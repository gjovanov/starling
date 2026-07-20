"""In-process asyncio job queue backed by SQLite.

Two workers: one for OCR runs (serializes GPU engine access), one for
extractions (separate GPU process — may run concurrently with OCR).
Queued state lives in the DB, so jobs survive restarts: start() re-queues
anything left 'running' by a dead process and wakes the workers.
"""

import asyncio
import json
import logging
import time
from pathlib import Path

from . import pdf
from .config import Settings
from .db import Database
from .engines.manager import EngineManager
from .extraction.extractor import create_extractor
from .extraction.validate import validate_extraction
from .ingest import original_path
from .ws import Broadcaster

logger = logging.getLogger(__name__)


class JobQueue:
    def __init__(self, *, db: Database, ws: Broadcaster, engines: EngineManager,
                 settings: Settings, extractor_factory=None) -> None:
        self._db = db
        self._ws = ws
        self._engines = engines
        self._settings = settings
        self._extractor_factory = extractor_factory or create_extractor
        self._run_wake = asyncio.Event()
        self._extract_wake = asyncio.Event()
        self._tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        requeued = await self._db.reset_inflight()
        if requeued:
            logger.info("[Queue] re-queued %d in-flight job(s) from previous process", requeued)
        # Force an initial scan — queued rows may be waiting from a prior run.
        self._run_wake.set()
        self._extract_wake.set()
        self._tasks = [
            asyncio.create_task(self._worker(self._run_wake, self._db.next_queued_run,
                                             self._process_run), name="ocr-run-worker"),
            asyncio.create_task(self._worker(self._extract_wake, self._db.next_queued_extraction,
                                             self._process_extraction), name="ocr-extract-worker"),
        ]

    async def stop(self) -> None:
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []

    def notify_run(self) -> None:
        self._run_wake.set()

    def notify_extraction(self) -> None:
        self._extract_wake.set()

    async def _worker(self, wake: asyncio.Event, next_queued, process) -> None:
        while True:
            await wake.wait()
            try:
                row = await next_queued()
                if row is None:
                    # No work: sleep until notified. The re-set below start()
                    # and in notify_*() closes the lost-wakeup race.
                    wake.clear()
                    if await next_queued() is not None:
                        wake.set()
                    continue
                await process(row)
            except asyncio.CancelledError:
                raise
            except Exception:
                # process() handles per-job failure itself; this guards worker
                # survival against bugs in the failure path.
                logger.exception("[Queue] worker error (worker continues)")
                await asyncio.sleep(0.5)

    # ── stage 1: OCR runs ────────────────────────────────────────────

    async def _process_run(self, run: dict) -> None:
        run_id = run["id"]
        t0 = time.monotonic()
        try:
            doc = await self._db.get_document(run["document_id"])
            if doc is None:
                raise RuntimeError("document row missing")
            engine = self._engines.get(run["engine"])
            await self._db.mark_run_running(run_id, engine_version=engine.version)
            total = doc["page_count"]
            await self._broadcast_run(run_id, doc["id"], "running", run["engine"],
                                      done=0, total=total)

            media_root = Path(self._settings.media_dir)
            results_root = Path(self._settings.results_dir)
            src = original_path(media_root, doc)
            if not src.exists():
                raise RuntimeError(f"original file missing: {src}")

            pngs = await asyncio.to_thread(
                pdf.rasterize_pages, src, doc["mime"], src.parent,
                dpi=self._settings.raster_dpi)
            layers = await asyncio.to_thread(
                pdf.extract_text_layers, src, doc["mime"],
                min_chars=self._settings.text_layer_min_chars)

            run_dir = results_root / str(run_id)
            run_dir.mkdir(parents=True, exist_ok=True)
            language_hint = run["language_hint"] or self._settings.language_hint

            for i in range(total):
                page_no = i + 1
                p0 = time.monotonic()
                text = layers[i] if i < len(layers) else None
                if text is not None:
                    md, source = text, "text-layer"
                else:
                    # Lazy engine load happens here — text-layer-only documents
                    # never touch the GPU.
                    eng = await self._engines.ensure_ready(engine.name)
                    md = await asyncio.wait_for(
                        eng.ocr_page(pngs[i], language_hint),
                        timeout=self._settings.ocr_page_timeout_secs)
                    eng.note_activity()
                    source = "ocr"
                md_path = run_dir / f"page-{page_no:03d}.md"
                await asyncio.to_thread(md_path.write_text, md, "utf-8")
                ms = int((time.monotonic() - p0) * 1000)
                await self._db.upsert_page(
                    run_id=run_id, page_no=page_no, source=source,
                    markdown_path=f"{run_id}/page-{page_no:03d}.md",
                    image_path=f"{doc['id']}/page-{page_no:03d}.png", ms=ms)
                await self._broadcast_run(run_id, doc["id"], "running", run["engine"],
                                          done=page_no, total=total)
                logger.info("[Run %d] page %d/%d via %s in %dms", run_id, page_no, total, source, ms)

            ms_total = int((time.monotonic() - t0) * 1000)
            await self._db.mark_run_finished(run_id, status="done", ms_total=ms_total,
                                             engine_version=engine.version)
            await self._broadcast_run(run_id, doc["id"], "done", run["engine"],
                                      done=total, total=total, ms_total=ms_total)

            if run["extract_auto"]:
                model = run["extract_model"] or self._settings.extract_model
                ex = await self._db.create_extraction(run_id=run_id, model=model)
                await self._broadcast_extraction(ex)
                self.notify_extraction()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            ms_total = int((time.monotonic() - t0) * 1000)
            logger.exception("[Run %d] failed", run_id)
            await self._db.mark_run_finished(
                run_id, status="failed", error=f"{type(exc).__name__}: {exc}", ms_total=ms_total)
            await self._broadcast_run(run_id, run["document_id"], "failed", run["engine"],
                                      error=str(exc))

    # ── stage 2: extractions ─────────────────────────────────────────

    async def _process_extraction(self, ex: dict) -> None:
        ex_id = ex["id"]
        t0 = time.monotonic()
        try:
            await self._db.mark_extraction_running(ex_id)
            await self._broadcast_extraction({**ex, "status": "running"})
            run = await self._db.get_run(ex["run_id"])
            if run is None:
                raise RuntimeError("run row missing")
            pages = await self._db.pages_for_run(ex["run_id"])
            if not pages:
                raise RuntimeError("run has no page results to extract from")

            results_root = Path(self._settings.results_dir)
            parts = []
            for p in pages:
                parts.append(await asyncio.to_thread(
                    (results_root / p["markdown_path"]).read_text, "utf-8"))
            markdown = "\n\n".join(parts)

            extractor = self._extractor_factory(ex["model"], self._settings)
            language_hint = run["language_hint"] or self._settings.language_hint
            data = await asyncio.wait_for(
                extractor.extract(markdown, language_hint=language_hint),
                timeout=self._settings.extract_timeout_secs)
            overall, checks = validate_extraction(data)

            run_dir = results_root / str(run["id"])
            run_dir.mkdir(parents=True, exist_ok=True)
            json_rel = f"{run['id']}/extraction-{ex_id}.json"
            val_rel = f"{run['id']}/validation-{ex_id}.json"
            await asyncio.to_thread(
                (results_root / json_rel).write_text,
                json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
            await asyncio.to_thread(
                (results_root / val_rel).write_text,
                json.dumps({"overall": overall, "checks": checks}, ensure_ascii=False, indent=2),
                "utf-8")

            ms = int((time.monotonic() - t0) * 1000)
            await self._db.mark_extraction_finished(
                ex_id, status="done", json_path=json_rel, validation_status=overall,
                validation_path=val_rel, ms=ms)
            await self._broadcast_extraction(
                {**ex, "status": "done", "validation_status": overall, "ms": ms})
            logger.info("[Extraction %d] run %d via %s in %dms → %s",
                        ex_id, ex["run_id"], ex["model"], ms, overall)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            ms = int((time.monotonic() - t0) * 1000)
            logger.exception("[Extraction %d] failed", ex_id)
            await self._db.mark_extraction_finished(
                ex_id, status="failed", error=f"{type(exc).__name__}: {exc}", ms=ms)
            await self._broadcast_extraction(
                {**ex, "status": "failed", "error": str(exc), "ms": ms})

    # ── WS events ────────────────────────────────────────────────────

    async def _broadcast_run(self, run_id: int, document_id: int, status: str, engine: str,
                             *, done: int | None = None, total: int | None = None,
                             ms_total: int | None = None, error: str | None = None) -> None:
        msg: dict = {"type": "run", "id": run_id, "document_id": document_id,
                     "status": status, "engine": engine}
        if done is not None:
            msg["page"] = {"done": done, "total": total}
        if ms_total is not None:
            msg["ms_total"] = ms_total
        if error:
            msg["error"] = error
        await self._ws.broadcast(msg)

    async def _broadcast_extraction(self, ex: dict) -> None:
        msg: dict = {"type": "extraction", "id": ex["id"], "run_id": ex["run_id"],
                     "status": ex.get("status"), "model": ex.get("model")}
        if ex.get("validation_status"):
            msg["validation_status"] = ex["validation_status"]
        if ex.get("ms") is not None:
            msg["ms"] = ex["ms"]
        if ex.get("error"):
            msg["error"] = ex["error"]
        await self._ws.broadcast(msg)
