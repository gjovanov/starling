"""Upload + document listing + bake-off re-runs + deletion."""

import asyncio
import shutil
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ..config import settings
from ..db import Database
from ..ingest import IngestError, ingest_upload
from ..models import ApiResponse, RunCreateRequest
from ..state import app_state

router = APIRouter()


def _db() -> Database:
    if app_state.db is None:
        raise HTTPException(status_code=503, detail="server not ready")
    return app_state.db


async def _enqueue_run(doc: dict, *, engine: str, extract_auto: bool,
                       extract_model: str | None, language_hint: str | None) -> dict:
    db = _db()
    run = await db.create_run(
        document_id=doc["id"], engine=engine, extract_auto=extract_auto,
        extract_model=extract_model, language_hint=language_hint or settings.language_hint)
    await app_state.ws.broadcast({
        "type": "run", "id": run["id"], "document_id": doc["id"], "status": "queued",
        "engine": engine, "page": {"done": 0, "total": doc["page_count"]},
    })
    app_state.queue.notify_run()
    return run


def _validate_engine(name: str) -> None:
    if app_state.engines is None or name not in app_state.engines.names():
        available = app_state.engines.names() if app_state.engines else []
        raise HTTPException(status_code=400,
                            detail=f"unknown engine: {name!r} (available: {available})")


@router.post("/api/ocr/documents")
async def upload_documents(
    files: list[UploadFile] = File(...),
    engine: str | None = Form(None),
    extract: str | None = Form(None),
    extract_model: str | None = Form(None),
    language_hint: str | None = Form(None),
) -> ApiResponse:
    db = _db()
    engine_name = engine or settings.engine
    _validate_engine(engine_name)
    if extract not in (None, "auto", "off"):
        raise HTTPException(status_code=400, detail="extract must be 'auto' or 'off'")
    extract_auto = (extract == "auto") if extract is not None else settings.extract_auto

    documents: list[dict] = []
    errors: list[dict] = []
    for f in files:
        name = f.filename or "upload"
        content = await f.read()
        try:
            doc, created = await ingest_upload(
                db, Path(settings.media_dir), filename=name, content=content,
                max_bytes=settings.max_upload_bytes)
        except IngestError as exc:
            if len(files) == 1:
                raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
            errors.append({"filename": name, "error": str(exc), "status": exc.status_code})
            continue
        run = await _enqueue_run(doc, engine=engine_name, extract_auto=extract_auto,
                                 extract_model=extract_model, language_hint=language_hint)
        documents.append({"document_id": doc["id"], "run_id": run["id"],
                          "filename": name, "deduped": not created})

    if not documents and errors:
        raise HTTPException(status_code=errors[0]["status"], detail=errors[0]["error"])
    return ApiResponse.ok({"documents": documents, "errors": errors})


async def _run_summary(db: Database, run: dict) -> dict:
    done = await db.pages_done(run["id"])
    latest_ex = await db.latest_extraction_for_run(run["id"])
    return {**run, "pages_done": done, "latest_extraction": latest_ex}


@router.get("/api/ocr/documents")
async def list_documents() -> ApiResponse:
    db = _db()
    out = []
    for d in await db.list_documents():
        latest = await db.latest_run_for_document(d["id"])
        out.append({**d, "latest_run": await _run_summary(db, latest) if latest else None})
    return ApiResponse.ok({"documents": out})


@router.get("/api/ocr/documents/{doc_id}/runs")
async def document_runs(doc_id: int) -> ApiResponse:
    db = _db()
    doc = await db.get_document(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="document not found")
    runs = [await _run_summary(db, r) for r in await db.runs_for_document(doc_id)]
    return ApiResponse.ok({"document": doc, "runs": runs})


def _prune_dir(root_dir: str, name: str) -> None:
    """rmtree root_dir/name, but only if it's genuinely inside root_dir."""
    root = Path(root_dir).resolve()
    target = (root / name).resolve()
    if target.is_relative_to(root) and target.is_dir():
        shutil.rmtree(target, ignore_errors=True)


@router.delete("/api/ocr/documents/{doc_id}")
async def delete_document(doc_id: int) -> ApiResponse:
    """Delete a document + all its runs, extractions and on-disk artifacts.
    Rejects while a run is actively processing (its worker holds the files)."""
    db = _db()
    doc = await db.get_document(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="document not found")
    runs = await db.runs_for_document(doc_id)
    if any(r["status"] == "running" for r in runs):
        raise HTTPException(status_code=409,
                            detail="cannot delete while a run is in progress")

    run_ids = await db.delete_document(doc_id)
    await asyncio.to_thread(_prune_dir, settings.media_dir, str(doc_id))
    for rid in run_ids:
        await asyncio.to_thread(_prune_dir, settings.results_dir, str(rid))

    await app_state.ws.broadcast(
        {"type": "document", "id": doc_id, "status": "deleted", "run_ids": run_ids})
    return ApiResponse.ok({"deleted": doc_id, "runs": run_ids})


@router.post("/api/ocr/documents/{doc_id}/runs")
async def rerun_document(doc_id: int, req: RunCreateRequest) -> ApiResponse:
    db = _db()
    doc = await db.get_document(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="document not found")
    _validate_engine(req.engine)
    if req.extract not in (None, "auto", "off"):
        raise HTTPException(status_code=400, detail="extract must be 'auto' or 'off'")
    extract_auto = (req.extract == "auto") if req.extract is not None \
        else settings.extract_auto
    run = await _enqueue_run(doc, engine=req.engine, extract_auto=extract_auto,
                             extract_model=None, language_hint=req.language_hint)
    return ApiResponse.ok(run)
