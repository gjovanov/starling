"""Restart mid-queue: queued runs resume, crashed 'running' runs re-queue."""

import asyncio
import time
from pathlib import Path

from ocr_server.db import Database
from ocr_server.ingest import ingest_upload
from tests.helpers import LONG_TEXT, make_text_pdf


async def _seed(ocr_settings, *, mark_running: bool) -> int:
    """Create a document + run in the DB with no server running."""
    db = Database(ocr_settings.db_path)
    await db.connect()
    doc, _ = await ingest_upload(
        db, Path(ocr_settings.media_dir), filename="seed.pdf",
        content=make_text_pdf([LONG_TEXT]), max_bytes=10**8)
    run = await db.create_run(document_id=doc["id"], engine="mock", extract_auto=False)
    if mark_running:
        await db.mark_run_running(run["id"])
    await db.close()
    return run["id"]


async def _boot_and_wait(run_id: int, timeout: float = 10.0) -> dict:
    from ocr_server.state import app_state

    await app_state.startup()
    try:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            run = await app_state.db.get_run(run_id)
            if run["status"] in ("done", "failed"):
                return run
            await asyncio.sleep(0.02)
        raise TimeoutError(f"run {run_id} still {run['status']} after restart")
    finally:
        await app_state.shutdown()


async def test_queued_run_resumes_after_restart(ocr_settings):
    run_id = await _seed(ocr_settings, mark_running=False)
    run = await _boot_and_wait(run_id)
    assert run["status"] == "done"


async def test_running_run_requeues_after_crash(ocr_settings):
    # 'running' with no live worker == the process died mid-run.
    run_id = await _seed(ocr_settings, mark_running=True)
    run = await _boot_and_wait(run_id)
    assert run["status"] == "done"
