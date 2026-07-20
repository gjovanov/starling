"""Run detail, extraction reruns, extraction results."""

import asyncio
import json
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from ..config import settings
from ..db import Database
from ..extraction.csv_export import extraction_to_csv
from ..models import ApiResponse, ExtractRequest
from ..state import app_state

router = APIRouter()


def _db() -> Database:
    if app_state.db is None:
        raise HTTPException(status_code=503, detail="server not ready")
    return app_state.db


def _resolve_results(rel: str) -> Path:
    root = Path(settings.results_dir).resolve()
    p = (root / rel).resolve()
    if not p.is_relative_to(root):
        raise HTTPException(status_code=400, detail="invalid result path")
    return p


@router.get("/api/ocr/runs/{run_id}")
async def run_detail(run_id: int) -> ApiResponse:
    db = _db()
    run = await db.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    doc = await db.get_document(run["document_id"])
    pages = await db.pages_for_run(run_id)
    extractions = await db.extractions_for_run(run_id)
    return ApiResponse.ok({"run": run, "document": doc, "pages": pages,
                           "extractions": extractions})


@router.post("/api/ocr/runs/{run_id}/extract")
async def rerun_extraction(run_id: int, req: ExtractRequest) -> ApiResponse:
    db = _db()
    run = await db.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    if run["status"] != "done":
        raise HTTPException(status_code=409,
                            detail=f"run is not done (status: {run['status']})")
    model = req.model or run["extract_model"] or settings.extract_model
    ex = await db.create_extraction(run_id=run_id, model=model)
    await app_state.ws.broadcast({"type": "extraction", "id": ex["id"], "run_id": run_id,
                                  "status": "queued", "model": model})
    app_state.queue.notify_extraction()
    return ApiResponse.ok(ex)


@router.get("/api/ocr/extractions/{ex_id}")
async def extraction_detail(ex_id: int) -> ApiResponse:
    db = _db()
    ex = await db.get_extraction(ex_id)
    if ex is None:
        raise HTTPException(status_code=404, detail="extraction not found")
    payload: dict = {**ex, "data": None, "validation": None}
    if ex["json_path"]:
        p = _resolve_results(ex["json_path"])
        if p.is_file():
            payload["data"] = json.loads(await asyncio.to_thread(p.read_text, "utf-8"))
    if ex["validation_path"]:
        p = _resolve_results(ex["validation_path"])
        if p.is_file():
            payload["validation"] = json.loads(await asyncio.to_thread(p.read_text, "utf-8"))
    return ApiResponse.ok(payload)


@router.get("/api/ocr/extractions/{ex_id}/csv")
async def extraction_csv(ex_id: int) -> Response:
    """Extracted invoice as a CSV table (metadata + line items). UTF-8 BOM so
    Excel renders Cyrillic (ЕИК / BG supplier names) correctly."""
    db = _db()
    ex = await db.get_extraction(ex_id)
    if ex is None:
        raise HTTPException(status_code=404, detail="extraction not found")
    if ex["status"] != "done" or not ex["json_path"]:
        raise HTTPException(status_code=409,
                            detail=f"extraction not complete (status: {ex['status']})")
    p = _resolve_results(ex["json_path"])
    if not p.is_file():
        raise HTTPException(status_code=404, detail="extraction data missing on disk")
    data = json.loads(await asyncio.to_thread(p.read_text, "utf-8"))
    csv_text = extraction_to_csv(data)

    stem = "extraction"
    run = await db.get_run(ex["run_id"])
    if run is not None:
        doc = await db.get_document(run["document_id"])
        if doc is not None:
            stem = Path(doc["filename"]).stem
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_") or "extraction"
    filename = f"{safe}_run{ex['run_id']}_extraction.csv"
    return Response(
        content="\ufeff" + csv_text,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
