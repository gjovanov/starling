"""Per-page artifacts: rasterized images, page Markdown, concatenated Markdown."""

import asyncio
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse

from ..config import settings
from ..db import Database
from ..state import app_state

router = APIRouter()

_MD_MEDIA_TYPE = "text/markdown; charset=utf-8"


def _db() -> Database:
    if app_state.db is None:
        raise HTTPException(status_code=503, detail="server not ready")
    return app_state.db


def _resolve(root_dir: str, rel: str) -> Path:
    root = Path(root_dir).resolve()
    p = (root / rel).resolve()
    if not p.is_relative_to(root):
        raise HTTPException(status_code=400, detail="invalid path")
    return p


async def _page(run_id: int, page_no: int) -> dict:
    page = await _db().get_page(run_id, page_no)
    if page is None:
        raise HTTPException(status_code=404, detail="page not found (not processed yet?)")
    return page


@router.get("/api/ocr/runs/{run_id}/pages/{page_no}.png")
async def page_image(run_id: int, page_no: int) -> FileResponse:
    page = await _page(run_id, page_no)
    if not page["image_path"]:
        raise HTTPException(status_code=404, detail="page has no image")
    p = _resolve(settings.media_dir, page["image_path"])
    if not p.is_file():
        raise HTTPException(status_code=404, detail="page image missing on disk")
    return FileResponse(path=str(p), media_type="image/png")


@router.get("/api/ocr/runs/{run_id}/pages/{page_no}.md")
async def page_markdown(run_id: int, page_no: int) -> PlainTextResponse:
    page = await _page(run_id, page_no)
    p = _resolve(settings.results_dir, page["markdown_path"])
    if not p.is_file():
        raise HTTPException(status_code=404, detail="page markdown missing on disk")
    text = await asyncio.to_thread(p.read_text, "utf-8")
    return PlainTextResponse(text, media_type=_MD_MEDIA_TYPE)


@router.get("/api/ocr/runs/{run_id}/markdown")
async def run_markdown(run_id: int) -> PlainTextResponse:
    db = _db()
    run = await db.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    pages = await db.pages_for_run(run_id)
    parts = []
    for page in pages:
        p = _resolve(settings.results_dir, page["markdown_path"])
        if p.is_file():
            parts.append(await asyncio.to_thread(p.read_text, "utf-8"))
    return PlainTextResponse("\n\n".join(parts), media_type=_MD_MEDIA_TYPE)
