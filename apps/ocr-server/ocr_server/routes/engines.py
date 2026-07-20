"""Engine registry endpoints + the OCR tab bootstrap config."""

import asyncio
import logging

from fastapi import APIRouter, HTTPException

from ..config import settings
from ..models import ApiResponse
from ..state import app_state

logger = logging.getLogger(__name__)
router = APIRouter()


def _validate(name: str) -> None:
    if app_state.engines is None:
        raise HTTPException(status_code=503, detail="server not ready")
    if name not in app_state.engines.names():
        raise HTTPException(status_code=404, detail=f"unknown engine: {name!r}")


@router.get("/api/ocr/config")
async def ocr_config() -> ApiResponse:
    engines = app_state.engines.names() if app_state.engines else []
    return ApiResponse.ok({
        "engines": engines,
        "default_engine": settings.engine if settings.engine in engines
        else (engines[0] if engines else None),
        "extract_auto": settings.extract_auto,
        "extract_model": settings.extract_model,
        "language_hint": settings.language_hint,
        "max_upload_bytes": settings.max_upload_bytes,
        "accepted_types": [".pdf", ".png", ".jpg", ".jpeg"],
    })


@router.get("/api/ocr/engines")
async def list_engines() -> ApiResponse:
    if app_state.engines is None:
        raise HTTPException(status_code=503, detail="server not ready")
    extractor = None
    if app_state.extract_process is not None:
        state_map = {"ready": "ready", "starting": "loading"}
        extractor = {
            "model": settings.extract_model,
            "status": state_map.get(app_state.extract_process.state.value, "idle"),
            "port": settings.vllm_extract_port,
        }
    return ApiResponse.ok({"engines": app_state.engines.statuses(),
                           "default": settings.engine,
                           "extractor": extractor})


@router.post("/api/ocr/engines/{name}/load")
async def load_engine(name: str) -> ApiResponse:
    _validate(name)

    async def _load() -> None:
        try:
            await app_state.engines.ensure_ready(name)
        except Exception as exc:
            # ensure_ready already broadcast the 'unloaded' engine event.
            logger.warning("[Engines] manual load of %s failed: %s", name, exc)

    # Long boots (a vLLM spawn is ~1 min) run in background; the UI follows
    # the WS 'engine' events.
    asyncio.create_task(_load())
    return ApiResponse.ok({"name": name, "status": "loading"})


@router.post("/api/ocr/engines/{name}/unload")
async def unload_engine(name: str) -> ApiResponse:
    _validate(name)
    await app_state.engines.unload(name)
    return ApiResponse.ok({"name": name, "status": "unloaded"})
