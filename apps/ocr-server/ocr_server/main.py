"""Starling OCR server — FastAPI entry point (port 8092).

Serves the shared frontend, the /api/ocr/* REST API and the /api/ocr/ws
event stream. Model subprocesses (GLM-OCR on :8003, extraction LLM on
:8004) are spawned lazily by the engine manager.
"""

import logging
import os
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles

from . import ws as ws_module
from .config import settings
from .routes import documents, engines, jobs, results
from .state import app_state

# uvicorn only configures its own loggers — without this, per-page timing
# and engine lifecycle logs from ocr_server.* are silently dropped.
logging.basicConfig(
    level=os.environ.get("OCR_LOG_LEVEL", "INFO"),
    format="%(levelname)s %(name)s: %(message)s",
)


class NoCacheStaticFiles(StaticFiles):
    """Static frontend with Cache-Control: no-cache — the browser revalidates
    every asset on each load (304 keeps it fast), so a plain F5 always picks
    up current ES modules. Same policy as burn-server (a6ade53)."""

    async def get_response(self, path, scope):  # type: ignore[override]
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


def create_app(*, serve_frontend: bool = True) -> FastAPI:
    app = FastAPI(title="Starling OCR Server", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health() -> PlainTextResponse:
        return PlainTextResponse("OK")

    @app.get("/api/config")
    async def api_config() -> dict:
        # Shared-frontend bootstrap endpoint. No overrides needed for OCR;
        # the key identifies which backend is serving.
        return {"app": "ocr-server"}

    app.include_router(engines.router)
    app.include_router(documents.router)
    app.include_router(jobs.router)
    app.include_router(results.router)
    app.include_router(ws_module.router)

    if serve_frontend:
        frontend_path = Path(settings.frontend_path).resolve()
        if frontend_path.is_dir():
            # Must be last — catch-all.
            app.mount("/", NoCacheStaticFiles(directory=str(frontend_path), html=True),
                      name="frontend")
            print(f"[Server] Serving frontend from {frontend_path}", file=sys.stderr)
        else:
            print(f"[Server] WARNING: Frontend path not found: {frontend_path}", file=sys.stderr)
    return app


app = create_app()


@app.on_event("startup")
async def _startup() -> None:
    await app_state.startup()


@app.on_event("shutdown")
async def _shutdown() -> None:
    await app_state.shutdown()


def run() -> None:
    """Entry point for the `ocr-server` command."""
    print(f"[Server] Starting ocr-server on port {settings.port}", file=sys.stderr)
    print(f"[Server] Default engine: {settings.engine}", file=sys.stderr)
    print(f"[Server] Media dir: {Path(settings.media_dir).resolve()}", file=sys.stderr)
    print(f"[Server] Results dir: {Path(settings.results_dir).resolve()}", file=sys.stderr)
    uvicorn.run(
        "ocr_server.main:app",
        host="0.0.0.0",
        port=settings.port,
        log_level="info",
    )


if __name__ == "__main__":
    run()
