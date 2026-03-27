"""Voxtral ASR server — FastAPI app with WebRTC audio + vLLM transcription."""

from __future__ import annotations

import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles

from .api.config_routes import router as config_router
from .api.media_routes import router as media_router
from .api.model_routes import router as model_router
from .api.session_routes import router as session_router
from .config import settings
from .ws.handler import router as ws_router

app = FastAPI(title="Voxtral Server", version="0.1.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check
@app.get("/health")
async def health():
    return PlainTextResponse("OK")

# API routes
app.include_router(config_router)
app.include_router(model_router)
app.include_router(media_router)
app.include_router(session_router)
app.include_router(ws_router)

# Serve frontend static files (must be last — catch-all)
frontend_path = Path(settings.frontend_path).resolve()
if frontend_path.is_dir():
    app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="frontend")
    print(f"[Server] Serving frontend from {frontend_path}", file=sys.stderr)
else:
    print(f"[Server] WARNING: Frontend path not found: {frontend_path}", file=sys.stderr)


def run():
    """Entry point for `voxtral-server` command."""
    print(f"[Server] Starting on port {settings.port}", file=sys.stderr)
    print(f"[Server] vLLM URL: {settings.vllm_url}", file=sys.stderr)
    print(f"[Server] Media dir: {Path(settings.media_dir).resolve()}", file=sys.stderr)
    uvicorn.run(
        "voxtral_server.main:app",
        host="0.0.0.0",
        port=settings.port,
        log_level="info",
    )


if __name__ == "__main__":
    run()
