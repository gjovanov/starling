"""Shared pytest fixtures for vllm-server tests."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

# Ensure the package is importable when running `pytest` from the app root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(autouse=True)
def reset_app_state():
    """Clear out the global `app_state` between tests to avoid cross-contamination."""
    from voxtral_server.state import app_state

    # Cancel any lingering tasks from prior tests
    for sid in list(app_state.sessions.keys()):
        app_state.remove_session(sid)
    yield
    for sid in list(app_state.sessions.keys()):
        app_state.remove_session(sid)


@pytest.fixture
def test_app() -> FastAPI:
    """Return a FastAPI instance wired up the same way main.py does, minus startup hooks."""
    from voxtral_server.api import (
        config_routes,
        media_routes,
        model_routes,
        session_routes,
        tts_routes,
    )
    from voxtral_server.ws import handler as ws_handler

    app = FastAPI()
    app.include_router(config_routes.router)
    app.include_router(media_routes.router)
    app.include_router(model_routes.router)
    app.include_router(session_routes.router)
    app.include_router(tts_routes.router)
    app.include_router(ws_handler.router)
    return app


@pytest.fixture
async def client(test_app: FastAPI):
    """httpx AsyncClient bound to the in-process FastAPI app."""
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
