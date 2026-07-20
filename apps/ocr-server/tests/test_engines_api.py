"""Engine registry endpoints + WS engine events on load/unload."""

import asyncio
import time


async def test_config_shape(client):
    r = await client.get("/api/ocr/config")
    cfg = r.json()["data"]
    assert "mock" in cfg["engines"]
    assert cfg["default_engine"] == "mock"
    assert cfg["accepted_types"] == [".pdf", ".png", ".jpg", ".jpeg"]
    assert cfg["language_hint"]


async def test_engines_listing(client):
    r = await client.get("/api/ocr/engines")
    engines = r.json()["data"]["engines"]
    mock = next(e for e in engines if e["name"] == "mock")
    assert mock["kind"] == "mock"
    assert mock["vram_class"] == "light"
    assert mock["status"] == "idle"


async def _mock_status(client) -> str:
    engines = (await client.get("/api/ocr/engines")).json()["data"]["engines"]
    return next(e for e in engines if e["name"] == "mock")["status"]


async def test_load_unload_cycle_with_ws_events(client, started_state):
    events = started_state.ws.subscribe()

    r = await client.post("/api/ocr/engines/mock/load")
    assert r.json()["data"]["status"] == "loading"

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        st = await _mock_status(client)
        if st == "ready":
            break
        await asyncio.sleep(0.02)
    assert st == "ready"

    r = await client.post("/api/ocr/engines/mock/unload")
    assert r.status_code == 200
    assert await _mock_status(client) == "idle"

    seen = []
    while not events.empty():
        seen.append(events.get_nowait())
    started_state.ws.unsubscribe(events)
    statuses = [e["status"] for e in seen if e["type"] == "engine" and e["name"] == "mock"]
    assert statuses == ["loading", "ready", "unloaded"]


async def test_unknown_engine_actions_404(client):
    assert (await client.post("/api/ocr/engines/nope/load")).status_code == 404
    assert (await client.post("/api/ocr/engines/nope/unload")).status_code == 404
