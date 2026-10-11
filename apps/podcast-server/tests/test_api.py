import io
import wave

import pytest
from httpx import ASGITransport, AsyncClient

TWO_TURNS = [{"speaker": "michael", "text": "Hello there."}, {"speaker": "emma", "text": "Hi."}]


async def test_podcast_returns_a_labelled_wav(client):
    r = await client.post("/api/tts/podcast", json={"turns": TWO_TURNS, "seed": 7})
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "audio/wav"
    h = r.headers
    assert h["x-podcast-sample-rate"] == "24000"
    assert h["x-podcast-seed"] == "7"
    assert h["x-podcast-cfg-scale"] == "1.3"
    assert h["x-podcast-diffusion-steps"] == "10"
    assert h["x-podcast-speakers"] == "michael,emma"
    assert h["x-podcast-truncated"] == "false"
    assert h["x-podcast-ai-generated"] == "true"
    assert h["x-podcast-model"] == "mock"
    assert "x-podcast-cold-load-secs" in h
    with wave.open(io.BytesIO(r.content)) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 24000)
        assert abs(w.getnframes() / 24000 - float(h["x-podcast-duration-secs"])) < 1e-3
    assert b"AI-generated speech" in r.content[:1024]


async def test_defaults_and_warm_second_request(client):
    await client.post("/api/tts/podcast", json={"turns": TWO_TURNS})
    r = await client.post("/api/tts/podcast", json={"turns": TWO_TURNS})
    assert r.status_code == 200
    assert r.headers["x-podcast-seed"] == "42"
    assert "x-podcast-cold-load-secs" not in r.headers


async def test_unknown_speaker_is_400(client):
    r = await client.post("/api/tts/podcast", json={"turns": [{"speaker": "bob", "text": "hi"}]})
    assert r.status_code == 400
    assert "unknown speaker 'bob'" in r.json()["detail"] and "michael" in r.json()["detail"]


async def test_fifth_speaker_is_400(client):
    turns = [{"speaker": n, "text": "hi"} for n in ("michael", "emma", "alex", "sam", "kim")]
    r = await client.post("/api/tts/podcast", json={"turns": turns})
    assert r.status_code == 400 and "at most 4" in r.json()["detail"]


@pytest.mark.parametrize("body", [
    {"turns": []},
    {"turns": [{"speaker": "michael"}]},
    {"turns": [{"speaker": "michael", "text": ""}]},
    {"turns": TWO_TURNS, "cfg_scale": 9},
    {"turns": TWO_TURNS, "diffusion_steps": 0},
    {"turns": TWO_TURNS, "seed": -1},
    {"turns": TWO_TURNS, "speed": 1.2},
    {"turns": [{"speaker": "michael", "text": "hi", "voice": "x"}]},
])
async def test_schema_violations_are_422(client, body):
    r = await client.post("/api/tts/podcast", json=body)
    assert r.status_code == 422


async def test_voices_endpoint(client):
    r = await client.get("/api/tts/podcast/voices")
    names = [v["name"] for v in r.json()["voices"]]
    assert {"michael", "emma"} <= set(names)
    assert all(v["sample_rate"] == 24000 for v in r.json()["voices"])


async def test_status_load_unload(client):
    assert (await client.get("/api/tts/podcast/status")).json()["state"] == "idle"
    loaded = (await client.post("/api/tts/podcast/load")).json()
    assert loaded["state"] == "ready" and loaded["engine"] == "mock"
    unloaded = (await client.post("/api/tts/podcast/unload")).json()
    assert unloaded["unloaded"] is True and unloaded["state"] == "idle"


async def test_blocked_is_503_with_reason(voices_dir):
    from podcast_server.engine import PodcastEngine
    from podcast_server.main import create_app
    from podcast_server.worker import MockWorker

    async def short_on_vram():
        return 1.0

    engine = PodcastEngine(MockWorker, engine_name="mock", model_label="mock", idle_unload_secs=0,
                           min_free_vram_gib=8.0, gpu_probe=short_on_vram)
    app = create_app(engine=engine)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/tts/podcast", json={"turns": TWO_TURNS})
    assert r.status_code == 503 and r.json()["reason"] == "blocked"


async def test_health(client):
    assert (await client.get("/health")).text == "OK"
