"""Integration tests for the speakers source on vllm-server.

Exercises the HTTP API surface (session creation, listing, state) and
verifies that creating a session with ``source="speakers"`` produces the
expected SessionInfo. A full WebRTC roundtrip is covered by the Playwright
E2E suite.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.asyncio


async def test_create_speakers_session(client):
    resp = await client.post(
        "/api/sessions",
        json={
            "model_id": "voxtral-mini-4b",
            "language": "de",
            "mode": "speedy",
            "source": "speakers",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    data = body["data"]
    assert data["source_type"] == "speakers"
    assert data["state"] == "created"
    assert data["media_filename"] == "Speakers (live capture)"
    assert data["id"]


async def test_create_session_requires_source_when_no_media(client):
    resp = await client.post(
        "/api/sessions",
        json={
            "model_id": "voxtral-mini-4b",
            "language": "de",
            "mode": "speedy",
        },
    )
    body = resp.json()
    # ApiResponse returns success=False with an explanatory message
    assert body["success"] is False
    assert "source" in body["error"].lower() or "media" in body["error"].lower()


async def test_list_includes_speakers_sessions(client):
    for language in ("de", "en"):
        r = await client.post(
            "/api/sessions",
            json={
                "model_id": "voxtral-mini-4b",
                "language": language,
                "mode": "speedy",
                "source": "speakers",
            },
        )
        assert r.status_code == 200
        assert r.json()["success"] is True

    resp = await client.get("/api/sessions")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert len(data) == 2
    assert all(s["source_type"] == "speakers" for s in data)


async def test_speakers_and_media_sessions_coexist(client, tmp_path, monkeypatch):
    # Point the media directory at a temp path with one real file so the
    # media session route can resolve it without needing ffprobe.
    from voxtral_server.config import settings

    media_dir = tmp_path / "media"
    media_dir.mkdir()
    # Empty WAV-named file is enough for the lookup path (duration falls back to 0).
    (media_dir / "sample.wav").write_bytes(b"RIFF0000WAVEfmt ")

    monkeypatch.setattr(settings, "media_dir", str(media_dir))

    r1 = await client.post(
        "/api/sessions",
        json={"model_id": "voxtral-mini-4b", "mode": "speedy", "source": "speakers"},
    )
    assert r1.json()["success"] is True

    r2 = await client.post(
        "/api/sessions",
        json={
            "model_id": "voxtral-mini-4b",
            "mode": "speedy",
            "media_id": "sample",
        },
    )
    assert r2.json()["success"] is True, r2.json()

    listed = (await client.get("/api/sessions")).json()["data"]
    assert len(listed) == 2
    types = {s["source_type"] for s in listed}
    assert "speakers" in types
    assert "file" in types


async def test_get_speakers_session_by_id(client):
    r = await client.post(
        "/api/sessions",
        json={"model_id": "voxtral-mini-4b", "mode": "speedy", "source": "speakers"},
    )
    sid = r.json()["data"]["id"]

    resp = await client.get(f"/api/sessions/{sid}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["id"] == sid
    assert body["data"]["source_type"] == "speakers"


async def test_stop_speakers_session(client):
    r = await client.post(
        "/api/sessions",
        json={"model_id": "voxtral-mini-4b", "mode": "speedy", "source": "speakers"},
    )
    sid = r.json()["data"]["id"]

    resp = await client.delete(f"/api/sessions/{sid}")
    assert resp.status_code == 200
    assert resp.json()["success"] is True

    listed = (await client.get("/api/sessions")).json()["data"]
    assert listed == []


async def test_pump_inbound_audio_resamples_48k_to_16k():
    """End-to-end test of the inbound audio pump using a *real* aiortc Opus
    encode → decode → AudioResampler pipeline. This catches the class of bug
    where browser-sent stereo packed s16 frames were being mishandled."""
    import asyncio
    from fractions import Fraction
    import numpy as np
    from av import AudioFrame
    from aiortc.codecs.opus import OpusEncoder, OpusDecoder
    from aiortc.jitterbuffer import JitterFrame

    from voxtral_server.ws import handler

    # Build a 480 ms 1 kHz tone at 48 kHz STEREO (matches what browsers send).
    sample_rate = 48000
    duration_s = 0.480
    n = int(sample_rate * duration_s)
    t = np.arange(n) / sample_rate
    tone = (0.5 * np.sin(2 * np.pi * 1000.0 * t)).astype(np.float32)

    # Encode in 20ms chunks (the standard Opus frame size aiortc uses)
    frame_samples = int(sample_rate * 0.020)
    enc = OpusEncoder()
    payloads = []
    for off in range(0, n, frame_samples):
        chunk = tone[off:off + frame_samples]
        if len(chunk) < frame_samples:
            chunk = np.pad(chunk, (0, frame_samples - len(chunk)))
        # Interleaved stereo s16 → packed shape (1, samples * 2)
        stereo = np.stack([chunk, chunk], axis=1).flatten()
        s16 = (stereo * 32767).astype(np.int16).reshape(1, -1)
        f = AudioFrame.from_ndarray(s16, format="s16", layout="stereo")
        f.sample_rate = sample_rate
        f.pts = off
        f.time_base = Fraction(1, 48000)
        chunks, _ = enc.encode(f, force_keyframe=False)
        payloads.extend(chunks)

    # Decode and feed each frame to the pump via a fake track.
    dec = OpusDecoder()
    out_frames = []
    for p in payloads:
        out_frames.extend(dec.decode(JitterFrame(data=p, timestamp=0)))

    class FakeTrack:
        def __init__(self, frames):
            self._frames = iter(frames)
            self.kind = "audio"

        async def recv(self):
            try:
                return next(self._frames)
            except StopIteration:
                raise RuntimeError("end")

    queue: asyncio.Queue = asyncio.Queue(maxsize=4096)
    await handler._pump_inbound_audio(FakeTrack(out_frames), queue, client_id="unit")

    # Drain and validate
    total = 0
    max_abs = 0.0
    sentinel_seen = False
    while not queue.empty():
        item = queue.get_nowait()
        if item is None:
            sentinel_seen = True
            continue
        total += len(item)
        max_abs = max(max_abs, float(np.abs(np.asarray(item)).max()))

    assert sentinel_seen, "pump should push a None sentinel on EOF"
    # ~0.48 s at 16 kHz = 7680 samples; allow a small Opus / resampler delay.
    expected = int(0.48 * 16000)
    assert total > expected * 0.85, f"expected ~{expected} samples, got {total}"
    # The 1 kHz tone must NOT have been mangled to silence by the channel-layout fix.
    # Voxtral can't transcribe near-silence, which is the symptom we're guarding against.
    assert max_abs > 0.05, f"expected meaningful audio amplitude, got max|sample|={max_abs}"
