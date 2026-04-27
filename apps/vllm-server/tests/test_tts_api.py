"""Integration tests for the TTS routes.

The actual TTS model is hosted by a separate vllm-omni process. We don't
boot it here — instead we monkey-patch `voxtral_server.tts.client.get_tts_client`
to return a fake that emits a constant 0.5s sine WAV. That keeps the suite
fast (<1s) and free of GPU dependencies.
"""

from __future__ import annotations

import io
import struct
import wave

import numpy as np
import pytest

from voxtral_server.config import settings
from voxtral_server.tts import client as tts_client_mod


pytestmark = pytest.mark.asyncio


def _fake_wav_bytes(duration_secs: float = 0.5, sample_rate: int = 24000) -> bytes:
    """Build a tiny but valid 24 kHz mono WAV containing a 440 Hz tone."""
    n = int(duration_secs * sample_rate)
    t = np.arange(n) / sample_rate
    samples = (0.2 * np.sin(2 * np.pi * 440.0 * t) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())
    return buf.getvalue()


def _fake_pcm_chunks(total_secs: float = 0.4, sample_rate: int = 24000) -> list[bytes]:
    """Two roughly-equal-sized chunks of silent PCM16 mono — enough to verify
    the streaming pipeline yields the upstream payload through unchanged."""
    n = int(total_secs * sample_rate)
    samples = np.zeros(n, dtype=np.int16).tobytes()
    half = len(samples) // 2
    return [samples[:half], samples[half:]]


class _FakeTtsClient:
    """Stand-in for `TtsClient` that records calls and returns canned bytes."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.stream_calls: list[dict] = []
        self.upload_calls: list[dict] = []
        self.delete_calls: list[str] = []
        self.audio = _fake_wav_bytes()
        self.pcm_chunks = _fake_pcm_chunks()

    async def synthesize(self, *, text: str, voice: str, response_format: str = "wav"):
        self.calls.append({"text": text, "voice": voice, "format": response_format})
        return tts_client_mod.TtsResult(
            audio_bytes=self.audio,
            content_type="audio/wav",
            elapsed_secs=0.01,
        )

    async def synthesize_stream(self, *, text: str, voice: str, chunk_size=None):
        self.stream_calls.append({"text": text, "voice": voice})
        for c in self.pcm_chunks:
            yield c

    async def _model_id(self) -> str:
        # Real client discovers via /v1/models; fake just returns a stub.
        return "fake-model"

    async def synthesize_stream_concat(self, *, text_parts, voice, chunk_size=None):
        for part in text_parts:
            async for c in self.synthesize_stream(text=part, voice=voice, chunk_size=chunk_size):
                yield c

    async def upload_voice(self, *, voice_name, consent, audio_bytes, audio_mime="audio/wav", ref_text=None):
        self.upload_calls.append({
            "voice_name": voice_name,
            "consent": consent,
            "ref_text": ref_text,
            "bytes": len(audio_bytes),
        })
        return {"success": True, "voice": {"name": voice_name}}

    async def delete_voice(self, voice_name: str) -> bool:
        self.delete_calls.append(voice_name)
        return True

    async def health(self) -> bool:
        return True

    async def aclose(self) -> None:
        pass


class _FakeLifecycle:
    """Stand-in that auto-readies and accepts the activity counters."""
    def __init__(self) -> None:
        self.started = 0
        self.finished = 0
        self.ensure_started_calls = 0

    async def ensure_started(self, task_type: str = "CustomVoice") -> None:
        self.ensure_started_calls += 1
        self.last_task_type = task_type

    async def ensure_stopped(self) -> None:
        pass

    async def status(self):
        from voxtral_server.tts.lifecycle import StatusInfo
        return StatusInfo(state="ready")

    @property
    def autostart(self) -> bool:
        return True

    def note_activity(self) -> None:
        pass

    def synth_started(self) -> None:
        self.started += 1

    def synth_finished(self) -> None:
        self.finished += 1


@pytest.fixture(autouse=True)
def _stub_tts_client(monkeypatch, tmp_path):
    """Replace the singleton TTS client + lifecycle + output dir."""
    from voxtral_server.api import tts_routes
    from voxtral_server.tts import lifecycle as lifecycle_mod

    fake = _FakeTtsClient()
    fake_lc = _FakeLifecycle()
    tts_client_mod.reset_tts_client()
    lifecycle_mod.reset_lifecycle()
    # Patch BOTH the source-module symbol (for completeness) and the route
    # module's imported name (which is what the handler actually calls).
    monkeypatch.setattr(tts_client_mod, "get_tts_client", lambda: fake)
    monkeypatch.setattr(tts_routes, "get_tts_client", lambda: fake)
    monkeypatch.setattr(lifecycle_mod, "get_lifecycle", lambda: fake_lc)
    monkeypatch.setattr(tts_routes, "get_lifecycle", lambda: fake_lc)
    monkeypatch.setattr(settings, "tts_output_dir", str(tmp_path))
    fake.lifecycle = fake_lc
    yield fake
    tts_client_mod.reset_tts_client()
    lifecycle_mod.reset_lifecycle()


# ---------------------------------------------------------------------------
# Voices + config endpoints
# ---------------------------------------------------------------------------

async def test_list_voices(client) -> None:
    r = await client.get("/api/tts/voices")
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True
    voices = body["data"]
    assert len(voices) == 20
    ids = {v["id"] for v in voices}
    # Spot-check the canonical IDs from the model's params.json.
    assert {"casual_male", "de_female", "fr_male", "ar_male", "hi_female"} <= ids
    # Sanity: shape
    sample = voices[0]
    assert {"id", "display_name", "language", "language_code", "gender"} <= sample.keys()


async def test_get_tts_config(client, tmp_path) -> None:
    r = await client.get("/api/tts/config")
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True
    data = body["data"]
    assert data["sample_rate"] == 24000
    assert data["max_chars"] == settings.tts_max_chars
    assert data["default_voice"] == settings.tts_default_voice
    # output_dir was redirected to the test tmp_path
    assert str(tmp_path) in data["output_dir"]


# ---------------------------------------------------------------------------
# Synthesize — happy paths
# ---------------------------------------------------------------------------

async def test_synthesize_save_default_filename(client, _stub_tts_client) -> None:
    r = await client.post(
        "/api/tts/synthesize",
        json={"text": "Hello world", "voice": "casual_male", "save": True},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["success"] is True
    data = body["data"]
    assert data["voice"] == "casual_male"
    assert data["sample_rate"] == 24000
    assert data["bytes"] > 0
    assert data["filename"].endswith(".wav")
    # Client received the right call
    assert _stub_tts_client.calls == [
        {"text": "Hello world", "voice": "casual_male", "format": "wav"}
    ]


async def test_synthesize_save_custom_filename(client, tmp_path) -> None:
    r = await client.post(
        "/api/tts/synthesize",
        json={
            "text": "Hello",
            "voice": "de_male",
            "save": True,
            "save_filename": "greeting.wav",
        },
    )
    assert r.json()["success"] is True
    assert (tmp_path / "greeting.wav").is_file()


async def test_synthesize_uses_default_voice_when_omitted(client, _stub_tts_client) -> None:
    r = await client.post("/api/tts/synthesize", json={"text": "Hello", "save": True})
    assert r.json()["success"] is True
    assert _stub_tts_client.calls[-1]["voice"] == settings.tts_default_voice


# ---------------------------------------------------------------------------
# Synthesize — negative paths
# ---------------------------------------------------------------------------

async def test_synthesize_rejects_unknown_voice(client) -> None:
    r = await client.post(
        "/api/tts/synthesize",
        json={"text": "Hello", "voice": "klingon", "save": True},
    )
    body = r.json()
    assert body["success"] is False
    assert "voice" in body["error"].lower()


async def test_synthesize_rejects_oversized_text(client) -> None:
    too_long = "A" * (settings.tts_max_chars + 1)
    r = await client.post(
        "/api/tts/synthesize",
        json={"text": too_long, "voice": "casual_male", "save": True},
    )
    body = r.json()
    assert body["success"] is False
    assert "max length" in body["error"].lower()


async def test_synthesize_rejects_empty_text(client) -> None:
    r = await client.post(
        "/api/tts/synthesize",
        json={"text": "   ", "voice": "casual_male", "save": True},
    )
    body = r.json()
    assert body["success"] is False
    assert "empty" in body["error"].lower()


async def test_synthesize_play_mode_streams_wav(client, _stub_tts_client) -> None:
    """save=false returns chunked audio/wav with a streaming WAV header."""
    r = await client.post(
        "/api/tts/synthesize",
        json={"text": "Stream me", "voice": "casual_male", "save": False},
    )
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("audio/wav")

    body = r.content
    # Header (44 bytes) + the two PCM chunks the fake client emits.
    assert len(body) >= 44 + sum(len(c) for c in _stub_tts_client.pcm_chunks)
    # RIFF/WAVE magic
    assert body[:4] == b"RIFF" and body[8:12] == b"WAVE"
    # Streaming placeholder for the RIFF size field (0xFFFFFFFF)
    assert body[4:8] == b"\xff\xff\xff\xff"
    # The data chunk size is also a streaming placeholder.
    data_idx = body.find(b"data")
    assert data_idx > 0
    assert body[data_idx + 4 : data_idx + 8] == b"\xff\xff\xff\xff"
    # The synth_stream method (not the buffered one) was used.
    assert _stub_tts_client.stream_calls == [{"text": "Stream me", "voice": "casual_male"}]
    assert _stub_tts_client.calls == []


async def test_synthesize_play_mode_rejects_unknown_voice(client) -> None:
    """In streaming mode the route returns a 400 with JSON error body."""
    r = await client.post(
        "/api/tts/synthesize",
        json={"text": "Hello", "voice": "klingon", "save": False},
    )
    assert r.status_code == 400
    body = r.json()
    assert body["success"] is False
    assert "voice" in body["error"].lower()


# ---------------------------------------------------------------------------
# Long-form (Phase 5)
# ---------------------------------------------------------------------------

async def test_synthesize_long_form_streams_one_header_then_concatenated_pcm(
    client, _stub_tts_client
) -> None:
    """A multi-sentence input is split into N upstream calls; the response
    body contains one streaming WAV header followed by the concatenated
    PCM from every sentence (no inter-sentence headers)."""
    text = "First sentence. Second sentence. Third sentence."
    r = await client.post(
        "/api/tts/synthesize",
        json={"text": text, "voice": "casual_male", "save": False},
    )
    assert r.status_code == 200
    body = r.content
    # One WAV header (44 bytes) + 3 calls * 2 PCM chunks each.
    expected_pcm_bytes = 3 * sum(len(c) for c in _stub_tts_client.pcm_chunks)
    assert len(body) == 44 + expected_pcm_bytes
    # Three upstream synthesize_stream() calls — one per sentence.
    assert len(_stub_tts_client.stream_calls) == 3
    expected_texts = [
        "First sentence.",
        "Second sentence.",
        "Third sentence.",
    ]
    assert [c["text"] for c in _stub_tts_client.stream_calls] == expected_texts


async def test_synthesize_long_form_save_writes_one_wav(client, tmp_path, _stub_tts_client) -> None:
    """Long-form save concatenates per-sentence PCM into a single WAV file."""
    text = "Sentence A. Sentence B."
    r = await client.post(
        "/api/tts/synthesize",
        json={
            "text": text,
            "voice": "de_male",
            "save": True,
            "save_filename": "longform.wav",
        },
    )
    body = r.json()
    assert body["success"] is True, body
    saved = tmp_path / "longform.wav"
    assert saved.is_file()
    raw = saved.read_bytes()
    # Sanity: 44-byte header, RIFF/WAVE magic, real (not 0xFFFFFFFF) sizes.
    assert raw[:4] == b"RIFF"
    assert raw[8:12] == b"WAVE"
    import struct
    riff_size = struct.unpack("<I", raw[4:8])[0]
    assert riff_size != 0xFFFFFFFF
    assert riff_size == len(raw) - 8
    data_idx = raw.find(b"data")
    data_size = struct.unpack("<I", raw[data_idx + 4 : data_idx + 8])[0]
    assert data_size == len(raw) - data_idx - 8
    # 2 sentences × concatenated stub PCM length = expected payload size.
    expected_pcm = 2 * sum(len(c) for c in _stub_tts_client.pcm_chunks)
    assert data_size == expected_pcm
    # The route fired one upstream synthesize_stream per sentence.
    assert len(_stub_tts_client.stream_calls) == 2


# ---------------------------------------------------------------------------
# Voice cloning (Phase 7)
# ---------------------------------------------------------------------------

@pytest.fixture
def _stub_voice_refs(monkeypatch, tmp_path):
    """Point voice_refs_dir at a tmp dir AND stub ffmpeg/probe so save_ref
    works without a real ffmpeg + audio bytes."""
    refs_dir = tmp_path / "voice_refs"
    refs_dir.mkdir()
    monkeypatch.setattr(settings, "tts_voice_refs_dir", str(refs_dir))

    import wave as _wave
    from voxtral_server.tts import refs as refs_mod

    def _fake_run(src, dst):
        # Write a tiny silent WAV at dst (exactly 6 s).
        with _wave.open(str(dst), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(b"\x00\x00" * (6 * 24000))

    monkeypatch.setattr(refs_mod, "_run_ffmpeg", _fake_run)
    monkeypatch.setattr(refs_mod, "_probe_duration", lambda p: 6.0)
    return refs_dir


async def test_voices_endpoint_includes_kind_field(client) -> None:
    r = await client.get("/api/tts/voices")
    assert r.status_code == 200
    data = r.json()["data"]
    # Built-in voices now carry kind=builtin.
    assert all("kind" in v for v in data)
    assert {v["kind"] for v in data} == {"builtin"}


async def test_upload_voice_ref_happy_path(client, _stub_tts_client, _stub_voice_refs) -> None:
    audio_bytes = b"\x00\x01" * 5000
    r = await client.post(
        "/api/tts/voices/upload",
        files={"audio_sample": ("clip.wav", audio_bytes, "audio/wav")},
        data={"name": "My Voice", "ref_text": "ref transcript", "permission_confirmed": "true"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["success"] is True
    info = body["data"]
    assert info["name"] == "My Voice"
    assert info["kind"] == "cloned"
    # Upstream upload was forwarded.
    assert len(_stub_tts_client.upload_calls) == 1
    assert _stub_tts_client.upload_calls[0]["voice_name"] == info["id"]


async def test_upload_voice_ref_rejects_unconfirmed_permission(client, _stub_voice_refs) -> None:
    r = await client.post(
        "/api/tts/voices/upload",
        files={"audio_sample": ("clip.wav", b"\x00\x01" * 5000, "audio/wav")},
        data={"name": "X", "ref_text": "t", "permission_confirmed": "false"},
    )
    body = r.json()
    assert body["success"] is False
    assert "permission" in body["error"].lower()


async def test_upload_voice_ref_rejects_oversize(client, _stub_voice_refs, monkeypatch) -> None:
    monkeypatch.setattr(settings, "tts_max_ref_audio_bytes", 100)
    r = await client.post(
        "/api/tts/voices/upload",
        files={"audio_sample": ("clip.wav", b"\x00" * 1000, "audio/wav")},
        data={"name": "X", "ref_text": "t", "permission_confirmed": "true"},
    )
    body = r.json()
    assert body["success"] is False
    assert "too large" in body["error"].lower()


async def test_voices_list_includes_uploaded_after_save(client, _stub_tts_client, _stub_voice_refs) -> None:
    await client.post(
        "/api/tts/voices/upload",
        files={"audio_sample": ("clip.wav", b"\x00\x01" * 5000, "audio/wav")},
        data={"name": "My Cloned", "ref_text": "t", "permission_confirmed": "true"},
    )
    r = await client.get("/api/tts/voices")
    voices_data = r.json()["data"]
    cloned = [v for v in voices_data if v.get("kind") == "cloned"]
    assert len(cloned) == 1
    assert cloned[0]["name"] == "My Cloned"


async def test_delete_voice_ref(client, _stub_tts_client, _stub_voice_refs) -> None:
    up = await client.post(
        "/api/tts/voices/upload",
        files={"audio_sample": ("clip.wav", b"\x00\x01" * 5000, "audio/wav")},
        data={"name": "Bye", "ref_text": "t", "permission_confirmed": "true"},
    )
    ref_id = up.json()["data"]["id"]
    r = await client.delete(f"/api/tts/voices/{ref_id}")
    assert r.json()["success"] is True
    assert _stub_tts_client.delete_calls == [ref_id]
    # Now list should return no cloned voices.
    listing = (await client.get("/api/tts/voices")).json()["data"]
    assert all(v.get("kind") == "builtin" for v in listing)


async def test_delete_voice_ref_not_found(client) -> None:
    r = await client.delete("/api/tts/voices/" + "0" * 32)
    body = r.json()
    assert body["success"] is False
    assert "not found" in body["error"].lower()


async def test_delete_voice_ref_rejects_bad_id(client) -> None:
    r = await client.delete("/api/tts/voices/...")
    body = r.json()
    assert body["success"] is False
    assert "voice_ref_id" in body["error"].lower()


async def test_synthesize_with_voice_ref_id_uses_base_task(client, _stub_tts_client, _stub_voice_refs) -> None:
    up = await client.post(
        "/api/tts/voices/upload",
        files={"audio_sample": ("clip.wav", b"\x00\x01" * 5000, "audio/wav")},
        data={"name": "Cloned", "ref_text": "ref", "permission_confirmed": "true"},
    )
    ref_id = up.json()["data"]["id"]
    # Reset the stub call counter so we measure THIS request's behaviour.
    _stub_tts_client.upload_calls.clear()
    _stub_tts_client.calls.clear()

    r = await client.post(
        "/api/tts/synthesize",
        json={
            "text": "Hallo Welt",
            "voice_ref_id": ref_id,
            "save": True,
            "save_filename": "cloned.wav",
        },
    )
    body = r.json()
    assert body["success"] is True, body
    # Lifecycle saw the Base task type.
    assert _stub_tts_client.lifecycle.last_task_type == "Base"
    # Re-sync to upstream happened with the cloned voice's id.
    assert any(c["voice_name"] == ref_id for c in _stub_tts_client.upload_calls)
    # Synth was called with voice=<ref_id>, not the built-in default.
    assert _stub_tts_client.calls[-1]["voice"] == ref_id


async def test_synthesize_unknown_voice_ref_id_rejects(client) -> None:
    r = await client.post(
        "/api/tts/synthesize",
        json={"text": "Hi", "voice_ref_id": "0" * 32, "save": True},
    )
    body = r.json()
    assert body["success"] is False
    assert "not found" in body["error"].lower()


async def test_synthesize_max_chars_cap_unchanged(client) -> None:
    """20000 chars should be allowed; 20001 must reject."""
    just_ok = "A" * settings.tts_max_chars
    over = "A" * (settings.tts_max_chars + 1)

    r1 = await client.post(
        "/api/tts/synthesize",
        json={"text": just_ok, "voice": "casual_male", "save": False},
    )
    # 20k 'A's has no terminators → one chunk → upstream returns canned WAV;
    # we don't care about the body here, only that it wasn't rejected.
    assert r1.status_code == 200, r1.text

    r2 = await client.post(
        "/api/tts/synthesize",
        json={"text": over, "voice": "casual_male", "save": True},
    )
    body = r2.json()
    assert body["success"] is False
    assert "max length" in body["error"].lower()


async def test_synthesize_strips_dir_in_filename(client, tmp_path, _stub_tts_client) -> None:
    """`../foo.wav` strips to `foo.wav` and writes inside the sandbox.

    Real escape attempts that would write outside (`/etc/passwd`,
    `../etc/passwd`) all fail the `.wav` suffix check because Path.name
    drops the suffix. See test_synthesize_rejects_absolute_path.
    """
    r = await client.post(
        "/api/tts/synthesize",
        json={
            "text": "Hello",
            "voice": "casual_male",
            "save": True,
            "save_filename": "../escape.wav",
        },
    )
    body = r.json()
    assert body["success"] is True
    # The file landed inside the sandbox under its stripped name.
    assert (tmp_path / "escape.wav").is_file()
    # Nothing leaked above the sandbox.
    assert not (tmp_path.parent / "escape.wav").exists()


async def test_synthesize_rejects_absolute_path(client) -> None:
    r = await client.post(
        "/api/tts/synthesize",
        json={
            "text": "Hello",
            "voice": "casual_male",
            "save": True,
            "save_filename": "/etc/passwd",
        },
    )
    body = r.json()
    assert body["success"] is False
    assert "filename" in body["error"].lower()


async def test_synthesize_rejects_bad_extension(client) -> None:
    r = await client.post(
        "/api/tts/synthesize",
        json={
            "text": "Hello",
            "voice": "casual_male",
            "save": True,
            "save_filename": "shell.sh",
        },
    )
    body = r.json()
    assert body["success"] is False
    assert "filename" in body["error"].lower()


async def test_synthesize_refuses_overwrite_by_default(client, tmp_path) -> None:
    # Plant an existing file
    (tmp_path / "x.wav").write_bytes(b"existing")

    r = await client.post(
        "/api/tts/synthesize",
        json={
            "text": "Hello",
            "voice": "casual_male",
            "save": True,
            "save_filename": "x.wav",
        },
    )
    body = r.json()
    assert body["success"] is False
    assert "exists" in body["error"].lower()
    assert (tmp_path / "x.wav").read_bytes() == b"existing"


async def test_synthesize_overwrite_when_requested(client, tmp_path) -> None:
    (tmp_path / "x.wav").write_bytes(b"existing")

    r = await client.post(
        "/api/tts/synthesize",
        json={
            "text": "Hello",
            "voice": "casual_male",
            "save": True,
            "save_filename": "x.wav",
            "overwrite": True,
        },
    )
    assert r.json()["success"] is True
    assert (tmp_path / "x.wav").read_bytes() != b"existing"


# ---------------------------------------------------------------------------
# Output management
# ---------------------------------------------------------------------------

async def test_list_outputs_after_synthesize(client) -> None:
    await client.post(
        "/api/tts/synthesize",
        json={"text": "First", "voice": "casual_male", "save": True, "save_filename": "a.wav"},
    )
    await client.post(
        "/api/tts/synthesize",
        json={"text": "Second", "voice": "casual_male", "save": True, "save_filename": "b.wav"},
    )

    r = await client.get("/api/tts/output")
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True
    names = {f["name"] for f in body["data"]}
    assert names == {"a.wav", "b.wav"}


async def test_download_output(client) -> None:
    await client.post(
        "/api/tts/synthesize",
        json={"text": "Hi", "voice": "casual_male", "save": True, "save_filename": "dl.wav"},
    )

    r = await client.get("/api/tts/output/dl.wav")
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/wav"
    assert len(r.content) > 0


async def test_download_rejects_bad_filename(client) -> None:
    """Filenames that fail the sanitizer (no .wav suffix) hit our 400 handler."""
    r = await client.get("/api/tts/output/passwd")
    assert r.status_code == 400


async def test_delete_output_round_trip(client, tmp_path) -> None:
    await client.post(
        "/api/tts/synthesize",
        json={"text": "Hi", "voice": "casual_male", "save": True, "save_filename": "del.wav"},
    )
    assert (tmp_path / "del.wav").is_file()

    r = await client.delete("/api/tts/output/del.wav")
    assert r.json()["success"] is True
    assert not (tmp_path / "del.wav").exists()


async def test_delete_rejects_bad_filename(client) -> None:
    """A filename without the required .wav suffix is rejected by the sanitizer."""
    r = await client.delete("/api/tts/output/passwd")
    body = r.json()
    assert body["success"] is False
    assert "filename" in body["error"].lower()


# ---------------------------------------------------------------------------
# Model-id discovery (fix for the relative-path 404)
# ---------------------------------------------------------------------------

import httpx as _httpx_mod
from voxtral_server.tts.client import TtsClient


class _MockTransport(_httpx_mod.AsyncBaseTransport):
    """Tiny in-memory httpx transport for unit-testing the real TtsClient."""

    def __init__(self, *, models_response: dict | None = None,
                 audio_response: bytes = b"\x00" * 100,
                 audio_status: int = 200) -> None:
        self.models_response = models_response
        self.audio_response = audio_response
        self.audio_status = audio_status
        self.synth_calls: list[dict] = []

    async def handle_async_request(self, request: _httpx_mod.Request) -> _httpx_mod.Response:
        path = request.url.path
        if path.endswith("/models") and request.method == "GET":
            if self.models_response is None:
                return _httpx_mod.Response(500, text="upstream down")
            return _httpx_mod.Response(200, json=self.models_response)
        if path.endswith("/audio/speech") and request.method == "POST":
            import json as _json
            self.synth_calls.append(_json.loads(request.content))
            return _httpx_mod.Response(self.audio_status, content=self.audio_response,
                                       headers={"content-type": "audio/wav"})
        return _httpx_mod.Response(404, text=f"unhandled: {request.method} {path}")


async def test_client_discovers_model_id_from_upstream() -> None:
    """The first synth call queries /v1/models and uses the returned id —
    handles the relative-vs-absolute path mismatch the user hit."""
    transport = _MockTransport(models_response={
        "object": "list",
        "data": [{"id": "/abs/path/to/tts", "object": "model"}],
    })
    client = TtsClient(base_url="http://localhost:8002/v1", model="./relative/path")
    client._client = _httpx_mod.AsyncClient(transport=transport)

    await client.synthesize(text="hi", voice="casual_male", response_format="wav")
    assert client.cached_model_id == "/abs/path/to/tts"
    assert transport.synth_calls[0]["model"] == "/abs/path/to/tts"


async def test_client_falls_back_when_models_endpoint_unreachable() -> None:
    """If /v1/models is broken, the client uses the configured fallback."""
    transport = _MockTransport(models_response=None)  # 500
    client = TtsClient(base_url="http://localhost:8002/v1", model="/configured/path")
    client._client = _httpx_mod.AsyncClient(transport=transport)

    await client.synthesize(text="hi", voice="casual_male", response_format="wav")
    assert client.cached_model_id == "/configured/path"
    assert transport.synth_calls[0]["model"] == "/configured/path"
