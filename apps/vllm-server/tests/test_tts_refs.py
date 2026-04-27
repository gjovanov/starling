"""Tests for the voice-reference storage layer.

We never invoke the real ffmpeg — tests inject `run_ffmpeg` and
`probe_duration` hooks. The point is to exercise the validation gauntlet
and the sidecar/audit-log persistence, not ffmpeg.
"""

from __future__ import annotations

import json
import wave

import pytest

from voxtral_server.tts import refs as refs_mod
from voxtral_server.tts.storage import StorageError


# ── helpers ───────────────────────────────────────────────────────────


def _stub_ffmpeg(duration_secs: float = 6.0):
    """Return a (run_ffmpeg, probe_duration) pair that fakes a successful
    re-encode by writing a tiny silent WAV at `dst`."""

    def run_ffmpeg(src, dst):
        n_samples = int(duration_secs * 24000)
        with wave.open(str(dst), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(b"\x00\x00" * n_samples)

    def probe_duration(path):
        return duration_secs

    return run_ffmpeg, probe_duration


# ── Save happy paths ─────────────────────────────────────────────────


def test_save_ref_writes_audio_and_sidecar(tmp_path) -> None:
    run_ff, probe = _stub_ffmpeg(duration_secs=10.0)
    info = refs_mod.save_ref(
        audio_bytes=b"FAKE-MP3-DATA",
        audio_filename="my voice sample.mp3",
        name="My Voice",
        ref_text="The quick brown fox jumps over the lazy dog.",
        permission_confirmed=True,
        voice_refs_dir=tmp_path,
        run_ffmpeg=run_ff,
        probe_duration=probe,
    )
    assert info.name == "My Voice"
    assert info.duration_secs == 10.0
    assert info.permission_confirmed is True
    audio = tmp_path / f"{info.id}.wav"
    sidecar = tmp_path / f"{info.id}.json"
    assert audio.is_file()
    assert sidecar.is_file()

    # Sidecar JSON round-trips.
    data = json.loads(sidecar.read_text())
    assert data["name"] == "My Voice"
    assert data["ref_text"].startswith("The quick brown fox")
    assert data["sample_rate"] == 24000

    # Audit log was appended to.
    audit = (tmp_path / "_audit.log").read_text()
    assert "upload" in audit
    assert info.id in audit
    # Audit must NOT contain the transcript or audio bytes.
    assert "quick brown fox" not in audit


def test_list_then_delete_round_trips(tmp_path) -> None:
    run_ff, probe = _stub_ffmpeg()
    a = refs_mod.save_ref(
        audio_bytes=b"x" * 100, audio_filename="a.wav", name="A",
        ref_text="ref-a", permission_confirmed=True, voice_refs_dir=tmp_path,
        run_ffmpeg=run_ff, probe_duration=probe,
    )
    b = refs_mod.save_ref(
        audio_bytes=b"y" * 100, audio_filename="b.wav", name="B",
        ref_text="ref-b", permission_confirmed=True, voice_refs_dir=tmp_path,
        run_ffmpeg=run_ff, probe_duration=probe,
    )
    listed = {r.id: r.name for r in refs_mod.list_refs(tmp_path)}
    assert listed == {a.id: "A", b.id: "B"}

    assert refs_mod.delete_ref(tmp_path, a.id) is True
    assert refs_mod.delete_ref(tmp_path, a.id) is False    # already gone
    assert {r.id for r in refs_mod.list_refs(tmp_path)} == {b.id}


def test_get_ref_returns_none_for_unknown(tmp_path) -> None:
    # Missing → None (must NOT raise just because the id wasn't found —
    # caller maps None to a 404).
    assert refs_mod.get_ref(tmp_path, "deadbeefdeadbeef") is None


# ── Validation: permission ───────────────────────────────────────────


def test_save_ref_rejects_when_permission_not_confirmed(tmp_path) -> None:
    run_ff, probe = _stub_ffmpeg()
    with pytest.raises(StorageError, match="permission_confirmed"):
        refs_mod.save_ref(
            audio_bytes=b"x", audio_filename="a.wav", name="A", ref_text="t",
            permission_confirmed=False, voice_refs_dir=tmp_path,
            run_ffmpeg=run_ff, probe_duration=probe,
        )


def test_save_ref_permission_check_can_be_disabled(tmp_path) -> None:
    """Local-dev mode where the operator turns off the gate."""
    run_ff, probe = _stub_ffmpeg()
    info = refs_mod.save_ref(
        audio_bytes=b"x", audio_filename="a.wav", name="A", ref_text="t",
        permission_confirmed=False, voice_refs_dir=tmp_path,
        require_permission=False,
        run_ffmpeg=run_ff, probe_duration=probe,
    )
    assert info.permission_confirmed is False


# ── Validation: size + duration ──────────────────────────────────────


def test_save_ref_rejects_oversize(tmp_path) -> None:
    run_ff, probe = _stub_ffmpeg()
    with pytest.raises(StorageError, match="too large"):
        refs_mod.save_ref(
            audio_bytes=b"x" * 10_000_000,
            audio_filename="a.wav", name="A", ref_text="t",
            permission_confirmed=True, voice_refs_dir=tmp_path,
            max_audio_bytes=5_000_000,
            run_ffmpeg=run_ff, probe_duration=probe,
        )


def test_save_ref_rejects_too_short(tmp_path) -> None:
    run_ff, probe = _stub_ffmpeg(duration_secs=2.0)
    with pytest.raises(StorageError, match="too short"):
        refs_mod.save_ref(
            audio_bytes=b"x", audio_filename="a.wav", name="A", ref_text="t",
            permission_confirmed=True, voice_refs_dir=tmp_path,
            min_duration_secs=5.0,
            run_ffmpeg=run_ff, probe_duration=probe,
        )


def test_save_ref_rejects_too_long(tmp_path) -> None:
    run_ff, probe = _stub_ffmpeg(duration_secs=120.0)
    with pytest.raises(StorageError, match="too long"):
        refs_mod.save_ref(
            audio_bytes=b"x", audio_filename="a.wav", name="A", ref_text="t",
            permission_confirmed=True, voice_refs_dir=tmp_path,
            max_duration_secs=30.0,
            run_ffmpeg=run_ff, probe_duration=probe,
        )


# ── Validation: name + filename ──────────────────────────────────────


@pytest.mark.parametrize("name", ["", "   ", "a" * 65, "name<script>", "../etc"])
def test_save_ref_rejects_bad_name(tmp_path, name) -> None:
    run_ff, probe = _stub_ffmpeg()
    with pytest.raises(StorageError, match="name"):
        refs_mod.save_ref(
            audio_bytes=b"x", audio_filename="a.wav", name=name, ref_text="t",
            permission_confirmed=True, voice_refs_dir=tmp_path,
            run_ffmpeg=run_ff, probe_duration=probe,
        )


@pytest.mark.parametrize("filename", ["a.exe", "a.txt", "a", "a.WAV.exe"])
def test_save_ref_rejects_non_audio_filename(tmp_path, filename) -> None:
    run_ff, probe = _stub_ffmpeg()
    with pytest.raises(StorageError, match="audio_filename"):
        refs_mod.save_ref(
            audio_bytes=b"x", audio_filename=filename, name="A", ref_text="t",
            permission_confirmed=True, voice_refs_dir=tmp_path,
            run_ffmpeg=run_ff, probe_duration=probe,
        )


# ── Validation: voice_id ─────────────────────────────────────────────


@pytest.mark.parametrize("vid", ["../escape", "..", "/etc", "x" * 200, "@@@"])
def test_voice_id_sanitizer_rejects_traversal(tmp_path, vid) -> None:
    with pytest.raises(StorageError, match="voice_ref_id"):
        refs_mod.delete_ref(tmp_path, vid)


def test_get_ref_rejects_bad_voice_id(tmp_path) -> None:
    with pytest.raises(StorageError, match="voice_ref_id"):
        refs_mod.get_ref(tmp_path, "..")


# ── ref_text validation ──────────────────────────────────────────────


def test_save_ref_rejects_empty_ref_text(tmp_path) -> None:
    run_ff, probe = _stub_ffmpeg()
    with pytest.raises(StorageError, match="ref_text"):
        refs_mod.save_ref(
            audio_bytes=b"x", audio_filename="a.wav", name="A", ref_text="   ",
            permission_confirmed=True, voice_refs_dir=tmp_path,
            run_ffmpeg=run_ff, probe_duration=probe,
        )


def test_save_ref_rejects_huge_ref_text(tmp_path) -> None:
    run_ff, probe = _stub_ffmpeg()
    with pytest.raises(StorageError, match="ref_text"):
        refs_mod.save_ref(
            audio_bytes=b"x", audio_filename="a.wav", name="A", ref_text="x" * 1500,
            permission_confirmed=True, voice_refs_dir=tmp_path,
            run_ffmpeg=run_ff, probe_duration=probe,
        )
