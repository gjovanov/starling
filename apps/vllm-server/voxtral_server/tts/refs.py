"""Voice-reference storage for zero-shot voice cloning.

Voice cloning is gated behind `task_type="Base"` on the upstream
vllm-omni TTS process. This module owns the local catalog of uploaded
reference audio + the user-supplied transcript:

    voice_refs/<id>.wav       — audio re-encoded to 24 kHz mono PCM16
    voice_refs/<id>.json      — sidecar with name, transcript, permission flag
    voice_refs/_audit.log     — append-only log of every upload/delete

Path-traversal safety reuses the same gauntlet as `tts/storage.py`. The
client never controls the on-disk filename — we always write to
`<uuid>.wav` and store the user-friendly name in the sidecar JSON.

`ffmpeg` is used to re-encode + duration-probe. Tests stub the
`_run_ffmpeg` and `_probe_duration` hooks so they can drive the storage
without ffmpeg installed.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

# Reuse the StorageError sentinel + tests.
from .storage import StorageError


# ── Public types ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class VoiceRef:
    id: str                        # uuid4 — used as on-disk filename + voice id
    name: str                      # user-friendly display name
    ref_text: str                  # transcript of the reference audio
    permission_confirmed: bool
    sample_rate: int               # 24000 after re-encoding
    duration_secs: float
    created_at: float              # unix epoch seconds
    audio_path: str                # absolute path to the .wav
    sidecar_path: str              # absolute path to the .json


# ── Validation rules ──────────────────────────────────────────────────

_NAME_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9 _\-]{1,64}$")
_VOICE_ID_RE: Final[re.Pattern[str]] = re.compile(r"^[a-f0-9-]{8,64}$")
_AUDIO_EXTENSIONS: Final[frozenset[str]] = frozenset({".wav", ".mp3", ".flac", ".ogg", ".m4a"})


def _sanitize_name(name: str) -> str:
    name = name.strip()
    if not _NAME_RE.match(name):
        raise StorageError(
            "name must be 1–64 chars of [A-Za-z0-9 _-]"
        )
    return name


def _sanitize_voice_id(voice_id: str) -> str:
    if not _VOICE_ID_RE.match(voice_id):
        raise StorageError(
            "voice_ref_id must look like a UUID hex"
        )
    return voice_id


# ── Hooks the tests can monkey-patch ──────────────────────────────────


def _run_ffmpeg(src: Path, dst: Path) -> None:
    """Re-encode `src` to 24 kHz mono PCM16 WAV at `dst`. Raises StorageError
    on ffmpeg failure (exit code, missing binary, malformed input)."""
    if shutil.which("ffmpeg") is None:
        raise StorageError("ffmpeg not installed (apt install ffmpeg)")
    proc = subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(src),
            "-ar", "24000",
            "-ac", "1",
            "-acodec", "pcm_s16le",
            str(dst),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if proc.returncode != 0:
        # Surface the last few lines of ffmpeg stderr — they're cryptic but
        # better than nothing.
        tail = (proc.stderr or "").strip().splitlines()[-3:]
        raise StorageError(f"ffmpeg failed (rc={proc.returncode}): {' / '.join(tail)}")


def _probe_duration(path: Path) -> float:
    """Return audio duration in seconds via the stdlib `wave` module
    (assumes the file is a real PCM WAV — i.e. post-`_run_ffmpeg`)."""
    import wave

    with wave.open(str(path), "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate() or 1
        return frames / rate


# ── Public API ────────────────────────────────────────────────────────


def save_ref(
    *,
    audio_bytes: bytes,
    audio_filename: str,
    name: str,
    ref_text: str,
    permission_confirmed: bool,
    voice_refs_dir: Path | str,
    max_audio_bytes: int = 5_000_000,
    min_duration_secs: float = 5.0,
    max_duration_secs: float = 30.0,
    require_permission: bool = True,
    audit_log: bool = True,
    # injectable hooks for tests
    run_ffmpeg: Any = None,
    probe_duration: Any = None,
) -> VoiceRef:
    """Persist a reference audio + transcript. Returns the new VoiceRef.

    `voice_refs_dir` is created on demand. The caller has already pulled
    the bytes out of the multipart upload; we own everything else
    (re-encoding, sanitization, sidecar write).
    """
    if require_permission and not permission_confirmed:
        raise StorageError(
            "permission_confirmed must be true (you must confirm you have "
            "permission from the speaker before uploading their voice)"
        )

    if len(audio_bytes) == 0:
        raise StorageError("audio file is empty")
    if len(audio_bytes) > max_audio_bytes:
        raise StorageError(
            f"audio file is too large ({len(audio_bytes)} > {max_audio_bytes} bytes)"
        )

    name = _sanitize_name(name)
    ref_text = ref_text.strip()
    if not ref_text:
        raise StorageError("ref_text must be a non-empty transcript of the audio")
    if len(ref_text) > 1000:
        raise StorageError("ref_text must be ≤ 1000 chars")

    # Validate the source filename's extension *before* we write anything
    # (prevents accidental text/binary uploads sneaking through).
    src_suffix = Path(audio_filename).suffix.lower()
    if src_suffix not in _AUDIO_EXTENSIONS:
        raise StorageError(
            f"audio_filename must end in one of {sorted(_AUDIO_EXTENSIONS)} (got {src_suffix!r})"
        )

    out_dir = Path(voice_refs_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    voice_id = uuid.uuid4().hex
    src_tmp = out_dir / f"{voice_id}.src{src_suffix}"
    dst_audio = out_dir / f"{voice_id}.wav"
    dst_sidecar = out_dir / f"{voice_id}.json"
    src_tmp.write_bytes(audio_bytes)

    try:
        (run_ffmpeg or _run_ffmpeg)(src_tmp, dst_audio)
    except StorageError:
        src_tmp.unlink(missing_ok=True)
        dst_audio.unlink(missing_ok=True)
        raise
    finally:
        src_tmp.unlink(missing_ok=True)

    duration = (probe_duration or _probe_duration)(dst_audio)
    if duration < min_duration_secs:
        dst_audio.unlink(missing_ok=True)
        raise StorageError(
            f"audio is too short ({duration:.1f}s < {min_duration_secs:.1f}s)"
        )
    if duration > max_duration_secs:
        dst_audio.unlink(missing_ok=True)
        raise StorageError(
            f"audio is too long ({duration:.1f}s > {max_duration_secs:.1f}s)"
        )

    info = VoiceRef(
        id=voice_id,
        name=name,
        ref_text=ref_text,
        permission_confirmed=bool(permission_confirmed),
        sample_rate=24000,
        duration_secs=round(duration, 3),
        created_at=time.time(),
        audio_path=str(dst_audio),
        sidecar_path=str(dst_sidecar),
    )
    dst_sidecar.write_text(json.dumps(_to_dict(info), ensure_ascii=False, indent=2))

    if audit_log:
        _append_audit(out_dir, "upload", info)

    return info


def list_refs(voice_refs_dir: Path | str) -> list[VoiceRef]:
    out_dir = Path(voice_refs_dir).expanduser().resolve()
    if not out_dir.is_dir():
        return []
    refs: list[VoiceRef] = []
    for p in sorted(out_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text())
            refs.append(_from_dict(data))
        except (OSError, ValueError, KeyError) as exc:
            # Don't let one corrupt sidecar take down the whole listing.
            print(f"[refs] WARN: skipping corrupt sidecar {p}: {exc}")
    refs.sort(key=lambda r: r.created_at, reverse=True)
    return refs


def get_ref(voice_refs_dir: Path | str, voice_id: str) -> VoiceRef | None:
    voice_id = _sanitize_voice_id(voice_id)
    sidecar = Path(voice_refs_dir).expanduser().resolve() / f"{voice_id}.json"
    if not sidecar.is_file():
        return None
    try:
        return _from_dict(json.loads(sidecar.read_text()))
    except (OSError, ValueError, KeyError):
        return None


def delete_ref(voice_refs_dir: Path | str, voice_id: str) -> bool:
    voice_id = _sanitize_voice_id(voice_id)
    out_dir = Path(voice_refs_dir).expanduser().resolve()
    audio = out_dir / f"{voice_id}.wav"
    sidecar = out_dir / f"{voice_id}.json"
    if not audio.exists() and not sidecar.exists():
        return False
    info = get_ref(out_dir, voice_id)
    audio.unlink(missing_ok=True)
    sidecar.unlink(missing_ok=True)
    if info is not None:
        _append_audit(out_dir, "delete", info)
    return True


# ── helpers ───────────────────────────────────────────────────────────


def _to_dict(info: VoiceRef) -> dict:
    return {
        "id": info.id,
        "name": info.name,
        "ref_text": info.ref_text,
        "permission_confirmed": info.permission_confirmed,
        "sample_rate": info.sample_rate,
        "duration_secs": info.duration_secs,
        "created_at": info.created_at,
        "audio_path": info.audio_path,
        "sidecar_path": info.sidecar_path,
    }


def _from_dict(data: dict) -> VoiceRef:
    return VoiceRef(
        id=data["id"],
        name=data["name"],
        ref_text=data["ref_text"],
        permission_confirmed=bool(data.get("permission_confirmed", False)),
        sample_rate=int(data.get("sample_rate", 24000)),
        duration_secs=float(data.get("duration_secs", 0)),
        created_at=float(data.get("created_at", 0)),
        audio_path=data["audio_path"],
        sidecar_path=data["sidecar_path"],
    )


def _append_audit(out_dir: Path, action: str, info: VoiceRef) -> None:
    """Append-only audit log. Intentionally minimal — id, name, when,
    permission flag. NEVER includes the audio bytes or the transcript."""
    log = out_dir / "_audit.log"
    when = datetime.fromtimestamp(info.created_at, tz=timezone.utc).isoformat()
    line = (
        f"{when}\t{action}\t{info.id}\t"
        f"name={info.name!r}\t"
        f"permission={info.permission_confirmed}\t"
        f"duration_secs={info.duration_secs}\n"
    )
    try:
        with log.open("a", encoding="utf-8") as fh:
            fh.write(line)
    except OSError as exc:
        print(f"[refs] WARN: audit log write failed: {exc}")
