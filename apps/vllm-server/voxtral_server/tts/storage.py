"""Path-traversal-safe filesystem helpers for synthesized TTS output.

Goal: callers can supply a filename like `"my_voice.wav"` but cannot escape
the configured output directory via `..`, absolute paths, symlinks, or
encoding tricks. All sanitization happens here so reviewers have ONE place
to audit.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# Whitelist: alphanumeric, dot, dash, underscore. Case-insensitive. Length 1–128.
# We anchor with ^…$ in the validator below; the pattern itself is just the
# character class.
_FILENAME_BODY_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
_REQUIRED_SUFFIX = ".wav"


class StorageError(ValueError):
    """Raised when a filename / path argument fails validation."""


@dataclass(frozen=True)
class OutputFile:
    name: str           # Filename only, no directory.
    path: str           # Absolute path.
    size_bytes: int
    created_at: float   # Unix epoch seconds.


def sanitize_filename(filename: str) -> str:
    """Normalize a user-supplied filename and reject anything suspicious.

    Returns the canonical filename (always ending in `.wav`). Raises
    `StorageError` on any rejection. NEVER returns a path — callers must
    join with `output_dir` via `safe_join()`.
    """
    if not isinstance(filename, str):
        raise StorageError("filename must be a string")

    # Strip any directory components a client might have prefixed. `Path.name`
    # collapses both `/` and `\`, returns "" for "." or "..".
    base = Path(filename).name
    if not base or base in {".", ".."}:
        raise StorageError("filename must not be empty or refer to a directory")

    # Reject embedded NUL — `os.path` and most syscalls choke on these.
    if "\x00" in base:
        raise StorageError("filename must not contain NUL bytes")

    # Force the .wav suffix. Reject anything else *before* whitelist check so
    # we give a useful error.
    if not base.lower().endswith(_REQUIRED_SUFFIX):
        raise StorageError(f"filename must end with {_REQUIRED_SUFFIX}")

    if not _FILENAME_BODY_RE.match(base):
        raise StorageError(
            "filename must be 1–128 chars of [A-Za-z0-9._-] and end in .wav"
        )

    # Hidden files would clutter `ls` and confuse the user — disallow.
    if base.startswith("."):
        raise StorageError("filename must not start with a dot")

    return base


def safe_join(output_dir: Path | str, filename: str) -> Path:
    """Resolve `output_dir / sanitize_filename(filename)` and verify the
    result is contained within `output_dir`.

    The contained-within check defeats symlink games and any residual
    traversal that slipped past `sanitize_filename`. We resolve the parent
    (which must exist) and re-check.
    """
    base = sanitize_filename(filename)
    out_dir = Path(output_dir).resolve()

    if not out_dir.is_dir():
        raise StorageError(f"output directory does not exist: {out_dir}")

    candidate = (out_dir / base).resolve()

    # `Path.is_relative_to` is the single canonical check. Available in 3.9+.
    if not candidate.is_relative_to(out_dir):
        raise StorageError(
            "filename resolves outside the output directory (refusing to write)"
        )

    return candidate


def write_wav(path: Path, audio_bytes: bytes, *, overwrite: bool = False) -> int:
    """Write `audio_bytes` to `path`. Returns bytes written.

    By default refuses to overwrite an existing file (returns the same error
    type as the validator so the route can map both to a single 409/422).
    """
    if path.exists() and not overwrite:
        raise StorageError(f"file already exists (pass overwrite=true): {path.name}")

    # Atomic-ish write: write to a temp file in the same dir, then rename.
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(audio_bytes)
    os.replace(tmp, path)
    return len(audio_bytes)


def list_outputs(output_dir: Path | str) -> list[OutputFile]:
    """Return all `.wav` files in `output_dir`, newest first."""
    out_dir = Path(output_dir).resolve()
    if not out_dir.is_dir():
        return []

    items: list[OutputFile] = []
    for p in out_dir.iterdir():
        # Skip non-files, non-wav, dotfiles, anything we wouldn't have written.
        if not p.is_file():
            continue
        if p.suffix.lower() != _REQUIRED_SUFFIX:
            continue
        if p.name.startswith("."):
            continue
        try:
            stat = p.stat()
        except OSError:
            continue
        items.append(
            OutputFile(
                name=p.name,
                path=str(p),
                size_bytes=stat.st_size,
                created_at=stat.st_mtime,
            )
        )

    items.sort(key=lambda o: o.created_at, reverse=True)
    return items


def delete_output(output_dir: Path | str, filename: str) -> bool:
    """Delete a file under `output_dir`. Returns True if removed, False if
    the file didn't exist. Validates the filename through the same gauntlet
    as `safe_join`."""
    target = safe_join(output_dir, filename)
    try:
        target.unlink()
        return True
    except FileNotFoundError:
        return False


def auto_filename(voice: str) -> str:
    """Generate a filename when the caller doesn't supply one.

    Format: `tts_<voice>_<utc-timestamp>.wav` — sortable, traceable, and
    passes `sanitize_filename` by construction.
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    # voice IDs are already constrained to [a-z0-9_], no extra escape needed.
    return f"tts_{voice}_{ts}.wav"
