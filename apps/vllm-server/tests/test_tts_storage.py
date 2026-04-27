"""Unit tests for the TTS path-traversal sanitizer.

This sanitizer is the only thing standing between a client-supplied filename
and an arbitrary write on the server, so we attack it with the usual suspects:
`..`, absolute paths, NUL bytes, length tricks, hidden files, symlinks, and
non-`.wav` extensions.
"""

from __future__ import annotations

import os
import pytest

from voxtral_server.tts import storage


# ---------------------------------------------------------------------------
# sanitize_filename — accept good names
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name",
    [
        "a.wav",
        "name.wav",
        "Name_With-Mixed.case.wav",
        "tts_de_male_20260427T084500Z.wav",
        ("x" * 124) + ".wav",  # 128 chars total: 124 body + ".wav"
    ],
)
def test_sanitize_accepts_valid(name: str) -> None:
    assert storage.sanitize_filename(name) == name


# ---------------------------------------------------------------------------
# sanitize_filename — reject bad names
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name",
    [
        "../etc/passwd",             # Path.name -> "passwd" -> no .wav suffix
        "/etc/passwd",
        ".",
        "..",
        "",
        "   ",                       # whitespace fails the body regex
        "with space.wav",
        "name?.wav",                 # special char
        "name|.wav",
        "name.txt",                  # wrong extension
        "name",                      # missing extension
        ".hidden.wav",               # hidden
        ("y" * 200) + ".wav",        # body too long
        "name\x00trick.wav",         # NUL injection
        "name\nnewline.wav",
    ],
)
def test_sanitize_rejects_bad(name: str) -> None:
    with pytest.raises(storage.StorageError):
        storage.sanitize_filename(name)


def test_sanitize_strips_directory_components() -> None:
    # `sanitize_filename` strips directory components via Path.name. The
    # contained-within escape protection lives in `safe_join`, not here.
    # ".../foo.wav" → "foo.wav" is by design.
    assert storage.sanitize_filename("a/b/foo.wav") == "foo.wav"
    assert storage.sanitize_filename("../foo.wav") == "foo.wav"

    # But once stripped, the residue still has to pass the whitelist:
    # "a/b/.." → ".." → rejected as a directory reference.
    with pytest.raises(storage.StorageError):
        storage.sanitize_filename("a/b/..")


# ---------------------------------------------------------------------------
# safe_join — containment check
# ---------------------------------------------------------------------------

def test_safe_join_inside_dir(tmp_path) -> None:
    target = storage.safe_join(tmp_path, "voice.wav")
    assert target.parent == tmp_path.resolve()
    assert target.name == "voice.wav"


def test_safe_join_strips_traversal(tmp_path) -> None:
    """`../escape.wav` is stripped to `escape.wav` and lands inside the sandbox.

    The escape-prevention boundary is the `.wav`-suffix requirement (which
    catches `/etc/passwd` style absolutes) plus the symlink containment check
    below. See `test_safe_join_rejects_absolute`.
    """
    target = storage.safe_join(tmp_path, "../escape.wav")
    assert target == (tmp_path / "escape.wav").resolve()


def test_safe_join_rejects_absolute(tmp_path) -> None:
    """Absolute paths get their last segment as the candidate filename, which
    lacks the `.wav` suffix → rejected by sanitize_filename."""
    with pytest.raises(storage.StorageError):
        storage.safe_join(tmp_path, "/tmp/oops")  # no .wav suffix
    with pytest.raises(storage.StorageError):
        storage.safe_join(tmp_path, "/etc/passwd")


def test_safe_join_rejects_symlink_to_outside(tmp_path) -> None:
    """If a malicious user (or a prior bad write) plants a symlink in the
    output dir pointing outside, refuse to write through it."""
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    link = out_dir / "evil.wav"
    os.symlink(outside, link)

    with pytest.raises(storage.StorageError):
        storage.safe_join(out_dir, "evil.wav")


def test_safe_join_missing_dir(tmp_path) -> None:
    with pytest.raises(storage.StorageError):
        storage.safe_join(tmp_path / "does-not-exist", "voice.wav")


# ---------------------------------------------------------------------------
# write_wav, list_outputs, delete_output
# ---------------------------------------------------------------------------

def test_write_then_list_then_delete(tmp_path) -> None:
    path = storage.safe_join(tmp_path, "a.wav")
    n = storage.write_wav(path, b"fake-wav-bytes")
    assert n == len(b"fake-wav-bytes")
    assert path.read_bytes() == b"fake-wav-bytes"

    files = storage.list_outputs(tmp_path)
    assert [f.name for f in files] == ["a.wav"]

    assert storage.delete_output(tmp_path, "a.wav") is True
    assert storage.list_outputs(tmp_path) == []


def test_write_refuses_overwrite_by_default(tmp_path) -> None:
    path = storage.safe_join(tmp_path, "x.wav")
    storage.write_wav(path, b"first")
    with pytest.raises(storage.StorageError):
        storage.write_wav(path, b"second")
    # Original survives.
    assert path.read_bytes() == b"first"


def test_write_overwrite_when_requested(tmp_path) -> None:
    path = storage.safe_join(tmp_path, "x.wav")
    storage.write_wav(path, b"first")
    storage.write_wav(path, b"second", overwrite=True)
    assert path.read_bytes() == b"second"


def test_list_outputs_sorts_newest_first(tmp_path) -> None:
    a = storage.safe_join(tmp_path, "a.wav")
    b = storage.safe_join(tmp_path, "b.wav")
    storage.write_wav(a, b"1")
    storage.write_wav(b, b"2")
    # Bump b's mtime to "now + 100s" so it's clearly newer.
    import time
    now = time.time()
    os.utime(a, (now - 100, now - 100))
    os.utime(b, (now, now))

    files = storage.list_outputs(tmp_path)
    assert [f.name for f in files] == ["b.wav", "a.wav"]


def test_list_outputs_skips_non_wav_and_dotfiles(tmp_path) -> None:
    (tmp_path / "ok.wav").write_bytes(b"ok")
    (tmp_path / "skip.txt").write_bytes(b"skip")
    (tmp_path / ".hidden.wav").write_bytes(b"hide")
    files = [f.name for f in storage.list_outputs(tmp_path)]
    assert files == ["ok.wav"]


def test_delete_returns_false_for_missing(tmp_path) -> None:
    assert storage.delete_output(tmp_path, "ghost.wav") is False


def test_delete_strips_dir_components(tmp_path) -> None:
    """Like safe_join, delete_output strips the dir part. `../bypass.wav` → `bypass.wav`,
    which doesn't exist in the sandbox, so the call returns False (no escape)."""
    assert storage.delete_output(tmp_path, "../bypass.wav") is False


def test_delete_rejects_bad_extension(tmp_path) -> None:
    with pytest.raises(storage.StorageError):
        storage.delete_output(tmp_path, "../etc/passwd")  # no .wav


def test_auto_filename_passes_validation() -> None:
    name = storage.auto_filename("de_male")
    # Auto-generated names must round-trip through sanitize_filename.
    assert storage.sanitize_filename(name) == name
    assert name.startswith("tts_de_male_")
    assert name.endswith(".wav")
