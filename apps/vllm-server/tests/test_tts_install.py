"""Tests for the TTS install verification script.

The script's whole job is to fail loudly when the venv has drifted into a
state vllm-omni won't tolerate (most importantly: torch >= 2.11 dragged in
by an accidental `pip install --upgrade vllm-omni`). We exercise the
version-parser directly, then run the script as a subprocess and assert it
exits 0 against the current venv (which has a known-good install — the
spike step set this up).
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_tts_install.py"


def _load_script_module():
    """Import `check_tts_install` as a module so we can unit-test the helpers."""
    spec = importlib.util.spec_from_file_location("check_tts_install", SCRIPT)
    if spec is None or spec.loader is None:
        pytest.skip("Could not load check_tts_install.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_parse_major_minor_strips_cuda_tag() -> None:
    mod = _load_script_module()
    assert mod._parse_major_minor("2.10.0+cu128") == (2, 10)
    assert mod._parse_major_minor("2.11.1") == (2, 11)
    assert mod._parse_major_minor("3.0.0") == (3, 0)


def test_parse_major_minor_rejects_garbage() -> None:
    mod = _load_script_module()
    assert mod._parse_major_minor("nightly") is None
    assert mod._parse_major_minor("2") is None
    assert mod._parse_major_minor("") is None


def test_torch_version_is_below_max_supported() -> None:
    """Sanity-check the constants: the currently-pinned torch in our venv
    must fall in the supported range. If this fails, either the venv has
    drifted (real bug — fix the install) or the constants need bumping
    (intentional toolchain upgrade — bump them deliberately and update the
    spike memo)."""
    mod = _load_script_module()
    parsed = mod._parse_major_minor("2.10.0+cu128")  # what init.sh installs today
    assert parsed is not None
    assert parsed >= mod.MIN_ALLOWED_TORCH
    assert parsed < mod.MAX_ALLOWED_TORCH


@pytest.mark.skipif(
    importlib.util.find_spec("vllm_omni") is None,
    reason="vllm_omni not installed (run init.sh --tts first)",
)
def test_check_tts_install_exits_zero_against_current_venv() -> None:
    """End-to-end: with the spike's install in place, the script must pass."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"check_tts_install.py failed.\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert "All checks passed" in result.stdout
