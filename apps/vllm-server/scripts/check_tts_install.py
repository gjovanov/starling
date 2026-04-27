"""Verify the vllm + vllm-omni install is sane after `init.sh --tts`.

Exits 0 with a one-line summary on success. Exits non-zero with a clear
explanation on regression. Designed to be invoked as a one-shot:

    python scripts/check_tts_install.py

Checked invariants:

1. Both `vllm` and `vllm_omni` import.
2. `torch` is importable, CUDA is visible.
3. `torch.__version__` is below MAX_ALLOWED_TORCH_MAJOR_MINOR.
   The 0.18 branch of vllm does not work with torch >= 2.11; if a stray
   `pip install --upgrade vllm-omni` bumped torch, the user has to roll
   back. We surface that loudly here rather than letting the user discover
   it the hard way at vllm import time.
4. `vllm_omni.model_executor.models.voxtral_tts` imports — confirms the
   architecture registry was wired up. The very first version of
   vllm-omni 0.18 had a broken egg layout; this catches that.
"""

from __future__ import annotations

import sys
from typing import Final

# vllm 0.18.0 was tested with torch 2.10.x; HF discussion #29 documented
# that torch 2.11+ produces undefined-symbol ImportErrors at vllm import
# time. Bump these constants if/when you upgrade the supported toolchain.
MIN_ALLOWED_TORCH: Final[tuple[int, int]] = (2, 8)
MAX_ALLOWED_TORCH: Final[tuple[int, int]] = (2, 11)   # exclusive upper bound


def _parse_major_minor(version: str) -> tuple[int, int] | None:
    """Best-effort `(major, minor)` parse. Returns None on unparseable input.

    `torch.__version__` looks like ``2.10.0+cu128`` — we only care about the
    leading numeric segment; the CUDA tag is dropped.
    """
    head = version.split("+", 1)[0]
    parts = head.split(".")
    if len(parts) < 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def _fail(msg: str) -> "NoReturn":  # type: ignore[name-defined]
    print(f"[CHECK-TTS] FAIL: {msg}", file=sys.stderr)
    raise SystemExit(1)


def _ok(msg: str) -> None:
    print(f"[CHECK-TTS] OK: {msg}")


def main() -> int:
    # ── 1. torch importable + CUDA ────────────────────────────────────
    try:
        import torch
    except Exception as exc:  # noqa: BLE001 — surface verbatim
        _fail(f"could not import torch: {exc}")

    if not torch.cuda.is_available():
        _fail(
            "torch is installed but CUDA is not visible. "
            "Re-run init.sh after fixing your driver/toolkit."
        )

    parsed = _parse_major_minor(torch.__version__)
    if parsed is None:
        _fail(f"could not parse torch version string {torch.__version__!r}")

    if parsed < MIN_ALLOWED_TORCH:
        _fail(
            f"torch {torch.__version__} is too old (need >= {MIN_ALLOWED_TORCH[0]}.{MIN_ALLOWED_TORCH[1]})"
        )

    if parsed >= MAX_ALLOWED_TORCH:
        _fail(
            f"torch {torch.__version__} is newer than the supported range "
            f"(< {MAX_ALLOWED_TORCH[0]}.{MAX_ALLOWED_TORCH[1]}). "
            "vllm-omni's compiled extensions break against torch >= 2.11. "
            "Roll back with: pip install 'torch<2.11' --index-url "
            "https://download.pytorch.org/whl/cu126"
        )
    _ok(f"torch {torch.__version__} (CUDA available)")

    # ── 2. vllm importable ────────────────────────────────────────────
    try:
        import vllm
    except Exception as exc:  # noqa: BLE001
        _fail(f"could not import vllm: {exc}")
    _ok(f"vllm {vllm.__version__}")

    # ── 3. vllm-omni importable ───────────────────────────────────────
    try:
        import vllm_omni  # noqa: F401 — import side-effect IS the test
    except Exception as exc:  # noqa: BLE001
        _fail(
            f"could not import vllm_omni: {exc}. "
            "If you ran `pip install --upgrade vllm-omni`, that's the "
            "likely cause; reinstall without --upgrade."
        )
    _ok("vllm_omni")

    # ── 4. vllm_omni's voxtral_tts model registry is wired up ─────────
    try:
        from vllm_omni.model_executor.models.voxtral_tts import (  # noqa: F401
            configuration_voxtral_tts,
        )
    except Exception as exc:  # noqa: BLE001
        _fail(
            f"vllm_omni installed but the voxtral_tts model is missing: {exc}. "
            "The egg may be partial — try a clean reinstall."
        )
    _ok("vllm_omni.model_executor.models.voxtral_tts")

    print("[CHECK-TTS] All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
