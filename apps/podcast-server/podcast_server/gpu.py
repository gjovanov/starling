"""Free-VRAM probe (nvidia-smi), shared by the engine and the CLI."""

from __future__ import annotations

import shutil
import subprocess


def gpu_free_gib() -> float | None:
    """Free memory on GPU 0 in GiB, or None when it cannot be read."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5.0, check=True,
        ).stdout
        return float(out.strip().splitlines()[0]) / 1024.0
    except (subprocess.SubprocessError, OSError, ValueError, IndexError):
        return None
