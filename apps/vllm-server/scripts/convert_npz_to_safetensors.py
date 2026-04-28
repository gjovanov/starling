#!/usr/bin/env python3
"""Convert tts_golden/*.npz fixtures to matching .safetensors.

Why: the burn-server Rust crate already has `safetensors` as a dep but
no `.npz` reader. Rather than add a new Rust dep just for tests, we
ship a parallel `.safetensors` for every captured `.npz`. Both formats
are committed; the npz remains for human-readable numpy inspection.

Usage:
    python3 scripts/convert_npz_to_safetensors.py
    python3 scripts/convert_npz_to_safetensors.py --dir <custom_dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file


def convert_one(npz_path: Path) -> Path:
    out_path = npz_path.with_suffix(".safetensors")
    metadata: dict[str, str] = {}
    tensors: dict[str, np.ndarray] = {}
    with np.load(npz_path) as data:
        for k in data.files:
            arr = data[k]
            # Python str values come back as 0-d numpy arrays of dtype <U…>.
            if arr.dtype.kind == "U":
                metadata[k] = str(arr.item() if arr.ndim == 0 else arr.tolist())
                continue
            # Make sure the array is contiguous and a safetensors-supported
            # dtype. Anything `int*` or `*float*` is fine; uint16/bool need
            # casting.
            if arr.dtype == np.bool_:
                arr = arr.astype(np.uint8)
            tensors[k] = np.ascontiguousarray(arr)
    save_file(tensors, str(out_path), metadata=metadata or None)
    return out_path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dir",
        default="/home/gjovanov/gjovanov/starling/apps/burn-server/test_data/tts_golden",
    )
    args = p.parse_args()

    root = Path(args.dir)
    if not root.is_dir():
        print(f"[FAIL] {root} is not a directory", file=sys.stderr)
        return 1

    npz_files = sorted(root.glob("*.npz"))
    if not npz_files:
        print(f"[FAIL] no .npz files under {root}", file=sys.stderr)
        return 1

    for npz in npz_files:
        out = convert_one(npz)
        print(f"  {npz.name} → {out.name} ({out.stat().st_size:,} bytes)")
    print(f"\nConverted {len(npz_files)} fixtures.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
