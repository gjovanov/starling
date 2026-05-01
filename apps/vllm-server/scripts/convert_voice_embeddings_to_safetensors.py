#!/usr/bin/env python3
"""Convert voice_embedding/*.pt to companion .safetensors files.

Each upstream voice file is a single torch.Tensor [163, 3072] BF16. We
write a parallel .safetensors with one entry named `voice` so the Rust
port can mmap-load them via the safetensors crate (no .pt parser
needed).

Usage:
    python3 scripts/convert_voice_embeddings_to_safetensors.py
    python3 scripts/convert_voice_embeddings_to_safetensors.py --dir <custom>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dir",
        default="/home/gjovanov/gjovanov/starling/models/cache/tts/voice_embedding",
    )
    args = p.parse_args()

    root = Path(args.dir)
    if not root.is_dir():
        print(f"[FAIL] {root} not a directory", file=sys.stderr)
        return 1

    pt_files = sorted(root.glob("*.pt"))
    if not pt_files:
        print(f"[FAIL] no .pt files under {root}", file=sys.stderr)
        return 1

    for pt in pt_files:
        out = pt.with_suffix(".safetensors")
        t = torch.load(pt, map_location="cpu", weights_only=True)
        if not isinstance(t, torch.Tensor):
            print(f"  [SKIP] {pt.name}: not a Tensor (got {type(t)})")
            continue
        save_file({"voice": t.contiguous()}, str(out))
        print(f"  {pt.name} → {out.name}  shape={list(t.shape)}  dtype={t.dtype}")
    print(f"\nConverted {len(pt_files)} voices.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
