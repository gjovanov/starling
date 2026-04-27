#!/usr/bin/env python3
"""Capture end-to-end TTS golden references via the running vllm-omni HTTP API.

For each canonical (text, voice) pair, hits `/v1/audio/speech` and saves:
  - <output_dir>/e2e_<name>.wav   — raw 24 kHz mono PCM wrapped in WAV
  - <output_dir>/e2e_<name>.json  — request body + timing + audio stats sidecar

These are the bit-mostly-exact reference outputs the Rust port must
match (deterministic up to the `torch.randn` seed in flow-matching, which
upstream does not currently fix — so e2e refs are seed-dependent).

Pre-req: vllm-omni TTS running on localhost:8002 (`./start-vllm-tts.sh`
under apps/vllm-server, or via the lifecycle manager — first /api/tts/*
request from the FastAPI server auto-spawns it).
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import httpx


CANONICAL_INPUTS = [
    # name, voice, text
    ("en_short_neutral_male", "neutral_male", "Hello, this is a test."),
    ("en_short_neutral_female", "neutral_female", "Hello, this is a test."),
    ("de_short_de_male", "de_male", "Hallo, das ist ein Test."),
    ("fr_short_fr_female", "fr_female", "Bonjour, ceci est un test."),
    (
        "en_medium_casual_male",
        "casual_male",
        "The quick brown fox jumps over the lazy dog. The early bird catches the worm.",
    ),
    (
        "en_long_cheerful_female",
        "cheerful_female",
        "When in the course of human events it becomes necessary for one people to dissolve "
        "the political bands which have connected them with another, and to assume among "
        "the powers of the earth the separate and equal station to which the laws of nature entitle them.",
    ),
]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:8002/v1")
    p.add_argument("--model-path", default="/home/gjovanov/gjovanov/starling/models/cache/tts")
    p.add_argument(
        "--output-dir",
        default="/home/gjovanov/gjovanov/starling/apps/burn-server/test_data/tts_golden",
    )
    p.add_argument("--timeout", type=float, default=300.0)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    try:
        import soundfile as sf  # type: ignore
    except ImportError:
        sf = None
        print("[WARN] soundfile not installed; sanity stats will be omitted")

    for name, voice, text in CANONICAL_INPUTS:
        wav_path = out / f"e2e_{name}.wav"
        meta_path = out / f"e2e_{name}.json"

        payload = {
            "model": args.model_path,
            "input": text,
            "voice": voice,
            "response_format": "wav",
        }
        print(f"\n[{name}] voice={voice} chars={len(text)}")
        t0 = time.perf_counter()
        try:
            resp = httpx.post(f"{args.base_url}/audio/speech", json=payload, timeout=args.timeout)
        except httpx.HTTPError as exc:
            print(f"  [FAIL] HTTP error: {exc}")
            continue
        elapsed = time.perf_counter() - t0

        if resp.status_code != 200:
            print(f"  [FAIL] status={resp.status_code} body={resp.text[:300]}")
            continue

        audio_bytes = resp.content
        wav_path.write_bytes(audio_bytes)

        meta = {
            "name": name,
            "voice": voice,
            "text": text,
            "request_payload": payload,
            "elapsed_seconds": elapsed,
            "wav_bytes": len(audio_bytes),
        }
        if sf is not None:
            try:
                data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
                meta.update(
                    sample_rate=int(sr),
                    samples=int(len(data)),
                    duration_seconds=float(len(data) / sr),
                    rms_dbfs=float(20 * (data**2).mean() ** 0.5),  # raw RMS, not dBFS yet
                    peak=float(abs(data).max()),
                )
                # convert raw RMS to dBFS
                import math
                rms = (data**2).mean() ** 0.5
                meta["rms_dbfs"] = float(20 * math.log10(rms + 1e-12))
            except Exception as exc:  # noqa: BLE001 — diagnostic-only
                meta["sanity_check_error"] = str(exc)

        meta_path.write_text(json.dumps(meta, indent=2))
        dur_str = f" {meta.get('duration_seconds', 0):.2f}s @ {meta.get('sample_rate', '?')} Hz" if "duration_seconds" in meta else ""
        rtf_str = f" rtf={elapsed / meta['duration_seconds']:.2f}x" if "duration_seconds" in meta and meta["duration_seconds"] > 0 else ""
        print(f"  → {wav_path}  ({len(audio_bytes):,} bytes,{dur_str}, elapsed={elapsed:.2f}s{rtf_str})")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
