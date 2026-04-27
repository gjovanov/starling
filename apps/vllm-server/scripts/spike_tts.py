"""Phase 0 TTS spike — verify the local vllm-omni server can synthesize.

Pre-req: vllm-omni serve must be running on http://127.0.0.1:8002 (see
`apps/vllm-server/start-vllm-tts.sh`).

Usage:
    python scripts/spike_tts.py
    python scripts/spike_tts.py --voice de_male --text "Hallo Welt" --out /tmp/tts.wav
    python scripts/spike_tts.py --bench   # synth a 30s utterance and report timing
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import httpx


# Canonical voice list, captured from the model's params.json.
VOICES = [
    "casual_female", "casual_male", "cheerful_female", "neutral_female",
    "neutral_male", "pt_male", "pt_female", "nl_male", "nl_female", "it_male",
    "it_female", "fr_male", "fr_female", "es_male", "es_female", "de_male",
    "de_female", "ar_male", "hi_male", "hi_female",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8002/v1")
    parser.add_argument("--voice", default="casual_male", choices=VOICES)
    parser.add_argument("--text", default="Hello, this is a test of the Voxtral TTS spike.")
    parser.add_argument("--out", default="/tmp/spike_tts.wav")
    parser.add_argument("--format", default="wav", choices=["wav", "pcm", "flac", "mp3", "aac", "opus"])
    parser.add_argument("--model", default="/home/gjovanov/gjovanov/starling/models/cache/tts")
    parser.add_argument(
        "--bench",
        action="store_true",
        help="Run a 30s-utterance benchmark instead of the short test.",
    )
    args = parser.parse_args()

    if args.bench:
        # ~30s of speech is roughly 75 words at 150 wpm (English). Keep it
        # innocuous so the model doesn't go off the rails.
        args.text = (
            "The quick brown fox jumps over the lazy dog. " * 8
        ).strip()
        args.out = "/tmp/spike_tts_30s.wav"

    payload = {
        "model": args.model,
        "input": args.text,
        "voice": args.voice,
        "response_format": args.format,
    }
    print(f"POST {args.base_url}/audio/speech  voice={args.voice}  format={args.format}", flush=True)
    print(f"text ({len(args.text)} chars): {args.text[:80]}{'...' if len(args.text) > 80 else ''}", flush=True)

    t0 = time.perf_counter()
    try:
        resp = httpx.post(f"{args.base_url}/audio/speech", json=payload, timeout=300.0)
    except httpx.HTTPError as exc:
        print(f"[FAIL] HTTP error: {exc}", file=sys.stderr)
        return 2

    elapsed = time.perf_counter() - t0
    if resp.status_code != 200:
        print(f"[FAIL] status={resp.status_code} body={resp.text[:500]}", file=sys.stderr)
        return 3

    audio_bytes = resp.content
    Path(args.out).write_bytes(audio_bytes)
    print(f"[OK]   wrote {len(audio_bytes):,} bytes in {elapsed:.2f}s -> {args.out}")

    # Best-effort decode for sanity check
    try:
        import soundfile as sf  # noqa: WPS433  — optional dep
        import io
        data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        dur = len(data) / sr
        print(
            f"[OK]   audio: {len(data):,} samples @ {sr} Hz = {dur:.2f}s "
            f"(realtime factor {elapsed / dur:.2f}x — lower is faster than realtime)"
        )
    except Exception as exc:  # noqa: BLE001 — diagnostic-only
        print(f"[WARN] could not decode for sanity check: {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
