#!/usr/bin/env python3
"""
Benchmark comparison: burn-server (candle-native-flash) vs vllm-server.
Runs both on broadcast_1.wav (5 minutes) and outputs segment-level results.

Usage:
  python3 scripts/benchmark_comparison.py [--duration 300]

Outputs:
  results/burn_segments.txt    — timestamped segments
  results/vllm_segments.txt    — timestamped segments
  results/comparison.txt       — side-by-side metrics

Requires: jiwer (pip install jiwer)
"""

import argparse
import asyncio
import base64
import json
import os
import re
import struct
import subprocess
import sys
import time
from pathlib import Path

# Reference transcript
MEDIA_DIR = Path(__file__).parent.parent / "media"
REF_FILE = MEDIA_DIR / "broadcast_1.txt"
AUDIO_FILE = MEDIA_DIR / "broadcast_1.wav"

def parse_reference(path, max_words=None):
    """Parse broadcast_1.txt: strip speaker labels and timestamps."""
    lines = []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("SPK_") or (len(line) < 6 and ":" in line):
            continue
        lines.append(line)
    text = " ".join(lines)
    if max_words:
        text = " ".join(text.split()[:max_words])
    return text

def compute_wer(ref, hyp):
    """Word Error Rate."""
    try:
        from jiwer import wer
        return wer(ref.lower(), hyp.lower())
    except ImportError:
        return -1.0

def compute_cer(ref, hyp):
    """Character Error Rate."""
    try:
        from jiwer import cer
        return cer(ref.lower(), hyp.lower())
    except ImportError:
        return -1.0

def extract_sentences(text):
    """Split text into sentences at [.!?] + uppercase boundaries."""
    pattern = re.compile(r'(?<=[a-zäöüß][.!?])\s+(?=[A-ZÄÖÜ])')
    parts = pattern.split(text.strip())
    return [p.strip() for p in parts if p.strip()]

def run_burn_benchmark(duration, backend="candle-native-flash"):
    """Run burn-server benchmark via subprocess."""
    binary = Path(__file__).parent.parent / "target" / "release" / "benchmark"
    if not binary.exists():
        binary = Path(__file__).parent.parent / "apps" / "burn-server" / "target" / "release" / "benchmark"

    # Create temp 5min clip
    clip_path = f"/tmp/broadcast_{duration}s.wav"
    subprocess.run([
        "ffmpeg", "-y", "-i", str(AUDIO_FILE), "-t", str(duration),
        "-ar", "16000", "-ac", "1", "-acodec", "pcm_s16le", clip_path
    ], capture_output=True)

    env = os.environ.copy()
    env["CANDLE_STREAMING"] = "1"

    t0 = time.monotonic()
    result = subprocess.run(
        [str(binary), "--backend", backend, "--audio", clip_path, "--models-dir", "models/cache"],
        capture_output=True, text=True, env=env, cwd=str(Path(__file__).parent.parent)
    )
    elapsed = time.monotonic() - t0

    text = result.stdout.strip()
    stderr = result.stderr

    # Parse metrics from stderr
    metrics = {"total_time": elapsed, "text": text, "stderr": stderr}
    for line in stderr.split("\n"):
        if "Decode:" in line:
            m = re.search(r"(\d+) steps.*?(\d+)ms.*?([\d.]+)ms/step.*?(\d+) text tokens.*?(\d+) rotations", line)
            if m:
                metrics["decode_steps"] = int(m.group(1))
                metrics["decode_ms"] = int(m.group(2))
                metrics["ms_per_step"] = float(m.group(3))
                metrics["text_tokens"] = int(m.group(4))
                metrics["rotations"] = int(m.group(5))
        if "Encoder:" in line:
            m = re.search(r"(\d+) adapter.*?([\d.]+)s", line)
            if m:
                metrics["adapter_tokens"] = int(m.group(1))
                metrics["encoder_secs"] = float(m.group(2))

    return metrics

async def run_vllm_benchmark(duration):
    """Run vllm benchmark via WebSocket."""
    import soundfile as sf
    import websockets

    data, sr = sf.read(str(AUDIO_FILE), dtype='float32')
    data = data[:duration * 16000]
    audio_secs = len(data) / 16000

    ws = await websockets.connect("ws://localhost:8001/v1/realtime", max_size=10*1024*1024)
    await ws.send(json.dumps({
        "type": "session.update",
        "model": "mistralai/Voxtral-Mini-4B-Realtime-2602",
        "input_audio_format": "pcm16", "language": "de", "turn_detection": None,
    }))
    for _ in range(5):
        try:
            r = await asyncio.wait_for(ws.recv(), timeout=2.0)
            if "session.updated" in r: break
        except: break

    all_text = []
    async def reader():
        try:
            async for msg in ws:
                d = json.loads(msg)
                if "delta" in d.get("type", ""):
                    all_text.append(d.get("delta", ""))
        except: pass

    rtask = asyncio.create_task(reader())
    t0 = time.monotonic()
    batch = 8000
    nc = 0

    for i in range(0, len(data), batch):
        chunk = data[i:i+batch]
        raw = struct.pack(f"<{len(chunk)}h", *[int(max(-32768, min(32767, s*32768))) for s in chunk])
        await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(raw).decode()}))
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        nc += 1
        await asyncio.sleep(0.05)
        if nc >= 200:
            rtask.cancel()
            try: await rtask
            except: pass
            await ws.close()
            ws = await websockets.connect("ws://localhost:8001/v1/realtime", max_size=10*1024*1024)
            await ws.send(json.dumps({
                "type": "session.update",
                "model": "mistralai/Voxtral-Mini-4B-Realtime-2602",
                "input_audio_format": "pcm16", "language": "de", "turn_detection": None,
            }))
            for _ in range(5):
                try:
                    r = await asyncio.wait_for(ws.recv(), timeout=2.0)
                    if "session.updated" in r: break
                except: break
            rtask = asyncio.create_task(reader())
            nc = 0

    await asyncio.sleep(3.0)
    elapsed = time.monotonic() - t0
    rtask.cancel()
    try: await rtask
    except: pass
    try: await ws.close()
    except: pass

    text = "".join(all_text)
    return {"total_time": elapsed, "text": text, "commits": nc}

def write_segments(text, out_path, label):
    """Write segment-level output."""
    sentences = extract_sentences(text)
    with open(out_path, "w") as f:
        f.write(f"# {label} — {len(sentences)} segments\n")
        f.write(f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        for i, s in enumerate(sentences):
            f.write(f"[seg {i+1:3d}] FINAL: {s}\n")
    return sentences

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=300)
    parser.add_argument("--skip-vllm", action="store_true")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    ref_text = parse_reference(REF_FILE)
    print(f"Reference: {len(ref_text.split())} words", file=sys.stderr)

    # Run burn-server
    print("Running burn-server (candle-native-flash)...", file=sys.stderr)
    burn = run_burn_benchmark(args.duration)
    burn_segs = write_segments(burn["text"], "results/burn_segments.txt", "burn-server (candle 0.10 + flash)")
    print(f"  Done: {burn.get('total_time', 0):.1f}s, {len(burn['text'].split())} words", file=sys.stderr)

    # Run vllm
    vllm = None
    if not args.skip_vllm:
        try:
            import websockets
            print("Running vllm-server...", file=sys.stderr)
            vllm = asyncio.run(run_vllm_benchmark(args.duration))
            write_segments(vllm["text"], "results/vllm_segments.txt", "vllm-server")
            print(f"  Done: {vllm['total_time']:.1f}s, {len(vllm['text'].split())} words", file=sys.stderr)
        except Exception as e:
            print(f"  vllm skipped: {e}", file=sys.stderr)

    # Compute WER
    burn_words = len(burn["text"].split())
    burn_ref = " ".join(ref_text.split()[:burn_words])
    burn_wer = compute_wer(burn_ref, burn["text"])
    burn_cer = compute_cer(burn_ref, burn["text"])

    vllm_wer, vllm_cer = -1.0, -1.0
    if vllm:
        vllm_words = len(vllm["text"].split())
        vllm_ref = " ".join(ref_text.split()[:vllm_words])
        vllm_wer = compute_wer(vllm_ref, vllm["text"])
        vllm_cer = compute_cer(vllm_ref, vllm["text"])

    # Write comparison
    with open("results/comparison.txt", "w") as f:
        f.write(f"{'='*72}\n")
        f.write(f"BURN-SERVER vs VLLM-SERVER — {args.duration}s broadcast_1.wav\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*72}\n\n")

        f.write(f"{'Metric':<30} {'burn-server':<25} {'vllm-server':<25}\n")
        f.write(f"{'-'*72}\n")
        f.write(f"{'Backend':<30} {'candle 0.10+flash BF16':<25} {'vLLM PyTorch BF16':<25}\n")
        f.write(f"{'Total time':<30} {burn.get('total_time',0):.1f}s{'':<20} ")
        if vllm: f.write(f"{vllm['total_time']:.1f}s")
        else: f.write("N/A")
        f.write("\n")
        f.write(f"{'Decode/step':<30} {burn.get('ms_per_step',0):.1f}ms{'':<20} {'~5ms':<25}\n")
        f.write(f"{'Text tokens':<30} {burn.get('text_tokens',0):<25} ")
        if vllm: f.write(f"{len(vllm['text'].split())}")
        else: f.write("N/A")
        f.write("\n")
        f.write(f"{'Words':<30} {burn_words:<25} ")
        if vllm: f.write(f"{len(vllm['text'].split())}")
        else: f.write("N/A")
        f.write("\n")
        f.write(f"{'Segments':<30} {len(burn_segs):<25} ")
        if vllm: f.write(f"{len(extract_sentences(vllm['text']))}")
        else: f.write("N/A")
        f.write("\n")
        f.write(f"{'Rotations':<30} {burn.get('rotations',0):<25} {'~3':<25}\n")
        f.write(f"{'WER':<30} {burn_wer:.1%}{'':<20} ")
        if vllm: f.write(f"{vllm_wer:.1%}")
        else: f.write("N/A")
        f.write("\n")
        f.write(f"{'CER':<30} {burn_cer:.1%}{'':<20} ")
        if vllm: f.write(f"{vllm_cer:.1%}")
        else: f.write("N/A")
        f.write("\n\n")

        f.write("BURN TEXT (first 500 chars):\n")
        f.write(burn["text"][:500] + "\n\n")
        if vllm:
            f.write("VLLM TEXT (first 500 chars):\n")
            f.write(vllm["text"][:500] + "\n")

    print(f"\nResults written to results/", file=sys.stderr)
    print(open("results/comparison.txt").read())

if __name__ == "__main__":
    main()
