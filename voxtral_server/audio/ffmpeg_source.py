"""FFmpeg audio source — decode file to PCM s16le 16kHz mono in realtime."""

from __future__ import annotations

import asyncio
import struct
import sys
from pathlib import Path
from typing import AsyncIterator


SAMPLE_RATE = 16000
CHANNELS = 1
BYTES_PER_SAMPLE = 2  # s16le
CHUNK_SAMPLES = 8000  # 0.5s chunks
CHUNK_BYTES = CHUNK_SAMPLES * BYTES_PER_SAMPLE


async def stream_pcm(
    source: Path | str,
    cancel_event: asyncio.Event,
) -> AsyncIterator[list[float]]:
    """
    Spawn FFmpeg to decode audio source to PCM and yield f32 sample chunks.

    source can be a file Path or a URL string (e.g. srt://host:port).
    Uses -re for realtime pacing on files; SRT streams are inherently realtime.
    Yields lists of float samples in [-1.0, 1.0] range.
    """
    source_str = str(source)
    is_srt = source_str.startswith("srt://")

    cmd = ["ffmpeg"]
    if not is_srt:
        cmd += ["-re"]  # realtime pacing (not needed for live SRT)
    cmd += [
        "-i", source_str,
        "-f", "s16le",
        "-ar", str(SAMPLE_RATE),
        "-ac", str(CHANNELS),
        "-loglevel", "error",
        "-",  # stdout
    ]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    label = Path(source_str).name if not is_srt else source_str
    print(f"[FFmpeg] Started PID={proc.pid} for {label}", file=sys.stderr)

    try:
        assert proc.stdout is not None
        while not cancel_event.is_set():
            try:
                raw = await asyncio.wait_for(
                    proc.stdout.read(CHUNK_BYTES),
                    timeout=2.0,
                )
            except asyncio.TimeoutError:
                continue

            if not raw:
                break  # EOF

            # Convert s16le bytes to f32 samples in [-1.0, 1.0]
            n_samples = len(raw) // BYTES_PER_SAMPLE
            samples = struct.unpack(f"<{n_samples}h", raw[:n_samples * BYTES_PER_SAMPLE])
            yield [s / 32768.0 for s in samples]

    finally:
        if proc.returncode is None:
            proc.kill()
            await proc.wait()
        print(f"[FFmpeg] Finished PID={proc.pid}", file=sys.stderr)


def resample_16k_to_48k(samples_16k: list[float]) -> list[float]:
    """Simple 3x upsample from 16kHz to 48kHz (sample triplication)."""
    out = []
    for s in samples_16k:
        out.append(s)
        out.append(s)
        out.append(s)
    return out
