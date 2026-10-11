"""WAV encoding: float32 mono → 16-bit PCM RIFF with an INFO chunk.

The INFO chunk (ISFT software, ICMT comment) labels the file as
AI-generated speech — ffprobe shows it as the `encoder` / `comment` tags.
It is metadata only: the samples are the model output, clipped to
[-1, 1] and quantized, with no other processing.
"""

from __future__ import annotations

import struct

import numpy as np


def _info_chunk(tags: list[tuple[bytes, str]]) -> bytes:
    body = b"INFO"
    for tag, value in tags:
        data = value.encode("utf-8") + b"\x00"
        body += tag + struct.pack("<I", len(data)) + data + (b"\x00" if len(data) % 2 else b"")
    return b"LIST" + struct.pack("<I", len(body)) + body


def encode_wav(samples: np.ndarray, sample_rate: int, *,
               software: str | None = None, comment: str | None = None) -> bytes:
    pcm = (np.clip(np.asarray(samples, dtype=np.float32).reshape(-1), -1.0, 1.0) * 32767.0)
    pcm_bytes = np.round(pcm).astype("<i2").tobytes()
    fmt = b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
    tags = [(t, v) for t, v in ((b"ISFT", software), (b"ICMT", comment)) if v]
    info = _info_chunk(tags) if tags else b""
    data = b"data" + struct.pack("<I", len(pcm_bytes)) + pcm_bytes
    body = b"WAVE" + fmt + info + data
    return b"RIFF" + struct.pack("<I", len(body)) + body
