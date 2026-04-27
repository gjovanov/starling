"""Tests for the streaming-WAV header helper."""

from __future__ import annotations

import io
import struct
import wave

import numpy as np
import pytest

from voxtral_server.tts import wav as wavmod


def test_streaming_header_size() -> None:
    assert len(wavmod.streaming_header()) == 44


def test_streaming_header_fields() -> None:
    h = wavmod.streaming_header(sample_rate=24000, channels=1, bits_per_sample=16)
    riff, riff_sz, wave_tag = struct.unpack("<4sI4s", h[:12])
    assert riff == b"RIFF"
    assert riff_sz == 0xFFFFFFFF
    assert wave_tag == b"WAVE"

    fmt, fmt_sz, fmt_id, channels, rate, byte_rate, block_align, bps = struct.unpack(
        "<4sIHHIIHH", h[12:36]
    )
    assert fmt == b"fmt "
    assert fmt_sz == 16
    assert fmt_id == 1            # PCM
    assert channels == 1
    assert rate == 24000
    assert byte_rate == 24000 * 2
    assert block_align == 2
    assert bps == 16

    data_tag, data_sz = struct.unpack("<4sI", h[36:44])
    assert data_tag == b"data"
    assert data_sz == 0xFFFFFFFF


def test_streaming_header_other_rates() -> None:
    h = wavmod.streaming_header(sample_rate=16000, channels=2, bits_per_sample=16)
    # Same layout as test_streaming_header_fields but different parameters.
    fmt, fmt_sz, fmt_id, channels, rate, byte_rate, block_align, bps = struct.unpack(
        "<4sIHHIIHH", h[12:36]
    )
    assert channels == 2
    assert rate == 16000
    assert byte_rate == 16000 * 2 * 2          # rate * channels * 2 bytes/sample
    assert block_align == 2 * 2
    assert bps == 16


def test_streaming_header_rejects_bad_bits() -> None:
    with pytest.raises(ValueError):
        wavmod.streaming_header(bits_per_sample=12)


def test_header_plus_pcm_round_trips_through_wave_module() -> None:
    """The Python `wave` module is permissive enough to read our streaming
    header + raw PCM payload as long as we splice in real lengths first.
    """
    n = 1200
    samples = (np.sin(np.linspace(0, np.pi * 4, n)) * 32000).astype(np.int16).tobytes()

    header = bytearray(wavmod.streaming_header(sample_rate=24000))
    # Patch the lengths so wave.open can parse it (browsers don't need this,
    # but the stdlib `wave` reader does).
    riff_size = 36 + len(samples)
    header[4:8] = riff_size.to_bytes(4, "little")
    header[40:44] = len(samples).to_bytes(4, "little")
    blob = bytes(header) + samples

    with wave.open(io.BytesIO(blob), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getframerate() == 24000
        assert wf.getsampwidth() == 2
        assert wf.getnframes() == n
