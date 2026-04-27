"""WAV header helpers for streamed PCM-to-WAV repackaging.

The vllm-omni server's native `response_format=wav` streaming path is broken
in 0.18.0 (asserts on missing sample-rate metadata before yielding the first
chunk). Workaround: ask for `response_format=pcm` + `stream=true`, prepend a
streaming WAV header on our side, and pass the raw PCM frames through
unchanged.

Stream format (this module's output, matching vllm-omni's intended behaviour
and what most browsers accept for chunked audio/wav):

    RIFF <0xFFFFFFFF> WAVE
    fmt  <16> 1 <channels> <rate> <byte_rate> <block_align> <bits_per_sample>
    data <0xFFFFFFFF> <pcm payload …>

The two `0xFFFFFFFF` placeholders signal "unknown / streaming" to clients.
Chrome, Firefox, Safari, and `<audio>` all play it back.
"""

from __future__ import annotations

import struct
from typing import Final

PLACEHOLDER_SIZE: Final[int] = 0xFFFFFFFF


def streaming_header(
    *,
    sample_rate: int = 24000,
    channels: int = 1,
    bits_per_sample: int = 16,
) -> bytes:
    """Return the 44-byte RIFF/WAVE/PCM header for an unknown-length stream.

    Defaults match Voxtral-4B-TTS output (24 kHz, mono, signed-int16 PCM).
    """
    if bits_per_sample % 8 != 0:
        raise ValueError("bits_per_sample must be a multiple of 8")
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        PLACEHOLDER_SIZE,        # File size — unknown for stream
        b"WAVE",
        b"fmt ",
        16,                      # fmt chunk size
        1,                       # PCM
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        PLACEHOLDER_SIZE,        # data chunk size — unknown for stream
    )
