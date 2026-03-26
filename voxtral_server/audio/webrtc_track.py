"""Custom aiortc AudioStreamTrack that feeds PCM frames from the audio pipeline."""

from __future__ import annotations

import asyncio
import time
from fractions import Fraction

import numpy as np
from aiortc import MediaStreamTrack
from av import AudioFrame


SAMPLE_RATE = 48000  # WebRTC expects 48kHz
FRAME_SAMPLES = 960  # 20ms at 48kHz
CHANNELS = 1


class PcmAudioTrack(MediaStreamTrack):
    """
    Audio track that reads PCM samples from an asyncio queue and produces
    av.AudioFrame objects at 48kHz mono, 20ms per frame.
    """

    kind = "audio"

    def __init__(self) -> None:
        super().__init__()
        self._queue: asyncio.Queue[list[float]] = asyncio.Queue(maxsize=100)
        self._buffer: list[float] = []
        self._timestamp = 0
        self._start_time: float | None = None

    def push_samples(self, samples_48k: list[float]) -> None:
        """Push 48kHz samples into the buffer (called from audio pipeline)."""
        try:
            self._queue.put_nowait(samples_48k)
        except asyncio.QueueFull:
            pass  # Drop oldest — real-time audio cannot block

    async def recv(self) -> AudioFrame:
        """Called by aiortc to get the next audio frame (20ms)."""
        # Pace ourselves to produce frames at real-time rate
        if self._start_time is None:
            self._start_time = time.monotonic()

        target_time = self._start_time + (self._timestamp / SAMPLE_RATE)
        now = time.monotonic()
        if target_time > now:
            await asyncio.sleep(target_time - now)

        # Fill buffer from queue until we have enough samples for one frame
        while len(self._buffer) < FRAME_SAMPLES:
            try:
                chunk = await asyncio.wait_for(self._queue.get(), timeout=0.1)
                self._buffer.extend(chunk)
            except asyncio.TimeoutError:
                # No data available — emit silence
                self._buffer.extend([0.0] * FRAME_SAMPLES)
                break

        # Extract one frame
        frame_samples = self._buffer[:FRAME_SAMPLES]
        self._buffer = self._buffer[FRAME_SAMPLES:]

        # Build av.AudioFrame
        arr = np.array(frame_samples, dtype=np.float32)
        arr = np.clip(arr, -1.0, 1.0)
        # Convert to s16
        arr_s16 = (arr * 32767).astype(np.int16)
        frame = AudioFrame.from_ndarray(
            arr_s16.reshape(1, -1),  # (channels, samples)
            format="s16",
            layout="mono",
        )
        frame.sample_rate = SAMPLE_RATE
        frame.pts = self._timestamp
        frame.time_base = Fraction(1, 48000)
        self._timestamp += FRAME_SAMPLES

        return frame
