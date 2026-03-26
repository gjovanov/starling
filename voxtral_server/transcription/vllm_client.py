"""vLLM Realtime API WebSocket client — sends audio, receives transcription deltas."""

from __future__ import annotations

import asyncio
import base64
import json
import struct
import sys
from typing import AsyncIterator

import websockets
from websockets.asyncio.client import ClientConnection

from ..config import settings


class VLLMClient:
    """
    WebSocket client for vLLM's /v1/realtime endpoint (OpenAI Realtime API format).

    Uses a background reader task to continuously consume deltas from vLLM,
    avoiding the problem of missing tokens due to short polling timeouts.

    For long-running streams (SRT), automatically rotates the vLLM session
    before the context window fills up, preventing EngineCore crashes.
    """

    # Each commit adds ~50-80 audio tokens. At 16384 max_model_len,
    # rotate well before the limit. 0.5s batches × 200 commits ≈ 100s of audio.
    MAX_COMMITS_BEFORE_ROTATE = 200

    def __init__(self, language: str = "de") -> None:
        self._url = settings.vllm_url
        self._language = language
        self._ws: ClientConnection | None = None
        self._connected = False
        # Background reader pushes deltas here
        self._delta_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=4096)
        self._reader_task: asyncio.Task | None = None
        self._commit_count = 0

    async def connect(self) -> None:
        """Connect to vLLM WebSocket and start background reader."""
        try:
            self._ws = await websockets.connect(
                self._url,
                close_timeout=5,
                max_size=10 * 1024 * 1024,  # 10MB
            )
            self._connected = True
            print(f"[vLLM] Connected to {self._url}", file=sys.stderr)

            # Send session.update to validate the model (required by vLLM Realtime API)
            # vLLM expects "model" at the top level of the event
            session_update = json.dumps({
                "type": "session.update",
                "model": "mistralai/Voxtral-Mini-4B-Realtime-2602",
                "input_audio_format": "pcm16",
                "language": self._language,
                "turn_detection": None,
            })
            await self._ws.send(session_update)
            print(f"[vLLM] Session update sent (language={self._language})", file=sys.stderr)

            # Wait for session.updated confirmation
            try:
                raw = await asyncio.wait_for(self._ws.recv(), timeout=5.0)
                msg = json.loads(raw)
                if msg.get("type") == "session.updated":
                    print(f"[vLLM] Session validated", file=sys.stderr)
                else:
                    print(f"[vLLM] Unexpected response: {msg.get('type', 'unknown')}", file=sys.stderr)
            except asyncio.TimeoutError:
                print(f"[vLLM] No session.updated response (continuing anyway)", file=sys.stderr)

            # Start background reader
            self._reader_task = asyncio.create_task(self._background_reader())

        except Exception as e:
            self._connected = False
            print(f"[vLLM] Connection failed: {e}", file=sys.stderr)
            raise

    async def _background_reader(self) -> None:
        """Continuously read from vLLM WebSocket and enqueue deltas."""
        assert self._ws is not None
        delta_count = 0
        other_count = 0
        try:
            async for raw in self._ws:
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                msg_type = msg.get("type", "")
                if msg_type in ("response.audio_transcript.delta", "transcription.delta"):
                    delta = msg.get("delta", "")
                    if delta:
                        delta_count += 1
                        try:
                            self._delta_queue.put_nowait(delta)
                        except asyncio.QueueFull:
                            try:
                                self._delta_queue.get_nowait()
                            except asyncio.QueueEmpty:
                                pass
                            self._delta_queue.put_nowait(delta)
                elif msg_type == "error":
                    print(f"[vLLM] Error: {msg.get('error', msg)}", file=sys.stderr)
                else:
                    other_count += 1
                    if other_count <= 5:
                        print(f"[vLLM] Reader got: {msg_type}", file=sys.stderr)

        except websockets.exceptions.ConnectionClosed as e:
            print(f"[vLLM] Connection closed: {e} (received {delta_count} deltas)", file=sys.stderr)
        except asyncio.CancelledError:
            print(f"[vLLM] Reader cancelled (received {delta_count} deltas)", file=sys.stderr)
        except Exception as e:
            print(f"[vLLM] Reader error: {e} (received {delta_count} deltas)", file=sys.stderr)
        finally:
            self._connected = False
            print(f"[vLLM] Reader stopped (total {delta_count} deltas, {other_count} other msgs)", file=sys.stderr)

    async def disconnect(self) -> None:
        """Close vLLM WebSocket and stop reader."""
        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()
            try:
                await self._reader_task
            except (asyncio.CancelledError, Exception):
                pass
            self._reader_task = None

        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
            self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected and self._ws is not None

    async def send_audio(self, samples_f32: list[float]) -> None:
        """
        Send audio samples to vLLM.

        Converts f32 samples to s16le bytes, base64-encodes, and sends as
        input_audio_buffer.append message.
        """
        if not self.is_connected:
            print(f"[vLLM] WARNING: send_audio skipped — not connected (queue={self._delta_queue.qsize()})", file=sys.stderr)
            return

        # Convert f32 to s16le bytes
        n = len(samples_f32)
        raw = struct.pack(f"<{n}h", *[int(max(-32768, min(32767, s * 32768))) for s in samples_f32])

        # Base64 encode
        audio_b64 = base64.b64encode(raw).decode("ascii")

        # Send append message
        msg = json.dumps({
            "type": "input_audio_buffer.append",
            "audio": audio_b64,
        })
        try:
            await self._ws.send(msg)
        except Exception as e:
            self._connected = False
            print(f"[vLLM] Send error: {e}", file=sys.stderr)

    async def commit(self) -> None:
        """Send commit message to trigger transcription."""
        if not self.is_connected:
            return
        try:
            await self._ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
            self._commit_count += 1

            # Rotate context before hitting max_model_len
            if self._commit_count >= self.MAX_COMMITS_BEFORE_ROTATE:
                await self._rotate_session()
        except Exception as e:
            self._connected = False
            print(f"[vLLM] Commit error: {e}", file=sys.stderr)

    async def _rotate_session(self) -> None:
        """Disconnect and reconnect to reset the vLLM context window.

        This prevents EngineCore crashes when accumulated audio tokens
        exceed max_model_len on long-running streams.
        """
        print(f"[vLLM] Rotating session (after {self._commit_count} commits)", file=sys.stderr)
        await self.disconnect()
        self._commit_count = 0
        try:
            await self.connect()
            print(f"[vLLM] Session rotated successfully", file=sys.stderr)
        except Exception as e:
            print(f"[vLLM] Rotation failed: {e}", file=sys.stderr)

    def drain_deltas(self) -> str:
        """
        Drain all available deltas from the queue (non-blocking).

        Returns concatenated delta text. Called from the session runner
        after each audio batch.
        """
        text = ""
        while True:
            try:
                delta = self._delta_queue.get_nowait()
                text += delta
            except asyncio.QueueEmpty:
                break
        return text
