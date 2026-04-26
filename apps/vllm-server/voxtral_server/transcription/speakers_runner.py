"""Speakers session runner — consumes 16kHz PCM from a browser uplink.

Unlike ``session_runner.run_session`` (which drives audio from an FFmpeg source),
this variant reads audio chunks that the WebSocket handler pushes into
``ctx.audio_in_queue`` from the inbound WebRTC track.
"""

from __future__ import annotations

import asyncio
import sys
import time

from ..state import SessionContext
from ..models import SessionState
from .session_runner import (
    BATCH_SAMPLES,
    _extract_complete_sentences,
)
from .vllm_client import VLLMClient


SAMPLE_RATE = 16000


async def run_speakers_session(ctx: SessionContext) -> None:
    """Transcribe audio arriving from the browser uplink.

    Flow:
      1. Connect to vLLM (skipped for without_transcription sessions).
      2. Consume PCM chunks from ``ctx.audio_in_queue`` (fed by WS ``on_track``).
      3. Accumulate into 0.5s batches, send to vLLM, commit, drain deltas.
      4. Emit subtitles via ``ctx.broadcast``.
    """
    session_id = ctx.info.id
    vllm = VLLMClient(language=ctx.info.language)

    if ctx.audio_in_queue is None:
        ctx.info.state = SessionState.ERROR
        await ctx.broadcast({"type": "error", "message": "Speakers session started without an inbound audio queue"})
        return

    try:
        if not ctx.info.without_transcription:
            try:
                await vllm.connect()
            except Exception as e:
                print(f"[Speakers {session_id}] vLLM connection failed: {e}", file=sys.stderr)
                ctx.info.state = SessionState.ERROR
                await ctx.broadcast({"type": "error", "message": f"vLLM connection failed: {e}"})
                return

        ctx.info.state = SessionState.RUNNING
        await ctx.broadcast({"type": "start"})

        growing_text = ""
        full_transcript = ""
        segment_count = 0
        total_samples = 0
        audio_batch: list[float] = []
        start_time = time.monotonic()

        while not ctx.cancel_event.is_set():
            try:
                chunk = await asyncio.wait_for(ctx.audio_in_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            if chunk is None:
                # Sentinel meaning the uplink disconnected
                break

            total_samples += len(chunk)
            ctx.info.progress_secs = total_samples / SAMPLE_RATE
            audio_batch.extend(chunk)

            if ctx.info.without_transcription:
                # Nothing more to do until we have enough data
                audio_batch = audio_batch[-SAMPLE_RATE:]
                continue

            if len(audio_batch) >= BATCH_SAMPLES:
                t0 = time.monotonic()
                await vllm.send_audio(audio_batch)
                await vllm.commit()
                audio_batch = []
                await asyncio.sleep(0.1)

                batch_delta = vllm.drain_deltas()
                if batch_delta:
                    growing_text += batch_delta
                    current_time = total_samples / SAMPLE_RATE
                    sentences, remainder = _extract_complete_sentences(growing_text)

                    for sentence in sentences:
                        segment_count += 1
                        full_transcript = (full_transcript + " " + sentence).strip()
                        await ctx.broadcast({
                            "type": "subtitle",
                            "text": sentence,
                            "growing_text": None,
                            "full_transcript": full_transcript,
                            "delta": sentence,
                            "tail_changed": False,
                            "speaker": None,
                            "start": max(0, current_time - 2.0),
                            "end": current_time,
                            "is_final": True,
                            "inference_time_ms": int((time.monotonic() - t0) * 1000),
                        })

                    growing_text = remainder
                    if growing_text.strip():
                        await ctx.broadcast({
                            "type": "subtitle",
                            "text": growing_text.strip(),
                            "growing_text": growing_text.strip(),
                            "full_transcript": (full_transcript + " " + growing_text).strip(),
                            "delta": batch_delta,
                            "tail_changed": False,
                            "speaker": None,
                            "start": max(0, current_time - 2.0),
                            "end": current_time,
                            "is_final": False,
                            "inference_time_ms": None,
                        })

        # Flush remaining audio on disconnect
        if audio_batch and not ctx.info.without_transcription and vllm.is_connected:
            await vllm.send_audio(audio_batch)
            await vllm.commit()
            await asyncio.sleep(2.0)
            final_delta = vllm.drain_deltas()
            if final_delta:
                growing_text += final_delta

            if growing_text.strip():
                current_time = total_samples / SAMPLE_RATE
                sentences, remainder = _extract_complete_sentences(growing_text)
                if remainder.strip():
                    sentences.append(remainder.strip())
                for sentence in sentences:
                    segment_count += 1
                    full_transcript = (full_transcript + " " + sentence).strip()
                    await ctx.broadcast({
                        "type": "subtitle",
                        "text": sentence,
                        "growing_text": None,
                        "full_transcript": full_transcript,
                        "delta": sentence,
                        "tail_changed": False,
                        "speaker": None,
                        "start": max(0, current_time - 2.0),
                        "end": current_time,
                        "is_final": True,
                        "inference_time_ms": None,
                    })

        total_duration = total_samples / SAMPLE_RATE
        ctx.info.state = SessionState.COMPLETED
        await ctx.broadcast({"type": "end", "total_duration": round(total_duration, 2)})
        wall = time.monotonic() - start_time
        print(
            f"[Speakers {session_id}] Completed: {total_duration:.1f}s audio, "
            f"{segment_count} segments, {wall:.1f}s wall time",
            file=sys.stderr,
        )

    except asyncio.CancelledError:
        ctx.info.state = SessionState.STOPPED
        print(f"[Speakers {session_id}] Cancelled", file=sys.stderr)
    except Exception as e:
        ctx.info.state = SessionState.ERROR
        print(f"[Speakers {session_id}] Error: {e}", file=sys.stderr)
        await ctx.broadcast({"type": "error", "message": str(e)})
    finally:
        await vllm.disconnect()
