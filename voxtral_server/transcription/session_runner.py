"""Session runner — orchestrates FFmpeg, vLLM transcription, and subtitle emission."""

from __future__ import annotations

import asyncio
import re
import sys
import time
from pathlib import Path

from ..audio.ffmpeg_source import resample_16k_to_48k, stream_pcm, SAMPLE_RATE
from ..audio.webrtc_track import PcmAudioTrack
from ..state import SessionContext
from ..models import SessionState
from .vllm_client import VLLMClient


# How many seconds of audio to batch before sending to vLLM
BATCH_INTERVAL_SECS = 0.5
BATCH_SAMPLES = int(SAMPLE_RATE * BATCH_INTERVAL_SECS)

# Sentence boundary: period/exclamation/question followed by whitespace and uppercase.
# Must have a word (not just a number or abbreviation) before the punctuation.
# This avoids splitting on "19. November" or "Dr. Müller".
_SENTENCE_BOUNDARY = re.compile(
    r'(?<=[a-zäöüß][.!?])\s+(?=[A-ZÄÖÜ])'
)


def _extract_complete_sentences(text: str) -> tuple[list[str], str]:
    """
    Split text into complete sentences and a remaining fragment.

    Returns (sentences, remainder) where:
    - sentences: list of complete sentences (ending with . ! ?)
    - remainder: text after the last sentence boundary (may be empty)
    """
    parts = _SENTENCE_BOUNDARY.split(text)
    if not parts:
        return [], text

    sentences = []
    for part in parts[:-1]:
        s = part.strip()
        if s:
            sentences.append(s)

    # Last part: check if it ends with sentence punctuation
    last = parts[-1].strip()
    if last and last[-1] in '.!?':
        sentences.append(last)
        return sentences, ""
    else:
        return sentences, last


async def run_session(ctx: SessionContext, media_path: Path | str | None) -> None:
    """
    Main session loop:
    1. Start FFmpeg to decode audio
    2. Split PCM: 48kHz to WebRTC track, 16kHz to vLLM
    3. Accumulate vLLM deltas into subtitle messages
    4. Broadcast subtitles to WebSocket clients
    """
    session_id = ctx.info.id
    vllm = VLLMClient(language=ctx.info.language)

    # Create audio track immediately so WS handler can attach it to WebRTC
    ctx.audio_track = PcmAudioTrack()

    try:
        # Connect to vLLM
        if not ctx.info.without_transcription:
            try:
                await vllm.connect()
            except Exception as e:
                print(f"[Session {session_id}] vLLM connection failed: {e}", file=sys.stderr)
                ctx.info.state = SessionState.ERROR
                await ctx.broadcast({"type": "error", "message": f"vLLM connection failed: {e}"})
                return

        ctx.info.state = SessionState.RUNNING

        if media_path is None:
            print(f"[Session {session_id}] No media file — waiting", file=sys.stderr)
            return

        # Wait for a client to send "ready" before starting audio
        print(f"[Session {session_id}] Waiting for client...", file=sys.stderr)
        try:
            await asyncio.wait_for(ctx.client_ready.wait(), timeout=10.0)
            print(f"[Session {session_id}] Client ready", file=sys.stderr)
        except asyncio.TimeoutError:
            print(f"[Session {session_id}] No client after 10s, starting anyway", file=sys.stderr)

        # Brief delay for WebRTC setup (if applicable)
        await asyncio.sleep(0.3)

        # Broadcast start
        await ctx.broadcast({"type": "start"})

        # State for subtitle accumulation
        growing_text = ""
        full_transcript = ""
        segment_count = 0
        total_samples = 0
        audio_batch: list[float] = []
        start_time = time.monotonic()

        # Stream audio from FFmpeg
        async for samples_16k in stream_pcm(media_path, ctx.cancel_event):
            if ctx.cancel_event.is_set():
                break

            total_samples += len(samples_16k)
            ctx.info.progress_secs = total_samples / SAMPLE_RATE

            # Push to WebRTC track (48kHz)
            samples_48k = resample_16k_to_48k(samples_16k)
            ctx.audio_track.push_samples(samples_48k)

            # Batch audio for vLLM
            if not ctx.info.without_transcription:
                audio_batch.extend(samples_16k)

                if len(audio_batch) >= BATCH_SAMPLES:
                    # Send to vLLM
                    await vllm.send_audio(audio_batch)
                    await vllm.commit()
                    audio_batch = []

                    # Small wait for vLLM to produce some deltas
                    await asyncio.sleep(0.1)

                    # Drain all deltas accumulated by the background reader
                    batch_delta = vllm.drain_deltas()

                    if batch_delta:
                        growing_text += batch_delta
                        current_time = total_samples / SAMPLE_RATE

                        # Extract complete sentences from growing_text
                        sentences, remainder = _extract_complete_sentences(growing_text)

                        # Emit each complete sentence as a FINAL
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
                                "inference_time_ms": int((time.monotonic() - start_time) * 1000) % 1000,
                            })

                        # Keep the remainder as the new growing_text
                        growing_text = remainder

                        # Emit PARTIAL for the remainder if non-empty
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

        # Flush remaining audio batch
        if audio_batch and not ctx.info.without_transcription and vllm.is_connected:
            await vllm.send_audio(audio_batch)
            await vllm.commit()
            # Wait longer for final deltas
            await asyncio.sleep(2.0)

            final_delta = vllm.drain_deltas()
            if final_delta:
                growing_text += final_delta

            if growing_text.strip():
                # Emit everything remaining as FINAL
                sentences, remainder = _extract_complete_sentences(growing_text)
                # Add remainder as final sentence too
                if remainder.strip():
                    sentences.append(remainder.strip())

                current_time = total_samples / SAMPLE_RATE
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

        # Broadcast end
        total_duration = total_samples / SAMPLE_RATE
        ctx.info.state = SessionState.COMPLETED
        await ctx.broadcast({
            "type": "end",
            "total_duration": round(total_duration, 2),
        })

        wall_time = time.monotonic() - start_time
        print(
            f"[Session {session_id}] Completed: {total_duration:.1f}s audio, "
            f"{segment_count} segments, {wall_time:.1f}s wall time",
            file=sys.stderr,
        )

    except asyncio.CancelledError:
        print(f"[Session {session_id}] Cancelled", file=sys.stderr)
        ctx.info.state = SessionState.STOPPED
    except Exception as e:
        print(f"[Session {session_id}] Error: {e}", file=sys.stderr)
        ctx.info.state = SessionState.ERROR
        await ctx.broadcast({"type": "error", "message": str(e)})
    finally:
        await vllm.disconnect()
