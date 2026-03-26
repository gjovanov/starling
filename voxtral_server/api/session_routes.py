"""Session CRUD + start endpoints."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import dotenv_values
from fastapi import APIRouter

from ..config import settings
from ..media.manager import get_media_path
from ..models import ApiResponse, CreateSessionRequest, SessionInfo, SessionState
from ..state import app_state


def _resolve_srt_url(channel_id: int) -> str | None:
    """Build an SRT URL from channel ID using SRT_ENCODER_IP + SRT_CHANNELS."""
    env: dict[str, str | None] = {}
    for path in [Path(".env"), Path("../.env")]:
        if path.is_file():
            env.update(dotenv_values(path))
    env.update(os.environ)

    encoder_ip = env.get("SRT_ENCODER_IP", "") or ""
    channels_json = env.get("SRT_CHANNELS", "") or ""
    if not encoder_ip or not channels_json:
        return None

    try:
        channels = json.loads(channels_json)
    except json.JSONDecodeError:
        return None

    if channel_id < 0 or channel_id >= len(channels):
        return None

    port = channels[channel_id].get("port", "")
    latency = env.get("SRT_LATENCY", "200000") or "200000"
    rcvbuf = env.get("SRT_RCVBUF", "2097152") or "2097152"
    return f"srt://{encoder_ip}:{port}?mode=caller&latency={latency}&rcvbuf={rcvbuf}"

router = APIRouter()


@router.get("/api/sessions")
async def list_sessions():
    sessions = app_state.list_sessions()
    return ApiResponse.ok([s.model_dump() for s in sessions])


@router.post("/api/sessions")
async def create_session(req: CreateSessionRequest):
    # Resolve media file or SRT channel
    media_path = None
    media_filename = ""
    duration_secs = 0.0
    source_type = "file"
    srt_url = ""

    if req.srt_channel_id is not None:
        srt_url = _resolve_srt_url(req.srt_channel_id) or ""
        if not srt_url:
            return ApiResponse.err(f"SRT channel {req.srt_channel_id} not found or SRT not configured")
        source_type = "srt"
        media_filename = srt_url
    elif req.media_id:
        media_path = get_media_path(req.media_id)
        if media_path is None:
            return ApiResponse.err(f"Media '{req.media_id}' not found")
        media_filename = media_path.name
        from ..media.manager import get_duration
        duration_secs = await get_duration(media_path)
    else:
        return ApiResponse.err("Either media_id or srt_channel_id is required")

    info = SessionInfo(
        model_id=req.model_id or "voxtral-mini-4b",
        model_name="Voxtral Mini 4B Realtime",
        media_id=req.media_id or srt_url,
        media_filename=media_filename,
        duration_secs=duration_secs,
        mode=req.mode,
        language=req.language,
        noise_cancellation=req.noise_cancellation,
        diarization=req.diarization,
        sentence_completion=req.sentence_completion,
        without_transcription=req.without_transcription,
        source_type=source_type,
    )

    app_state.add_session(info)
    print(f"[Session {info.id}] Created (model={info.model_id}, source={source_type}, media={info.media_id}, lang={info.language})", file=sys.stderr)
    return ApiResponse.ok(info.model_dump())


@router.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    ctx = app_state.get_session(session_id)
    if ctx is None:
        return ApiResponse.err(f"Session '{session_id}' not found")
    return ApiResponse.ok(ctx.info.model_dump())


@router.delete("/api/sessions/{session_id}")
async def stop_session(session_id: str):
    ctx = app_state.get_session(session_id)
    if ctx is None:
        return ApiResponse.err(f"Session '{session_id}' not found")

    ctx.info.state = SessionState.STOPPED
    ctx.cancel_event.set()
    if ctx.task and not ctx.task.done():
        ctx.task.cancel()

    print(f"[Session {session_id}] Stopped", file=sys.stderr)
    app_state.remove_session(session_id)
    return ApiResponse.ok({"stopped": session_id})


@router.post("/api/sessions/{session_id}/start")
async def start_session(session_id: str):
    ctx = app_state.get_session(session_id)
    if ctx is None:
        return ApiResponse.err(f"Session '{session_id}' not found")

    if ctx.info.state != SessionState.CREATED:
        return ApiResponse.err(f"Session already in state '{ctx.info.state}'")

    ctx.info.state = SessionState.STARTING

    # Resolve audio source: SRT URL or media file path
    if ctx.info.source_type == "srt":
        # media_id holds the SRT URL for SRT sessions
        audio_source: Path | str = ctx.info.media_id
    else:
        media_path = get_media_path(ctx.info.media_id)
        if media_path is None and not ctx.info.without_transcription:
            ctx.info.state = SessionState.ERROR
            return ApiResponse.err(f"Media '{ctx.info.media_id}' not found")
        audio_source = media_path

    # Import here to avoid circular imports
    from ..transcription.session_runner import run_session

    ctx.task = asyncio.create_task(
        run_session(ctx, audio_source),
        name=f"session-{session_id}",
    )

    ctx.info.state = SessionState.RUNNING
    print(f"[Session {session_id}] Started (source={audio_source})", file=sys.stderr)
    return ApiResponse.ok(ctx.info.model_dump())
