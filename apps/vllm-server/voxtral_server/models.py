"""Pydantic models matching parakeet-rs API contract."""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# API envelope
# ---------------------------------------------------------------------------

class ApiResponse(BaseModel):
    success: bool = True
    data: Any = None
    error: str | None = None

    @classmethod
    def ok(cls, data: Any = None) -> "ApiResponse":
        return cls(success=True, data=data)

    @classmethod
    def err(cls, msg: str) -> "ApiResponse":
        return cls(success=False, error=msg)


# ---------------------------------------------------------------------------
# Models & modes
# ---------------------------------------------------------------------------

class ModelInfo(BaseModel):
    id: str
    display_name: str
    description: str
    supports_diarization: bool = False
    languages: list[str] = []
    is_loaded: bool = True


class ModeInfo(BaseModel):
    id: str
    label: str
    description: str


# ---------------------------------------------------------------------------
# Media
# ---------------------------------------------------------------------------

class MediaFile(BaseModel):
    id: str
    filename: str
    format: str = "wav"
    duration_secs: float = 0.0
    size_bytes: int = 0


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

class SessionState(str, Enum):
    CREATED = "created"
    STARTING = "starting"
    RUNNING = "running"
    COMPLETED = "completed"
    STOPPED = "stopped"
    ERROR = "error"


class SessionInfo(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    model_id: str = ""
    model_name: str = ""
    media_id: str = ""
    media_filename: str = ""
    state: str = SessionState.CREATED
    client_count: int = 0
    duration_secs: float = 0.0
    progress_secs: float = 0.0
    created_at: float = Field(default_factory=time.time)
    mode: str = "speedy"
    language: str = "de"
    noise_cancellation: str = "none"
    diarization: bool = False
    source_type: str = "file"
    sentence_completion: str = "minimal"
    without_transcription: bool = False


class CreateSessionRequest(BaseModel):
    model_id: str = ""
    mode: str = "speedy"
    language: str = "de"
    media_id: str | None = None
    srt_channel_id: int | None = None
    noise_cancellation: str = "none"
    diarization: bool = False
    sentence_completion: str = "minimal"
    without_transcription: bool = False
    pause_config: dict | None = None
    growing_segments_config: dict | None = None
    fab_enabled: str | None = None
    fab_url: str | None = None
    fab_send_type: str | None = None


# ---------------------------------------------------------------------------
# Subtitle message
# ---------------------------------------------------------------------------

class SubtitleMessage(BaseModel):
    type: str = "subtitle"
    text: str = ""
    growing_text: str | None = None
    full_transcript: str | None = None
    delta: str | None = None
    tail_changed: bool | None = None
    speaker: int | None = None
    start: float = 0.0
    end: float = 0.0
    is_final: bool = False
    inference_time_ms: int | None = None
