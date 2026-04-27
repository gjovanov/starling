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
    # "media" (default), "srt", or "speakers". When "speakers", no media file
    # is required — audio is uploaded from the browser via WebRTC.
    source: str | None = None
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


# ---------------------------------------------------------------------------
# Text-to-speech
# ---------------------------------------------------------------------------

class TtsConfig(BaseModel):
    """Static TTS server config the frontend needs to render the TTS tab."""

    output_dir: str
    max_chars: int
    default_voice: str
    sample_rate: int = 24000
    supported_formats: list[str] = ["wav"]


class TtsSynthesizeRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to synthesize.")
    voice: str = Field(default="", description="Voice id from /api/tts/voices. Empty → default.")
    voice_ref_id: str | None = Field(
        default=None,
        description=(
            "Uploaded voice reference id (from /api/tts/voices kind=cloned). "
            "When set, the upstream is restarted into --task-type Base and "
            "the cloned voice is used instead of the built-in `voice`."
        ),
    )
    save: bool = Field(default=True, description="Persist the result under VOXTRAL_TTS_OUTPUT_DIR.")
    save_filename: str | None = Field(
        default=None,
        description=(
            "Optional filename (must end in .wav, [A-Za-z0-9._-]{1,128}). "
            "Auto-generated when omitted."
        ),
    )
    overwrite: bool = Field(
        default=False,
        description="Allow replacing an existing file with the same name.",
    )


class TtsSynthesizeResponse(BaseModel):
    """Returned by POST /api/tts/synthesize when save=true."""

    filename: str
    path: str
    bytes: int
    voice: str
    sample_rate: int
    duration_secs: float | None = None
    elapsed_secs: float


class TtsOutputFile(BaseModel):
    name: str
    bytes: int
    created_at: float


# ---------------------------------------------------------------------------
# Voice cloning (Phase 7)
# ---------------------------------------------------------------------------

class VoiceRefInfo(BaseModel):
    """Public representation of an uploaded voice reference. Maps directly
    onto the on-disk sidecar JSON minus internal paths."""
    id: str
    name: str
    ref_text: str
    permission_confirmed: bool
    sample_rate: int = 24000
    duration_secs: float
    created_at: float
    kind: str = "cloned"
