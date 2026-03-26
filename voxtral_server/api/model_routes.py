"""Model listing, modes, and stub endpoints."""

import json
import os
from pathlib import Path

from dotenv import dotenv_values
from fastapi import APIRouter

from ..models import ApiResponse, ModeInfo, ModelInfo

router = APIRouter()

VOXTRAL_MODEL = ModelInfo(
    id="voxtral-mini-4b",
    display_name="Voxtral Mini 4B Realtime",
    description="Mistral Voxtral Mini 4B — streaming ASR via vLLM (13 languages)",
    supports_diarization=False,
    languages=["en", "fr", "es", "de", "ru", "zh", "ja", "it", "pt", "nl", "ar", "hi", "ko"],
    is_loaded=True,
)

MODES = [
    ModeInfo(
        id="speedy",
        label="Speedy",
        description="Low-latency streaming transcription",
    ),
]


@router.get("/api/models")
async def list_models():
    return ApiResponse.ok([VOXTRAL_MODEL.model_dump()])


@router.get("/api/modes")
async def list_modes():
    return ApiResponse.ok([m.model_dump() for m in MODES])


@router.get("/api/noise-cancellation")
async def list_noise_cancellation():
    return ApiResponse.ok([
        {"id": "none", "label": "None", "description": "No noise cancellation"},
    ])


@router.get("/api/diarization")
async def list_diarization():
    return ApiResponse.ok([
        {"id": "none", "label": "None", "description": "No diarization"},
    ])


def _load_srt_env() -> dict[str, str | None]:
    """Load SRT config from .env files (shared with parakeet-rs, no VOXTRAL_ prefix)."""
    env: dict[str, str | None] = {}
    # Check both voxtral-server/.env and root .env
    for path in [Path(".env"), Path("../.env")]:
        if path.is_file():
            env.update(dotenv_values(path))
    # os.environ overrides .env files
    env.update(os.environ)
    return env


@router.get("/api/srt-streams")
async def list_srt_streams():
    env = _load_srt_env()
    encoder_ip = env.get("SRT_ENCODER_IP", "") or ""
    channels_json = env.get("SRT_CHANNELS", "") or ""

    if not encoder_ip or not channels_json:
        return {"success": True, "streams": [], "configured": False}

    try:
        channels = json.loads(channels_json)
    except json.JSONDecodeError:
        return {"success": True, "streams": [], "configured": False}

    streams = []
    for i, ch in enumerate(channels):
        name = ch.get("name", f"Channel {i}")
        port = ch.get("port", "")
        streams.append({
            "id": i,
            "name": name,
            "port": port,
            "display": f"{name} (srt://{encoder_ip}:{port})",
        })

    return {"success": True, "streams": streams, "configured": True}
