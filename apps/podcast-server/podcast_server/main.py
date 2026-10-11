"""Starling podcast-server — FastAPI entry point (loopback :8093).

    POST /api/tts/podcast          turns → audio/wav (one-pass, multi-speaker)
    GET  /api/tts/podcast/voices   named voice prompts
    GET  /api/tts/podcast/status   engine lifecycle snapshot
    POST /api/tts/podcast/load     warm the model now
    POST /api/tts/podcast/unload   free the GPU now
    GET  /health

The model lives in a worker process the engine spawns on first use and
stops after PODCAST_IDLE_UNLOAD_SECS idle; this process never imports torch.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from pydantic import BaseModel, ConfigDict, Field

from . import __version__, disclosure
from .config import settings
from .engine import EngineError, PodcastEngine, build_engine
from .script import (CFG_SCALE_RANGE, DIFFUSION_STEPS_RANGE, SEED_MAX, ScriptError, Turn,
                     check_generation, plan)
from .voices import load_voices
from .wav import encode_wav

logging.basicConfig(
    level=os.environ.get("PODCAST_LOG_LEVEL", "INFO"),
    format="%(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# EngineError.reason → HTTP status
_STATUS = {
    "blocked": 503,
    "load_failed": 503,
    "load_timeout": 503,
    "synth_timeout": 504,
    "synth_failed": 500,
    "worker_died": 500,
}


class TurnIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    speaker: str = Field(min_length=1, max_length=64, description="voice name, e.g. 'michael'")
    text: str = Field(min_length=1, description="what this speaker says (whitespace is collapsed)")


class PodcastRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    turns: list[TurnIn] = Field(min_length=1)
    seed: int | None = Field(default=None, ge=0, le=SEED_MAX)
    cfg_scale: float | None = Field(default=None, ge=CFG_SCALE_RANGE[0], le=CFG_SCALE_RANGE[1])
    diffusion_steps: int | None = Field(default=None, ge=DIFFUSION_STEPS_RANGE[0],
                                        le=DIFFUSION_STEPS_RANGE[1])


def create_app(engine: PodcastEngine | None = None) -> FastAPI:
    engine = engine or build_engine(settings)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        await engine.shutdown()

    app = FastAPI(title="Starling Podcast Server", version=__version__, lifespan=lifespan)
    app.state.engine = engine

    @app.exception_handler(ScriptError)
    async def _script_error(_req, exc: ScriptError):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.exception_handler(EngineError)
    async def _engine_error(_req, exc: EngineError):
        return JSONResponse({"detail": str(exc), "reason": exc.reason},
                            status_code=_STATUS.get(exc.reason, 500))

    @app.get("/health")
    async def health() -> PlainTextResponse:
        return PlainTextResponse("OK")

    @app.get("/api/config")
    async def api_config() -> dict:
        return {"app": "podcast-server"}

    @app.get("/api/tts/podcast/voices")
    async def voices() -> dict:
        return {"voices": [v.describe() for v in load_voices(settings.voices_dir).values()]}

    @app.get("/api/tts/podcast/status")
    async def status() -> dict:
        return engine.status()

    @app.post("/api/tts/podcast/load")
    async def load() -> dict:
        await engine.load()
        return engine.status()

    @app.post("/api/tts/podcast/unload")
    async def unload() -> dict:
        unloaded = await engine.unload()
        return {"unloaded": unloaded, **engine.status()}

    @app.post("/api/tts/podcast")
    async def podcast(req: PodcastRequest) -> Response:
        seed = settings.default_seed if req.seed is None else req.seed
        cfg = settings.default_cfg_scale if req.cfg_scale is None else req.cfg_scale
        steps = settings.default_diffusion_steps if req.diffusion_steps is None else req.diffusion_steps
        check_generation(seed, cfg, steps)
        registry = load_voices(settings.voices_dir)
        p = plan([Turn(t.speaker, t.text) for t in req.turns], set(registry), settings)
        result = await engine.synthesize(
            p, [str(registry[name].path) for name in p.speakers],
            seed=seed, cfg_scale=cfg, diffusion_steps=steps)
        if not result.finished:
            logger.warning("podcast render hit the max_new_tokens guard (%d) — audio is truncated",
                           p.max_new_tokens)
        wav = encode_wav(result.audio, result.sample_rate, software=disclosure.SOFTWARE,
                         comment=disclosure.comment(engine.model_label, p.speakers))
        headers = {
            "X-Podcast-Sample-Rate": str(result.sample_rate),
            "X-Podcast-Duration-Secs": f"{result.duration_secs:.3f}",
            "X-Podcast-Generation-Secs": f"{result.gen_secs:.2f}",
            "X-Podcast-Seed": str(seed),
            "X-Podcast-Cfg-Scale": f"{cfg:g}",
            "X-Podcast-Diffusion-Steps": str(steps),
            "X-Podcast-Speakers": ",".join(p.speakers),
            "X-Podcast-Model": engine.model_label,
            "X-Podcast-Truncated": "false" if result.finished else "true",
            "X-Podcast-Peak-Vram-Gib": f"{result.peak_vram_gib:.2f}",
            "X-Podcast-AI-Generated": "true",
        }
        if result.cold_load_secs is not None:
            headers["X-Podcast-Cold-Load-Secs"] = f"{result.cold_load_secs:.2f}"
        return Response(content=wav, media_type="audio/wav", headers=headers)

    return app


app = create_app()


def run() -> None:
    """Entry point for the `podcast-server` command — loopback only, by design."""
    import uvicorn

    uvicorn.run("podcast_server.main:app", host="127.0.0.1", port=settings.port, log_level="info")


if __name__ == "__main__":
    run()
