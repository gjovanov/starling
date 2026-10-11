"""Podcast engine lifecycle: lazy load, one job at a time, idle unload.

Same shape as vllm-server's TTS lifecycle (voxtral_server/tts/lifecycle.py):

    idle ──synthesize()/load()──► loading ──worker ready──► ready
      ▲                              │                        │
      │                         (load fails)                  │
      └───────────────◄──────────────┴──── idle unload / unload() / worker crash

    blocked: the free-VRAM pre-check failed (another app holds the GPU).
             Not sticky — the next request checks again.

One asyncio lock serializes everything that touches the worker (load,
synth, unload), so requests queue rather than race for the GPU. The idle
unload never fires while a request is queued or running.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum

import numpy as np

from .gpu import gpu_free_gib
from .script import Plan
from .worker import MockWorker, ProcessWorker, Worker, WorkerError, WorkerTimeout

logger = logging.getLogger(__name__)


class State(str, Enum):
    IDLE = "idle"
    LOADING = "loading"
    READY = "ready"
    BLOCKED = "blocked"


class EngineError(RuntimeError):
    """A failure the API maps to an HTTP status (see `reason`)."""

    def __init__(self, message: str, *, reason: str) -> None:
        super().__init__(message)
        # blocked | load_failed | load_timeout | synth_failed | synth_timeout | worker_died
        self.reason = reason


@dataclass
class Synthesis:
    audio: np.ndarray                 # float32 mono
    sample_rate: int
    gen_secs: float
    peak_vram_gib: float
    audio_tokens: int
    finished: bool                    # False: stopped at the max_new_tokens guard
    cold_load_secs: float | None      # set when this request paid the model load

    @property
    def duration_secs(self) -> float:
        return self.audio.size / self.sample_rate if self.sample_rate else 0.0


class PodcastEngine:
    def __init__(
        self,
        worker_factory: Callable[[], Worker],
        *,
        engine_name: str,
        model_label: str,
        idle_unload_secs: float = 600.0,
        load_timeout_secs: float = 300.0,
        synth_timeout_secs: float = 1800.0,
        min_free_vram_gib: float = 0.0,
        gpu_probe: Callable[[], Awaitable[float | None]] | None = None,
    ) -> None:
        self._factory = worker_factory
        self.engine_name = engine_name
        self.model_label = model_label
        self._idle_secs = idle_unload_secs
        self._load_timeout = load_timeout_secs
        self._synth_timeout = synth_timeout_secs
        self._min_free = min_free_vram_gib
        self._gpu_probe = gpu_probe

        self._lock = asyncio.Lock()
        self._state = State.IDLE
        self._worker: Worker | None = None
        self._inflight = 0
        self._jobs_done = 0
        self._last_activity: float | None = None
        self._loaded_at: float | None = None
        self._ready_info: dict = {}
        self._blocked_reason: str | None = None
        self._last_error: str | None = None
        self._idle_task: asyncio.Task | None = None

    @property
    def state(self) -> State:
        return self._state

    # ── public API ─────────────────────────────────────────────────────

    async def load(self) -> float | None:
        """Warm the model. Returns the cold-load seconds if this call loaded it."""
        async with self._lock:
            return await self._ensure_ready_locked()

    async def synthesize(self, plan: Plan, voice_paths: list[str], *, seed: int,
                         cfg_scale: float, diffusion_steps: int) -> Synthesis:
        self._inflight += 1
        try:
            async with self._lock:
                cold = await self._ensure_ready_locked()
                payload = {
                    "type": "synth",
                    "turns": [[role, turn.text] for role, turn in zip(plan.roles, plan.turns)],
                    "voices": voice_paths,
                    "seed": seed,
                    "cfg_scale": cfg_scale,
                    "diffusion_steps": diffusion_steps,
                    "max_new_tokens": plan.max_new_tokens,
                }
                assert self._worker is not None
                try:
                    msg = await self._worker.request(payload, self._synth_timeout)
                except WorkerTimeout as exc:
                    await self._drop_worker_locked(
                        f"synthesis timed out after {self._synth_timeout:.0f}s; worker stopped")
                    raise EngineError(str(exc), reason="synth_timeout") from exc
                except WorkerError as exc:
                    if self._worker is None or not self._worker.alive():
                        await self._drop_worker_locked(f"worker died: {exc}")
                        raise EngineError(f"worker died: {exc}", reason="worker_died") from exc
                    self._last_error = str(exc)
                    raise EngineError(str(exc), reason="synth_failed") from exc
                self._jobs_done += 1
                return Synthesis(
                    audio=np.frombuffer(msg["pcm"], dtype=np.float32),
                    sample_rate=int(msg["sample_rate"]),
                    gen_secs=float(msg["gen_secs"]),
                    peak_vram_gib=float(msg["peak_vram_gib"]),
                    audio_tokens=int(msg["audio_tokens"]),
                    finished=bool(msg["finished"]),
                    cold_load_secs=cold,
                )
        finally:
            self._inflight -= 1
            self._last_activity = time.monotonic()

    async def unload(self) -> bool:
        """Stop the worker (waits for a running job). True if one was running."""
        async with self._lock:
            return await self._unload_locked()

    async def shutdown(self) -> None:
        """Server exit: stop the worker now, without queueing behind a
        running job (that job fails; its client gets a 500)."""
        if self._idle_task and not self._idle_task.done():
            self._idle_task.cancel()
        worker, self._worker = self._worker, None
        if worker is not None:
            pid = worker.pid
            await worker.stop()
            logger.info("[PodcastEngine] shutdown: worker pid %s stopped", pid)
        self._set_state(State.IDLE)

    def status(self) -> dict:
        now = time.monotonic()
        idle_for = None if self._last_activity is None else round(now - self._last_activity, 1)
        unload_in = None
        if self._state is State.READY and self._idle_secs > 0 and idle_for is not None:
            unload_in = max(0.0, round(self._idle_secs - idle_for, 1))
        return {
            "state": self._state.value,
            "engine": self.engine_name,
            "model": self.model_label,
            "worker_pid": self._worker.pid if self._worker is not None else None,
            "loaded_at": self._loaded_at,
            "cold_load_secs": self._ready_info.get("cold_load_secs"),
            "import_secs": self._ready_info.get("import_secs"),
            "load_secs": self._ready_info.get("load_secs"),
            "resident_vram_gib": self._ready_info.get("resident_vram_gib"),
            "inflight": self._inflight,
            "jobs_done": self._jobs_done,
            "idle_secs": idle_for,
            "idle_unload_secs": self._idle_secs,
            "idle_unload_in_secs": unload_in,
            "blocked_reason": self._blocked_reason if self._state is State.BLOCKED else None,
            "last_error": self._last_error,
        }

    # ── internals (call with self._lock held) ──────────────────────────

    async def _ensure_ready_locked(self) -> float | None:
        if self._state is State.READY and self._worker is not None and self._worker.alive():
            return None
        if self._worker is not None:                  # died while READY
            await self._drop_worker_locked("worker exited unexpectedly")

        if self._min_free > 0:
            free = await (self._gpu_probe() if self._gpu_probe else asyncio.to_thread(gpu_free_gib))
            if free is not None and free < self._min_free:
                self._blocked_reason = (f"insufficient free GPU memory ({free:.1f} GiB free, "
                                        f"need {self._min_free:.1f})")
                self._set_state(State.BLOCKED)
                raise EngineError(self._blocked_reason, reason="blocked")

        self._set_state(State.LOADING)
        started = time.monotonic()
        worker = self._factory()
        try:
            info = await worker.start(self._load_timeout)
        except WorkerTimeout as exc:
            self._last_error = f"model load timed out after {self._load_timeout:.0f}s"
            self._set_state(State.IDLE)
            raise EngineError(self._last_error, reason="load_timeout") from exc
        except WorkerError as exc:
            self._last_error = str(exc)
            self._set_state(State.IDLE)
            raise EngineError(str(exc), reason="load_failed") from exc
        cold = round(time.monotonic() - started, 2)
        self._worker = worker
        self._ready_info = {**info, "cold_load_secs": cold}
        self._loaded_at = time.time()
        self._last_error = None
        self._last_activity = time.monotonic()
        self._set_state(State.READY)
        logger.info("[PodcastEngine] loaded %s in %.1fs (pid %s)", self.model_label, cold, worker.pid)
        self._schedule_idle_check()
        return cold

    async def _unload_locked(self) -> bool:
        if self._worker is None:
            if self._state is State.BLOCKED:
                self._set_state(State.IDLE)
            return False
        pid = self._worker.pid
        await self._worker.stop()
        self._worker = None
        self._loaded_at = None
        self._ready_info = {}
        self._set_state(State.IDLE)
        logger.info("[PodcastEngine] unloaded (worker pid %s stopped)", pid)
        return True

    async def _drop_worker_locked(self, reason: str) -> None:
        self._last_error = reason
        logger.warning("[PodcastEngine] %s", reason)
        await self._unload_locked()

    def _set_state(self, state: State) -> None:
        if state is not self._state:
            logger.info("[PodcastEngine] %s → %s", self._state.value, state.value)
            self._state = state

    def _schedule_idle_check(self) -> None:
        if self._idle_secs <= 0:
            return
        if self._idle_task and not self._idle_task.done():
            return
        self._idle_task = asyncio.create_task(self._idle_loop())

    async def _idle_loop(self) -> None:
        while True:
            await asyncio.sleep(min(self._idle_secs, 30.0))
            if self._state is not State.READY:
                return
            if self._inflight > 0 or self._last_activity is None:
                continue
            if time.monotonic() - self._last_activity < self._idle_secs:
                continue
            async with self._lock:
                if (self._state is State.READY and self._inflight == 0
                        and time.monotonic() - self._last_activity >= self._idle_secs):
                    logger.info("[PodcastEngine] idle for %.0fs — unloading", self._idle_secs)
                    await self._unload_locked()
                    return


def build_engine(settings) -> PodcastEngine:
    """The engine selected by PODCAST_ENGINE ("vibevoice" or "mock")."""
    common = dict(
        idle_unload_secs=settings.idle_unload_secs,
        load_timeout_secs=settings.load_timeout_secs,
        synth_timeout_secs=settings.synth_timeout_secs,
    )
    if settings.engine == "mock":
        return PodcastEngine(lambda: MockWorker(delay_secs=settings.mock_delay_secs),
                             engine_name="mock", model_label="mock", **common)
    if settings.engine == "vibevoice":
        cfg = {"model_id": settings.model_id, "model_revision": settings.model_revision,
               "device": settings.device, "dtype": settings.dtype}
        return PodcastEngine(lambda: ProcessWorker(cfg), engine_name="vibevoice",
                             model_label=f"{settings.model_id}@{settings.model_revision[:8]}",
                             min_free_vram_gib=settings.min_free_vram_gib, **common)
    raise ValueError(f"unknown PODCAST_ENGINE {settings.engine!r} (vibevoice | mock)")
