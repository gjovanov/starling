"""On-demand TTS process lifecycle.

The vllm-omni TTS engine + the existing ASR vLLM together overflow the
24 GiB GPU we benchmark on (~22 GiB ASR + ~12 GiB TTS = OOM, see
Phase 1.1 in docs/tts_spike.md). This module owns the TTS subprocess so
we can:

  - lazy-start it on the first /api/tts/* request,
  - unload it after `tts_idle_unload_secs` of inactivity,
  - refuse to start if ASR is currently using the GPU
    (returns "blocked by ASR session" instead of OOMing).

The public API is `lifecycle.ensure_started()` (called from each TTS
route) and `lifecycle.status()` (polled by the frontend status badge).
Tests stub out `_spawn_subprocess` and `_check_gpu_free` to exercise the
state machine without a real GPU.

State machine:

    idle ──ensure_started()──► starting ──health-check ok──► ready
       ▲                          │                            │
       │                          ▼                            │
       │                      (boot fail)                      │
       │                          │                            │
       │                          ▼                            │
       └──────────── stopping ◄───┴──── idle-unload OR ensure_stopped()
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import signal
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class State(str, Enum):
    IDLE = "idle"
    STARTING = "starting"
    READY = "ready"
    STOPPING = "stopping"
    BLOCKED = "blocked"          # GPU pre-check failed (ASR is on the card)


class LifecycleError(RuntimeError):
    """Raised when start/stop fails for a reason the caller should surface
    to the user (boot timeout, blocked by ASR, etc.)."""

    def __init__(self, message: str, *, reason: str = "error") -> None:
        super().__init__(message)
        self.reason = reason


@dataclass
class StatusInfo:
    state: str
    pid: int | None = None
    boot_started_at: float | None = None
    boot_elapsed_secs: float | None = None
    boot_timeout_secs: float | None = None
    last_activity_at: float | None = None
    inflight_synths: int = 0
    blocked_reason: str | None = None
    error: str | None = None
    task_type: str | None = None


@dataclass
class _Probe:
    """Runtime hooks the tests can monkey-patch to avoid real subprocess
    + nvidia-smi calls. Production wiring uses the defaults."""

    spawn_subprocess: Any = None
    check_gpu_free: Any = None
    health_check: Any = None
    sleep: Any = None


class TtsLifecycle:
    """Owns the lifecycle of the vllm-omni TTS subprocess."""

    def __init__(
        self,
        *,
        start_command: list[str] | None = None,
        cwd: str | None = None,
        upstream_url: str = "http://127.0.0.1:8002/v1",
        idle_unload_secs: float = 600.0,
        boot_timeout_secs: float = 180.0,
        min_free_vram_gib: float = 12.0,
        asr_port: int = 8001,
        autostart: bool = True,
        probe: _Probe | None = None,
    ) -> None:
        self._start_command = start_command
        self._cwd = cwd
        self._upstream_url = upstream_url.rstrip("/")
        self._idle_unload_secs = idle_unload_secs
        self._boot_timeout_secs = boot_timeout_secs
        self._min_free_vram_gib = min_free_vram_gib
        self._asr_port = asr_port
        self._autostart = autostart
        self._probe = probe or _Probe()

        self._state = State.IDLE
        self._proc: Any = None             # asyncio.subprocess.Process or fake
        self._boot_lock = asyncio.Lock()
        self._boot_started_at: float | None = None
        self._last_activity_at: float | None = None
        self._inflight_synths = 0
        self._blocked_reason: str | None = None
        self._last_error: str | None = None
        self._idle_task: asyncio.Task | None = None
        # Phase 7: which `--task-type` the running upstream was launched
        # with. None when not running. ensure_started() restarts on
        # mismatch.
        self._task_type: str | None = None

    # ── public API ───────────────────────────────────────────────────

    @property
    def state(self) -> State:
        return self._state

    @property
    def autostart(self) -> bool:
        return self._autostart

    async def ensure_started(self, task_type: str = "CustomVoice") -> None:
        """Block until TTS is ready, spawning it if needed.

        Idempotent: if already READY *with the right task type*, returns
        immediately. If the running task type doesn't match, restarts the
        subprocess (~75 s on the spike hardware). Concurrent callers
        share a single boot via the lock.
        """
        if self._state is State.READY and self._task_type == task_type:
            return

        async with self._boot_lock:
            # Re-check after grabbing the lock — another caller may have
            # finished the boot while we were waiting.
            if self._state is State.READY and self._task_type == task_type:
                return

            # If we're ready but with a different task type, tear it down
            # first. We hold the lock so no synth requests can sneak in
            # between stop and start.
            if self._state is State.READY and self._task_type != task_type:
                logger.info(
                    "[TtsLifecycle] task_type mismatch (%s → %s), restarting",
                    self._task_type, task_type,
                )
                self._set_state(State.STOPPING)
                await self._terminate_subprocess()
                self._proc = None
                self._task_type = None
                self._set_state(State.IDLE)

            if not self._autostart:
                # Operator wants manual control. Just probe the upstream
                # and treat it as ready/blocked accordingly.
                if await self._is_upstream_healthy():
                    self._set_state(State.READY)
                    return
                raise LifecycleError(
                    "TTS server is not running and autostart is disabled.",
                    reason="not_running",
                )

            free = await self._gpu_free_gib()
            if free is not None and free < self._min_free_vram_gib:
                # If ASR is on the card, surface that specifically.
                if await self._is_asr_running():
                    self._blocked_reason = "blocked by ASR session"
                else:
                    self._blocked_reason = (
                        f"insufficient free GPU memory ({free:.1f} GiB free, "
                        f"need {self._min_free_vram_gib:.0f})"
                    )
                self._set_state(State.BLOCKED)
                raise LifecycleError(self._blocked_reason, reason="blocked")

            self._set_state(State.STARTING)
            self._boot_started_at = time.time()
            self._task_type = task_type
            try:
                await self._spawn_and_wait_ready()
            except LifecycleError:
                self._task_type = None
                self._set_state(State.IDLE)
                raise
            self._set_state(State.READY)
            self._schedule_idle_check()

    async def ensure_stopped(self) -> None:
        """Tear the subprocess down. No-op when already idle/blocked."""
        async with self._boot_lock:
            if self._state in {State.IDLE, State.BLOCKED}:
                return
            self._set_state(State.STOPPING)
            await self._terminate_subprocess()
            self._proc = None
            self._task_type = None
            self._set_state(State.IDLE)
            if self._idle_task and not self._idle_task.done():
                self._idle_task.cancel()
                self._idle_task = None

    async def status(self) -> StatusInfo:
        """Cheap snapshot for the frontend badge."""
        info = StatusInfo(
            state=self._state.value,
            pid=getattr(self._proc, "pid", None),
            boot_started_at=self._boot_started_at,
            boot_timeout_secs=self._boot_timeout_secs,
            last_activity_at=self._last_activity_at,
            inflight_synths=self._inflight_synths,
            blocked_reason=self._blocked_reason if self._state is State.BLOCKED else None,
            error=self._last_error,
            task_type=self._task_type,
        )
        if self._state is State.STARTING and self._boot_started_at is not None:
            info.boot_elapsed_secs = time.time() - self._boot_started_at
        return info

    def note_activity(self) -> None:
        """Reset the idle timer. Called after every successful synth."""
        self._last_activity_at = time.time()

    def synth_started(self) -> None:
        """Inc the in-flight synth counter (delays idle-unload)."""
        self._inflight_synths += 1

    def synth_finished(self) -> None:
        self._inflight_synths = max(0, self._inflight_synths - 1)
        self.note_activity()

    # ── internals ─────────────────────────────────────────────────────

    def _set_state(self, next_state: State) -> None:
        if self._state is next_state:
            return
        logger.info("[TtsLifecycle] %s → %s", self._state.value, next_state.value)
        self._state = next_state
        if next_state is not State.BLOCKED:
            self._blocked_reason = None
        if next_state is State.READY:
            # Reset error on a fresh successful boot.
            self._last_error = None

    async def _spawn_and_wait_ready(self) -> None:
        """Spawn the child + poll the upstream health endpoint until ready."""
        try:
            self._proc = await self._spawn()
        except Exception as exc:  # noqa: BLE001 — re-wrap for the route
            self._last_error = f"failed to spawn: {exc}"
            raise LifecycleError(self._last_error, reason="spawn_failed") from exc

        sleep = self._probe.sleep or asyncio.sleep
        deadline = time.monotonic() + self._boot_timeout_secs
        while time.monotonic() < deadline:
            if await self._is_upstream_healthy():
                return
            # Detect early death of the child — the upstream may have
            # crashed during model load.
            if self._proc is not None and self._proc.returncode is not None:
                self._last_error = (
                    f"TTS subprocess exited during boot (code={self._proc.returncode})"
                )
                raise LifecycleError(self._last_error, reason="early_exit")
            await sleep(1.0)

        self._last_error = (
            f"TTS subprocess did not become healthy within "
            f"{self._boot_timeout_secs:.0f}s"
        )
        await self._terminate_subprocess()
        raise LifecycleError(self._last_error, reason="boot_timeout")

    async def _spawn(self) -> Any:
        """Hook seam for tests (`_probe.spawn_subprocess`) and prod."""
        if self._probe.spawn_subprocess is not None:
            return await self._probe.spawn_subprocess()
        if not self._start_command:
            raise LifecycleError(
                "no start_command configured (set start_command= or supply a probe)",
                reason="config_missing",
            )
        # Spawn detached enough that our SIGTERM doesn't accidentally
        # propagate from a parent. We don't pipe stdout; the start script
        # tees its own log.
        env = os.environ.copy()
        if self._task_type is not None:
            env["VOXTRAL_TTS_TASK_TYPE"] = self._task_type
        return await asyncio.create_subprocess_exec(
            *self._start_command,
            cwd=self._cwd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            start_new_session=True,
            env=env,
        )

    async def _terminate_subprocess(self) -> None:
        if self._proc is None:
            return
        if self._proc.returncode is not None:
            return
        try:
            self._proc.terminate()
        except ProcessLookupError:
            return
        # Wait up to 30s for graceful exit, then SIGKILL.
        try:
            await asyncio.wait_for(self._proc.wait(), timeout=30.0)
        except asyncio.TimeoutError:
            logger.warning("[TtsLifecycle] SIGTERM timed out, sending SIGKILL")
            try:
                self._proc.kill()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=10.0)
            except asyncio.TimeoutError:
                logger.error("[TtsLifecycle] SIGKILL also timed out — pid leaked")

    async def _is_upstream_healthy(self) -> bool:
        if self._probe.health_check is not None:
            return bool(await self._probe.health_check())
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.get(f"{self._upstream_url}/models")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    async def _gpu_free_gib(self) -> float | None:
        if self._probe.check_gpu_free is not None:
            return await self._probe.check_gpu_free()
        if shutil.which("nvidia-smi") is None:
            return None
        try:
            proc = await asyncio.create_subprocess_exec(
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5.0)
            line = stdout.decode().strip().splitlines()[0]
            return float(line) / 1024.0
        except (asyncio.TimeoutError, ValueError, IndexError, OSError):
            return None

    async def _is_asr_running(self) -> bool:
        # Best-effort port probe — `lsof -ti:<port>` returns a non-empty
        # output if a listener is bound. Useful even without nvidia-smi.
        try:
            proc = await asyncio.create_subprocess_exec(
                "lsof", "-ti", f":{self._asr_port}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=2.0)
            return stdout.strip() != b""
        except (asyncio.TimeoutError, OSError):
            return False

    def _schedule_idle_check(self) -> None:
        if self._idle_unload_secs <= 0:
            return  # disabled
        if self._idle_task and not self._idle_task.done():
            return
        self._idle_task = asyncio.create_task(self._idle_loop())

    async def _idle_loop(self) -> None:
        sleep = self._probe.sleep or asyncio.sleep
        while True:
            try:
                await sleep(min(self._idle_unload_secs, 30.0))
            except asyncio.CancelledError:
                return
            if self._state is not State.READY:
                return
            if self._inflight_synths > 0:
                continue  # do not unload during a long-form synth
            if self._last_activity_at is None:
                continue
            since = time.time() - self._last_activity_at
            if since >= self._idle_unload_secs:
                logger.info(
                    "[TtsLifecycle] idle for %.0fs (>= %.0fs), unloading",
                    since, self._idle_unload_secs,
                )
                await self.ensure_stopped()
                return


# Module-level singleton — created lazily by main.py / tests.
_singleton: TtsLifecycle | None = None


def get_lifecycle() -> TtsLifecycle:
    """Return the process-wide lifecycle singleton."""
    global _singleton
    if _singleton is None:
        from ..config import settings

        # Default start command: invoke start-vllm-tts.sh from the app dir.
        # Path is relative-from-CWD-tolerant: the FastAPI server boots
        # with cwd = apps/vllm-server (per start.sh).
        app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # voxtral_server/ -> .. -> apps/vllm-server
        app_dir = os.path.dirname(app_dir)
        script = os.path.join(app_dir, "start-vllm-tts.sh")
        _singleton = TtsLifecycle(
            start_command=["bash", script],
            cwd=app_dir,
            upstream_url=settings.tts_vllm_url,
            idle_unload_secs=settings.tts_idle_unload_secs,
            boot_timeout_secs=settings.tts_boot_timeout_secs,
            min_free_vram_gib=settings.tts_min_free_vram_gib,
            autostart=settings.tts_autostart,
        )
    return _singleton


def reset_lifecycle() -> None:
    """Test helper — drop the cached singleton."""
    global _singleton
    _singleton = None
