"""Generic vLLM subprocess lifecycle: lazy spawn, health poll, idle unload.

Mirrors voxtral_server/tts/lifecycle.py (state machine + _Probe injection for
tests). Used by the GLM-OCR engine (:8003) and the extraction LLM (:8004).

  idle ──ensure_started()──► starting ──health ok──► ready
    ▲                           │(boot fail)            │
    └────────── stopping ◄──────┴── idle-unload OR ensure_stopped()
"""

import asyncio
import logging
import os
import signal
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable

import httpx

from .base import EngineError

logger = logging.getLogger(__name__)


class ProcState(str, Enum):
    IDLE = "idle"
    STARTING = "starting"
    READY = "ready"
    STOPPING = "stopping"


@dataclass
class Probe:
    """Test seams — production leaves all fields None."""

    spawn_subprocess: Callable[[], Awaitable[Any]] | None = None
    health_check: Callable[[], Awaitable[bool]] | None = None
    sleep: Callable[[float], Awaitable[None]] | None = None
    extra_env: dict[str, str] = field(default_factory=dict)


class VllmProcess:
    def __init__(
        self,
        *,
        name: str,
        start_command: list[str],
        base_url: str,
        cwd: str | None = None,
        boot_timeout_secs: float = 300.0,
        idle_unload_secs: float = 600.0,
        probe: Probe | None = None,
    ) -> None:
        self.name = name
        self._start_command = start_command
        self._cwd = cwd
        self._base_url = base_url.rstrip("/")
        self._boot_timeout_secs = boot_timeout_secs
        self._idle_unload_secs = idle_unload_secs
        self._probe = probe or Probe()

        self._state = ProcState.IDLE
        self._proc: Any = None
        self._boot_lock = asyncio.Lock()
        self._last_activity_at: float | None = None
        self._idle_task: asyncio.Task | None = None
        self._on_idle_unload: Callable[[], Awaitable[None]] | None = None
        self.last_error: str | None = None

    # ── public API ───────────────────────────────────────────────────

    @property
    def state(self) -> ProcState:
        # A crashed child must read as IDLE — the engine manager's fast path
        # trusts this state, and a stale READY would skip the respawn in
        # ensure_started() (run 24 during Phase 5 hardening).
        if self._state is ProcState.READY and not self._alive():
            return ProcState.IDLE
        return self._state

    @property
    def base_url(self) -> str:
        return self._base_url

    def note_activity(self) -> None:
        self._last_activity_at = time.time()

    async def is_healthy(self) -> bool:
        """Live upstream probe (used to distinguish 'empty OCR result'
        from 'upstream died mid-request')."""
        return await self._is_healthy()

    def set_idle_unload_callback(self, cb: Callable[[], Awaitable[None]]) -> None:
        """Called instead of plain ensure_stopped() when the idle timer fires,
        so the owner (engine/manager) can broadcast WS events."""
        self._on_idle_unload = cb

    async def ensure_started(self) -> None:
        """Idempotent, serialized boot. Detects a crashed-under-us process and
        respawns it."""
        if self._state is ProcState.READY and self._alive():
            self.note_activity()
            return
        async with self._boot_lock:
            if self._state is ProcState.READY:
                if self._alive():
                    self.note_activity()
                    return
                logger.warning("[%s] process died while READY — respawning", self.name)
                self._proc = None
                self._set_state(ProcState.IDLE)

            self._set_state(ProcState.STARTING)
            try:
                await self._spawn_and_wait_ready()
            except Exception:
                self._set_state(ProcState.IDLE)
                raise
            self._set_state(ProcState.READY)
            self.note_activity()
            self._schedule_idle_check()

    async def ensure_stopped(self) -> None:
        async with self._boot_lock:
            if self._state in (ProcState.IDLE, ProcState.STOPPING):
                return
            self._set_state(ProcState.STOPPING)
            await self._terminate()
            self._proc = None
            self._set_state(ProcState.IDLE)

    # ── internals ────────────────────────────────────────────────────

    def _set_state(self, state: ProcState) -> None:
        if state is not self._state:
            logger.info("[%s] %s → %s", self.name, self._state.value, state.value)
            self._state = state

    def _alive(self) -> bool:
        # An externally-started server (no child proc) counts as alive; the
        # health check during boot verified it.
        return self._proc is None or self._proc.returncode is None

    async def _spawn(self) -> Any:
        if self._probe.spawn_subprocess is not None:
            return await self._probe.spawn_subprocess()
        env = os.environ.copy()
        env.update(self._probe.extra_env)
        # Boot failures must be debuggable — tee the subprocess into a log
        # file (same convention as burn-server's /tmp/burn-server.log).
        log_path = os.environ.get(f"OCR_{self.name.upper().replace('-', '_')}_LOG",
                                  f"/tmp/ocr-{self.name}.log")
        log_file = open(log_path, "ab")
        logger.info("[%s] spawning (log: %s)", self.name, log_path)
        try:
            return await asyncio.create_subprocess_exec(
                *self._start_command,
                cwd=self._cwd,
                stdout=log_file,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
                env=env,
            )
        finally:
            log_file.close()

    async def _is_healthy(self) -> bool:
        if self._probe.health_check is not None:
            return bool(await self._probe.health_check())
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.get(f"{self._base_url}/health")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    async def _spawn_and_wait_ready(self) -> None:
        # An already-running upstream (manual ./start-vllm-ocr.sh) is adopted
        # instead of spawned twice.
        if await self._is_healthy():
            self._proc = None
            return
        try:
            self._proc = await self._spawn()
        except Exception as exc:
            self.last_error = f"failed to spawn {self.name}: {exc}"
            raise EngineError(self.last_error) from exc

        sleep = self._probe.sleep or asyncio.sleep
        deadline = time.monotonic() + self._boot_timeout_secs
        while time.monotonic() < deadline:
            if await self._is_healthy():
                return
            if self._proc is not None and self._proc.returncode is not None:
                self.last_error = (
                    f"{self.name} subprocess exited during boot "
                    f"(code={self._proc.returncode})")
                raise EngineError(self.last_error)
            await sleep(1.0)

        self.last_error = (
            f"{self.name} did not become healthy within {self._boot_timeout_secs:.0f}s")
        await self._terminate()
        self._proc = None
        raise EngineError(self.last_error)

    async def _terminate(self) -> None:
        if self._proc is None or self._proc.returncode is not None:
            return
        # vLLM forks EngineCore + resource_tracker children (setproctitle'd,
        # so name-based pkill misses them). Signaling only the master leaks
        # EngineCores that keep holding VRAM and the listen socket — signal
        # the whole process group (start_new_session made the child leader).
        use_group = self._probe.spawn_subprocess is None  # fakes have fake pids

        def _signal_group(sig: int) -> bool:
            if not use_group:
                return False
            try:
                os.killpg(self._proc.pid, sig)
                return True
            except (ProcessLookupError, PermissionError, OSError):
                return False

        if not _signal_group(signal.SIGTERM):
            try:
                self._proc.terminate()
            except ProcessLookupError:
                return
        try:
            await asyncio.wait_for(self._proc.wait(), timeout=30.0)
        except asyncio.TimeoutError:
            logger.warning("[%s] SIGTERM timed out, sending SIGKILL", self.name)
            if not _signal_group(signal.SIGKILL):
                try:
                    self._proc.kill()
                except ProcessLookupError:
                    pass
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=10.0)
            except asyncio.TimeoutError:
                logger.error("[%s] SIGKILL also timed out — pid leaked", self.name)
        # Sweep stragglers: EngineCore children can briefly outlive the master.
        _signal_group(signal.SIGKILL)

    def _schedule_idle_check(self) -> None:
        if self._idle_unload_secs <= 0:
            return
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
            if self._state is not ProcState.READY:
                return
            if self._last_activity_at is None:
                continue
            since = time.time() - self._last_activity_at
            if since >= self._idle_unload_secs:
                logger.info("[%s] idle for %.0fs (>= %.0fs), unloading",
                            self.name, since, self._idle_unload_secs)
                if self._on_idle_unload is not None:
                    await self._on_idle_unload()
                else:
                    await self.ensure_stopped()
                return
