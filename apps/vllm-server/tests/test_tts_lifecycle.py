"""Tests for the on-demand TTS subprocess lifecycle.

We exercise the full state machine without spawning real subprocesses or
querying real GPUs. The lifecycle takes a `_Probe` object that lets tests
inject:

  - `spawn_subprocess()` — returns a fake process with `pid`/`returncode`/
    `terminate()`/`wait()` etc.
  - `check_gpu_free()` — returns free VRAM in GiB, or None.
  - `health_check()` — returns True once the boot is "complete".
  - `sleep()` — replaces asyncio.sleep so we can fast-forward the idle
    timer without burning wall-clock.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from voxtral_server.tts.lifecycle import (
    LifecycleError,
    State,
    TtsLifecycle,
    _Probe,
)


pytestmark = pytest.mark.asyncio


class _FakeProc:
    """Minimal stand-in for `asyncio.subprocess.Process`."""

    def __init__(self) -> None:
        self.pid = 12345
        self.returncode: int | None = None
        self.terminated = False
        self.killed = False
        self._exited = asyncio.Event()

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = 0
        self._exited.set()

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9
        self._exited.set()

    async def wait(self) -> int:
        await self._exited.wait()
        return self.returncode or 0


def _make_lifecycle(
    *,
    spawn=None,
    health_calls=None,
    gpu_free: float | None = None,
    autostart: bool = True,
    idle_unload_secs: float = 0.05,    # tiny to let idle test fire fast
    boot_timeout_secs: float = 5.0,
) -> tuple[TtsLifecycle, _FakeProc]:
    """Build a lifecycle with a fake subprocess + injectable health probe."""
    proc = _FakeProc()

    async def _spawn() -> _FakeProc:
        if spawn is not None:
            return await spawn()
        return proc

    health_iter = iter(health_calls or [True])

    async def _health_check() -> bool:
        try:
            return next(health_iter)
        except StopIteration:
            return True

    async def _gpu() -> float | None:
        return gpu_free

    async def _sleep(secs: float) -> None:
        # Yield control without burning wall clock.
        await asyncio.sleep(0)

    probe = _Probe(
        spawn_subprocess=_spawn,
        check_gpu_free=_gpu,
        health_check=_health_check,
        sleep=_sleep,
    )
    lc = TtsLifecycle(
        autostart=autostart,
        idle_unload_secs=idle_unload_secs,
        boot_timeout_secs=boot_timeout_secs,
        probe=probe,
    )
    return lc, proc


# ─── State machine ────────────────────────────────────────────────────

async def test_ensure_started_transitions_to_ready() -> None:
    lc, proc = _make_lifecycle()
    assert lc.state is State.IDLE
    await lc.ensure_started()
    assert lc.state is State.READY
    assert (await lc.status()).pid == proc.pid


async def test_concurrent_ensure_started_collapses_to_one_boot() -> None:
    """Two callers awaiting ensure_started simultaneously must share one
    subprocess, not race-spawn two."""
    spawn_count = 0

    async def _spawn() -> _FakeProc:
        nonlocal spawn_count
        spawn_count += 1
        await asyncio.sleep(0.01)
        return _FakeProc()

    lc, _ = _make_lifecycle(spawn=_spawn)
    await asyncio.gather(lc.ensure_started(), lc.ensure_started(), lc.ensure_started())
    assert spawn_count == 1
    assert lc.state is State.READY


async def test_ensure_stopped_releases_subprocess() -> None:
    lc, proc = _make_lifecycle()
    await lc.ensure_started()
    await lc.ensure_stopped()
    assert lc.state is State.IDLE
    assert proc.terminated is True


async def test_idle_unload_fires_after_threshold() -> None:
    """With no activity, the idle timer must trigger ensure_stopped."""
    lc, proc = _make_lifecycle(idle_unload_secs=0.05)
    await lc.ensure_started()
    lc.note_activity()  # set last_activity_at so the loop has a baseline
    # Immediately backdate the activity so the loop's "since" exceeds the
    # threshold on first wake-up.
    lc._last_activity_at = time.time() - 10.0
    # Give the idle loop a few asyncio ticks to run.
    for _ in range(20):
        await asyncio.sleep(0)
        if lc.state is State.IDLE:
            break
    assert lc.state is State.IDLE
    assert proc.terminated is True


async def test_idle_unload_skipped_during_inflight_synth() -> None:
    """Long-form syntheses that span the idle window must not be killed."""
    lc, proc = _make_lifecycle(idle_unload_secs=0.05)
    await lc.ensure_started()
    lc.synth_started()                              # in-flight = 1
    lc._last_activity_at = time.time() - 10.0       # would trigger unload otherwise
    for _ in range(10):
        await asyncio.sleep(0)
    assert lc.state is State.READY                   # not torn down
    lc.synth_finished()                              # in-flight = 0


async def test_blocked_when_gpu_short_and_asr_running(monkeypatch) -> None:
    """If free VRAM is below the threshold AND ASR is on the card, surface
    a 'blocked by ASR session' error, not a generic OOM."""
    lc, _ = _make_lifecycle(gpu_free=2.0)
    monkeypatch.setattr(lc, "_is_asr_running", lambda: _async_true())

    with pytest.raises(LifecycleError) as exc:
        await lc.ensure_started()
    assert "blocked by ASR session" in str(exc.value)
    assert exc.value.reason == "blocked"
    assert lc.state is State.BLOCKED


async def test_blocked_generic_when_asr_not_running(monkeypatch) -> None:
    lc, _ = _make_lifecycle(gpu_free=2.0)
    monkeypatch.setattr(lc, "_is_asr_running", lambda: _async_false())

    with pytest.raises(LifecycleError) as exc:
        await lc.ensure_started()
    assert "insufficient free GPU memory" in str(exc.value)
    assert lc.state is State.BLOCKED


async def test_boot_timeout_kills_subprocess() -> None:
    """If the upstream never reports healthy within boot_timeout_secs,
    the lifecycle must SIGTERM the child and surface the timeout."""
    proc = _FakeProc()

    async def _never_healthy() -> bool:
        return False

    probe = _Probe(
        spawn_subprocess=lambda: _ret(proc),
        check_gpu_free=lambda: _ret(None),
        health_check=_never_healthy,
        sleep=lambda secs: asyncio.sleep(0),
    )
    lc = TtsLifecycle(autostart=True, boot_timeout_secs=0.05, probe=probe)

    with pytest.raises(LifecycleError) as exc:
        await lc.ensure_started()
    assert "did not become healthy" in str(exc.value)
    assert exc.value.reason == "boot_timeout"
    assert lc.state is State.IDLE


async def test_autostart_false_returns_not_running_when_upstream_down() -> None:
    """When autostart is disabled, ensure_started must NOT spawn —
    it just probes the upstream and surfaces 'not running' if absent."""
    spawn_count = 0

    async def _spawn():
        nonlocal spawn_count
        spawn_count += 1
        return _FakeProc()

    probe = _Probe(
        spawn_subprocess=_spawn,
        check_gpu_free=lambda: _ret(None),
        health_check=lambda: _ret(False),
        sleep=lambda secs: asyncio.sleep(0),
    )
    lc = TtsLifecycle(autostart=False, probe=probe)

    with pytest.raises(LifecycleError) as exc:
        await lc.ensure_started()
    assert exc.value.reason == "not_running"
    assert spawn_count == 0
    assert lc.state is State.IDLE


async def test_autostart_false_with_healthy_upstream_marks_ready() -> None:
    probe = _Probe(
        spawn_subprocess=lambda: _ret(_FakeProc()),
        check_gpu_free=lambda: _ret(None),
        health_check=lambda: _ret(True),
        sleep=lambda secs: asyncio.sleep(0),
    )
    lc = TtsLifecycle(autostart=False, probe=probe)
    await lc.ensure_started()
    assert lc.state is State.READY


# ─── helpers ──────────────────────────────────────────────────────────

async def _ret(value):
    return value


async def _async_true() -> bool:
    return True


async def _async_false() -> bool:
    return False
