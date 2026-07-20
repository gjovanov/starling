"""VllmProcess lifecycle unit tests — probe-injected fakes, no GPU/subprocess."""

import asyncio

import pytest

from ocr_server.engines.base import EngineError
from ocr_server.engines.vllm_process import Probe, ProcState, VllmProcess


class FakeProc:
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

    def die(self, code: int = 1) -> None:
        self.returncode = code
        self._exited.set()


def make(*, healthy, proc: FakeProc | None = None, idle_unload_secs: float = 0.0,
         boot_timeout_secs: float = 5.0, sleeps: list | None = None) -> VllmProcess:
    the_proc = proc or FakeProc()

    async def spawn():
        return the_proc

    async def fast_sleep(secs: float) -> None:
        if sleeps is not None:
            sleeps.append(secs)
        await asyncio.sleep(0)

    return VllmProcess(
        name="test-vllm",
        start_command=["true"],
        base_url="http://127.0.0.1:9",
        boot_timeout_secs=boot_timeout_secs,
        idle_unload_secs=idle_unload_secs,
        probe=Probe(spawn_subprocess=spawn, health_check=healthy, sleep=fast_sleep),
    )


async def test_boot_success():
    calls = {"n": 0}

    async def healthy():
        calls["n"] += 1
        return calls["n"] >= 3  # ready on the 3rd poll

    p = make(healthy=healthy)
    await p.ensure_started()
    assert p.state is ProcState.READY


async def test_boot_adopts_running_upstream():
    async def healthy():
        return True

    proc = FakeProc()
    p = make(healthy=healthy, proc=proc)
    await p.ensure_started()
    assert p.state is ProcState.READY
    # Healthy before spawn → adopted, never spawned.
    await p.ensure_stopped()
    assert proc.terminated is False


async def test_boot_timeout_kills_child():
    async def healthy():
        return False

    proc = FakeProc()
    p = make(healthy=healthy, proc=proc, boot_timeout_secs=0.01)
    with pytest.raises(EngineError, match="did not become healthy"):
        await p.ensure_started()
    assert p.state is ProcState.IDLE
    assert proc.terminated


async def test_early_exit_detected():
    proc = FakeProc()
    calls = {"n": 0}

    async def healthy():
        calls["n"] += 1
        if calls["n"] == 2:
            proc.die(3)  # child crashes after the first poll
        return False

    p = make(healthy=healthy, proc=proc)
    with pytest.raises(EngineError, match="exited during boot"):
        await p.ensure_started()
    assert p.state is ProcState.IDLE


async def test_crash_while_ready_respawns():
    flags = {"healthy_once": False}

    async def healthy():
        return True

    proc = FakeProc()
    spawned = []

    async def spawn():
        p2 = FakeProc()
        spawned.append(p2)
        return p2

    p = VllmProcess(
        name="t", start_command=["true"], base_url="http://127.0.0.1:9",
        boot_timeout_secs=5.0, idle_unload_secs=0.0,
        probe=Probe(spawn_subprocess=spawn, health_check=healthy,
                    sleep=lambda s: asyncio.sleep(0)))
    # First boot is adopted (healthy immediately, no spawn).
    await p.ensure_started()
    assert p.state is ProcState.READY and not spawned

    # Simulate: we own a proc and it dies.
    p._proc = proc
    proc.die(9)
    await p.ensure_started()
    assert p.state is ProcState.READY  # respawned (adopted again via health)


async def test_state_property_reflects_dead_child():
    calls = {"n": 0}

    async def healthy():
        calls["n"] += 1
        return calls["n"] >= 2

    proc = FakeProc()
    p = make(healthy=healthy, proc=proc)
    await p.ensure_started()
    assert p.state is ProcState.READY
    # Child dies underneath us (kill -9 from outside): public state must
    # read IDLE so manager fast paths re-enter ensure_started.
    proc.die(9)
    assert p.state is ProcState.IDLE
    await p.ensure_started()  # respawns (fake spawn returns same proc obj —
    proc.returncode = None    # resurrect the fake for the healthy re-check)
    assert p.state is ProcState.READY


async def test_ensure_stopped_terminates():
    async def healthy():
        return True

    proc = FakeProc()

    async def spawn():
        return proc

    p = VllmProcess(
        name="t", start_command=["true"], base_url="http://127.0.0.1:9",
        probe=Probe(spawn_subprocess=spawn,
                    health_check=None, sleep=lambda s: asyncio.sleep(0)))
    # Force the spawn path: unhealthy once, then healthy.
    calls = {"n": 0}

    async def health2():
        calls["n"] += 1
        return calls["n"] >= 2

    p._probe.health_check = health2
    await p.ensure_started()
    assert p.state is ProcState.READY
    await p.ensure_stopped()
    assert p.state is ProcState.IDLE
    assert proc.terminated


async def test_idle_unload_fires():
    calls = {"n": 0}

    async def healthy():
        calls["n"] += 1
        return calls["n"] >= 2

    p = make(healthy=healthy, idle_unload_secs=0.05)
    await p.ensure_started()
    assert p.state is ProcState.READY
    p._last_activity_at -= 10  # pretend the last activity was long ago
    # Let the idle loop tick (probe sleep is instant).
    for _ in range(50):
        await asyncio.sleep(0)
        if p.state is ProcState.IDLE:
            break
    assert p.state is ProcState.IDLE


async def test_idle_unload_callback_used():
    calls = {"n": 0, "cb": 0}

    async def healthy():
        calls["n"] += 1
        return calls["n"] >= 2

    p = make(healthy=healthy, idle_unload_secs=0.05)

    async def cb():
        calls["cb"] += 1
        await p.ensure_stopped()

    p.set_idle_unload_callback(cb)
    await p.ensure_started()
    p._last_activity_at -= 10
    for _ in range(50):
        await asyncio.sleep(0)
        if p.state is ProcState.IDLE:
            break
    assert calls["cb"] == 1
    assert p.state is ProcState.IDLE
