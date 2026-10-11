import asyncio

import pytest

from podcast_server.config import Settings
from podcast_server.engine import EngineError, PodcastEngine, State
from podcast_server.script import Turn, plan
from podcast_server.worker import MockWorker, WorkerError, WorkerTimeout

VOICES = {"michael", "emma"}


class Factory:
    """Counts spawns; hands out MockWorkers (or a scripted subclass)."""

    def __init__(self, cls=MockWorker, **kw):
        self.cls, self.kw, self.made = cls, kw, []

    def __call__(self):
        worker = self.cls(**self.kw)
        self.made.append(worker)
        return worker


def _engine(factory, **kw):
    kw.setdefault("idle_unload_secs", 0)
    return PodcastEngine(factory, engine_name="mock", model_label="mock", **kw)


async def _synth(engine, *texts):
    turns = [Turn("michael" if i % 2 == 0 else "emma", t) for i, t in enumerate(texts or ("hi",))]
    return await engine.synthesize(plan(turns, VOICES, Settings()), ["m.wav", "e.wav"],
                                   seed=1, cfg_scale=1.3, diffusion_steps=10)


async def _wait_for(predicate, timeout=3.0):
    for _ in range(int(timeout / 0.02)):
        if predicate():
            return True
        await asyncio.sleep(0.02)
    return predicate()


async def test_lazy_load_then_reuse():
    f = Factory()
    e = _engine(f)
    assert e.state is State.IDLE and not f.made
    first = await _synth(e, "hello")
    assert e.state is State.READY and len(f.made) == 1
    assert first.cold_load_secs is not None
    second = await _synth(e, "again")
    assert len(f.made) == 1 and second.cold_load_secs is None
    assert e.status()["jobs_done"] == 2
    await e.shutdown()
    assert e.state is State.IDLE and not f.made[0].alive()


async def test_mock_render_shape():
    e = _engine(Factory())
    r = await _synth(e, "x" * 20, "y" * 5)        # 1.2 s + 0.25 s gap + 0.5 s minimum
    assert r.sample_rate == 24000 and r.finished
    assert abs(r.duration_secs - (1.2 + 0.25 + 0.5)) < 1e-3
    await e.shutdown()


async def test_unload_then_reload():
    f = Factory()
    e = _engine(f)
    await _synth(e)
    assert await e.unload() is True
    assert e.state is State.IDLE and not f.made[0].alive()
    assert await e.unload() is False
    await _synth(e)
    assert len(f.made) == 2 and e.state is State.READY
    await e.shutdown()


async def test_idle_unload_fires():
    f = Factory()
    e = _engine(f, idle_unload_secs=0.05)
    await _synth(e)
    assert await _wait_for(lambda: e.state is State.IDLE)
    assert not f.made[0].alive()


async def test_idle_unload_never_interrupts_a_running_job():
    f = Factory(delay_secs=0.3)                     # load 0.3 s + synth 0.3 s
    e = _engine(f, idle_unload_secs=0.05)
    job = asyncio.create_task(_synth(e, "slow"))
    await asyncio.sleep(0.45)                       # far past idle_unload_secs, mid-synth
    assert e.state is State.READY and e.status()["inflight"] == 1
    result = await job
    assert result.audio.size > 0
    assert await _wait_for(lambda: e.state is State.IDLE)


async def test_blocked_when_vram_is_short_then_recovers():
    free = {"gib": 2.0}

    async def probe():
        return free["gib"]

    f = Factory()
    e = _engine(f, min_free_vram_gib=8.0, gpu_probe=probe)
    with pytest.raises(EngineError) as err:
        await _synth(e)
    assert err.value.reason == "blocked" and e.state is State.BLOCKED
    assert "2.0 GiB free" in e.status()["blocked_reason"] and not f.made
    free["gib"] = 20.0
    await _synth(e)
    assert e.state is State.READY and e.status()["blocked_reason"] is None
    await e.shutdown()


async def test_unreadable_vram_does_not_block():
    async def probe():
        return None

    e = _engine(Factory(), min_free_vram_gib=8.0, gpu_probe=probe)
    await _synth(e)
    assert e.state is State.READY
    await e.shutdown()


class FailingLoad(MockWorker):
    async def start(self, timeout):
        raise WorkerError("model load failed: boom")


class TimingOutLoad(MockWorker):
    async def start(self, timeout):
        raise WorkerTimeout("worker did not answer within 300s")


@pytest.mark.parametrize("cls, reason", [(FailingLoad, "load_failed"), (TimingOutLoad, "load_timeout")])
async def test_load_failures_return_to_idle(cls, reason):
    e = _engine(Factory(cls))
    with pytest.raises(EngineError) as err:
        await _synth(e)
    assert err.value.reason == reason and e.state is State.IDLE
    assert e.status()["last_error"]


class HangingSynth(MockWorker):
    async def request(self, payload, timeout):
        raise WorkerTimeout(f"worker did not answer within {timeout:.0f}s")


async def test_synth_timeout_drops_the_worker():
    f = Factory(HangingSynth)
    e = _engine(f)
    with pytest.raises(EngineError) as err:
        await _synth(e)
    assert err.value.reason == "synth_timeout"
    assert e.state is State.IDLE and not f.made[0].alive()


class CrashingSynth(MockWorker):
    async def request(self, payload, timeout):
        self._running = False
        raise WorkerError("worker exited (code=-9)")


async def test_worker_crash_is_reported_and_the_next_request_respawns():
    f = Factory(CrashingSynth)
    e = _engine(f)
    with pytest.raises(EngineError) as err:
        await _synth(e)
    assert err.value.reason == "worker_died" and e.state is State.IDLE
    with pytest.raises(EngineError):
        await _synth(e)
    assert len(f.made) == 2


async def test_worker_that_died_between_requests_is_replaced():
    f = Factory()
    e = _engine(f)
    await _synth(e)
    await f.made[0].stop()                          # e.g. OOM-killed while idle
    await _synth(e)                                 # must respawn, not error
    assert len(f.made) == 2 and e.state is State.READY
    await e.shutdown()


class ErroringSynth(MockWorker):
    async def request(self, payload, timeout):
        raise WorkerError("ValueError: bad input")


async def test_synth_error_keeps_the_worker():
    f = Factory(ErroringSynth)
    e = _engine(f)
    with pytest.raises(EngineError) as err:
        await _synth(e)
    assert err.value.reason == "synth_failed"
    assert e.state is State.READY and f.made[0].alive()
    await e.shutdown()


class ConcurrencyProbe(MockWorker):
    active = 0
    peak = 0

    async def request(self, payload, timeout):
        cls = ConcurrencyProbe
        cls.active += 1
        cls.peak = max(cls.peak, cls.active)
        await asyncio.sleep(0.05)
        cls.active -= 1
        return await super().request(payload, timeout)


async def test_jobs_are_serialized():
    e = _engine(Factory(ConcurrencyProbe))
    await asyncio.gather(*(_synth(e, f"job {i}") for i in range(4)))
    assert ConcurrencyProbe.peak == 1 and e.status()["jobs_done"] == 4
    await e.shutdown()


async def test_explicit_load_reports_cold_then_warm():
    e = _engine(Factory())
    assert await e.load() is not None
    assert await e.load() is None
    status = e.status()
    assert status["state"] == "ready" and status["cold_load_secs"] is not None
    await e.shutdown()
