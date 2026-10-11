"""ProcessWorker mechanics against a real spawned child (tests/fake_worker.py)."""

import numpy as np
import pytest

from podcast_server.worker import ProcessWorker, WorkerError, WorkerTimeout
from tests.fake_worker import fake_worker_main


def _payload(text):
    return {"type": "synth", "turns": [[0, text]], "voices": [], "seed": 1,
            "cfg_scale": 1.3, "diffusion_steps": 10, "max_new_tokens": 10}


async def test_start_request_stop():
    w = ProcessWorker({}, target=fake_worker_main)
    info = await w.start(timeout=30)
    assert info["type"] == "ready" and w.alive() and w.pid
    msg = await w.request(_payload("hello"), timeout=10)
    assert np.frombuffer(msg["pcm"], np.float32).size == 2400
    proc = w._proc
    await w.stop()
    assert not proc.is_alive() and w.pid is None


async def test_load_error_is_raised_and_the_child_reaped():
    w = ProcessWorker({"fail_load": True}, target=fake_worker_main)
    with pytest.raises(WorkerError, match="model load failed: fake"):
        await w.start(timeout=30)
    assert not w.alive()


async def test_error_reply_keeps_the_child():
    w = ProcessWorker({}, target=fake_worker_main)
    await w.start(timeout=30)
    with pytest.raises(WorkerError, match="ValueError: fake"):
        await w.request(_payload("error"), timeout=10)
    assert w.alive()
    await w.request(_payload("fine"), timeout=10)
    await w.stop()


async def test_crash_is_detected():
    w = ProcessWorker({}, target=fake_worker_main)
    await w.start(timeout=30)
    with pytest.raises(WorkerError, match="code=3"):
        await w.request(_payload("crash"), timeout=10)
    assert not w.alive()
    await w.stop()


async def test_hang_times_out_and_stop_kills_it():
    w = ProcessWorker({}, target=fake_worker_main)
    await w.start(timeout=30)
    with pytest.raises(WorkerTimeout):
        await w.request(_payload("hang"), timeout=0.5)
    proc = w._proc
    await w.stop(grace=0.5)                 # child is asleep: SIGTERM path
    assert not proc.is_alive()


async def test_stale_reply_is_discarded():
    w = ProcessWorker({}, target=fake_worker_main)
    await w.start(timeout=30)
    msg = await w.request(_payload("stale"), timeout=10)
    assert msg["pcm"] != b"stale" and msg["id"] == w._seq
    await w.stop()
