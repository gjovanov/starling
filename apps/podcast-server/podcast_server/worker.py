"""Model workers: where synthesis actually runs.

The API process never imports torch. `ProcessWorker` hosts the model in a
child process (multiprocessing "spawn"), so unloading means the child
exits and every byte of its VRAM — CUDA context included — goes back to
the other starling apps. `MockWorker` is the deterministic, torch-free
stand-in for tests and GPU-less demos.

Wire protocol (dicts over a multiprocessing Pipe):
    child  → parent  {"type": "ready", import_secs, load_secs, resident_vram_gib}
    parent → child   {"type": "synth", turns: [[role, text]], voices: [path],
                      seed, cfg_scale, diffusion_steps, max_new_tokens}
    child  → parent  {"type": "result", pcm: float32 bytes, sample_rate,
                      gen_secs, peak_vram_gib, audio_tokens, finished, segments}
    child  → parent  {"type": "error", error: str}
    parent → child   {"type": "stop"}
"""

from __future__ import annotations

import asyncio
import math
import multiprocessing as mp
import os
import signal
import time
from typing import Any, Protocol

import numpy as np


class WorkerError(RuntimeError):
    """The worker failed (load error, synth error, crash)."""


class WorkerTimeout(WorkerError):
    """The worker did not answer in time."""


class Worker(Protocol):
    @property
    def pid(self) -> int | None: ...

    def alive(self) -> bool: ...

    async def start(self, timeout: float) -> dict: ...

    async def request(self, payload: dict, timeout: float) -> dict: ...

    async def stop(self) -> None: ...


# ── child side ─────────────────────────────────────────────────────────

def _die_with_parent() -> None:
    """Linux: SIGTERM this process when the parent dies (best effort), so
    a SIGKILLed server never strands a GPU-holding orphan."""
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(1, signal.SIGTERM)          # PR_SET_PDEATHSIG
    except Exception:  # noqa: BLE001 — non-Linux / no libc: rely on pipe EOF
        pass


def worker_main(conn, cfg: dict) -> None:
    _die_with_parent()
    t0 = time.monotonic()
    try:
        from .synth import Renderer, load_voice

        import_secs = time.monotonic() - t0
        renderer = Renderer(cfg["model_id"], cfg["model_revision"], cfg["device"], cfg["dtype"])
    except Exception as exc:  # noqa: BLE001 — reported to the parent
        conn.send({"type": "error", "error": f"model load failed: {type(exc).__name__}: {exc}"})
        return
    conn.send({
        "type": "ready",
        "import_secs": round(import_secs, 2),
        "load_secs": round(renderer.load_secs, 2),
        "resident_vram_gib": round(renderer.resident_vram_gib, 2),
    })
    voice_cache: dict[str, np.ndarray] = {}
    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            return                                 # parent went away
        if msg.get("type") == "stop":
            return
        if msg.get("type") != "synth":
            conn.send({"type": "error", "error": f"unknown message {msg.get('type')!r}"})
            continue
        try:
            arrays = []
            for path in msg["voices"]:
                if path not in voice_cache:
                    voice_cache[path] = load_voice(path)
                arrays.append(voice_cache[path])
            res = renderer.render(
                [tuple(t) for t in msg["turns"]], arrays,
                seed=msg["seed"], cfg_scale=msg["cfg_scale"],
                diffusion_steps=msg["diffusion_steps"], max_new_tokens=msg["max_new_tokens"],
            )
            conn.send({
                "type": "result",
                "id": msg.get("id"),
                "pcm": res.audio.astype(np.float32).tobytes(),
                "sample_rate": 24000,
                "gen_secs": res.gen_secs,
                "peak_vram_gib": res.peak_vram_gib,
                "audio_tokens": res.audio_tokens,
                "finished": res.finished,
                "segments": res.segments,
            })
        except Exception as exc:  # noqa: BLE001 — reported, worker stays up
            conn.send({"type": "error", "id": msg.get("id"),
                       "error": f"{type(exc).__name__}: {exc}"})


# ── parent side ────────────────────────────────────────────────────────

class ProcessWorker:
    """Parent-side handle on the model child process."""

    def __init__(self, cfg: dict, target: Any = None) -> None:
        self._cfg = cfg
        self._target = target or worker_main     # tests inject a fake child main
        self._proc: Any = None
        self._conn: Any = None
        self._seq = 0

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc is not None else None

    def alive(self) -> bool:
        return self._proc is not None and self._proc.is_alive()

    async def start(self, timeout: float) -> dict:
        ctx = mp.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe()
        self._proc = ctx.Process(target=self._target, args=(child_conn, self._cfg),
                                 name="podcast-worker", daemon=True)
        self._proc.start()
        child_conn.close()
        self._conn = parent_conn
        try:
            msg = await self._recv(timeout)
        except WorkerError:
            await self.stop()
            raise
        if msg.get("type") != "ready":
            await self.stop()
            raise WorkerError(msg.get("error", f"unexpected message {msg!r}"))
        return msg

    async def request(self, payload: dict, timeout: float) -> dict:
        if not self.alive():
            raise WorkerError("worker is not running")
        self._seq += 1
        self._conn.send({**payload, "id": self._seq})
        deadline = time.monotonic() + timeout
        while True:
            msg = await self._recv(max(0.0, deadline - time.monotonic()))
            if msg.get("id") == self._seq:
                break                              # anything else is a stale reply
        if msg.get("type") == "error":
            raise WorkerError(msg["error"])
        return msg

    async def _recv(self, timeout: float) -> dict:
        loop = asyncio.get_running_loop()
        ready = await loop.run_in_executor(None, self._conn.poll, timeout)
        if not ready:
            raise WorkerTimeout(f"worker did not answer within {timeout:.0f}s")
        try:
            return await loop.run_in_executor(None, self._conn.recv)
        except (EOFError, OSError) as exc:
            code = self._proc.exitcode if self._proc is not None else None
            raise WorkerError(f"worker exited (code={code})") from exc

    async def stop(self, grace: float = 10.0) -> None:
        """Ask the child to exit; SIGTERM after `grace` s, SIGKILL after 10 more."""
        proc, conn = self._proc, self._conn
        self._proc = self._conn = None
        if proc is None:
            return
        loop = asyncio.get_running_loop()
        if proc.is_alive() and conn is not None:
            try:
                conn.send({"type": "stop"})
            except (OSError, ValueError):
                pass
            await loop.run_in_executor(None, proc.join, grace)
        if proc.is_alive():
            proc.terminate()
            await loop.run_in_executor(None, proc.join, 10.0)
        if proc.is_alive():
            proc.kill()
            await loop.run_in_executor(None, proc.join, 5.0)
        if conn is not None:
            conn.close()


class MockWorker:
    """Deterministic, torch-free stand-in: one sine tone per turn.

    Pitch encodes the role (220 Hz · 1.25^role), length the text (60 ms per
    char, min 0.5 s), with 0.25 s of silence between turns — tests can
    assert on duration and content without a GPU.
    """

    SAMPLE_RATE = 24000

    def __init__(self, delay_secs: float = 0.0) -> None:
        self._delay = delay_secs
        self._running = False
        self.starts = 0

    @property
    def pid(self) -> int | None:
        return os.getpid() if self._running else None

    def alive(self) -> bool:
        return self._running

    async def start(self, timeout: float) -> dict:
        if self._delay:
            await asyncio.sleep(self._delay)
        self._running = True
        self.starts += 1
        return {"type": "ready", "import_secs": 0.0, "load_secs": 0.0, "resident_vram_gib": 0.0}

    async def request(self, payload: dict, timeout: float) -> dict:
        if not self._running:
            raise WorkerError("worker is not running")
        if self._delay:
            await asyncio.sleep(self._delay)
        sr = self.SAMPLE_RATE
        gap = np.zeros(int(0.25 * sr), dtype=np.float32)
        parts: list[np.ndarray] = []
        for i, (role, text) in enumerate(payload["turns"]):
            n = int(max(0.5, 0.06 * len(text)) * sr)
            t = np.arange(n, dtype=np.float32) / sr
            parts.append((0.2 * np.sin(2 * math.pi * 220.0 * 1.25 ** role * t)).astype(np.float32))
            if i < len(payload["turns"]) - 1:
                parts.append(gap)
        audio = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
        return {
            "type": "result",
            "pcm": audio.tobytes(),
            "sample_rate": sr,
            "gen_secs": self._delay,
            "peak_vram_gib": 0.0,
            "audio_tokens": int(math.ceil(audio.size / 3200)),
            "finished": True,
            "segments": [[0, int(audio.size)]],
        }

    async def stop(self, grace: float = 10.0) -> None:
        self._running = False
