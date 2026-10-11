"""A torch-free stand-in for worker_main, run in a REAL spawned process.

Behaviour is chosen by the first turn's text: "crash" exits the process,
"hang" never answers, "error" replies with an error, "stale" sends a reply
with a wrong id before the real one; anything else returns 0.1 s of silence.
"""

import os
import time

import numpy as np


def fake_worker_main(conn, cfg):
    if cfg.get("fail_load"):
        conn.send({"type": "error", "error": "model load failed: fake"})
        return
    conn.send({"type": "ready", "import_secs": 0.0, "load_secs": 0.0, "resident_vram_gib": 0.0})
    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            return
        if msg["type"] == "stop":
            return
        text = msg["turns"][0][1]
        if text == "crash":
            os._exit(3)
        if text == "hang":
            time.sleep(60)
            continue
        if text == "error":
            conn.send({"type": "error", "id": msg.get("id"), "error": "ValueError: fake"})
            continue
        result = {"type": "result", "pcm": np.zeros(2400, np.float32).tobytes(), "sample_rate": 24000,
                  "gen_secs": 0.01, "peak_vram_gib": 0.0, "audio_tokens": 1, "finished": True,
                  "segments": [[0, 2400]]}
        if text == "stale":
            conn.send({**result, "id": -1, "pcm": b"stale"})
        conn.send({**result, "id": msg.get("id")})
