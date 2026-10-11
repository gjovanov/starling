"""podcast-tts — render a `<speaker>: <text>` script to a WAV.

    podcast-tts script.txt -o out.wav [--seed 42] [--cfg-scale 1.3] [--diffusion-steps 10]
    podcast-tts script.txt -o out.wav --server http://127.0.0.1:8093
    podcast-tts --list-voices

Default: renders in THIS process (loads the model, renders, exits — no
server needed; the GPU is free again when the command returns). With
--server it posts to a running podcast-server instead, sharing its warm
model. Either way it prints one JSON summary line on stdout.

Exit codes: 0 ok · 1 runtime failure (GPU busy, model/server error) · 2 bad script.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import disclosure
from .config import settings
from .gpu import gpu_free_gib
from .script import ScriptError, check_generation, parse_script, plan
from .voices import load_voices
from .wav import encode_wav


def _render_local(p, registry, out: Path, seed: int, cfg: float, steps: int) -> dict:
    if settings.min_free_vram_gib > 0:
        free = gpu_free_gib()
        if free is not None and free < settings.min_free_vram_gib:
            raise RuntimeError(f"insufficient free GPU memory ({free:.1f} GiB free, need "
                               f"{settings.min_free_vram_gib:.1f}; PODCAST_MIN_FREE_VRAM_GIB=0 skips)")
    t0 = time.monotonic()
    from .synth import SAMPLE_RATE, Renderer, load_voice

    renderer = Renderer(settings.model_id, settings.model_revision, settings.device, settings.dtype)
    cold = time.monotonic() - t0
    res = renderer.render(
        [(role, turn.text) for role, turn in zip(p.roles, p.turns)],
        [load_voice(registry[name].path) for name in p.speakers],
        seed=seed, cfg_scale=cfg, diffusion_steps=steps, max_new_tokens=p.max_new_tokens,
    )
    label = f"{settings.model_id}@{settings.model_revision[:8]}"
    out.write_bytes(encode_wav(res.audio, SAMPLE_RATE, software=disclosure.SOFTWARE,
                               comment=disclosure.comment(label, p.speakers)))
    secs = res.audio.size / SAMPLE_RATE
    return {
        "out": str(out),
        "sample_rate": SAMPLE_RATE,
        "duration_secs": round(secs, 3),
        "generation_secs": round(res.gen_secs, 2),
        "rtf": round(res.gen_secs / secs, 3) if secs else None,
        "cold_load_secs": round(cold, 2),
        "peak_vram_gib": round(res.peak_vram_gib, 2),
        "truncated": not res.finished,
        "seed": seed, "cfg_scale": cfg, "diffusion_steps": steps,
        "speakers": list(p.speakers), "turns": len(p.turns), "model": label,
    }


def _render_remote(server: str, p, out: Path, seed: int, cfg: float, steps: int) -> dict:
    body = json.dumps({
        "turns": [{"speaker": t.speaker, "text": t.text} for t in p.turns],
        "seed": seed, "cfg_scale": cfg, "diffusion_steps": steps,
    }).encode()
    req = urllib.request.Request(server.rstrip("/") + "/api/tts/podcast", data=body,
                                 headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=settings.synth_timeout_secs + 600) as resp:
            wav, h = resp.read(), resp.headers
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"server answered {exc.code}: {exc.read().decode(errors='replace')}") from exc
    out.write_bytes(wav)
    secs = float(h["X-Podcast-Duration-Secs"])
    gen = float(h["X-Podcast-Generation-Secs"])
    return {
        "out": str(out),
        "sample_rate": int(h["X-Podcast-Sample-Rate"]),
        "duration_secs": secs,
        "generation_secs": gen,
        "rtf": round(gen / secs, 3) if secs else None,
        "cold_load_secs": float(h["X-Podcast-Cold-Load-Secs"]) if h.get("X-Podcast-Cold-Load-Secs") else None,
        "request_secs": round(time.monotonic() - t0, 2),
        "peak_vram_gib": float(h["X-Podcast-Peak-Vram-Gib"]),
        "truncated": h["X-Podcast-Truncated"] == "true",
        "seed": int(h["X-Podcast-Seed"]), "cfg_scale": float(h["X-Podcast-Cfg-Scale"]),
        "diffusion_steps": int(h["X-Podcast-Diffusion-Steps"]),
        "speakers": h["X-Podcast-Speakers"].split(","), "turns": len(p.turns),
        "model": h["X-Podcast-Model"],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="podcast-tts", description=__doc__.split("\n\n")[0])
    ap.add_argument("script", nargs="?", help="script file, one '<speaker>: <text>' per line ('-' = stdin)")
    ap.add_argument("-o", "--out", help="output WAV path (24 kHz mono 16-bit)")
    ap.add_argument("--seed", type=int, default=settings.default_seed)
    ap.add_argument("--cfg-scale", type=float, default=settings.default_cfg_scale)
    ap.add_argument("--diffusion-steps", type=int, default=settings.default_diffusion_steps)
    ap.add_argument("--server", help="render on a running podcast-server, e.g. http://127.0.0.1:8093")
    ap.add_argument("--list-voices", action="store_true", help="print the available voices and exit")
    args = ap.parse_args(argv)

    registry = load_voices(settings.voices_dir)
    if args.list_voices:
        print(json.dumps([v.describe() for v in registry.values()], indent=2))
        return 0
    if not args.script or not args.out:
        ap.error("SCRIPT and -o/--out are required")

    try:
        source = sys.stdin.read() if args.script == "-" else Path(args.script).read_text(encoding="utf-8")
        check_generation(args.seed, args.cfg_scale, args.diffusion_steps)
        p = plan(parse_script(source), set(registry), settings)
    except (ScriptError, OSError) as exc:
        print(f"podcast-tts: {exc}", file=sys.stderr)
        return 2

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        if args.server:
            summary = _render_remote(args.server, p, out, args.seed, args.cfg_scale, args.diffusion_steps)
        else:
            summary = _render_local(p, registry, out, args.seed, args.cfg_scale, args.diffusion_steps)
    except Exception as exc:  # noqa: BLE001 — CLI boundary
        print(f"podcast-tts: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if summary["truncated"]:
        print("podcast-tts: WARNING: render hit the max_new_tokens guard — audio is truncated",
              file=sys.stderr)
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
