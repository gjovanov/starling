#!/usr/bin/env python3
"""Dump golden-reference tensors from `FlowMatchingAudioTransformer`.

Drives the upstream FMA class offline (no vLLM runtime, no HTTP) with
a fixed RNG seed and synthetic `llm_hidden` inputs. Captures the boundary
tensors needed to bit-exact-regression-test the Rust port:

    inputs:
      - llm_hidden_input          [1, 3072]   our synthetic conditioning
      - x_0                       [1, 36]     RNG noise (reproducible from seed)

    intermediate (per Euler step, 7 steps total):
      - v_t_steps                 [7, 1, 36]  velocity AFTER CFG mix
      - sampled_steps             [7, 1, 36]  x_t AFTER step

    outputs:
      - semantic_logit            [1, 8320]   pre-mask
      - semantic_code             [1, 1]      argmax post-mask
      - audio_codes_output        [1, 37]     [semantic, *acoustic]
      - timesteps                 [8]         linspace 0..1 (constant; for sanity)

    config:
      - seed, dtype, device

Each input fixture (zeros / ones_small / alternating) goes to its own
.npz under `apps/burn-server/test_data/tts_golden/`. The npz format is
chosen because it's well-supported by every Rust ndarray-style crate
and human-inspectable via numpy.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

# Allow running directly from the venv site-packages without the parent
# voxtral_server importable.
sys.path.insert(0, "/home/gjovanov/gjovanov/starling/apps/vllm-server/.venv/lib/python3.12/site-packages")

from vllm_omni.model_executor.models.voxtral_tts.voxtral_tts_audio_generation import (  # noqa: E402
    AudioSpecialTokens,
    FlowMatchingAudioTransformer,
)


def load_acoustic_transformer_weights(model: torch.nn.Module, safetensors_path: str) -> int:
    """Copy all `acoustic_transformer.*` tensors from safetensors into `model`.

    Safetensors keys are flat (`acoustic_transformer.layers.0.attention.wq.weight`).
    Our FMA module's state_dict is nested without that prefix, so we strip it.
    Returns the number of tensors loaded.
    """
    state = model.state_dict()
    loaded = 0
    skipped = []
    with safe_open(safetensors_path, framework="pt") as f:
        for k in f.keys():
            if not k.startswith("acoustic_transformer."):
                continue
            local = k[len("acoustic_transformer.") :]
            if local not in state:
                skipped.append(local)
                continue
            t = f.get_tensor(k).to(state[local].dtype).to(state[local].device)
            if t.shape != state[local].shape:
                raise RuntimeError(f"shape mismatch {local}: ckpt {t.shape} vs model {state[local].shape}")
            state[local].copy_(t)
            loaded += 1
    if skipped:
        print(f"  WARNING: {len(skipped)} unmatched ckpt keys, e.g. {skipped[:3]}")
    missing = [k for k in state if k not in [k for k in state if k in dict(model.named_parameters())]]
    return loaded


def patched_decode_one_frame(fma, store):
    """Wrap `fma.decode_one_frame` to dump per-step intermediates into `store`."""
    original = fma.decode_one_frame  # noqa: F841

    def hook(semantic_code: torch.Tensor, llm_hidden: torch.Tensor) -> torch.Tensor:
        B = semantic_code.shape[0]
        should_decode = semantic_code != fma._end_audio_token_id

        x_0 = torch.randn(B, fma.model_args.n_acoustic_codebook).to(dtype=llm_hidden.dtype, device=llm_hidden.device)
        x_0 = fma._noise_scale * x_0
        store["x_0"] = x_0.detach().cpu().float().numpy()

        timesteps = fma._timesteps.to(dtype=llm_hidden.dtype)
        store["timesteps"] = timesteps.detach().cpu().float().numpy()
        llm_hidden_zero = torch.zeros_like(llm_hidden)

        sampled = x_0
        v_t_steps = []
        sampled_steps = []
        for i in range(len(timesteps) - 1):
            t = timesteps[i]
            dt = timesteps[i + 1] - timesteps[i]

            t_emb = fma.time_embedding(t.view(-1, 1).repeat(B, 1)).to(llm_hidden.dtype)
            x_batched = torch.cat([sampled, sampled], dim=0)
            llm_batched = torch.cat([llm_hidden, llm_hidden_zero], dim=0)
            t_emb_batched = torch.cat([t_emb, t_emb], dim=0)

            v_all = fma._predict_velocity(x_t=x_batched, llm_output=llm_batched, t_emb=t_emb_batched)
            v_t, uncond_v_t = v_all[:B], v_all[B:]
            v_t = fma._cfg_alpha * v_t + (1 - fma._cfg_alpha) * uncond_v_t
            sampled = sampled + v_t * dt

            v_t_steps.append(v_t.detach().cpu().float().numpy())
            sampled_steps.append(sampled.detach().cpu().float().numpy())

        store["v_t_steps"] = np.stack(v_t_steps)
        store["sampled_steps"] = np.stack(sampled_steps)

        sampled = torch.clamp(sampled, -1, 1)
        scaled_x = ((sampled + 1) / 2) * (fma.acoustic_embeddings_levels - 1)
        output_codes = scaled_x.round().long()
        output_codes[~should_decode] = fma._empty_audio_token_id
        return output_codes + len(AudioSpecialTokens.all_special_tokens())

    fma.decode_one_frame = hook


def patched_forward(fma, store):
    """Wrap `fma.forward` to also capture semantic_logit + semantic_code."""
    from vllm_omni.model_executor.models.voxtral_tts.voxtral_tts_audio_generation import AudioSpecialTokens

    def hook(llm_hidden: torch.Tensor) -> torch.Tensor:
        semantic_logit = fma.semantic_codebook_output(llm_hidden).float()
        store["semantic_logit_pre_mask"] = semantic_logit.detach().cpu().float().numpy()

        semantic_logit = semantic_logit.clone()
        semantic_logit[:, fma._empty_audio_token_id] = -float("inf")
        semantic_logit[:, (len(AudioSpecialTokens.all_special_tokens()) + fma.model_args.semantic_codebook_size) :] = -float("inf")
        store["semantic_logit_post_mask"] = semantic_logit.detach().cpu().float().numpy()

        semantic_code = semantic_logit.argmax(dim=-1, keepdim=True)
        store["semantic_code"] = semantic_code.detach().cpu().long().numpy()

        acoustic_codes = fma.decode_one_frame(semantic_code.squeeze(1), llm_hidden)

        audio_codes = torch.cat([semantic_code, acoustic_codes], dim=1)
        return audio_codes

    fma.forward = hook


def run_capture(name: str, llm_hidden: torch.Tensor, fma, seed: int, output_dir: Path, dtype_name: str):
    store: dict[str, np.ndarray] = {}
    patched_decode_one_frame(fma, store)
    patched_forward(fma, store)

    torch.manual_seed(seed)
    with torch.no_grad():
        audio_codes = fma.forward(llm_hidden)

    store["llm_hidden_input"] = llm_hidden.detach().cpu().float().numpy()
    store["audio_codes_output"] = audio_codes.detach().cpu().long().numpy()
    store["seed"] = np.array(seed, dtype=np.int64)
    store["dtype"] = np.array(dtype_name)

    out_path = output_dir / f"fma_{name}_{dtype_name}.npz"
    np.savez_compressed(out_path, **store)
    print(f"  → {out_path}  ({len(store)} entries)")
    print(f"    audio_codes shape={audio_codes.shape} first8={audio_codes.flatten()[:8].tolist()}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--params", default="/home/gjovanov/gjovanov/starling/models/cache/tts/params.json")
    p.add_argument("--safetensors", default="/home/gjovanov/gjovanov/starling/models/cache/tts/consolidated.safetensors")
    p.add_argument("--output-dir", default="/home/gjovanov/gjovanov/starling/apps/burn-server/test_data/tts_golden")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    with open(args.params) as f:
        cfg = json.load(f)

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]

    print(f"Building FMA on {args.device} with {args.dtype}...")
    fma = FlowMatchingAudioTransformer(cfg["multimodal"]["audio_model_args"])
    fma = fma.to(dtype=dtype, device=args.device).eval()

    print("Loading acoustic_transformer.* weights from consolidated.safetensors...")
    n = load_acoustic_transformer_weights(fma, args.safetensors)
    print(f"  loaded {n} tensors")

    fixtures = [
        ("zeros", torch.zeros(1, 3072, dtype=dtype, device=args.device)),
        ("ones_small", torch.full((1, 3072), 0.01, dtype=dtype, device=args.device)),
        (
            "alternating",
            torch.tensor(
                [0.05 * (1 if i % 2 == 0 else -1) for i in range(3072)],
                dtype=dtype,
                device=args.device,
            ).view(1, 3072),
        ),
    ]

    for name, x in fixtures:
        print(f"\n[{name}]")
        run_capture(name, x, fma, args.seed, Path(args.output_dir), args.dtype)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
