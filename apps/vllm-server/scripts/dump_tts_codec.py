#!/usr/bin/env python3
"""Dump golden-reference tensors from the Voxtral TTS codec decoder.

Drives `VoxtralTTSAudioTokenizer.decode()` offline (no vLLM runtime, no
HTTP) on synthetic audio-code fixtures and captures:

  inputs:
    - codes_input          [1, 37, T]    audio-code tensor fed to decode
  intermediates:
    - quantizer_emb        [1, K_emb, T] embedding after VQ lookup
    - per_block_<i>_out    varying       output of each decoder_blocks[i]
  outputs:
    - pcm_output           [1, T_pcm]    24 kHz PCM (after squeeze)

Each fixture goes to a separate .npz under
`apps/burn-server/test_data/tts_golden/`.

Notes:
  - The shipped checkpoint contains only decoder + quantizer + token-
    embedding weights. Encoder weights are absent. The decoder path is
    what we need to port; encoder support is out of Phase 2 scope.
  - The script runs on CPU for determinism. The full audio_tokenizer
    fits in <2 GB at BF16.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, "/home/gjovanov/gjovanov/starling/apps/vllm-server/.venv/lib/python3.12/site-packages")

from vllm_omni.model_executor.models.voxtral_tts.voxtral_tts_audio_tokenizer import (  # noqa: E402
    VoxtralTTSAudioTokenizer,
)


def make_mock_vllm_config(params: dict) -> SimpleNamespace:
    """Construct the minimal vllm_config attribute path that
    `VoxtralTTSAudioTokenizer.__init__` reads."""
    audio_config = {
        "codec_args": params["multimodal"]["audio_tokenizer_args"],
        "audio_model_args": params["multimodal"]["audio_model_args"],
    }
    text_config = SimpleNamespace(hidden_size=params["dim"])
    hf_config = SimpleNamespace(audio_config=audio_config, text_config=text_config)
    model_config = SimpleNamespace(hf_config=hf_config)
    return SimpleNamespace(model_config=model_config)


def load_audio_tokenizer_weights(model: torch.nn.Module, safetensors_path: str) -> tuple[int, int]:
    """Load `audio_tokenizer.*` weights into `model` (strips prefix).

    Also handles the special remapping: upstream renames
    `mm_audio_embeddings.audio_codebook_embeddings.embeddings.*`
    to `audio_token_embedding.embeddings.*`. We do the same.

    Returns (loaded_count, skipped_count).
    """
    state = model.state_dict()
    loaded = 0
    skipped: list[str] = []
    encoder_skipped = 0
    with safe_open(safetensors_path, framework="pt") as f:
        for k in f.keys():
            local = None
            if k.startswith("audio_tokenizer."):
                local = k[len("audio_tokenizer.") :]
            elif k.startswith("mm_audio_embeddings.audio_codebook_embeddings."):
                # `audio_token_embedding.embeddings.weight` <-
                # `mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight`
                local = k.replace(
                    "mm_audio_embeddings.audio_codebook_embeddings.",
                    "audio_token_embedding.",
                    1,
                )
            else:
                continue

            if local not in state:
                # quantizer.semantic_codebook.{cluster_usage,embedding_sum} are
                # registered as buffers, not parameters; they're set via
                # setattr in VoxtralTTSAudioTokenizer.load_weight. Mirror that.
                if local == "quantizer.semantic_codebook.cluster_usage":
                    setattr(model.quantizer.semantic_codebook, "cluster_usage", f.get_tensor(k))
                    loaded += 1
                    continue
                if local == "quantizer.semantic_codebook.embedding_sum":
                    setattr(model.quantizer.semantic_codebook, "embedding_sum", f.get_tensor(k))
                    loaded += 1
                    continue
                skipped.append(local)
                continue
            t = f.get_tensor(k).to(state[local].dtype).to(state[local].device)
            if t.shape != state[local].shape:
                raise RuntimeError(f"shape mismatch {local}: ckpt {t.shape} vs model {state[local].shape}")
            state[local].copy_(t)
            loaded += 1

    if skipped:
        print(f"  WARNING: {len(skipped)} unmatched ckpt keys, e.g. {skipped[:3]}")

    # Encoder weights are absent in the shipped checkpoint. Mark zeroed
    # so the model still runs decode-only without errors.
    for n, _ in model.named_parameters():
        if n.startswith("input_proj.") or n.startswith("encoder_blocks."):
            encoder_skipped += 1

    return loaded, encoder_skipped


def attach_block_hooks(tok, store: dict[str, np.ndarray]):
    hooks = []
    # quantizer output
    def quantizer_hook(_m, _i, out):
        store["quantizer_emb"] = out.detach().cpu().float().numpy()

    hooks.append(tok.quantizer.register_forward_hook(quantizer_hook))

    # per decoder block output
    for idx, block in enumerate(tok.decoder_blocks):
        def make(i=idx):
            def h(_m, _i, out):
                # Some blocks return tuples (e.g. Transformer), unwrap:
                t = out[0] if isinstance(out, tuple) else out
                store[f"decoder_block_{i:02d}_out"] = t.detach().cpu().float().numpy()
            return h
        hooks.append(block.register_forward_hook(make()))

    # output_proj final
    hooks.append(
        tok.output_proj.register_forward_hook(
            lambda _m, _i, o: store.__setitem__(
                "output_proj_pre_rearrange", o.detach().cpu().float().numpy()
            )
        )
    )
    return hooks


def run_capture(name: str, codes: torch.Tensor, tok, output_dir: Path, dtype_name: str, dtype: torch.dtype):
    store: dict[str, np.ndarray] = {}
    hooks = attach_block_hooks(tok, store)
    try:
        with torch.no_grad():
            pcm = tok.decode(codes, dtype=dtype)
    finally:
        for h in hooks:
            h.remove()

    store["codes_input"] = codes.detach().cpu().long().numpy()
    store["pcm_output"] = pcm.detach().cpu().float().numpy()
    store["dtype"] = np.array(dtype_name)

    out_path = output_dir / f"codec_{name}_{dtype_name}.npz"
    np.savez_compressed(out_path, **store)
    print(
        f"  → {out_path} ({len(store)} entries) "
        f"pcm shape={tuple(pcm.shape)} range=[{pcm.min().item():.3f},{pcm.max().item():.3f}]"
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--params", default="/home/gjovanov/gjovanov/starling/models/cache/tts/params.json")
    p.add_argument("--safetensors", default="/home/gjovanov/gjovanov/starling/models/cache/tts/consolidated.safetensors")
    p.add_argument("--output-dir", default="/home/gjovanov/gjovanov/starling/apps/burn-server/test_data/tts_golden")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(args.params) as f:
        params = json.load(f)

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    print(f"Building audio_tokenizer on {args.device} with {args.dtype}...")
    cfg = make_mock_vllm_config(params)
    tok = VoxtralTTSAudioTokenizer(vllm_config=cfg)
    tok = tok.to(dtype=dtype, device=args.device).eval()

    print("Loading audio_tokenizer.* weights from consolidated.safetensors...")
    loaded, enc_count = load_audio_tokenizer_weights(tok, args.safetensors)
    print(f"  loaded {loaded} tensors  (encoder still uninitialised: {enc_count} params — decode path only)")

    K = 37  # num_codebooks
    sem_size = params["multimodal"]["audio_model_args"]["semantic_codebook_size"]
    acoust_size = params["multimodal"]["audio_model_args"]["acoustic_codebook_size"]

    torch.manual_seed(args.seed)

    fixtures = []

    # Fixture 1: minimal — single frame of mid-codebook codes
    f1_codes = torch.zeros(1, K, 1, dtype=torch.long, device=args.device)
    f1_codes[0, 0, 0] = sem_size // 2  # semantic
    for i in range(1, K):
        f1_codes[0, i, 0] = acoust_size // 2  # acoustic
    fixtures.append(("single_frame_mid", f1_codes))

    # Fixture 2: 25 frames (~2 s) random-but-valid codes
    f2_codes = torch.zeros(1, K, 25, dtype=torch.long, device=args.device)
    f2_codes[0, 0, :] = torch.randint(0, sem_size, (25,))
    for i in range(1, K):
        f2_codes[0, i, :] = torch.randint(0, acoust_size, (25,))
    fixtures.append(("random_25_frames", f2_codes))

    # Fixture 3: alternating-pattern 10 frames (cheap deterministic check)
    f3_codes = torch.zeros(1, K, 10, dtype=torch.long, device=args.device)
    for t in range(10):
        f3_codes[0, 0, t] = (t * 17) % sem_size  # arbitrary deterministic
        for i in range(1, K):
            f3_codes[0, i, t] = (t + i) % acoust_size
    fixtures.append(("deterministic_10_frames", f3_codes))

    for name, codes in fixtures:
        print(f"\n[{name}] codes shape={tuple(codes.shape)}")
        run_capture(name, codes, tok, Path(args.output_dir), args.dtype, dtype)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
