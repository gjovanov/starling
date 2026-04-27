# Voxtral-TTS golden references

Bit-mostly-exact reference outputs from the upstream `vllm-omni` Python
implementation, captured for regression-testing the Rust port.

**Source scripts** (run from `apps/vllm-server` with the venv activated):
- `scripts/dump_tts_flow_matching.py` — `FlowMatchingAudioTransformer` driven
  offline (no vLLM runtime, no HTTP).
- `scripts/dump_tts_codec.py` — `VoxtralTTSAudioTokenizer.decode()` driven
  offline.
- `scripts/dump_tts_e2e_golden.py` — full-pipeline outputs via the running
  `vllm-omni` HTTP API.

**Re-derive:**
```bash
cd apps/vllm-server
source .venv/bin/activate
python3 scripts/dump_tts_flow_matching.py
python3 scripts/dump_tts_flow_matching.py --dtype float32
python3 scripts/dump_tts_codec.py
python3 scripts/dump_tts_codec.py --dtype float32
# E2E requires vllm-omni running on :8002 (lifecycle manager spawns it
# on the first /api/tts/* hit, or run start-vllm-tts.sh manually).
python3 scripts/dump_tts_e2e_golden.py
```

## Files

### Flow-matching velocity field — `fma_*.npz`

Three input fixtures × two dtypes = 6 files. Each `.npz` contains:
- `llm_hidden_input` `[1, 3072]` — synthetic conditioning we fed in.
- `seed`, `dtype` — for reproducibility.
- `timesteps` `[8]` — `linspace(0, 1, 8)`, constant; sanity-check.
- `x_0` `[1, 36]` — initial RNG noise (deterministic from `seed=42`).
- `semantic_logit_pre_mask` `[1, 8320]` — output of `semantic_codebook_output`.
- `semantic_logit_post_mask` `[1, 8320]` — `EMPTY_AUDIO` and out-of-vocab → -inf.
- `semantic_code` `[1, 1]` — argmax post-mask. `=1` means `[END_AUDIO]`.
- `v_t_steps` `[7, 1, 36]` — CFG-mixed velocity per Euler step.
- `sampled_steps` `[7, 1, 36]` — `x_t` after each step.
- `audio_codes_output` `[1, 37]` — `[semantic_code, *acoustic_codes]` final.

Note the synthetic inputs (`zeros`, `ones_small`, `alternating`) are not
real `llm_hidden` from the AR LLM — they are deterministic patterns to
enable bit-exact unit-testing of the velocity field. End-to-end refs
(below) cover the full pipeline.

### Codec decoder — `codec_*.npz`

Three fixtures × two dtypes = 6 files. Each `.npz` contains:
- `codes_input` `[1, 37, T]` — synthetic audio-code tensor.
- `quantizer_emb` `[1, K_emb, T]` — embedding after VQ lookup.
- `decoder_block_<i>_out` — output of each `decoder_blocks[i]`.
- `output_proj_pre_rearrange` — final `output_proj` conv output before
  the `rearrange` (which produces the 240-sample-per-frame patches).
- `pcm_output` `[1, 1, T*1920]` — final 24 kHz PCM.
- `dtype` — `bfloat16` or `float32`.

Frame counts: 1 / 25 / 10. PCM samples per frame = 1920 (downsample
factor; sampling rate / frame rate = 24000 / 12.5 = 1920).

The `random_25_frames` fixture intentionally produces incoherent audio
(uniform-random codebook indices) — its only job is to exercise the
decoder graph at "long" lengths. The `pcm_output.max()` exceeding 1.0 is
expected for incoherent codes; the upstream warns but does not clamp.

### End-to-end — `e2e_*.wav` + `e2e_*.json`

Six (text, voice) inputs spanning EN/DE/FR and four voices. Each pair:
- `e2e_<name>.wav` — 24 kHz mono PCM, WAV-wrapped, 2.2 s – 15.4 s long.
- `e2e_<name>.json` — request payload + timing + sanity stats (sr,
  duration, peak, rms_dbfs).

End-to-end is **seed-dependent**: upstream's `decode_one_frame` calls
`torch.randn` per frame without a fixed seed, so re-running synthesis
will produce a different waveform (audibly the same speech). The Rust
port should be able to match upstream **only when fed the same noise
samples**. For bit-exact e2e testing, capture the noise as well — the
current `dump_tts_e2e_golden.py` does not. Adding a noise-fixed mode is
deferred to Phase 2-A.3 if needed.

## Total

21 files, ~8.4 MB. Cheap to commit.
