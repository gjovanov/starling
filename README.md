# Starling

Multi-app monorepo for real-time **speech recognition (ASR)**, **text-to-speech (TTS)**, **OCR + document understanding** and **multi-speaker "podcast" TTS**. Four server apps, one shared frontend.

- ASR: [Voxtral-Mini-4B-Realtime](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602)
- TTS: [Voxtral-4B-TTS-2603](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603)
- OCR: [GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) (stage 1) + [Qwen3-8B-FP8](https://huggingface.co/Qwen/Qwen3-8B-FP8) structured extraction (stage 2)
- Podcast TTS: [VibeVoice-1.5B](https://huggingface.co/microsoft/VibeVoice-1.5B) (up to 4 speakers, one continuous render) via Hugging Face transformers

## Servers

| Server | Stack | Serves | Where to run |
|---|---|---|---|
| [**vllm-server**](apps/vllm-server/) | Python · FastAPI · vLLM realtime · aiortc · vllm-omni | ASR + TTS (BF16, GPU) | Reference / production-stable |
| [**burn-server**](apps/burn-server/) | Rust · Axum · candle 0.10 + FlashAttention v2 (GPU) / ggml AVX-512 (CPU) | ASR (BF16 GPU, Q4 CPU, Q4 WASM — partial) + TTS (native Rust port) | Single-binary deployments, CPU laptops, in-browser experiments |
| [**ocr-server**](apps/ocr-server/) | Python · FastAPI · vLLM · glmocr SDK | OCR → Markdown → validated invoice JSON; multi-engine bake-offs (GLM-OCR, tesseract) | Document digitization, engine comparisons on BG/EN financial documents |
| [**podcast-server**](apps/podcast-server/) | Python · FastAPI · transformers (VibeVoice) in a spawned GPU worker | Script of turns → one multi-speaker WAV (API on 127.0.0.1:8093 + `podcast-tts` CLI) | Two-host narration, dialogue voice-overs — read its responsible-use section first |

The speech servers expose the same HTTP surface to a shared frontend in `frontend/`. A given browser session can hit either backend; both implement `POST /api/sessions` (ASR) and `POST /api/tts/synthesize` (TTS) with identical request/response shapes. The frontend reports which backend produced the current audio via an engine-identity badge. ocr-server serves the same frontend and adds the OCR tab (`/api/ocr/*` + live WebSocket job updates).

## ASR — Voxtral-Mini-4B-Realtime

13 languages (English, German, French, Spanish, Russian, Chinese, Japanese, Italian, Portuguese, Dutch, Arabic, Hindi, Korean). Audio in over WebRTC, live captions out over WebSocket. Three transcription modes (speedy, growing-segments, pause-segmented), context rotation every ~100 s to keep KV bounded, regex-driven sentence splitter that handles `Dr.`, `19. November`, `3.14`, `…`.

| Backend | Quant | Hardware | Decode/step | Throughput | Status |
|---|---|---|---|---|---|
| `vllm-server` (vLLM realtime) | BF16 | CUDA | — | ~5× RT on RTX 5090 | Production |
| `burn-server` / `candle-native-flash` | BF16 | CUDA + FlashAttention v2 | **14.6 ms** | 1.74× faster than candle 0.9 | Production |
| `burn-server` / `candle-cpu-ggml` (speculative) | Q4_0 | AVX-512 + VNNI CPU | — | **1.50× RT** on Ryzen 9 laptop | Production |
| `burn-server` / wgpu Q4 (WASM) | Q4_0 | Browser WebGPU | — | English-biased on German due to cumulative fp32 drift across 26 decoder layers — WGSL spec footnote; ggml-WASM port is the path forward | Partial |

The `candle-cpu-ggml` backend is built on top of ggml's AVX-512+VNNI Q4 matmul and a Voxtral-tuned **speculative decoding** scheme (draft → truncate → verify → accept up to first mismatch + sequential fallback). For a deep dive see [`docs/blog/voxtral-asr-burn/`](docs/blog/voxtral-asr-burn/blog-post.md).

## TTS — Voxtral-4B-TTS-2603

20 built-in voices across 9 languages (English, German, French, Spanish, Italian, Dutch, Portuguese, Arabic, Hindi), 24 kHz mono, 12.5 fps, 37 codebooks (1 semantic + 36 acoustic), per-frame autoregressive generation through an AR LLM ↔ flow-matching autoregressor ↔ codec pipeline.

| Backend | Stack | Engine badge | Status |
|---|---|---|---|
| `vllm-server` | vllm-omni HTTP subprocess proxied through FastAPI | `voxtral-4b-tts via vllm-omni` | Production |
| `burn-server` | Native Rust port (candle 0.10) in the Axum process | `voxtral-4b-tts via burn-server-native (cuda+bf16)` | Production (since 2026-05-02) |

Validation: the FMA is **F32 bit-exact** vs vllm-omni across all 3 layers + every Euler step; the codec is single-frame bit-exact through every one of its 8 blocks. Fixtures live in `apps/burn-server/test_data/tts_golden/`.

**Long-form synthesis**: regex sentence splitter (shared with ASR) breaks the input into sentences; each sentence streams independently through `Body::from_stream`. Cap is 20 000 chars or 180 s wall-clock, whichever hits first.

**Voice cloning**: storage works end-to-end (`POST /api/tts/voices/upload`, `_audit.log`, listing, `DELETE`). Synthesis returns `501` because the open-source Voxtral-4B-TTS-2603 checkpoint ships without the codec encoder weights (`audio_tokenizer.input_proj.*`, `audio_tokenizer.encoder_blocks.*`). Affects both servers identically — `cargo run --release --features voxtral-tts --bin dump-tts-weights` reports it.

**Frontend playback** (`frontend/js/modules/tts-player.js`): four layered DSP fixes on top of "just play the bytes" — byte-alignment carry across odd-byte network chunks, no-skip scheduling clamp on `AudioBufferSourceNode.start()`, ~500 ms chunk aggregation, per-block loudness normaliser with `tanh` soft-clip. Voxtral itself emits long sentences ~30 dB quieter than short ones; the normaliser is what makes that listenable. State machine `idle → buffering → playing → paused/ended` exposed via `window.__ttsPlayerState` for Playwright assertions.

**GPU coexistence** (vllm-server): a 24 GiB GPU cannot hold ASR (~9 GB) + TTS (~12 GB) simultaneously. The TTS lifecycle auto-spawns the vllm-omni subprocess on first `/api/tts/*` request and unloads after 10 min idle. `BURN_TTS_IDLE_UNLOAD_SECS` (default 600) controls the same behaviour on burn-server.

## Frontend

Shared HTML/CSS/JS under `frontend/` is served by both servers. Same DOM, same API contract, same caption div, same TTS player. No build step.

- **STT tab** — Sessions list, Create New (model + quant + source + mode + language + diarization + sentence-completion + pause-segmented config + growing-segments config + fab webhook), Media Files (upload + list).
- **TTS tab** — text area (≤ 20 000 chars), voice dropdown (20 builtins + uploaded clones), speed (0.85×–1.5×), play vs save mode, live engine identity badge, in-browser audio player that survives the four DSP quirks above.
- **Audio sources** — Media Files (uploaded WAV/MP3), SRT Live Streams, **Speakers** (browser tab / display capture for real-time transcription of whatever's playing).
- **WebGPU experiment** — Q4 ASR weights load into a wasm-engine pipeline for in-browser inference. Currently produces English-biased output on German source audio due to cumulative fp32 drift; see Post 2 of the blog series for the full bisection.

## Repository layout

```
starling/
├── apps/
│   ├── vllm-server/         # Python · FastAPI · vLLM realtime · vllm-omni
│   │   ├── voxtral_server/  # FastAPI app (sessions, media, TTS routes)
│   │   ├── start-vllm.sh    # ASR vLLM realtime on :8001
│   │   ├── start-vllm-tts.sh # TTS vllm-omni on :8002 (auto-spawned)
│   │   └── start.sh         # FastAPI on :8090
│   ├── burn-server/         # Rust · Axum · candle 0.10
│   │   ├── src/inference/
│   │   │   ├── candle_native_flash/    # GPU BF16, FlashAttn v2, fused RmsNorm/RoPE
│   │   │   ├── candle_cpu/             # CPU Q4 streaming engine, speculative decode
│   │   │   └── voxtral_tts/            # Native Rust TTS (AR LLM + FMA + codec)
│   │   └── start.sh         # Axum on :8091
│   ├── ocr-server/          # Python · FastAPI · OCR + extraction pipeline
│   │   ├── ocr_server/      # App (queue, engines, extraction, routes, WS)
│   │   ├── start-vllm-ocr.sh     # GLM-OCR vLLM on :8003 (auto-spawned)
│   │   ├── start-vllm-extract.sh # Qwen3-8B-FP8 vLLM on :8004 (auto-spawned)
│   │   └── start.sh         # FastAPI on :8092
│   └── podcast-server/      # Python · FastAPI · VibeVoice-1.5B multi-speaker TTS
│       ├── podcast_server/  # App (lifecycle engine, GPU worker process, routes, CLI)
│       ├── voices/          # michael.wav + emma.wav (synthetic Kokoro prompts)
│       └── start.sh         # FastAPI on 127.0.0.1:8093
├── frontend/                # Shared HTML/CSS/JS (no build step)
│   └── js/modules/tts-player.js  # 4-fix audio path
├── media/                   # Shared audio files (uploads land here; media/ocr for documents)
├── models/                  # Shared model weights
│   ├── download.sh          # Universal downloader (BF16 + Q4 + TTS + tokenizer + OCR)
│   └── cache/{bf16,q4,tts,tokenizer,ocr}/
├── results/ocr/             # OCR run artifacts (markdown, JSON, bench CSVs)
├── tools/ocr-bench.py       # Multi-engine OCR bake-off bench (CSV output)
├── libs/                    # Shared Rust crates (voxtral-core, ggml-matmul, wasm-engine)
├── docs/
│   └── blog/                # 4-post technical blog series + demo videos
└── tests/e2e/               # Playwright frontend tests
```

All apps reference shared `frontend/`, `media/`, `models/` at the repo root. Set `STARLING_MODELS_DIR` to override the model cache location.

## Quick start

### vllm-server (Python, GPU)

```bash
cd apps/vllm-server
sudo ./prerequisites.sh        # Python, CUDA, FFmpeg

# ASR-only (~9 GB model, lighter venv)
./init.sh
./start-vllm.sh                # ASR vLLM realtime — port 8001
./start.sh                     # FastAPI web server — port 8090

# ASR + TTS (+ ~8 GB TTS model + vllm-omni)
./init.sh --tts                # Idempotent. Order matters: vllm BEFORE vllm-omni.
./start-vllm.sh                # ASR — port 8001
./start.sh                     # FastAPI — port 8090
#   (the TTS subprocess auto-spawns on first /api/tts/* request)
```

### burn-server (Rust, CPU or GPU)

```bash
cd apps/burn-server
./prerequisites.sh             # Rust toolchain, optional wasm-pack

# Pick a backend in .env:
#   BURN_BACKEND=candle-native-flash  # GPU BF16, FlashAttn v2 (default)
#   BURN_BACKEND=candle-cpu           # CPU Q4 streaming (laptop, no GPU)

./init.sh                      # ASR only
./init.sh --tts                # ASR + TTS in one binary
./start.sh                     # Axum — port 8091
```

For CPU streaming the ggml static libs from llama.cpp are needed; see [`apps/burn-server/README.md`](apps/burn-server/README.md) for the build recipe.

### ocr-server (Python, GPU — mock engine works without)

```bash
cd apps/ocr-server
./prerequisites.sh             # Python 3.10+, GPU check, optional tesseract recipe

./init.sh                      # venv + vLLM + glmocr SDK + models (~11 GB)
./init.sh --no-gpu --no-model  # CI / tests-only setup (mock engine)
./start.sh                     # FastAPI — port 8092
#   GLM-OCR (:8003) and the extraction LLM (:8004) auto-spawn on first use
#   and idle-unload after 10 min. Upload invoices in the OCR tab; runs and
#   extractions stream live over /api/ocr/ws.
```

Bake-off bench across engines:

```bash
./apps/ocr-server/.venv/bin/python tools/ocr-bench.py \
    --dir apps/ocr-server/tests/fixtures --engines glm-ocr,tesseract-bul --extract auto
# → results/ocr/bench-<timestamp>.csv + per-engine summary
```

### podcast-server (Python, GPU — mock engine works without)

```bash
cd apps/podcast-server
./init.sh                      # venv + torch (cu130) + transformers 5.19.0 + VibeVoice-1.5B (~5.4 GB, HF cache)
./start.sh                     # FastAPI — 127.0.0.1:8093 only
#   POST /api/tts/podcast {"turns":[{"speaker":"michael","text":"…"},{"speaker":"emma","text":"…"}]} → audio/wav
#   The GPU worker spawns on first use and exits after 10 min idle (all VRAM returned).
.venv/bin/podcast-tts script.txt -o out.wav   # batch CLI, lines of `<speaker>: <text>`
```

API only — it does not serve the shared frontend. Read the [responsible-use section](apps/podcast-server/README.md#responsible-use-and-licence) before publishing its output.

### Frontend

All servers except podcast-server serve the same `frontend/` automatically — open `http://localhost:8090` (vllm), `http://localhost:8091` (burn) or `http://localhost:8092` (ocr) in Chrome. No npm install, no bundler. Tabs whose backend is absent hide themselves.

## Tests

```bash
# Rust (burn-server) — including bit-exact TTS golden refs
cargo test --release --features voxtral-tts --lib voxtral_tts::

# Python (vllm-server) — 145 pytest cases for TTS, ASR, lifecycle, audit log, refs
cd apps/vllm-server && pytest -xvs

# Python (podcast-server) — 78 pytest cases, no GPU (mock engine + a real spawned fake worker)
cd apps/podcast-server && .venv/bin/pytest

# Playwright (shared frontend, mock-server) — 28 cases for speakers / TTS UI / cloning
cd tests/e2e && bunx playwright test
```

## Blog series

Four deep-dive posts under [`docs/blog/`](docs/blog/) cover each (model × server) cell of the matrix. Each post ships with mermaid sources, pre-rendered PNGs for Medium, and a ≤ 2-minute demo video with audio:

1. [`voxtral-asr-vllm/`](docs/blog/voxtral-asr-vllm/blog-post.md) — vLLM realtime, FastAPI, aiortc plumbing, sentence splitter, context rotation, ASR/TTS coexistence.
2. [`voxtral-asr-burn/`](docs/blog/voxtral-asr-burn/blog-post.md) — three GPU backends, CPU streaming + speculative decoding, the five-month WebGPU Q4 bisection.
3. [`voxtral-tts-vllm/`](docs/blog/voxtral-tts-vllm/blog-post.md) — 20 voices, streaming WAV, lifecycle, the four frontend DSP fixes.
4. [`voxtral-tts-burn/`](docs/blog/voxtral-tts-burn/blog-post.md) — 386 BF16 tensors, bit-exact FMA, native Rust pipeline, the COUNT-offset fix.

Demo recording infrastructure lives under [`docs/blog/_shared/`](docs/blog/_shared/README.md) — one bash wrapper + two Playwright specs drive both ASR and TTS demos against either backend with the right voice/language pre-selected.

## Related projects

- [parakeet-rs](https://github.com/gjovanov/parakeet-rs) — sibling Rust ASR server using ONNX (Parakeet TDT, Canary 1B) and whisper.cpp. Same frontend, same API contract.
- [voxtral-mini-realtime-rs](https://github.com/TrevorS/voxtral-mini-realtime-rs) — reference Rust Voxtral inference with Burn/WebGPU. burn-server's ASR path derives from this.
- [voxtral.c](https://github.com/antirez/voxtral.c) — pure-C reference implementation by antirez. candle-native's architecture matches this byte-for-byte.

## License

MIT
