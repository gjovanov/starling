# Starling

Multi-app monorepo for real-time ASR powered by Voxtral-Mini-4B-Realtime.

## Monorepo Structure
```
starling/
  frontend/                          # Shared web UI (served by all apps)
  media/                             # Shared audio files
  models/                            # Shared model weights
    download.sh                      # Universal model downloader (BF16 + Q4)
    cache/bf16/                      # SafeTensors (~9 GB) — vllm-server & burn-server BF16
    cache/q4/                        # GGUF Q4_0 (~2.5 GB) — burn-server Q4 & WASM
    cache/q4/chunks/                 # 64 MB shards for browser ArrayBuffer limit
    cache/tokenizer/                 # Tekken tokenizer (shared)
  scripts/
    benchmark_voxtral.py             # Quality benchmark
  apps/
    vllm-server/                     # App 1: Python/vLLM (BF16, GPU only)
    burn-server/                     # App 2: Rust/Candle+Burn (BF16+Q4, GPU + WASM)
```

## Apps

### vllm-server (Python/FastAPI + vLLM)
- **Port**: 8090 (server) + 8001 (vLLM GPU backend)
- **Quantization**: BF16 only (~9 GB VRAM)
- **Architecture**: FastAPI + aiortc (WebRTC) + websockets (vLLM client)
- vLLM serves Voxtral via OpenAI Realtime API on port 8001
- Starling proxies audio to vLLM via WebSocket, returns live subtitles
- Frontend is a pure pass-through display

#### Setup & Run
```bash
cd apps/vllm-server
./init.sh           # Creates venv, installs vLLM + deps, downloads model
./start-vllm.sh     # Terminal 1: vLLM GPU server (port 8001)
./start.sh           # Terminal 2: Starling server (port 8090)
```

#### Key Implementation Notes
- vLLM requires `--enforce-eager` flag (torch.compile has FakeTensorMode bug)
- vLLM requires `session.update` message before accepting audio (OpenAI Realtime API)
- Background WebSocket reader task (not polling) to avoid dropping deltas
- Sentence boundary splitting with regex to avoid splitting on "19. November"
- Session runner waits for `client_ready` event before starting FFmpeg
- CPU mode NOT viable: encoder takes 2-3s per second of audio on CPU
- Context rotation every 200 commits (~100s) to prevent KV cache overflow

#### vllm-server Structure
```
apps/vllm-server/
  voxtral_server/
    main.py                     # FastAPI app entry point
    config.py                   # Settings (VOXTRAL_ prefix env vars)
    models.py                   # Pydantic API models
    state.py                    # AppState (sessions, broadcast)
    api/                        # REST endpoints
    ws/handler.py               # WebSocket + WebRTC signaling
    transcription/
      vllm_client.py            # vLLM /v1/realtime WebSocket client
      session_runner.py         # FFmpeg + vLLM + subtitle orchestrator
    audio/
      ffmpeg_source.py          # FFmpeg -> PCM 16kHz
      webrtc_track.py           # aiortc AudioStreamTrack
    media/manager.py            # Media file listing/upload
```

### burn-server (Rust/Candle-Native + Burn)
- **Port**: 8091
- **Backends**: candle-native (BF16 CUDA), Burn/wgpu (Q4), Burn/CUDA (BF16)
- **Architecture**: Axum HTTP server + pure candle CUDA inference + WebRTC
- Runs model directly on GPU (no separate model server process)
- Real-time streaming: ~300ms per 0.5s audio commit

#### candle-native Backend (Primary — BF16 CUDA, candle 0.9)
Pure candle implementation matching [voxtral.c](https://github.com/antirez/voxtral.c) architecture. Produces identical transcription output.

**Key architecture details:**
- **Interleaved RoPE** (GPT-J style): pairs `(x[0],x[1]), (x[2],x[3])` — NOT split-half NeoX
- **Silence padding**: left=32×1280 + right=17×1280 zeros around audio before mel
- **Streaming prefix**: BOS(1) + STREAMING_PAD(32)×38 = 39 tokens with audio at prefix positions
- **Autoregressive decode**: prev_token fed back as text embedding
- **Encoder**: CausalConv → 32 layers (interleaved RoPE, sliding window 750) → 4x reshape → adapter
- **Decoder**: 26 layers (GQA 32h/8kv, interleaved RoPE, ADA FFN, sliding window 8192)
- **Incremental streaming**: encoder KV cache persists across commits, only new conv frames processed
- **Boundary safety**: 8-frame margin at conv edge — boundary mel frames may shift with new audio
- **No right-padding on intermediate commits** — only final flush gets right-pad
- **Context rotation**: decoder KV caches reset every ~1250 positions (~100s) for constant decode speed

#### candle-native-flash Backend (Fastest — BF16 CUDA, candle 0.10 + FlashAttention v2)
Same architecture as candle-native but with fused CUDA kernels for 1.75× faster decode:
- **candle-core 0.10.2** + **candle-flash-attn 0.10.2** (FlashAttention v2 with native GQA)
- **Fused RmsNorm**: `candle_nn::ops::rms_norm` — 1 CUDA kernel replaces 7 ops (53 calls/step)
- **Fused interleaved RoPE**: `candle_nn::rotary_emb::rope_i` — 1 kernel replaces ~36 ops (26 calls/step)
- **Zero-copy RoPE reshape**: for seq=1 decode, reshape instead of transpose+contiguous (no GPU copy)
- **Pre-cast BF16 RoPE tables**: cos/sin cast to BF16 at load time, not per-step
- **flash_attn_varlen decode**: avoids KV cache contiguous copies via squeeze(0) contiguous view
- **Disabled CUDA event tracking**: single-stream inference, no cross-stream sync needed
- **GPU argmax**: transfers 1 u32 instead of 131K floats per decode step

**Performance (candle-native-flash, BF16, RTX 5090, 300s audio):**
| Metric | candle-native (0.9) | candle-native-flash (0.10) | Speedup |
|--------|--------------------|-----------------------------|---------|
| Decode/step | 25.7ms | **14.6ms** | **1.76×** |
| Encode (300s) | 7.0s | **4.6s** | 1.52× |
| Total (300s) | 106.1s | **60.9s** | **1.74×** |
| Text tokens | 692 | 691 | identical |

**Decode step breakdown (14.6ms, profiled):**
- Attention (QKV + RoPE + flash_attn + output proj): ~6ms (41%)
- FFN (gate+up + SiLU + mul + down): ~7ms (45%)
- Norms + residuals + ADA: ~2ms (14%)
- ~74% is candle dispatch overhead (~22μs/op × ~500 ops), ~26% is GPU compute (bandwidth-limited)

**Streaming performance (incremental, per 0.5s commit):**
- ~6 new adapter tokens per commit
- ~200ms total (mel+conv+encoder+decoder) with flash backend
- Text appears every 0.5-1s

#### candle-cpu Backend (CPU-only, Q4 GGUF) — STREAMING with KV resets
Pure CPU inference using Q4 GGUF weights + streaming engine with periodic KV cache resets.
- **candle-cpu-ggml**: ggml graph API with AVX-512 + VNNI + BF16 native matmul

**Prerequisites for candle-cpu-ggml:**
```bash
# Build ggml static libraries from llama.cpp (one-time)
cd ~/gjovanov/llama.cpp
mkdir -p build-cpu && cd build-cpu
cmake .. -DGGML_CUDA=OFF -DGGML_CPU=ON -DGGML_AVX512=ON \
  -DGGML_AVX512_VNNI=ON -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
make -j$(nproc) ggml
```

**Performance (AMD Ryzen 9 9955HX3D, 16c/32t, 96MB L3, DDR5-5600, streaming 0.5s commits, 300s audio):**

Three decode modes available (recommended: speculative):

| Mode | Env | Speed | Text quality |
|------|-----|-------|--------------|
| **Speculative** ⭐ | `VOXTRAL_SPEC=1` | **1.50× RT** | ✅ Full German content |
| Sequential | (default) | 1.81× RT | ✅ Correct |
| Batched | `VOXTRAL_BATCH=6` | 0.86× RT | ❌ Broken ("Jahre Der" only) |

**Speculative decoding** (Leviathan et al. 2023, adapted for Voxtral's dual-stream):
1. DRAFT pass: batched forward, all N positions share `prev_token` (cheap, wrong)
2. TRUNCATE: roll back KV cache via `KVCache::truncate_to`
3. VERIFY pass: batched forward with corrected text embeddings (`text_ids = [prev_token, draft[0], …, draft[N-2]]`)
4. ACCEPT up to first mismatch + sequential fallback for rejected positions

Per-commit cost: 2 batched forwards (~150ms each) + 0-2 sequential fallback steps. Real speedup ~17% over pure sequential because acceptance rate is high for ASR (audio dominates the prediction).

**Key streaming engine fixes (all modes):**
1. **Periodic encoder KV reset** (`VOXTRAL_ENC_RESET=150`): prevents encoder from growing past L3 cache
2. **Periodic decoder KV reset** (`VOXTRAL_DEC_RESET=300`): bounds decoder attention cost
3. **Incremental mel + conv caching**: only new audio processed per commit

**Environment variables:**
- `VOXTRAL_SPEC=1` — enable speculative decoding (RECOMMENDED)
- `VOXTRAL_BATCH=6` — batched (broken quality, only for benchmarking)
- `GGML_THREADS=16` — thread count for ggml matmul (default: 16)
- `VOXTRAL_ENC_RESET=150` — encoder KV cache reset threshold
- `VOXTRAL_DEC_RESET=300` — decoder KV cache reset threshold
- `CANDLE_PROFILE=1` — per-component timing (decoder_forward, lm_head)
- `VOXTRAL_PROMPT_LEN=39` — streaming prefix length (TrevorS uses 38, no quality difference)

#### Setup & Run
```bash
cd apps/burn-server

# Edit .env for your setup:
#   BURN_QUANT=q4                    # Q4 for CPU, bf16 for GPU
#   BURN_BACKEND=candle-cpu          # CPU streaming (or candle-native-flash for GPU)
#   TURN_SERVER=turn:your-turn-server:3478
#   TURN_SHARED_SECRET=your-secret
#   FORCE_RELAY=true

# One-stop build (reads BURN_BACKEND from .env):
./init.sh

# Or build manually:
# GPU BF16:    cargo build --release --features candle-native-flash
# CPU Q4:      RUSTFLAGS="-C target-cpu=native" cargo build --release --features candle-cpu-ggml

# Start the server (port 8091)
./start.sh
```

**For CPU streaming with the frontend:**
```bash
# Set .env: BURN_BACKEND=candle-cpu, BURN_QUANT=q4
./init.sh          # builds with candle-cpu-ggml feature + AVX-512
./start.sh         # starts server on :8091
# Open http://localhost:8091 in browser → record/upload audio via WebRTC
```

#### Benchmark
```bash
# candle-native (0.9, baseline)
cargo build --release --features candle-native --bin benchmark
CANDLE_STREAMING=1 ./target/release/benchmark --backend candle-native --audio ../../media/broadcast_1.wav --models-dir ../../models/cache --duration 300

# candle-native-flash (0.10 + FlashAttention v2, fastest GPU)
cargo build --release --features candle-native-flash --bin benchmark
CANDLE_STREAMING=1 ./target/release/benchmark --backend candle-native-flash --audio ../../media/broadcast_1.wav --models-dir ../../models/cache --duration 300

# candle-cpu-ggml (CPU streaming engine — correct quality at 1.68× RT)
RUSTFLAGS="-C target-cpu=native" cargo build --release --features candle-cpu-ggml --bin benchmark
GGML_THREADS=16 ./target/release/benchmark --backend candle-cpu-engine \
  --audio ../../media/broadcast_1.wav --models-dir ../../models/cache --duration 300

# Batched mode (faster 0.76× RT but has duplicated tokens):
# Note: batched mode requires the older commit path — see engine.rs history.

# Profile CPU streaming breakdown (per-commit enc/dec timing)
CANDLE_PROFILE=1 GGML_THREADS=16 ./target/release/benchmark --backend candle-cpu-engine \
  --audio ../../media/broadcast_1.wav --models-dir ../../models/cache --duration 20

# Profile decode step breakdown (per-section GPU timing)
CANDLE_STREAMING=1 CANDLE_PROFILE=1 ./target/release/benchmark --backend candle-native-flash --audio ../../media/broadcast_1.wav --models-dir ../../models/cache --duration 5

# F32 mode (for debugging, matches voxtral.c exactly)
CANDLE_NATIVE_F32=1 ./target/release/benchmark --backend candle-native --audio /tmp/broadcast_5s.wav --models-dir ../../models/cache
```

#### burn-server Structure
```
apps/burn-server/
  src/
    main.rs                              # Axum server entry point
    config.rs                            # CLI args + env config
    inference/
      mod.rs                             # InferenceEngine/InferenceSession traits
      candle_native/
        mod.rs                           # Module declaration
        model.rs                         # Full Voxtral model (encoder+decoder+transcribe)
        engine.rs                        # Incremental streaming InferenceSession
      candle_native_flash/
        mod.rs                           # Module declaration
        model.rs                         # Fused kernels (RmsNorm, RoPE, FlashAttn v2)
        engine.rs                        # Streaming session (same API as candle_native)
      candle_cpu/
        mod.rs                           # Module declaration
        model.rs                         # CPU Q4 model (ggml graph API or candle QMatMul)
        engine.rs                        # Streaming session for CPU
      bf16/                              # Burn BF16 backend (burn-candle)
      q4/                                # Burn Q4 backend (wgpu)
    transcription/
      session.rs                         # Session runner (FFmpeg → inference → subtitles)
      streaming.rs                       # Sentence splitting
    audio/
      mel.rs                             # Mel spectrogram (Slaney, matching Whisper)
      pad.rs                             # Audio padding (left/right silence)
      ffmpeg.rs                          # FFmpeg PCM extraction
      opus.rs                            # Opus encoding for WebRTC
    server/
      routes.rs                          # REST API + TURN credentials
      ws.rs                              # WebSocket + WebRTC signaling
      state.rs                           # AppState, sessions
    web/                                 # WebRTC utilities
  Cargo.toml                             # Features: candle-native, candle-native-flash, candle-cpu, candle-cpu-ggml, cuda, candle
libs/
  ggml-matmul/                           # FFI wrapper for ggml's AVX-512 Q4 matmul
    csrc/ggml_matmul_wrapper.c           # C wrapper using ggml graph compute API
    src/lib.rs                           # Rust FFI bindings
    build.rs                             # Links pre-built llama.cpp ggml static libs
```

#### Q4 Padding Workaround
The upstream left-pads audio with 32 silence tokens. After mel/conv/reshape, this covers only 16 of 38 decoder prefix positions. BF16 handles this fine, but Q4_0 makes the decoder sensitive to speech content in the prefix. Left padding is increased to 76 tokens → exactly 38 decoder positions of silence, covering the full streaming prefix.

#### WASM Browser Constraints
- 2 GB ArrayBuffer limit → sharded cursor reads
- 4 GB address space → two-phase loading
- 1.5 GiB embedding table → Q4 on GPU + CPU byte lookups
- No sync GPU readback → `into_data_async().await`
- 256 workgroup invocation limit → patched CubeCL

## Shared Infrastructure

### Model Storage
All apps use `STARLING_MODELS_DIR` (default: `./models/cache/`).
- `models/download.sh` downloads both BF16 and Q4 variants
- vllm-server's `start-vllm.sh` auto-detects shared BF16 weights
- burn-server loads from Q4 or BF16 path directly

### Frontend
Shared `frontend/` served by all apps. Same API contract as parakeet-rs:
- `GET /api/models` — Available models
- `GET /api/modes` — Transcription modes
- `GET /api/media` — Media file listing
- `POST /api/media/upload` — File upload
- `POST /api/sessions` — Create session
- `WS /ws/{session_id}` — Subtitle streaming + WebRTC signaling

### Config
All env vars prefixed `VOXTRAL_` for vllm-server, `BURN_` for burn-server.
- `VOXTRAL_VLLM_URL` — vLLM WebSocket URL (default: ws://localhost:8001/v1/realtime)
- `VOXTRAL_PORT` — vllm-server port (default: 8090)
- `BURN_PORT` — burn-server port (default: 8091)
- `BURN_BACKEND` — GPU backend: candle-native, cuda, candle, wgpu (default: wgpu)
- `BURN_QUANT` — Quantization: bf16, q4 (default: q4)
- `STARLING_MODELS_DIR` — Shared model cache (default: ./models/cache)
- `TURN_SERVER` — TURN relay server URL
- `TURN_SHARED_SECRET` — COTURN ephemeral credentials shared secret
- `FORCE_RELAY` — Force TURN relay mode (true/false)

## Transcription Modes
- **speedy** — Low-latency, pause-based word confirmation
- **growing_segments** — Word-by-word PARTIAL updates toward FINAL sentences
- **pause_segmented** — Segment audio by pauses, transcribe each chunk once

## Benchmark
```bash
python3 scripts/benchmark_voxtral.py --duration 300
```

## llama.cpp Voxtral Port (WIP)

Local fork at `~/gjovanov/llama.cpp` (issue: ggml-org/llama.cpp#20914).
Adds `LLM_ARCH_VOXTRAL` with encoder RoPE, causal attention, dual-stream summation.

**Status:** Both models load, encoder produces meaningful embeddings with RoPE + causal mask.
Output contains real words from audio but not yet correct transcription.

**Remaining:** ADA-RmsNorm decoder graph (`src/models/voxtral.cpp`) + autoregressive prev_token feedback.

**Build:** `cd ~/gjovanov/llama.cpp/build-cpu && make -j16 llama-mtmd-cli`
**Test:** `llama-mtmd-cli -m /tmp/voxtral-text-f16.gguf --mmproj <mmproj.gguf> --audio <file.wav> -p "Transcribe this audio." -n 256 --no-warmup`

## Related Projects
- [parakeet-rs](https://github.com/gjovanov/parakeet-rs) — Rust ASR server using ONNX (Parakeet TDT, Canary 1B) and whisper.cpp. Shares frontend and API contract.
- [voxtral-mini-realtime-rs](https://github.com/TrevorS/voxtral-mini-realtime-rs) — Reference Rust implementation of Voxtral inference with Burn/WebGPU. burn-server is derived from this.
- [voxtral.c](https://github.com/antirez/voxtral.c) — Pure C implementation by antirez. candle-native engine architecture matches this reference.
