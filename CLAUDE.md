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

#### candle-native Backend (Primary — BF16 CUDA)
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

**Performance (BF16, RTX 5090):**
| Duration | Encode | Decode/step | Total | Realtime |
|----------|--------|-------------|-------|----------|
| 5s | 0.2s | 38ms | 3.1s | 0.7× |
| 60s | 1.3s | 37ms | 30s | 0.9× |
| 300s | 6.0s | 44ms | 173s | 1.1× |
| 600s (w/ rotation) | 12s | 38ms | 299s | 1.0× |

**Streaming performance (incremental, per 0.5s commit):**
- ~6 new adapter tokens per commit
- ~300ms total (mel+conv+encoder+decoder)
- Text appears every 0.5-1s

#### Setup & Run
```bash
cd apps/burn-server

# Edit .env for your setup:
#   BURN_QUANT=bf16
#   BURN_BACKEND=candle-native
#   TURN_SERVER=turn:your-turn-server:3478
#   TURN_SHARED_SECRET=your-secret
#   FORCE_RELAY=true

# Build (requires CUDA toolkit)
cargo build --release --features candle-native

# Start
./start.sh
```

#### Benchmark
```bash
# Batch transcription
cargo build --release --features candle-native --bin benchmark
./target/release/benchmark --backend candle-native --audio ../../media/broadcast_1.wav --models-dir ../../models/cache --duration 60

# Streaming mode (chunked encoder)
CANDLE_STREAMING=1 ./target/release/benchmark --backend candle-native --audio ../../media/broadcast_1.wav --models-dir ../../models/cache --duration 300

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
  Cargo.toml                             # Features: candle-native, cuda, candle
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

## Related Projects
- [parakeet-rs](https://github.com/gjovanov/parakeet-rs) — Rust ASR server using ONNX (Parakeet TDT, Canary 1B) and whisper.cpp. Shares frontend and API contract.
- [voxtral-mini-realtime-rs](https://github.com/TrevorS/voxtral-mini-realtime-rs) — Reference Rust implementation of Voxtral inference with Burn/WebGPU. burn-server is derived from this.
- [voxtral.c](https://github.com/antirez/voxtral.c) — Pure C implementation by antirez. candle-native engine architecture matches this reference.
