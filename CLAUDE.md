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
    burn-server/                     # App 2: Rust/Burn (Q4+BF16, GPU + WASM browser)
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

### burn-server (Rust/Burn + WebGPU) — PLANNED
- **Port**: 8091
- **Quantization**: Q4_0 (~700 MB VRAM) and BF16 (~9 GB VRAM)
- **Architecture**: Axum HTTP server + Burn ML framework + wgpu backend
- Runs model directly on GPU (no separate model server process)
- WASM mode: entire model runs in browser via WebGPU
- Q4 uses fused dequant+matmul WGSL shaders (weights never materialized to f32)

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
- `STARLING_MODELS_DIR` — Shared model cache (default: ./models/cache)

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
