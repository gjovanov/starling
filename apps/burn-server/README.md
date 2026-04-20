# Burn Server

Real-time ASR server powered by [Voxtral-Mini-4B-Realtime](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602) using the [Burn](https://burn.dev/) ML framework with WebGPU/Vulkan/Metal backend.

Unlike vllm-server, burn-server runs the model directly in the same process (no separate model server). Supports both Q4 quantized inference (~700 MB VRAM) and full BF16 (~9 GB VRAM). Can also run entirely in the browser via WASM + WebGPU.

## Features

- **CPU streaming (candle-cpu-ggml)**: Q4 GGUF + ggml AVX-512, **1.68× realtime** on AMD Ryzen 9 9955HX3D with correct German transcription. Periodic KV cache resets (vllm-style) for stable long-audio processing.
- **GPU BF16 (candle-native-flash)**: FlashAttention v2, ~0.3× realtime on RTX 5090, highest accuracy
- **Q4 GPU inference**: Fused dequant+matmul WGSL shaders, ~700 MB VRAM
- **WASM browser mode**: Entire model runs in browser tab via WebGPU (Q4 only)
- **Same API contract**: Drop-in replacement for vllm-server (shared frontend works with both)
- **Single process**: No separate model server needed
- **13 languages**: Same Voxtral-Mini-4B language support

## Requirements

### Native (GPU)
- **GPU**: Any GPU with Vulkan/Metal/WebGPU support
  - Q4: ~700 MB VRAM (runs on most GPUs)
  - BF16: ~9 GB VRAM (needs RTX 4080+ or equivalent)
- **Rust**: 1.75+
- **FFmpeg**: for media file processing
- **System**: Linux, macOS, or Windows

### Browser (WASM)
- Chrome 113+ or Firefox 117+ with WebGPU support
- ~2.5 GB download for model weights (cached after first load)

## Quick Start

```bash
# 1. Install prerequisites
./prerequisites.sh

# 2. Build binary + download models
./init.sh

# 3. Start server
./start.sh

# 4. Open http://localhost:8091
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `BURN_PORT` | `8091` | Server port |
| `BURN_QUANT` | `q4` | Quantization: `q4` or `bf16` |
| `STARLING_MODELS_DIR` | `../../models/cache` | Shared model cache |
| `BURN_MEDIA_DIR` | `../../media` | Shared media directory |
| `BURN_FRONTEND_PATH` | `../../frontend` | Shared frontend |
| `PUBLIC_IP` | auto-detected | Public IP for WebRTC |
| `TURN_SERVER` | | TURN server for NAT traversal |

## Architecture

```
Browser ──WebRTC──> Burn Server (Axum on :8091)
   ^                     |
   |                     v
   |              ┌─────────────┐
   |              │ Burn/wgpu   │
   |              │ Q4 or BF16  │
   |              │ Voxtral-4B  │
   |              └─────────────┘
   |                     |
   └──WebSocket──── Live Subtitles

OR (WASM browser mode):

Browser ──Microphone──> Web Worker ──> Burn/WASM ──> Subtitles
                         (Q4 model runs entirely in browser)
```

## Q4 Padding Workaround

Q4_0 quantization makes the decoder sensitive to speech content in the streaming prefix. Left-padding is increased from 32 to 76 silence tokens to cover all 38 decoder prefix positions with silence. Without this, audio starting immediately with speech produces all-pad tokens instead of text.

## WASM Build

```bash
./init.sh --wasm                # Build native + WASM
# OR
wasm-pack build --target web    # WASM only
```

The WASM build produces `pkg/` with the JS/WASM bundle. Model weights are loaded as 64 MB chunks to work around the browser's 2 GB ArrayBuffer limit.

## Project Structure

```
apps/burn-server/
  src/
    main.rs               # Axum server entry point
    config.rs             # CLI args + env config
    server/
      state.rs            # AppState, sessions, WebRTC peers
      routes.rs           # REST API (same contract as vllm-server)
      ws.rs               # WebSocket + WebRTC signaling
    audio/
      mel.rs              # Mel spectrogram (128 bins, rustfft)
      resample.rs         # 16kHz resampling (rubato)
      ffmpeg.rs           # FFmpeg subprocess for media files
    inference/
      bf16/               # BF16 path (Burn tensors, SafeTensors)
      q4/                 # Q4 path (GGUF, fused WGSL shaders)
    transcription/
      session.rs          # Session runner (audio → inference → text)
      streaming.rs        # Prefix tokens, greedy decoding
    web/                  # WASM bindings (wasm-bindgen)
```

## Benchmarks (streaming 0.5s commits)

| Mode | Realtime factor | Mean commit | Text quality | Hardware |
|------|-----------------|-------------|--------------|----------|
| **candle-cpu-ggml (sequential)** | **1.68×** | 838ms | Correct German | Ryzen 9 9955HX3D AVX-512 |
| **candle-cpu-ggml (batched)** | **0.76×** | 381ms | Duplicated tokens | Ryzen 9 9955HX3D AVX-512 |
| **candle-native-flash (BF16)** | **0.3×** | ~150ms | Correct | RTX 5090 |
| Q4 WGPU | ~14× | — | Correct | browser WebGPU |

Benchmark:
```bash
# CPU streaming (sequential, correct quality, 1.68× RT)
RUSTFLAGS="-C target-cpu=native" cargo build --release --features candle-cpu-ggml --bin benchmark -p burn-server
GGML_THREADS=16 ../../target/release/benchmark --backend candle-cpu-engine \
  --audio ../../media/broadcast_1.wav --models-dir ../../models/cache --duration 60

# GPU (BF16 flash attention, 0.3× RT)
cargo build --release --features candle-native-flash --bin benchmark -p burn-server
CANDLE_STREAMING=1 ../../target/release/benchmark --backend candle-native-flash \
  --audio ../../media/broadcast_1.wav --models-dir ../../models/cache --duration 60
```

**Performance tuning env vars:**
- `GGML_THREADS=16` — CPU thread count
- `VOXTRAL_ENC_RESET=150` — Encoder KV cache reset threshold
- `VOXTRAL_DEC_RESET=300` — Decoder KV cache reset threshold
- `CANDLE_PROFILE=1` — Per-commit enc/dec/lm_head breakdown
