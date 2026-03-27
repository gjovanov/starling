# Burn Server

Real-time ASR server powered by [Voxtral-Mini-4B-Realtime](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602) using the [Burn](https://burn.dev/) ML framework with WebGPU/Vulkan/Metal backend.

Unlike vllm-server, burn-server runs the model directly in the same process (no separate model server). Supports both Q4 quantized inference (~700 MB VRAM) and full BF16 (~9 GB VRAM). Can also run entirely in the browser via WASM + WebGPU.

## Features

- **Q4 inference**: Fused dequant+matmul WGSL shaders, ~700 MB VRAM, faster than real-time on modern GPUs
- **BF16 inference**: Full precision, ~9 GB VRAM, highest accuracy
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

## Benchmarks

| Mode | RTF | Tokens/s | VRAM |
|------|-----|----------|------|
| Q4 native | ~0.4 | ~19 tok/s | ~700 MB |
| BF16 native | ~1.5 | ~5 tok/s | ~9 GB |
| Q4 WASM | ~14 | ~1 tok/s | browser GPU |

(Reference benchmarks from voxtral-mini-realtime-rs on NVIDIA DGX Spark)
