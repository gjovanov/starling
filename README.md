# Starling

Multi-app monorepo for real-time speech recognition powered by [Voxtral-Mini-4B-Realtime](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602).

Each app provides a different inference backend for the same model, sharing a common frontend, media library, and model weights.

## Apps

| App | Backend | Quantization | Performance | Port |
|-----|---------|-------------|-------------|------|
| [**vllm-server**](apps/vllm-server/) | Python/vLLM (GPU) | BF16 | 0.2× realtime | 8090 |
| [**burn-server**](apps/burn-server/) | Rust/Candle+ggml | Q4 (CPU) / BF16 (GPU) | **1.68× RT (CPU) / 0.3× RT (GPU)** | 8091 |

## Quick Start

### vllm-server (Python, GPU required)

```bash
cd apps/vllm-server
sudo ./prerequisites.sh    # System deps (Python, CUDA, FFmpeg)
./init.sh                  # Python venv, vLLM, model download
./start-vllm.sh            # Terminal 1: GPU model server
./start.sh                 # Terminal 2: Web server on :8090
```

### burn-server (Rust, CPU or GPU)

```bash
cd apps/burn-server
./prerequisites.sh         # Rust toolchain, wasm-pack

# Edit .env to pick backend:
#   BURN_BACKEND=candle-cpu           # CPU streaming (Q4, 1.68× realtime on AVX-512)
#   BURN_BACKEND=candle-native-flash  # GPU BF16 (requires CUDA toolkit)

./init.sh                  # Build + model download (reads BURN_BACKEND)
./start.sh                 # Server on :8091

# For CPU streaming, llama.cpp must be pre-built (see apps/burn-server/README.md)
```

**See [HANDOVER.md](HANDOVER.md) for detailed CPU streaming setup and performance tuning.**

## Shared Resources

```
starling/
  frontend/                # Shared web UI (all apps serve this)
  media/                   # Shared audio files (uploaded via any app)
  models/                  # Shared model weights (BF16 + Q4 GGUF)
    download.sh            # Downloads all model variants
    cache/                 # Downloaded weights (gitignored)
  scripts/                 # Shared benchmarks
  apps/
    vllm-server/           # Python/vLLM app
    burn-server/           # Rust/Burn app (planned)
```

All apps reference the shared `frontend/`, `media/`, and `models/` directories at the repo root. Set `STARLING_MODELS_DIR` to override the model cache location.

## Features

- **13 languages** natively supported by Voxtral-Mini-4B
- **Real-time streaming** via WebRTC audio + WebSocket subtitles
- **3 transcription modes**: speedy, growing segments, pause-segmented
- **Multi-session** support with concurrent transcription
- **Media file** playback with synchronized subtitles
- **Web frontend** included (no build step)
- **Shared model cache** across all apps (no duplicate downloads)

## Related Projects

- [parakeet-rs](https://github.com/gjovanov/parakeet-rs) - Rust ASR server using ONNX Runtime (Parakeet TDT, Canary 1B) and whisper.cpp. Shares the same frontend and API contract.

## License

MIT
