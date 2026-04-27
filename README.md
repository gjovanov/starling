# Starling

Multi-app monorepo for real-time **speech recognition (ASR)** and **text-to-speech (TTS)** powered by Mistral's Voxtral models:

- ASR: [Voxtral-Mini-4B-Realtime](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602)
- TTS: [Voxtral-4B-TTS-2603](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603) (vllm-server only)

Each ASR app provides a different inference backend for the same model, sharing a common frontend, media library, and model weights.

## Apps

| App | Backend | Quantization | ASR perf | Extras |
|-----|---------|-------------|---------|--------|
| [**vllm-server**](apps/vllm-server/) | Python/vLLM (GPU) | BF16 | 0.2× realtime | **+ TTS tab** (20 voices, voice cloning, long-form) |
| [**burn-server**](apps/burn-server/) | Rust/Candle+ggml | Q4 (CPU) / BF16 (GPU) | **1.68× RT (CPU) / 0.3× RT (GPU)** | ASR only |

## Quick Start

### vllm-server (Python, GPU required)

```bash
cd apps/vllm-server
sudo ./prerequisites.sh    # System deps (Python, CUDA, FFmpeg)

# ASR only (default — lighter venv)
./init.sh                  # Python venv, vLLM, model download
./start-vllm.sh            # Terminal 1: GPU ASR server
./start.sh                 # Terminal 2: Web server on :8090

# Or: ASR + TTS (adds vllm-omni + Voxtral-4B-TTS, ~3 GB extra deps + 8 GB model)
./init.sh --tts            # Idempotent; safe to re-run on an existing setup
./start-vllm.sh            # ASR server (port 8001)
./start-vllm-tts.sh        # TTS server (port 8002) — auto-spawned by the
                           # FastAPI server on first /api/tts/* request, so
                           # this terminal is optional
./start.sh                 # Web server (port 8090) — exposes both ASR + TTS tabs
```

**Note:** ASR and TTS each fit on their own but **not both simultaneously** on a 24 GiB GPU. The TTS lifecycle auto-spawns/unloads the TTS process so a single GPU can host both workflows at different times — see [apps/vllm-server/docs/tts_spike.md](apps/vllm-server/docs/tts_spike.md).

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

### ASR (both apps)
- **13 languages** natively supported by Voxtral-Mini-4B
- **Real-time streaming** via WebRTC audio + WebSocket subtitles
- **3 transcription modes**: speedy, growing segments, pause-segmented
- **Multi-session** support with concurrent transcription
- **Media file** playback with synchronized subtitles
- **System-audio capture** ("Speakers" tab) — transcribe whatever is playing in the browser tab
- **Shared model cache** across all apps (no duplicate downloads)
- **Web frontend** included (no build step)

### TTS (vllm-server only)
- **20 built-in voices** across 9 languages (English, German, French, Spanish, Italian, Dutch, Portuguese, Arabic, Hindi)
- **Progressive playback** via Web Audio API — first sound at TTFB ≈ 50 ms, not after the full file lands
- **Long-form synthesis** — paste up to 20 000 chars; a server-side splitter produces one continuous audio stream
- **Save to disk** with path-traversal-safe filenames + auto-generated timestamps
- **Zero-shot voice cloning** — upload 5–30 s of reference audio + transcript, synthesise in that voice
- **On-demand engine lifecycle** — TTS subprocess auto-starts on first request, unloads after 10 min idle

## Related Projects

- [parakeet-rs](https://github.com/gjovanov/parakeet-rs) - Rust ASR server using ONNX Runtime (Parakeet TDT, Canary 1B) and whisper.cpp. Shares the same frontend and API contract.

## License

MIT
