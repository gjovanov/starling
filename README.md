# Starling

Real-time speech recognition server powered by [Voxtral-Mini-4B-Realtime](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602) via [vLLM](https://github.com/vllm-project/vllm).

Starling streams audio from the browser via WebRTC (or from media files via FFmpeg), sends it to a vLLM GPU backend for transcription, and delivers live subtitles over WebSocket.

## Features

- **13 languages** natively supported by Voxtral-Mini-4B
- **Real-time streaming** via WebRTC audio + WebSocket subtitles
- **3 transcription modes**: speedy, growing segments, pause-segmented
- **Multi-session** support with concurrent transcription
- **Media file** playback with synchronized subtitles
- **Web frontend** included (no separate build step)

## Requirements

- **GPU**: NVIDIA with >= 16GB VRAM (model uses ~9GB in BF16)
- **CUDA**: 12.x toolkit
- **Python**: 3.10+
- **FFmpeg**: for media file processing
- **System**: Linux (tested on Ubuntu 22.04)

## Quick Start

### 1. Setup

```bash
./init.sh
```

This creates a Python venv, installs vLLM + dependencies, and downloads the Voxtral model (~8GB).

### 2. Start vLLM (Terminal 1)

```bash
./start-vllm.sh
```

Serves the model on port 8001. Wait for `"Started server process"` before proceeding.

### 3. Start Starling (Terminal 2)

```bash
./start.sh
```

Opens the web UI on port 8090.

### 4. Open Browser

Navigate to `http://localhost:8090`, select a media file, choose a transcription mode, and start.

## Configuration

Copy `.env.example` to `.env` and adjust:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|----------|---------|-------------|
| `VOXTRAL_VLLM_URL` | `ws://localhost:8001/v1/realtime` | vLLM WebSocket URL |
| `VOXTRAL_PORT` | `8090` | Server port |
| `VOXTRAL_MEDIA_DIR` | `./media` | Directory for audio files |
| `VOXTRAL_FRONTEND_PATH` | `./frontend` | Frontend static files |
| `VOXTRAL_PUBLIC_IP` | auto-detected | Public IP for WebRTC |
| `VOXTRAL_TURN_SERVER` | | TURN server for NAT traversal |
| `VOXTRAL_TURN_SHARED_SECRET` | | COTURN ephemeral credentials |
| `VOXTRAL_FORCE_RELAY` | `false` | Force TURN relay mode |

## Transcription Modes

| Mode | Behavior |
|------|----------|
| **Speedy** | Low-latency streaming with pause-based word confirmation |
| **Growing Segments** | Word-by-word PARTIAL updates building toward FINAL sentences |
| **Pause-Segmented** | Segment audio by acoustic pauses, transcribe each chunk once |

## Benchmark

```bash
python3 scripts/benchmark_voxtral.py --duration 300   # 5-min benchmark
```

Results on Austrian German broadcast audio (ORF news, 5 min):
- **WER**: 3.3%
- **CER**: 1.3%
- **Key Phrase Recall**: 77%

## Architecture

```
Browser ──WebRTC──> Starling Server ──WebSocket──> vLLM GPU Server
   ^                     |                              |
   |                     v                              v
   └──WebSocket──── Live Subtitles <──── Voxtral-Mini-4B (BF16)
```

- **FastAPI** + **aiortc** for WebRTC/WebSocket handling
- **vLLM** serves Voxtral-Mini-4B-Realtime via OpenAI Realtime API
- **FFmpeg** decodes media files to PCM 16kHz mono
- Frontend is a pure pass-through display (no client-side ASR logic)

## Project Structure

```
starling/
  voxtral_server/
    main.py                   # FastAPI app entry point
    config.py                 # Settings (VOXTRAL_ prefixed env vars)
    models.py                 # Pydantic models (API contract)
    state.py                  # AppState (sessions, broadcast)
    api/                      # REST endpoints (sessions, media, models)
    ws/handler.py             # WebSocket + WebRTC signaling
    transcription/
      vllm_client.py          # vLLM /v1/realtime WebSocket client
      session_runner.py       # FFmpeg + vLLM + subtitle orchestrator
    audio/
      ffmpeg_source.py        # FFmpeg -> PCM 16kHz
      webrtc_track.py         # aiortc AudioStreamTrack
    media/manager.py          # Media file listing/upload
  frontend/                   # Web UI
  scripts/
    benchmark_voxtral.py      # Quality benchmark
  init.sh                     # Full setup script
  start-vllm.sh               # Start vLLM GPU server
  start.sh                    # Start Starling server
```

## Related Projects

- [parakeet-rs](https://github.com/gjovanov/parakeet-rs) - Rust ASR server using ONNX Runtime (Parakeet TDT, Canary 1B) and whisper.cpp (Whisper). Shares the same frontend and API contract.

## License

MIT
