# vLLM Server

Real-time ASR server powered by [Voxtral-Mini-4B-Realtime](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602) via [vLLM](https://github.com/vllm-project/vllm).

Streams audio from the browser via WebRTC (or from media files via FFmpeg), sends it to a vLLM GPU backend for transcription, and delivers live subtitles over WebSocket.

## Requirements

- **GPU**: NVIDIA with >= 16GB VRAM (model uses ~9GB in BF16)
- **CUDA**: 12.x toolkit
- **Python**: 3.10+
- **FFmpeg**: for media file processing
- **System**: Linux (tested on Ubuntu 22.04)

## Quick Start

```bash
# 1. Install system prerequisites (if needed)
sudo ./prerequisites.sh

# 2. Setup Python venv, install vLLM, download model
./init.sh

# 3. Start vLLM GPU server (Terminal 1)
./start-vllm.sh

# 4. Start Starling server (Terminal 2)
./start.sh

# 5. Open http://localhost:8090
```

## Configuration

Copy `.env.example` to `.env` and adjust:

| Variable | Default | Description |
|----------|---------|-------------|
| `VOXTRAL_VLLM_URL` | `ws://localhost:8001/v1/realtime` | vLLM WebSocket URL |
| `VOXTRAL_PORT` | `8090` | Server port |
| `VOXTRAL_MEDIA_DIR` | `../../media` | Shared media directory |
| `VOXTRAL_FRONTEND_PATH` | `../../frontend` | Shared frontend |
| `VOXTRAL_PUBLIC_IP` | auto-detected | Public IP for WebRTC |
| `VOXTRAL_TURN_SERVER` | | TURN server for NAT traversal |

## Transcription Modes

| Mode | Behavior |
|------|----------|
| **Speedy** | Low-latency streaming with pause-based word confirmation |
| **Growing Segments** | Word-by-word PARTIAL updates building toward FINAL sentences |
| **Pause-Segmented** | Segment audio by acoustic pauses, transcribe each chunk once |

## Architecture

```
Browser ──WebRTC──> Starling Server ──WebSocket──> vLLM GPU Server
   ^                     |                              |
   |                     v                              v
   └──WebSocket──── Live Subtitles <──── Voxtral-Mini-4B (BF16)
```

## Benchmark

```bash
python3 ../../scripts/benchmark_voxtral.py --duration 300
```

Results on Austrian German broadcast audio (ORF news, 5 min):
- **WER**: 3.3%
- **CER**: 1.3%
- **Key Phrase Recall**: 77%

## Project Structure

```
apps/vllm-server/
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
```
