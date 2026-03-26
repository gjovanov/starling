# Starling

Real-time ASR server powered by Voxtral-Mini-4B-Realtime via vLLM (GPU only).

## Architecture
- FastAPI + aiortc (WebRTC) + websockets (vLLM client)
- vLLM serves Voxtral-Mini-4B-Realtime-2602 on port 8001 (GPU, BF16, ~9GB VRAM)
- Starling runs on port 8090, proxies audio to vLLM via WebSocket
- Frontend is a pure pass-through display (same UI as parakeet-rs)

## Setup & Run
```bash
./init.sh           # Creates venv, installs vLLM + deps, downloads model
./start-vllm.sh     # Terminal 1: vLLM GPU server (port 8001)
./start.sh          # Terminal 2: Starling server (port 8090)
```

## Benchmark
```bash
python3 scripts/benchmark_voxtral.py --duration 300
```

## Transcription Modes
- **speedy** — Low-latency, pause-based word confirmation
- **growing_segments** — Word-by-word PARTIAL updates toward FINAL sentences
- **pause_segmented** — Segment audio by pauses, transcribe each chunk once

## Config
All env vars prefixed `VOXTRAL_` (see `.env.example`):
- `VOXTRAL_VLLM_URL` — vLLM WebSocket URL (default: ws://localhost:8001/v1/realtime)
- `VOXTRAL_PORT` — Server port (default: 8090)
- `VOXTRAL_MEDIA_DIR` — Media directory (default: ./media)
- `VOXTRAL_FRONTEND_PATH` — Frontend directory (default: ./frontend)

## Project Structure
```
voxtral_server/
  main.py                     # FastAPI app entry point
  config.py                   # Settings (VOXTRAL_ prefix env vars)
  models.py                   # Pydantic API models
  state.py                    # AppState (sessions, broadcast)
  api/                        # REST endpoints
  ws/handler.py               # WebSocket + WebRTC signaling
  transcription/
    vllm_client.py            # vLLM /v1/realtime WebSocket client
    session_runner.py          # FFmpeg + vLLM + subtitle orchestrator
  audio/
    ffmpeg_source.py           # FFmpeg -> PCM 16kHz
    webrtc_track.py            # aiortc AudioStreamTrack
  media/manager.py             # Media file listing/upload
frontend/                      # Web UI (shared API contract with parakeet-rs)
scripts/
  benchmark_voxtral.py         # Quality benchmark
```

## Key Implementation Notes
- vLLM requires `--enforce-eager` flag (torch.compile has FakeTensorMode bug)
- vLLM requires `session.update` message before accepting audio (OpenAI Realtime API)
- Background WebSocket reader task (not polling) to avoid dropping deltas
- Sentence boundary splitting with regex to avoid splitting on "19. November"
- Session runner waits for `client_ready` event before starting FFmpeg
- CPU mode NOT viable: encoder takes 2-3s per second of audio on CPU
