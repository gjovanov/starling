# vLLM Server

Real-time ASR + TTS server powered by Mistral's Voxtral models via [vLLM](https://github.com/vllm-project/vllm) and [vllm-omni](https://github.com/vllm-project/vllm-omni):

- **ASR** (always on): [Voxtral-Mini-4B-Realtime](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602)
- **TTS** (optional `--tts` flag): [Voxtral-4B-TTS-2603](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603)

For ASR: streams audio from the browser via WebRTC (or from media files via FFmpeg), sends it to a vLLM GPU backend, delivers live subtitles over WebSocket.

For TTS: synthesises speech with 20 built-in voices or a user-uploaded reference voice (zero-shot cloning); streams chunked WAV back to the browser for progressive playback.

## Requirements

- **GPU**: NVIDIA with >= 16 GB VRAM. ASR alone uses ~12 GB; TTS alone uses ~12 GB. **Both at once does NOT fit on a 24 GB GPU** — see [TTS coexistence](docs/tts_spike.md). The on-demand lifecycle (Phase 6) auto-swaps them.
- **CUDA**: 12.x toolkit
- **Python**: 3.10+
- **FFmpeg**: for media file processing AND for re-encoding TTS reference audio
- **System**: Linux (tested on Ubuntu 22.04, WSL2)

## Quick Start

```bash
# 1. Install system prerequisites (if needed)
sudo ./prerequisites.sh

# 2a. ASR only (default — lighter venv, ~9 GB model)
./init.sh
./start-vllm.sh        # Terminal 1: ASR GPU server (port 8001)
./start.sh             # Terminal 2: Web server (port 8090)

# 2b. ASR + TTS (adds vllm-omni + Voxtral-4B-TTS, ~3 GB deps + 8 GB model)
./init.sh --tts        # Idempotent; safe on existing setup
./start-vllm.sh        # Terminal 1: ASR (port 8001)
./start.sh             # Terminal 2: Web server (port 8090)
                       # The TTS server (port 8002) auto-spawns on first
                       # /api/tts/* request and unloads after 10 min idle.
                       # If you prefer manual control, run ./start-vllm-tts.sh.

# 3. Open http://localhost:8090
```

## Configuration

Copy `.env.example` to `.env` and adjust. ASR-side knobs:

| Variable | Default | Description |
|----------|---------|-------------|
| `VOXTRAL_VLLM_URL` | `ws://localhost:8001/v1/realtime` | vLLM WebSocket URL |
| `VOXTRAL_PORT` | `8090` | Server port |
| `VOXTRAL_MEDIA_DIR` | `../../media` | Shared media directory |
| `VOXTRAL_FRONTEND_PATH` | `../../frontend` | Shared frontend |
| `VOXTRAL_PUBLIC_IP` | auto-detected | Public IP for WebRTC |
| `VOXTRAL_TURN_SERVER` | | TURN server for NAT traversal |
| `VOXTRAL_VLLM_GPU_UTIL` | `0.45` | ASR vLLM `--gpu-memory-utilization` (lowered from 0.90 to share the GPU with TTS — raise to 0.90 if TTS is disabled) |
| `VOXTRAL_VLLM_MAX_MODEL_LEN` | `4096` | ASR vLLM `--max-model-len` (lowered to fit at the smaller KV-cache budget; ASR sessions rotate context every ~100 s anyway) |

TTS-side knobs (only used when `--tts` was selected):

| Variable | Default | Description |
|----------|---------|-------------|
| `VOXTRAL_TTS_VLLM_URL` | `http://127.0.0.1:8002/v1` | vllm-omni HTTP base URL |
| `VOXTRAL_TTS_MODEL_PATH` | `../../models/cache/tts` | TTS model dir |
| `VOXTRAL_TTS_OUTPUT_DIR` | `./tts_output` | Where save-mode WAVs land |
| `VOXTRAL_TTS_VOICE_REFS_DIR` | `./voice_refs` | Local cache of uploaded clones |
| `VOXTRAL_TTS_MAX_CHARS` | `20000` | Per-request input cap (~20 min of speech) |
| `VOXTRAL_TTS_DEFAULT_VOICE` | `casual_male` | One of the 20 built-in voice ids |
| `VOXTRAL_TTS_AUTOSTART` | `true` | Lazy-spawn the TTS subprocess on first use |
| `VOXTRAL_TTS_IDLE_UNLOAD_SECS` | `600` | Unload TTS after this many seconds idle (`0` = never) |
| `VOXTRAL_TTS_BOOT_TIMEOUT_SECS` | `180` | Max wall-clock for a TTS warm-up |
| `VOXTRAL_TTS_MIN_FREE_VRAM_GIB` | `12` | Refuse to start TTS if free VRAM is below this |
| `VOXTRAL_TTS_REQUIRE_PERMISSION` | `true` | Server-side gate on the voice-cloning consent checkbox |

## Transcription Modes

| Mode | Behavior |
|------|----------|
| **Speedy** | Low-latency streaming with pause-based word confirmation |
| **Growing Segments** | Word-by-word PARTIAL updates building toward FINAL sentences |
| **Pause-Segmented** | Segment audio by acoustic pauses, transcribe each chunk once |

## Architecture

```
                                          ┌── vLLM (port 8001)  ── Voxtral-Mini-4B (ASR, BF16)
Browser ──WebRTC/HTTP──> Starling Server ─┤
   ^                          (port 8090) └── vllm-omni (port 8002, optional)
   │                                          ── Voxtral-4B-TTS (BF16)
   └──WebSocket / chunked WAV ── Live subtitles + synthesised audio
```

The ASR vLLM is started by `./start-vllm.sh` and stays running.
The TTS vllm-omni is started by `./start-vllm-tts.sh` (or auto-spawned by the FastAPI server when `VOXTRAL_TTS_AUTOSTART=true`) and unloads after `VOXTRAL_TTS_IDLE_UNLOAD_SECS` of inactivity.

## Text-to-speech

The TTS tab in the web UI talks to a small REST surface that proxies to vllm-omni:

| Endpoint | Purpose |
|----------|---------|
| `GET /api/tts/voices` | Built-in (20) + user-uploaded (cloned) voices, with a `kind: builtin\|cloned` discriminator |
| `GET /api/tts/config` | Output dir, max chars, default voice, sample rate |
| `GET /api/tts/status` | Lifecycle snapshot — `idle / starting / ready / stopping / blocked` + boot ETA |
| `POST /api/tts/synthesize` | `{text, voice, voice_ref_id?, save, save_filename?, overwrite?}` — returns either a streaming `audio/wav` (save=false) or a JSON envelope with the saved path (save=true) |
| `GET /api/tts/output` | List saved files |
| `GET /api/tts/output/{name}` | Download a saved file |
| `DELETE /api/tts/output/{name}` | Remove a saved file |
| `POST /api/tts/voices/upload` | Multipart: `audio_sample`, `name`, `ref_text`, `permission_confirmed` — register a cloned voice |
| `DELETE /api/tts/voices/{ref_id}` | Remove a cloned voice |

### Built-in voices (20)

`casual_female`, `casual_male`, `cheerful_female`, `neutral_female`, `neutral_male`,
`de_female`, `de_male`, `fr_female`, `fr_male`, `es_female`, `es_male`,
`it_female`, `it_male`, `nl_female`, `nl_male`, `pt_female`, `pt_male`,
`ar_male`, `hi_female`, `hi_male`.

(Source-of-truth: `models/cache/tts/params.json → multimodal.audio_tokenizer_args.voice`.)

### Streaming play vs save-to-disk

- **Play in browser (`save=false`)** — the server proxies upstream PCM, prepends a streaming WAV header (`0xFFFFFFFF` size placeholders), returns chunked `audio/wav`. The browser-side `TtsPlayer` (Web Audio API) starts playback on the first chunk (TTFB ≈ 50 ms in tests) and schedules `AudioBufferSourceNode`s for sample-accurate stitching. Pause / Resume / Stop buttons + a pulsing live-dot indicator. Speed control (0.85× – 1.5×) via `playbackRate`. The player also runs **per-block loudness normalisation** because Voxtral itself emits long sentences at much lower amplitude than short ones (~30 dB delta in measurements). See `docs/tts_spike.md` Phase 8 for the full audio-quality pipeline (byte-alignment carry, no-skip scheduling, ~500 ms chunk aggregation, normaliser, soft-clip).
- **Save to server (`save=true`)** — the server fully synthesises, wraps in a real (non-streaming) WAV with proper sizes, writes to `VOXTRAL_TTS_OUTPUT_DIR/<filename>`. Filename is sanitised against path-traversal (`[A-Za-z0-9._-]{1,128}` + `.wav` suffix); auto-generated when omitted (`tts_<voice>_<utc-timestamp>.wav`). Save-mode does NOT run the client-side normaliser; if you want a level-matched WAV for export, do save-mode + post-process with `ffmpeg -af loudnorm` or similar.

### Long-form

Inputs over a single sentence are split server-side on punctuation boundaries (handles `Dr.`, `19. November`, `3.14`, ellipses, German question-mark variants). Each sentence becomes one upstream synth call; the output stream concatenates them under a single WAV header. Cap defaults to **20 000 chars** (~20 minutes of speech) — raise/lower with `VOXTRAL_TTS_MAX_CHARS`.

### Voice cloning

Upload a 5–30 s reference WAV/MP3/FLAC + the exact transcript; the server re-encodes via `ffmpeg -ar 24000 -ac 1`, stores the canonical bytes plus a sidecar JSON, forwards the registration to upstream's `/v1/audio/voices`. The next synth with `voice_ref_id=<id>` triggers a lifecycle reload into `--task-type Base` (~75 s on first cold-start, then warm).

A consent checkbox is mandatory before upload (`permission_confirmed: true` — server-side enforced). Every upload + delete is recorded in an append-only `voice_refs/_audit.log` (audit log records id, name, timestamp, and the permission flag — never the audio bytes or transcript). **Voice cloning carries real legal risk; document your usage policy.**

### Engine lifecycle (when `--tts` is enabled)

The TTS subprocess is owned by an async state machine in `voxtral_server/tts/lifecycle.py`:

```
  idle ──ensure_started()──► starting ──/v1/models healthy──► ready
   ▲                                                            │
   │                                                            ▼
   └─────────── stopping ◄── idle-timer (>= idle_unload_secs) ◄──┘
```

A pre-check on each cold-start verifies the GPU has at least `VOXTRAL_TTS_MIN_FREE_VRAM_GIB` free; otherwise the request returns **HTTP 503** with `blocked by ASR session`. The frontend renders a status badge that polls `/api/tts/status` every 5 s.

### TTS deep-dive

Full design notes, VRAM math, the GPU-coexistence finding, the upstream `vllm-omni 0.18` quirks (broken `wav` streaming, terminal-chunk-drop tolerance, `--task-type` switch cost), and the per-phase test counts live in [docs/tts_spike.md](docs/tts_spike.md).

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
  init.sh                       # Setup script — accepts --tts, --no-model, --dry-run
  start-vllm.sh                 # ASR GPU server (port 8001)
  start-vllm-tts.sh             # TTS GPU server (port 8002, --tts only)
  start.sh                      # FastAPI web server (port 8090)
  configs/
    voxtral_tts.yaml            # vllm-omni stage-config override (lowered util/mml)
  scripts/
    check_tts_install.py        # Verifies vllm + vllm-omni + torch range after install
    spike_tts.py                # Standalone smoke-test against the running TTS server
  docs/
    tts_spike.md                # TTS Phase 0–7 design memo
  voxtral_server/
    main.py                     # FastAPI app entry point + TTS shutdown hook
    config.py                   # Settings (VOXTRAL_ prefixed env vars)
    models.py                   # Pydantic models (API contract)
    state.py                    # AppState (sessions, broadcast)
    api/
      tts_routes.py             # GET /api/tts/* + POST /api/tts/synthesize + voices upload/delete
      …                         # ASR session/media/model routes
    ws/handler.py               # WebSocket + WebRTC signaling
    tts/
      voices.py                 # Canonical 20-voice catalog
      client.py                 # httpx wrapper around vllm-omni /v1/audio/*
      storage.py                # Path-traversal-safe save-to-disk
      wav.py                    # Streaming WAV header (0xFFFFFFFF placeholders)
      text.py                   # Sentence splitter for long-form input
      lifecycle.py              # Async state machine — auto-spawn / idle-unload / task-type switch
      refs.py                   # Voice-cloning ref storage + audit log
    transcription/
      vllm_client.py            # vLLM /v1/realtime WebSocket client
      session_runner.py         # FFmpeg + vLLM + subtitle orchestrator
    audio/
      ffmpeg_source.py          # FFmpeg -> PCM 16kHz
      webrtc_track.py           # aiortc AudioStreamTrack
    media/manager.py            # Media file listing/upload
  tests/
    test_tts_*.py               # 7 TTS test files — see test counts in docs/tts_spike.md
    scripts/test_init_tts_flag.sh  # Bash test for the installer's --tts flag ordering
```
