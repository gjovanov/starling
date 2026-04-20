# Starling Voxtral — Session Handover

**Last session:** 2026-04-20
**Status:** Real-time CPU inference near-achieved. Quality/speed tradeoff documented.

---

## TL;DR

**GPU (candle-native-flash):** 0.2× realtime (5× faster than RT), correct transcription.
**CPU (candle-cpu-ggml):** 1.68× realtime (slightly slower than RT), correct transcription.

Both engines use **sequential autoregressive decode** + **periodic KV cache resets** (vllm-style).

Earlier batched-decode experiments (commits 9decf61, earlier state) produced "Zum Zum Wir Wir" duplication because all positions in a batch shared the same prev_token. Fixed in commits **31b448e** (CPU) and **310fd81** (GPU).

**llama.cpp port:** 28 commits on local fork. End-to-end works, produces German text. Encoder precision gap (0.959 vs candle's BF16) is inherent to ggml's F32 compute — unsolvable without native BF16 kernels. Not blocking anything.

---

## Current State (key commits)

| Commit | Change |
|--------|--------|
| `310fd81` | **sequential decode in GPU engine** — correct quality, 0.2× RT |
| `31b448e` | **sequential decode in CPU engine** — correct quality, 1.68× RT |
| `e84006c` | **periodic KV cache resets** — encoder@150, decoder@300 (bounds compute) |
| `9decf61` | batched decode for streaming commits (reverted in both engines) |
| `f8b8e40` | incremental mel+conv caching (huge CPU win) |
| `67fe06a` | zero-copy input for ggml matmul |

---

## How to Build, Start, and Test CPU Streaming via Frontend

### Prerequisites

1. **Rust toolchain** (stable)
2. **Native CPU features** — Ryzen 9 9955HX3D or similar AVX-512+VNNI+BF16 CPU
3. **llama.cpp built statically** (for ggml_matmul FFI):
   ```bash
   cd ~/gjovanov/llama.cpp
   mkdir -p build-cpu && cd build-cpu
   cmake .. -DGGML_CUDA=OFF -DGGML_CPU=ON -DGGML_AVX512=ON \
     -DGGML_AVX512_VNNI=ON -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release \
     -DBUILD_SHARED_LIBS=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
   make -j$(nproc) ggml
   ```
4. **Models** (~3 GB): `./models/download.sh` (downloads both BF16 and Q4 from Hugging Face)

### 1. Configure .env for CPU

```bash
cd apps/burn-server
cat > .env <<'EOF'
BURN_PORT=8091
BURN_QUANT=q4
BURN_BACKEND=candle-cpu
STARLING_MODELS_DIR=../../models/cache
BURN_FRONTEND_PATH=../../frontend
BURN_MEDIA_DIR=../../media

# WebRTC (adjust to your TURN)
TURN_SERVER=turn:94.130.141.98:3478
TURN_SHARED_SECRET=<your-secret>
FORCE_RELAY=true
EOF
```

### 2. Build

The `init.sh` reads `BURN_BACKEND=candle-cpu` from `.env` and builds with the right feature flags automatically:

```bash
./init.sh
```

This runs:
```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release \
  --bin burn-server --features candle-cpu-ggml
```

Output: `../../target/release/burn-server` (~80MB).

### 3. Start the Server

```bash
./start.sh
```

The server listens on **port 8091**. On first start it:
- Loads Q4 GGUF (~4s)
- Initializes ggml AVX-512 backend
- Serves REST API + WebSocket + static frontend

### 4. Test via Frontend (WebRTC)

Open **http://localhost:8091** in Chrome/Firefox.

**Test flow:**
1. Click **"Start Recording"** → browser asks for microphone permission
2. WebRTC connects to burn-server via TURN (if FORCE_RELAY=true)
3. Audio flows: browser → TURN → FFmpeg → PCM 16kHz → candle-cpu-ggml engine
4. Engine processes in 0.5s commits, each returning transcribed text delta
5. WebSocket pushes live subtitles to the browser

**Expected behavior on CPU:**
- First commit ~5s (model loading + initial encoder)
- Steady-state commits: 700-850ms for 500ms audio = **1.68× realtime** (subtitles lag slightly behind audio)
- Transcription quality: correct German text with occasional word substitutions
- After ~50s audio: periodic KV resets kick in (visible as brief 2s pause in subtitles on reset)

### 5. Test via Benchmark (no WebRTC)

```bash
# Build benchmark binary (separate from server)
RUSTFLAGS="-C target-cpu=native" cargo build --release \
  --features candle-cpu-ggml --bin benchmark -p burn-server

# Run 60s streaming benchmark (simulates WebRTC 0.5s commits)
GGML_THREADS=16 /home/gjovanov/gjovanov/starling/target/release/benchmark \
  --backend candle-cpu-engine \
  --audio ../../media/broadcast_1.wav \
  --models-dir ../../models/cache \
  --duration 60

# With profiling (shows enc/dec breakdown per commit)
CANDLE_PROFILE=1 GGML_THREADS=16 /home/gjovanov/gjovanov/starling/target/release/benchmark \
  --backend candle-cpu-engine \
  --audio ../../media/broadcast_1.wav \
  --models-dir ../../models/cache \
  --duration 20
```

### 6. Test via Media Files (no microphone needed)

```bash
# Server must be running
curl -X POST http://localhost:8091/api/sessions \
  -H 'Content-Type: application/json' \
  -d '{"media_file": "broadcast_1.wav", "mode": "speedy"}'

# Or upload via frontend: select from "Media Files" dropdown → play
```

---

## Env Vars (tuning)

| Variable | Default | Purpose |
|----------|---------|---------|
| `GGML_THREADS` | 16 | ggml matmul thread count |
| `VOXTRAL_ENC_RESET` | 150 | Encoder KV cache reset threshold (positions) |
| `VOXTRAL_DEC_RESET` | 300 | Decoder KV cache reset threshold (positions) |
| `CANDLE_PROFILE` | unset | Enable per-commit encoder/decoder/lm_head timing |
| `VOXTRAL_BATCH` | 1 | Batch size for transcribe_streaming (GPU benchmark only) |
| `BURN_BACKEND` | wgpu | `candle-cpu` for CPU streaming, `candle-native-flash` for GPU |

---

## Known Issues & Open Work

### Quality: mid-audio word errors
- Sequential decode produces "Team drin" vs reference "Thema Ortszentren", "Utey" vs "Rubay"
- Root cause: Q4 quantization loses precision on some tokens
- Could be improved with Q4_K_M (K-quants) or Q5 weights (larger model)

### Speed: 1.68× realtime (not quite real-time)
- Bottleneck: `decoder_forward` = 90ms per token on CPU (memory-bandwidth-bound, Q4 weight loading)
- Per commit: 140ms encoder + 6×90ms decoder + 6×8ms lm_head = ~720-850ms for 500ms audio
- To reach <500ms/commit, need decoder to run at ~60ms/token

### Possible speed optimizations (not yet tried):
1. **Speculative decoding** — batch draft + sequential verify, typical 2-3× speedup if many drafts accepted
2. **Q4_K_M weights** — better cache locality than Q4_0
3. **Layer culling** — skip 4-8 decoder layers (quality tradeoff)
4. **Hybrid CPU encoder + iGPU/dGPU decoder** — encoder is small (~140ms), decoder dominates
5. **Token parallelism** — run 2-3 sessions in parallel on separate cores

### llama.cpp port status (local fork, not blocking)
- Repo: `~/gjovanov/llama.cpp`
- Produces German text end-to-end
- 4% encoder correlation gap vs candle (ggml F32 vs BF16 arithmetic difference)
- Not worth more investment — burn-server candle path is the production path

---

## File Pointers

**Engine code (CPU streaming):**
- `apps/burn-server/src/inference/candle_cpu/engine.rs` — commit() path with periodic resets
- `apps/burn-server/src/inference/candle_cpu/model.rs` — VoxtralModel + transcribe_streaming
- `libs/ggml-matmul/` — FFI wrapper for ggml Q4 matmul

**Server:**
- `apps/burn-server/src/main.rs` — loads `CandleCpuEngine` when `BURN_BACKEND=candle-cpu`
- `apps/burn-server/src/server/ws.rs` — WebSocket handler (receives audio chunks, returns text)
- `apps/burn-server/src/transcription/session.rs` — FFmpeg → inference → subtitles

**Frontend:**
- `frontend/index.html` + `frontend/app.js` — WebRTC client, subtitle display

**Memory (survives across sessions):**
- `~/.claude/projects/-home-gjovanov-gjovanov-starling/memory/MEMORY.md` — Index
- `~/.claude/projects/-home-gjovanov-gjovanov-starling/memory/project_cpu_inference_findings.md` — CPU detail

---

## Next Session Starting Points

### If user wants **production CPU real-time**:
1. Look at speculative decoding (batched draft → sequential verify)
2. Profile decoder_forward to find hot path (expect matmul dominates)
3. Try Q4_K_M weights (need to reconvert model)

### If user wants **better quality**:
1. Investigate Q4_K_M vs Q4_0 WER difference
2. Or bump to Q5/Q6 (larger but still under 5GB)
3. Or use BF16 weights on GPU (already works, 0.3× RT)

### If user wants to **continue llama.cpp port**:
1. Branch at `~/gjovanov/llama.cpp` has 28 commits
2. Remaining: fix encoder BF16 precision gap (requires ggml kernel changes, likely blocked)
3. Worth submitting as PR to issue #20914 for community to improve

### If user wants to **ship what's there**:
1. Sequential mode in engine.rs is production-ready at quality-priority
2. Add UI hint: "CPU mode is ~1.7× slower than real-time — subtitles lag ~500ms"
3. Offer GPU mode as "real-time" and CPU as "offline/background"
