# Voxtral-4B-TTS Phase 0 Spike

Findings from the spike that informs Phase 1 implementation. Hardware: RTX 5090
Laptop GPU, 24 GB VRAM, WSL2.

## Hosting model — chosen approach

**Run a second `vllm-omni serve` process on port 8002.** The TTS model is exposed
via the OpenAI-compatible `POST /v1/audio/speech` endpoint. Our existing FastAPI
on port 8090 will proxy `POST /api/tts/synthesize → http://127.0.0.1:8002/v1/audio/speech`.

Why:

- The model card documents *only* the `vllm-omni` path. There is no
  `transformers`-based reference implementation.
- `vllm-omni` registers a `voxtral_tts` model class that pulls in both the
  decoder LLM and the acoustic transformer/audio tokenizer. Plain `vllm serve`
  resolves the architecture as `MistralForCausalLM` and crashes on
  `acoustic_transformer` weights (verified — see Boot failure 1 below).
- Two processes give clean fault isolation — an OOM in TTS won't take down ASR.

## Install

```bash
source .venv/bin/activate
pip install vllm-omni            # vllm 0.18.0 must already be installed
```

Notes:

- Install vllm BEFORE vllm-omni (HF discussion #29) — reverse order causes
  undefined-symbol ImportErrors.
- Do NOT pass `--upgrade`. It pulls torch 2.11 which breaks vllm extensions.
- The install pulls heavy deps (gradio, diffusers, librosa, openai-whisper).
  Side-effects on existing deps in the venv:
  - websockets 16.0 → 15.0.1 (still compatible with our aiortc 1.14)
  - pydantic 2.12.5 → 2.12.3, pillow 12.1.1 → 11.3.0, aiofiles 25.1.0 → 24.1.0
  All re-checked: existing imports still succeed.

## Serve command

```bash
vllm-omni serve <models/cache/tts> \
  --port 8002 \
  --gpu-memory-utilization 0.45 \
  --enforce-eager \
  --config-format mistral \
  --load-format mistral \
  --tokenizer-mode mistral
```

The three `mistral` flags are required because the TTS repo ships only
`params.json` (no `config.json`). vllm-omni's `MistralConfigParser` reads
`params.json` and dispatches to its `voxtral_tts` model class via the
`@register_config_parser("mistral")` hook in
`.venv/.../vllm_omni/model_executor/models/voxtral_tts/configuration_voxtral_tts.py`.

`--enforce-eager` mirrors the existing ASR start script — torch.compile has a
FakeTensorMode bug we already documented for the ASR path.

`--gpu-memory-utilization 0.45` is the value picked for the spike. Tune in
Phase 1 once both processes are running together.

## VRAM math (24 GB total, RTX 5090 Laptop)

| Process | Model weights | KV cache pool (`util * total - weights`) | Total reserved |
|---|---:|---:|---:|
| ASR (current `start-vllm.sh`, util 0.90) | ~9 GB | ~13 GB | ~22 GB |
| TTS (proposed, util 0.45) | ~7.5 GB | ~3 GB | ~10.5 GB |

Both at current utilization will OOM. **Phase 0 outcome: lower the ASR
process's `--gpu-memory-utilization` from 0.90 to 0.55 in `start-vllm.sh`.**
That gives ASR ~13 GB total (9 GB weights + ~4 GB KV cache) and leaves ~11 GB
for TTS — enough for both to be resident simultaneously.

The ASR KV cache shrinkage from ~13 GB → ~4 GB is acceptable: starling's
realtime sessions are single-user and rotate context every 200 commits anyway
(documented in CLAUDE.md). Verified after Phase 0 boot test passes.

## Voice catalog (canonical, from `params.json`)

The 20 preset voices are enumerated in
`models/cache/tts/params.json` under
`multimodal.audio_tokenizer_args.voice` as a `name → index` map. The matching
`.pt` embedding files live in `models/cache/tts/voice_embedding/`.

Pass the `name` (e.g. `"casual_male"`) as the `voice` field in the
`/v1/audio/speech` request body — vllm-omni does the embedding lookup
internally.

| Voice ID | Index | Embedding file | Language hint |
|---|---:|---|---|
| casual_female | 0 | casual_female.pt | en (casual) |
| casual_male | 1 | casual_male.pt | en (casual) |
| cheerful_female | 2 | cheerful_female.pt | en (cheerful) |
| neutral_female | 3 | neutral_female.pt | en (neutral) |
| neutral_male | 4 | neutral_male.pt | en (neutral) |
| pt_male | 5 | pt_male.pt | Portuguese |
| pt_female | 6 | pt_female.pt | Portuguese |
| nl_male | 7 | nl_male.pt | Dutch |
| nl_female | 8 | nl_female.pt | Dutch |
| it_male | 9 | it_male.pt | Italian |
| it_female | 10 | it_female.pt | Italian |
| fr_male | 11 | fr_male.pt | French |
| fr_female | 12 | fr_female.pt | French |
| es_male | 13 | es_male.pt | Spanish |
| es_female | 14 | es_female.pt | Spanish |
| de_male | 15 | de_male.pt | German |
| de_female | 16 | de_female.pt | German |
| ar_male | 17 | ar_male.pt | Arabic |
| hi_male | 18 | hi_male.pt | Hindi |
| hi_female | 19 | hi_female.pt | Hindi |

Phase 1 should hard-code this 20-entry list in `voxtral_server/tts/voices.py`
(it's static for the released checkpoint) and surface it via
`GET /api/tts/voices` with friendlier display names + language tags.

## Audio output format (from `params.json`)

- Sample rate: **24000 Hz** (NOT the 16 kHz used by ASR)
- Frame rate: 12.5 audio frames/s
- vLLM-omni accepts `response_format`: `wav | pcm | flac | mp3 | aac | opus`
- Default in model card example: `wav`, dtype float32

## Boot failures observed during spike

### Boot failure 1 — missing `--config-format mistral`

```
INFO ... Resolved architecture: MistralForCausalLM
ERROR ... ValueError: There is no module or parameter named 'acoustic_transformer'
                     in MistralForCausalLM. The available parameters are: {...}
```

Root cause: TTS repo has no `config.json`, only `params.json`. Without
`--config-format mistral`, vllm-omni's `voxtral_tts` config parser is never
invoked and vllm falls back to plain `MistralForCausalLM` which cannot bind
the audio decoder weights. Fixed by passing all three `mistral` flags.

## Boot failure 2 — missing `--omni`, also missing `--task-type`

`vllm-omni serve` without `--omni` runs the vanilla single-stage path → same
`acoustic_transformer` error as Boot failure 1. The `--omni` flag flips it
into the multi-stage pipeline that vllm-omni's `voxtral_tts.yaml`
(`vllm_omni/model_executor/stage_configs/voxtral_tts.yaml`) describes:

- Stage 0: AR text→audio-tokens (AsyncOmniEngine, vLLM core, `gpu_memory_utilization: 0.8`)
- Stage 1: audio_tokenizer (CUDA-graph capture path, `gpu_memory_utilization: 0.1`)
- Stage 2: audio_generation (acoustic transformer)

`--task-type CustomVoice` selects the built-in 20-voice path. Other modes
(`Base` for voice cloning with `ref_audio` + `ref_text`, and `VoiceDesign`)
are not part of this feature.

## Smoke test results

After the successful boot, three syntheses on the same TTS process:

| Test | Voice | Input | Output audio | Wall | Realtime factor |
|---|---|---|---|---:|---:|
| Short EN | casual_male | 47 chars | 4.56s @ 24 kHz mono PCM16 | 1.68s | **0.37× RT** |
| Long EN (8× pangram) | casual_male | 359 chars | 21.12s | 5.80s | **0.27× RT** |
| Short DE | de_male | 49 chars | 2.24s | 0.69s | **0.31× RT** |

All three WAVs decoded correctly with `soundfile`: 24 kHz mono PCM16
(`file(1)` confirms `RIFF (little-endian) data, WAVE audio, Microsoft PCM,
16 bit, mono 24000 Hz`).

**Latency for the 30s-class benchmark is fast enough that batched playback
is fine for v1; streaming is a Phase-2 nice-to-have, not required.**

## VRAM verdict (the surprise)

Idle, with TTS-only resident: **20.97 GiB used / 3.17 GiB free** on the 24 GiB
RTX 5090 Laptop. The outer `--gpu-memory-utilization 0.45` flag we passed was
ignored — vllm-omni honours per-stage `gpu_memory_utilization` from the YAML
(`stage 0: 0.8`, `stage 1: 0.1`, plus stage 2 overhead). 0.8 of 24 GiB ≈ 19 GiB
matches what we observed.

This **breaks the original VRAM math.** With ASR also resident (~13 GiB at
util 0.55), the total target is ~33 GiB on a 24 GiB GPU — won't fit.

### Phase-1 implication

We need to **override the per-stage utilization** rather than the outer flag.
Two paths:

1. **Copy `voxtral_tts.yaml` to `apps/vllm-server/configs/voxtral_tts.yaml`,
   lower `gpu_memory_utilization` for stage 0 from 0.8 → ~0.45, then pass
   `--stage-configs-path apps/vllm-server/configs/voxtral_tts.yaml`** to
   `vllm-omni serve`. Cleanest, but ties us to one snapshot of the upstream
   YAML.
2. **Mutual-exclusion at the proxy layer.** Keep TTS off by default; spin it
   up on first request and accept that ASR sessions can't run simultaneously
   with a TTS request. Bad UX for our use case; reject.
3. **Lazy unload.** Run a single GPU-resource manager that loads ASR by
   default, swaps in TTS on demand. Possible later, but adds significant
   complexity.

**Phase 1 picks option 1.** A copied stage YAML is cheap and pinned to the
behaviour we tested. Document the source revision so the upstream change
detector is obvious.

(Update to `start-vllm.sh`: `--gpu-memory-utilization` lowered from 0.90 →
0.55 — *necessary but not sufficient*. Phase 1 must also override TTS-side
utilisation in the copied YAML.)

## Phase 0 status

- [x] Boot succeeds with `--omni --task-type CustomVoice --tokenizer-mode mistral
      --config-format mistral --load-format mistral`.
- [x] `POST /v1/audio/speech` returns a playable WAV (3 voices verified).
- [x] Latency: 0.27–0.37× RT — fast enough that streaming is not required for v1.
- [ ] `nvidia-smi` snapshot of ASR + TTS *simultaneously* resident — **blocked**
      on stage-config override (Phase 1 first task).

This is the gate for Phase 1: implement the copied stage YAML override and
verify both processes coexist. Everything else in Phase 0 is green.

## Files added by Phase 0

- `models/download.sh` — added `--tts-only`, `--no-tts`, downloads
  `mistralai/Voxtral-4B-TTS-2603` to `models/cache/tts/`.
- `apps/vllm-server/start-vllm-tts.sh` — boots the second `vllm-omni` process
  on port 8002.
- `apps/vllm-server/start-vllm.sh` — `--gpu-memory-utilization` 0.90 → 0.55
  (overridable via `VOXTRAL_VLLM_GPU_UTIL`).
- `apps/vllm-server/scripts/spike_tts.py` — standalone smoke-test against
  the local TTS server.
- `apps/vllm-server/docs/tts_spike.md` — this file.

## Phase 1 update (the coexistence limitation)

Phase 1 added the stage-config override (`apps/vllm-server/configs/voxtral_tts.yaml`)
to bring TTS VRAM down dramatically by capping per-stage utilisation +
batching:

- Stage 0 (AR LLM): `gpu_memory_utilization` 0.8 → 0.40, `max_num_seqs` 32
  → 4, `max_model_len` 4096 → 1024.
- Stage 1 (audio_tokenizer): unchanged at 0.10.

**Result of the YAML override (verified):**
- TTS-only resident: **11.4 GiB / 24 GiB** (was ~21 GiB before the override).
- E2E synth via FastAPI proxy works: 3.2s of 24 kHz mono WAV in 1.1s wall
  (0.34× RT). German voice (de_male) verified.

**Coexistence outcome (also verified, three times):** ASR + TTS *cannot*
both load simultaneously on the 24 GiB GPU. With ASR at util 0.45–0.55
(11–13 GiB resident) and TTS YAML overrides (11.4 GiB resident), the second
process to boot reliably hits `ValueError: No available memory for the cache
blocks` — vLLM's per-process budget exceeds what's actually free after the
first process loaded. Tested orderings: ASR-then-TTS, TTS-then-ASR. Both
fail.

The root cause is vLLM's own workspace overhead (CUDA streams, NCCL
buffers, optimizer state for warmup) — it needs ~3-5% of total GPU memory
on top of the declared `gpu_memory_utilization`, and 24 GiB has no headroom
once both `weights + KV cache` budgets are stacked.

**Phase-1 recommendation: ASR-only OR TTS-only, not both.** Operators
choose by starting one of `start-vllm.sh` / `start-vllm-tts.sh` (the
FastAPI server tolerates either or both being absent — TTS routes simply
return upstream errors when port 8002 is unreachable).

### Future-work options (none required for v1)

- **Lazy-load TTS, idle-unload.** Spin TTS up on first `/api/tts/synthesize`
  request, kill it after N idle seconds. Adds ~75s startup latency on first
  request but lets ASR keep running otherwise. Needs a small process
  manager in FastAPI.
- **CPU swap for ASR.** `vllm serve --swap-space N` lets vLLM page KV cache
  to system RAM. Allows ASR to run with much smaller GPU util but adds
  per-token latency.
- **Larger GPU.** A 32 GiB or 48 GiB card has the headroom for both at
  generous settings. RTX 5090 desktop (32 GiB) is the obvious target.

Each of these is meaningful work; v1 takes the simpler path and lets the
operator choose at startup.

## Phase 1 backend deliverables

- `apps/vllm-server/configs/voxtral_tts.yaml` — stage-config override.
- `apps/vllm-server/start-vllm.sh` — util/mml lowered (still doesn't fit
  alongside TTS — see above; users typically run one or the other).
- `apps/vllm-server/start-vllm-tts.sh` — passes `--stage-configs-path`.
- `apps/vllm-server/voxtral_server/tts/` — voices.py, storage.py,
  client.py, __init__.py.
- `apps/vllm-server/voxtral_server/api/tts_routes.py` — REST surface.
- `apps/vllm-server/voxtral_server/config.py` — `tts_*` settings.
- `apps/vllm-server/voxtral_server/models.py` — TTS Pydantic models.
- `apps/vllm-server/tests/test_tts_storage.py` — 30 sanitizer + storage tests.
- `apps/vllm-server/tests/test_tts_api.py` — 18 API integration tests.
- All 61 pytest cases pass.

## Phase 2 — streaming playback

### Upstream behaviour discovered

`POST /v1/audio/speech` with `stream=true` accepts `response_format=pcm` or
`response_format=wav`, but vllm-omni 0.18's WAV-streaming path is broken:

```
AssertionError: First audio chunk must include sample rate metadata for
                WAV streaming
```

PCM streaming works correctly: `audio/pcm` chunked, ~10 chunks/utterance,
**TTFB ~290 ms** measured directly against vllm-omni.

Workaround (implemented in this app): request PCM upstream and **prepend a
streaming WAV header on our side** (44 bytes, RIFF + data sizes both
`0xFFFFFFFF` placeholders). The browser plays the resulting `audio/wav`
just fine.

### Backend implementation

- `voxtral_server/tts/wav.py` — `streaming_header()` builds the 44-byte
  RIFF/WAVE/PCM header with placeholder sizes.
- `voxtral_server/tts/client.py` — `TtsClient.synthesize_stream(text, voice)`
  is an async generator yielding raw PCM chunks via `httpx.stream(...)`.
  Tolerates vllm-omni's habit of dropping the terminal HTTP chunk (treats
  `RemoteProtocolError` after first byte as clean EOF).
- `voxtral_server/api/tts_routes.py` — `POST /api/tts/synthesize` now
  branches on `save`:
  - `save=true` (default): same as Phase 1 — buffered synth + save-to-disk +
    `ApiResponse` envelope.
  - `save=false`: returns `StreamingResponse(media_type="audio/wav")` with
    `Cache-Control: no-store` and `Content-Disposition: inline`. Header is
    yielded eagerly; upstream PCM chunks are streamed through.

### Frontend

- `frontend/js/modules/tts-manager.js` — `synthesizeForPlayback()` POSTs
  `save:false` and uses `await response.blob()` + `URL.createObjectURL` for
  `<audio src=>`.
- `frontend/js/main-sessions.js` — revokes the previous object URL across
  syntheses to avoid leaking blobs.
- The mode toggle in the UI hasn't changed: "Play in browser" → streaming,
  "Save to server" → Phase 1 path. Status text now distinguishes
  `Streamed …` vs `Saved …`.

### Phase-2 latency (verified end-to-end through FastAPI proxy)

| Mode | TTFB | Wall | Audio | Notes |
|---|---:|---:|---:|---|
| `save=false` (streaming) | **39 ms** | 2.85 s | 9.76 s | header lands first |
| `save=true` (buffered)   | ~2100 ms | 2.10 s | 7.76 s | Phase 1 baseline |

The ~54× TTFB drop confirms the streaming pipeline. *Perceived* latency in
the browser is still gated by `<audio src=blob>` waiting for the full body
in this v1; partial-playback via `MediaSource` Extensions is the obvious
Phase 2.x follow-up. For typical 5-10 s utterances generated at ~0.3× RT,
the buffered mode is already fast enough that MSE is a UX nice-to-have.

### Tests

- `apps/vllm-server/tests/test_tts_wav.py` — 5 unit tests (header layout,
  sample rates, bad bps, round-trip via stdlib `wave`).
- `apps/vllm-server/tests/test_tts_api.py` — added 2 cases:
  `test_synthesize_play_mode_streams_wav` (verifies the chunked body has
  RIFF/WAVE magic, both sizes are `0xFFFFFFFF`, and the fake client's
  `synthesize_stream` was called instead of the buffered one) and
  `test_synthesize_play_mode_rejects_unknown_voice` (confirms the streaming
  branch returns 400 + JSON on validation failures).
- **pytest: 67 passed** (61 from Phase 1 + 5 wav + 2 API).
- `tests/e2e/specs/tts.spec.ts` — adjusted "Generate (play mode)" to assert
  `save=false` and `audio.src` matches `/^blob:/`. Added "Play mode does
  NOT add anything to the saved-files list".
- `tests/e2e/fixtures/mockServer.ts` — `/api/tts/synthesize?save=false`
  serves a stub WAV with `Transfer-Encoding: chunked`; validation errors
  map to 400 + JSON.
- **playwright: 17 passed** (7 speakers + 10 tts).

### Phase-2 deliverables

- `apps/vllm-server/voxtral_server/tts/wav.py` (new).
- Updates to `voxtral_server/tts/client.py`, `voxtral_server/api/tts_routes.py`,
  `frontend/js/modules/tts-manager.js`, `frontend/js/main-sessions.js`,
  `tests/e2e/specs/tts.spec.ts`, `tests/e2e/fixtures/mockServer.ts`.
- `apps/vllm-server/tests/test_tts_wav.py` (new).

## Phase 3 — installer integration

`pip install vllm-omni` was a manual post-`init.sh` step. Phase 3 wires it
into the existing initializer behind a `--tts` flag (or
`VOXTRAL_INSTALL_TTS=1` env var) so a fresh deployment can get to the TTS
tab in one command:

```bash
./init.sh --tts        # full setup including TTS
./init.sh              # ASR-only (default; lighter venv, no TTS deps)
```

### What the flag does

- Reorders into 7 numbered steps (vs 6 for ASR-only). The new step is
  `[5/7] Installing vllm-omni`, slotted **between** vLLM install and
  `voxtral-server` install. This preserves HF discussion #29's hard
  ordering rule: install `vllm` *before* `vllm-omni`.
- Never passes `--upgrade` to `pip install vllm-omni`. The discussion
  documented that `--upgrade` pulls torch 2.11 which crashes vLLM's
  compiled extensions; a stray rerun would silently break the venv.
- After the install, runs `scripts/check_tts_install.py` automatically to
  verify both packages import, torch is in the supported range
  `2.8 ≤ x < 2.11`, and the `voxtral_tts` model registry is wired up.
  Same script also runs on every `--tts` rerun (idempotency: skips the
  pip install but re-validates).
- Auto-downloads the TTS model via the shared `models/download.sh
  --tts-only` (8 GB) when `--tts` is set. The BF16 ASR model is still
  downloaded by default.
- Updates the "Next steps" output to include the `start-vllm-tts.sh`
  reminder *and* a one-line callout that ASR + TTS can't coexist on a
  24 GiB GPU (with a pointer back to this memo's "Coexistence outcome"
  section).

### `--dry-run` flag (used by tests)

`./init.sh --dry-run` (and `--tts --dry-run`) prints every install command
without executing it. Each side-effecting line is wrapped through a
`_run()` helper that echoes `[DRY-RUN] <command>` instead of running.
The bash test below uses this to assert the install order without
actually mutating the venv.

### Tests added

- `apps/vllm-server/tests/test_tts_install.py` — pytest, 4 cases. Unit-tests
  the version parser in `check_tts_install.py`. The end-to-end case is
  skipped when `vllm_omni` isn't importable (so CI without GPU still
  passes; the spike-set-up venv does run it).
- `apps/vllm-server/tests/scripts/test_init_tts_flag.sh` — bash, 7
  assertions. Covers: vllm-before-vllm-omni order, no `--upgrade` flag,
  step-label numbering for both flow shapes, `VOXTRAL_INSTALL_TTS=1`
  env-var equivalence to `--tts`, skip-marker visibility when TTS is off.

Run tests:
```bash
source apps/vllm-server/.venv/bin/activate
python -m pytest apps/vllm-server/tests/test_tts_install.py
bash apps/vllm-server/tests/scripts/test_init_tts_flag.sh
```

### Phase-3 deliverables

- `apps/vllm-server/init.sh` — `--tts` / `--dry-run` flags, `_run()`
  wrapper, `install_vllm_omni()` step, dynamic step counts (`/6` vs `/7`),
  TTS-aware "Next steps" output.
- `apps/vllm-server/scripts/check_tts_install.py` (new) — integrity
  verifier with parseable error messages.
- `apps/vllm-server/start-vllm-tts.sh` — pre-reqs comment now points at
  `./init.sh --tts`.
- `apps/vllm-server/tests/test_tts_install.py` (new) — pytest.
- `apps/vllm-server/tests/scripts/test_init_tts_flag.sh` (new) — bash.

### Open Phase-3 items

- The vllm-omni install pulls ~3 GB of unrelated deps (gradio, diffusers,
  librosa, openai-whisper). Documented; not slimmed. Slimming would
  require upstream PRs against vllm-omni and is out of scope.
- A future `./init.sh --tts-only` (skip ASR vLLM weights for boxes that
  *only* run TTS) would save ~9 GB of disk; not in v1.

## Phase 4 — progressive playback (Web Audio)

Phase 2's `save=false` streaming endpoint already returns chunked
`audio/wav` with `0xFFFFFFFF`-placeholder sizes; **TTFB measured at 39 ms**
through the FastAPI proxy. The frontend, however, awaited
`response.blob()` before letting the `<audio>` element play it, so first
sound was still gated by the full body. Phase 4 swaps that for true
progressive playback driven by the Web Audio API.

### Why Web Audio, not MediaSource Extensions

| Path | Verdict |
|---|---|
| `MediaSource` + `audio/wav` SourceBuffer | Rejected — `MediaSource.isTypeSupported('audio/wav')` returns `false` in Chrome (MSE only supports fragmented MP4 / WebM). |
| Re-mux PCM to fMP4 server-side, then MSE | Rejected — significant server-side plumbing for marginal benefit. |
| **Web Audio API: chunked PCM → `AudioBufferSourceNode` per chunk** | **Chosen.** Sample-accurate `start(when)` scheduling is exactly what Phase 5 needs to stitch sentence streams gap-free, and there's nothing to add server-side. |

### `frontend/js/modules/tts-player.js`

A standalone `TtsPlayer` class that:

- Consumes the streaming response via `response.body.getReader()`.
- Parses the 44-byte WAV header on the first read; subsequent bytes are
  raw PCM frames.
- Schedules each chunk as a fresh `AudioBufferSourceNode`, starting at
  the previous source's `endedTime` so back-to-back samples are
  contiguous in the output stream.
- Tracks state through `idle → buffering → playing → paused/ended`,
  emitting `tts:state` `CustomEvent`s on its `EventTarget` and exposing
  the latest snapshot on `window.__ttsPlayerState` for Playwright.
- `pause()` / `resume()` use `AudioContext.suspend()` /
  `AudioContext.resume()` (the entire schedule pauses atomically).
- `cancel()` aborts the fetch reader, stops + disconnects every live
  source, closes the `AudioContext`.
- A `visibilitychange` handler pauses on tab-blur to avoid Chrome's
  AudioContext clock-throttling glitches; resumes on tab-focus.

Public API surface:

```js
const p = new TtsPlayer({ onStateChange });
await p.start(response);   // resolves on EOF, throws on protocol error
p.pause();
p.resume();
p.cancel();                // tear down everything
p.getState();              // {state, currentTimeSecs, bufferedSecs, sampleRate}
p.addEventListener('tts:state', e => …);
```

### Frontend wiring changes

- `index.html` — added `#tts-pause-btn`, `#tts-resume-btn`,
  `#tts-stop-btn` inside the TTS tab. The existing `<audio>` element
  remains, used only by save-mode and hidden during stream-mode.
- `tts-manager.js::synthesizeForPlayback` no longer awaits a blob;
  returns the raw `Response`.
- `main-sessions.js::generateTts` (play branch) instantiates a
  `TtsPlayer`, hooks `onStateChange` to update status/buttons, and hands
  the streaming `Response` straight to `player.start()`.
- The status text now reflects the player's actual progress: `Buffering…`
  → `Playing (3.2 s, 0.7 s buffered)` → `Done (9.1 s)`.

### Tests

- **Playwright (12 cases total in `tests/e2e/specs/tts.spec.ts`)** — added
  three Phase-4 cases on top of Phase 1/2's eight: progressive-play state
  transitions via `window.__ttsPlayerState`, Pause/Resume toggle, Stop
  button transitions to `Stopped`. The mock server in
  `tests/e2e/fixtures/mockServer.ts` now emits a streaming WAV header +
  three PCM chunks with 25 ms gaps so the player exercises real chunked
  scheduling.
- pytest unchanged — Phase 4 is frontend-only.

### Open Phase-4 items / known limitations

- AudioContext requires a user gesture to start. The Generate button
  click satisfies the gesture; if a future code path triggers playback
  without one (autoplay-style flows), `_ctx.resume()` will reject
  silently and the user will hear nothing. Document if/when that path
  ships.
- AudioContext sample-rate mismatch (some Chromes default to 48 kHz)
  is handled inside `AudioBuffer` by setting `sampleRate=24000` and
  letting the node graph resample on playback. Verified with the test
  rig.
- Pause-on-blur is purely a glitch-avoidance affordance; Phase 5
  long-form syntheses might want the user to leave the tab and come
  back. Revisit if that's actually a use case.

### Phase-4 deliverables

- `frontend/js/modules/tts-player.js` (new).
- `frontend/index.html` — Pause/Resume/Stop buttons.
- `frontend/js/modules/tts-manager.js` — `synthesizeForPlayback` returns
  `Response`.
- `frontend/js/main-sessions.js` — `setupTtsTab` instantiates `TtsPlayer`
  and wires the new buttons.
- `tests/e2e/fixtures/mockServer.ts` — `streamingWavHeader()` + chunked
  PCM stub.
- `tests/e2e/specs/tts.spec.ts` — three new cases for Phase 4.

## Phase 5 — long-form text (sentence-segmented streaming concat)

Phase 5 raises the per-request input cap from **2 000 → 20 000 chars**
(~20 minutes of speech) and switches the route to a sentence-by-sentence
synthesis pipeline so we don't have to feed the model a 20-minute prompt
and wait for the whole audio at the end.

### Server-side sentence splitter (`voxtral_server/tts/text.py`)

A standalone, deterministic splitter that handles the common European
edge cases the burn-server's transcription splitter doesn't (it had to be
generic enough for any prose, not just transcript-style):

- **Abbreviations:** `Dr.`, `Mr.`, `Prof.`, `Frau`, `Z.B.`, `bzw.`, `vs.`,
  `etc.`, `ggf.`, `Inc.`, `Ltd.`, … (50+ entries, all lowercase). Match
  is case-insensitive on the trailing word.
- **Numbered prefixes:** `19. November` doesn't split; `Sektion 3.
  Beachten Sie …` doesn't split (the rule: bare integer followed by `.`
  isn't a terminator). But **decimal numbers do split**:
  `Pi ist 3.14. Schön, oder?` → `["Pi ist 3.14.", "Schön, oder?"]`.
  We disambiguate by checking what precedes the digit run: if it's a `.`,
  it's a decimal terminator; otherwise it's a numbered prefix.
- **Terminators:** `.`, `!`, `?`, `…`, and the literal three-dot `...`.
- **Lower-case follow-ups:** if the next word starts lowercase, no split
  (mid-sentence punctuation: URLs, decimals, abbreviations not in our list).
- **Soft cap:** any chunk longer than `MAX_SENTENCE_CHARS` (1500) is
  re-split on commas to keep per-upstream latency bounded.

24 pytest cases verify the splitter, including a full German news
paragraph that round-trips correctly.

### Per-sentence streaming + save

`TtsClient.synthesize_stream_concat(text_parts, voice)` opens one
upstream `/v1/audio/speech?stream=true` per sentence, yields its PCM
bytes, closes, repeats. First-sentence errors propagate (caller can
return 5xx); per-sentence errors on later sentences are logged and
skipped — partial audio is better than tearing down the whole stream.

The `POST /api/tts/synthesize` route:

- Splits `text` once with `text.split_sentences`.
- **Single-sentence path:** unchanged from Phase 1/2 (buffered or
  streaming).
- **Long-form streaming (`save=False`):** yields a single streaming WAV
  header, then `synthesize_stream_concat` is consumed straight to the
  client. Polls `request.is_disconnected()` between sentences so a
  closed tab stops upstream work cleanly. Honours
  `settings.tts_long_max_secs` as a wall-clock cap.
- **Long-form save (`save=True`):** buffers all PCM in memory, wraps in
  a real (non-streaming) WAV with proper RIFF/data sizes, writes one
  file. Memory usage at the 20-min cap is ~58 MB — acceptable.

### Settings

```python
tts_max_chars: int = 20000             # was 2000 in Phase 1/2
tts_long_max_secs: float = 300.0
tts_long_max_concurrency: int = 1      # serial; bump to 2+ once measured
```

The frontend reads `max_chars` from `/api/tts/config` so the textarea
caps at 20 000 automatically; the mock server emits the same value.

`tts_long_max_concurrency=1` (serial) is the default until we measure
stage-0 VRAM under parallel load. Phase 5 ships without sentence
prefetching; bumping to 2 is a one-line setting change once verified.

### Tests

- **`tests/test_tts_text.py` (24 cases)** — splitter table tests:
  empty/whitespace, single sentence, multiple sentences, English +
  German abbreviations, numbered prefixes, decimals, ellipses, URLs
  with dots, multi-newline gaps, soft-comma split for >1500-char chunks.
- **`tests/test_tts_api.py` (3 new cases)** —
  `test_synthesize_long_form_streams_one_header_then_concatenated_pcm`
  (assert one header + N×PCM, one upstream call per sentence),
  `test_synthesize_long_form_save_writes_one_wav` (real RIFF/data sizes,
  not 0xFFFFFFFF), `test_synthesize_max_chars_cap_unchanged` (20 000
  ok, 20 001 rejected).
- **Playwright** — new `Long-form input is accepted` case fills 5 400
  chars and asserts the player reaches `ended` plus the request body
  has >2 000 chars (proving the textarea cap was raised).

Final test counts after Phase 5:
- pytest: **98 passed**
- playwright: **20 passed** (7 speakers + 13 tts).

### Risks called out and how Phase 5 handles them

- **Splitter mishandling a real-world edge case**: 24 unit tests cover
  the common ones; failures degrade gracefully (over-long chunk → soft
  comma split → still works, just slower).
- **Per-sentence boundary clicks**: not addressed in v1. If listener QA
  flags artefacts, add a 5–10 ms linear ramp inside the
  `synthesize_stream_concat` PCM stream. Document at the time.
- **vllm-omni terminal-chunk-drop compounding across N sentences**: the
  per-sentence wrapper already tolerates the drop after the first byte
  has landed. Across many sentences each drop is independent.
- **Long-form save memory**: 20 min × 24 kHz × 2 bytes ≈ 58 MB peak. OK
  for typical hardware; document.
- **Concurrency at vllm-omni stage 0**: kept at 1 for Phase 5 v1. No
  measured regression.

### Phase-5 deliverables

- `apps/vllm-server/voxtral_server/tts/text.py` (new) — sentence
  splitter.
- `apps/vllm-server/voxtral_server/tts/client.py` — added
  `synthesize_stream_concat`.
- `apps/vllm-server/voxtral_server/api/tts_routes.py` — long-form save
  + streaming branches; `_pcm_to_wav` helper for the save path.
- `apps/vllm-server/voxtral_server/config.py` — bumped `tts_max_chars`,
  added `tts_long_max_secs`, `tts_long_max_concurrency`.
- `frontend/index.html` — textarea `maxlength=20000` + updated label.
- `tests/e2e/fixtures/mockServer.ts` — `/api/tts/config` returns
  `max_chars: 20000`.
- `tests/e2e/specs/tts.spec.ts` — long-form happy-path case.
- `apps/vllm-server/tests/test_tts_text.py` (new).
- `apps/vllm-server/tests/test_tts_api.py` — three new long-form cases.

## Phase 6 — TTS-on-demand with idle unload + auto-start/stop

Phase 1 documented the hard limit: ASR + TTS together overflow the 24 GiB
GPU. Phase 6 sidesteps it: the TTS subprocess starts lazily on the first
TTS request and unloads after configurable idle. Operators no longer need
to remember to run `start-vllm-tts.sh` themselves, and ASR sessions are
not crashed by an attempted TTS load.

### `voxtral_server/tts/lifecycle.py`

A `TtsLifecycle` async state machine (`idle → starting → ready → stopping
→ idle` plus `blocked` for the GPU-availability path) that owns the child
process via `asyncio.create_subprocess_exec(start-vllm-tts.sh)`. Public
API:

- `await ensure_started()` — idempotent. Returns immediately when
  already `ready`. Otherwise: pre-checks GPU free memory; spawns the
  subprocess; polls `/v1/models` until 200 OR `tts_boot_timeout_secs`;
  starts the idle-unload watcher.
- `await ensure_stopped()` — `SIGTERM` with a 30 s grace, `SIGKILL`
  fallback. Cancels the idle watcher.
- `await status()` — cheap snapshot for the frontend badge.
- `note_activity()` / `synth_started()` / `synth_finished()` —
  reset the idle timer; in-flight counter prevents the timer from
  firing during a long-form synth.

A `_Probe` dataclass exposes hooks (`spawn_subprocess`, `check_gpu_free`,
`health_check`, `sleep`) so tests can drive the full state machine
without real subprocesses. The pytest suite covers: ready transition,
concurrent-boot collapse, idle unload, in-flight-protection,
GPU-blocked-by-ASR vs generic-low-VRAM, boot timeout, autostart=false
fast-path.

### Routes wiring

- `GET /api/tts/status` — new endpoint returning the lifecycle snapshot
  (`state`, `pid`, `boot_elapsed_secs`, `inflight_synths`,
  `blocked_reason`, …). The frontend polls this every 5 s while the TTS
  tab is active.
- `POST /api/tts/synthesize` — first awaits `ensure_started()`. On
  `LifecycleError(reason='blocked')` it returns **HTTP 503** with the
  `{success:false, error:"blocked by ASR session"}` body. On
  `boot_timeout` it returns 4xx with the wall-clock-cap message.
  `synth_started()` / `synth_finished()` bracket every successful synth
  to keep the idle timer accurate.
- `voxtral_server/main.py` — `@app.on_event("shutdown")` calls
  `ensure_stopped()` so the GPU is clean for the next boot.

### Settings

```python
tts_autostart: bool = True
tts_idle_unload_secs: float = 600.0    # 10 min; 0 disables
tts_boot_timeout_secs: float = 180.0   # observed ~75 s in the spike
tts_min_free_vram_gib: float = 12.0    # < this → refuse to start
tts_asr_port: int = 8001               # for the "blocked by ASR" probe
```

`autostart=false` keeps the previous behaviour: ASR-only operators can
start `start-vllm-tts.sh` manually and the lifecycle just probes its
health.

### Frontend status badge

A small pill in the TTS tab header polls `/api/tts/status` every 5 s
and renders one of:

- `idle` — grey
- `warming up · 35s left` — amber, ticks down based on
  `boot_timeout_secs - boot_elapsed_secs`
- `ready` — green
- `stopping…` — amber
- `blocked by ASR session` — red, **also disables the Generate button**
  with a tooltip explaining why

When the upstream (`/api/tts/status`) is unreachable the badge shows
`unreachable` rather than crashing the tab.

### Risks and how Phase 6 handles them

- **Subprocess SIGTERM on WSL2 sometimes leaves CUDA pinned.** Mitigated
  with a 30 s SIGTERM-grace + SIGKILL fallback, plus an explicit
  `await proc.wait()` so the lifecycle doesn't return until the OS has
  reaped the child. The "GPU pinned" failure is logged but not blocking.
- **Race between GPU pre-check and ASR session start.** The pre-check
  is best-effort; if ASR grabs memory between the check and the spawn,
  the upstream returns `ValueError: No available memory for the cache
  blocks` during boot, and the lifecycle catches that as `early_exit`
  with a clear surface to the user.
- **Idle-unload during a long-form synth.** `synth_started()` /
  `synth_finished()` form a counter that the idle loop honours — the
  timer waits until the counter is back to 0 before firing.
- **Concurrent boot requests** — `asyncio.Lock` plus a re-check after
  the lock is acquired makes ensure_started idempotent under contention
  (verified by the `test_concurrent_ensure_started_collapses_to_one_boot`
  pytest case).

### UX notes

A cold-start synth now takes the boot time (~75 s observed) on top of
the synthesis time. Users see the badge sequence
`idle → warming up · 75s left → … → ready`, and the Generate request
awaits the boot before kicking off. In the most common single-user case
(open the TTS tab, type, click Generate), this is a one-time per-session
cost; subsequent generations within `tts_idle_unload_secs` are
warm-fast.

### Tests

- **pytest** — `tests/test_tts_lifecycle.py`, **10 cases** covering the
  full state machine without real subprocesses.
- **pytest** — `tests/test_tts_api.py` updated with a `_FakeLifecycle`
  stub so the existing 18 API tests still pass alongside the lifecycle.
- **Playwright** — three new badge cases: default `ready`, `warming up
  · …s left` rendering, `blocked by ASR` disables Generate.

Final test counts after Phase 6:
- pytest: **108 passed**
- playwright: **23 passed** (7 speakers + 16 tts).

### Phase-6 deliverables

- `apps/vllm-server/voxtral_server/tts/lifecycle.py` (new).
- `apps/vllm-server/voxtral_server/api/tts_routes.py` —
  `GET /api/tts/status`, `ensure_started()` gate on every synth, 503 on
  `blocked` reason, in-flight counters.
- `apps/vllm-server/voxtral_server/config.py` — five new lifecycle
  settings.
- `apps/vllm-server/voxtral_server/main.py` — shutdown hook tears down
  the TTS subprocess.
- `frontend/index.html` — engine-status badge inside `#tts-content`.
- `frontend/js/modules/tts-manager.js` — `fetchTtsStatus`.
- `frontend/js/main-sessions.js` — `refreshTtsEngineStatus` polling
  every 5 s; disables Generate while `state=blocked`.
- `tests/e2e/fixtures/mockServer.ts` — `/api/tts/status` route + mutable
  `ttsStatus` test handle.
- `tests/e2e/specs/tts.spec.ts` — three Phase-6 cases.
- `apps/vllm-server/tests/test_tts_lifecycle.py` (new) — 10 unit tests.
- `apps/vllm-server/tests/test_tts_api.py` — `_FakeLifecycle` stub.

## Phase 7 — voice cloning (`task_type=Base`)

Phase 7 lets users **clone any voice from a 5–30 s audio clip + transcript**
via Voxtral's `task_type=Base`. The Phase-6 lifecycle now restarts the
upstream into Base mode on demand and back to CustomVoice afterwards.

### Spike findings (the new mode is real, with caveats)

- vllm-omni 0.18 already exposes a built-in voice store: `GET/POST/DELETE
  /v1/audio/voices`. Multipart fields: `audio_sample` (file), `name`,
  `consent`, optional `ref_text`. We forward our uploads through to it
  and use the voice's `name` as the synth `voice` field.
- Synthesizing with an uploaded voice in a CustomVoice-launched server
  fails with `Unknown voice 'xxx'`. The server **must be running with
  `--task-type Base`** for cloning to work.
- Sending `ref_audio` inline (data URL) in a synth request crashed the
  upstream orchestrator. The viable path is the upload → register →
  synth-with-`voice=<name>` flow, not inline ref bytes.
- On task-type switch, the in-memory uploaded-voice store is wiped
  (only `/tmp/voice_samples/` audio files persist). The route therefore
  re-uploads the chosen voice to the upstream right after switching to
  Base.

### Lifecycle reload (`ensure_started(task_type)`)

`TtsLifecycle.ensure_started()` gained a `task_type` arg. State machine
now: if `state==ready` AND `_task_type==requested`, no-op. Otherwise
`stop+start` with the new task type. The lifecycle passes
`VOXTRAL_TTS_TASK_TYPE=<value>` as an env var to `start-vllm-tts.sh`,
which now reads + validates it (`CustomVoice|Base|VoiceDesign`).

`StatusInfo.task_type` is now part of the `/api/tts/status` payload so
the frontend can show "switching engine to Base mode" if we ever want
to.

### `voxtral_server/tts/refs.py`

Local catalog of uploaded reference clips:

```
voice_refs/
    <uuid>.wav       — re-encoded to 24 kHz mono PCM16 via ffmpeg
    <uuid>.json      — sidecar: {id, name, ref_text, permission_confirmed, …}
    _audit.log       — append-only "<ts>\tupload|delete\t<uuid>\t…"
```

The sanitizer is the Phase-1 `storage.StorageError` family:

- `name` must match `[A-Za-z0-9 _\-]{1,64}` and is metadata only.
- `voice_ref_id` must look like a uuid hex (validated before any
  filesystem access).
- `audio_filename` extension must be one of `wav/mp3/flac/ogg/m4a`
  (drives the temp-file extension; ffmpeg picks up the real format).
- `audio_bytes` capped (`tts_max_ref_audio_bytes`, 5 MB default).
- `duration` re-probed via stdlib `wave` after re-encoding; reject if
  outside `[5, 30]` seconds.
- `permission_confirmed=False` rejects unless
  `tts_require_permission=False` (intended only for local dev).
- `ref_text` capped at 1000 chars; can't be empty.

`_run_ffmpeg` and `_probe_duration` are injectable hooks so tests skip
real ffmpeg.

### Routes

- `POST /api/tts/voices/upload` — multipart with `audio_sample`,
  `name`, `ref_text`, `permission_confirmed` (bool). Saves locally
  AND forwards (via `TtsClient.upload_voice`) to upstream `/v1/audio/voices`.
  Failures on the upstream side are **logged but not fatal** — the
  local entry persists, and the server will re-sync the voice the next
  time the user synthesizes with it.
- `DELETE /api/tts/voices/{ref_id}` — removes locally + upstream.
- `GET /api/tts/voices` — extended to mix built-ins (`kind=builtin`)
  with cloned (`kind=cloned`).
- `POST /api/tts/synthesize` — accepts new `voice_ref_id`. When set:
  1. Look up `refs.get_ref(voice_refs_dir, voice_ref_id)` → 404 if
     missing.
  2. `lifecycle.ensure_started(task_type='Base')` → may take ~75 s if
     the upstream was running CustomVoice.
  3. Re-upload the ref's audio bytes to upstream so the in-memory
     voice store has it.
  4. Synthesize with `voice=<voice_ref_id>` (we use the uuid as the
     upstream voice name).

### Settings

```python
tts_voice_refs_dir: str = "./voice_refs"
tts_max_ref_audio_bytes: int = 5_000_000
tts_min_ref_duration_secs: float = 5.0
tts_max_ref_duration_secs: float = 30.0
tts_max_voice_refs: int = 50
tts_require_permission: bool = True
```

### Frontend UX

A collapsible `<details>` "Custom voices (zero-shot cloning)" section
inside the TTS tab:

- Name input, transcript textarea, file picker (audio/*), permission
  checkbox with explicit consent language, Upload button (disabled until
  every required field + the checkbox are filled).
- Uploaded clones appear in the existing voice `<select>` under a
  "Custom voices" optgroup with `value="ref:<uuid>"`. The frontend
  detects the `ref:` prefix and sends `voice_ref_id` instead of `voice`
  on the next synth call.
- Per-clone Delete buttons in a list under the upload form.
- Server-side permission check mirrors the client-side gate (Playwright
  test bypasses the client-side gate via `fetch()` to verify the server
  is the source of truth).

### Risks and how Phase 7 handles them

- **Voice-cloning abuse.** Mitigations: explicit checkbox copy
  ("I confirm I have permission … may be illegal …"), server-side
  enforcement of the flag, append-only audit log under
  `voice_refs/_audit.log` (records id/name/timestamp/permission flag —
  **NOT** the audio bytes or transcript). Document the legal reality
  in the README (TODO).
- **Reload latency.** First synth after a CustomVoice→Base switch
  costs the full ~75 s boot. Documented; acceptable for occasional
  cloning. If a user alternates rapidly, consider pinning to Base.
- **Upstream voice-store loss on restart.** Mitigated by re-uploading
  the chosen ref to upstream right before each cloned synth. Cheap
  (a few hundred KB).
- **Upstream `--task-type Base` may have different VRAM characteristics.**
  Spike not yet run in Base mode at scale. Phase 7 v1 ships with the
  same stage YAML; if Base needs different per-stage util, file a
  follow-up.
- **Path traversal / arbitrary file write.** Reused the Phase-1
  sanitizer + replaced the on-disk filename with our generated uuid;
  user-supplied `name` is metadata only.
- **Permission flag being trivially flipped client-side.** Server is
  the authoritative gate; Playwright `Permission checkbox must be
  ticked or upload is rejected at the server` test verifies.

### Tests

- **pytest `test_tts_refs.py` (new, 25 cases)** — sanitizer (name,
  filename extension, voice_id), permission gate, oversize/short/long
  duration, ref_text empty/oversized, list+delete round-trip,
  audit-log redaction.
- **pytest `test_tts_api.py` (10 new cases)** — voices listing
  includes `kind` field, upload happy path, upload rejects unconfirmed
  permission, oversize rejected, list-after-upload, delete happy path
  + not-found + bad-id, synth with `voice_ref_id` switches lifecycle
  to Base, unknown `voice_ref_id` rejects.
- **Playwright `tts.spec.ts` (5 new cases)** — collapsed by default,
  upload button gated by all-fields-set, upload posts multipart and
  populates the dropdown, server enforces permission gate (bypassing
  client-side disable), delete removes from list.

Final test counts after Phase 7:
- pytest: **143 passed**
- playwright: **28 passed** (7 speakers + 21 tts).

### Phase-7 deliverables

- `apps/vllm-server/voxtral_server/tts/refs.py` (new).
- `apps/vllm-server/voxtral_server/tts/client.py` — `upload_voice`
  + `delete_voice` wrappers around upstream's `/v1/audio/voices`.
- `apps/vllm-server/voxtral_server/tts/lifecycle.py` —
  `ensure_started(task_type)` reload; `_task_type` state; env-var
  spawn.
- `apps/vllm-server/voxtral_server/api/tts_routes.py` —
  `POST /api/tts/voices/upload`, `DELETE /api/tts/voices/{ref_id}`,
  `voice_ref_id` synth path.
- `apps/vllm-server/voxtral_server/config.py` — six voice-clone settings.
- `apps/vllm-server/voxtral_server/models.py` — `VoiceRefInfo` +
  `voice_ref_id` field on `TtsSynthesizeRequest`.
- `apps/vllm-server/start-vllm-tts.sh` — reads
  `VOXTRAL_TTS_TASK_TYPE` (CustomVoice/Base/VoiceDesign).
- `frontend/index.html` — collapsible "Custom voices" section.
- `frontend/js/modules/tts-manager.js` — `uploadVoiceRef`,
  `deleteVoiceRef`, `voiceRefId` field on synth helpers.
- `frontend/js/main-sessions.js` — `setupTtsTab` adds
  `renderVoiceSelect` + clone-form wiring + delete buttons; `generateTts`
  routes `ref:<uuid>` values to `voice_ref_id`.
- `tests/e2e/fixtures/mockServer.ts` — `/api/tts/voices/upload` + DELETE,
  `clonedVoices` test handle, multipart parser.
- `tests/e2e/specs/tts.spec.ts` — five Phase-7 cases.
- `apps/vllm-server/tests/test_tts_refs.py` (new).
- `apps/vllm-server/tests/test_tts_api.py` — voice-cloning section
  appended.

### Open Phase-7 items

- README needs a "responsible use" callout + a link to relevant
  legal warnings — not a code change.
- VRAM measurement in `--task-type Base` mode pending; Phase 1's
  numbers were CustomVoice-only.
- Audit log is local-filesystem-only; for multi-tenant deployments
  this should land in a centralized logger. Not required for v1.
