# Starling E2E browser tests

Playwright tests for the shared frontend. Currently focused on the Speakers
(live system-audio) source flow for both `burn-server` and `vllm-server`.

## Setup

```bash
cd tests/e2e
bun install
bunx playwright install chromium
```

## Run

```bash
bunx playwright test                # headless
bunx playwright test --headed       # visible browser
bunx playwright test --ui           # UI runner
```

## How it works

The tests do not require the real servers to be running. `fixtures/mockServer.ts`
boots a tiny HTTP + WebSocket signalling server that:

- Serves the shared `frontend/` directory statically.
- Fakes the `/api/*` surface (models, media, modes, sessions, config).
- Accepts both WebSocket flows: the existing receiver flow (media / SRT
  sessions) and the new uplink flow used for Speakers sessions
  (`ready {role:"uplink"}` → welcome → offer → answer).

The mock does not actually decode Opus — the test assertions stop at the
SDP signalling level. That's sufficient to guarantee:

1. The UI exposes the Speakers source tab and device dropdown.
2. Enumerated devices populate correctly.
3. Creating a Speakers session posts `source: "speakers"` (no `media_id`).
4. The frontend opens a WebSocket, sends `{type: "ready", role: "uplink"}`,
   then an SDP offer.

Real end-to-end audio transcription is covered by the server-side Rust /
Python integration tests (in `apps/burn-server/tests/speakers_api.rs` and
`apps/vllm-server/tests/test_speakers_api.py`).
