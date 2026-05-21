#!/usr/bin/env bash
#
# Record one demo video for the Starling blog series, with audio muxed in.
#
# Usage:
#   ./record-video.sh --post <slug>
#
# Where <slug> is one of:
#   voxtral-asr-vllm | voxtral-asr-burn | voxtral-tts-vllm | voxtral-tts-burn
#
# What it does:
#   1. Pre-warms the right model on the target server (curl a tiny request)
#      so the recording itself doesn't start with a cold cache + spinner.
#   2. Runs the matching Playwright spec, which records the page video to webm.
#   3. The spec logs an `audio_start_ms` marker so we know when in the video
#      timeline the ASR/TTS audio should start.
#   4. For ASR posts, the audio track is broadcast_1_90s.wav itself (the file
#      being transcribed).  For TTS posts, we call /api/tts/synthesize against
#      the same server with the same German voice + excerpt to produce a clean
#      WAV (no browser-output capture needed).
#   5. ffmpeg transcodes webm→mp4 AND mixes the audio at the right offset.
#
# Output:
#   docs/blog/<slug>/video.mp4

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BLOG_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_ROOT="$(cd "$BLOG_DIR/../.." && pwd)"
SHARED_MEDIA="$BLOG_DIR/_shared/media"

POST=""
while [ $# -gt 0 ]; do
  case "$1" in
    --post)
      POST="$2"; shift 2 ;;
    -h|--help)
      sed -n '3,30p' "$0"; exit 0 ;;
    *)
      echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [ -z "$POST" ]; then
  echo "ERROR: --post <slug> is required (run with --help)" >&2
  exit 2
fi

case "$POST" in
  voxtral-asr-vllm)  SPEC="record-asr-demo.spec.ts"; BACKEND="vllm"; DEFAULT_PORT=8090; KIND=asr ;;
  voxtral-asr-burn)  SPEC="record-asr-demo.spec.ts"; BACKEND="burn"; DEFAULT_PORT=8091; KIND=asr ;;
  voxtral-tts-vllm)  SPEC="record-tts-demo.spec.ts"; BACKEND="vllm"; DEFAULT_PORT=8090; KIND=tts ;;
  voxtral-tts-burn)  SPEC="record-tts-demo.spec.ts"; BACKEND="burn"; DEFAULT_PORT=8091; KIND=tts ;;
  *)
    echo "ERROR: unknown post slug \"$POST\"" >&2
    echo "Valid: voxtral-asr-vllm | voxtral-asr-burn | voxtral-tts-vllm | voxtral-tts-burn" >&2
    exit 2 ;;
esac

# BASE_URL env overrides the default port (e.g. production on :80).
if [ -n "${BASE_URL:-}" ]; then
  BASE_URL_FINAL="$BASE_URL"
else
  BASE_URL_FINAL="http://localhost:$DEFAULT_PORT"
fi

OUT_DIR="$BLOG_DIR/$POST"
OUT_MP4="$OUT_DIR/video.mp4"
WORK_DIR="$(mktemp -d)"
trap "rm -rf $WORK_DIR" EXIT
AUDIO_WAV="$WORK_DIR/audio.wav"
VIDEO_NOAUDIO_MP4="$WORK_DIR/video-noaudio.mp4"

# Playwright config has outputDir = path.resolve(SCRIPT_DIR/../test-results)
TEST_RESULTS="$SCRIPT_DIR/../test-results"
STDOUT_LOG="$WORK_DIR/playwright.stdout"

echo "=== Starling blog series — video recorder ==="
echo "Post     : $POST"
echo "Spec     : $SPEC"
echo "Backend  : $BACKEND ($KIND)"
echo "Base URL : $BASE_URL_FINAL"
echo "Output   : $OUT_MP4"
echo

# 1. Health-check the target server.
if ! curl -sf --max-time 3 "$BASE_URL_FINAL/api/config" >/dev/null 2>&1; then
  echo "ERROR: no response from $BASE_URL_FINAL/api/config" >&2
  echo "       Start the server first — see CLAUDE.md for the right commands." >&2
  exit 3
fi
echo "[1/6] Server is up at $BASE_URL_FINAL"

# 2. For TTS, synthesise the demo audio NOW (before Playwright runs). This
# doubles as a warm-up AND gives us the audio bytes for muxing. The browser's
# in-spec Generate click later may or may not produce real audio — we don't
# care for the muxed track, only for the visual.
#
# For ASR, the audio track is broadcast_1_90s.wav itself; just ping /api/models
# to confirm the runtime is reachable.
PREGEN_WAV="$WORK_DIR/pregen.wav"
echo "[2/6] Pre-warming $KIND on $BACKEND..."
if [ "$KIND" = "tts" ]; then
  EXCERPT_TXT="$SHARED_MEDIA/broadcast_1_excerpt.txt"
  TEXT="$(tr '\n' ' ' < "$EXCERPT_TXT" | sed -e 's/  */ /g' -e 's/^ //; s/ $//')"
  PAYLOAD="$(python3 -c '
import json, sys
text = sys.argv[1]
print(json.dumps({"text": text, "voice": "de_male", "save": False}))
' "$TEXT")"
  echo "       Synthesising warm-up + demo audio (de_male, ~150 words DE)..."
  PREWARM_RESP=$(curl -sf -X POST "$BASE_URL_FINAL/api/tts/synthesize" \
    -H 'content-type: application/json' \
    --data "$PAYLOAD" \
    --max-time 600 \
    -o "$PREGEN_WAV" \
    -w "HTTP %{http_code} bytes=%{size_download} time=%{time_total}s\n" 2>&1) || true
  echo "       $PREWARM_RESP"

  if [ -s "$PREGEN_WAV" ]; then
    AUDIO_DUR=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$PREGEN_WAV" 2>/dev/null || echo "0")
    echo "       Captured ${AUDIO_DUR}s of audio for muxing"
  fi
  # Wait until status reports ready (mostly for the rare cold-start case).
  for i in $(seq 1 30); do
    STATE=$(curl -sf "$BASE_URL_FINAL/api/tts/status" 2>/dev/null \
      | sed -n 's/.*"state":"\([^"]*\)".*/\1/p' | head -1)
    if [ "$STATE" = "ready" ]; then
      echo "       TTS lifecycle: ready"
      break
    fi
    sleep 1
  done
else
  curl -sf --max-time 5 "$BASE_URL_FINAL/api/models" >/dev/null 2>&1 || true
  echo "       ASR: model registry reachable"
fi

# 3. Clear previous test-results so we know which webm is new.
rm -rf "$TEST_RESULTS"
mkdir -p "$TEST_RESULTS"

# 4. Run the Playwright spec.
echo "[3/6] Running Playwright spec ($SPEC, BACKEND=$BACKEND)..."
pushd "$SCRIPT_DIR" >/dev/null
PLAYWRIGHT_CLI="$SCRIPT_DIR/node_modules/@playwright/test/cli.js"
if [ ! -f "$PLAYWRIGHT_CLI" ]; then
  echo "ERROR: $PLAYWRIGHT_CLI not found — run \`bun install\` in $SCRIPT_DIR" >&2
  exit 4
fi
NODE_BIN=""
for candidate in "$HOME/.local/node/bin/node" "$(command -v node 2>/dev/null || true)"; do
  if [ -n "$candidate" ] && [ -x "$candidate" ] && [ "$($candidate --version 2>/dev/null | head -1)" != "" ]; then
    if [ "$($candidate -e 'console.log(typeof Bun)' 2>/dev/null)" = "undefined" ]; then
      NODE_BIN="$candidate"
      break
    fi
  fi
done
if [ -z "$NODE_BIN" ]; then
  echo "ERROR: no real Node.js found (only bun-bundled wrapper detected)." >&2
  exit 4
fi
echo "       Using node: $NODE_BIN ($($NODE_BIN --version))"
# Don't fail the whole pipeline if Playwright assertions fail — Playwright
# still saves the video.webm under test-results/, and for the burn-server TTS
# demo specifically the in-browser Generate click may not produce real audio
# (a known CUDA quirk after the prewarm). The audio bytes for muxing come from
# the prewarm WAV anyway.
set +e
BACKEND="$BACKEND" BASE_URL="$BASE_URL_FINAL" \
  NODE_PATH="$SCRIPT_DIR/node_modules" \
  "$NODE_BIN" "$PLAYWRIGHT_CLI" test --config=playwright.config.ts "$SPEC" --reporter=list \
  2>&1 | tee "$STDOUT_LOG"
PLAYWRIGHT_EXIT=${PIPESTATUS[0]}
set -e
if [ "$PLAYWRIGHT_EXIT" -ne 0 ]; then
  echo "       Playwright exited $PLAYWRIGHT_EXIT — continuing with the captured webm."
fi
popd >/dev/null

# 5. Find the newest .webm and capture the audio-start marker.
WEBM="$(find "$TEST_RESULTS" -name '*.webm' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)"
if [ -z "$WEBM" ] || [ ! -f "$WEBM" ]; then
  echo "ERROR: no .webm produced under $TEST_RESULTS" >&2
  exit 5
fi
echo "[4/6] WebM: $WEBM"

AUDIO_START_MS="$(grep -oE 'BLOG_VIDEO_AUDIO_START_MS\] [0-9]+' "$STDOUT_LOG" | tail -1 | awk '{print $NF}')"
if [ -z "$AUDIO_START_MS" ]; then
  echo "       WARNING: no audio_start marker found, defaulting to 0 ms"
  AUDIO_START_MS=0
fi
# Add a small video-pipeline offset: Playwright starts the video before the
# spec's first action by roughly 150 ms.
AUDIO_START_MS=$(( AUDIO_START_MS + 150 ))
AUDIO_START_S=$(awk -v ms="$AUDIO_START_MS" 'BEGIN{printf "%.3f", ms/1000.0}')
echo "       Audio start offset: ${AUDIO_START_S}s (from spec marker)"

# 6. Prepare the audio track.
if [ "$KIND" = "asr" ]; then
  SRC_WAV="$SHARED_MEDIA/broadcast_1_90s.wav"
  if [ ! -f "$SRC_WAV" ]; then
    echo "ERROR: missing $SRC_WAV — run the ffmpeg trim step from CLAUDE.md" >&2
    exit 6
  fi
  cp "$SRC_WAV" "$AUDIO_WAV"
  echo "[5/6] ASR audio = broadcast_1_90s.wav (90 s)"
else
  # We synth'd into $PREGEN_WAV during step [2/6]. If that file is missing
  # or only contains the 44-byte streaming WAV header (i.e. the backend
  # crashed mid-stream), fall back to the vllm-omni server at $TTS_FALLBACK_URL
  # so the muxed audio still matches the on-screen content.
  FILE_SIZE=$(stat -c '%s' "$PREGEN_WAV" 2>/dev/null || echo 0)
  if [ -s "$PREGEN_WAV" ] && [ "$FILE_SIZE" -gt 1024 ]; then
    cp "$PREGEN_WAV" "$AUDIO_WAV"
    AUDIO_DUR=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$AUDIO_WAV" 2>/dev/null || echo "0")
    echo "[5/6] TTS audio reused from prewarm: ${AUDIO_DUR}s (${FILE_SIZE} B)"
  else
    FB_URL="${TTS_FALLBACK_URL:-http://localhost:80}"
    echo "[5/6] Prewarm produced only ${FILE_SIZE} B — falling back to $FB_URL"
    EXCERPT_TXT="$SHARED_MEDIA/broadcast_1_excerpt.txt"
    TEXT="$(tr '\n' ' ' < "$EXCERPT_TXT" | sed -e 's/  */ /g' -e 's/^ //; s/ $//')"
    PAYLOAD="$(python3 -c '
import json, sys
text = sys.argv[1]
print(json.dumps({"text": text, "voice": "de_male", "save": False}))
' "$TEXT")"
    curl -sf -X POST "$FB_URL/api/tts/synthesize" \
      -H 'content-type: application/json' \
      --data "$PAYLOAD" \
      --max-time 600 \
      -o "$AUDIO_WAV"
    AUDIO_DUR=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$AUDIO_WAV" 2>/dev/null || echo "0")
    echo "       Fallback synthesised ${AUDIO_DUR}s of audio"
  fi
fi

# 7. Transcode the video, then mux audio.
mkdir -p "$OUT_DIR"
ffmpeg -y -i "$WEBM" \
  -c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p \
  -movflags +faststart \
  -an \
  "$VIDEO_NOAUDIO_MP4" 2>&1 | tail -3

VIDEO_DUR=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$VIDEO_NOAUDIO_MP4")
echo "       Video (pre-mux): ${VIDEO_DUR}s"

# Generate padded audio: silence up to AUDIO_START_S, then the actual audio,
# then trailing silence padding up to the full video duration.  apad without
# args pads with infinite silence; -t clamps to VIDEO_DUR so the final audio
# track is exactly the video length.  This avoids -itsoffset corner cases.
ffmpeg -y \
  -f lavfi -i "anullsrc=channel_layout=mono:sample_rate=24000" \
  -i "$AUDIO_WAV" \
  -filter_complex "[0:a]atrim=0:${AUDIO_START_S},asetpts=PTS-STARTPTS[silence];[silence][1:a]concat=n=2:v=0:a=1,apad[a]" \
  -map "[a]" -t "$VIDEO_DUR" \
  -ac 2 -ar 48000 -c:a aac -b:a 160k \
  "$WORK_DIR/audio-final.aac" 2>&1 | tail -3

ffmpeg -y -i "$VIDEO_NOAUDIO_MP4" -i "$WORK_DIR/audio-final.aac" \
  -c:v copy -c:a copy \
  -map 0:v -map 1:a \
  -movflags +faststart \
  "$OUT_MP4" 2>&1 | tail -3

FINAL_DUR=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$OUT_MP4")
FINAL_HAS_AUDIO=$(ffprobe -v error -select_streams a -show_entries stream=codec_type -of default=noprint_wrappers=1:nokey=1 "$OUT_MP4")
FINAL_SIZE=$(ls -lh "$OUT_MP4" | awk '{print $5}')
echo "[6/6] MP4 written: $OUT_MP4"
echo "       Duration: ${FINAL_DUR}s   Size: ${FINAL_SIZE}   Audio: ${FINAL_HAS_AUDIO}"

DUR_INT=${FINAL_DUR%.*}
if [ "${DUR_INT:-0}" -gt 125 ]; then
  echo "WARNING: video is longer than 125 s. Tighten the scenes." >&2
fi

echo
echo "Done. Upload to YouTube and update the post when ready."
