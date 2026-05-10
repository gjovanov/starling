#!/bin/bash
#
# Burn Server Initialization Script
# Builds the Rust binary and downloads model weights.
#
# Usage:
#   ./init.sh              # Build + download models (ASR only)
#   ./init.sh --tts        # Build with the voxtral-tts feature + download TTS model
#   ./init.sh --no-model   # Build only, skip model download
#   ./init.sh --wasm       # Also build WASM target
#   ./init.sh --tts --no-model
#                          # Combines: enable TTS routes, skip downloads
#
# Prerequisites:
#   - Rust toolchain (rustc, cargo)
#   - Run ./prerequisites.sh first if needed
#
# Notes:
#   - --tts enables the `voxtral-tts` cargo feature, which is required for
#     the /api/tts/* HTTP routes (text-to-speech). Without it, the binary
#     boots normally but the TTS routes are absent.
#   - The TTS path uses CPU + F32 by default (the configured BURN_BACKEND
#     only governs ASR). TTS works on any host with enough RAM (~12 GiB
#     resident while loaded; idle-unloads after 600 s by default).
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

SKIP_MODEL=false
BUILD_WASM=false
BUILD_TTS=false
for arg in "$@"; do
    case "$arg" in
        --no-model) SKIP_MODEL=true ;;
        --wasm)     BUILD_WASM=true ;;
        --tts)      BUILD_TTS=true ;;
    esac
done

echo "========================================"
echo "  Burn Server Initialization"
echo "========================================"
echo ""

# ─── 1. Check prerequisites ──────────────────────────────────────

echo "[1/4] Checking prerequisites..."

if ! command -v rustc &>/dev/null; then
    echo "  [FAIL] Rust not found. Run ./prerequisites.sh first."
    exit 1
fi
echo "  [OK] Rust $(rustc --version | awk '{print $2}')"

if ! command -v cargo &>/dev/null; then
    echo "  [FAIL] Cargo not found."
    exit 1
fi
echo "  [OK] Cargo"

if command -v ffmpeg &>/dev/null; then
    echo "  [OK] FFmpeg"
else
    echo "  [WARN] FFmpeg not found — media file transcription won't work"
fi

# ─── 2. Build native binary ──────────────────────────────────────

echo ""
echo "[2/4] Building burn-server (release)..."
cd "$SCRIPT_DIR"

# Load .env to detect backend and derive cargo features
[ -f "$SCRIPT_DIR/.env" ] && set -a && source "$SCRIPT_DIR/.env" && set +a

# Derive cargo features in two parts so `--tts` can compose cleanly with
# whichever ASR backend is selected. The wgpu backend is the cargo
# default (no feature flag needed); every other ASR backend pulls in
# its named feature. `--tts` appends `voxtral-tts` to the same list.
BACKEND_FEATURE=""
RUSTFLAGS_EXTRA=""
case "${BURN_BACKEND:-wgpu}" in
    candle-native-flash) BACKEND_FEATURE="candle-native-flash" ;;
    candle-native)       BACKEND_FEATURE="candle-native" ;;
    candle)              BACKEND_FEATURE="candle" ;;
    cuda)                BACKEND_FEATURE="cuda" ;;
    candle-cpu)          BACKEND_FEATURE="candle-cpu-ggml"
                          RUSTFLAGS_EXTRA="-C target-cpu=native" ;;
    candle-cpu-ggml)     BACKEND_FEATURE="candle-cpu-ggml"
                          RUSTFLAGS_EXTRA="-C target-cpu=native" ;;
esac

TTS_FEATURE=""
[ "$BUILD_TTS" = true ] && TTS_FEATURE="voxtral-tts"

# Build the combined --features arg. Cargo accepts comma-separated
# feature names inside a single --features flag.
FEATURES=""
if [ -n "$BACKEND_FEATURE" ] && [ -n "$TTS_FEATURE" ]; then
    FEATURES="--features $BACKEND_FEATURE,$TTS_FEATURE"
elif [ -n "$BACKEND_FEATURE" ]; then
    FEATURES="--features $BACKEND_FEATURE"
elif [ -n "$TTS_FEATURE" ]; then
    FEATURES="--features $TTS_FEATURE"
fi

echo "  Backend: ${BURN_BACKEND:-wgpu}, features: ${FEATURES:-default}"
echo "  TTS:     $([ "$BUILD_TTS" = true ] && echo "enabled (voxtral-tts)" || echo "disabled (pass --tts to enable)")"
if [ -n "$RUSTFLAGS_EXTRA" ]; then
    echo "  RUSTFLAGS: $RUSTFLAGS_EXTRA"
    RUSTFLAGS="$RUSTFLAGS_EXTRA" cargo build --release --bin burn-server $FEATURES 2>&1
else
    cargo build --release --bin burn-server $FEATURES 2>&1
fi

# Check workspace target first, then local
if [ -f "$PROJECT_DIR/target/release/burn-server" ]; then
    BINARY="$PROJECT_DIR/target/release/burn-server"
elif [ -f "$SCRIPT_DIR/target/release/burn-server" ]; then
    BINARY="$SCRIPT_DIR/target/release/burn-server"
fi

if [ -f "$BINARY" ]; then
    echo "  [DONE] Binary: $BINARY"
else
    echo "  [FAIL] Build failed — binary not found"
    exit 1
fi

# ─── 3. Build WASM (optional) ────────────────────────────────────

if [ "$BUILD_WASM" = true ]; then
    echo ""
    echo "[3/4] Building WASM target..."

    if ! command -v wasm-pack &>/dev/null; then
        echo "  [FAIL] wasm-pack not found. Install with: curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh"
        exit 1
    fi

    wasm-pack build --target web --features wasm --no-default-features
    echo "  [DONE] WASM build at $SCRIPT_DIR/pkg/"
else
    echo ""
    echo "[3/4] Skipping WASM build (use --wasm to enable)"
fi

# ─── 4. Download models ──────────────────────────────────────────

if [ "$SKIP_MODEL" = true ]; then
    echo ""
    echo "[4/4] Skipping model download (--no-model)"
else
    echo ""
    echo "[4/4] Downloading models..."

    DOWNLOAD_SCRIPT="$PROJECT_DIR/models/download.sh"
    if [ -x "$DOWNLOAD_SCRIPT" ]; then
        "$DOWNLOAD_SCRIPT"
    else
        echo "  [WARN] Model download script not found at $DOWNLOAD_SCRIPT"
        echo "         Run manually: cd $PROJECT_DIR/models && ./download.sh"
    fi
fi

# ─── Done ─────────────────────────────────────────────────────────

echo ""
echo "========================================"
echo "  Initialization Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo ""
echo "  1. Start the server:"
echo "     ./start.sh"
echo ""
echo "  2. Open http://localhost:8091 in your browser"
echo ""
if [ "$BUILD_TTS" = true ]; then
    echo "  TTS routes available at:"
    echo "    GET    /api/tts/voices          GET    /api/tts/config"
    echo "    GET    /api/tts/status          POST   /api/tts/synthesize"
    echo "    POST   /api/tts/voices/upload   DELETE /api/tts/voices/{ref_id}"
    echo "    GET    /api/tts/output[/{name}] DELETE /api/tts/output/{name}"
    echo ""
    echo "  TTS pipeline loads lazily on the first /api/tts/* hit (~30 s on CPU)"
    echo "  and idle-unloads after BURN_TTS_IDLE_UNLOAD_SECS (default 600 s)."
    echo ""
fi
