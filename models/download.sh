#!/bin/bash
#
# Starling Shared Model Downloader
# Downloads Voxtral-Mini-4B-Realtime model weights to the shared models/cache/ directory.
# All apps in this monorepo reference this directory to avoid duplicate downloads.
#
# Usage:
#   ./download.sh              # Download all variants (BF16 + Q4 + TTS)
#   ./download.sh --bf16-only  # Download BF16 SafeTensors only (~9GB)
#   ./download.sh --q4-only    # Download Q4 GGUF only (~2.5GB)
#   ./download.sh --tts-only   # Download Voxtral TTS only (~8GB)
#   ./download.sh --no-tts     # Skip TTS download (useful on small disks)
#   ./download.sh --chunk      # Also chunk Q4 GGUF into 64MB shards for WASM
#
# Model storage layout:
#   models/cache/
#     bf16/                    # SafeTensors — used by vllm-server & burn-server BF16 mode
#     q4/                      # GGUF Q4_0 — used by burn-server Q4 mode
#       voxtral-q4.gguf
#       chunks/                # 64MB shards for browser WASM loading
#     tts/                     # Voxtral-4B-TTS-2603 — served by vllm-server's TTS tab
#       consolidated.safetensors
#       voice_embedding/       # 20 .pt voice presets bundled by Mistral
#     tokenizer/               # Tekken tokenizer (shared)
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CACHE_DIR="$SCRIPT_DIR/cache"

BF16_MODEL_ID="mistralai/Voxtral-Mini-4B-Realtime-2602"
Q4_REPO_ID="TrevorJS/voxtral-mini-realtime-gguf"
TTS_MODEL_ID="mistralai/Voxtral-4B-TTS-2603"

# Parse args
DOWNLOAD_BF16=true
DOWNLOAD_Q4=true
DOWNLOAD_TTS=true
DO_CHUNK=false
for arg in "$@"; do
    case "$arg" in
        --bf16-only) DOWNLOAD_Q4=false; DOWNLOAD_TTS=false ;;
        --q4-only)   DOWNLOAD_BF16=false; DOWNLOAD_TTS=false ;;
        --tts-only)  DOWNLOAD_BF16=false; DOWNLOAD_Q4=false ;;
        --no-tts)    DOWNLOAD_TTS=false ;;
        --chunk)     DO_CHUNK=true ;;
    esac
done

echo "========================================"
echo "  Starling Model Downloader"
echo "========================================"
echo "  Cache: $CACHE_DIR"
echo "  BF16:  $DOWNLOAD_BF16"
echo "  Q4:    $DOWNLOAD_Q4"
echo "  TTS:   $DOWNLOAD_TTS"
echo "  Chunk: $DO_CHUNK"
echo ""

# ─── Resolve HF token ─────────────────────────────────────────────

resolve_hf_token() {
    local hf_token=""
    # Check root .env
    if [ -f "$PROJECT_DIR/.env" ]; then
        hf_token=$(grep -E "^HF_TOKEN=" "$PROJECT_DIR/.env" 2>/dev/null | cut -d'=' -f2 || echo "")
    fi
    hf_token="${hf_token:-$HF_TOKEN}"
    hf_token="${hf_token:-$HUGGING_FACE_HUB_TOKEN}"

    if [ -n "$hf_token" ]; then
        export HF_TOKEN="$hf_token"
        echo "[INFO] Using HF token for authentication"
    fi
}

# ─── Download helper ───────────────────────────────────────────────

hf_download() {
    local repo_id="$1"
    local local_dir="$2"

    mkdir -p "$local_dir"

    if command -v hf &>/dev/null; then
        hf download "$repo_id" --local-dir "$local_dir" --quiet 2>&1 | tail -3
    elif command -v huggingface-cli &>/dev/null; then
        huggingface-cli download "$repo_id" --local-dir "$local_dir" --quiet 2>&1 | tail -3
    else
        # Try with Python
        python3 -c "
from huggingface_hub import snapshot_download
path = snapshot_download('$repo_id', local_dir='$local_dir')
print(f'  Downloaded to: {path}')
" 2>/dev/null || {
            echo "  [ERROR] No download tool found. Install huggingface_hub:"
            echo "          pip install huggingface_hub"
            return 1
        }
    fi
}

# ─── BF16 SafeTensors ──────────────────────────────────────────────

download_bf16() {
    if [ "$DOWNLOAD_BF16" = false ]; then
        return 0
    fi

    local bf16_dir="$CACHE_DIR/bf16"

    if [ -f "$bf16_dir/consolidated.safetensors" ]; then
        echo "[BF16] Already downloaded at $bf16_dir"
        return 0
    fi

    echo "[BF16] Downloading $BF16_MODEL_ID (~9GB)..."
    hf_download "$BF16_MODEL_ID" "$bf16_dir"
    echo "[BF16] Done."
}

# ─── TTS BF16 SafeTensors ──────────────────────────────────────────

download_tts() {
    if [ "$DOWNLOAD_TTS" = false ]; then
        return 0
    fi

    local tts_dir="$CACHE_DIR/tts"

    if [ -f "$tts_dir/consolidated.safetensors" ]; then
        echo "[TTS]  Already downloaded at $tts_dir"
        return 0
    fi

    echo "[TTS]  Downloading $TTS_MODEL_ID (~8GB, includes voice_embedding/)..."
    hf_download "$TTS_MODEL_ID" "$tts_dir"
    echo "[TTS]  Done."
}

# ─── Q4 GGUF ───────────────────────────────────────────────────────

download_q4() {
    if [ "$DOWNLOAD_Q4" = false ]; then
        return 0
    fi

    local q4_dir="$CACHE_DIR/q4"

    if [ -f "$q4_dir/voxtral-q4.gguf" ]; then
        echo "[Q4]   Already downloaded at $q4_dir"
    else
        echo "[Q4]   Downloading $Q4_REPO_ID (~2.5GB)..."
        hf_download "$Q4_REPO_ID" "$q4_dir"
        echo "[Q4]   Done."
    fi

    # Copy tokenizer to shared location
    local tok_dir="$CACHE_DIR/tokenizer"
    if [ -f "$q4_dir/tekken.json" ] && [ ! -f "$tok_dir/tekken.json" ]; then
        mkdir -p "$tok_dir"
        cp "$q4_dir/tekken.json" "$tok_dir/tekken.json"
        echo "[TOK]  Tokenizer copied to $tok_dir"
    fi
}

# ─── Chunk GGUF for WASM ──────────────────────────────────────────

chunk_gguf() {
    if [ "$DO_CHUNK" = false ]; then
        return 0
    fi

    local gguf_file="$CACHE_DIR/q4/voxtral-q4.gguf"
    local chunk_dir="$CACHE_DIR/q4/chunks"

    if [ ! -f "$gguf_file" ]; then
        echo "[CHUNK] No GGUF file found at $gguf_file — download Q4 first"
        return 1
    fi

    if [ -d "$chunk_dir" ] && [ "$(ls "$chunk_dir/" 2>/dev/null | wc -l)" -gt 0 ]; then
        echo "[CHUNK] Chunks already exist at $chunk_dir"
        return 0
    fi

    echo "[CHUNK] Splitting GGUF into 64MB chunks..."
    mkdir -p "$chunk_dir"
    split -b 64m "$gguf_file" "$chunk_dir/chunk_"
    local count
    count=$(ls "$chunk_dir/" | wc -l)
    echo "[CHUNK] Created $count chunks in $chunk_dir"
}

# ─── Main ──────────────────────────────────────────────────────────

resolve_hf_token
download_bf16
download_q4
download_tts
chunk_gguf

echo ""
echo "========================================"
echo "  Model Download Complete"
echo "========================================"
echo ""
ls -lh "$CACHE_DIR/" 2>/dev/null || echo "  (empty)"
echo ""
