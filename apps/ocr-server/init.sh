#!/bin/bash
#
# ocr-server setup: venv + deps + model download.
#
# Usage:
#   ./init.sh              # Full setup (venv, deps incl. vLLM + glmocr, models ~11GB)
#   ./init.sh --no-model   # Skip the model download
#   ./init.sh --no-gpu     # Skip GPU deps (vLLM, glmocr) — mock engine / CI setup
#   ./init.sh --dry-run    # Print commands without executing
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

SKIP_MODEL=false
SKIP_GPU=false
DRY_RUN=false
for arg in "$@"; do
    case "$arg" in
        --no-model) SKIP_MODEL=true ;;
        --no-gpu)   SKIP_GPU=true ;;
        --dry-run)  DRY_RUN=true ;;
    esac
done

_run() {
    if [ "$DRY_RUN" = true ]; then
        printf "  [DRY-RUN] %s\n" "$*"
    else
        eval "$@"
    fi
}

echo "========================================"
echo "  ocr-server init"
echo "========================================"
echo "  venv:     $VENV_DIR"
echo "  gpu deps: $([ "$SKIP_GPU" = true ] && echo skipped || echo included)"
echo "  models:   $([ "$SKIP_MODEL" = true ] && echo skipped || echo "~11GB via models/download.sh --ocr-only")"
echo ""

bash "$SCRIPT_DIR/prerequisites.sh" || true
echo ""

if [ ! -d "$VENV_DIR" ]; then
    echo "[Init] Creating venv..."
    _run "python3 -m venv '$VENV_DIR'"
fi

echo "[Init] Installing ocr-server (editable) + dev deps..."
_run "'$VENV_DIR/bin/pip' install --upgrade pip --quiet"
_run "'$VENV_DIR/bin/pip' install -e '$SCRIPT_DIR[dev]' --quiet"

if [ "$SKIP_GPU" = false ]; then
    echo "[Init] Installing GPU inference deps (vLLM + glmocr SDK) — this is large..."
    _run "'$VENV_DIR/bin/pip' install 'vllm>=0.19.0' --quiet"
    _run "'$VENV_DIR/bin/pip' install 'glmocr[selfhosted]' --quiet"
fi

if [ "$SKIP_MODEL" = false ]; then
    echo "[Init] Downloading OCR models (GLM-OCR + Qwen3-8B-FP8)..."
    _run "bash '$PROJECT_DIR/models/download.sh' --ocr-only"
fi

echo ""
echo "[Init] Done. Start the server with: ./start.sh"
