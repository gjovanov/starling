#!/bin/bash
#
# podcast-server setup: venv + deps + model download.
#
# Usage:
#   ./init.sh              # Full setup: venv, torch (CUDA), transformers, model (~5.4 GB → HF cache)
#   ./init.sh --no-model   # Skip the model download
#   ./init.sh --no-gpu     # API + mock engine only (no torch/transformers) — tests/CI
#   ./init.sh --dry-run    # Print commands without executing
#
# Never pass --upgrade to pip here: torch comes from the CUDA wheel index and
# an upgrade resolve against PyPI can silently replace it.
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

# RTX 50xx (Blackwell, sm_120) needs CUDA >= 12.8 wheels; cu130 is the
# newest index carrying a current torch (2.14.x). Override both for other GPUs.
TORCH_VERSION="${PODCAST_TORCH_VERSION:-2.14.1}"
TORCH_INDEX="${PODCAST_TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"

SKIP_MODEL=false
SKIP_GPU=false
DRY_RUN=false
for arg in "$@"; do
    case "$arg" in
        --no-model) SKIP_MODEL=true ;;
        --no-gpu)   SKIP_GPU=true; SKIP_MODEL=true ;;
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
echo "  podcast-server init"
echo "========================================"
echo "  venv:     $VENV_DIR"
echo "  gpu deps: $([ "$SKIP_GPU" = true ] && echo skipped || echo "torch $TORCH_VERSION ($TORCH_INDEX) + transformers 5.19.0")"
echo "  model:    $([ "$SKIP_MODEL" = true ] && echo skipped || echo "~5.4 GB into the HF cache (pinned revision, see podcast_server/config.py)")"
echo ""

if ! python3 -c "import venv, ensurepip" 2>/dev/null; then
    echo "ERROR: python3 venv/ensurepip missing — sudo apt install python3-venv"
    exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "[Init] Creating venv..."
    _run "python3 -m venv '$VENV_DIR'"
fi

_run "'$VENV_DIR/bin/pip' install --upgrade pip --quiet"

if [ "$SKIP_GPU" = false ]; then
    echo "[Init] Installing torch $TORCH_VERSION from $TORCH_INDEX (large)..."
    _run "'$VENV_DIR/bin/pip' install 'torch==$TORCH_VERSION' --index-url '$TORCH_INDEX' --quiet"
    echo "[Init] Installing podcast-server (editable) + gpu + dev deps..."
    _run "'$VENV_DIR/bin/pip' install -e '$SCRIPT_DIR[gpu,dev]' --quiet"
    echo "[Init] Verifying the CUDA build of torch survived the resolve..."
    _run "'$VENV_DIR/bin/python' -c \"import torch, transformers; print('  torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available()); print('  transformers', transformers.__version__); assert torch.version.cuda, 'CPU-only torch got installed'\""
else
    echo "[Init] Installing podcast-server (editable) + dev deps (no GPU stack)..."
    _run "'$VENV_DIR/bin/pip' install -e '$SCRIPT_DIR[dev]' --quiet"
fi

if [ "$SKIP_MODEL" = false ]; then
    echo "[Init] Downloading the model into the HF cache..."
    _run "'$VENV_DIR/bin/python' -m podcast_server.download"
fi

echo ""
echo "[Init] Done. Start the server with: ./start.sh   (loopback :8093)"
echo "       Batch CLI:                    .venv/bin/podcast-tts script.txt -o out.wav"
