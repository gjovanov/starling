#!/bin/bash
#
# Voxtral Server Initialization Script
# Sets up Python venv, vLLM with CUDA, voxtral-server deps, and downloads the model.
# Idempotent — safe to re-run; skips steps that are already done.
#
# Usage:
#   ./init.sh              # Full setup (venv + vLLM + voxtral-server + model download)
#   ./init.sh --no-model   # Skip model download (if you already have it cached)
#
# Prerequisites:
#   - Python 3.10+
#   - NVIDIA GPU with ≥16GB VRAM
#   - CUDA toolkit (nvcc)
#   - FFmpeg
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$SCRIPT_DIR/.venv"
MODEL_ID="mistralai/Voxtral-Mini-4B-Realtime-2602"

# Parse args
SKIP_MODEL=false
for arg in "$@"; do
    case "$arg" in
        --no-model) SKIP_MODEL=true ;;
    esac
done

echo "========================================"
echo "  Voxtral Server Initialization"
echo "========================================"
echo ""

# ─── 1. Check prerequisites ─────────────────────────────────────────

check_prerequisites() {
    echo "[1/6] Checking prerequisites..."
    local ok=true

    # Python
    if command -v python3 &>/dev/null; then
        local pyver
        pyver=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        echo "  [OK] Python $pyver"
        # Check version >= 3.10
        local major minor
        major=$(echo "$pyver" | cut -d. -f1)
        minor=$(echo "$pyver" | cut -d. -f2)
        if [ "$major" -lt 3 ] || { [ "$major" -eq 3 ] && [ "$minor" -lt 10 ]; }; then
            echo "  [FAIL] Python 3.10+ required (found $pyver)"
            ok=false
        fi
    else
        echo "  [FAIL] python3 not found"
        ok=false
    fi

    # Python dev headers (needed by Triton/vLLM for CUDA kernel compilation)
    if python3 -c "import sysconfig; p=sysconfig.get_path('include'); import os; assert os.path.exists(os.path.join(p,'Python.h'))" 2>/dev/null; then
        echo "  [OK] Python dev headers"
    else
        echo "  [FAIL] Python dev headers not found"
        echo "         Install with: sudo apt install python3-dev"
        ok=false
    fi

    # NVIDIA GPU
    if command -v nvidia-smi &>/dev/null; then
        local gpu_name gpu_mem
        gpu_name=$(timeout 5 nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "unknown")
        gpu_mem=$(timeout 5 nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null || echo "0")
        # Strip whitespace for numeric comparison
        gpu_mem_num="${gpu_mem%% *}"
        gpu_mem_num="${gpu_mem_num// /}"

        if [ "$gpu_mem_num" -ge 16000 ] 2>/dev/null; then
            echo "  [OK] GPU: $gpu_name (${gpu_mem} MiB)"
        elif [ "$gpu_mem_num" -gt 0 ] 2>/dev/null; then
            echo "  [FAIL] GPU: $gpu_name (${gpu_mem} MiB)"
            echo ""
            echo "  ┌──────────────────────────────────────────────────────────────┐"
            echo "  │  Voxtral-Mini-4B-Realtime requires ≥16GB GPU VRAM.          │"
            echo "  │                                                              │"
            echo "  │  Model: 4.4B params in BF16 = ~9GB weights + KV cache       │"
            echo "  │  Your GPU: ${gpu_name}"
            printf "  │  VRAM: %-5s MiB (need ≥16,384 MiB)                        │\n" "$gpu_mem_num"
            echo "  │                                                              │"
            echo "  │  Compatible GPUs:                                            │"
            echo "  │    • RTX 4080/4090/5080/5090 (16–24 GB)                      │"
            echo "  │    • A10, L4 (24 GB)                                         │"
            echo "  │    • A100, H100 (40–80 GB)                                   │"
            echo "  │    • T4 (16 GB — tight, may need --max-model-len 32768)      │"
            echo "  └──────────────────────────────────────────────────────────────┘"
            echo ""
            ok=false
        else
            echo "  [WARN] GPU: $gpu_name (could not read VRAM)"
        fi
    else
        echo "  [FAIL] nvidia-smi not found"
        echo ""
        echo "  ┌──────────────────────────────────────────────────────────────┐"
        echo "  │  Voxtral-Mini-4B-Realtime requires an NVIDIA GPU with      │"
        echo "  │  ≥16GB VRAM and CUDA support.                              │"
        echo "  │                                                              │"
        echo "  │  Install NVIDIA drivers:                                     │"
        echo "  │    sudo apt install nvidia-driver-550                        │"
        echo "  │                                                              │"
        echo "  │  Minimum GPUs: RTX 4080, T4, A10, L4, A100                  │"
        echo "  └──────────────────────────────────────────────────────────────┘"
        echo ""
        ok=false
    fi

    # CUDA toolkit
    if command -v nvcc &>/dev/null; then
        local cuda_ver
        cuda_ver=$(nvcc --version 2>/dev/null | grep "release" | sed 's/.*release //' | cut -d, -f1)
        echo "  [OK] CUDA toolkit $cuda_ver"
    else
        echo "  [WARN] nvcc not found — vLLM may still work with driver CUDA"
    fi

    # FFmpeg
    if command -v ffmpeg &>/dev/null; then
        echo "  [OK] FFmpeg"
    else
        echo "  [FAIL] FFmpeg not found — install with: sudo apt install ffmpeg"
        ok=false
    fi

    if [ "$ok" = false ]; then
        echo ""
        echo "  [ERROR] Prerequisites not met. Fix the above issues and re-run."
        exit 1
    fi
}

# ─── 2. Create Python virtual environment ────────────────────────────

setup_venv() {
    if [ -f "$VENV_DIR/bin/activate" ]; then
        echo "[2/6] Python venv already exists at $VENV_DIR"
    else
        # Remove broken venv if it exists without activate
        if [ -d "$VENV_DIR" ]; then
            echo "[2/6] Removing broken venv..."
            rm -rf "$VENV_DIR"
        fi

        echo "[2/6] Creating Python venv at $VENV_DIR..."

        # Try with ensurepip first, fall back to without + manual pip install
        if python3 -m venv "$VENV_DIR" 2>/dev/null; then
            echo "  [DONE] Virtual environment created"
        else
            echo "  [INFO] ensurepip not available, creating venv without pip..."
            python3 -m venv --without-pip "$VENV_DIR"
            # Install pip manually via get-pip.py
            curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
            "$VENV_DIR/bin/python" /tmp/get-pip.py --quiet
            rm -f /tmp/get-pip.py
            echo "  [DONE] Virtual environment created (with manual pip)"
        fi
    fi

    # Activate
    source "$VENV_DIR/bin/activate"
    echo "  Python: $(python --version) ($(which python))"

    # Upgrade pip
    pip install --quiet --upgrade pip
}

# ─── 3. Install PyTorch with CUDA ────────────────────────────────────

install_pytorch() {
    # Check if torch is already installed with CUDA
    if python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'" 2>/dev/null; then
        local torch_ver
        torch_ver=$(python -c "import torch; print(torch.__version__)")
        echo "[3/6] PyTorch $torch_ver with CUDA already installed"
        return 0
    fi

    echo "[3/6] Installing PyTorch with CUDA support..."

    # Detect best CUDA version for torch
    # PyTorch supports cu118, cu121, cu124, cu126
    local cuda_ver=""
    if [ -f "/usr/local/cuda-12.6/bin/nvcc" ]; then
        cuda_ver="cu126"
    elif [ -f "/usr/local/cuda-12.4/bin/nvcc" ]; then
        cuda_ver="cu124"
    elif [ -f "/usr/local/cuda-12.1/bin/nvcc" ] || [ -f "/usr/local/cuda-12/bin/nvcc" ]; then
        cuda_ver="cu121"
    else
        # Default to cu126 (works with driver CUDA 13.x)
        cuda_ver="cu126"
    fi

    echo "  Installing torch with $cuda_ver..."
    pip install --quiet torch --index-url "https://download.pytorch.org/whl/${cuda_ver}"

    # Verify
    if python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'" 2>/dev/null; then
        local torch_ver
        torch_ver=$(python -c "import torch; print(torch.__version__)")
        echo "  [DONE] PyTorch $torch_ver with CUDA"
    else
        echo "  [WARN] PyTorch installed but CUDA not available. vLLM may not work."
    fi
}

# ─── 4. Install vLLM ────────────────────────────────────────────────

install_vllm() {
    if python -c "import vllm; print(vllm.__version__)" 2>/dev/null; then
        local vllm_ver
        vllm_ver=$(python -c "import vllm; print(vllm.__version__)")
        echo "[4/6] vLLM $vllm_ver already installed"
        return 0
    fi

    echo "[4/6] Installing vLLM + mistral_common..."
    pip install --quiet vllm "mistral_common[soundfile]>=1.9.0"

    local vllm_ver
    vllm_ver=$(python -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "unknown")
    echo "  [DONE] vLLM $vllm_ver"
}

# ─── 5. Install voxtral-server ──────────────────────────────────────

install_voxtral_server() {
    if python -c "import voxtral_server" 2>/dev/null; then
        echo "[5/6] voxtral-server already installed"
    else
        echo "[5/6] Installing voxtral-server..."
    fi

    # Always reinstall in editable mode (fast, picks up code changes)
    pip install --quiet -e "$SCRIPT_DIR"
    echo "  [DONE] voxtral-server installed"
}

# ─── 6. Download model / Verify ─────────────────────────────────────

download_model() {
    if [ "$SKIP_MODEL" = true ]; then
        echo "[6/6] Skipping model download (--no-model)"
        return 0
    fi

    echo "[6/6] Ensuring model is cached: $MODEL_ID"

    # Check HF token
    local hf_token=""
    if [ -f "$PROJECT_DIR/.env" ]; then
        hf_token=$(grep -E "^HF_TOKEN=" "$PROJECT_DIR/.env" 2>/dev/null | cut -d'=' -f2 || echo "")
    fi
    hf_token="${hf_token:-$HF_TOKEN}"
    hf_token="${hf_token:-$HUGGING_FACE_HUB_TOKEN}"

    if [ -n "$hf_token" ]; then
        export HF_TOKEN="$hf_token"
        echo "  [INFO] Using HF token for authentication"
    fi

    # Use hf cli to download (comes with huggingface_hub)
    # "hf download" is the current command; "huggingface-cli download" is deprecated
    if command -v hf &>/dev/null; then
        echo "  Downloading via hf cli (this may take a while on first run)..."
        hf download "$MODEL_ID" --quiet 2>&1 | tail -3
        echo "  [DONE] Model cached"
    elif command -v huggingface-cli &>/dev/null; then
        echo "  Downloading via huggingface-cli (this may take a while on first run)..."
        huggingface-cli download "$MODEL_ID" --quiet 2>&1 | tail -3
        echo "  [DONE] Model cached"
    else
        # Install huggingface_hub and download
        pip install --quiet huggingface_hub
        python -c "
from huggingface_hub import snapshot_download
print('  Downloading model (this may take a while on first run)...')
path = snapshot_download('$MODEL_ID')
print(f'  [DONE] Model cached at: {path}')
"
    fi
}

verify_setup() {
    echo ""
    echo "========================================"
    echo "  Verification"
    echo "========================================"

    local ok=true

    # Python venv
    if [ -f "$VENV_DIR/bin/python" ]; then
        echo "  [OK] Python venv"
    else
        echo "  [FAIL] Python venv"
        ok=false
    fi

    # PyTorch + CUDA
    if "$VENV_DIR/bin/python" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        local torch_ver
        torch_ver=$("$VENV_DIR/bin/python" -c "import torch; print(torch.__version__)")
        echo "  [OK] PyTorch $torch_ver (CUDA)"
    else
        echo "  [FAIL] PyTorch with CUDA"
        ok=false
    fi

    # vLLM
    if "$VENV_DIR/bin/python" -c "import vllm" 2>/dev/null; then
        local vllm_ver
        vllm_ver=$("$VENV_DIR/bin/python" -c "import vllm; print(vllm.__version__)")
        echo "  [OK] vLLM $vllm_ver"
    else
        echo "  [FAIL] vLLM"
        ok=false
    fi

    # voxtral-server
    if "$VENV_DIR/bin/python" -c "import voxtral_server" 2>/dev/null; then
        echo "  [OK] voxtral-server"
    else
        echo "  [FAIL] voxtral-server"
        ok=false
    fi

    # FFmpeg
    if command -v ffmpeg &>/dev/null; then
        echo "  [OK] FFmpeg"
    else
        echo "  [FAIL] FFmpeg"
        ok=false
    fi

    # Frontend
    if [ -d "$PROJECT_DIR/frontend" ]; then
        echo "  [OK] Frontend at $PROJECT_DIR/frontend"
    else
        echo "  [WARN] Frontend not found at $PROJECT_DIR/frontend"
    fi

    # Media
    if [ -d "$PROJECT_DIR/media" ]; then
        local count
        count=$(ls "$PROJECT_DIR/media/"*.wav 2>/dev/null | wc -l)
        echo "  [OK] Media directory ($count .wav files)"
    else
        echo "  [WARN] Media directory not found at $PROJECT_DIR/media"
    fi

    # GPU info
    local gpu_name gpu_mem
    gpu_name=$(timeout 5 nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "unknown")
    gpu_mem=$(timeout 5 nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null || echo "unknown")
    echo "  [INFO] GPU: $gpu_name ($gpu_mem)"

    if [ "$ok" = false ]; then
        echo ""
        echo "  [WARN] Some checks failed — review above."
    fi
}

# ─── Main ─────────────────────────────────────────────────────────────

check_prerequisites
setup_venv
install_pytorch
install_vllm
install_voxtral_server
download_model
verify_setup

echo ""
echo "========================================"
echo "  Initialization Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo ""
echo "  1. Activate the venv:"
echo "     source $VENV_DIR/bin/activate"
echo ""
echo "  2. Start vLLM (GPU server, separate terminal):"
echo "     vllm serve $MODEL_ID --port 8001"
echo ""
echo "  3. Start voxtral-server (another terminal):"
echo "     cd $SCRIPT_DIR && ./start.sh"
echo ""
echo "  4. Open http://localhost:8090 in your browser"
echo ""
