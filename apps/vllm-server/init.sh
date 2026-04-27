#!/bin/bash
#
# Voxtral Server Initialization Script
# Sets up Python venv, vLLM with CUDA, voxtral-server deps, and downloads the model.
# Idempotent — safe to re-run; skips steps that are already done.
#
# Usage:
#   ./init.sh              # ASR-only setup (venv + vLLM + voxtral-server + BF16 model)
#   ./init.sh --tts        # Also install vllm-omni + Voxtral-4B-TTS for the TTS tab
#   ./init.sh --no-model   # Skip model download (if you already have it cached)
#   ./init.sh --dry-run    # Print every install command without executing it (used by tests)
#
# Env-var equivalents:
#   VOXTRAL_INSTALL_TTS=1  same as --tts
#
# Prerequisites:
#   - Python 3.10+
#   - NVIDIA GPU with ≥16GB VRAM
#   - CUDA toolkit (nvcc)
#   - FFmpeg
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
MODELS_DIR="${STARLING_MODELS_DIR:-$PROJECT_DIR/models/cache}"
MODEL_ID="mistralai/Voxtral-Mini-4B-Realtime-2602"

# Parse args
SKIP_MODEL=false
DRY_RUN=false
INSTALL_TTS="${VOXTRAL_INSTALL_TTS:-false}"
for arg in "$@"; do
    case "$arg" in
        --no-model) SKIP_MODEL=true ;;
        --tts)      INSTALL_TTS=true ;;
        --dry-run)  DRY_RUN=true ;;
    esac
done

# Normalise the env-var form to a strict boolean
case "$INSTALL_TTS" in
    1|true|TRUE|yes|YES|y|Y) INSTALL_TTS=true ;;
    *) INSTALL_TTS=false ;;
esac

# Step counts (we add one extra step when TTS is requested)
TOTAL_STEPS=6
[ "$INSTALL_TTS" = true ] && TOTAL_STEPS=7

# `_run` is the canonical wrapper for any side-effecting command.
# In dry-run mode it prints the command instead of executing it; otherwise
# it evals it. Use `_run cmd args…` (NOT inside another quoted string).
_run() {
    if [ "$DRY_RUN" = true ]; then
        printf "  [DRY-RUN] %s\n" "$*"
    else
        eval "$@"
    fi
}

echo "========================================"
echo "  Voxtral Server Initialization"
echo "========================================"
echo "  TTS support:  $INSTALL_TTS"
echo "  Skip model:   $SKIP_MODEL"
echo "  Dry-run:      $DRY_RUN"
echo ""

# ─── 1. Check prerequisites ─────────────────────────────────────────

check_prerequisites() {
    echo "[1/${TOTAL_STEPS}] Checking prerequisites..."
    if [ "$DRY_RUN" = true ]; then
        echo "  [DRY-RUN] (skipping host-environment checks)"
        return 0
    fi
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
    if [ "$DRY_RUN" = true ]; then
        echo "[2/${TOTAL_STEPS}] Create Python venv at $VENV_DIR (dry-run)"
        echo "  [DRY-RUN] python3 -m venv $VENV_DIR"
        return 0
    fi

    if [ -f "$VENV_DIR/bin/activate" ]; then
        echo "[2/${TOTAL_STEPS}] Python venv already exists at $VENV_DIR"
    else
        # Remove broken venv if it exists without activate
        if [ -d "$VENV_DIR" ]; then
            echo "[2/${TOTAL_STEPS}] Removing broken venv..."
            rm -rf "$VENV_DIR"
        fi

        echo "[2/${TOTAL_STEPS}] Creating Python venv at $VENV_DIR..."

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
    if [ "$DRY_RUN" = false ] && python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'" 2>/dev/null; then
        local torch_ver
        torch_ver=$(python -c "import torch; print(torch.__version__)")
        echo "[3/${TOTAL_STEPS}] PyTorch $torch_ver with CUDA already installed"
        return 0
    fi

    echo "[3/${TOTAL_STEPS}] Installing PyTorch with CUDA support..."

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
    _run pip install --quiet torch --index-url "https://download.pytorch.org/whl/${cuda_ver}"

    # Verify (only when actually installed)
    if [ "$DRY_RUN" = false ]; then
        if python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'" 2>/dev/null; then
            local torch_ver
            torch_ver=$(python -c "import torch; print(torch.__version__)")
            echo "  [DONE] PyTorch $torch_ver with CUDA"
        else
            echo "  [WARN] PyTorch installed but CUDA not available. vLLM may not work."
        fi
    fi
}

# ─── 4. Install vLLM ────────────────────────────────────────────────

install_vllm() {
    if [ "$DRY_RUN" = false ] && python -c "import vllm; print(vllm.__version__)" 2>/dev/null; then
        local vllm_ver
        vllm_ver=$(python -c "import vllm; print(vllm.__version__)")
        echo "[4/${TOTAL_STEPS}] vLLM $vllm_ver already installed"
        return 0
    fi

    echo "[4/${TOTAL_STEPS}] Installing vLLM + mistral_common..."
    _run pip install --quiet vllm 'mistral_common[soundfile]>=1.9.0'

    if [ "$DRY_RUN" = false ]; then
        local vllm_ver
        vllm_ver=$(python -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "unknown")
        echo "  [DONE] vLLM $vllm_ver"
    fi
}

# ─── 5. (optional) Install vllm-omni for the TTS tab ──────────────────
#
# vllm-omni MUST be installed AFTER vllm. Order is enforced by the
# install_vllm() step running first; this step asserts vllm is importable
# before pulling in vllm-omni. Do NOT pass --upgrade here — it pulls
# torch 2.11 which breaks vllm's compiled extensions (HF discussion #29).

install_vllm_omni() {
    if [ "$INSTALL_TTS" != true ]; then
        # Not a numbered step when skipped — keeps slot 5 reserved for
        # voxtral-server in the no-TTS flow.
        echo "      vllm-omni: skipped (re-run with --tts to install)"
        return 0
    fi

    if [ "$DRY_RUN" = false ] && python -c "import vllm_omni" 2>/dev/null; then
        echo "[5/${TOTAL_STEPS}] vllm-omni already installed"
        # Even when present, run the integrity check — torch may have been
        # bumped by some unrelated install since.
        _run_tts_integrity_check
        return 0
    fi

    echo "[5/${TOTAL_STEPS}] Installing vllm-omni (TTS engine)..."
    echo "  Note: pulls gradio + diffusers + librosa (~3 GB extra)."

    # Pre-condition: vllm must already be present (otherwise vllm-omni's
    # compiled extensions look up the wrong torch ABI and fail to load).
    if [ "$DRY_RUN" = false ] && ! python -c "import vllm" 2>/dev/null; then
        echo "  [FAIL] vllm is not importable — install_vllm() must run first"
        exit 1
    fi

    # Capture torch version before the install so we can detect a bump.
    local torch_before=""
    if [ "$DRY_RUN" = false ]; then
        torch_before=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "unknown")
    fi

    # IMPORTANT: do NOT pass --upgrade — see file header comment.
    _run pip install --quiet vllm-omni

    if [ "$DRY_RUN" = false ]; then
        local torch_after
        torch_after=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "unknown")
        if [ "$torch_before" != "$torch_after" ]; then
            echo "  [WARN] torch was changed during install: $torch_before → $torch_after"
            echo "         Run scripts/check_tts_install.py manually to verify."
        else
            echo "  [OK] torch unchanged ($torch_after)"
        fi
        _run_tts_integrity_check
        echo "  [DONE] vllm-omni installed"
    fi
}

_run_tts_integrity_check() {
    if [ -x "$SCRIPT_DIR/scripts/check_tts_install.py" ] || [ -f "$SCRIPT_DIR/scripts/check_tts_install.py" ]; then
        python "$SCRIPT_DIR/scripts/check_tts_install.py" || {
            echo "  [FAIL] vllm-omni integrity check failed (see above)"
            exit 1
        }
    fi
}

# ─── 5. Install voxtral-server ──────────────────────────────────────

install_voxtral_server() {
    # Step number: 5 when no TTS, 6 when TTS (since vllm-omni took slot 5).
    local step=5
    [ "$INSTALL_TTS" = true ] && step=6

    if [ "$DRY_RUN" = false ] && python -c "import voxtral_server" 2>/dev/null; then
        echo "[${step}/${TOTAL_STEPS}] voxtral-server already installed"
    else
        echo "[${step}/${TOTAL_STEPS}] Installing voxtral-server..."
    fi

    # Always reinstall in editable mode (fast, picks up code changes)
    _run pip install --quiet -e "$SCRIPT_DIR"
    if [ "$DRY_RUN" = false ]; then
        echo "  [DONE] voxtral-server installed"
    fi
}

# ─── 6. Download model / Verify ─────────────────────────────────────

download_model() {
    # Step number: 6 when no TTS, 7 when TTS (vllm-omni in slot 5 pushed +1).
    local step=6
    [ "$INSTALL_TTS" = true ] && step=7

    if [ "$SKIP_MODEL" = true ]; then
        echo "[${step}/${TOTAL_STEPS}] Skipping model download (--no-model)"
        return 0
    fi

    echo "[${step}/${TOTAL_STEPS}] Ensuring model(s) are cached..."

    # Use shared model download script if available
    local download_script="$PROJECT_DIR/models/download.sh"
    if [ -x "$download_script" ]; then
        echo "  Using shared model downloader..."
        _run "$download_script" --bf16-only
        if [ "$INSTALL_TTS" = true ]; then
            _run "$download_script" --tts-only
        fi
        return 0
    fi

    # Fallback: download directly
    # Check HF token from app .env, root .env, or environment
    local hf_token=""
    if [ -f "$SCRIPT_DIR/.env" ]; then
        hf_token=$(grep -E "^HF_TOKEN=" "$SCRIPT_DIR/.env" 2>/dev/null | cut -d'=' -f2 || echo "")
    fi
    if [ -z "$hf_token" ] && [ -f "$PROJECT_DIR/.env" ]; then
        hf_token=$(grep -E "^HF_TOKEN=" "$PROJECT_DIR/.env" 2>/dev/null | cut -d'=' -f2 || echo "")
    fi
    hf_token="${hf_token:-$HF_TOKEN}"
    hf_token="${hf_token:-$HUGGING_FACE_HUB_TOKEN}"

    if [ -n "$hf_token" ]; then
        export HF_TOKEN="$hf_token"
        echo "  [INFO] Using HF token for authentication"
    fi

    # Download to shared models directory
    mkdir -p "$MODELS_DIR/bf16"

    if command -v hf &>/dev/null; then
        echo "  Downloading via hf cli (this may take a while on first run)..."
        hf download "$MODEL_ID" --local-dir "$MODELS_DIR/bf16" --quiet 2>&1 | tail -3
        echo "  [DONE] Model cached at $MODELS_DIR/bf16"
    elif command -v huggingface-cli &>/dev/null; then
        echo "  Downloading via huggingface-cli (this may take a while on first run)..."
        huggingface-cli download "$MODEL_ID" --local-dir "$MODELS_DIR/bf16" --quiet 2>&1 | tail -3
        echo "  [DONE] Model cached at $MODELS_DIR/bf16"
    else
        pip install --quiet huggingface_hub
        python -c "
from huggingface_hub import snapshot_download
print('  Downloading model (this may take a while on first run)...')
path = snapshot_download('$MODEL_ID', local_dir='$MODELS_DIR/bf16')
print(f'  [DONE] Model cached at: {path}')
"
    fi
}

verify_setup() {
    if [ "$DRY_RUN" = true ]; then
        echo ""
        echo "[DRY-RUN] Skipping verification (no files were actually installed)."
        return 0
    fi

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

    # vllm-omni (only when --tts was selected)
    if [ "$INSTALL_TTS" = true ]; then
        if "$VENV_DIR/bin/python" -c "import vllm_omni" 2>/dev/null; then
            echo "  [OK] vllm-omni (TTS engine)"
        else
            echo "  [FAIL] vllm-omni (--tts requested but import failed)"
            ok=false
        fi
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
install_vllm_omni
install_voxtral_server
download_model
verify_setup

echo ""
echo "========================================"
if [ "$DRY_RUN" = true ]; then
    echo "  Dry-run Complete (no changes)"
else
    echo "  Initialization Complete!"
fi
echo "========================================"
echo ""
echo "Next steps:"
echo ""
echo "  1. Activate the venv:"
echo "     source $VENV_DIR/bin/activate"
echo ""
echo "  2. Start vLLM (ASR, separate terminal):"
echo "     ./start-vllm.sh"
echo ""
if [ "$INSTALL_TTS" = true ]; then
    echo "  3. (optional) Start vllm-omni (TTS, another terminal):"
    echo "     ./start-vllm-tts.sh"
    echo ""
    echo "     NOTE: ASR + TTS cannot both fit on a 24 GiB GPU. See"
    echo "     apps/vllm-server/docs/tts_spike.md for the coexistence"
    echo "     finding and workarounds."
    echo ""
    echo "  4. Start voxtral-server (another terminal):"
    echo "     cd $SCRIPT_DIR && ./start.sh"
    echo ""
    echo "  5. Open http://localhost:8090 in your browser"
else
    echo "  3. Start voxtral-server (another terminal):"
    echo "     cd $SCRIPT_DIR && ./start.sh"
    echo ""
    echo "  4. Open http://localhost:8090 in your browser"
    echo ""
    echo "  Tip: re-run with --tts to enable the TTS tab (Voxtral-4B-TTS)."
fi
echo ""
