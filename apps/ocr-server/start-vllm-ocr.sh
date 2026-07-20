#!/bin/bash
#
# GLM-OCR vLLM server (stage-1 OCR, port 8003).
# Canonical serving recipe from zai-org/GLM-OCR: vllm serve + MTP speculative
# decoding. Normally spawned lazily by the ocr-server engine manager; run
# manually for debugging.
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

# Load .env files (local overrides root)
[ -f "$PROJECT_DIR/.env" ] && set -a && source "$PROJECT_DIR/.env" && set +a
[ -f "$SCRIPT_DIR/.env" ] && set -a && source "$SCRIPT_DIR/.env" && set +a
[ -n "${STARLING_MODELS_DIR:-}" ] && export OCR_MODELS_DIR="${OCR_MODELS_DIR:-$STARLING_MODELS_DIR}"

MODELS_DIR="${OCR_MODELS_DIR:-$PROJECT_DIR/models/cache}"
MODEL_PATH="${OCR_GLM_MODEL_PATH:-$MODELS_DIR/ocr/glm-ocr}"
PORT="${OCR_VLLM_OCR_PORT:-8003}"
GPU_UTIL="${OCR_GPU_UTIL_OCR:-0.20}"
MAX_LEN="${OCR_VLLM_OCR_MAX_LEN:-16384}"
# FlashInfer's JIT arch probe fails on WSL2 + sm_120 (RTX 5090) with
# "FlashInfer requires GPUs with sm75 or higher"; Triton attention works.
ATTN_BACKEND="${OCR_VLLM_ATTN_BACKEND:-TRITON_ATTN}"

if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
else
    echo "ERROR: venv not found. Run ./init.sh first."
    exit 1
fi

if ! python -c "import vllm" 2>/dev/null; then
    echo "ERROR: vllm not installed. Run ./init.sh (without --no-gpu)."
    exit 1
fi

if [ ! -f "$MODEL_PATH/config.json" ]; then
    echo "ERROR: GLM-OCR model not found at $MODEL_PATH"
    echo "       Run: $PROJECT_DIR/models/download.sh --ocr-only"
    exit 1
fi

# WSL2: CUDA lacks UVA host-mapped memory, and vLLM's V2 model runner
# hard-requires it ("RuntimeError: UVA is not available"). Force the V1
# runner; override with VLLM_USE_V2_MODEL_RUNNER=1 on native Linux.
export VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-0}"
# FlashInfer's JIT arch probe also breaks on WSL2 + sm_120 (empty arch list →
# "requires GPUs with sm75 or higher"), taking the top-k/top-p sampler down
# with it. The PyTorch-native sampler is fine at OCR batch sizes.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

# MTP speculative decoding (~1.5x decode) — from the official serving guide.
# Disable with OCR_VLLM_OCR_MTP=0 if a vLLM upgrade breaks it.
SPEC_ARGS=()
if [ "${OCR_VLLM_OCR_MTP:-1}" = "1" ]; then
    SPEC_ARGS=(--speculative-config '{"method": "mtp", "num_speculative_tokens": 3}')
fi

echo "[vLLM-OCR] GLM-OCR on port $PORT (util $GPU_UTIL, max-len $MAX_LEN, attn $ATTN_BACKEND, mtp ${OCR_VLLM_OCR_MTP:-1})"
exec vllm serve "$MODEL_PATH" \
    --served-model-name glm-ocr \
    --host 127.0.0.1 \
    --port "$PORT" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --max-model-len "$MAX_LEN" \
    --attention-backend "$ATTN_BACKEND" \
    "${SPEC_ARGS[@]}"
