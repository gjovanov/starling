#!/bin/bash
#
# Extraction LLM vLLM server (stage-2 structured extraction, port 8004).
# Default model: Qwen/Qwen3-8B-FP8 (guided JSON via xgrammar). Normally
# spawned lazily by ocr-server; run manually for debugging.
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
MODEL_PATH="${OCR_EXTRACT_MODEL_PATH:-$MODELS_DIR/ocr/qwen3-8b-fp8}"
SERVED_NAME="${OCR_EXTRACT_MODEL:-qwen3-8b-fp8}"
PORT="${OCR_VLLM_EXTRACT_PORT:-8004}"
# 0.45 leaves <1 GiB KV after the ~9.2 GiB FP8 weights; 16k ctx needs 2.25 GiB.
GPU_UTIL="${OCR_GPU_UTIL_EXTRACT:-0.55}"
MAX_LEN="${OCR_VLLM_EXTRACT_MAX_LEN:-16384}"
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
    echo "ERROR: extraction model not found at $MODEL_PATH"
    echo "       Run: $PROJECT_DIR/models/download.sh --ocr-only"
    exit 1
fi

# Same WSL2 workarounds as start-vllm-ocr.sh (UVA-less CUDA + FlashInfer's
# broken sm_120 arch probe).
export VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-0}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

# ngram speculative decoding — invoice JSON is highly repetitive (field names +
# structure repeat per line item, and schema-constrained decoding makes the
# scaffolding deterministic), so drafting n-grams from the prompt/output-so-far
# lands the structural tokens for free. Speculation is exact (verified, mismatches
# rejected) → identical output, faster decode. Disable with OCR_EXTRACT_NGRAM=0.
SPEC_ARGS=()
if [ "${OCR_EXTRACT_NGRAM:-1}" = "1" ]; then
    SPEC_ARGS=(--speculative-config "{\"method\": \"ngram\", \"num_speculative_tokens\": ${OCR_EXTRACT_NGRAM_TOKENS:-5}, \"prompt_lookup_max\": ${OCR_EXTRACT_NGRAM_MAX:-4}, \"prompt_lookup_min\": ${OCR_EXTRACT_NGRAM_MIN:-2}}")
fi

echo "[vLLM-Extract] $SERVED_NAME on port $PORT (util $GPU_UTIL, max-len $MAX_LEN, attn $ATTN_BACKEND, ngram ${OCR_EXTRACT_NGRAM:-1})"
exec vllm serve "$MODEL_PATH" \
    --served-model-name "$SERVED_NAME" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --max-model-len "$MAX_LEN" \
    --attention-backend "$ATTN_BACKEND" \
    "${SPEC_ARGS[@]}"
