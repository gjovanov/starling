#!/bin/bash
# Start vLLM serving Voxtral-Mini-4B-Realtime
#
# Prerequisite: Run ./init.sh first (sets up venv + downloads model)
#
# The model uses ~9GB VRAM in BF16. Requires GPU with ≥16GB.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
MODELS_DIR="${STARLING_MODELS_DIR:-$PROJECT_DIR/models/cache}"

# Activate venv
if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
else
    echo "ERROR: venv not found. Run ./init.sh first."
    exit 1
fi

MODEL_ID="mistralai/Voxtral-Mini-4B-Realtime-2602"
PORT="${VOXTRAL_VLLM_PORT:-8001}"

# Use shared model dir if it exists, otherwise fall back to HF cache
MODEL_PATH="$MODEL_ID"
if [ -d "$MODELS_DIR/bf16" ] && [ -f "$MODELS_DIR/bf16/consolidated.safetensors" ]; then
    MODEL_PATH="$MODELS_DIR/bf16"
fi

echo "=== vLLM Server ==="
echo "  Model: $MODEL_PATH"
echo "  Port:  $PORT"
echo "  Python: $(python --version)"
echo ""

# --gpu-memory-utilization is intentionally lowered from 0.90 → 0.45 so a
# second vllm-omni TTS process (start-vllm-tts.sh, stage-0 util 0.40 +
# stage-1 0.10 = 0.50) can fit on the same GPU. Override via
# VOXTRAL_VLLM_GPU_UTIL if you don't run TTS alongside ASR (e.g. set it to
# 0.90 + raise --max-model-len for more KV cache).
GPU_UTIL="${VOXTRAL_VLLM_GPU_UTIL:-0.45}"

# --max-model-len lowered from 16384 → 4096 to fit the smaller KV cache budget
# at GPU_UTIL=0.50. Realtime ASR sessions rotate context every ~200 commits
# (~100s ≈ 1200 tokens) so 4096 is well above what's used in practice.
MAX_MODEL_LEN="${VOXTRAL_VLLM_MAX_MODEL_LEN:-4096}"

vllm serve "$MODEL_PATH" \
    --port "$PORT" \
    --dtype bfloat16 \
    --enforce-eager \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_UTIL"
