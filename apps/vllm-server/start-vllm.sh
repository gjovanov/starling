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

vllm serve "$MODEL_PATH" \
    --port "$PORT" \
    --dtype bfloat16 \
    --enforce-eager \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.90
