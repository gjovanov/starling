#!/bin/bash
# Start vLLM serving Voxtral-Mini-4B-Realtime
#
# Prerequisite: Run ./init.sh first (sets up venv + downloads model)
#
# The model uses ~9GB VRAM in BF16. Requires GPU with ≥16GB.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

# Activate venv
if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
else
    echo "ERROR: venv not found. Run ./init.sh first."
    exit 1
fi

MODEL_ID="mistralai/Voxtral-Mini-4B-Realtime-2602"
PORT="${VOXTRAL_VLLM_PORT:-8001}"

echo "=== vLLM Server ==="
echo "  Model: $MODEL_ID"
echo "  Port:  $PORT"
echo "  Python: $(python --version)"
echo ""

vllm serve "$MODEL_ID" \
    --port "$PORT" \
    --dtype bfloat16 \
    --enforce-eager \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.90
