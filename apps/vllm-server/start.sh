#!/bin/bash
# Start the Voxtral ASR server
#
# Prerequisites:
#   1. Run ./init.sh first (sets up venv, vLLM, deps)
#   2. Start vLLM in a separate terminal:
#      source .venv/bin/activate
#      vllm serve mistralai/Voxtral-Mini-4B-Realtime-2602 --port 8001

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

# Activate venv if it exists
if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
elif ! python3 -c "import voxtral_server" 2>/dev/null; then
    echo "ERROR: voxtral-server not installed. Run ./init.sh first."
    exit 1
fi

export VOXTRAL_FRONTEND_PATH="${VOXTRAL_FRONTEND_PATH:-$PROJECT_DIR/frontend}"
export VOXTRAL_MEDIA_DIR="${VOXTRAL_MEDIA_DIR:-$PROJECT_DIR/media}"
export VOXTRAL_PORT="${VOXTRAL_PORT:-80}"
export VOXTRAL_VLLM_URL="${VOXTRAL_VLLM_URL:-ws://localhost:8001/v1/realtime}"

echo "=== Starling Server ==="
echo "  Port:     $VOXTRAL_PORT"
echo "  vLLM:     $VOXTRAL_VLLM_URL"
echo "  Media:    $VOXTRAL_MEDIA_DIR"
echo "  Frontend: $VOXTRAL_FRONTEND_PATH"
echo "  Python:   $(python3 --version) ($(which python3))"
echo ""

cd "$SCRIPT_DIR"
python3 -m uvicorn voxtral_server.main:app --host 0.0.0.0 --port "$VOXTRAL_PORT" --log-level warning
