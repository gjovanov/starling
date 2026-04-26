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

# Load .env files (local overrides root) — mirrors burn-server/start.sh.
# Without this, settings like VOXTRAL_TURN_SERVER, VOXTRAL_FORCE_RELAY,
# VOXTRAL_PUBLIC_IP, etc. are silently ignored.
[ -f "$PROJECT_DIR/.env" ] && set -a && source "$PROJECT_DIR/.env" && set +a
[ -f "$SCRIPT_DIR/.env" ] && set -a && source "$SCRIPT_DIR/.env" && set +a

# Map unprefixed legacy vars from the shared .env to the VOXTRAL_-prefixed
# names that pydantic-settings expects. (burn-server reads the unprefixed
# names; vllm-server reads the prefixed ones, so a single shared .env can
# drive both servers.)
[ -n "$PUBLIC_IP" ]            && export VOXTRAL_PUBLIC_IP="${VOXTRAL_PUBLIC_IP:-$PUBLIC_IP}"
[ -n "$TURN_SERVER" ]          && export VOXTRAL_TURN_SERVER="${VOXTRAL_TURN_SERVER:-$TURN_SERVER}"
[ -n "$TURN_USERNAME" ]        && export VOXTRAL_TURN_USERNAME="${VOXTRAL_TURN_USERNAME:-$TURN_USERNAME}"
[ -n "$TURN_PASSWORD" ]        && export VOXTRAL_TURN_PASSWORD="${VOXTRAL_TURN_PASSWORD:-$TURN_PASSWORD}"
[ -n "$TURN_SHARED_SECRET" ]   && export VOXTRAL_TURN_SHARED_SECRET="${VOXTRAL_TURN_SHARED_SECRET:-$TURN_SHARED_SECRET}"
[ -n "$FORCE_RELAY" ]          && export VOXTRAL_FORCE_RELAY="${VOXTRAL_FORCE_RELAY:-$FORCE_RELAY}"

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
echo "  Port:           $VOXTRAL_PORT"
echo "  vLLM:           $VOXTRAL_VLLM_URL"
echo "  Media:          $VOXTRAL_MEDIA_DIR"
echo "  Frontend:       $VOXTRAL_FRONTEND_PATH"
echo "  Public IP:      ${VOXTRAL_PUBLIC_IP:-(auto)}"
echo "  TURN server:    ${VOXTRAL_TURN_SERVER:-(none)}"
echo "  Force relay:    ${VOXTRAL_FORCE_RELAY:-false}"
echo "  Python:         $(python3 --version) ($(which python3))"
echo ""

cd "$SCRIPT_DIR"
# `--log-level warning` is fine — our diagnostic prints go to stderr directly,
# bypassing the logging level filter.
python3 -m uvicorn voxtral_server.main:app --host 0.0.0.0 --port "$VOXTRAL_PORT" --log-level warning
