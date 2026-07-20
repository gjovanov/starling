#!/bin/bash
#
# Start the OCR FastAPI web server (port 8092).
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

# Load .env files (local overrides root)
[ -f "$PROJECT_DIR/.env" ] && set -a && source "$PROJECT_DIR/.env" && set +a
[ -f "$SCRIPT_DIR/.env" ] && set -a && source "$SCRIPT_DIR/.env" && set +a

# Map unprefixed shared vars to OCR_-prefixed names
[ -n "${STARLING_MODELS_DIR:-}" ] && export OCR_MODELS_DIR="${OCR_MODELS_DIR:-$STARLING_MODELS_DIR}"

export OCR_PORT="${OCR_PORT:-8092}"
export OCR_FRONTEND_PATH="${OCR_FRONTEND_PATH:-$PROJECT_DIR/frontend}"
export OCR_MEDIA_DIR="${OCR_MEDIA_DIR:-$PROJECT_DIR/media/ocr}"
export OCR_RESULTS_DIR="${OCR_RESULTS_DIR:-$PROJECT_DIR/results/ocr}"
export OCR_MODELS_DIR="${OCR_MODELS_DIR:-$PROJECT_DIR/models/cache}"

if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
elif ! python3 -c "import ocr_server" 2>/dev/null; then
    echo "ERROR: ocr-server not installed. Run ./init.sh first."
    exit 1
fi

echo "[Start] ocr-server on port $OCR_PORT (engine: ${OCR_ENGINE:-glm-ocr})"
cd "$SCRIPT_DIR"
exec python3 -m uvicorn ocr_server.main:app --host 0.0.0.0 --port "$OCR_PORT" --log-level info
