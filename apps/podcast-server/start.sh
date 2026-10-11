#!/bin/bash
#
# Start the podcast FastAPI server on 127.0.0.1:8093 — loopback ONLY.
#
# The endpoint has no auth and drives the GPU, so unlike the other apps'
# start.sh it never binds a public interface. The model worker spawns on
# the first request and exits after PODCAST_IDLE_UNLOAD_SECS idle (600 s).
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

# Load .env files (local overrides root)
[ -f "$PROJECT_DIR/.env" ] && set -a && source "$PROJECT_DIR/.env" && set +a
[ -f "$SCRIPT_DIR/.env" ] && set -a && source "$SCRIPT_DIR/.env" && set +a

export PODCAST_PORT="${PODCAST_PORT:-8093}"

if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
elif ! python3 -c "import podcast_server" 2>/dev/null; then
    echo "ERROR: podcast-server not installed. Run ./init.sh first."
    exit 1
fi

echo "[Start] podcast-server on 127.0.0.1:$PODCAST_PORT (engine: ${PODCAST_ENGINE:-vibevoice})"
cd "$SCRIPT_DIR"
exec python3 -m uvicorn podcast_server.main:app --host 127.0.0.1 --port "$PODCAST_PORT" --log-level info
