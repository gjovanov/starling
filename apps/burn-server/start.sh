#!/bin/bash
# Start the Burn ASR server
#
# Prerequisites:
#   1. Run ./init.sh first (builds binary + downloads models)
#
# Unlike vllm-server, burn-server runs the model directly (no separate model server).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Load .env files (local overrides root)
[ -f "$PROJECT_DIR/.env" ] && set -a && source "$PROJECT_DIR/.env" && set +a
[ -f "$SCRIPT_DIR/.env" ] && set -a && source "$SCRIPT_DIR/.env" && set +a

# DZN (Vulkan-over-DX12 on WSL2) support
# - Allow non-conformant adapter (DZN reports conformance 0.0.0.0)
# - Restrict autotune to avoid plane/subgroup ops unsupported by DZN
export WGPU_ALLOW_UNDERLYING_NONCOMPLIANT_ADAPTER="${WGPU_ALLOW_UNDERLYING_NONCOMPLIANT_ADAPTER:-1}"
export CUBECL_AUTOTUNE_LEVEL="${CUBECL_AUTOTUNE_LEVEL:-1}"
# Prefer DZN over llvmpipe if both are installed
if [ -f /usr/share/vulkan/icd.d/dzn_icd.json ]; then
    export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/dzn_icd.json}"
fi

# Prefer workspace-level binary (cargo workspace builds go to root target/)
if [ -f "$PROJECT_DIR/target/release/burn-server" ]; then
    BINARY="$PROJECT_DIR/target/release/burn-server"
else
    BINARY="$SCRIPT_DIR/target/release/burn-server"
fi

if [ ! -f "$BINARY" ]; then
    echo "ERROR: Binary not found at $BINARY"
    echo "       Run ./init.sh first to build."
    exit 1
fi

export BURN_FRONTEND_PATH="${BURN_FRONTEND_PATH:-$PROJECT_DIR/frontend}"
export BURN_MEDIA_DIR="${BURN_MEDIA_DIR:-$PROJECT_DIR/media}"
export BURN_PORT="${BURN_PORT:-8091}"
export STARLING_MODELS_DIR="${STARLING_MODELS_DIR:-$PROJECT_DIR/models/cache}"

echo "=== Burn Server ==="
echo "  Port:     $BURN_PORT"
echo "  Quant:    ${BURN_QUANT:-q4}"
echo "  Backend:  ${BURN_BACKEND:-wgpu}"
echo "  Models:   $STARLING_MODELS_DIR"
echo "  Media:    $BURN_MEDIA_DIR"
echo "  Frontend: $BURN_FRONTEND_PATH"
echo "  TURN:     ${TURN_SERVER:-none}"
echo "  Relay:    ${FORCE_RELAY:-false}"
echo ""

ARGS=(
    --port "$BURN_PORT"
    --quant "${BURN_QUANT:-q4}"
    --backend "${BURN_BACKEND:-wgpu}"
    --models-dir "$STARLING_MODELS_DIR"
    --frontend "$BURN_FRONTEND_PATH"
    --media-dir "$BURN_MEDIA_DIR"
)

# TURN/WebRTC configuration
[ -n "$TURN_SERVER" ] && ARGS+=(--turn-server "$TURN_SERVER")
[ -n "$TURN_SHARED_SECRET" ] && ARGS+=(--turn-shared-secret "$TURN_SHARED_SECRET")
[ -n "$TURN_USERNAME" ] && ARGS+=(--turn-username "$TURN_USERNAME")
[ -n "$TURN_PASSWORD" ] && ARGS+=(--turn-password "$TURN_PASSWORD")
[ "$FORCE_RELAY" = "true" ] && ARGS+=(--force-relay)

"$BINARY" "${ARGS[@]}"
