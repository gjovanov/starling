#!/bin/bash
# Start vllm-omni serving Voxtral-4B-TTS-2603
#
# Prerequisites:
#   1. ./init.sh --tts          (one-shot: venv + vllm + vllm-omni + TTS model)
#      OR equivalently:
#          ./init.sh
#          source .venv/bin/activate && pip install vllm-omni
#          ../../models/download.sh --tts-only
#   2. ./scripts/check_tts_install.py (auto-run by init.sh; manual sanity)
#
# This is a SECOND vLLM process (port 8002) that runs alongside the ASR
# vLLM (port 8001). The ASR start script (start-vllm.sh) lowers its
# --gpu-memory-utilization to leave room for this process.
#
# Endpoint: POST http://127.0.0.1:8002/v1/audio/speech
#   body: {"model": "<path>", "input": "<text>", "voice": "<id>",
#          "response_format": "wav"}
#
# Voice presets: 20 built-in (casual_male, de_female, fr_male, …).
# See apps/vllm-server/docs/tts_spike.md for the full list.

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

# Verify vllm-omni is installed (separate package from vllm)
if ! python -c "import vllm_omni" 2>/dev/null; then
    echo "ERROR: vllm-omni not installed."
    echo "       Run: source $VENV_DIR/bin/activate && pip install vllm-omni"
    exit 1
fi

PORT="${VOXTRAL_VLLM_TTS_PORT:-8002}"
TTS_MODEL_PATH="${VOXTRAL_TTS_MODEL_PATH:-$MODELS_DIR/tts}"

if [ ! -f "$TTS_MODEL_PATH/consolidated.safetensors" ]; then
    echo "ERROR: TTS model not found at $TTS_MODEL_PATH"
    echo "       Run: $PROJECT_DIR/models/download.sh --tts-only"
    exit 1
fi

echo "=== vLLM-Omni TTS Server ==="
echo "  Model: $TTS_MODEL_PATH"
echo "  Port:  $PORT"
echo "  Python: $(python --version)"
echo ""

# Mistral-format flags are required: the TTS repo ships only params.json
# (no config.json), so vllm-omni's voxtral_tts config parser activates only
# when --config-format=mistral.
#
# --task-type CustomVoice = use the 20 built-in voice presets.
# --task-type Base        = zero-shot voice cloning (Phase 7), requires a
#                            ref_audio + ref_text per request.
# Override at boot via VOXTRAL_TTS_TASK_TYPE; the lifecycle manager
# (voxtral_server/tts/lifecycle.py) sets this when restarting the
# subprocess to switch modes.
TASK_TYPE="${VOXTRAL_TTS_TASK_TYPE:-CustomVoice}"
case "$TASK_TYPE" in
    CustomVoice|Base|VoiceDesign) ;;
    *)
        echo "ERROR: VOXTRAL_TTS_TASK_TYPE=$TASK_TYPE is not one of CustomVoice|Base|VoiceDesign"
        exit 1
        ;;
esac

# --stage-configs-path overrides the bundled stage YAML (which sets stage 0 to
# gpu_memory_utilization=0.8 and overflows the 24 GiB GPU when ASR is also
# resident). Our override drops it to 0.45.
STAGE_CONFIG="$SCRIPT_DIR/configs/voxtral_tts.yaml"
if [ ! -f "$STAGE_CONFIG" ]; then
    echo "ERROR: Stage config not found at $STAGE_CONFIG"
    exit 1
fi

echo "  Task type:    $TASK_TYPE"
exec vllm-omni serve "$TTS_MODEL_PATH" \
    --omni \
    --task-type "$TASK_TYPE" \
    --port "$PORT" \
    --tokenizer-mode mistral \
    --config-format mistral \
    --load-format mistral \
    --stage-configs-path "$STAGE_CONFIG"
