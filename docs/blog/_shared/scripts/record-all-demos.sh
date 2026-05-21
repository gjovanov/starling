#!/usr/bin/env bash
#
# Record all four demo videos for the Starling blog series.
#
# This script does NOT manage the servers \u2014 it pauses before each recording
# so you can switch which server is running (ASR vs TTS, vllm-server vs
# burn-server). Press ENTER when the server is ready, or 's' to skip a post.
#
# Usage:
#   ./record-all-demos.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

POSTS=(
  "voxtral-asr-vllm:Start vllm-server (./start-vllm.sh + ./start.sh in apps/vllm-server)"
  "voxtral-asr-burn:Start burn-server (./start.sh in apps/burn-server)"
  "voxtral-tts-vllm:Start vllm-server with TTS (./init.sh --tts + ./start.sh)"
  "voxtral-tts-burn:Start burn-server with TTS (./init.sh --tts + ./start.sh)"
)

for entry in "${POSTS[@]}"; do
  POST="${entry%%:*}"
  HINT="${entry#*:}"
  echo
  echo "================================================================"
  echo " Next post: $POST"
  echo " Setup    : $HINT"
  echo "================================================================"
  read -r -p "Press ENTER to record, or 's' + ENTER to skip: " ans
  if [ "${ans:-}" = "s" ] || [ "${ans:-}" = "S" ]; then
    echo "Skipped $POST"
    continue
  fi
  "$SCRIPT_DIR/record-video.sh" --post "$POST"
done

echo
echo "All four videos done. Upload each to YouTube and fill in the placeholder"
echo "URLs in the matching docs/blog/<post>/blog-post-medium.md files."
