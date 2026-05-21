#!/usr/bin/env bash
#
# Render every .mmd file under docs/blog/<post>/diagrams/ to .png.
# Used by the Medium-style blog posts, which reference pre-rendered images
# (Medium doesn't support inline mermaid).
#
# Usage:
#   ./render-mermaid.sh                       # render all 4 posts
#   ./render-mermaid.sh voxtral-asr-vllm      # render one post
#
# Requirements:
#   * `mmdc` (mermaid-cli) on PATH:
#       npm install -g @mermaid-js/mermaid-cli

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# SCRIPT_DIR is .../docs/blog/_shared/scripts → BLOG_DIR is .../docs/blog
BLOG_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Prefer the locally-installed mmdc (bun-installed under node_modules/.bin)
# over any global one — version pinning + PATH-free invocation.
if [ -x "$SCRIPT_DIR/node_modules/.bin/mmdc" ]; then
  MMDC="$SCRIPT_DIR/node_modules/.bin/mmdc"
elif command -v mmdc >/dev/null 2>&1; then
  MMDC="$(command -v mmdc)"
else
  echo "ERROR: mmdc not found." >&2
  echo "       Install locally:  cd $SCRIPT_DIR && bun add --dev @mermaid-js/mermaid-cli" >&2
  echo "       Or globally:      npm install -g @mermaid-js/mermaid-cli" >&2
  exit 1
fi

# mmdc invokes node — prefer the real node binary, not the bun-bundled
# wrapper (which can't load the puppeteer chromium adapter cleanly).
NODE_BIN=""
for candidate in "$HOME/.local/node/bin/node" "$(command -v node 2>/dev/null || true)"; do
  if [ -n "$candidate" ] && [ -x "$candidate" ]; then
    if [ "$("$candidate" -e 'console.log(typeof Bun)' 2>/dev/null)" = "undefined" ]; then
      NODE_BIN="$candidate"
      break
    fi
  fi
done
if [ -n "$NODE_BIN" ]; then
  export PATH="$(dirname "$NODE_BIN"):$PATH"
fi

POSTS=()
if [ $# -gt 0 ]; then
  POSTS=("$@")
else
  POSTS=(voxtral-asr-vllm voxtral-asr-burn voxtral-tts-vllm voxtral-tts-burn)
fi

FAILED=()

for post in "${POSTS[@]}"; do
  dir="$BLOG_DIR/$post/diagrams"
  if [ ! -d "$dir" ]; then
    echo "Skip $post (no diagrams/ directory)"
    continue
  fi
  shopt -s nullglob
  for src in "$dir"/*.mmd; do
    out="${src%.mmd}.png"
    echo "Rendering $src -> $out"
    if ! "$MMDC" -i "$src" -o "$out" -b "#0f1219" -w 1600 --quiet; then
      echo "  (skipped: mmdc failed; the .mmd may have an unquoted character)"
      FAILED+=("$src")
    fi
  done
  shopt -u nullglob
done

if [ ${#FAILED[@]} -gt 0 ]; then
  echo
  echo "Failed to render ${#FAILED[@]} diagram(s):"
  for f in "${FAILED[@]}"; do
    echo "  - $f"
  done
  exit 2
fi

echo "Done."
