#!/bin/bash
#
# ocr-server prerequisites check. Informational — reports what is missing
# and how to install it; exits non-zero if a hard requirement is absent.
#
set -u

FAIL=0

check() {
    local label="$1" ok="$2" hint="$3"
    if [ "$ok" = true ]; then
        echo "[OK]   $label"
    else
        echo "[MISS] $label — $hint"
    fi
}

# ─── Hard requirements ────────────────────────────────────────────

if command -v python3 >/dev/null && python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
    check "python3 >= 3.10 ($(python3 --version 2>&1))" true ""
else
    check "python3 >= 3.10" false "install Python 3.10+"
    FAIL=1
fi

# ─── GPU (needed for glm-ocr / extraction; mock engine runs without) ──

if command -v nvidia-smi >/dev/null; then
    check "nvidia-smi ($(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1))" true ""
else
    check "nvidia-smi" false "GPU engines unavailable; OCR_ENGINE=mock still works"
fi

# ─── Optional: tesseract CPU baseline (Phase 4 bake-off engine) ────
# ocr-server auto-detects either a PATH install or the user-space prefix
# at ~/.local/tesseract (no root required — see the recipe below).

LOCAL_TESS="$HOME/.local/tesseract/usr/bin/tesseract"
if command -v tesseract >/dev/null; then
    if tesseract --list-langs 2>/dev/null | grep -q '^bul$'; then
        check "tesseract + bul traineddata (PATH)" true ""
    else
        check "tesseract (bul language missing)" false "sudo apt install tesseract-ocr-bul"
    fi
elif [ -x "$LOCAL_TESS" ]; then
    check "tesseract (user-space ~/.local/tesseract)" true ""
else
    check "tesseract" false "optional CPU baseline. With root: sudo apt install tesseract-ocr tesseract-ocr-bul tesseract-ocr-eng. Without root:
         mkdir -p /tmp/tessdeb ~/.local/tesseract && cd /tmp/tessdeb &&
         apt-get download tesseract-ocr tesseract-ocr-bul tesseract-ocr-eng tesseract-ocr-osd libtesseract5 liblept5 &&
         for d in *.deb; do dpkg -x \"\$d\" ~/.local/tesseract; done"
fi

# PyMuPDF is pure-wheel (bundles MuPDF) — no system poppler/mupdf needed.
echo "[OK]   PDF rendering via PyMuPDF wheel (no system packages required)"

exit $FAIL
