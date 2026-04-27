#!/bin/bash
#
# Bash test: verify `./init.sh --tts --no-model --dry-run` invokes the right
# install commands in the right order — vllm BEFORE vllm-omni, no --upgrade.
#
# Run from the repo root or from anywhere; uses absolute paths.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

fail() {
    echo "[TEST FAIL] $*" >&2
    exit 1
}

pass() {
    echo "[TEST PASS] $*"
}

# ─── Case 1: --tts flag enables vllm-omni install in the right spot ──

OUT=$(bash "$APP_DIR/init.sh" --tts --no-model --dry-run 2>&1)

# Ordering: vllm install must appear BEFORE vllm-omni install
VLLM_LINE=$(printf '%s\n' "$OUT" | grep -n "pip install --quiet vllm " | head -1 | cut -d: -f1 || true)
VLLM_OMNI_LINE=$(printf '%s\n' "$OUT" | grep -n "pip install --quiet vllm-omni" | head -1 | cut -d: -f1 || true)

[ -n "$VLLM_LINE" ]      || fail "--tts dry-run: 'pip install ... vllm' line not found"
[ -n "$VLLM_OMNI_LINE" ] || fail "--tts dry-run: 'pip install ... vllm-omni' line not found"
[ "$VLLM_LINE" -lt "$VLLM_OMNI_LINE" ] || fail "vllm-omni line ($VLLM_OMNI_LINE) appeared before vllm line ($VLLM_LINE)"
pass "--tts: vllm installed before vllm-omni (lines $VLLM_LINE → $VLLM_OMNI_LINE)"

# Must NOT pass --upgrade to vllm-omni install (the HF #29 trap)
if printf '%s\n' "$OUT" | grep -q -E "pip install.*--upgrade.*vllm-omni"; then
    fail "--tts dry-run: vllm-omni install used --upgrade (forbidden)"
fi
pass "--tts: no --upgrade flag on vllm-omni install"

# Step labels are 7 when --tts is set
if ! printf '%s\n' "$OUT" | grep -q "\[5/7\] Installing vllm-omni"; then
    fail "--tts dry-run: expected '[5/7] Installing vllm-omni' marker"
fi
pass "--tts: step labels reflect the +1 TTS slot ([5/7])"

# ─── Case 2: no --tts → vllm-omni must be skipped ────────────────────

OUT_NOTTS=$(bash "$APP_DIR/init.sh" --no-model --dry-run 2>&1)

if printf '%s\n' "$OUT_NOTTS" | grep -q "pip install --quiet vllm-omni"; then
    fail "no-TTS dry-run: vllm-omni was installed (should be skipped)"
fi
pass "no-TTS: vllm-omni install skipped"

if ! printf '%s\n' "$OUT_NOTTS" | grep -q "vllm-omni: skipped"; then
    fail "no-TTS dry-run: expected 'vllm-omni: skipped' marker"
fi
pass "no-TTS: skip marker present"

# Step labels are /6 when --tts is off
if ! printf '%s\n' "$OUT_NOTTS" | grep -q "\[5/6\] Installing voxtral-server"; then
    fail "no-TTS dry-run: expected '[5/6] Installing voxtral-server' marker"
fi
pass "no-TTS: step labels reflect the no-TTS layout ([5/6])"

# ─── Case 3: VOXTRAL_INSTALL_TTS=1 env var equivalent to --tts ───────

OUT_ENV=$(VOXTRAL_INSTALL_TTS=1 bash "$APP_DIR/init.sh" --no-model --dry-run 2>&1)

if ! printf '%s\n' "$OUT_ENV" | grep -q "pip install --quiet vllm-omni"; then
    fail "env-var: VOXTRAL_INSTALL_TTS=1 did not trigger vllm-omni install"
fi
pass "env-var: VOXTRAL_INSTALL_TTS=1 triggers TTS install"

echo ""
echo "All install-script tests passed."
