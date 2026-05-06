#!/usr/bin/env python3
"""Capture upstream-tokenized Voxtral-TTS prompt_ids for the canonical
(text, voice) pairs.

This is the byte-exact reference our Rust Tekken loader and chat-template
builder must match (Phase 2-H.2/2-H.3). Unlike `dump_tts_e2e_golden.py`,
this script does NOT require the vllm-omni HTTP server — it just loads
the Mistral tokenizer locally and runs `encode_speech_request`.

Output: <output_dir>/prompt_ids_<name>.json with keys:
  - name, text, voice, voice_n_audio_tokens
  - prompt_ids (list[int]) — the full tokenized sequence
  - text_token_ids (list[int]) — the BPE encoding of text alone
  - special_token_ids: {bos, begin_audio, audio, text_to_audio, audio_to_text}

Pre-req: apps/vllm-server's venv must have mistral_common installed
(it's a transitive dep of vllm). Run from the repo root or apps/vllm-server.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mistral_common.protocol.speech.request import SpeechRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer


# Same canonical (name, voice, text) inputs as dump_tts_e2e_golden.py so
# the prompt_ids fixtures align 1:1 with the e2e wav fixtures.
CANONICAL_INPUTS = [
    ("en_short_neutral_male", "neutral_male", "Hello, this is a test."),
    ("en_short_neutral_female", "neutral_female", "Hello, this is a test."),
    ("de_short_de_male", "de_male", "Hallo, das ist ein Test."),
    ("fr_short_fr_female", "fr_female", "Bonjour, ceci est un test."),
    (
        "en_medium_casual_male",
        "casual_male",
        "The quick brown fox jumps over the lazy dog. The early bird catches the worm.",
    ),
    (
        "en_long_cheerful_female",
        "cheerful_female",
        "When in the course of human events it becomes necessary for one people to dissolve "
        "the political bands which have connected them with another, and to assume among "
        "the powers of the earth the separate and equal station to which the laws of nature entitle them.",
    ),
]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--tokenizer",
        default="/home/gjovanov/gjovanov/starling/models/cache/tts/tekken.json",
    )
    p.add_argument(
        "--output-dir",
        default="/home/gjovanov/gjovanov/starling/apps/burn-server/test_data/tts_golden",
    )
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = MistralTokenizer.from_file(args.tokenizer)
    instr = tokenizer.instruct_tokenizer
    audio_enc = instr.audio_encoder

    # Snapshot the special-token IDs once so the JSON is self-describing.
    special = {
        "bos": tokenizer.tokenizer_v3.bos_id if hasattr(tokenizer, "tokenizer_v3") else 1,
        "audio": audio_enc.audio_token,
        "begin_audio": audio_enc.begin_audio_token,
        "text_to_audio": audio_enc.text_to_audio_token,
        "audio_to_text": audio_enc.audio_to_text_token,
    }

    for name, voice, text in CANONICAL_INPUTS:
        tokenized = instr.encode_speech_request(SpeechRequest(input=text, voice=voice))
        prompt_ids = list(tokenized.tokens)
        # Standalone text BPE — no BOS/EOS — for unit-testing the
        # tokenizer in isolation.
        text_ids = list(instr.tokenizer.encode(text, bos=False, eos=False))
        n_audio_for_voice = audio_enc.audio_config.voice_num_audio_tokens[voice]
        meta = {
            "name": name,
            "text": text,
            "voice": voice,
            "voice_n_audio_tokens": n_audio_for_voice,
            "prompt_ids": prompt_ids,
            "text_token_ids": text_ids,
            "special_token_ids": special,
        }
        out_path = out / f"prompt_ids_{name}.json"
        out_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        print(
            f"  {name}: total={len(prompt_ids)}, text_ids={len(text_ids)}, "
            f"n_audio={n_audio_for_voice} → {out_path}"
        )

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
