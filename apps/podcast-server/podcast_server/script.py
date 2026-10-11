"""Podcast scripts: turns, the `<speaker>: <text>` file format, validation.

Shared by the HTTP API and the batch CLI so both refuse the same inputs.
No torch here — the API process never imports it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_WS = re.compile(r"\s+")

# Generation-parameter bounds, shared by the API schema and the CLI.
SEED_MAX = 2**32 - 1                     # numpy's seed range
CFG_SCALE_RANGE = (1.0, 3.0)             # checkpoint default 1.3
DIFFUSION_STEPS_RANGE = (1, 50)          # checkpoint default 10


class ScriptError(ValueError):
    """An input the server refuses (HTTP 400, CLI exit code 2)."""


def check_generation(seed: int, cfg_scale: float, diffusion_steps: int) -> None:
    if not 0 <= seed <= SEED_MAX:
        raise ScriptError(f"seed must be in [0, {SEED_MAX}]")
    if not CFG_SCALE_RANGE[0] <= cfg_scale <= CFG_SCALE_RANGE[1]:
        raise ScriptError(f"cfg_scale must be in [{CFG_SCALE_RANGE[0]}, {CFG_SCALE_RANGE[1]}]")
    if not DIFFUSION_STEPS_RANGE[0] <= diffusion_steps <= DIFFUSION_STEPS_RANGE[1]:
        raise ScriptError(
            f"diffusion_steps must be in [{DIFFUSION_STEPS_RANGE[0]}, {DIFFUSION_STEPS_RANGE[1]}]")


@dataclass(frozen=True)
class Turn:
    speaker: str
    text: str


@dataclass(frozen=True)
class Plan:
    """A validated script, ready for the engine.

    `speakers` are voice names in role order (order of first appearance);
    `roles[i]` is turn i's index into `speakers`. VibeVoice addresses
    speakers as "Speaker 0..3" and pairs voice prompt k with role k, which is
    exactly the original processor's 0-based convention.
    """

    turns: tuple[Turn, ...]
    speakers: tuple[str, ...]
    roles: tuple[int, ...]
    total_chars: int
    max_new_tokens: int


def normalize_text(text: str) -> str:
    """Collapse all whitespace (incl. newlines) to single spaces.

    Each turn becomes one ` Speaker N: <text>` line of the model prompt, so
    an embedded newline would forge a malformed extra line.
    """
    return _WS.sub(" ", text).strip()


def parse_script(source: str) -> list[Turn]:
    """Parse `<speaker>: <text>` lines. Blank lines and `#` comments are skipped.

    Only the first colon splits, so the text may contain colons. Speaker
    names are case-insensitive (stored lowercase).
    """
    turns: list[Turn] = []
    for lineno, raw in enumerate(source.lstrip("﻿").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        speaker, sep, text = line.partition(":")
        if not sep or not speaker.strip() or not text.strip():
            raise ScriptError(f"line {lineno}: expected '<speaker>: <text>', got {raw.strip()!r}")
        turns.append(Turn(speaker=speaker.strip().lower(), text=normalize_text(text)))
    if not turns:
        raise ScriptError("script has no turns")
    return turns


def plan(turns: list[Turn], known_voices: set[str], settings) -> Plan:
    """Validate turns against the voice registry and the request limits."""
    if not turns:
        raise ScriptError("no turns")
    if len(turns) > settings.max_turns:
        raise ScriptError(f"{len(turns)} turns exceeds the limit of {settings.max_turns}")

    clean: list[Turn] = []
    speakers: list[str] = []
    roles: list[int] = []
    total = 0
    for i, turn in enumerate(turns, 1):
        name = turn.speaker.strip().lower()
        text = normalize_text(turn.text)
        if not text:
            raise ScriptError(f"turn {i}: empty text")
        if len(text) > settings.max_chars_per_turn:
            raise ScriptError(
                f"turn {i}: {len(text)} chars exceeds the per-turn limit of {settings.max_chars_per_turn}")
        if name not in known_voices:
            raise ScriptError(
                f"turn {i}: unknown speaker {name!r} (voices: {', '.join(sorted(known_voices)) or 'none'})")
        if name not in speakers:
            if len(speakers) == settings.max_speakers:
                raise ScriptError(
                    f"turn {i}: speaker {name!r} would be speaker #{len(speakers) + 1}; "
                    f"the model supports at most {settings.max_speakers} per script")
            speakers.append(name)
        roles.append(speakers.index(name))
        clean.append(Turn(speaker=name, text=text))
        total += len(text)
    if total > settings.max_total_chars:
        raise ScriptError(f"{total} chars exceeds the per-script limit of {settings.max_total_chars}")

    budget = int(total * settings.tokens_per_char) + settings.token_slack
    return Plan(
        turns=tuple(clean),
        speakers=tuple(speakers),
        roles=tuple(roles),
        total_chars=total,
        max_new_tokens=min(settings.max_new_tokens_cap, budget),
    )
