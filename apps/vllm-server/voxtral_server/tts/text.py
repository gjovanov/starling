"""Sentence splitter for long-form TTS input.

The splitter's job is to chop a 20 000-char block of prose into chunks that
the TTS model can synthesize one at a time without losing prosody. The
heuristic is intentionally conservative: when in doubt, **don't split**.
Producing one over-long sentence is better than producing two sentences
that should have been one (which would make the model re-prime its prosody
and sound robotic at the seam).

Rules (all anchored on common European punctuation):

1. A *terminator* is one of `.`, `!`, `?`, `…`, or the literal three-dot
   ellipsis `...`. After the terminator we need whitespace + an uppercase
   letter (Latin or German umlaut) to confirm we're really starting a new
   sentence. Lowercase next-char → not a sentence boundary.

2. The word *before* a terminator must not be a known abbreviation
   (`Dr.`, `Mr.`, `Prof.`, `vs.`, `etc.`, German `bzw.`, `ggf.`, …) and
   must not be a bare number (so `19. November` survives).

3. After splitting, any single chunk longer than `MAX_SENTENCE_CHARS`
   gets chopped further on the nearest comma. This keeps per-upstream
   call latency bounded — vllm-omni is OK with ~1500-char inputs but
   queues badly above that.

Languages where punctuation differs significantly (Arabic, Hindi) fall
back to a single chunk per call. Document the limitation; users with
those languages can pre-split client-side if needed.
"""

from __future__ import annotations

import re
from typing import Final


# Words ending with `.` that should NOT terminate a sentence.
# Stored *lowercased*, no trailing dot. The matcher strips the dot before
# checking.
_KNOWN_ABBREVIATIONS: Final[frozenset[str]] = frozenset({
    # English titles + Latin
    "dr", "mr", "mrs", "ms", "prof", "rev", "st", "sr", "jr",
    "vs", "etc", "cf", "ie", "eg", "i.e", "e.g",
    "inc", "ltd", "co", "corp", "no", "nr", "vol",
    # German
    "frau", "herr", "ing", "dipl",
    "jh", "jhr", "bzw", "evtl", "ggf", "uvm",
    "z", "u", "z.b", "u.a", "u.s.w",
    # Italian
    "sig", "sigg", "dott", "ing",
    # French
    "m", "mlle", "mme", "mm",
    # Spanish/Portuguese
    "sr", "sra", "snr", "snra",
})

# Regex: a terminator `.!?…` or literal `...` ending. We match into one
# capture group so we can reach back via the matched span.
_TERMINATOR_RE: Final[re.Pattern[str]] = re.compile(
    r'([.!?…]|\.{3})',
    re.UNICODE,
)

# After a terminator we want whitespace + uppercase. Use an inclusive
# unicode "uppercase letter" class via char-class enumeration; sticking
# to A-Z plus common Latin extensions keeps this readable.
_NEXT_SENTENCE_RE: Final[re.Pattern[str]] = re.compile(
    r'\s+(?=[A-ZÄÖÜÉÈÊËÀÂÎÏÔÖÙÛÜÇÑ])',
    re.UNICODE,
)

# How long a single chunk may be before we soft-split on commas.
MAX_SENTENCE_CHARS: Final[int] = 1500


def split_sentences(text: str) -> list[str]:
    """Split `text` into sentence-sized chunks.

    Returns a list of non-empty stripped strings. An empty / whitespace-only
    input returns `[]`. A single sentence without a terminator returns a
    single-element list (verbatim minus surrounding whitespace).

    The function is deterministic and side-effect-free; tests should drive
    it with table inputs.
    """
    text = text.strip()
    if not text:
        return []

    # Greedy single-pass split. Walk every terminator candidate; for each,
    # decide whether to commit a split here.
    chunks: list[str] = []
    last_split = 0
    for m in _TERMINATOR_RE.finditer(text):
        term_end = m.end()
        # Need whitespace+uppercase right after — otherwise this terminator
        # is mid-word punctuation (URL dots, decimal numbers, etc.) and we
        # do not split.
        boundary = _NEXT_SENTENCE_RE.match(text, pos=term_end)
        if not boundary:
            continue

        # Look at the word just before the terminator to filter abbreviations
        # and bare numbers. We slice from `last_split` not `0` so abbreviations
        # earlier in the text don't poison this check.
        candidate_sentence = text[last_split:term_end]
        word_region = candidate_sentence[:-len(m.group(0))]
        prev_word = _trailing_word(word_region)
        if prev_word is not None:
            lower = prev_word.lower()
            if lower in _KNOWN_ABBREVIATIONS:
                continue
            if lower.isdigit() and m.group(0) == ".":
                # Distinguish `19. November` (numbered prefix — don't split)
                # from `3.14.` (decimal terminator — split). The decimal
                # case has a `.` immediately before the digit run.
                before_word = word_region[: -len(prev_word)].rstrip()
                if not before_word.endswith("."):
                    continue

        sentence = candidate_sentence.strip()
        if sentence:
            chunks.append(sentence)
        last_split = boundary.end()

    tail = text[last_split:].strip()
    if tail:
        chunks.append(tail)

    # Soft-split any over-long chunk on commas to keep per-upstream
    # latency bounded.
    out: list[str] = []
    for chunk in chunks:
        if len(chunk) <= MAX_SENTENCE_CHARS:
            out.append(chunk)
            continue
        out.extend(_soft_split_on_commas(chunk, MAX_SENTENCE_CHARS))
    return out


def _trailing_word(s: str) -> str | None:
    """Return the trailing word (run of letters/digits) of `s`, or None."""
    m = re.search(r'([A-Za-zÄÖÜäöüß0-9]+)\s*$', s)
    return m.group(1) if m else None


def _soft_split_on_commas(chunk: str, max_chars: int) -> list[str]:
    """Split a long chunk on commas so no piece exceeds `max_chars`.

    If even after comma-splitting some piece is still over the limit,
    we return that piece verbatim — the upstream call will be slow, but
    the caller is at least informed (we don't truncate text).
    """
    pieces: list[str] = []
    current = ""
    # Keep the comma attached to the LEFT side ("Hello, world" → "Hello," + " world").
    # `re.split` with capture group preserves separators.
    parts = re.split(r'(,\s+)', chunk)
    for token in parts:
        if not token:
            continue
        if len(current) + len(token) <= max_chars:
            current += token
        else:
            if current.strip():
                pieces.append(current.strip())
            current = token
    if current.strip():
        pieces.append(current.strip())
    # If a single comma-bounded piece still exceeds max_chars, just emit
    # it verbatim — partial truncation would cut audio mid-thought.
    return pieces or [chunk]
