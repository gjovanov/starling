"""Tests for the long-form sentence splitter.

The splitter's job is to chop user input into TTS-able chunks without
mis-splitting on abbreviations, numbered prefixes, decimals, etc. We
drive it with a table of representative inputs and assert the exact list
of chunks.
"""

from __future__ import annotations

import pytest

from voxtral_server.tts.text import (
    MAX_SENTENCE_CHARS,
    split_sentences,
)


@pytest.mark.parametrize(
    "text, expected",
    [
        # Empty / whitespace
        ("", []),
        ("   ", []),
        ("\n\n\t  ", []),

        # Single sentence with and without terminator
        ("Hello, world.", ["Hello, world."]),
        ("Hello, world", ["Hello, world"]),

        # Trivial multi-sentence
        ("Hello. World.", ["Hello.", "World."]),
        ("It works! Does it?", ["It works!", "Does it?"]),
        ("Halt... Fertig?", ["Halt...", "Fertig?"]),
        ("Halt… Fertig?", ["Halt…", "Fertig?"]),

        # Abbreviations must not split (English + German)
        ("Dr. Müller called.", ["Dr. Müller called."]),
        ("Mr. Smith arrived.", ["Mr. Smith arrived."]),
        ("Prof. Schmidt sagte: ", ["Prof. Schmidt sagte:"]),
        ("Z.B. ist das Wetter schön.", ["Z.B. ist das Wetter schön."]),

        # Numbered prefix must not split on the dot but does on next sentence
        ("19. November 2025 ist Mittwoch.", ["19. November 2025 ist Mittwoch."]),
        ("Sektion 3. Beachten Sie das.", ["Sektion 3. Beachten Sie das."]),

        # Decimal number — no split
        ("Pi ist ungefähr 3.14. Schön, oder?", ["Pi ist ungefähr 3.14.", "Schön, oder?"]),

        # Mid-word punctuation (URLs etc.) — no split
        ("Visit https://example.com today.", ["Visit https://example.com today."]),

        # Lower-case after terminator (incomplete sentence) — no split
        ("Hallo. ja, gut.", ["Hallo. ja, gut."]),

        # Multiple consecutive whitespace + newline between sentences
        ("First.\n\nSecond.", ["First.", "Second."]),
        ("First.   Second.", ["First.", "Second."]),
    ],
)
def test_split_sentences_table(text: str, expected: list[str]) -> None:
    assert split_sentences(text) == expected


def test_split_sentences_long_input_breaks_on_commas() -> None:
    """A very long single 'sentence' gets soft-split on commas to keep
    upstream calls bounded by MAX_SENTENCE_CHARS."""
    # Build a 4000-char run with commas every 60 chars. No terminators
    # → returned as one logical chunk before soft-split.
    pieces = [f"clause {i:03d} of the long run" for i in range(120)]
    text = ", ".join(pieces)
    assert len(text) > 2 * MAX_SENTENCE_CHARS, "test setup needs more text"

    chunks = split_sentences(text)
    assert len(chunks) >= 2, "long input should be soft-split"
    for c in chunks:
        # Each chunk must be <= MAX_SENTENCE_CHARS, except a single
        # comma-bounded piece longer than the cap (impossible here).
        assert len(c) <= MAX_SENTENCE_CHARS, f"chunk too long: {len(c)} chars"

    # Round-trip: re-joining the chunks reproduces the original up to
    # whitespace normalisation.
    rejoined = " ".join(chunks)
    # Preserve commas (they were captured into the left-hand piece).
    assert rejoined.count(",") == text.count(",")


def test_split_sentences_real_paragraph() -> None:
    """A realistic German news paragraph — sanity check we get a sensible
    sentence count, not over- or under-split."""
    text = (
        "Am 19. November 2025 trafen sich Dr. Müller und Prof. Schmidt "
        "in Berlin. Sie sprachen über die Lage. Es war ein kalter Tag, "
        "aber die Stimmung blieb gut. Frau Weber kam ebenfalls vorbei. "
        "Pi ist ungefähr 3.14, das wussten alle."
    )
    chunks = split_sentences(text)
    assert len(chunks) == 5, f"expected 5 sentences, got {len(chunks)}: {chunks}"
    assert "Dr. Müller" in chunks[0]
    assert "Prof. Schmidt" in chunks[0]
    assert "19. November" in chunks[0]


def test_split_sentences_strips_surrounding_whitespace() -> None:
    assert split_sentences("   Hello.  World.   ") == ["Hello.", "World."]


def test_split_sentences_handles_unterminated_tail() -> None:
    assert split_sentences("First sentence. And a tail without dot") == [
        "First sentence.",
        "And a tail without dot",
    ]
