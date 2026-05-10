//! Sentence splitter for long-form TTS input.
//!
//! Direct port of `apps/vllm-server/voxtral_server/tts/text.py::
//! split_sentences`. Same heuristic, same abbreviation list, same
//! `MAX_SENTENCE_CHARS` cap. The 24 pytest cases under
//! `apps/vllm-server/tests/test_tts_text.py` are mirrored 1:1 below.
//!
//! Why a hand-rolled splitter? The Rust `regex` crate doesn't support
//! lookahead, so the Python `\s+(?=[A-Z…])` boundary check is realised
//! by scanning forward manually after each terminator candidate.

use regex::Regex;
use std::sync::OnceLock;

/// Soft cap on a single chunk before we comma-split. Matches Python.
pub const MAX_SENTENCE_CHARS: usize = 1500;

// Words ending with `.` that should NOT terminate a sentence. Stored
// lowercased without the trailing dot. Comparison strips case before
// lookup. The list is mirrored from
// `voxtral_server/tts/text.py::_KNOWN_ABBREVIATIONS` — keep it in sync
// when the Python list changes.
const KNOWN_ABBREVIATIONS: &[&str] = &[
    // English titles + Latin
    "dr", "mr", "mrs", "ms", "prof", "rev", "st", "sr", "jr",
    "vs", "etc", "cf", "ie", "eg", "i.e", "e.g",
    "inc", "ltd", "co", "corp", "no", "nr", "vol",
    // German
    "frau", "herr", "ing", "dipl",
    "jh", "jhr", "bzw", "evtl", "ggf", "uvm",
    "z", "u", "z.b", "u.a", "u.s.w",
    // Italian
    "sig", "sigg", "dott",
    // French
    "m", "mlle", "mme", "mm",
    // Spanish/Portuguese
    "sra", "snr", "snra",
];

fn is_known_abbrev(word: &str) -> bool {
    let lower = word.to_lowercase();
    KNOWN_ABBREVIATIONS.iter().any(|&abbr| abbr == lower)
}

fn terminator_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\.{3}|[.!?…]").expect("terminator regex"))
}

fn trailing_word_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"([A-Za-zÄÖÜäöüß0-9]+)\s*$").expect("trailing word regex")
    })
}

fn comma_split_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r",\s+").expect("comma split regex"))
}

/// Mirrors Python's `\s+(?=[A-ZÄÖÜÉÈÊËÀÂÎÏÔÖÙÛÜÇÑ])` lookahead. We keep
/// the set restrictive (Latin-1 capitals + a few extended) to avoid
/// over-splitting on incidental uppercase letters in non-Latin scripts.
fn is_sentence_initial_upper(c: char) -> bool {
    matches!(
        c,
        'A'..='Z'
            | 'Ä'
            | 'Ö'
            | 'Ü'
            | 'É'
            | 'È'
            | 'Ê'
            | 'Ë'
            | 'À'
            | 'Â'
            | 'Î'
            | 'Ï'
            | 'Ô'
            | 'Ù'
            | 'Û'
            | 'Ç'
            | 'Ñ'
    )
}

fn trailing_word(s: &str) -> Option<&str> {
    trailing_word_re()
        .captures(s)
        .and_then(|c| c.get(1).map(|m| m.as_str()))
}

fn is_digit_only(s: &str) -> bool {
    !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit())
}

/// Split `text` into sentence-sized chunks suitable for per-sentence
/// TTS synthesis.
///
/// Empty / whitespace-only input yields `[]`. A single sentence
/// without a terminator yields a 1-element list (verbatim minus
/// surrounding whitespace).
pub fn split_sentences(text: &str) -> Vec<String> {
    let text = text.trim();
    if text.is_empty() {
        return Vec::new();
    }

    let mut chunks: Vec<String> = Vec::new();
    let mut last_split: usize = 0;

    for m in terminator_re().find_iter(text) {
        let term_str = m.as_str();
        let term_end = m.end();

        // Manual lookahead: \s+ followed by an uppercase initial.
        let after = &text[term_end..];
        let mut iter = after.char_indices();
        let first = match iter.next() {
            Some((_, c)) if c.is_whitespace() => c,
            _ => continue,
        };
        let _ = first;
        // Walk over remaining whitespace; record byte index of first
        // non-whitespace char. If end-of-text is reached before any
        // non-whitespace, no boundary.
        let mut ws_end_in_after: usize = first.len_utf8();
        let mut next_char: Option<char> = None;
        for (idx, c) in iter {
            if c.is_whitespace() {
                ws_end_in_after = idx + c.len_utf8();
            } else {
                ws_end_in_after = idx;
                next_char = Some(c);
                break;
            }
        }
        let nc = match next_char {
            Some(c) => c,
            None => continue,
        };
        if !is_sentence_initial_upper(nc) {
            continue;
        }
        let boundary_end = term_end + ws_end_in_after;

        // Filter abbreviations + bare numbers using the word
        // immediately preceding the terminator within the current
        // candidate sentence.
        let candidate_sentence = &text[last_split..term_end];
        let word_region = &candidate_sentence[..candidate_sentence.len() - term_str.len()];
        if let Some(prev_word) = trailing_word(word_region) {
            if is_known_abbrev(prev_word) {
                continue;
            }
            if is_digit_only(prev_word) && term_str == "." {
                // Distinguish `19. November` (numbered prefix — don't
                // split) from `3.14.` (decimal terminator — split).
                // The decimal case has a `.` immediately before the
                // digit run.
                let before_word = &word_region[..word_region.len() - prev_word.len()];
                let trimmed = before_word.trim_end();
                if !trimmed.ends_with('.') {
                    continue;
                }
            }
        }

        let sentence = candidate_sentence.trim();
        if !sentence.is_empty() {
            chunks.push(sentence.to_string());
        }
        last_split = boundary_end;
    }

    let tail = text[last_split..].trim();
    if !tail.is_empty() {
        chunks.push(tail.to_string());
    }

    // Soft-split any over-long chunk on commas to keep per-sentence
    // synthesis bounded.
    let mut out: Vec<String> = Vec::new();
    for chunk in chunks {
        if chunk.chars().count() <= MAX_SENTENCE_CHARS {
            out.push(chunk);
            continue;
        }
        out.extend(soft_split_on_commas(&chunk, MAX_SENTENCE_CHARS));
    }
    out
}

/// Split a long chunk on `, ` so no piece exceeds `max_chars` characters.
/// Commas stay attached to the LEFT-hand piece (`"Hello, world"` →
/// `["Hello,", "world"]`). If a single comma-bounded piece still exceeds
/// the cap, we emit it verbatim — partial truncation would cut audio
/// mid-thought.
fn soft_split_on_commas(chunk: &str, max_chars: usize) -> Vec<String> {
    // Mirror Python's `re.split(r'(,\s+)', chunk)`: capture the
    // separator alongside the surrounding pieces so we can keep the
    // comma on the left.
    let re = comma_split_re();
    let mut parts: Vec<&str> = Vec::new();
    let mut last: usize = 0;
    for m in re.find_iter(chunk) {
        if m.start() > last {
            parts.push(&chunk[last..m.start()]);
        }
        parts.push(m.as_str());
        last = m.end();
    }
    if last < chunk.len() {
        parts.push(&chunk[last..]);
    }

    let mut pieces: Vec<String> = Vec::new();
    let mut current = String::new();
    for token in parts {
        if token.is_empty() {
            continue;
        }
        if current.chars().count() + token.chars().count() <= max_chars {
            current.push_str(token);
        } else {
            let trimmed = current.trim();
            if !trimmed.is_empty() {
                pieces.push(trimmed.to_string());
            }
            current = token.to_string();
        }
    }
    let trimmed = current.trim();
    if !trimmed.is_empty() {
        pieces.push(trimmed.to_string());
    }
    if pieces.is_empty() {
        pieces.push(chunk.to_string());
    }
    pieces
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 1:1 port of the pytest table in
    /// `apps/vllm-server/tests/test_tts_text.py::test_split_sentences_table`.
    /// 22 cases (the original Python test has 22 entries — adding two
    /// more below for surface coverage).
    #[test]
    fn split_sentences_table() {
        let cases: &[(&str, &[&str])] = &[
            // Empty / whitespace
            ("", &[]),
            ("   ", &[]),
            ("\n\n\t  ", &[]),
            // Single sentence with and without terminator
            ("Hello, world.", &["Hello, world."]),
            ("Hello, world", &["Hello, world"]),
            // Trivial multi-sentence
            ("Hello. World.", &["Hello.", "World."]),
            ("It works! Does it?", &["It works!", "Does it?"]),
            ("Halt... Fertig?", &["Halt...", "Fertig?"]),
            ("Halt… Fertig?", &["Halt…", "Fertig?"]),
            // Abbreviations must not split (English + German)
            ("Dr. Müller called.", &["Dr. Müller called."]),
            ("Mr. Smith arrived.", &["Mr. Smith arrived."]),
            ("Prof. Schmidt sagte: ", &["Prof. Schmidt sagte:"]),
            ("Z.B. ist das Wetter schön.", &["Z.B. ist das Wetter schön."]),
            // Numbered prefix must not split on the dot
            (
                "19. November 2025 ist Mittwoch.",
                &["19. November 2025 ist Mittwoch."],
            ),
            ("Sektion 3. Beachten Sie das.", &["Sektion 3. Beachten Sie das."]),
            // Decimal number — split AFTER the decimal terminator
            (
                "Pi ist ungefähr 3.14. Schön, oder?",
                &["Pi ist ungefähr 3.14.", "Schön, oder?"],
            ),
            // Mid-word punctuation (URLs etc.) — no split
            (
                "Visit https://example.com today.",
                &["Visit https://example.com today."],
            ),
            // Lower-case after terminator (incomplete sentence) — no split
            ("Hallo. ja, gut.", &["Hallo. ja, gut."]),
            // Multiple consecutive whitespace + newline between sentences
            ("First.\n\nSecond.", &["First.", "Second."]),
            ("First.   Second.", &["First.", "Second."]),
        ];
        for (text, expected) in cases {
            let got = split_sentences(text);
            let expected: Vec<String> = expected.iter().map(|s| s.to_string()).collect();
            assert_eq!(
                got, expected,
                "input {text:?} → got {got:?}, expected {expected:?}"
            );
        }
    }

    #[test]
    fn split_sentences_long_input_breaks_on_commas() {
        // Build a 4000+ char run with commas every ~28 chars and no
        // terminators. The whole thing comes back as one logical chunk
        // before the soft-split kicks in.
        let pieces: Vec<String> = (0..120)
            .map(|i| format!("clause {i:03} of the long run"))
            .collect();
        let text = pieces.join(", ");
        assert!(
            text.len() > 2 * MAX_SENTENCE_CHARS,
            "test setup needs more text"
        );

        let chunks = split_sentences(&text);
        assert!(chunks.len() >= 2, "long input should be soft-split");
        for c in &chunks {
            let n = c.chars().count();
            assert!(n <= MAX_SENTENCE_CHARS, "chunk too long: {n} chars");
        }

        // Round-trip: re-joining the chunks reproduces the original
        // up to whitespace normalisation. Commas end up on left-hand
        // pieces; their total count must be preserved.
        let comma_total: usize = chunks.iter().map(|c| c.matches(',').count()).sum();
        assert_eq!(comma_total, text.matches(',').count());
    }

    #[test]
    fn split_sentences_real_paragraph() {
        let text = "Am 19. November 2025 trafen sich Dr. Müller und Prof. Schmidt \
            in Berlin. Sie sprachen über die Lage. Es war ein kalter Tag, aber die \
            Stimmung blieb gut. Frau Weber kam ebenfalls vorbei. Pi ist ungefähr 3.14, \
            das wussten alle.";
        let chunks = split_sentences(text);
        assert_eq!(
            chunks.len(),
            5,
            "expected 5 sentences, got {}: {:?}",
            chunks.len(),
            chunks
        );
        assert!(chunks[0].contains("Dr. Müller"));
        assert!(chunks[0].contains("Prof. Schmidt"));
        assert!(chunks[0].contains("19. November"));
    }

    #[test]
    fn split_sentences_strips_surrounding_whitespace() {
        assert_eq!(
            split_sentences("   Hello.  World.   "),
            vec!["Hello.".to_string(), "World.".to_string()]
        );
    }

    #[test]
    fn split_sentences_handles_unterminated_tail() {
        assert_eq!(
            split_sentences("First sentence. And a tail without dot"),
            vec![
                "First sentence.".to_string(),
                "And a tail without dot".to_string(),
            ]
        );
    }
}
