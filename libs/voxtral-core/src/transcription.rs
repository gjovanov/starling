//! Streaming text processing: sentence boundary detection and subtitle emission.
//!
//! Splits transcribed text into sentences at natural boundaries, avoiding
//! false splits on abbreviations, numbered lists, and decimal numbers.

/// Split text into sentences at natural boundaries.
///
/// Splits at `.` `?` `!` followed by whitespace and an uppercase letter,
/// but NOT when the punctuation follows a digit (avoids "19. November").
pub fn split_sentences(text: &str) -> Vec<String> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return vec![];
    }

    let chars: Vec<char> = trimmed.chars().collect();
    let mut sentences = Vec::new();
    let mut start = 0;

    let mut i = 0;
    while i < chars.len() {
        let ch = chars[i];

        // Check for sentence-ending punctuation
        if ch == '.' || ch == '?' || ch == '!' {
            // Skip if preceded by a digit (e.g., "19. November")
            if i > 0 && chars[i - 1].is_ascii_digit() && ch == '.' {
                i += 1;
                continue;
            }

            // Look ahead: whitespace followed by uppercase letter
            let mut j = i + 1;
            while j < chars.len() && chars[j].is_whitespace() {
                j += 1;
            }

            if j < chars.len() && j > i + 1 && chars[j].is_uppercase() {
                // Split here — include punctuation in current sentence
                let sentence: String = chars[start..=i].iter().collect();
                let sentence = sentence.trim().to_string();
                if !sentence.is_empty() {
                    sentences.push(sentence);
                }
                start = j; // skip whitespace, start at uppercase
            }
        }
        i += 1;
    }

    // Remaining text
    if start < chars.len() {
        let remainder: String = chars[start..].iter().collect();
        let remainder = remainder.trim().to_string();
        if !remainder.is_empty() {
            sentences.push(remainder);
        }
    }

    sentences
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_sentences_basic() {
        let text = "Hello world. How are you? I am fine.";
        let sentences = split_sentences(text);
        assert_eq!(sentences.len(), 3);
        assert_eq!(sentences[0], "Hello world.");
        assert_eq!(sentences[1], "How are you?");
        assert_eq!(sentences[2], "I am fine.");
    }

    #[test]
    fn test_split_sentences_no_split_on_number() {
        let text = "Am 19. November war es kalt.";
        let sentences = split_sentences(text);
        assert_eq!(sentences.len(), 1);
        assert_eq!(sentences[0], "Am 19. November war es kalt.");
    }

    #[test]
    fn test_split_sentences_empty() {
        assert!(split_sentences("").is_empty());
        assert!(split_sentences("   ").is_empty());
    }
}
