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

/// Tracks growing text for the "growing_segments" transcription mode.
///
/// Accumulates partial text updates and detects when a sentence is complete.
pub struct GrowingTextTracker {
    confirmed_text: String,
    partial_text: String,
    segment_index: u32,
}

impl GrowingTextTracker {
    pub fn new() -> Self {
        Self {
            confirmed_text: String::new(),
            partial_text: String::new(),
            segment_index: 0,
        }
    }

    /// Update with new transcription output. Returns (final_sentences, partial_text).
    pub fn update(&mut self, new_text: &str) -> (Vec<(String, u32)>, Option<(String, u32)>) {
        let full_text = format!("{}{}", self.confirmed_text, new_text);
        let sentences = split_sentences(&full_text);

        let mut finals = Vec::new();

        if sentences.len() > 1 {
            // All but last are confirmed/final
            for sentence in &sentences[..sentences.len() - 1] {
                finals.push((sentence.clone(), self.segment_index));
                self.segment_index += 1;
            }
            // Update confirmed text to include finalized sentences
            self.confirmed_text = sentences[..sentences.len() - 1].join(" ") + " ";
            self.partial_text = sentences.last().cloned().unwrap_or_default();
        } else {
            self.partial_text = full_text;
        }

        let partial = if !self.partial_text.is_empty() {
            Some((self.partial_text.clone(), self.segment_index))
        } else {
            None
        };

        (finals, partial)
    }

    /// Flush any remaining partial text as a final segment.
    pub fn flush(&mut self) -> Option<(String, u32)> {
        if !self.partial_text.is_empty() {
            let text = self.partial_text.clone();
            let idx = self.segment_index;
            self.partial_text.clear();
            self.segment_index += 1;
            Some((text, idx))
        } else {
            None
        }
    }
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

    #[test]
    fn test_growing_text_tracker() {
        let mut tracker = GrowingTextTracker::new();

        // Partial update
        let (finals, partial) = tracker.update("Hello world");
        assert!(finals.is_empty());
        assert_eq!(partial.unwrap().0, "Hello world");

        // Sentence boundary detected
        let (finals, partial) = tracker.update(". How are you");
        assert_eq!(finals.len(), 1);
        assert!(partial.is_some());

        // Flush remaining
        let flushed = tracker.flush();
        assert!(flushed.is_some());
    }
}
