//! Minimal Tekken tokenizer — decode-only (token IDs → text).
//!
//! Voxtral uses Mistral's Tekken tokenizer (v7, BPE). For inference we only
//! need to decode generated token IDs back to text, not encode.

#[cfg(feature = "native")]
use std::path::Path;

/// Simple token ID to text decoder loaded from tekken.json.
pub struct TekkenDecoder {
    vocab: Vec<String>,
}

impl TekkenDecoder {
    /// Load from a tekken.json file.
    #[cfg(feature = "native")]
    pub fn from_file(path: &Path) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read tokenizer: {}", e))?;
        Self::from_json(&content)
    }

    /// Parse from JSON string.
    pub fn from_json(json: &str) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let data: serde_json::Value =
            serde_json::from_str(json).map_err(|e| format!("Invalid JSON: {}", e))?;

        // Tekken format: {"config": {...}, "vocab": [...], "special_tokens": [...]}
        // vocab is an array of base64-encoded byte strings
        let vocab_array = data
            .get("vocab")
            .and_then(|v| v.as_array())
            .ok_or("Missing 'vocab' array in tokenizer JSON")?;

        let mut vocab = Vec::with_capacity(vocab_array.len());
        for entry in vocab_array {
            let text = if let Some(s) = entry.as_str() {
                // Simple string entry (base64-encoded)
                let bytes = base64::Engine::decode(
                    &base64::engine::general_purpose::STANDARD,
                    s,
                )
                .unwrap_or_default();
                String::from_utf8_lossy(&bytes).to_string()
            } else if let Some(obj) = entry.as_object() {
                // Dict entry: {"rank": N, "token_bytes": "...", "token_str": "..."}
                if let Some(token_bytes) = obj.get("token_bytes").and_then(|v| v.as_str()) {
                    let bytes = base64::Engine::decode(
                        &base64::engine::general_purpose::STANDARD,
                        token_bytes,
                    )
                    .unwrap_or_default();
                    String::from_utf8_lossy(&bytes).to_string()
                } else if let Some(token_str) = obj.get("token_str").and_then(|v| v.as_str()) {
                    token_str.to_string()
                } else {
                    String::new()
                }
            } else {
                String::new()
            };
            vocab.push(text);
        }

        // Append special tokens
        if let Some(special) = data.get("special_tokens").and_then(|v| v.as_array()) {
            // Ensure vocab is large enough
            let config_vocab_size = data
                .get("config")
                .and_then(|c| c.get("default_vocab_size"))
                .and_then(|v| v.as_u64())
                .unwrap_or(131072) as usize;

            while vocab.len() < config_vocab_size {
                vocab.push(String::new());
            }

            // Special tokens go at the end (after default_vocab_size)
            for token in special {
                if let Some(text) = token.as_str() {
                    vocab.push(text.to_string());
                }
            }
        }

        eprintln!("[Tekken] Loaded {} vocab entries", vocab.len());
        Ok(Self { vocab })
    }

    /// Decode token IDs to text.
    ///
    /// Voxtral token IDs 0-999 are control tokens (32=STREAMING_PAD, 33=STREAMING_WORD).
    /// Text token IDs start at 1000 and are offset by 1000 from tekken.json vocab indices:
    ///   token_id 1000 → vocab[0], token_id 1362 → vocab[362] (" I"), etc.
    pub fn decode(&self, token_ids: &[i32]) -> String {
        let mut result = String::new();
        for &id in token_ids {
            if id >= 1000 {
                let vocab_idx = (id - 1000) as usize;
                if vocab_idx < self.vocab.len() {
                    result.push_str(&self.vocab[vocab_idx]);
                }
            }
        }
        result
    }

    /// Vocab size.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_empty() {
        let decoder = TekkenDecoder {
            vocab: vec!["a".into(), "b".into()],
        };
        assert_eq!(decoder.decode(&[]), "");
    }

    #[test]
    fn test_decode_filters_special() {
        let mut vocab = vec![String::new(); 1100];
        vocab[1000] = "hello".into();
        vocab[1001] = " world".into();
        let decoder = TekkenDecoder { vocab };

        // Special tokens (< 1000) filtered out
        assert_eq!(decoder.decode(&[0, 32, 33, 1000, 1001]), "hello world");
    }
}
