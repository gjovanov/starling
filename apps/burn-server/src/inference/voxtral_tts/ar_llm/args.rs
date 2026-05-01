//! Top-level AR LLM hyper-parameters parsed from `params.json`.
//!
//! Unlike the FMA + codec args (nested under `multimodal.*`), the AR
//! LLM args sit at the JSON root. Same shape as the Voxtral-Mini-
//! Realtime ASR decoder so we share the validator-friendly defaults.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::Path;

#[derive(Clone, Debug, Deserialize)]
pub struct ArLlmArgs {
    /// Hidden dim. 3072 in Voxtral-4B-TTS-2603.
    pub dim: usize,
    /// Number of transformer layers. 26 in Voxtral-4B-TTS-2603.
    pub n_layers: usize,
    pub head_dim: usize,
    /// FFN inner dim. 9216 (3 × dim) in Voxtral-4B-TTS-2603.
    pub hidden_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    #[serde(default)]
    pub use_biases: bool,
    /// `1_000_000` in Voxtral-4B-TTS-2603 (matches the ASR decoder's
    /// long-context RoPE).
    pub rope_theta: f64,
    #[serde(default = "d_norm_eps")]
    pub norm_eps: f64,
    pub vocab_size: usize,
    /// `true` upstream — `lm_head.weight` is `tok_embeddings.weight`
    /// (no separate output projection).
    #[serde(default = "d_true")]
    pub tied_embeddings: bool,
    #[serde(default = "d_max_seq_len")]
    pub max_seq_len: usize,
}

fn d_true() -> bool {
    true
}
fn d_norm_eps() -> f64 {
    1e-5
}
fn d_max_seq_len() -> usize {
    65536
}

impl ArLlmArgs {
    pub fn from_params_json_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = std::fs::read(path)
            .with_context(|| format!("reading {}", path.display()))?;
        // params.json is a flat map at the root for the AR LLM fields.
        // We use a subset-projection via serde_json::Value to skip the
        // multimodal-only branches without complaining about extra keys.
        let v: serde_json::Value = serde_json::from_slice(&bytes)
            .with_context(|| format!("parsing JSON from {}", path.display()))?;
        Ok(serde_json::from_value(v)
            .with_context(|| "deserialising AR LLM args from params.json")?)
    }

    /// Number of K/V heads per Q head — the GQA repeat factor. For
    /// Voxtral-4B-TTS-2603: `32 / 8 = 4`.
    pub fn gqa_repeats(&self) -> usize {
        self.n_heads / self.n_kv_heads
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn params_path() -> PathBuf {
        std::env::var_os("STARLING_TTS_PARAMS")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from("/home/gjovanov/gjovanov/starling/models/cache/tts/params.json")
            })
    }

    #[test]
    fn parses_real_params_when_present() {
        let path = params_path();
        if !path.exists() {
            eprintln!("skipping: {} not present", path.display());
            return;
        }
        let args = ArLlmArgs::from_params_json_path(&path).unwrap();
        assert_eq!(args.dim, 3072);
        assert_eq!(args.n_layers, 26);
        assert_eq!(args.head_dim, 128);
        assert_eq!(args.hidden_dim, 9216);
        assert_eq!(args.n_heads, 32);
        assert_eq!(args.n_kv_heads, 8);
        assert!(!args.use_biases);
        assert!((args.rope_theta - 1_000_000.0).abs() < 1.0);
        assert_eq!(args.vocab_size, 131072);
        assert!(args.tied_embeddings);
        assert_eq!(args.gqa_repeats(), 4);
    }
}
