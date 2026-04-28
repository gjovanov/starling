//! Config types for the flow-matching acoustic transformer.
//!
//! Mirrors the upstream Python dataclasses
//! `AcousticTransformerArgs` and `MultimodalAudioModelArgs` from
//! `voxtral_tts_audio_generation.py`. Loadable directly from
//! `models/cache/tts/params.json` via the `serde_json` path —
//! see [`MultimodalAudioModelArgs::from_params_json_path`].

use anyhow::{anyhow, Context, Result};
use serde::Deserialize;
use std::path::Path;

/// Acoustic-transformer (flow-matching velocity field) hyper-parameters.
///
/// For Voxtral-4B-TTS-2603 the values are:
/// `dim=3072, n_layers=3, head_dim=128, hidden_dim=9216,
///  n_heads=32, n_kv_heads=8, rope_theta=10_000` (RoPE θ different from
/// the AR LLM's 1M, but the velocity field doesn't apply RoPE — it is
/// bidirectional over a 3-token window of [x, t_emb, llm_hidden]).
#[derive(Clone, Debug, Deserialize)]
pub struct AcousticTransformerArgs {
    pub input_dim: usize,
    pub dim: usize,
    pub n_layers: usize,
    pub head_dim: usize,
    pub hidden_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    #[serde(default)]
    pub use_biases: bool,
    /// RoPE θ — present in the upstream dataclass but unused here, the
    /// velocity field is bidirectional with no RoPE applied. Kept for
    /// schema compatibility with `params.json`.
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_norm_eps")]
    pub norm_eps: f64,
    /// Flow-matching noise floor.
    #[serde(default = "default_sigma")]
    pub sigma: f64,
    /// Maximum noise scale at t=0. Always 1.0 in the published config.
    #[serde(default = "default_sigma_max")]
    pub sigma_max: f64,
}

fn default_rope_theta() -> f64 {
    10_000.0
}
fn default_norm_eps() -> f64 {
    1e-5
}
fn default_sigma() -> f64 {
    1e-5
}
fn default_sigma_max() -> f64 {
    1.0
}

/// Top-level audio-model config — wraps [`AcousticTransformerArgs`] and
/// adds codebook-size knobs that govern the output projections.
#[derive(Clone, Debug, Deserialize)]
pub struct MultimodalAudioModelArgs {
    pub semantic_codebook_size: usize,
    pub acoustic_codebook_size: usize,
    pub n_acoustic_codebook: usize,
    pub acoustic_transformer_args: AcousticTransformerArgs,
    #[serde(default)]
    pub p_uncond: f64,
    /// Special-token id used for the CFG null condition. Not used at
    /// inference time (the upstream null condition is `llm_hidden = 0`),
    /// but parsed for completeness.
    #[serde(default = "default_condition_dropped_token_id")]
    pub condition_dropped_token_id: u32,
}

fn default_condition_dropped_token_id() -> u32 {
    42
}

impl MultimodalAudioModelArgs {
    /// Load `multimodal.audio_model_args` from a path to `params.json`.
    pub fn from_params_json_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = std::fs::read(path)
            .with_context(|| format!("reading {}", path.display()))?;
        let v: serde_json::Value = serde_json::from_slice(&bytes)
            .with_context(|| format!("parsing JSON from {}", path.display()))?;
        let inner = v
            .get("multimodal")
            .and_then(|m| m.get("audio_model_args"))
            .cloned()
            .ok_or_else(|| {
                anyhow!(
                    "{} is missing the multimodal.audio_model_args object",
                    path.display()
                )
            })?;
        Ok(serde_json::from_value(inner)
            .with_context(|| "deserialising multimodal.audio_model_args")?)
    }

    /// Number of distinct codebooks per audio frame: 1 semantic + N
    /// acoustic. Always 37 for Voxtral-4B-TTS.
    pub fn num_codebooks(&self) -> usize {
        1 + self.n_acoustic_codebook
    }

    /// Number of quantization levels per acoustic codebook = the same
    /// value as `acoustic_codebook_size` (typically 21).
    pub fn acoustic_levels(&self) -> usize {
        self.acoustic_codebook_size
    }

    /// Padded semantic-output dimension: codebook_size + special tokens
    /// rounded up to multiple of 128.
    ///
    /// Always 8320 for Voxtral-4B-TTS-2603 (8192 + 2 special tokens →
    /// 8194 → ceil to 8320 via 128-aligned padding).
    pub fn semantic_output_dim(&self) -> usize {
        let raw = self.semantic_codebook_size + AudioSpecialTokens::COUNT;
        // Round up to multiple of 128.
        let mul = 128;
        ((raw + mul - 1) / mul) * mul
    }
}

/// Special-token IDs reserved by the audio quantizer. These are NOT
/// part of the codebook — they are predicted by codebook-0 head when
/// the AR LLM signals end-of-audio or empty-audio.
///
/// Upstream `AudioSpecialTokens` enum has `empty_audio = 0` and
/// `end_audio = 1` (in iteration order).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AudioSpecialTokens;

impl AudioSpecialTokens {
    pub const EMPTY_AUDIO: u32 = 0;
    pub const END_AUDIO: u32 = 1;
    pub const COUNT: usize = 2;
}

/// Decoder constants — hard-coded in the upstream class, mirrored here.
#[derive(Clone, Copy, Debug)]
pub struct FlowMatchingDecodeArgs {
    /// 8 timesteps × `linspace(0, 1)` → 7 Euler steps.
    pub decode_iters: usize,
    /// Classifier-Free Guidance scale. `α=1.2` upstream.
    pub cfg_alpha: f64,
    /// Multiplier on the initial noise sample. `1.0` upstream.
    pub noise_scale: f64,
}

impl Default for FlowMatchingDecodeArgs {
    fn default() -> Self {
        Self {
            decode_iters: 8,
            cfg_alpha: 1.2,
            noise_scale: 1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params_path() -> std::path::PathBuf {
        std::env::var_os("STARLING_TTS_PARAMS")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                std::path::PathBuf::from(
                    "/home/gjovanov/gjovanov/starling/models/cache/tts/params.json",
                )
            })
    }

    #[test]
    fn parses_real_params_json_when_present() {
        let path = params_path();
        if !path.exists() {
            eprintln!("skipping: {} not present", path.display());
            return;
        }
        let args = MultimodalAudioModelArgs::from_params_json_path(&path).unwrap();
        assert_eq!(args.semantic_codebook_size, 8192);
        assert_eq!(args.acoustic_codebook_size, 21);
        assert_eq!(args.n_acoustic_codebook, 36);
        assert_eq!(args.num_codebooks(), 37);
        assert_eq!(args.acoustic_levels(), 21);
        assert_eq!(args.semantic_output_dim(), 8320);

        let at = &args.acoustic_transformer_args;
        assert_eq!(at.input_dim, 3072);
        assert_eq!(at.dim, 3072);
        assert_eq!(at.n_layers, 3);
        assert_eq!(at.head_dim, 128);
        assert_eq!(at.hidden_dim, 9216);
        assert_eq!(at.n_heads, 32);
        assert_eq!(at.n_kv_heads, 8);
        assert_eq!(at.use_biases, false);
        assert!((at.rope_theta - 10_000.0).abs() < 1.0);
        assert!((at.sigma_max - 1.0).abs() < 1e-9);
    }

    #[test]
    fn special_tokens_count_matches_upstream() {
        assert_eq!(AudioSpecialTokens::COUNT, 2);
    }

    #[test]
    fn semantic_output_dim_padding() {
        // Hand-computed: 8192 + 2 → 8194 → next multiple of 128 = 8320.
        let args = MultimodalAudioModelArgs {
            semantic_codebook_size: 8192,
            acoustic_codebook_size: 21,
            n_acoustic_codebook: 36,
            acoustic_transformer_args: AcousticTransformerArgs {
                input_dim: 3072,
                dim: 3072,
                n_layers: 3,
                head_dim: 128,
                hidden_dim: 9216,
                n_heads: 32,
                n_kv_heads: 8,
                use_biases: false,
                rope_theta: 10_000.0,
                norm_eps: 1e-5,
                sigma: 1e-5,
                sigma_max: 1.0,
            },
            p_uncond: 0.0,
            condition_dropped_token_id: 42,
        };
        assert_eq!(args.semantic_output_dim(), 8320);
    }

    #[test]
    fn default_decode_args_match_upstream() {
        let d = FlowMatchingDecodeArgs::default();
        assert_eq!(d.decode_iters, 8);
        assert!((d.cfg_alpha - 1.2).abs() < 1e-9);
        assert!((d.noise_scale - 1.0).abs() < 1e-9);
    }
}
