//! Codec hyper-parameters parsed from `params.json`.
//!
//! Mirrors the upstream Python dataclass `AudioTokenizerArgs` from
//! `voxtral_tts_audio_tokenizer.py`. Loadable via
//! [`AudioTokenizerArgs::from_params_json_path`] which extracts the
//! `multimodal.audio_tokenizer_args` block.

use anyhow::{anyhow, Context, Result};
use serde::Deserialize;
use std::path::Path;

#[derive(Clone, Debug, Deserialize)]
pub struct AudioTokenizerArgs {
    // Audio settings
    #[serde(default = "d_channels")]
    pub channels: usize,
    #[serde(default = "d_sampling_rate")]
    pub sampling_rate: usize,
    #[serde(default = "d_pretransform_patch_size")]
    pub pretransform_patch_size: usize,
    #[serde(default = "d_patch_proj_kernel_size")]
    pub patch_proj_kernel_size: usize,

    // Quantizer settings
    #[serde(default = "d_semantic_codebook_size")]
    pub semantic_codebook_size: usize,
    #[serde(default = "d_semantic_dim")]
    pub semantic_dim: usize,
    #[serde(default = "d_acoustic_codebook_size")]
    pub acoustic_codebook_size: usize,
    #[serde(default = "d_acoustic_dim")]
    pub acoustic_dim: usize,

    // General architecture
    #[serde(default = "d_true")]
    pub conv_weight_norm: bool,
    #[serde(default = "d_true")]
    pub causal: bool,
    #[serde(default = "d_attn_sliding_window_size")]
    pub attn_sliding_window_size: usize,
    #[serde(default = "d_true")]
    pub half_attn_window_upon_downsampling: bool,
    #[serde(default = "d_dim")]
    pub dim: usize,
    #[serde(default = "d_hidden_dim")]
    pub hidden_dim: usize,
    #[serde(default = "d_head_dim")]
    pub head_dim: usize,
    #[serde(default = "d_n_heads")]
    pub n_heads: usize,
    #[serde(default = "d_n_kv_heads")]
    pub n_kv_heads: usize,
    #[serde(default = "d_qk_norm_eps")]
    pub qk_norm_eps: f64,
    #[serde(default = "d_true")]
    pub qk_norm: bool,
    #[serde(default)]
    pub use_biases: bool,
    #[serde(default = "d_norm_eps")]
    pub norm_eps: f64,
    #[serde(default = "d_true")]
    pub layer_scale: bool,
    /// `null` upstream selects a per-layer init schedule (0.1, 1e-5, 1e-6
    /// depending on `layer_id < 18 / ≤ 24 / >`); the published Voxtral-
    /// 4B-TTS-2603 config explicitly sets `0.01`. We store the value
    /// because the schedule is for training-time init and never reaches
    /// inference (the loaded checkpoint already has trained scales).
    #[serde(default)]
    pub layer_scale_init: Option<f64>,

    // Encoder strings — kept for params-schema compatibility; not used
    // by the decode-only port.
    #[serde(default = "d_encoder_transformer_lengths_str")]
    pub encoder_transformer_lengths_str: String,
    #[serde(default = "d_encoder_convs_kernels_str")]
    pub encoder_convs_kernels_str: String,
    #[serde(default = "d_encoder_convs_strides_str")]
    pub encoder_convs_strides_str: String,

    // Decoder strings — parsed lazily via the helpers below.
    #[serde(default = "d_decoder_transformer_lengths_str")]
    pub decoder_transformer_lengths_str: String,
    #[serde(default = "d_decoder_convs_kernels_str")]
    pub decoder_convs_kernels_str: String,
    #[serde(default = "d_decoder_convs_strides_str")]
    pub decoder_convs_strides_str: String,
}

fn d_true() -> bool {
    true
}
fn d_channels() -> usize {
    1
}
fn d_sampling_rate() -> usize {
    24000
}
fn d_pretransform_patch_size() -> usize {
    240
}
fn d_patch_proj_kernel_size() -> usize {
    7
}
fn d_semantic_codebook_size() -> usize {
    8192
}
fn d_semantic_dim() -> usize {
    256
}
fn d_acoustic_codebook_size() -> usize {
    21
}
fn d_acoustic_dim() -> usize {
    36
}
fn d_attn_sliding_window_size() -> usize {
    16
}
fn d_dim() -> usize {
    1024
}
fn d_hidden_dim() -> usize {
    4096
}
fn d_head_dim() -> usize {
    128
}
fn d_n_heads() -> usize {
    8
}
fn d_n_kv_heads() -> usize {
    8
}
fn d_qk_norm_eps() -> f64 {
    1e-6
}
fn d_norm_eps() -> f64 {
    1e-2
}
fn d_encoder_transformer_lengths_str() -> String {
    "2,2,2,2".into()
}
fn d_encoder_convs_kernels_str() -> String {
    "4,4,4,3".into()
}
fn d_encoder_convs_strides_str() -> String {
    "2,2,2,1".into()
}
fn d_decoder_transformer_lengths_str() -> String {
    "2,2,2,2".into()
}
fn d_decoder_convs_kernels_str() -> String {
    "3,4,4,4".into()
}
fn d_decoder_convs_strides_str() -> String {
    "1,2,2,2".into()
}

fn parse_csv_usize(s: &str) -> Result<Vec<usize>> {
    s.split(',')
        .map(|p| {
            p.trim()
                .parse::<usize>()
                .map_err(|e| anyhow!("parsing csv entry {p:?}: {e}"))
        })
        .collect()
}

impl AudioTokenizerArgs {
    pub fn from_params_json_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = std::fs::read(path)
            .with_context(|| format!("reading {}", path.display()))?;
        let v: serde_json::Value = serde_json::from_slice(&bytes)
            .with_context(|| format!("parsing JSON from {}", path.display()))?;
        let inner = v
            .get("multimodal")
            .and_then(|m| m.get("audio_tokenizer_args"))
            .cloned()
            .ok_or_else(|| {
                anyhow!(
                    "{} missing multimodal.audio_tokenizer_args",
                    path.display()
                )
            })?;
        Ok(serde_json::from_value(inner)
            .with_context(|| "deserialising multimodal.audio_tokenizer_args")?)
    }

    pub fn decoder_transformer_lengths(&self) -> Result<Vec<usize>> {
        parse_csv_usize(&self.decoder_transformer_lengths_str)
    }
    pub fn decoder_convs_kernels(&self) -> Result<Vec<usize>> {
        parse_csv_usize(&self.decoder_convs_kernels_str)
    }
    pub fn decoder_convs_strides(&self) -> Result<Vec<usize>> {
        parse_csv_usize(&self.decoder_convs_strides_str)
    }

    /// Codec frame rate: `sampling_rate / (patch_size × prod(decoder_strides))`.
    /// For Voxtral-4B-TTS-2603: 24000 / (240 × 8) = 12.5.
    pub fn frame_rate(&self) -> Result<f64> {
        let strides = self.decoder_convs_strides()?;
        let prod: usize = strides.iter().product();
        Ok((self.sampling_rate as f64) / ((self.pretransform_patch_size * prod) as f64))
    }

    /// Total samples per codec frame. For Voxtral-4B-TTS-2603: 1920.
    pub fn samples_per_frame(&self) -> Result<usize> {
        let strides = self.decoder_convs_strides()?;
        let prod: usize = strides.iter().product();
        Ok(self.pretransform_patch_size * prod)
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
        let args = AudioTokenizerArgs::from_params_json_path(&path).unwrap();
        assert_eq!(args.channels, 1);
        assert_eq!(args.sampling_rate, 24000);
        assert_eq!(args.pretransform_patch_size, 240);
        assert_eq!(args.semantic_codebook_size, 8192);
        assert_eq!(args.semantic_dim, 256);
        assert_eq!(args.acoustic_codebook_size, 21);
        assert_eq!(args.acoustic_dim, 36);
        assert_eq!(args.attn_sliding_window_size, 16);
        assert_eq!(args.dim, 1024);
        assert_eq!(args.hidden_dim, 4096);
        assert_eq!(args.n_heads, 8);
        assert_eq!(args.n_kv_heads, 8);
        assert!(args.qk_norm);
        assert!(args.layer_scale);
        assert_eq!(args.layer_scale_init, Some(0.01));
        assert_eq!(args.decoder_transformer_lengths().unwrap(), vec![2, 2, 2, 2]);
        assert_eq!(args.decoder_convs_kernels().unwrap(), vec![3, 4, 4, 4]);
        assert_eq!(args.decoder_convs_strides().unwrap(), vec![1, 2, 2, 2]);
        assert!((args.frame_rate().unwrap() - 12.5).abs() < 1e-9);
        assert_eq!(args.samples_per_frame().unwrap(), 1920);
    }

    #[test]
    fn parse_csv_usize_basic() {
        assert_eq!(parse_csv_usize("3,4,4,4").unwrap(), vec![3, 4, 4, 4]);
        assert_eq!(parse_csv_usize("1").unwrap(), vec![1]);
        assert!(parse_csv_usize("1,bad,3").is_err());
    }
}
