//! AR LLM module skeleton + weight loader.
//!
//! Phase 2-F.1: every layer struct is constructed and its weights are
//! loaded from the checkpoint, but `forward()` is not yet implemented.
//! Construction-only is enough to assert all 237 expected tensor names
//! are present + correctly shaped against the prefix
//! `<vb_root>.layers.{0..25}.*` plus
//! `<vb_root>.mm_audio_embeddings.tok_embeddings.weight`,
//! `<vb_root>.mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight`,
//! and `<vb_root>.norm.weight`.
//!
//! Phase 2-F.2 adds the actual `forward()` over a single token slice
//! with KV cache.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{Embedding, Linear, RmsNorm, VarBuilder};

use super::args::ArLlmArgs;

/// Multi-head attention with GQA. RoPE is applied at forward time
/// (Phase 2-F.2) — at load time we only carry the projection weights.
pub struct ArLlmAttention {
    pub wq: Linear,
    pub wk: Linear,
    pub wv: Linear,
    pub wo: Linear,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
}

impl ArLlmAttention {
    pub fn load(vb: VarBuilder, args: &ArLlmArgs) -> Result<Self> {
        let q_out = args.n_heads * args.head_dim;
        let kv_out = args.n_kv_heads * args.head_dim;
        let wq = linear(args.dim, q_out, args.use_biases, vb.pp("wq"))?;
        let wk = linear(args.dim, kv_out, false, vb.pp("wk"))?;
        let wv = linear(args.dim, kv_out, args.use_biases, vb.pp("wv"))?;
        let wo = linear(q_out, args.dim, args.use_biases, vb.pp("wo"))?;
        Ok(Self {
            wq,
            wk,
            wv,
            wo,
            n_heads: args.n_heads,
            n_kv_heads: args.n_kv_heads,
            head_dim: args.head_dim,
        })
    }
}

/// SwiGLU FFN — same structure as the FMA + codec FFNs but with
/// AR LLM's larger dim/hidden_dim.
pub struct ArLlmFeedForward {
    pub w1: Linear,
    pub w2: Linear,
    pub w3: Linear,
}

impl ArLlmFeedForward {
    pub fn load(vb: VarBuilder, args: &ArLlmArgs) -> Result<Self> {
        let w1 = linear(args.dim, args.hidden_dim, false, vb.pp("w1"))?;
        let w2 = linear(args.hidden_dim, args.dim, args.use_biases, vb.pp("w2"))?;
        let w3 = linear(args.dim, args.hidden_dim, false, vb.pp("w3"))?;
        Ok(Self { w1, w2, w3 })
    }
}

pub struct ArLlmBlock {
    pub attention: ArLlmAttention,
    pub feed_forward: ArLlmFeedForward,
    pub attention_norm: RmsNorm,
    pub ffn_norm: RmsNorm,
}

impl ArLlmBlock {
    pub fn load(vb: VarBuilder, args: &ArLlmArgs) -> Result<Self> {
        let attention = ArLlmAttention::load(vb.pp("attention"), args)?;
        let feed_forward = ArLlmFeedForward::load(vb.pp("feed_forward"), args)?;
        let attention_norm = rms_norm(args.dim, args.norm_eps, vb.pp("attention_norm"))?;
        let ffn_norm = rms_norm(args.dim, args.norm_eps, vb.pp("ffn_norm"))?;
        Ok(Self {
            attention,
            feed_forward,
            attention_norm,
            ffn_norm,
        })
    }
}

/// The full AR LLM. Holds:
/// - 26 transformer blocks (`layers.0..25`)
/// - Final RmsNorm (`norm`)
/// - Token embedding table (`mm_audio_embeddings.tok_embeddings`,
///   shape `[131072, 3072]`) — also serves as `lm_head.weight`
///   because `tied_embeddings=true`.
/// - Audio codebook embeddings (`mm_audio_embeddings
///   .audio_codebook_embeddings.embeddings`, shape `[9088, 3072]`)
///   used to embed the AR LLM's previous-step audio_codes when
///   feeding back into the input stream.
pub struct ArLlmModel {
    pub args: ArLlmArgs,
    pub layers: Vec<ArLlmBlock>,
    pub norm: RmsNorm,
    pub tok_embeddings: Embedding,
    pub audio_codebook_embeddings: Embedding,
    pub device: Device,
    pub dtype: DType,
}

impl ArLlmModel {
    pub fn load(
        vb: VarBuilder,
        args: ArLlmArgs,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(args.n_layers);
        for i in 0..args.n_layers {
            layers.push(ArLlmBlock::load(vb.pp(&format!("layers.{i}")), &args)?);
        }
        let norm = rms_norm(args.dim, args.norm_eps, vb.pp("norm"))?;

        // mm_audio_embeddings.tok_embeddings.weight  [vocab_size, dim]
        let tok_w = vb
            .pp("mm_audio_embeddings.tok_embeddings")
            .get((args.vocab_size, args.dim), "weight")?;
        let tok_embeddings = Embedding::new(tok_w, args.dim);

        // mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight  [9088, dim]
        // The 9088 figure is `pad_to_multiple_of_128(8192 + 36*21 + 2*COUNT)`
        // upstream — read the actual on-disk dim instead of computing it.
        let audio_emb_view = vb
            .pp("mm_audio_embeddings.audio_codebook_embeddings.embeddings")
            .get_unchecked("weight")?;
        let audio_emb_dims = audio_emb_view.dims();
        if audio_emb_dims.len() != 2 || audio_emb_dims[1] != args.dim {
            return Err(anyhow::anyhow!(
                "audio_codebook_embeddings expected [_, {}], got {:?}",
                args.dim,
                audio_emb_dims
            ));
        }
        let audio_codebook_embeddings = Embedding::new(audio_emb_view, args.dim);

        Ok(Self {
            args,
            layers,
            norm,
            tok_embeddings,
            audio_codebook_embeddings,
            device: device.clone(),
            dtype,
        })
    }

    /// `lm_head` is tied to `tok_embeddings` — return the embedding
    /// matrix for the caller to use as the projection weight.
    pub fn lm_head_weight(&self) -> &Tensor {
        self.tok_embeddings.embeddings()
    }
}

fn linear(in_features: usize, out_features: usize, has_bias: bool, vb: VarBuilder) -> Result<Linear> {
    let weight = vb.get((out_features, in_features), "weight")?;
    let bias = if has_bias {
        Some(vb.get(out_features, "bias")?)
    } else {
        None
    };
    Ok(Linear::new(weight, bias))
}

fn rms_norm(size: usize, eps: f64, vb: VarBuilder) -> Result<RmsNorm> {
    let weight = vb.get(size, "weight")?;
    Ok(RmsNorm::new(weight, eps))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn checkpoint_path() -> PathBuf {
        std::env::var_os("STARLING_TTS_SAFETENSORS")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from(
                    "/home/gjovanov/gjovanov/starling/models/cache/tts/consolidated.safetensors",
                )
            })
    }

    fn params_path() -> PathBuf {
        std::env::var_os("STARLING_TTS_PARAMS")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from("/home/gjovanov/gjovanov/starling/models/cache/tts/params.json")
            })
    }

    #[test]
    fn loads_real_ar_llm_weights() {
        let ckpt = checkpoint_path();
        let params = params_path();
        if !ckpt.exists() || !params.exists() {
            eprintln!(
                "skipping: {} or {} not present",
                ckpt.display(),
                params.display()
            );
            return;
        }
        let args = ArLlmArgs::from_params_json_path(&params).unwrap();
        let device = Device::Cpu;
        let dtype = DType::F32;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&ckpt], dtype, &device).unwrap()
        };
        let model = ArLlmModel::load(vb.clone(), args, &device, dtype).unwrap();
        assert_eq!(model.layers.len(), 26);
        assert_eq!(model.args.dim, 3072);
        assert_eq!(model.args.vocab_size, 131072);
        // lm_head shares storage with tok_embeddings.
        assert_eq!(model.lm_head_weight().dims(), &[131072, 3072]);
        // audio codebook embeddings size — verify on-disk shape matches
        // the inventory's 9088 figure.
        assert_eq!(model.audio_codebook_embeddings.embeddings().dims(), &[9088, 3072]);
    }
}
