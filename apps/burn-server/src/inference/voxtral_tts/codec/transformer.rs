//! Codec transformer block + n-layer Transformer.
//!
//! Adds two features over the FMA flow-matching block:
//! - **Layer-scale**: a per-channel learnable scale `attention_scale`
//!   and `ffn_scale` (both `[dim]`) multiplies the residual branch
//!   before the residual add. The published Voxtral-4B-TTS-2603 init is
//!   `0.01` (not the upstream's `None`-driven schedule, which is for
//!   training).
//! - **post_attention_norm / post_ffn_norm**: optional extra norms
//!   between residual and add. Voxtral does NOT enable these (no
//!   matching tensors in the checkpoint), but we keep the slot so a
//!   future variant doesn't surprise us.
//!
//! Reference: `voxtral_tts_audio_tokenizer.TransformerBlock`,
//! `Transformer`.

use anyhow::Result;
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{Linear, RmsNorm, VarBuilder};

use super::args::AudioTokenizerArgs;
use super::attention::CodecAttention;

/// SwiGLU FFN for the codec. Same shape as the FMA's FeedForward but
/// with codec-side dim (1024) and hidden_dim (4096).
pub struct CodecFeedForward {
    w1: Linear,
    w2: Linear,
    w3: Linear,
}

impl CodecFeedForward {
    pub fn load(vb: VarBuilder, args: &AudioTokenizerArgs) -> Result<Self> {
        let dim = args.dim;
        let hidden = args.hidden_dim;
        let w1 = linear(dim, hidden, false, vb.pp("w1"))?;
        let w2 = linear(hidden, dim, args.use_biases, vb.pp("w2"))?;
        let w3 = linear(dim, hidden, false, vb.pp("w3"))?;
        Ok(Self { w1, w2, w3 })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::ops::silu(&self.w1.forward(x)?)?;
        let up = self.w3.forward(x)?;
        Ok(self.w2.forward(&(gate * up)?)?)
    }
}

pub struct CodecTransformerBlock {
    attention: CodecAttention,
    feed_forward: CodecFeedForward,
    attention_norm: RmsNorm,
    ffn_norm: RmsNorm,
    /// Per-channel residual scale for the attention branch. Shape `[dim]`.
    attention_scale: Option<Tensor>,
    /// Per-channel residual scale for the FFN branch. Shape `[dim]`.
    ffn_scale: Option<Tensor>,
}

impl CodecTransformerBlock {
    pub fn load(
        vb: VarBuilder,
        args: &AudioTokenizerArgs,
        sliding_window: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let attention = CodecAttention::load(vb.pp("attention"), args, sliding_window, device, dtype)?;
        let feed_forward = CodecFeedForward::load(vb.pp("feed_forward"), args)?;
        let attention_norm = rms_norm(args.dim, args.norm_eps, vb.pp("attention_norm"))?;
        let ffn_norm = rms_norm(args.dim, args.norm_eps, vb.pp("ffn_norm"))?;

        let (attention_scale, ffn_scale) = if args.layer_scale {
            // The vector parameters live directly on the block.
            let attn_scale = vb.get(args.dim, "attention_scale")?;
            let ffn_scale = vb.get(args.dim, "ffn_scale")?;
            (Some(attn_scale), Some(ffn_scale))
        } else {
            (None, None)
        };

        Ok(Self {
            attention,
            feed_forward,
            attention_norm,
            ffn_norm,
            attention_scale,
            ffn_scale,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let r = self.attention.forward(&self.attention_norm.forward(x)?)?;
        let r = if let Some(s) = &self.attention_scale {
            // s [D], r [B, T, D] → broadcast multiply on last dim.
            r.broadcast_mul(s)?
        } else {
            r
        };
        let h = (x + r)?;
        let r = self.feed_forward.forward(&self.ffn_norm.forward(&h)?)?;
        let r = if let Some(s) = &self.ffn_scale {
            r.broadcast_mul(s)?
        } else {
            r
        };
        Ok((h + r)?)
    }
}

/// `Transformer` from upstream — `n_layers` blocks indexed by
/// `nn.ModuleDict["0", "1", ...]` (so safetensors keys look like
/// `<prefix>.layers.0.attention.wq.weight`).
pub struct CodecTransformer {
    pub layers: Vec<CodecTransformerBlock>,
}

impl CodecTransformer {
    pub fn load(
        vb: VarBuilder,
        args: &AudioTokenizerArgs,
        n_layers: usize,
        sliding_window: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            layers.push(CodecTransformerBlock::load(
                vb.pp(&format!("layers.{i}")),
                args,
                sliding_window,
                device,
                dtype,
            )?);
        }
        Ok(Self { layers })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = x.clone();
        for l in &self.layers {
            h = l.forward(&h)?;
        }
        Ok(h)
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
