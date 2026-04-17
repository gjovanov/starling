//! Voxtral Mini 4B Realtime — candle 0.10 + FlashAttention v2.
//!
//! Same architecture as candle_native but with:
//! - candle-core 0.10.2 (lower per-op overhead)
//! - candle-flash-attn for fused decoder attention (GQA native, BF16)

use candle_core::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn;
use candle_nn::VarBuilder;
use candle_flash_attn::flash_attn;
use std::time::Instant;

// Custom Linear that stores pre-transposed, pre-unsqueezed weight.
// Weight stored as [1, in, out] to avoid per-forward unsqueeze(0).
struct Linear {
    weight_3d: Tensor,  // [1, in_features, out_features] — ready for batched matmul
    bias: Option<Tensor>,
}

impl Linear {
    fn new(weight: Tensor, bias: Option<Tensor>) -> Result<Self> {
        let weight_3d = weight.t()?.contiguous()?.unsqueeze(0)?;
        Ok(Self { weight_3d, bias })
    }
}

impl Module for Linear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let out = x.matmul(&self.weight_3d)?;
        match &self.bias {
            None => Ok(out),
            Some(bias) => out.broadcast_add(bias),
        }
    }
}

fn linear_no_bias(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Linear> {
    let weight = vb.get((out_dim, in_dim), "weight")?;
    Linear::new(weight, None)
}

fn linear_b(in_dim: usize, out_dim: usize, has_bias: bool, vb: VarBuilder) -> Result<Linear> {
    let weight = vb.get((out_dim, in_dim), "weight")?;
    let bias = if has_bias {
        Some(vb.get(out_dim, "bias")?)
    } else {
        None
    };
    Linear::new(weight, bias)
}

// Custom RmsNorm matching burn's formula: x / sqrt(mean(x²) + eps) * weight
// candle-nn's RmsNorm uses: x / (sqrt(mean(x²)) + eps) * weight (eps OUTSIDE sqrt)
pub struct BurnRmsNorm {
    weight: Tensor,
    eps: f64,
}

impl BurnRmsNorm {
    fn load(size: usize, eps: f64, vb: &VarBuilder) -> Result<Self> {
        let weight = vb.get(size, "weight")?;
        Ok(Self { weight, eps })
    }

    fn weight(&self) -> &Tensor { &self.weight }
}

impl Module for BurnRmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Fused CUDA kernel: single launch replaces 7 decomposed ops.
        // Formula: x / sqrt(mean(x²) + eps) * weight
        // The CUDA kernel accumulates in F32 internally for bf16 inputs.
        // Requires contiguous input (residual-add outputs may have non-trivial strides).
        candle_nn::ops::rms_norm(&x.contiguous()?, &self.weight, self.eps as f32)
    }
}

// ─── RoPE ───────────────────────────────────────────────────────────────────

pub struct RoPE {
    cos_f32: Tensor,     // [max_seq, half] in F32 (for encoder)
    sin_f32: Tensor,
    cos_bf16: Tensor,    // [max_seq, half] in BF16 (pre-cast, for decoder)
    sin_bf16: Tensor,
}

impl RoPE {
    fn new(head_dim: usize, max_seq: usize, theta: f64, device: &Device) -> Result<Self> {
        let half = head_dim / 2;
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| 1.0 / (theta as f32).powf(2.0 * i as f32 / head_dim as f32))
            .collect();
        let inv_freq = Tensor::new(inv_freq, device)?; // [half]
        let positions: Vec<f32> = (0..max_seq).map(|p| p as f32).collect();
        let positions = Tensor::new(positions, device)?; // [max_seq]
        let freqs = positions.unsqueeze(1)?.matmul(&inv_freq.unsqueeze(0)?)?;
        let cos_f32 = freqs.cos()?;
        let sin_f32 = freqs.sin()?;
        // Pre-cast to BF16 once at load time — saves 2 to_dtype kernel launches per layer per step
        let cos_bf16 = cos_f32.to_dtype(DType::BF16)?.contiguous()?;
        let sin_bf16 = sin_f32.to_dtype(DType::BF16)?.contiguous()?;
        Ok(Self { cos_f32, sin_f32, cos_bf16, sin_bf16 })
    }

    /// Apply interleaved RoPE (is_neox_style=False) to q, k [B, seq, heads, hd].
    /// Uses candle_nn's fused CUDA kernel: single launch replaces ~36 decomposed ops.
    fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let (b, seq_len, n_heads, hd) = q.dims4()?;
        let n_kv_heads = k.dim(2)?;
        let dtype = q.dtype();
        let (cos, sin) = if dtype == DType::BF16 {
            (self.cos_bf16.narrow(0, offset, seq_len)?,
             self.sin_bf16.narrow(0, offset, seq_len)?)
        } else {
            (self.cos_f32.narrow(0, offset, seq_len)?.to_dtype(dtype)?.contiguous()?,
             self.sin_f32.narrow(0, offset, seq_len)?.to_dtype(dtype)?.contiguous()?)
        };

        if seq_len == 1 {
            // Decode path: [B, 1, heads, hd] → reshape to [B, heads, 1, hd] (zero-copy).
            // Both have the same flat memory layout when seq=1, so reshape is metadata-only.
            // Eliminates 4 contiguous() GPU copy kernels per layer.
            let q_bhsd = q.reshape((b, n_heads, 1, hd))?;
            let k_bhsd = k.reshape((b, n_kv_heads, 1, hd))?;
            let q_r = candle_nn::rotary_emb::rope_i(&q_bhsd, &cos, &sin)?;
            let k_r = candle_nn::rotary_emb::rope_i(&k_bhsd, &cos, &sin)?;
            Ok((q_r.reshape((b, 1, n_heads, hd))?, k_r.reshape((b, 1, n_kv_heads, hd))?))
        } else {
            // Prefill path: seq > 1, need actual transpose + contiguous
            let q_t = q.transpose(1, 2)?.contiguous()?;
            let k_t = k.transpose(1, 2)?.contiguous()?;
            let q_r = candle_nn::rotary_emb::rope_i(&q_t, &cos, &sin)?;
            let k_r = candle_nn::rotary_emb::rope_i(&k_t, &cos, &sin)?;
            Ok((q_r.transpose(1, 2)?.contiguous()?, k_r.transpose(1, 2)?.contiguous()?))
        }
    }
}

// ─── KV Cache ───────────────────────────────────────────────────────────────

pub struct KVCache {
    k: Option<Tensor>,
    v: Option<Tensor>,
    pub seq_len: usize,
    /// Logical position offset (for when cache is compacted)
    pub pos_offset: usize,
}

impl KVCache {
    pub fn new() -> Self {
        Self { k: None, v: None, seq_len: 0, pos_offset: 0 }
    }

    /// Pre-allocate cache with given capacity. Shape: [B, capacity, heads, hd]
    pub fn pre_allocate(b: usize, capacity: usize, heads: usize, hd: usize, dtype: DType, device: &Device) -> Result<Self> {
        let k = Tensor::zeros((b, capacity, heads, hd), dtype, device)?;
        let v = Tensor::zeros((b, capacity, heads, hd), dtype, device)?;
        Ok(Self { k: Some(k), v: Some(v), seq_len: 0, pos_offset: 0 })
    }

    /// Current logical position (pos_offset + seq_len)
    fn logical_len(&self) -> usize { self.pos_offset + self.seq_len }

    /// Compact: keep only the last `keep` positions. Updates pos_offset.
    pub fn compact(&mut self, keep: usize) -> Result<()> {
        if self.seq_len <= keep { return Ok(()); }
        let drop = self.seq_len - keep;
        if let (Some(ck), Some(cv)) = (&self.k, &self.v) {
            self.k = Some(ck.narrow(1, drop, keep)?.contiguous()?);
            self.v = Some(cv.narrow(1, drop, keep)?.contiguous()?);
        }
        self.pos_offset += drop;
        self.seq_len = keep;
        Ok(())
    }

    /// Append new K, V and return full cached K, V.
    /// Input shapes: [B, new_seq, heads, hd]
    fn update(&mut self, k: Tensor, v: Tensor) -> Result<(Tensor, Tensor)> {
        let new_seq = k.dim(1)?;
        let (k, v) = if new_seq == 1 && self.k.is_some() {
            // Single-token hot path: use slice_set to write in-place (avoids cat+alloc)
            let ck = self.k.as_mut().unwrap();
            let cv = self.v.as_mut().unwrap();
            let cur = self.seq_len;
            let cap = ck.dim(1)?;
            if cur < cap {
                // Write into pre-allocated slot
                ck.slice_set(&k.squeeze(1)?, 1, cur)?;
                cv.slice_set(&v.squeeze(1)?, 1, cur)?;
                self.seq_len += 1;
                return Ok((ck.narrow(1, 0, self.seq_len)?, cv.narrow(1, 0, self.seq_len)?));
            }
            // Capacity exceeded: grow with cat
            let k = Tensor::cat(&[ck as &Tensor, &k], 1)?.contiguous()?;
            let v = Tensor::cat(&[cv as &Tensor, &v], 1)?.contiguous()?;
            (k, v)
        } else {
            // Batch/first call: cat or init
            match (&self.k, &self.v) {
                (Some(ck), Some(cv)) => {
                    let k = Tensor::cat(&[ck, &k], 1)?.contiguous()?;
                    let v = Tensor::cat(&[cv, &v], 1)?.contiguous()?;
                    (k, v)
                }
                _ => (k.contiguous()?, v.contiguous()?),
            }
        };
        self.k = Some(k.clone());
        self.v = Some(v.clone());
        self.seq_len += new_seq;
        Ok((k, v))
    }
}

// ─── Encoder Layer ──────────────────────────────────────────────────────────

struct EncoderAttention {
    wq: Linear,
    wk: Linear,
    wv: Linear,
    wo: Linear,
    n_heads: usize,
    head_dim: usize,
    attn_dim: usize, // n_heads * head_dim (may differ from d_model)
    scale: f32,
    sliding_window: usize,
}

impl EncoderAttention {
    fn load(prefix: &str, d_model: usize, n_heads: usize, head_dim: usize, sliding_window: usize, vb: &VarBuilder) -> Result<Self> {
        let attn_dim = n_heads * head_dim; // 2048 for encoder (≠ d_model=1280)
        let wq = linear_b(d_model, attn_dim, true, vb.pp(&format!("{prefix}.attention.wq")))?;
        let wk = linear_no_bias(d_model, attn_dim, vb.pp(&format!("{prefix}.attention.wk")))?;
        let wv = linear_b(d_model, attn_dim, true, vb.pp(&format!("{prefix}.attention.wv")))?;
        let wo = linear_b(attn_dim, d_model, true, vb.pp(&format!("{prefix}.attention.wo")))?;
        Ok(Self {
            wq, wk, wv, wo,
            n_heads, head_dim, attn_dim,
            scale: (head_dim as f32).powf(-0.5),
            sliding_window,
        })
    }

    /// Forward with KV cache support (causal + sliding window).
    fn forward_with_cache(&self, x: &Tensor, rope: &RoPE, cache: &mut KVCache) -> Result<Tensor> {
        let (b, seq, _) = x.dims3()?;
        let offset = cache.logical_len(); // logical position for RoPE continuity
        let q = self.wq.forward(x)?.reshape((b, seq, self.n_heads, self.head_dim))?;
        let k = self.wk.forward(x)?.reshape((b, seq, self.n_heads, self.head_dim))?;
        let v = self.wv.forward(x)?.reshape((b, seq, self.n_heads, self.head_dim))?;
        let (q, k) = rope.apply(&q, &k, offset)?;

        // Update KV cache BEFORE transpose (cache stores [B, seq, heads, hd])
        let (full_k, full_v) = cache.update(k, v)?;

        // [B, seq, heads, hd] → [B, heads, seq, hd]
        let q = q.transpose(1, 2)?.contiguous()?;
        let full_k = full_k.transpose(1, 2)?.contiguous()?;
        let full_v = full_v.transpose(1, 2)?.contiguous()?;
        let kv_seq = full_k.dim(2)?;

        // scores = Q @ K^T * scale
        let scores = (q.matmul(&full_k.transpose(2, 3)?.contiguous()?)? * (self.scale as f64))?;

        // Causal + sliding window mask
        // KV positions in cache: cache.pos_offset + 0..kv_seq
        let kv_pos_offset = cache.pos_offset;
        let mask: Vec<f32> = (0..seq).flat_map(|i| {
            let qi_abs = offset + i; // logical position of query
            (0..kv_seq).map(move |j| {
                let kj_abs = kv_pos_offset + j; // logical position of key
                if kj_abs > qi_abs { f32::NEG_INFINITY }  // causal
                else if qi_abs - kj_abs >= self.sliding_window { f32::NEG_INFINITY } // sliding window
                else { 0.0 }
            })
        }).collect();
        let mask = Tensor::new(mask, x.device())?.to_dtype(scores.dtype())?.reshape((1, 1, seq, kv_seq))?;
        let scores = scores.broadcast_add(&mask)?;

        // Softmax
        let max_vals = scores.max_keepdim(D::Minus1)?;
        let shifted = scores.broadcast_sub(&max_vals)?;
        let exp_vals = shifted.exp()?;
        let sum_vals = exp_vals.sum_keepdim(D::Minus1)?;
        let attn = exp_vals.broadcast_div(&sum_vals)?;

        let out = attn.matmul(&full_v)?;
        let out = out.transpose(1, 2)?.contiguous()?.reshape((b, seq, self.attn_dim))?;
        self.wo.forward(&out)
    }

}

struct EncoderSwiGLU {
    w1: Linear,
    w2: Linear,
    w3: Linear,
}

impl EncoderSwiGLU {
    fn load(prefix: &str, d_model: usize, hidden_dim: usize, vb: &VarBuilder) -> Result<Self> {
        let w1 = linear_no_bias(d_model, hidden_dim, vb.pp(&format!("{prefix}.feed_forward.w1")))?;
        let w2 = linear_b(hidden_dim, d_model, true, vb.pp(&format!("{prefix}.feed_forward.w2")))?;
        let w3 = linear_no_bias(d_model, hidden_dim, vb.pp(&format!("{prefix}.feed_forward.w3")))?;
        Ok(Self { w1, w2, w3 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.w1.forward(x)?.silu()?;
        let up = self.w3.forward(x)?;
        self.w2.forward(&(gate * up)?)
    }
}

pub struct EncoderLayer {
    attention_norm: BurnRmsNorm,
    attention: EncoderAttention,
    ffn_norm: BurnRmsNorm,
    ffn: EncoderSwiGLU,
}

impl EncoderLayer {
    fn load(prefix: &str, d_model: usize, n_heads: usize, head_dim: usize, hidden_dim: usize, sliding_window: usize, eps: f64, vb: &VarBuilder) -> Result<Self> {
        let attention_norm = BurnRmsNorm::load(d_model, eps, &vb.pp(&format!("{prefix}.attention_norm")))?;
        let attention = EncoderAttention::load(prefix, d_model, n_heads, head_dim, sliding_window, vb)?;
        let ffn_norm = BurnRmsNorm::load(d_model, eps, &vb.pp(&format!("{prefix}.ffn_norm")))?;
        let ffn = EncoderSwiGLU::load(prefix, d_model, hidden_dim, vb)?;
        Ok(Self { attention_norm, attention, ffn_norm, ffn })
    }

    pub fn forward(&self, x: &Tensor, rope: &RoPE, cache: &mut KVCache) -> Result<Tensor> {
        let residual = x.clone();
        let x = self.attention_norm.forward(x)?;
        let x = (self.attention.forward_with_cache(&x, rope, cache)? + &residual)?;
        let residual = x.clone();
        let h = self.ffn_norm.forward(&x)?;
        (self.ffn.forward(&h)? + &residual)
    }

}

// ─── Audio Encoder ──────────────────────────────────────────────────────────

pub struct ConvDownsampler {
    conv1_weight: Tensor,
    conv1_bias: Tensor,
    conv2_weight: Tensor,
    conv2_bias: Tensor,
}

impl ConvDownsampler {
    fn load(prefix: &str, vb: &VarBuilder) -> Result<Self> {
        let conv1_weight = vb.get((1280, 128, 3), &format!("{prefix}.conv_layers.0.conv.weight"))?;
        let conv1_bias = vb.get(1280, &format!("{prefix}.conv_layers.0.conv.bias"))?;
        let conv2_weight = vb.get((1280, 1280, 3), &format!("{prefix}.conv_layers.1.conv.weight"))?;
        let conv2_bias = vb.get(1280, &format!("{prefix}.conv_layers.1.conv.bias"))?;
        Ok(Self { conv1_weight, conv1_bias, conv2_weight, conv2_bias })
    }

    /// mel [B, 128, T] → [B, 1280, T/2] → transposed to [B, T/2, 1280]
    pub fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        // Conv1: causal left-pad by (kernel - stride) = 3 - 1 = 2
        let x = mel.pad_with_zeros(2, 2, 0)?; // dim=2, left=2, right=0
        let x = x.conv1d(&self.conv1_weight, 0, 1, 1, 1)?;
        let bias1 = self.conv1_bias.unsqueeze(0)?.unsqueeze(D::Minus1)?;
        let x = x.broadcast_add(&bias1)?.gelu()?;
        // Conv2: causal left-pad by (kernel - stride) = 3 - 2 = 1
        let x = x.pad_with_zeros(2, 1, 0)?; // dim=2, left=1, right=0  ← FIXED: was 2, should be 1
        let x = x.conv1d(&self.conv2_weight, 0, 2, 1, 1)?;
        let bias2 = self.conv2_bias.unsqueeze(0)?.unsqueeze(D::Minus1)?;
        let x = x.broadcast_add(&bias2)?.gelu()?;
        x.transpose(1, 2) // [B, T/2, 1280]
    }
}

pub struct AudioEncoder {
    pub conv: ConvDownsampler,
    pub rope: RoPE,
    pub layers: Vec<EncoderLayer>,
    pub norm: BurnRmsNorm,
}

impl AudioEncoder {
    fn load(vb: &VarBuilder, device: &Device) -> Result<Self> {
        let prefix = "mm_streams_embeddings.embedding_module.whisper_encoder";
        // Encoder: d_model=1280, n_heads=32, head_dim=64 (attn_dim=2048), hidden=5120, sw=750
        let d_model = 1280;
        let conv = ConvDownsampler::load(prefix, vb)?;
        let rope = RoPE::new(64, 200_000, 1_000_000.0, device)?;
        let mut layers = Vec::with_capacity(32);
        for i in 0..32 {
            let layer_prefix = format!("{prefix}.transformer.layers.{i}");
            layers.push(EncoderLayer::load(&layer_prefix, d_model, 32, 64, 5120, 750, 1e-5, vb)?);
            if (i + 1) % 8 == 0 {
                eprintln!("[CandleNative] encoder layer {}/32", i + 1);
            }
        }
        let norm = BurnRmsNorm::load(d_model, 1e-5, &vb.pp(&format!("{prefix}.transformer.norm")))?;
        Ok(Self { conv, rope, layers, norm })
    }

}

// ─── Adapter ────────────────────────────────────────────────────────────────

pub struct AudioLanguageAdapter {
    linear1: Linear,
    linear2: Linear,
}

impl AudioLanguageAdapter {
    fn load(vb: &VarBuilder) -> Result<Self> {
        let prefix = "mm_streams_embeddings.embedding_module.audio_language_projection";
        // Actual weights: linear1 [3072, 5120] (out=3072, in=5120), linear2 [3072, 3072]
        let linear1 = linear_no_bias(5120, 3072, vb.pp(&format!("{prefix}.0")))?;
        let linear2 = linear_no_bias(3072, 3072, vb.pp(&format!("{prefix}.2")))?;
        Ok(Self { linear1, linear2 })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.linear1.forward(x)?.gelu()?;
        self.linear2.forward(&x)
    }
}

// ─── Decoder Layer ──────────────────────────────────────────────────────────

struct DecoderAttention {
    wqkv_t: Tensor,  // fused [d_model, q_dim+2*kv_dim] pre-transposed
    wo: Linear,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    q_dim: usize,
    kv_dim: usize,
    scale: f32,
    sliding_window: usize,
}

impl DecoderAttention {
    fn load(prefix: &str, vb: &VarBuilder) -> Result<Self> {
        let n_heads = 32;
        let n_kv_heads = 8;
        let head_dim = 128;
        let d_model = 3072;
        let q_dim = n_heads * head_dim;    // 4096
        let kv_dim = n_kv_heads * head_dim; // 1024

        // Load individual weights then fuse QKV into single matmul
        let wq = vb.get((q_dim, d_model), &format!("{prefix}.attention.wq.weight"))?;
        let wk = vb.get((kv_dim, d_model), &format!("{prefix}.attention.wk.weight"))?;
        let wv = vb.get((kv_dim, d_model), &format!("{prefix}.attention.wv.weight"))?;
        // Fuse: [q_dim+2*kv_dim, d_model] then transpose to [d_model, q_dim+2*kv_dim]
        let wqkv = Tensor::cat(&[&wq, &wk, &wv], 0)?; // [6144, 3072]
        let wqkv_t = wqkv.t()?.contiguous()?.unsqueeze(0)?; // [1, 3072, 6144] pre-unsqueezed

        let wo = linear_no_bias(q_dim, d_model, vb.pp(&format!("{prefix}.attention.wo")))?;

        Ok(Self {
            wqkv_t, wo,
            n_heads, n_kv_heads, head_dim,
            q_dim, kv_dim,
            scale: (head_dim as f32).powf(-0.5),
            sliding_window: 8192,
        })
    }

    fn forward_with_cache(&self, x: &Tensor, rope: &RoPE, cache: &mut KVCache) -> Result<Tensor> {
        let (b, seq, _) = x.dims3()?;
        let offset = cache.logical_len();

        // Fused QKV: single matmul (weight pre-unsqueezed to [1, in, out])
        let wqkv = x.matmul(&self.wqkv_t)?;
        let q = wqkv.narrow(D::Minus1, 0, self.q_dim)?.reshape((b, seq, self.n_heads, self.head_dim))?;
        let k = wqkv.narrow(D::Minus1, self.q_dim, self.kv_dim)?.reshape((b, seq, self.n_kv_heads, self.head_dim))?;
        let v = wqkv.narrow(D::Minus1, self.q_dim + self.kv_dim, self.kv_dim)?.reshape((b, seq, self.n_kv_heads, self.head_dim))?;

        let (q, k) = rope.apply(&q, &k, offset)?;
        let (k, v) = cache.update(k.contiguous()?, v.contiguous()?)?;

        // FlashAttention v2 — single fused kernel, GQA native (32 Q heads, 8 KV heads)
        // Input shapes: q [B, seq_q, n_heads, hd], k [B, seq_kv, n_kv_heads, hd]
        let causal = seq > 1; // causal during prefill, non-causal for single-token decode
        let out = flash_attn(&q.contiguous()?, &k.contiguous()?, &v.contiguous()?, self.scale, causal)?;
        // out: [B, seq_q, n_heads, hd]
        let out = out.reshape((b, seq, self.q_dim))?;
        self.wo.forward(&out)
    }
}

struct ADANorm {
    w0: Tensor,  // [d_model, t_cond_dim] — down projection
    w2: Tensor,  // [t_cond_dim, d_model] — up projection
}

impl ADANorm {
    fn load(prefix: &str, vb: &VarBuilder) -> Result<Self> {
        let w0 = vb.get((32, 3072), &format!("{prefix}.ada_rms_norm_t_cond.0.weight"))?;
        let w2 = vb.get((3072, 32), &format!("{prefix}.ada_rms_norm_t_cond.2.weight"))?;
        Ok(Self { w0, w2 })
    }

    /// Compute ADA scale: (1 + w2(gelu(w0(t_embed))))
    fn compute_scale(&self, t_embed: &Tensor) -> Result<Tensor> {
        // t_embed [1, 1, 3072], w0 [32, 3072] → need [1, 3072, 32] for matmul
        let w0_t = self.w0.t()?.unsqueeze(0)?; // [1, 3072, 32]
        let w2_t = self.w2.t()?.unsqueeze(0)?; // [1, 32, 3072]
        let h = t_embed.matmul(&w0_t)?.gelu()?; // [1, 1, 32]
        let scale = h.matmul(&w2_t)?; // [1, 1, 3072]
        // Add 1.0 in same dtype (bf16)
        let ones = Tensor::ones_like(&scale)?;
        (&scale + &ones)
    }
}

struct DecoderSwiGLU {
    w13_t: Tensor, // fused gate+up [d_model, 2*hidden_dim] pre-transposed
    w2: Linear,    // down projection
    hidden_dim: usize,
}

impl DecoderSwiGLU {
    fn load(prefix: &str, vb: &VarBuilder) -> Result<Self> {
        let d_model = 3072;
        let hidden_dim = 9216;
        // Load individual weights then fuse gate+up
        let w1 = vb.get((hidden_dim, d_model), &format!("{prefix}.feed_forward.w1.weight"))?;
        let w3 = vb.get((hidden_dim, d_model), &format!("{prefix}.feed_forward.w3.weight"))?;
        let w13 = Tensor::cat(&[&w1, &w3], 0)?; // [2*hidden, d_model]
        let w13_t = w13.t()?.contiguous()?.unsqueeze(0)?; // [1, d_model, 2*hidden] pre-unsqueezed

        let w2 = linear_no_bias(hidden_dim, d_model, vb.pp(&format!("{prefix}.feed_forward.w2")))?;
        Ok(Self { w13_t, w2, hidden_dim })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Fused gate+up matmul then SwiGLU activation
        let gate_up = x.matmul(&self.w13_t)?;
        let gate = gate_up.narrow(D::Minus1, 0, self.hidden_dim)?.silu()?;
        let up = gate_up.narrow(D::Minus1, self.hidden_dim, self.hidden_dim)?;
        self.w2.forward(&(gate * up)?)
    }
}

struct DecoderLayer {
    ada: ADANorm,
    attention_norm: BurnRmsNorm,
    attention: DecoderAttention,
    ffn_norm: BurnRmsNorm,
    ffn: DecoderSwiGLU,
}

impl DecoderLayer {
    fn load(i: usize, vb: &VarBuilder) -> Result<Self> {
        let prefix = format!("layers.{i}");
        let ada = ADANorm::load(&prefix, vb)?;
        // Use BurnRmsNorm (eps inside sqrt) matching PyTorch/voxtral.c
        let attention_norm = BurnRmsNorm::load(3072, 1e-5, &vb.pp(&format!("{prefix}.attention_norm")))?;
        let attention = DecoderAttention::load(&prefix, vb)?;
        let ffn_norm = BurnRmsNorm::load(3072, 1e-5, &vb.pp(&format!("{prefix}.ffn_norm")))?;
        let ffn = DecoderSwiGLU::load(&prefix, vb)?;
        Ok(Self { ada, attention_norm, attention, ffn_norm, ffn })
    }

    /// Forward with pre-computed ADA scale (eliminates 3 kernels/step).
    /// ADA modulation applies ONLY to the FFN path (after ffn_norm, before MLP).
    fn forward_with_cache_precomputed(
        &self, x: &Tensor, ada_scale: &Tensor, rope: &RoPE, cache: &mut KVCache,
    ) -> Result<Tensor> {
        // Attention path (no ADA)
        let residual = x.clone();
        let normed = self.attention_norm.forward(x)?;
        let attn_out = self.attention.forward_with_cache(&normed, rope, cache)?;
        let x = (&attn_out + &residual)?;

        // FFN path (with ADA scale)
        let residual = x.clone();
        let h = self.ffn_norm.forward(&x)?;
        let h = h.broadcast_mul(ada_scale)?;
        let ffn_out = self.ffn.forward(&h)?;
        (&ffn_out + &residual)
    }

    /// Debug forward: prints values at same measurement points as PyTorch debug script
    fn forward_debug(
        &self, x: &Tensor, ada_scale: &Tensor, rope: &RoPE, cache: &mut KVCache, layer_idx: usize,
    ) -> Result<Tensor> {
        let residual = x.clone();
        let normed = self.attention_norm.forward(x)?;
        let normed_abs = normed.to_dtype(DType::F32)?.abs()?.mean_all()?.to_scalar::<f32>()?;
        let attn_out = self.attention.forward_with_cache(&normed, rope, cache)?;
        let attn_abs = attn_out.to_dtype(DType::F32)?.abs()?.mean_all()?.to_scalar::<f32>()?;
        let x_after_attn = (&attn_out + &residual)?;
        let after_attn_abs = x_after_attn.to_dtype(DType::F32)?.abs()?.mean_all()?.to_scalar::<f32>()?;

        let residual2 = x_after_attn.clone();
        let h = self.ffn_norm.forward(&x_after_attn)?;
        let ffn_norm_abs = h.to_dtype(DType::F32)?.abs()?.mean_all()?.to_scalar::<f32>()?;
        let h = h.broadcast_mul(ada_scale)?;
        let h_ada_abs = h.to_dtype(DType::F32)?.abs()?.mean_all()?.to_scalar::<f32>()?;
        let ffn_out = self.ffn.forward(&h)?;
        let ffn_abs = ffn_out.to_dtype(DType::F32)?.abs()?.mean_all()?.to_scalar::<f32>()?;
        let output = (&ffn_out + &residual2)?;
        let out_abs = output.to_dtype(DType::F32)?.abs()?.mean_all()?.to_scalar::<f32>()?;

        eprintln!("[candle-dbg] layer={} attn_norm={:.4} attn_out={:.4} +res={:.4} ffn_norm={:.4} h_ada={:.4} ffn_out={:.4} output={:.4}",
            layer_idx, normed_abs, attn_abs, after_attn_abs, ffn_norm_abs, h_ada_abs, ffn_abs, out_abs);
        Ok(output)
    }
}

// ─── Full Model ─────────────────────────────────────────────────────────────

pub struct VoxtralModel {
    pub encoder: AudioEncoder,
    pub adapter: AudioLanguageAdapter,
    decoder_layers: Vec<DecoderLayer>,
    decoder_norm: BurnRmsNorm,
    decoder_rope: RoPE,
    tok_embed: Tensor,
    tok_embed_t: Tensor, // [1, D, V] pre-transposed+unsqueezed for lm_head
    device: Device,
}

impl VoxtralModel {
    pub fn load(safetensors_path: &std::path::Path, device: &Device) -> Result<Self> {
        let t0 = Instant::now();
        eprintln!("[CandleNative] Loading from {}", safetensors_path.display());

        // Disable CUDA event tracking — we use a single stream, no cross-stream sync needed.
        // Eliminates per-allocation overhead (2 CudaEvent objects per CudaSlice).
        if let Ok(cuda_dev) = device.as_cuda_device() {
            unsafe { cuda_dev.disable_event_tracking(); }
            eprintln!("[CandleNative] Disabled CUDA event tracking (single-stream)");
        }

        // Use env var to select dtype: CANDLE_NATIVE_F32=1 for f32, default bf16
        let dtype = if std::env::var("CANDLE_NATIVE_F32").is_ok() {
            eprintln!("[CandleNative] Using F32 (matching voxtral.c)");
            DType::F32
        } else {
            DType::BF16
        };
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[safetensors_path], dtype, device)?
        };

        // Encoder
        let encoder = AudioEncoder::load(&vb, device)?;
        let adapter = AudioLanguageAdapter::load(&vb)?;
        eprintln!("[CandleNative] Encoder+Adapter loaded ({:?}, {:.1}s)",
            encoder.norm.weight().dtype(), t0.elapsed().as_secs_f32());

        // Decoder
        let decoder_rope = RoPE::new(128, 16384, 1_000_000.0, device)?;
        let mut decoder_layers = Vec::with_capacity(26);
        for i in 0..26 {
            decoder_layers.push(DecoderLayer::load(i, &vb)?);
            if (i + 1) % 5 == 0 {
                eprintln!("[CandleNative] decoder layer {}/26 (bf16)", i + 1);
            }
        }
        let decoder_norm = BurnRmsNorm::load(3072, 1e-5, &vb.pp("norm"))?;

        // Token embeddings + pre-transposed for lm_head
        let tok_embed = vb.get(
            (131072, 3072),
            "mm_streams_embeddings.embedding_module.tok_embeddings.weight",
        )?;
        let tok_embed_t = tok_embed.t()?.contiguous()?.unsqueeze(0)?; // [1, 3072, 131072] pre-unsqueezed
        eprintln!("[CandleNative] Token embedding on GPU ({:.0} MB)",
            131072.0 * 3072.0 * 2.0 / 1e6);

        eprintln!("[CandleNative] Full model loaded in {:.1}s ({:?})", t0.elapsed().as_secs_f32(), dtype);
        Ok(Self { encoder, adapter, decoder_layers, decoder_norm, decoder_rope, tok_embed, tok_embed_t, device: device.clone() })
    }

    /// Encode audio with KV cache support for incremental processing.
    /// conv → 32 encoder layers (interleaved RoPE, KV cached) → norm → 4x reshape → adapter
    pub fn encode_audio(&self, mel: &Tensor, enc_caches: &mut Vec<KVCache>) -> Result<Tensor> {
        // Conv stem
        let conv_out = self.encoder.conv.forward(mel)?; // [1, T/2, 1280]
        let n_frames = conv_out.dim(1)?;

        // Truncate right to align to pool_size=4
        let n_aligned = (n_frames / 4) * 4;
        let x = if n_aligned < n_frames {
            conv_out.narrow(1, 0, n_aligned)?
        } else {
            conv_out
        };

        // Encoder transformer layers with KV cache
        let mut x = x;
        for (i, layer) in self.encoder.layers.iter().enumerate() {
            x = layer.forward(&x, &self.encoder.rope, &mut enc_caches[i])?;
        }
        let enc_out = self.encoder.norm.forward(&x)?;

        // 4x reshape + adapter
        let reshaped = reshape_encoder_output(&enc_out, 4)?;
        self.adapter.forward(&reshaped)
    }

    /// Create fresh encoder KV caches (32 layers).
    pub fn new_encoder_caches() -> Vec<KVCache> {
        (0..32).map(|_| KVCache::new()).collect()
    }

    /// Create pre-allocated decoder KV caches (26 layers, capacity for ~2000 positions).
    pub fn new_decoder_caches(&self, capacity: usize) -> Result<Vec<KVCache>> {
        let dtype = self.tok_embed.dtype();
        (0..26).map(|_| KVCache::pre_allocate(1, capacity, 8, 128, dtype, &self.device))
            .collect()
    }

    /// Pre-compute ADA scales for all 26 decoder layers (constant across decode steps).
    pub fn precompute_ada_scales(&self, t_embed: &Tensor) -> Result<Vec<Tensor>> {
        self.decoder_layers.iter()
            .map(|layer| layer.ada.compute_scale(t_embed))
            .collect()
    }

    /// Forward through all decoder layers with pre-computed ADA scales.
    pub fn decoder_forward(
        &self, mut x: Tensor, ada_scales: &[Tensor], caches: &mut Vec<KVCache>,
    ) -> Result<Tensor> {
        let num_layers = std::env::var("CANDLE_NATIVE_LAYERS")
            .ok().and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(self.decoder_layers.len());
        let diag = std::env::var("DIAG_LAYERS").is_ok();
        for (i, layer) in self.decoder_layers.iter().enumerate().take(num_layers) {
            x = layer.forward_with_cache_precomputed(&x, &ada_scales[i], &self.decoder_rope, &mut caches[i])?;
            if diag && caches[i].seq_len <= 40 {
                // Only dump during prefill (first forward call)
                let seq = x.dim(1)?;
                let dim = x.dim(2)?;
                let last: Vec<f32> = x.i((0, seq - 1, ..4))?.to_dtype(DType::F32)?.to_vec1()?;
                let full_row: Vec<f32> = x.i((0, seq - 1, ..))?.to_dtype(DType::F32)?.to_vec1()?;
                let norm: f32 = full_row.iter().map(|v| v * v).sum::<f32>().sqrt();
                eprintln!("[DIAG CandleNative layer {}] hidden L2={:.4} first4=[{:.4}, {:.4}, {:.4}, {:.4}]",
                    i, norm, last[0], last[1], last[2], last[3]);
            }
        }
        self.decoder_norm.forward(&x)
    }

    /// lm_head: hidden @ tok_embed^T → logits (uses pre-transposed weight)
    pub fn lm_head(&self, hidden: &Tensor) -> Result<Tensor> {
        hidden.contiguous()?.matmul(&self.tok_embed_t)
    }

    /// lm_head + argmax in one call — avoids GPU→CPU transfer of 131K floats.
    /// Returns (token_id, logits_tensor)
    pub fn lm_head_argmax(&self, hidden: &Tensor) -> Result<u32> {
        let logits = self.lm_head(hidden)?; // [1, 1, 131072]
        // argmax on GPU
        let token = logits.argmax(D::Minus1)?; // [1, 1]
        let token_id: u32 = token.to_dtype(DType::U32)?.reshape(1)?.to_vec1::<u32>()?[0];
        Ok(token_id)
    }

    /// GPU embedding lookup for a single token ID.
    pub fn embed_token(&self, token_id: u32) -> Result<Tensor> {
        self.tok_embed.i(token_id as usize)?.contiguous()?.unsqueeze(0)?.unsqueeze(0)
    }

    /// Embed multiple token IDs: [len] → [1, len, d_model]
    pub fn embed_tokens(&self, ids: &[u32]) -> Result<Tensor> {
        let indices = Tensor::new(ids, &self.device)?;
        self.tok_embed.index_select(&indices, 0)?.contiguous()?.unsqueeze(0)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}

// ─── Helpers ────────────────────────────────────────────────────────────────

pub fn reshape_encoder_output(x: &Tensor, factor: usize) -> Result<Tensor> {
    let (batch, seq, dim) = x.dims3()?;
    let new_seq = seq / factor;
    let truncated = new_seq * factor;
    let x = if truncated < seq { x.narrow(1, 0, truncated)? } else { x.clone() };
    x.reshape((batch, new_seq, dim * factor))
}

/// Compute sinusoidal time embedding for transcription delay.
pub fn compute_time_embedding(t: f32, dim: usize, device: &Device) -> Result<Tensor> {
    let half_dim = dim / 2;
    let log_theta = 10000.0f32.ln();
    let mut embedding = Vec::with_capacity(dim);
    for i in 0..half_dim {
        let freq = (-log_theta * (i as f32) / (half_dim as f32)).exp();
        embedding.push((t * freq).cos());
    }
    for i in 0..half_dim {
        let freq = (-log_theta * (i as f32) / (half_dim as f32)).exp();
        embedding.push((t * freq).sin());
    }
    // Compute in f32, convert to model dtype
    let dtype = if std::env::var("CANDLE_NATIVE_F32").is_ok() { DType::F32 } else { DType::BF16 };
    Tensor::new(embedding, device)?.to_dtype(dtype)?.reshape((1, 1, dim))
}

/// Transcription matching voxtral.c's architecture:
/// 1. Encode all audio → adapter output [1, N, 3072]
/// 2. Prefill: adapter[0..prompt_len] + embed(BOS/STREAMING_PAD) — audio+text at prefix positions
/// 3. Autoregressive decode: adapter[i] + embed(prev_token) — one position at a time
pub fn transcribe(model: &VoxtralModel, mel: &Tensor, t_embed: &Tensor) -> Result<Vec<u32>> {
    let t0 = Instant::now();

    let mut enc_caches = VoxtralModel::new_encoder_caches();
    let audio_embeds = model.encode_audio(mel, &mut enc_caches)?;
    let audio_seq = audio_embeds.dim(1)?;
    let d_model = audio_embeds.dim(2)?;
    eprintln!("[CandleNative] Encoded: seq_len={} ({:.1}s)", audio_seq, t0.elapsed().as_secs_f32());

    let mut caches = (0..26).map(|_| KVCache::new()).collect(); // capacity = audio + prompt
    let ada_scales = model.precompute_ada_scales(t_embed)?;

    // Prompt: BOS(1) + STREAMING_PAD(32)*38 = 39 tokens
    // voxtral.c uses 1 + 32 + delay_tokens(6) = 39
    let prompt_len = 39usize.min(audio_seq); // can't exceed audio length
    let prompt_ids: Vec<u32> = std::iter::once(1u32)
        .chain(std::iter::repeat(32u32).take(prompt_len - 1))
        .collect();

    // Phase 1: Prefill — matching voxtral.c exactly:
    // voxtral.c prefills prompt_len-1 positions, then forwards last position separately
    // prompt_embeds[i] = adapter_buf[i] + tok_embedding[BOS/PAD]
    let prefill_count = prompt_len - 1; // voxtral.c: prefill_count = prompt_len - 1
    let prompt_audio = audio_embeds.narrow(1, 0, prompt_len)?;
    let prompt_text = model.embed_tokens(&prompt_ids)?;
    let prompt_input = (&prompt_audio + &prompt_text)?;

    let t1 = Instant::now();
    // Prefill first prompt_len-1 positions
    let prefill_input = prompt_input.narrow(1, 0, prefill_count)?;
    let _prefill_hidden = model.decoder_forward(prefill_input, &ada_scales, &mut caches)?;

    // Forward last prompt position separately (matching voxtral.c)
    let last_input = prompt_input.narrow(1, prefill_count, 1)?;
    let last_hidden = model.decoder_forward(last_input, &ada_scales, &mut caches)?;
    let prompt_logits = model.lm_head(&last_hidden)?;
    let prefill_ms = t1.elapsed().as_millis();

    // Get first token from last prompt position
    let last_logits: Vec<f32> = prompt_logits.to_dtype(DType::F32)?.reshape(131072)?.to_vec1()?;
    let mut prev_token = last_logits.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32).unwrap_or(0);
    eprintln!("[CandleNative] Prefill {}: {}ms", prompt_len, prefill_ms);

    // Diagnostic: dump top-k logits and encoder output for comparison with Q4
    if std::env::var("DIAG_LOGITS").is_ok() {
        // Top-10 logits after prefill
        let mut indexed: Vec<(usize, f32)> = last_logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top10: Vec<String> = indexed.iter().take(10).map(|(i, v)| format!("{}:{:.4}", i, v)).collect();
        let min_v = last_logits.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_v = last_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mean_v: f32 = last_logits.iter().sum::<f32>() / last_logits.len() as f32;
        eprintln!("[DIAG CandleNative] prefill logits: vocab={} min={:.4} max={:.4} mean={:.6} first_token={} top10=[{}]",
            last_logits.len(), min_v, max_v, mean_v, prev_token, top10.join(", "));

        // Encoder output: first 8 values at position 0
        let enc_first: Vec<f32> = audio_embeds.i((0, 0, ..8))?.to_dtype(DType::F32)?.to_vec1()?;
        let enc_last_pos = audio_seq - 1;
        let enc_last: Vec<f32> = audio_embeds.i((0, enc_last_pos, ..8))?.to_dtype(DType::F32)?.to_vec1()?;
        let enc_norm0: f32 = {
            let row: Vec<f32> = audio_embeds.i((0, 0, ..))?.to_dtype(DType::F32)?.to_vec1()?;
            row.iter().map(|x| x * x).sum::<f32>().sqrt()
        };
        eprintln!("[DIAG CandleNative] encoder_out[0,0,:8]={:?} L2_norm={:.4}", enc_first, enc_norm0);
        eprintln!("[DIAG CandleNative] encoder_out[0,{},:8]={:?}", enc_last_pos, enc_last);

        // Decoder hidden state at last prefill position
        let last_hidden_f32: Vec<f32> = last_hidden.i((0, 0, ..8))?.to_dtype(DType::F32)?.to_vec1()?;
        let hidden_norm: f32 = {
            let row: Vec<f32> = last_hidden.i((0, 0, ..))?.to_dtype(DType::F32)?.to_vec1()?;
            row.iter().map(|x| x * x).sum::<f32>().sqrt()
        };
        eprintln!("[DIAG CandleNative] hidden_last_prefill[:8]={:?} L2_norm={:.4}", last_hidden_f32, hidden_norm);

        // Token 32 embedding
        let tok32_embed: Vec<f32> = model.tok_embed.i(32)?.to_dtype(DType::F32)?.to_vec1()?;
        let tok32_norm: f32 = tok32_embed.iter().map(|x| x * x).sum::<f32>().sqrt();
        eprintln!("[DIAG CandleNative] tok32_embed[:8]={:?} L2_norm={:.4}", &tok32_embed[..8], tok32_norm);
    }

    // Phase 2: Autoregressive decode — adapter[i] + embed(prev_token)
    let t2 = Instant::now();
    let eos_token = 2u32;
    let mut generated = Vec::with_capacity(audio_seq);
    generated.push(prev_token);

    for i in prompt_len..audio_seq {
        let audio_pos = audio_embeds.narrow(1, i, 1)?;
        let text_embed = model.embed_token(prev_token)?;
        let x = (&audio_pos + &text_embed)?;

        let hidden = model.decoder_forward(x, &ada_scales, &mut caches)?;
        let token = model.lm_head_argmax(&hidden)?;

        generated.push(token);
        prev_token = token;

        if token == eos_token { break; }
    }

    let decode_ms = t2.elapsed().as_millis();
    let n_steps = generated.len() - 1;
    let ms_per_step = if n_steps > 0 { decode_ms as f32 / n_steps as f32 } else { 0.0 };
    let total_s = t0.elapsed().as_secs_f32();
    let audio_secs = audio_seq as f32 / 12.5;
    let text_tokens: Vec<u32> = generated.iter().copied().filter(|&t| t >= 1000).collect();
    eprintln!("[CandleNative] Decode: {} steps, {}ms, {:.1}ms/step, {} text tokens",
        n_steps, decode_ms, ms_per_step, text_tokens.len());
    eprintln!("[CandleNative] Total: {:.1}s ({:.1}× realtime for {:.0}s audio)",
        total_s, total_s / audio_secs, audio_secs);

    Ok(generated)
}

/// Streaming transcription: processes audio in chunks matching voxtral.c's approach.
/// Conv is computed once (fast), then encoder processes in 2s chunks with KV cache.
/// This enables transcription of unlimited audio without O(n²) encoder blowup.
pub fn transcribe_streaming(model: &VoxtralModel, mel: &Tensor, t_embed: &Tensor) -> Result<Vec<u32>> {
    let t0 = Instant::now();

    // Step 1: Conv stem on full mel (fast, no attention)
    let conv_out = model.encoder.conv.forward(mel)?; // [1, T/2, 1280]
    let n_conv = conv_out.dim(1)?;
    let n_aligned = (n_conv / 4) * 4;
    let conv_out = if n_aligned < n_conv { conv_out.narrow(1, 0, n_aligned)? } else { conv_out };
    let n_conv = conv_out.dim(1)?;
    eprintln!("[Streaming] Conv: {} frames ({:.2}s)", n_conv, t0.elapsed().as_secs_f32());

    // Step 2: Encoder in chunks (matching voxtral.c's 2s interval = ~200 mel frames = ~100 conv frames)
    let enc_chunk_size = 100; // conv frames per chunk (~1s of audio after stride-2 conv)
    let sliding_window = 750; // encoder attention window
    let mut enc_caches = VoxtralModel::new_encoder_caches();
    let mut adapter_tokens: Vec<Tensor> = Vec::new();
    let mut enc_residual: Option<Tensor> = None; // leftover frames for 4x alignment

    let t_enc = Instant::now();
    let mut pos = 0;
    while pos < n_conv {
        let chunk_end = (pos + enc_chunk_size).min(n_conv);
        let chunk = conv_out.narrow(1, pos, chunk_end - pos)?;

        // Compact encoder KV caches if approaching sliding window limit
        if enc_caches[0].seq_len + (chunk_end - pos) > sliding_window {
            let keep = sliding_window / 2; // keep half, drop old
            for cache in &mut enc_caches {
                cache.compact(keep)?;
            }
        }

        // Encoder transformer layers with KV cache
        let mut x = chunk;
        for (i, layer) in model.encoder.layers.iter().enumerate() {
            x = layer.forward(&x, &model.encoder.rope, &mut enc_caches[i])?;
        }
        let enc_out = model.encoder.norm.forward(&x)?;

        // Handle 4x alignment residual: combine with previous leftover
        let to_adapt = if let Some(residual) = enc_residual.take() {
            Tensor::cat(&[&residual, &enc_out], 1)?
        } else {
            enc_out
        };

        let total = to_adapt.dim(1)?;
        let usable = (total / 4) * 4;
        let leftover = total - usable;

        if usable > 0 {
            let adapter_input = to_adapt.narrow(1, 0, usable)?;
            let reshaped = reshape_encoder_output(&adapter_input, 4)?;
            let adapted = model.adapter.forward(&reshaped)?;
            adapter_tokens.push(adapted);
        }

        if leftover > 0 {
            enc_residual = Some(to_adapt.narrow(1, usable, leftover)?);
        }

        pos = chunk_end;
    }

    // Flush any remaining residual (pad with zeros to align)
    if let Some(residual) = enc_residual {
        let left = residual.dim(1)?;
        let pad_needed = 4 - left;
        let zeros = Tensor::zeros((1, pad_needed, 1280), residual.dtype(), residual.device())?;
        let padded = Tensor::cat(&[&residual, &zeros], 1)?;
        let reshaped = reshape_encoder_output(&padded, 4)?;
        let adapted = model.adapter.forward(&reshaped)?;
        adapter_tokens.push(adapted);
    }

    let audio_embeds = Tensor::cat(&adapter_tokens.iter().collect::<Vec<_>>(), 1)?;
    let audio_seq = audio_embeds.dim(1)?;
    let d_model = audio_embeds.dim(2)?;
    eprintln!("[Streaming] Encoder: {} adapter tokens, {} chunks ({:.2}s)",
        audio_seq, (n_conv + enc_chunk_size - 1) / enc_chunk_size, t_enc.elapsed().as_secs_f32());

    // Step 3: Decoder with context rotation
    let ada_scales = model.precompute_ada_scales(t_embed)?;
    let prompt_len = 39usize.min(audio_seq);
    let prompt_ids: Vec<u32> = std::iter::once(1u32)
        .chain(std::iter::repeat(32u32).take(prompt_len - 1))
        .collect();
    let rotation_interval = 1250; // ~100s of audio (12.5 tokens/s × 100s)
    let eos_token = 2u32;

    // Helper: prefill decoder with streaming prefix at given audio offset
    let prefill_decoder = |dec_caches: &mut Vec<KVCache>, audio_offset: usize| -> Result<u32> {
        let pl = prompt_len.min(audio_seq - audio_offset);
        let prompt_audio = audio_embeds.narrow(1, audio_offset, pl)?;
        let prompt_text = model.embed_tokens(&prompt_ids[..pl])?;
        let prompt_input = (&prompt_audio + &prompt_text)?;
        let prefill_count = pl - 1;
        let _h = model.decoder_forward(prompt_input.narrow(1, 0, prefill_count)?, &ada_scales, dec_caches)?;
        let last_h = model.decoder_forward(prompt_input.narrow(1, prefill_count, 1)?, &ada_scales, dec_caches)?;
        let logits = model.lm_head(&last_h)?;
        let logits_f32: Vec<f32> = logits.to_dtype(DType::F32)?.reshape(131072)?.to_vec1()?;
        Ok(logits_f32.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32).unwrap_or(0))
    };

    let t_dec = Instant::now();
    let mut dec_caches = (0..26).map(|_| KVCache::new()).collect();
    let mut prev_token = prefill_decoder(&mut dec_caches, 0)?;
    let mut generated = Vec::with_capacity(audio_seq);
    generated.push(prev_token);
    let mut steps_since_rotation = 0usize;
    let mut n_rotations = 0usize;

    for i in prompt_len..audio_seq {
        // Context rotation: clean restart from current position (like vllm's disconnect/reconnect)
        // Accept a small text gap (~3s) rather than producing garbled output
        if steps_since_rotation >= rotation_interval {
            dec_caches = (0..26).map(|_| KVCache::new()).collect();
            prev_token = prefill_decoder(&mut dec_caches, i)?;
            // Skip the prompt-length positions (prefill covers them)
            let skip = prompt_len.min(audio_seq - i);
            for _ in 0..skip.saturating_sub(1) {
                // These positions are consumed by prefill, just advance i
                generated.push(32); // pad placeholder
            }
            prev_token = 0; // fresh start — model will produce pad then text
            steps_since_rotation = 0;
            n_rotations += 1;
        }

        let audio_pos = audio_embeds.narrow(1, i, 1)?;
        let text_embed = model.embed_token(prev_token)?;
        let x = (&audio_pos + &text_embed)?;

        let hidden = model.decoder_forward(x, &ada_scales, &mut dec_caches)?;
        // GPU argmax: transfers 1 u32 instead of 131K floats
        let token = model.lm_head_argmax(&hidden)?;

        generated.push(token);
        prev_token = token;
        steps_since_rotation += 1;
        if token == eos_token { break; }
    }

    let decode_ms = t_dec.elapsed().as_millis();
    let n_steps = generated.len() - 1;
    let ms_per_step = if n_steps > 0 { decode_ms as f32 / n_steps as f32 } else { 0.0 };
    let total_s = t0.elapsed().as_secs_f32();
    let audio_secs = audio_seq as f32 / 12.5;
    let text_tokens: Vec<u32> = generated.iter().copied().filter(|&t| t >= 1000).collect();
    eprintln!("[Streaming] Decode: {} steps, {}ms, {:.1}ms/step, {} text tokens, {} rotations",
        n_steps, decode_ms, ms_per_step, text_tokens.len(), n_rotations);
    eprintln!("[Streaming] Total: {:.1}s ({:.1}× realtime for {:.0}s audio)",
        total_s, total_s / audio_secs, audio_secs);

    Ok(generated)
}
