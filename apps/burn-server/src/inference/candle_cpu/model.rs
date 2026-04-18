//! Voxtral Mini 4B Realtime — CPU inference with Q4 GGUF quantization.
//!
//! Architecture (identical to candle_native/GPU version):
//!   Encoder: CausalConv(128→1280, stride 2) → 32 layers (MHA, interleaved RoPE, SW=750)
//!   Adapter: 4x reshape → Linear(5120→3072) → GELU → Linear(3072→3072)
//!   Decoder: 26 layers (GQA 32h/8kv, interleaved RoPE, ADA FFN, SW=8192)
//!
//! Key differences from GPU version:
//!   - Device::Cpu, F32 dtype for activations/KV caches
//!   - QMatMul (candle quantized module) for all weight-heavy layers
//!   - Loads Q4 GGUF weights directly (no SafeTensors)
//!   - Manual softmax attention (no FlashAttention)

use candle_core_cpu::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_core_cpu::quantized::{self, GgmlDType, QMatMul, QTensor};
use candle_nn_cpu as candle_nn;
use std::sync::Arc;
use std::time::Instant;

// ─── Q4 Linear (quantized matmul) ─────────────────────────────────────────

/// Linear layer backed by QMatMul — weight stored in Q4 GGUF format,
/// dequantized on-the-fly with AVX2/AVX-512 SIMD during matmul.
///
/// With `candle-cpu-ggml` feature: uses ggml's AVX-512 vec_dot kernels.
/// Without: falls back to candle's AVX2 QMatMul.
struct QLinear {
    #[cfg(not(feature = "candle-cpu-ggml"))]
    inner: QMatMul,
    #[cfg(feature = "candle-cpu-ggml")]
    q4_data: Vec<u8>,  // raw Q4_0 blocks [n_out, k/32 * 18]
    #[cfg(feature = "candle-cpu-ggml")]
    n_out: usize,
    #[cfg(feature = "candle-cpu-ggml")]
    k: usize,
    bias: Option<Tensor>,
}

impl QLinear {
    fn from_qtensor(qt: Arc<QTensor>, bias: Option<Tensor>) -> Self {
        #[cfg(not(feature = "candle-cpu-ggml"))]
        {
            Self {
                inner: QMatMul::QTensor(qt),
                bias,
            }
        }
        #[cfg(feature = "candle-cpu-ggml")]
        {
            let shape = qt.shape();
            let dims = shape.dims();
            let (n_out, k) = if dims.len() == 2 {
                (dims[0], dims[1])
            } else if dims.len() == 1 {
                (1, dims[0])
            } else {
                panic!("QLinear: unexpected QTensor shape {:?}", dims);
            };
            let q4_data = qt.data().expect("QTensor data access").to_vec();
            Self {
                q4_data,
                n_out,
                k,
                bias,
            }
        }
    }

    #[cfg(not(feature = "candle-cpu-ggml"))]
    fn from_tensor(weight: Tensor, bias: Option<Tensor>) -> Self {
        Self {
            inner: QMatMul::Tensor(weight),
            bias,
        }
    }

    #[cfg(feature = "candle-cpu-ggml")]
    fn from_tensor(_weight: Tensor, _bias: Option<Tensor>) -> Self {
        panic!("F32 decoder mode is not supported with candle-cpu-ggml — use Q4 mode");
    }
}

impl Module for QLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        #[cfg(not(feature = "candle-cpu-ggml"))]
        let out = self.inner.forward(x)?;

        #[cfg(feature = "candle-cpu-ggml")]
        let out = {
            let x_dims = x.dims();
            // Ensure F32 contiguous
            let x_f32 = if x.dtype() == DType::F32 && x.is_contiguous() {
                x.clone()
            } else if x.dtype() == DType::F32 {
                x.contiguous()?
            } else {
                x.to_dtype(DType::F32)?.contiguous()?
            };
            let x_flat = x_f32.flatten_all()?;

            // Zero-copy input: access CpuStorage's f32 slice directly
            let (storage, layout) = x_flat.storage_and_layout();
            let x_slice = match &*storage {
                candle_core_cpu::Storage::Cpu(cpu) => match cpu {
                    candle_core_cpu::CpuStorage::F32(v) => {
                        let offset = layout.start_offset();
                        &v[offset..offset + layout.shape().elem_count()]
                    }
                    _ => unreachable!("expected F32 storage"),
                },
                _ => unreachable!("expected CPU storage"),
            };
            let m = x_slice.len() / self.k;

            let mut dst = vec![0.0f32; m * self.n_out];
            ggml_matmul::q4_matmul(m, self.k, self.n_out, x_slice, &self.q4_data, &mut dst);
            drop(storage); // release read lock before creating output tensor

            let mut out_shape = x_dims.to_vec();
            *out_shape.last_mut().unwrap() = self.n_out;
            Tensor::new(dst, x.device())?.reshape(out_shape)?
        };

        match &self.bias {
            None => Ok(out),
            Some(bias) => out.broadcast_add(bias),
        }
    }
}

// ─── F32 Linear (for small layers: norms, ADA, adapter) ───────────────────

struct Linear {
    weight_t: Tensor, // [in_features, out_features] — pre-transposed, contiguous
    bias: Option<Tensor>,
}

impl Linear {
    fn new(weight: Tensor, bias: Option<Tensor>) -> Result<Self> {
        let weight_t = weight.t()?.contiguous()?;
        Ok(Self { weight_t, bias })
    }
}

impl Module for Linear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let w = if x.dims().len() == 3 {
            self.weight_t.unsqueeze(0)?
        } else {
            self.weight_t.clone()
        };
        let out = x.matmul(&w)?;
        match &self.bias {
            None => Ok(out),
            Some(bias) => out.broadcast_add(bias),
        }
    }
}

// ─── RmsNorm ──────────────────────────────────────────────────────────────

pub struct BurnRmsNorm {
    weight: Tensor,
    eps: f64,
}

impl BurnRmsNorm {
    pub fn new(weight: Tensor, eps: f64) -> Self {
        Self { weight, eps }
    }

    fn weight(&self) -> &Tensor {
        &self.weight
    }
}

impl Module for BurnRmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let hidden_size = x.dim(D::Minus1)?;
        let x_sq = x.sqr()?;
        let mean_sq = (x_sq.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
        let rms = (mean_sq + self.eps)?.sqrt()?;
        x.broadcast_div(&rms)?.broadcast_mul(&self.weight)
    }
}

// ─── RoPE ─────────────────────────────────────────────────────────────────

pub struct RoPE {
    cos: Tensor,
    sin: Tensor,
}

impl RoPE {
    pub fn new(head_dim: usize, max_seq: usize, theta: f64, device: &Device) -> Result<Self> {
        let half = head_dim / 2;
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| 1.0 / (theta as f32).powf(2.0 * i as f32 / head_dim as f32))
            .collect();
        let inv_freq = Tensor::new(inv_freq, device)?;
        let positions: Vec<f32> = (0..max_seq).map(|p| p as f32).collect();
        let positions = Tensor::new(positions, device)?;
        let freqs = positions.unsqueeze(1)?.matmul(&inv_freq.unsqueeze(0)?)?;
        let cos = freqs.cos()?;
        let sin = freqs.sin()?;
        Ok(Self { cos, sin })
    }

    /// Apply interleaved RoPE (is_neox_style=False) to q, k [B, seq, heads, hd].
    fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let seq_len = q.dim(1)?;
        let hd = q.dim(3)?;
        let half = hd / 2;
        let cos = self
            .cos
            .i(offset..offset + seq_len)?
            .unsqueeze(0)?
            .unsqueeze(2)?;
        let sin = self
            .sin
            .i(offset..offset + seq_len)?
            .unsqueeze(0)?
            .unsqueeze(2)?;

        let apply_rope = |x: &Tensor| -> Result<Tensor> {
            let (b, s, h, _d) = x.dims4()?;
            let x_pairs = x.reshape((b, s, h, half, 2))?;
            let x1 = x_pairs.narrow(4, 0, 1)?.squeeze(4)?;
            let x2 = x_pairs.narrow(4, 1, 1)?.squeeze(4)?;
            let o1 = x1.broadcast_mul(&cos)?.broadcast_sub(&x2.broadcast_mul(&sin)?)?;
            let o2 = x2.broadcast_mul(&cos)?.broadcast_add(&x1.broadcast_mul(&sin)?)?;
            let o1 = o1.unsqueeze(4)?;
            let o2 = o2.unsqueeze(4)?;
            Tensor::cat(&[&o1, &o2], 4)?.reshape((b, s, h, hd))
        };

        Ok((apply_rope(q)?, apply_rope(k)?))
    }
}

// ─── KV Cache ─────────────────────────────────────────────────────────────

pub struct KVCache {
    k: Option<Tensor>,
    v: Option<Tensor>,
    pub seq_len: usize,
    pub pos_offset: usize,
}

impl KVCache {
    pub fn new() -> Self {
        Self {
            k: None,
            v: None,
            seq_len: 0,
            pos_offset: 0,
        }
    }

    fn logical_len(&self) -> usize {
        self.pos_offset + self.seq_len
    }

    pub fn compact(&mut self, keep: usize) -> Result<()> {
        if self.seq_len <= keep {
            return Ok(());
        }
        let drop = self.seq_len - keep;
        if let (Some(ck), Some(cv)) = (&self.k, &self.v) {
            self.k = Some(ck.narrow(1, drop, keep)?.contiguous()?);
            self.v = Some(cv.narrow(1, drop, keep)?.contiguous()?);
        }
        self.pos_offset += drop;
        self.seq_len = keep;
        Ok(())
    }

    fn update(&mut self, k: Tensor, v: Tensor) -> Result<(Tensor, Tensor)> {
        let new_seq = k.dim(1)?;
        let (k, v) = match (&self.k, &self.v) {
            (Some(ck), Some(cv)) => {
                let k = Tensor::cat(&[ck, &k], 1)?.contiguous()?;
                let v = Tensor::cat(&[cv, &v], 1)?.contiguous()?;
                (k, v)
            }
            _ => (k.contiguous()?, v.contiguous()?),
        };
        self.k = Some(k.clone());
        self.v = Some(v.clone());
        self.seq_len += new_seq;
        Ok((k, v))
    }
}

// ─── Encoder Layer ────────────────────────────────────────────────────────

struct EncoderAttention {
    wq: QLinear,
    wk: QLinear,
    wv: QLinear,
    wo: QLinear,
    n_heads: usize,
    head_dim: usize,
    attn_dim: usize,
    scale: f32,
    sliding_window: usize,
}

impl EncoderAttention {
    fn forward_with_cache(&self, x: &Tensor, rope: &RoPE, cache: &mut KVCache) -> Result<Tensor> {
        let (b, seq, _) = x.dims3()?;
        let offset = cache.logical_len();
        let q = self
            .wq
            .forward(x)?
            .reshape((b, seq, self.n_heads, self.head_dim))?;
        let k = self
            .wk
            .forward(x)?
            .reshape((b, seq, self.n_heads, self.head_dim))?;
        let v = self
            .wv
            .forward(x)?
            .reshape((b, seq, self.n_heads, self.head_dim))?;
        let (q, k) = rope.apply(&q, &k, offset)?;

        let (full_k, full_v) = cache.update(k, v)?;

        // [B, seq, heads, hd] → [B, heads, seq, hd]
        let q = q.transpose(1, 2)?.contiguous()?;
        let full_k = full_k.transpose(1, 2)?.contiguous()?;
        let full_v = full_v.transpose(1, 2)?.contiguous()?;
        let kv_seq = full_k.dim(2)?;

        let scores =
            (q.matmul(&full_k.transpose(2, 3)?.contiguous()?)? * (self.scale as f64))?;

        // Causal + sliding window mask
        let kv_pos_offset = cache.pos_offset;
        let mask: Vec<f32> = (0..seq)
            .flat_map(|i| {
                let qi_abs = offset + i;
                (0..kv_seq).map(move |j| {
                    let kj_abs = kv_pos_offset + j;
                    if kj_abs > qi_abs {
                        f32::NEG_INFINITY
                    } else if qi_abs - kj_abs >= self.sliding_window {
                        f32::NEG_INFINITY
                    } else {
                        0.0
                    }
                })
            })
            .collect();
        let mask = Tensor::new(mask, x.device())?
            .to_dtype(scores.dtype())?
            .reshape((1, 1, seq, kv_seq))?;
        let scores = scores.broadcast_add(&mask)?;

        // Softmax
        let max_vals = scores.max_keepdim(D::Minus1)?;
        let shifted = scores.broadcast_sub(&max_vals)?;
        let exp_vals = shifted.exp()?;
        let sum_vals = exp_vals.sum_keepdim(D::Minus1)?;
        let attn = exp_vals.broadcast_div(&sum_vals)?;

        let out = attn.matmul(&full_v)?;
        let out = out
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, seq, self.attn_dim))?;
        self.wo.forward(&out)
    }
}

struct EncoderSwiGLU {
    w1: QLinear,
    w2: QLinear,
    w3: QLinear,
}

impl EncoderSwiGLU {
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
    pub fn forward(&self, x: &Tensor, rope: &RoPE, cache: &mut KVCache) -> Result<Tensor> {
        let residual = x.clone();
        let x = self.attention_norm.forward(x)?;
        let x = (self.attention.forward_with_cache(&x, rope, cache)? + &residual)?;
        let residual = x.clone();
        let h = self.ffn_norm.forward(&x)?;
        (self.ffn.forward(&h)? + &residual)
    }
}

// ─── Audio Encoder ────────────────────────────────────────────────────────

pub struct ConvDownsampler {
    conv1_weight: Tensor,
    conv1_bias: Tensor,
    conv2_weight: Tensor,
    conv2_bias: Tensor,
}

impl ConvDownsampler {
    pub fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        let x = mel.pad_with_zeros(2, 2, 0)?;
        let x = x.conv1d(&self.conv1_weight, 0, 1, 1, 1)?;
        let bias1 = self.conv1_bias.unsqueeze(0)?.unsqueeze(D::Minus1)?;
        let x = x.broadcast_add(&bias1)?.gelu()?;
        let x = x.pad_with_zeros(2, 1, 0)?;
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

// ─── Adapter ──────────────────────────────────────────────────────────────

pub struct AudioLanguageAdapter {
    linear1: QLinear,
    linear2: QLinear,
}

impl AudioLanguageAdapter {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.linear1.forward(x)?.gelu()?;
        self.linear2.forward(&x)
    }
}

// ─── Decoder Layer ────────────────────────────────────────────────────────

struct DecoderAttention {
    wq: QLinear,
    wk: QLinear,
    wv: QLinear,
    wo: QLinear,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    q_dim: usize,
    kv_dim: usize,
    scale: f32,
}

impl DecoderAttention {
    fn forward_with_cache(&self, x: &Tensor, rope: &RoPE, cache: &mut KVCache) -> Result<Tensor> {
        let (b, seq, _) = x.dims3()?;
        let offset = cache.logical_len();

        let q = self
            .wq
            .forward(x)?
            .reshape((b, seq, self.n_heads, self.head_dim))?;
        let k = self
            .wk
            .forward(x)?
            .reshape((b, seq, self.n_kv_heads, self.head_dim))?;
        let v = self
            .wv
            .forward(x)?
            .reshape((b, seq, self.n_kv_heads, self.head_dim))?;

        let (q, k) = rope.apply(&q, &k, offset)?;
        let (k, v) = cache.update(k.contiguous()?, v.contiguous()?)?;

        // [B, seq/kv_seq, heads, hd] → [B, heads, seq/kv_seq, hd]
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        // GQA expand
        let repeat = self.n_heads / self.n_kv_heads;
        let kv_seq = k.dim(2)?;
        let k = k
            .unsqueeze(2)?
            .repeat(&[1, 1, repeat, 1, 1])?
            .reshape((b, self.n_heads, kv_seq, self.head_dim))?
            .contiguous()?;
        let v = v
            .unsqueeze(2)?
            .repeat(&[1, 1, repeat, 1, 1])?
            .reshape((b, self.n_heads, kv_seq, self.head_dim))?
            .contiguous()?;

        let scores = (q.matmul(&k.t()?)? * (self.scale as f64))?;

        let scores = if seq > 1 {
            let mask: Vec<f32> = (0..seq)
                .flat_map(|i| {
                    (0..kv_seq).map(move |j| {
                        if j > i + offset {
                            f32::NEG_INFINITY
                        } else {
                            0.0
                        }
                    })
                })
                .collect();
            let mask = Tensor::new(mask, x.device())?
                .to_dtype(scores.dtype())?
                .reshape((1, 1, seq, kv_seq))?;
            scores.broadcast_add(&mask)?
        } else {
            scores
        };

        let attn = candle_nn::ops::softmax_last_dim(&scores)?;
        let out = attn.matmul(&v)?;
        let out = out
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, seq, self.q_dim))?;
        self.wo.forward(&out)
    }
}

struct ADANorm {
    w0: Tensor, // [d_model, t_cond_dim] — down projection
    w2: Tensor, // [t_cond_dim, d_model] — up projection
}

impl ADANorm {
    fn compute_scale(&self, t_embed: &Tensor) -> Result<Tensor> {
        let w0_t = self.w0.t()?.unsqueeze(0)?;
        let w2_t = self.w2.t()?.unsqueeze(0)?;
        let h = t_embed.matmul(&w0_t)?.gelu()?;
        let scale = h.matmul(&w2_t)?;
        let ones = Tensor::ones_like(&scale)?;
        (&scale + &ones)
    }
}

struct DecoderSwiGLU {
    w1: QLinear, // gate
    w3: QLinear, // up
    w2: QLinear, // down
}

impl DecoderSwiGLU {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.w1.forward(x)?.silu()?;
        let up = self.w3.forward(x)?;
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
    fn forward_with_cache_precomputed(
        &self,
        x: &Tensor,
        ada_scale: &Tensor,
        rope: &RoPE,
        cache: &mut KVCache,
    ) -> Result<Tensor> {
        let residual = x.clone();
        let normed = self.attention_norm.forward(x)?;
        let attn_out = self.attention.forward_with_cache(&normed, rope, cache)?;
        let x = (&attn_out + &residual)?;
        let residual = x.clone();
        let h = self.ffn_norm.forward(&x)?;
        let h = h.broadcast_mul(ada_scale)?;
        let ffn_out = self.ffn.forward(&h)?;
        (&ffn_out + &residual)
    }
}

// ─── Full Model ───────────────────────────────────────────────────────────

pub struct VoxtralModel {
    pub encoder: AudioEncoder,
    pub adapter: AudioLanguageAdapter,
    decoder_layers: Vec<DecoderLayer>,
    decoder_norm: BurnRmsNorm,
    decoder_rope: RoPE,
    tok_embed_q: QMatMul,       // Q4 for lm_head (matmul)
    tok_embed_f32: Tensor,      // F32 dequantized for embed_token lookups
    device: Device,
}

/// GGUF loading context — avoids closure borrow issues with &mut File.
struct GgufLoader {
    file: std::fs::File,
    tensor_infos: std::collections::HashMap<String, quantized::gguf_file::TensorInfo>,
    tensor_data_offset: u64,
}

impl GgufLoader {
    fn open(path: &std::path::Path) -> Result<Self> {
        let mut file = std::fs::File::open(path)
            .map_err(|e| candle_core_cpu::Error::Msg(format!("open GGUF: {}", e)))?;
        let content = quantized::gguf_file::Content::read(&mut file)
            .map_err(|e| candle_core_cpu::Error::Msg(format!("read GGUF: {}", e)))?;
        eprintln!(
            "[CandleCpu] GGUF: {} tensors, version {:?}",
            content.tensor_infos.len(),
            content.magic
        );
        Ok(Self {
            file,
            tensor_infos: content.tensor_infos,
            tensor_data_offset: content.tensor_data_offset,
        })
    }

    fn load_qtensor(&mut self, name: &str, device: &Device) -> Result<Arc<QTensor>> {
        let info = self.tensor_infos.get(name).ok_or_else(|| {
            candle_core_cpu::Error::Msg(format!("tensor '{}' not found in GGUF", name))
        })?;
        let qt = info.read(&mut self.file, self.tensor_data_offset, device)?;
        Ok(Arc::new(qt))
    }

    fn load_qlinear(&mut self, name: &str, device: &Device) -> Result<QLinear> {
        let qt = self.load_qtensor(name, device)?;
        Ok(QLinear::from_qtensor(qt, None))
    }

    fn load_f32(&mut self, name: &str, device: &Device) -> Result<Tensor> {
        let qt = self.load_qtensor(name, device)?;
        qt.dequantize(device)
    }

    fn load_qlinear_b(
        &mut self,
        weight_name: &str,
        bias_name: &str,
        device: &Device,
    ) -> Result<QLinear> {
        let qt = self.load_qtensor(weight_name, device)?;
        let bias = self.load_f32(bias_name, device)?;
        Ok(QLinear::from_qtensor(qt, Some(bias)))
    }

    /// Load weight as F32 Linear (dequantize Q4 → F32 at load time).
    /// Uses gemm-backed matmul (multi-threaded, cache-tiled) instead of Q4 dequant-on-the-fly.
    #[cfg(not(feature = "candle-cpu-ggml"))]
    fn load_f32_linear(&mut self, name: &str, device: &Device) -> Result<QLinear> {
        let qt = self.load_qtensor(name, device)?;
        let weight = qt.dequantize(device)?; // [out, in] F32
        Ok(QLinear::from_tensor(weight, None))
    }
}

impl VoxtralModel {
    /// Load from Q4 GGUF file.
    pub fn load(gguf_path: &std::path::Path, device: &Device) -> Result<Self> {
        let t0 = Instant::now();
        eprintln!("[CandleCpu] Loading Q4 GGUF from {}", gguf_path.display());

        let mut gguf = GgufLoader::open(gguf_path)?;

        // ─── Encoder ──────────────────────────────────────────────────
        let enc_prefix = "mm_streams_embeddings.embedding_module.whisper_encoder";

        // Conv (keep in F32 — small, needs precision for GELU activation)
        let conv = ConvDownsampler {
            conv1_weight: gguf.load_f32(&format!("{enc_prefix}.conv_layers.0.conv.weight"), device)?,
            conv1_bias: gguf.load_f32(&format!("{enc_prefix}.conv_layers.0.conv.bias"), device)?,
            conv2_weight: gguf.load_f32(&format!("{enc_prefix}.conv_layers.1.conv.weight"), device)?,
            conv2_bias: gguf.load_f32(&format!("{enc_prefix}.conv_layers.1.conv.bias"), device)?,
        };

        let enc_rope = RoPE::new(64, 200_000, 1_000_000.0, device)?;

        let mut enc_layers = Vec::with_capacity(32);
        for i in 0..32 {
            let lp = format!("{enc_prefix}.transformer.layers.{i}");
            let attention_norm =
                BurnRmsNorm::new(gguf.load_f32(&format!("{lp}.attention_norm.weight"), device)?, 1e-5);
            let ffn_norm = BurnRmsNorm::new(gguf.load_f32(&format!("{lp}.ffn_norm.weight"), device)?, 1e-5);
            let attention = EncoderAttention {
                wq: gguf.load_qlinear_b(
                    &format!("{lp}.attention.wq.weight"),
                    &format!("{lp}.attention.wq.bias"),
                    device,
                )?,
                wk: gguf.load_qlinear(&format!("{lp}.attention.wk.weight"), device)?,
                wv: gguf.load_qlinear_b(
                    &format!("{lp}.attention.wv.weight"),
                    &format!("{lp}.attention.wv.bias"),
                    device,
                )?,
                wo: gguf.load_qlinear_b(
                    &format!("{lp}.attention.wo.weight"),
                    &format!("{lp}.attention.wo.bias"),
                    device,
                )?,
                n_heads: 32,
                head_dim: 64,
                attn_dim: 2048,
                scale: (64.0f32).powf(-0.5),
                sliding_window: 750,
            };
            let ffn = EncoderSwiGLU {
                w1: gguf.load_qlinear(&format!("{lp}.feed_forward.w1.weight"), device)?,
                w2: gguf.load_qlinear_b(
                    &format!("{lp}.feed_forward.w2.weight"),
                    &format!("{lp}.feed_forward.w2.bias"),
                    device,
                )?,
                w3: gguf.load_qlinear(&format!("{lp}.feed_forward.w3.weight"), device)?,
            };
            enc_layers.push(EncoderLayer {
                attention_norm,
                attention,
                ffn_norm,
                ffn,
            });
            if (i + 1) % 8 == 0 {
                eprintln!("[CandleCpu] encoder layer {}/32", i + 1);
            }
        }

        let enc_norm = BurnRmsNorm::new(
            gguf.load_f32(&format!("{enc_prefix}.transformer.norm.weight"), device)?,
            1e-5,
        );

        let encoder = AudioEncoder {
            conv,
            rope: enc_rope,
            layers: enc_layers,
            norm: enc_norm,
        };

        // ─── Adapter ─────────────────────────────────────────────────
        let adp_prefix = "mm_streams_embeddings.embedding_module.audio_language_projection";
        let adapter = AudioLanguageAdapter {
            linear1: gguf.load_qlinear(&format!("{adp_prefix}.0.weight"), device)?,
            linear2: gguf.load_qlinear(&format!("{adp_prefix}.2.weight"), device)?,
        };

        eprintln!(
            "[CandleCpu] Encoder+Adapter loaded ({:.1}s)",
            t0.elapsed().as_secs_f32()
        );

        // ─── Decoder ─────────────────────────────────────────────────
        // F32 mode: dequantize decoder weights at load time for gemm-backed matmul.
        // Not available with ggml feature (ggml uses Q4 directly with AVX-512).
        #[cfg(not(feature = "candle-cpu-ggml"))]
        let f32_decoder = std::env::var("CANDLE_CPU_F32_DECODER").is_ok();
        #[cfg(feature = "candle-cpu-ggml")]
        let f32_decoder = false;
        if f32_decoder {
            eprintln!("[CandleCpu] F32 decoder mode — dequantizing Q4 weights to F32 for threaded GEMM");
        }

        let decoder_rope = RoPE::new(128, 16384, 1_000_000.0, device)?;
        let mut decoder_layers = Vec::with_capacity(26);
        for i in 0..26 {
            let lp = format!("layers.{i}");
            let ada = ADANorm {
                w0: gguf.load_f32(&format!("{lp}.ada_rms_norm_t_cond.0.weight"), device)?,
                w2: gguf.load_f32(&format!("{lp}.ada_rms_norm_t_cond.2.weight"), device)?,
            };
            let attention_norm =
                BurnRmsNorm::new(gguf.load_f32(&format!("{lp}.attention_norm.weight"), device)?, 1e-5);
            let ffn_norm = BurnRmsNorm::new(gguf.load_f32(&format!("{lp}.ffn_norm.weight"), device)?, 1e-5);

            let mut load_weight = |name: &str| -> Result<QLinear> {
                #[cfg(not(feature = "candle-cpu-ggml"))]
                if f32_decoder {
                    return gguf.load_f32_linear(name, device);
                }
                gguf.load_qlinear(name, device)
            };

            let attention = DecoderAttention {
                wq: load_weight(&format!("{lp}.attention.wq.weight"))?,
                wk: load_weight(&format!("{lp}.attention.wk.weight"))?,
                wv: load_weight(&format!("{lp}.attention.wv.weight"))?,
                wo: load_weight(&format!("{lp}.attention.wo.weight"))?,
                n_heads: 32,
                n_kv_heads: 8,
                head_dim: 128,
                q_dim: 4096,
                kv_dim: 1024,
                scale: (128.0f32).powf(-0.5),
            };
            let ffn = DecoderSwiGLU {
                w1: load_weight(&format!("{lp}.feed_forward.w1.weight"))?,
                w3: load_weight(&format!("{lp}.feed_forward.w3.weight"))?,
                w2: load_weight(&format!("{lp}.feed_forward.w2.weight"))?,
            };
            decoder_layers.push(DecoderLayer {
                ada,
                attention_norm,
                attention,
                ffn_norm,
                ffn,
            });
            if (i + 1) % 5 == 0 {
                eprintln!("[CandleCpu] decoder layer {}/26", i + 1);
            }
        }
        let decoder_norm = BurnRmsNorm::new(gguf.load_f32("norm.weight", device)?, 1e-5);

        // ─── Token Embeddings ─────────────────────────────────────────
        let tok_name = "mm_streams_embeddings.embedding_module.tok_embeddings.weight";
        let tok_qt = gguf.load_qtensor(tok_name, device)?;
        // Keep Q4 version for lm_head matmul (efficient)
        let tok_embed_q = QMatMul::QTensor(tok_qt.clone());
        // Dequantize to F32 for embed_token lookups (index_select needs dense tensor)
        let tok_embed_f32 = tok_qt.dequantize(device)?;
        eprintln!(
            "[CandleCpu] Token embedding dequantized ({:.0} MB F32)",
            131072.0 * 3072.0 * 4.0 / 1e6
        );

        eprintln!(
            "[CandleCpu] Full model loaded in {:.1}s",
            t0.elapsed().as_secs_f32()
        );

        Ok(Self {
            encoder,
            adapter,
            decoder_layers,
            decoder_norm,
            decoder_rope,
            tok_embed_q,
            tok_embed_f32,
            device: device.clone(),
        })
    }

    /// Encode audio with KV cache support for incremental processing.
    pub fn encode_audio(&self, mel: &Tensor, enc_caches: &mut Vec<KVCache>) -> Result<Tensor> {
        let conv_out = self.encoder.conv.forward(mel)?;
        let n_frames = conv_out.dim(1)?;
        let n_aligned = (n_frames / 4) * 4;
        let x = if n_aligned < n_frames {
            conv_out.narrow(1, 0, n_aligned)?
        } else {
            conv_out
        };
        let mut x = x;
        for (i, layer) in self.encoder.layers.iter().enumerate() {
            x = layer.forward(&x, &self.encoder.rope, &mut enc_caches[i])?;
        }
        let enc_out = self.encoder.norm.forward(&x)?;
        let reshaped = reshape_encoder_output(&enc_out, 4)?;
        self.adapter.forward(&reshaped)
    }

    pub fn new_encoder_caches() -> Vec<KVCache> {
        (0..32).map(|_| KVCache::new()).collect()
    }

    pub fn new_decoder_caches() -> Vec<KVCache> {
        (0..26).map(|_| KVCache::new()).collect()
    }

    pub fn precompute_ada_scales(&self, t_embed: &Tensor) -> Result<Vec<Tensor>> {
        self.decoder_layers
            .iter()
            .map(|layer| layer.ada.compute_scale(t_embed))
            .collect()
    }

    pub fn decoder_forward(
        &self,
        mut x: Tensor,
        ada_scales: &[Tensor],
        caches: &mut Vec<KVCache>,
    ) -> Result<Tensor> {
        let profile = std::env::var("CANDLE_PROFILE").is_ok();
        let t0 = if profile { Some(Instant::now()) } else { None };
        for (i, layer) in self.decoder_layers.iter().enumerate() {
            x = layer.forward_with_cache_precomputed(
                &x,
                &ada_scales[i],
                &self.decoder_rope,
                &mut caches[i],
            )?;
        }
        let out = self.decoder_norm.forward(&x)?;
        if let Some(t) = t0 {
            eprintln!("[CandleCpu PROFILE] decoder_forward: {:.1}ms", t.elapsed().as_secs_f32() * 1000.0);
        }
        Ok(out)
    }

    /// lm_head: hidden @ tok_embed^T → logits (Q4 quantized matmul)
    pub fn lm_head(&self, hidden: &Tensor) -> Result<Tensor> {
        let profile = std::env::var("CANDLE_PROFILE").is_ok();
        let t0 = if profile { Some(Instant::now()) } else { None };
        let out = self.tok_embed_q.forward(hidden)?;
        if let Some(t) = t0 {
            eprintln!("[CandleCpu PROFILE] lm_head: {:.1}ms", t.elapsed().as_secs_f32() * 1000.0);
        }
        Ok(out)
    }

    /// lm_head + CPU argmax.
    pub fn lm_head_argmax(&self, hidden: &Tensor) -> Result<u32> {
        let logits = self.lm_head(hidden)?;
        let logits_f32: Vec<f32> = logits.to_dtype(DType::F32)?.reshape(131072)?.to_vec1()?;
        Ok(logits_f32
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(0))
    }

    /// CPU embedding lookup for a single token ID.
    pub fn embed_token(&self, token_id: u32) -> Result<Tensor> {
        self.tok_embed_f32
            .i(token_id as usize)?
            .contiguous()?
            .unsqueeze(0)?
            .unsqueeze(0)
    }

    /// Embed multiple token IDs: [len] → [1, len, d_model]
    pub fn embed_tokens(&self, ids: &[u32]) -> Result<Tensor> {
        let indices = Tensor::new(ids, &self.device)?;
        self.tok_embed_f32
            .index_select(&indices, 0)?
            .contiguous()?
            .unsqueeze(0)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────

pub fn reshape_encoder_output(x: &Tensor, factor: usize) -> Result<Tensor> {
    let (batch, seq, dim) = x.dims3()?;
    let new_seq = seq / factor;
    let truncated = new_seq * factor;
    let x = if truncated < seq {
        x.narrow(1, 0, truncated)?
    } else {
        x.clone()
    };
    x.reshape((batch, new_seq, dim * factor))
}

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
    Tensor::new(embedding, device)?.reshape((1, 1, dim))
}

/// Full transcription (non-streaming) — for benchmarking.
pub fn transcribe(model: &VoxtralModel, mel: &Tensor, t_embed: &Tensor) -> Result<Vec<u32>> {
    let t0 = Instant::now();

    let mut enc_caches = VoxtralModel::new_encoder_caches();
    let audio_embeds = model.encode_audio(mel, &mut enc_caches)?;
    let audio_seq = audio_embeds.dim(1)?;
    eprintln!(
        "[CandleCpu] Encoded: seq_len={} ({:.1}s)",
        audio_seq,
        t0.elapsed().as_secs_f32()
    );

    let mut caches = VoxtralModel::new_decoder_caches();
    let ada_scales = model.precompute_ada_scales(t_embed)?;

    let prompt_len = 39usize.min(audio_seq);
    let prompt_ids: Vec<u32> = std::iter::once(1u32)
        .chain(std::iter::repeat(32u32).take(prompt_len - 1))
        .collect();

    let prefill_count = prompt_len - 1;
    let prompt_audio = audio_embeds.narrow(1, 0, prompt_len)?;
    let prompt_text = model.embed_tokens(&prompt_ids)?;
    let prompt_input = (&prompt_audio + &prompt_text)?;

    let t1 = Instant::now();
    let prefill_input = prompt_input.narrow(1, 0, prefill_count)?;
    let _prefill_hidden = model.decoder_forward(prefill_input, &ada_scales, &mut caches)?;

    let last_input = prompt_input.narrow(1, prefill_count, 1)?;
    let last_hidden = model.decoder_forward(last_input, &ada_scales, &mut caches)?;
    let mut prev_token = model.lm_head_argmax(&last_hidden)?;
    let prefill_ms = t1.elapsed().as_millis();
    eprintln!("[CandleCpu] Prefill {}: {}ms", prompt_len, prefill_ms);

    // Autoregressive decode
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
        if token == eos_token {
            break;
        }
    }

    let decode_ms = t2.elapsed().as_millis();
    let n_steps = generated.len() - 1;
    let ms_per_step = if n_steps > 0 {
        decode_ms as f32 / n_steps as f32
    } else {
        0.0
    };
    let total_s = t0.elapsed().as_secs_f32();
    let audio_secs = audio_seq as f32 / 12.5;
    let text_tokens: Vec<u32> = generated.iter().copied().filter(|&t| t >= 1000).collect();
    eprintln!(
        "[CandleCpu] Decode: {} steps, {}ms, {:.1}ms/step, {} text tokens",
        n_steps, decode_ms, ms_per_step,
        text_tokens.len()
    );
    eprintln!(
        "[CandleCpu] Total: {:.1}s ({:.1}× realtime for {:.0}s audio)",
        total_s,
        total_s / audio_secs,
        audio_secs
    );

    Ok(generated)
}

/// Streaming transcription with context rotation.
pub fn transcribe_streaming(
    model: &VoxtralModel,
    mel: &Tensor,
    t_embed: &Tensor,
) -> Result<Vec<u32>> {
    let t0 = Instant::now();

    // Step 1: Conv stem on full mel
    let conv_out = model.encoder.conv.forward(mel)?;
    let n_conv = conv_out.dim(1)?;
    let n_aligned = (n_conv / 4) * 4;
    let conv_out = if n_aligned < n_conv {
        conv_out.narrow(1, 0, n_aligned)?
    } else {
        conv_out
    };
    let n_conv = conv_out.dim(1)?;
    eprintln!(
        "[CandleCpu Streaming] Conv: {} frames ({:.2}s)",
        n_conv,
        t0.elapsed().as_secs_f32()
    );

    // Step 2: Encoder in chunks
    let enc_chunk_size = 100;
    let sliding_window = 750;
    let mut enc_caches = VoxtralModel::new_encoder_caches();
    let mut adapter_tokens: Vec<Tensor> = Vec::new();
    let mut enc_residual: Option<Tensor> = None;

    let t_enc = Instant::now();
    let mut pos = 0;
    while pos < n_conv {
        let chunk_end = (pos + enc_chunk_size).min(n_conv);
        let chunk = conv_out.narrow(1, pos, chunk_end - pos)?;

        if enc_caches[0].seq_len + (chunk_end - pos) > sliding_window {
            let keep = sliding_window / 2;
            for cache in &mut enc_caches {
                cache.compact(keep)?;
            }
        }

        let mut x = chunk;
        for (i, layer) in model.encoder.layers.iter().enumerate() {
            x = layer.forward(&x, &model.encoder.rope, &mut enc_caches[i])?;
        }
        let enc_out = model.encoder.norm.forward(&x)?;

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

    // Flush residual
    if let Some(residual) = enc_residual {
        let left = residual.dim(1)?;
        let pad_needed = 4 - left;
        let zeros = Tensor::zeros((1, pad_needed, 1280), DType::F32, residual.device())?;
        let padded = Tensor::cat(&[&residual, &zeros], 1)?;
        let reshaped = reshape_encoder_output(&padded, 4)?;
        let adapted = model.adapter.forward(&reshaped)?;
        adapter_tokens.push(adapted);
    }

    let audio_embeds = Tensor::cat(&adapter_tokens.iter().collect::<Vec<_>>(), 1)?;
    let audio_seq = audio_embeds.dim(1)?;
    eprintln!(
        "[CandleCpu Streaming] Encoder: {} adapter tokens ({:.2}s)",
        audio_seq,
        t_enc.elapsed().as_secs_f32()
    );

    // Step 3: Decoder
    let ada_scales = model.precompute_ada_scales(t_embed)?;
    let prompt_len = 39usize.min(audio_seq);
    let prompt_ids: Vec<u32> = std::iter::once(1u32)
        .chain(std::iter::repeat(32u32).take(prompt_len - 1))
        .collect();
    let rotation_interval = 1250;
    let eos_token = 2u32;

    let prefill_decoder =
        |dec_caches: &mut Vec<KVCache>, audio_offset: usize| -> Result<u32> {
            let pl = prompt_len.min(audio_seq - audio_offset);
            let prompt_audio = audio_embeds.narrow(1, audio_offset, pl)?;
            let prompt_text = model.embed_tokens(&prompt_ids[..pl])?;
            let prompt_input = (&prompt_audio + &prompt_text)?;
            let prefill_count = pl - 1;
            let _h = model.decoder_forward(
                prompt_input.narrow(1, 0, prefill_count)?,
                &ada_scales,
                dec_caches,
            )?;
            let last_h = model.decoder_forward(
                prompt_input.narrow(1, prefill_count, 1)?,
                &ada_scales,
                dec_caches,
            )?;
            model.lm_head_argmax(&last_h)
        };

    let t_dec = Instant::now();
    let mut dec_caches = VoxtralModel::new_decoder_caches();
    let mut prev_token = prefill_decoder(&mut dec_caches, 0)?;
    let mut generated = Vec::with_capacity(audio_seq);
    generated.push(prev_token);
    let mut steps_since_rotation = 0usize;

    for i in prompt_len..audio_seq {
        if steps_since_rotation >= rotation_interval {
            dec_caches = VoxtralModel::new_decoder_caches();
            prev_token = prefill_decoder(&mut dec_caches, i)?;
            steps_since_rotation = 0;
        }

        let audio_pos = audio_embeds.narrow(1, i, 1)?;
        let text_embed = model.embed_token(prev_token)?;
        let x = (&audio_pos + &text_embed)?;
        let hidden = model.decoder_forward(x, &ada_scales, &mut dec_caches)?;
        let token = model.lm_head_argmax(&hidden)?;
        generated.push(token);
        prev_token = token;
        steps_since_rotation += 1;
        if token == eos_token {
            break;
        }
    }

    let decode_ms = t_dec.elapsed().as_millis();
    let n_steps = generated.len() - 1;
    let ms_per_step = if n_steps > 0 {
        decode_ms as f32 / n_steps as f32
    } else {
        0.0
    };
    let total_s = t0.elapsed().as_secs_f32();
    let audio_secs = audio_seq as f32 / 12.5;
    let text_tokens: Vec<u32> = generated.iter().copied().filter(|&t| t >= 1000).collect();
    eprintln!(
        "[CandleCpu Streaming] Decode: {} steps, {}ms, {:.1}ms/step, {} text tokens",
        n_steps, decode_ms, ms_per_step,
        text_tokens.len()
    );
    eprintln!(
        "[CandleCpu Streaming] Total: {:.1}s ({:.1}× realtime for {:.0}s audio)",
        total_s,
        total_s / audio_secs,
        audio_secs
    );

    Ok(generated)
}
