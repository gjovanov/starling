//! AR LLM forward pass with KV cache.
//!
//! Single-step + prefill forward: take an `[B, S, D]` input embedding
//! stream + a starting position, run through 26 layers, return the
//! `[B, S, D]` hidden states (or just the final position for the
//! autoregressive decode case). The caller handles `lm_head` projection
//! and sampling.
//!
//! KV cache is per-layer and grows on each call. Callers must supply
//! a fresh `ArLlmKvCache::new(...)` per generation and feed it to
//! every step.

use anyhow::{anyhow, Result};
use candle_core::{DType, Device, IndexOp, Module, Tensor, D};

use super::args::ArLlmArgs;
use super::model::{ArLlmAttention, ArLlmBlock, ArLlmModel};

/// Per-layer growing KV cache. Stores `[B, S_total, n_kv_heads, head_dim]`
/// tensors that get appended on every forward step.
pub struct LayerKvCache {
    pub k: Option<Tensor>,
    pub v: Option<Tensor>,
}

impl LayerKvCache {
    pub fn empty() -> Self {
        Self { k: None, v: None }
    }

    /// `k_new`, `v_new` shape: `[B, S, n_kv_heads, head_dim]`.
    /// Grows the stored tensor along axis 1.
    pub fn append(&mut self, k_new: &Tensor, v_new: &Tensor) -> Result<()> {
        self.k = Some(match self.k.take() {
            None => k_new.clone(),
            Some(prev) => Tensor::cat(&[&prev, k_new], 1)?.contiguous()?,
        });
        self.v = Some(match self.v.take() {
            None => v_new.clone(),
            Some(prev) => Tensor::cat(&[&prev, v_new], 1)?.contiguous()?,
        });
        Ok(())
    }

    pub fn seq_len(&self) -> usize {
        self.k.as_ref().map(|k| k.dim(1).unwrap_or(0)).unwrap_or(0)
    }
}

pub struct ArLlmKvCache {
    pub layers: Vec<LayerKvCache>,
    /// Position of the next token to be added. Starts at 0; advances
    /// by `S` on every forward call.
    pub pos: usize,
}

impl ArLlmKvCache {
    pub fn new(n_layers: usize) -> Self {
        Self {
            layers: (0..n_layers).map(|_| LayerKvCache::empty()).collect(),
            pos: 0,
        }
    }

    pub fn reset(&mut self) {
        for l in self.layers.iter_mut() {
            l.k = None;
            l.v = None;
        }
        self.pos = 0;
    }
}

/// RoPE cos/sin tables precomputed for `max_seq_len` × `head_dim/2`.
/// Stored at the model's compute dtype.
///
/// Layout matches what `candle_nn::rotary_emb::rope_i` expects: one
/// entry per *pair* of head dims. The kernel internally applies the
/// pair to both `(2k, 2k+1)` slots of the input.
pub struct RopeCache {
    pub cos: Tensor, // [max_seq_len, head_dim/2]
    pub sin: Tensor, // [max_seq_len, head_dim/2]
}

impl RopeCache {
    pub fn new(
        max_seq_len: usize,
        head_dim: usize,
        theta: f64,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        // Interleaved RoPE: for each pair index k ∈ [0, head_dim/2):
        //   freq[k] = theta^(-2k/d)
        //   cos[p, k] = cos(p * freq[k])
        //   sin[p, k] = sin(p * freq[k])
        // candle's `rope_i` duplicates per-pair internally.
        let half = head_dim / 2;
        let mut freqs = Vec::with_capacity(half);
        for k in 0..half {
            let exponent = -((2 * k) as f32) / head_dim as f32;
            let f = (theta as f32).powf(exponent);
            freqs.push(f);
        }
        let mut cos_data = Vec::with_capacity(max_seq_len * half);
        let mut sin_data = Vec::with_capacity(max_seq_len * half);
        for p in 0..max_seq_len {
            for k in 0..half {
                let arg = p as f32 * freqs[k];
                cos_data.push(arg.cos());
                sin_data.push(arg.sin());
            }
        }
        let cos = Tensor::from_vec(cos_data, (max_seq_len, half), device)?.to_dtype(dtype)?;
        let sin = Tensor::from_vec(sin_data, (max_seq_len, half), device)?.to_dtype(dtype)?;
        Ok(Self { cos, sin })
    }

    /// Slice `[start..start+seq, :]` of cos/sin for a forward call.
    pub fn slice(&self, start: usize, seq_len: usize) -> Result<(Tensor, Tensor)> {
        let cos = self.cos.i((start..start + seq_len, ..))?.contiguous()?;
        let sin = self.sin.i((start..start + seq_len, ..))?.contiguous()?;
        Ok((cos, sin))
    }
}

/// Apply interleaved RoPE: paired (2k, 2k+1) rotation per head.
/// `x` shape: `[B, S, H, D]`. `cos`, `sin` shape: `[S, D]`.
fn apply_rope_i(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    // candle has rope_i for [B, H, S, D] layout. Transpose, apply, transpose back.
    let x_bhsd = x.transpose(1, 2)?.contiguous()?; // [B, H, S, D]
    let out = candle_nn::rotary_emb::rope_i(&x_bhsd, cos, sin)?; // [B, H, S, D]
    Ok(out.transpose(1, 2)?.contiguous()?) // back to [B, S, H, D]
}

/// Run one attention block.
fn attention_forward(
    a: &ArLlmAttention,
    x: &Tensor,
    cache: &mut LayerKvCache,
    cos: &Tensor,
    sin: &Tensor,
    causal_mask: Option<&Tensor>,
) -> Result<Tensor> {
    let (b, s, _) = x.dims3()?;

    let xq = a.wq.forward(x)?.reshape((b, s, a.n_heads, a.head_dim))?;
    let xk = a.wk.forward(x)?.reshape((b, s, a.n_kv_heads, a.head_dim))?;
    let xv = a.wv.forward(x)?.reshape((b, s, a.n_kv_heads, a.head_dim))?;

    let xq = apply_rope_i(&xq, cos, sin)?;
    let xk = apply_rope_i(&xk, cos, sin)?;

    cache.append(&xk, &xv)?;
    let k_full = cache.k.as_ref().expect("just appended").clone();
    let v_full = cache.v.as_ref().expect("just appended").clone();

    // GQA: repeat KV heads if needed.
    let repeats = a.n_heads / a.n_kv_heads;
    let k_full = repeat_kv(&k_full, repeats)?;
    let v_full = repeat_kv(&v_full, repeats)?;

    // [B, H, S_q, D] / [B, H, S_k, D] for matmul.
    let xq = xq.transpose(1, 2)?.contiguous()?;
    let k = k_full.transpose(1, 2)?.contiguous()?;
    let v = v_full.transpose(1, 2)?.contiguous()?;

    let scale = 1.0 / (a.head_dim as f64).sqrt();
    let xq_scaled = (xq * scale)?;
    let scores = xq_scaled.matmul(&k.transpose(D::Minus2, D::Minus1)?.contiguous()?)?;

    let scores = if let Some(mask) = causal_mask {
        scores.broadcast_add(mask)?
    } else {
        scores
    };
    let attn = candle_nn::ops::softmax_last_dim(&scores)?;
    let out = attn.matmul(&v)?; // [B, H, S, D]
    let out = out
        .transpose(1, 2)?
        .contiguous()?
        .reshape((b, s, a.n_heads * a.head_dim))?;
    Ok(a.wo.forward(&out)?)
}

fn repeat_kv(t: &Tensor, repeats: usize) -> Result<Tensor> {
    if repeats == 1 {
        return Ok(t.clone());
    }
    let (b, s, n_kv, d) = t.dims4()?;
    Ok(t.unsqueeze(3)?
        .expand(&[b, s, n_kv, repeats, d])?
        .reshape((b, s, n_kv * repeats, d))?)
}

fn ffn_forward(b: &super::model::ArLlmFeedForward, x: &Tensor) -> Result<Tensor> {
    let gate = candle_nn::ops::silu(&b.w1.forward(x)?)?;
    let up = b.w3.forward(x)?;
    Ok(b.w2.forward(&(gate * up)?)?)
}

fn block_forward(
    block: &ArLlmBlock,
    x: &Tensor,
    cache: &mut LayerKvCache,
    cos: &Tensor,
    sin: &Tensor,
    causal_mask: Option<&Tensor>,
) -> Result<Tensor> {
    let r = attention_forward(
        &block.attention,
        &block.attention_norm.forward(x)?,
        cache,
        cos,
        sin,
        causal_mask,
    )?;
    let h = (x + r)?;
    let r = ffn_forward(&block.feed_forward, &block.ffn_norm.forward(&h)?)?;
    Ok((h + r)?)
}

/// Build a `[1, 1, S_q, S_k]` attention bias matrix that masks
/// attention to future positions. `q_pos[i]` (the `i`-th query) can
/// attend to `k_pos[j]` iff `k_pos[j] <= q_pos[i]`. Positions:
/// `q_pos = pos_start..pos_start + s_q` and `k_pos = 0..pos_start + s_q`.
fn build_causal_mask(
    pos_start: usize,
    s_q: usize,
    s_k: usize,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let mut data: Vec<f32> = Vec::with_capacity(s_q * s_k);
    for i in 0..s_q {
        let q_pos = pos_start + i;
        for j in 0..s_k {
            let k_pos = j;
            data.push(if k_pos > q_pos { f32::NEG_INFINITY } else { 0.0 });
        }
    }
    let mask = Tensor::from_vec(data, (s_q, s_k), device)?
        .to_dtype(dtype)?
        .unsqueeze(0)?
        .unsqueeze(0)?; // [1, 1, S_q, S_k]
    Ok(mask)
}

impl ArLlmModel {
    /// Run the forward pass on `input_embeds` `[B, S, D]`. Mutates
    /// `kv_cache` in place, advancing its `pos` by `S`. Returns the
    /// `[B, S, D]` hidden states from the final RmsNorm.
    pub fn forward_embeds(
        &self,
        input_embeds: &Tensor,
        rope: &RopeCache,
        kv_cache: &mut ArLlmKvCache,
    ) -> Result<Tensor> {
        if kv_cache.layers.len() != self.layers.len() {
            return Err(anyhow!(
                "kv_cache has {} layers, model has {}",
                kv_cache.layers.len(),
                self.layers.len()
            ));
        }
        let (_b, s, _d) = input_embeds.dims3()?;
        let pos_start = kv_cache.pos;

        let (cos, sin) = rope.slice(pos_start, s)?;
        let s_k = pos_start + s; // total key length after this call
        let causal_mask = if s > 1 || s_k > 1 {
            Some(build_causal_mask(
                pos_start,
                s,
                s_k,
                &self.device,
                self.dtype,
            )?)
        } else {
            None
        };

        let mut h = input_embeds.clone();
        for (i, block) in self.layers.iter().enumerate() {
            h = block_forward(
                block,
                &h,
                &mut kv_cache.layers[i],
                &cos,
                &sin,
                causal_mask.as_ref(),
            )?;
        }
        let h = self.norm.forward(&h)?;

        kv_cache.pos += s;
        Ok(h)
    }

    /// Project `hidden [B, S, D]` to vocab logits `[B, S, vocab_size]`
    /// using the tied tok_embeddings as the projection weight.
    pub fn lm_head(&self, hidden: &Tensor) -> Result<Tensor> {
        let w = self.tok_embeddings.embeddings(); // [vocab_size, dim]
        // logits = hidden @ w.T → [B, S, vocab_size]
        let w_t = w.t()?.contiguous()?;
        Ok(hidden.broadcast_matmul(&w_t)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_cache_first_position_is_identity() {
        // At pos=0: cos=1, sin=0, so RoPE is the identity.
        let device = Device::Cpu;
        let dtype = DType::F32;
        let head_dim = 8;
        let half = head_dim / 2;
        let cache = RopeCache::new(4, head_dim, 10000.0, &device, dtype).unwrap();
        assert_eq!(cache.cos.dims(), &[4, half]);
        assert_eq!(cache.sin.dims(), &[4, half]);
        let cos_v: Vec<f32> = cache.cos.flatten_all().unwrap().to_vec1().unwrap();
        let sin_v: Vec<f32> = cache.sin.flatten_all().unwrap().to_vec1().unwrap();
        // First row (pos=0) should be all 1s in cos, all 0s in sin.
        for i in 0..half {
            assert!((cos_v[i] - 1.0).abs() < 1e-7);
            assert!(sin_v[i].abs() < 1e-7);
        }
    }

    #[test]
    fn kv_cache_grows_on_append() {
        let device = Device::Cpu;
        let dtype = DType::F32;
        let mut cache = LayerKvCache::empty();
        let k1 = Tensor::zeros((1, 3, 2, 4), dtype, &device).unwrap();
        let v1 = Tensor::zeros((1, 3, 2, 4), dtype, &device).unwrap();
        cache.append(&k1, &v1).unwrap();
        assert_eq!(cache.seq_len(), 3);
        let k2 = Tensor::zeros((1, 5, 2, 4), dtype, &device).unwrap();
        let v2 = Tensor::zeros((1, 5, 2, 4), dtype, &device).unwrap();
        cache.append(&k2, &v2).unwrap();
        assert_eq!(cache.seq_len(), 8);
        assert_eq!(cache.k.as_ref().unwrap().dims(), &[1, 8, 2, 4]);
    }

    fn checkpoint_path() -> std::path::PathBuf {
        std::env::var_os("STARLING_TTS_SAFETENSORS")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                std::path::PathBuf::from(
                    "/home/gjovanov/gjovanov/starling/models/cache/tts/consolidated.safetensors",
                )
            })
    }

    fn params_path() -> std::path::PathBuf {
        std::env::var_os("STARLING_TTS_PARAMS")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                std::path::PathBuf::from(
                    "/home/gjovanov/gjovanov/starling/models/cache/tts/params.json",
                )
            })
    }

    /// End-to-end smoke test against the real checkpoint. Confirms the
    /// AR LLM forward path runs without crashing on Voxtral-4B-TTS-2603
    /// weights, that the KV cache grows correctly across calls, and
    /// that the lm_head projection produces sane logit shapes.
    /// Bit-exactness against the upstream is Phase 2-F.4 territory
    /// (needs captured llm_hidden golden refs).
    #[test]
    fn ar_llm_forward_real_weights_smoke() {
        let ckpt = checkpoint_path();
        let params = params_path();
        if !ckpt.exists() || !params.exists() {
            eprintln!("skipping: real-weights smoke test (checkpoint absent)");
            return;
        }
        let args = ArLlmArgs::from_params_json_path(&params).unwrap();
        let device = Device::Cpu;
        let dtype = DType::F32;
        let vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(&[&ckpt], dtype, &device).unwrap()
        };
        let model = ArLlmModel::load(vb, args.clone(), &device, dtype).unwrap();
        let rope = RopeCache::new(64, args.head_dim, args.rope_theta, &device, dtype).unwrap();

        let mut cache = ArLlmKvCache::new(args.n_layers);

        // Prefill: small token sequence (5 tokens, BOS-like trivial input).
        let token_ids = Tensor::from_vec(vec![1u32, 24, 25, 26, 27], (1, 5), &device).unwrap();
        let embeds = model.tok_embeddings.forward(&token_ids).unwrap();
        assert_eq!(embeds.dims(), &[1, 5, args.dim]);

        let h_prefill = model.forward_embeds(&embeds, &rope, &mut cache).unwrap();
        assert_eq!(h_prefill.dims(), &[1, 5, args.dim]);
        assert_eq!(cache.pos, 5);
        assert_eq!(cache.layers[0].seq_len(), 5);
        assert_eq!(cache.layers[args.n_layers - 1].seq_len(), 5);

        // Decode: one more token, cache should grow to 6.
        let next_id = Tensor::from_vec(vec![42u32], (1, 1), &device).unwrap();
        let next_emb = model.tok_embeddings.forward(&next_id).unwrap();
        let h_step = model.forward_embeds(&next_emb, &rope, &mut cache).unwrap();
        assert_eq!(h_step.dims(), &[1, 1, args.dim]);
        assert_eq!(cache.pos, 6);
        assert_eq!(cache.layers[0].seq_len(), 6);

        // lm_head projection.
        let logits = model.lm_head(&h_step).unwrap();
        assert_eq!(logits.dims(), &[1, 1, args.vocab_size]);
    }

    #[test]
    fn causal_mask_correctness() {
        // pos_start=2, s_q=3, s_k=5 → query positions 2,3,4; key positions 0..5.
        // q=2 attends to k∈{0,1,2}
        // q=3 attends to k∈{0,1,2,3}
        // q=4 attends to k∈{0,1,2,3,4}
        let device = Device::Cpu;
        let dtype = DType::F32;
        let mask = build_causal_mask(2, 3, 5, &device, dtype).unwrap();
        let v: Vec<f32> = mask.flatten_all().unwrap().to_vec1().unwrap();
        let inf = f32::NEG_INFINITY;
        let expected = [
            // q=2
            0.0, 0.0, 0.0, inf, inf,
            // q=3
            0.0, 0.0, 0.0, 0.0, inf,
            // q=4
            0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        for (i, e) in expected.iter().enumerate() {
            if e.is_infinite() {
                assert!(v[i].is_infinite() && v[i].is_sign_negative(), "[{i}] {} not -inf", v[i]);
            } else {
                assert!((v[i] - e).abs() < 1e-7, "[{i}] {} vs {}", v[i], e);
            }
        }
    }
}
