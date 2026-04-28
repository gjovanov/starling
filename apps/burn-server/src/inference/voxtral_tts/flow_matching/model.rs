//! Candle implementation of the flow-matching acoustic transformer.
//!
//! Mirrors `vllm_omni.model_executor.models.voxtral_tts
//! .voxtral_tts_audio_generation.FlowMatchingAudioTransformer` exactly,
//! including:
//!
//! - Sinusoidal time embedding (concatenated cos/sin halves, **not**
//!   interleaved as in the standard Vaswani-style impl).
//! - Bidirectional GQA attention with no RoPE, no mask.
//! - SwiGLU FFN.
//! - 3-layer pre-norm transformer block.
//! - 8-step Euler integration with Classifier-Free Guidance (α=1.2).
//! - Scaled-uniform quantisation `(x+1)/2 * 20 → round → long` and
//!   `+len(special_tokens)` offset on the final acoustic-codebook output.
//!
//! Phase 2-C scope: bit-mostly-exact match against the
//! `apps/burn-server/test_data/tts_golden/fma_*.npz` fixtures captured
//! in Phase 2-A. Tolerance: 1e-3 abs for BF16, 5e-5 abs for F32.

use anyhow::{anyhow, Result};
use candle_core::{DType, Device, IndexOp, Module, Tensor, D};
use candle_nn::{Linear, RmsNorm, VarBuilder};

use super::args::{
    AcousticTransformerArgs, AudioSpecialTokens, FlowMatchingDecodeArgs, MultimodalAudioModelArgs,
};

/// Sinusoidal time embedding. Maps a time step `t ∈ [0, 1]` to a
/// `dim`-vector by concatenating `cos(t · inv_freq)` and `sin(t · inv_freq)`,
/// where `inv_freq[k] = θ^(-k / (dim/2))`.
///
/// **Not** the same as the conventional sinusoidal-position embedding —
/// the upstream concatenates the cos and sin halves rather than
/// interleaving them.
pub struct TimeEmbedding {
    /// Pre-computed `[dim/2]` of `θ^(-k / half)` for `k ∈ [0, half)`.
    inv_freq: Tensor,
}

impl TimeEmbedding {
    pub fn new(dim: usize, theta: f64, device: &Device, dtype: DType) -> Result<Self> {
        let half = dim / 2;
        let arange: Vec<f32> = (0..half).map(|k| k as f32).collect();
        let arange = Tensor::from_vec(arange, half, device)?;
        let log_theta = -(theta.ln()) as f32;
        let scale = log_theta / (half as f32);
        let inv_freq = (arange * scale as f64)?.exp()?;
        let inv_freq = inv_freq.to_dtype(dtype)?;
        Ok(Self { inv_freq })
    }

    /// `t` shape: `[B, 1]` (the upstream calls with `t.view(-1, 1).repeat(B, 1)`).
    /// Returns `[B, dim]`.
    pub fn forward(&self, t: &Tensor) -> Result<Tensor> {
        let inv = self.inv_freq.unsqueeze(0)?; // [1, dim/2]
        let emb = t.broadcast_mul(&inv)?; // [B, dim/2]
        let cos = emb.cos()?;
        let sin = emb.sin()?;
        Ok(Tensor::cat(&[&cos, &sin], 1)?)
    }
}

/// Bidirectional Group-Query Attention with no positional encoding.
///
/// In the velocity-field network the input has only 3 positions —
/// `[x_emb, t_emb, llm_proj]` — so attention is dense, no mask, no
/// RoPE. GQA expands `n_kv_heads` k/v projections to `n_heads` query
/// heads via `repeat_kv`.
pub struct BidirectionalAttention {
    wq: Linear,
    wk: Linear,
    wv: Linear,
    wo: Linear,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    repeats: usize,
    /// `1 / sqrt(head_dim)` baked in.
    softmax_scale: f64,
}

impl BidirectionalAttention {
    pub fn load(vb: VarBuilder, args: &AcousticTransformerArgs) -> Result<Self> {
        let qkv_in = args.dim;
        let q_out = args.n_heads * args.head_dim;
        let kv_out = args.n_kv_heads * args.head_dim;
        // Upstream sets bias differently per linear: wq/wv/wo follow `use_biases`,
        // wk is always bias-less. For Voxtral-4B-TTS `use_biases=false`, so they're
        // all unbiased — but we mirror the schema precisely.
        let wq = linear(qkv_in, q_out, args.use_biases, vb.pp("wq"))?;
        let wk = linear(qkv_in, kv_out, false, vb.pp("wk"))?;
        let wv = linear(qkv_in, kv_out, args.use_biases, vb.pp("wv"))?;
        let wo = linear(q_out, args.dim, args.use_biases, vb.pp("wo"))?;
        if args.n_heads % args.n_kv_heads != 0 {
            return Err(anyhow!(
                "n_heads ({}) must be a multiple of n_kv_heads ({})",
                args.n_heads,
                args.n_kv_heads
            ));
        }
        Ok(Self {
            wq,
            wk,
            wv,
            wo,
            n_heads: args.n_heads,
            n_kv_heads: args.n_kv_heads,
            head_dim: args.head_dim,
            repeats: args.n_heads / args.n_kv_heads,
            softmax_scale: 1.0 / (args.head_dim as f64).sqrt(),
        })
    }

    /// `x` shape `[B, S, D]`. Returns `[B, S, D]`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        let xq = self
            .wq
            .forward(x)?
            .reshape((b, s, self.n_heads, self.head_dim))?;
        let xk = self
            .wk
            .forward(x)?
            .reshape((b, s, self.n_kv_heads, self.head_dim))?;
        let xv = self
            .wv
            .forward(x)?
            .reshape((b, s, self.n_kv_heads, self.head_dim))?;

        let xk = repeat_kv(&xk, self.repeats)?;
        let xv = repeat_kv(&xv, self.repeats)?;

        // [B, n_heads, S, head_dim] for matmul-friendly layout.
        let xq = xq.transpose(1, 2)?.contiguous()?;
        let xk = xk.transpose(1, 2)?.contiguous()?;
        let xv = xv.transpose(1, 2)?.contiguous()?;

        // Native attention. The upstream `_native_attention` does
        // `query *= scale`, then `softmax(query @ key.T)`. We bake the
        // scale into the query before the matmul to match.
        let xq_scaled = (xq * self.softmax_scale)?;
        let attn = xq_scaled.matmul(&xk.transpose(D::Minus2, D::Minus1)?.contiguous()?)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let out = attn.matmul(&xv)?; // [B, n_heads, S, head_dim]
        let out = out
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, s, self.n_heads * self.head_dim))?;
        Ok(self.wo.forward(&out)?)
    }
}

/// SwiGLU feed-forward: `w2(silu(w1(x)) * w3(x))`.
pub struct FeedForward {
    w1: Linear,
    w2: Linear,
    w3: Linear,
}

impl FeedForward {
    pub fn load(vb: VarBuilder, args: &AcousticTransformerArgs) -> Result<Self> {
        let dim = args.dim;
        let hidden = args.hidden_dim;
        // Upstream: w1/w3 always bias-less, w2 follows use_biases.
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

/// One block: pre-norm attention residual, pre-norm FFN residual.
pub struct AcousticTransformerBlock {
    attention: BidirectionalAttention,
    feed_forward: FeedForward,
    attention_norm: RmsNorm,
    ffn_norm: RmsNorm,
}

impl AcousticTransformerBlock {
    pub fn load(vb: VarBuilder, args: &AcousticTransformerArgs) -> Result<Self> {
        let attention = BidirectionalAttention::load(vb.pp("attention"), args)?;
        let feed_forward = FeedForward::load(vb.pp("feed_forward"), args)?;
        let attention_norm = rms_norm(args.dim, args.norm_eps, vb.pp("attention_norm"))?;
        let ffn_norm = rms_norm(args.dim, args.norm_eps, vb.pp("ffn_norm"))?;
        Ok(Self {
            attention,
            feed_forward,
            attention_norm,
            ffn_norm,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let r = self.attention.forward(&self.attention_norm.forward(x)?)?;
        let h = (x + r)?;
        let r = self.feed_forward.forward(&self.ffn_norm.forward(&h)?)?;
        Ok((h + r)?)
    }
}

/// Full flow-matching velocity-field network.
pub struct FlowMatchingAudioTransformer {
    args: MultimodalAudioModelArgs,
    decode_args: FlowMatchingDecodeArgs,

    // Embedding layers.
    time_embedding: TimeEmbedding,
    input_projection: Linear,  // [3072, 36]
    time_projection: Linear,   // [3072, 3072]
    llm_projection: Linear,    // [3072, 3072]

    // Output projections.
    semantic_codebook_output: Linear, // [3072, 8320]
    acoustic_codebook_output: Linear, // [3072, 36]

    // 3-layer transformer + final norm.
    layers: Vec<AcousticTransformerBlock>,
    norm: RmsNorm,

    /// Pre-computed `linspace(0, 1, decode_iters)` cached at the model's dtype.
    timesteps: Tensor,

    device: Device,
    dtype: DType,
}

impl FlowMatchingAudioTransformer {
    pub fn load(
        vb: VarBuilder,
        args: MultimodalAudioModelArgs,
        decode_args: FlowMatchingDecodeArgs,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let at = &args.acoustic_transformer_args;

        let time_embedding = TimeEmbedding::new(at.dim, at.rope_theta.max(1.0), device, dtype)?;

        let input_projection = linear(args.n_acoustic_codebook, at.dim, false, vb.pp("input_projection"))?;
        let time_projection = linear(at.dim, at.dim, false, vb.pp("time_projection"))?;
        let llm_projection = linear(at.input_dim, at.dim, false, vb.pp("llm_projection"))?;

        let semantic_codebook_output = linear(
            at.dim,
            args.semantic_output_dim(),
            at.use_biases,
            vb.pp("semantic_codebook_output"),
        )?;
        let acoustic_codebook_output = linear(
            at.dim,
            args.n_acoustic_codebook,
            false,
            vb.pp("acoustic_codebook_output"),
        )?;

        // Upstream uses `nn.ModuleDict` with string keys "0", "1", ... ;
        // safetensors flattens that to `layers.0.*`, `layers.1.*`, ...
        let mut layers = Vec::with_capacity(at.n_layers);
        for i in 0..at.n_layers {
            let block = AcousticTransformerBlock::load(vb.pp(&format!("layers.{i}")), at)?;
            layers.push(block);
        }
        let norm = rms_norm(at.dim, at.norm_eps, vb.pp("norm"))?;

        // Note: upstream uses dtype= sigma_max-typed timesteps but in
        // practice they're cast to llm_hidden.dtype on first use. We
        // pre-cast to `dtype` for cheap inner-loop access.
        let timesteps = Tensor::arange(0u32, decode_args.decode_iters as u32, device)?
            .to_dtype(DType::F32)?;
        let timesteps = (timesteps / ((decode_args.decode_iters - 1) as f64))?;
        let timesteps = timesteps.to_dtype(dtype)?;

        Ok(Self {
            args,
            decode_args,
            time_embedding,
            input_projection,
            time_projection,
            llm_projection,
            semantic_codebook_output,
            acoustic_codebook_output,
            layers,
            norm,
            timesteps,
            device: device.clone(),
            dtype,
        })
    }

    pub fn args(&self) -> &MultimodalAudioModelArgs {
        &self.args
    }

    pub fn decode_args(&self) -> &FlowMatchingDecodeArgs {
        &self.decode_args
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn timesteps(&self) -> &Tensor {
        &self.timesteps
    }

    /// `_predict_velocity` — runs the velocity field at one Euler step,
    /// returning `[B, n_acoustic_codebook]` velocity.
    ///
    /// `x_t`: `[B, n_acoustic_codebook]`
    /// `llm_output`: `[B, dim]` (zero for the unconditioned branch)
    /// `t_emb`: `[B, dim]` (the sinusoidal time embedding pre-projection)
    pub fn predict_velocity(
        &self,
        x_t: &Tensor,
        llm_output: &Tensor,
        t_emb: &Tensor,
    ) -> Result<Tensor> {
        let x_t = x_t.to_dtype(llm_output.dtype())?;

        let t_emb = self.time_projection.forward(t_emb)?;
        let llm_output = self.llm_projection.forward(llm_output)?;

        let x_emb = self.input_projection.forward(&x_t.unsqueeze(1)?)?; // [B, 1, dim]
        let t_emb = t_emb.unsqueeze(1)?; // [B, 1, dim]
        let llm_output = llm_output.unsqueeze(1)?; // [B, 1, dim]
        let inputs = Tensor::cat(&[&x_emb, &t_emb, &llm_output], 1)?; // [B, 3, dim]

        let mut h = inputs;
        for layer in &self.layers {
            h = layer.forward(&h)?;
        }
        let h = self.norm.forward(&h)?;
        // Upstream: `final_hidden.view(-1, S, D)` — already that shape.
        // `acoustic_codebook_output(final_hidden[:, 0, :])`.
        // CUDA matmul requires contiguous; `i((.., 0, ..))` is strided.
        let pos0 = h.i((.., 0, ..))?.contiguous()?;
        Ok(self.acoustic_codebook_output.forward(&pos0)?)
    }

    /// Drive the 7-step Euler integration with Classifier-Free Guidance.
    ///
    /// `semantic_code`: `[B]` — used for the `END_AUDIO` short-circuit mask.
    /// `llm_hidden`: `[B, dim]` — AR LLM output.
    /// `x_0`: `[B, n_acoustic_codebook]` — initial noise sample.
    ///
    /// Returns offset acoustic codes `[B, n_acoustic_codebook]`,
    /// matching the upstream `output_codes + len(AudioSpecialTokens)`.
    pub fn decode_one_frame(
        &self,
        semantic_code: &Tensor,
        llm_hidden: &Tensor,
        x_0: &Tensor,
    ) -> Result<Tensor> {
        let b = semantic_code.dim(0)?;
        let n_acoustic = self.args.n_acoustic_codebook;

        // should_decode[b] = (semantic_code[b] != END_AUDIO)
        let end_audio_mask = semantic_code.eq(AudioSpecialTokens::END_AUDIO as u32)?;
        // We'll apply this mask post-hoc when scattering the special-token id.

        let llm_hidden_zero = llm_hidden.zeros_like()?;
        let dtype = llm_hidden.dtype();

        let mut sampled = (x_0.to_dtype(dtype)? * self.decode_args.noise_scale)?;
        let timesteps = self.timesteps.to_dtype(dtype)?;

        for i in 0..(self.decode_args.decode_iters - 1) {
            let t = timesteps.i(i)?; // scalar
            let t_next = timesteps.i(i + 1)?;
            let dt = (t_next - &t)?;

            // Build [B, 1] time tensor and its sinusoidal embedding.
            let t_b1 = t.reshape((1, 1))?.broadcast_as((b, 1))?.to_dtype(dtype)?;
            let t_emb = self.time_embedding.forward(&t_b1)?;

            // Batch cond + uncond in the heads-dim → 2·B batch.
            let x_batched = Tensor::cat(&[&sampled, &sampled], 0)?;
            let llm_batched = Tensor::cat(&[llm_hidden, &llm_hidden_zero], 0)?;
            let t_emb_batched = Tensor::cat(&[&t_emb, &t_emb], 0)?;

            let v_all = self.predict_velocity(&x_batched, &llm_batched, &t_emb_batched)?;
            let v_cond = v_all.i(..b)?;
            let v_uncond = v_all.i(b..)?;
            let alpha = self.decode_args.cfg_alpha;
            let v_t = ((v_cond * alpha)?.add(&(v_uncond * (1.0 - alpha))?))?;
            sampled = sampled.add(&v_t.broadcast_mul(&dt)?)?;
        }

        // Quantize: clamp [-1, 1], scale to [0, levels-1], round to long.
        let levels_minus_one = (self.args.acoustic_codebook_size - 1) as f64;
        let sampled = sampled.clamp(-1.0_f64, 1.0_f64)?;
        let scaled = (((sampled + 1.0_f64)? * 0.5_f64)? * levels_minus_one)?;
        // candle has no `round`, but quantising-toward-nearest can be
        // expressed as `(x + 0.5).floor()` for non-negative values
        // (which is true here because scaled ∈ [0, levels-1]).
        let scaled_plus_half = (scaled + 0.5_f64)?;
        let codes_f = scaled_plus_half.floor()?;
        let codes_u = codes_f.to_dtype(DType::U32)?; // [B, n_acoustic]

        // For rows where semantic_code == END_AUDIO, replace with EMPTY_AUDIO.
        // end_audio_mask: [B] u8 -> broadcast along axis 1 to [B, n_acoustic].
        let mask = end_audio_mask
            .to_dtype(DType::U32)?
            .reshape((b, 1))?
            .broadcast_as((b, n_acoustic))?;
        let empty = Tensor::full(AudioSpecialTokens::EMPTY_AUDIO, (b, n_acoustic), &self.device)?;
        let codes_u = mask.where_cond(&empty, &codes_u)?; // mask=1 => empty, else codes_u

        // Apply +len(special_tokens) offset.
        let offset = AudioSpecialTokens::COUNT as u32;
        let offset_t =
            Tensor::full(offset, (b, n_acoustic), &self.device)?;
        Ok((codes_u + offset_t)?)
    }

    /// Top-level forward: `llm_hidden [B, dim]` plus per-frame noise
    /// `x_0 [B, n_acoustic_codebook]` → `audio_codes [B, 1 + n_acoustic_codebook]`.
    ///
    /// The `x_0` argument lets callers feed a captured noise tensor for
    /// bit-exact regression-testing; in production it would be drawn
    /// from a CSPRNG matching `torch.randn`.
    pub fn forward_with_noise(&self, llm_hidden: &Tensor, x_0: &Tensor) -> Result<Tensor> {
        let semantic_logit = self.semantic_codebook_output.forward(llm_hidden)?;
        let semantic_logit = semantic_logit.to_dtype(DType::F32)?;
        let semantic_logit = mask_semantic_logits(
            &semantic_logit,
            AudioSpecialTokens::EMPTY_AUDIO,
            self.args.semantic_codebook_size + AudioSpecialTokens::COUNT,
        )?;
        let semantic_code = semantic_logit.argmax_keepdim(D::Minus1)?; // [B, 1] u32
        let semantic_code_squeezed = semantic_code.squeeze(1)?;
        let acoustic_codes = self.decode_one_frame(&semantic_code_squeezed, llm_hidden, x_0)?;
        let audio_codes = Tensor::cat(&[&semantic_code, &acoustic_codes], 1)?;
        Ok(audio_codes)
    }
}

/// Apply the upstream's two semantic-logit masks before argmax:
/// 1. `[:, EMPTY_AUDIO] = -inf`  → never predict `[EMPTY_AUDIO]` directly.
/// 2. `[:, COUNT + semantic_codebook_size:] = -inf` → zero out
///    output-padding positions (8194..8320 for Voxtral-4B-TTS).
fn mask_semantic_logits(logits: &Tensor, empty_id: u32, hard_cutoff: usize) -> Result<Tensor> {
    use candle_core::IndexOp;

    let (b, d) = logits.dims2()?;
    // Build a mask vector [d] with -inf at positions to mask, 0 elsewhere.
    let mut mask = vec![0.0f32; d];
    mask[empty_id as usize] = f32::NEG_INFINITY;
    for i in hard_cutoff..d {
        mask[i] = f32::NEG_INFINITY;
    }
    let mask_t = Tensor::from_vec(mask, d, logits.device())?
        .to_dtype(logits.dtype())?
        .unsqueeze(0)?
        .broadcast_as((b, d))?;
    Ok(logits.broadcast_add(&mask_t)?)
}

// ---------- helpers ----------

/// Repeat KV heads along axis 2: `[B, S, n_kv, D] -> [B, S, n_kv*r, D]`.
/// Mirrors upstream's `_repeat_interleave`.
fn repeat_kv(t: &Tensor, repeats: usize) -> Result<Tensor> {
    if repeats == 1 {
        return Ok(t.clone());
    }
    let (b, s, n_kv, d) = t.dims4()?;
    let out = t
        .unsqueeze(3)?
        .expand(&[b, s, n_kv, repeats, d])?
        .reshape((b, s, n_kv * repeats, d))?;
    Ok(out)
}

/// Construct a Linear from a VarBuilder. `weight` is `[out_features,
/// in_features]` (PyTorch row-major convention).
fn linear(in_features: usize, out_features: usize, has_bias: bool, vb: VarBuilder) -> Result<Linear> {
    let weight = vb.get((out_features, in_features), "weight")?;
    let bias = if has_bias { Some(vb.get(out_features, "bias")?) } else { None };
    Ok(Linear::new(weight, bias))
}

/// Construct an RmsNorm (parametrized by a per-channel weight vector).
fn rms_norm(size: usize, eps: f64, vb: VarBuilder) -> Result<RmsNorm> {
    let weight = vb.get(size, "weight")?;
    Ok(RmsNorm::new(weight, eps))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu() -> Device {
        Device::Cpu
    }

    #[test]
    fn time_embedding_zero_t_is_constant() {
        let dim = 32;
        let theta = 10000.0;
        let te = TimeEmbedding::new(dim, theta, &cpu(), DType::F32).unwrap();

        let t = Tensor::zeros((4, 1), DType::F32, &cpu()).unwrap();
        let out = te.forward(&t).unwrap();
        assert_eq!(out.dims(), &[4, dim]);

        // cos(0) = 1, sin(0) = 0 → first half should be all 1, second half all 0.
        let v: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        for row in 0..4 {
            for j in 0..dim / 2 {
                assert!((v[row * dim + j] - 1.0).abs() < 1e-5, "cos half not 1");
                assert!(v[row * dim + dim / 2 + j].abs() < 1e-5, "sin half not 0");
            }
        }
    }

    #[test]
    fn time_embedding_matches_upstream_formula() {
        // Manually compute inv_freq for dim=8, theta=10.0, then verify
        // emb at t=0.5.
        let dim = 8;
        let theta = 10.0_f64;
        let half = dim / 2;
        let te = TimeEmbedding::new(dim, theta, &cpu(), DType::F32).unwrap();
        let t = Tensor::from_vec(vec![0.5f32], (1, 1), &cpu()).unwrap();
        let out: Vec<f32> = te.forward(&t).unwrap().flatten_all().unwrap().to_vec1().unwrap();

        // expected[j] = cos(0.5 * theta^(-j/half)) for j in 0..half
        // expected[half+j] = sin(0.5 * theta^(-j/half))
        for j in 0..half {
            let inv = (theta).powf(-(j as f64) / (half as f64));
            let arg = 0.5_f64 * inv;
            let cos_expected = arg.cos() as f32;
            let sin_expected = arg.sin() as f32;
            assert!((out[j] - cos_expected).abs() < 1e-5, "cos[{}] {} vs {}", j, out[j], cos_expected);
            assert!(
                (out[half + j] - sin_expected).abs() < 1e-5,
                "sin[{}] {} vs {}",
                j,
                out[half + j],
                sin_expected
            );
        }
    }

    #[test]
    fn repeat_kv_expand() {
        // shape [B=1, S=2, n_kv=2, D=3] with values [0..12]
        let t = Tensor::from_vec(
            (0..12).map(|i| i as f32).collect::<Vec<_>>(),
            (1, 2, 2, 3),
            &cpu(),
        )
        .unwrap();
        let out = repeat_kv(&t, 2).unwrap();
        assert_eq!(out.dims(), &[1, 2, 4, 3]);
        let v: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        // Each kv-head should appear `repeats` times in a row.
        // [B=0, S=0, kv=0, :] = [0,1,2]   → out[0..3] and out[3..6]
        // [B=0, S=0, kv=1, :] = [3,4,5]   → out[6..9] and out[9..12]
        assert_eq!(&v[0..3], &[0.0, 1.0, 2.0]);
        assert_eq!(&v[3..6], &[0.0, 1.0, 2.0]);
        assert_eq!(&v[6..9], &[3.0, 4.0, 5.0]);
        assert_eq!(&v[9..12], &[3.0, 4.0, 5.0]);
    }
}
