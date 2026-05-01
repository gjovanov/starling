//! Codec attention with ALiBi + qk_norm + sliding-window.
//!
//! Differences vs the FMA attention:
//! - **No RoPE.** Position is encoded via ALiBi: a per-head bias added
//!   to the attention scores, `slope[h] * (j - i)`, with geometric
//!   slopes `r^i` for `r = 2^(-8/H)`.
//! - **qk_norm.** RmsNorm is applied to the full `(B, T, n_heads*head_dim)`
//!   tensor BEFORE viewing into heads. Same eps `qk_norm_eps=1e-6`.
//! - **Sliding window.** Only positions `j ∈ [i - W, i]` participate
//!   (causal + window cap). `W` is per-block: 16 / 32 / 64 / 128 across
//!   the four decoder transformer blocks (doubles on each ×2 upsample).
//! - **GQA**: the codec ships with `n_heads = n_kv_heads = 8`, so no
//!   actual KV repetition — the path is preserved for parity with the
//!   FMA implementation.
//!
//! Reference: `voxtral_tts_audio_tokenizer.Attention`.

use anyhow::{anyhow, Result};
use candle_core::{DType, Device, IndexOp, Module, Tensor, D};
use candle_nn::{Linear, RmsNorm, VarBuilder};

use super::args::AudioTokenizerArgs;

/// ALiBi slopes for `n_heads` attention heads.
/// `slope[h] = 2^(-8h/n_heads)` when `n_heads` is a power of 2;
/// otherwise the upstream's mixed-power scheme (largest power-of-2
/// prefix + interleaved doubling). For `n_heads=8` this is just the
/// power-of-2 path: `[1, 2^-1, 2^-2, ..., 2^-7]`.
pub fn alibi_slopes(n_heads: usize) -> Vec<f32> {
    fn pow2_slopes(n: usize) -> Vec<f32> {
        let r = 2.0_f32.powf(-8.0 / n as f32);
        (0..n).map(|i| r.powi(i as i32)).collect()
    }
    if n_heads.is_power_of_two() {
        pow2_slopes(n_heads)
    } else {
        let m = 1usize << (usize::BITS as usize - 1 - n_heads.leading_zeros() as usize);
        let mut out = pow2_slopes(m);
        let extra = pow2_slopes(2 * m);
        // Take every other element (`extra[::2]`), starting from index 0.
        for i in 0..(n_heads - m) {
            out.push(extra[2 * i]);
        }
        out
    }
}

/// Build the [H, T, T] attention bias used by the codec attention:
/// `bias[h, i, j] = slope[h] * (j - i)` for positions with
/// `i - sliding_window <= j <= i` (causal + window), `-inf` elsewhere.
///
/// Returned shape: `[1, H, T, T]` so it broadcasts over the batch.
fn build_attention_bias(
    slopes: &Tensor,
    seqlen: usize,
    sliding_window: usize,
    causal: bool,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let positions: Vec<f32> = (0..seqlen).map(|i| i as f32).collect();
    let pos_t = Tensor::from_vec(positions, seqlen, device)?;
    let pos_i = pos_t.unsqueeze(1)?; // [T, 1]
    let pos_j = pos_t.unsqueeze(0)?; // [1, T]
    let rel_pos = pos_j.broadcast_sub(&pos_i)?; // [T, T]
    let rel_pos = rel_pos.to_dtype(dtype)?;

    // ALiBi bias: slopes [H] × rel_pos [T, T] → [H, T, T]
    let slopes = slopes.to_dtype(dtype)?.unsqueeze(1)?.unsqueeze(2)?; // [H, 1, 1]
    let mut bias = slopes.broadcast_mul(&rel_pos.unsqueeze(0)?)?; // [H, T, T]

    // Build the in-window mask (1 where allowed, 0 otherwise) on f32.
    let window_left = sliding_window as f32;
    let window_right = if causal { 0.0 } else { sliding_window as f32 };
    let mut mask: Vec<f32> = Vec::with_capacity(seqlen * seqlen);
    for i in 0..seqlen {
        for j in 0..seqlen {
            let d = j as f32 - i as f32;
            // Future-mask when causal.
            let causal_ok = if causal { d <= 0.0 } else { true };
            let window_ok = d >= -window_left && d <= window_right;
            mask.push(if causal_ok && window_ok { 0.0 } else { f32::NEG_INFINITY });
        }
    }
    let mask_t = Tensor::from_vec(mask, (seqlen, seqlen), device)?
        .to_dtype(dtype)?
        .unsqueeze(0)?; // [1, T, T]
    bias = bias.broadcast_add(&mask_t)?;

    // Add batch dim → [1, H, T, T].
    Ok(bias.unsqueeze(0)?)
}

pub struct CodecAttention {
    wq: Linear,
    wk: Linear,
    wv: Linear,
    wo: Linear,
    q_norm: Option<RmsNorm>,
    k_norm: Option<RmsNorm>,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    sliding_window: usize,
    causal: bool,
    /// Cached `[n_heads]` slopes — pre-cast to the model's dtype.
    slopes: Tensor,
}

impl CodecAttention {
    pub fn load(
        vb: VarBuilder,
        args: &AudioTokenizerArgs,
        sliding_window: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        if args.n_heads % args.n_kv_heads != 0 {
            return Err(anyhow!(
                "n_heads ({}) must be a multiple of n_kv_heads ({})",
                args.n_heads,
                args.n_kv_heads
            ));
        }
        let q_out = args.n_heads * args.head_dim;
        let kv_out = args.n_kv_heads * args.head_dim;
        let wq = linear(args.dim, q_out, false, vb.pp("wq"))?;
        let wk = linear(args.dim, kv_out, false, vb.pp("wk"))?;
        let wv = linear(args.dim, kv_out, false, vb.pp("wv"))?;
        let wo = linear(q_out, args.dim, args.use_biases, vb.pp("wo"))?;

        let (q_norm, k_norm) = if args.qk_norm {
            (
                Some(rms_norm(q_out, args.qk_norm_eps, vb.pp("q_norm"))?),
                Some(rms_norm(kv_out, args.qk_norm_eps, vb.pp("k_norm"))?),
            )
        } else {
            (None, None)
        };

        let slopes_vec = alibi_slopes(args.n_heads);
        let slopes = Tensor::from_vec(slopes_vec, args.n_heads, device)?.to_dtype(dtype)?;

        Ok(Self {
            wq,
            wk,
            wv,
            wo,
            q_norm,
            k_norm,
            n_heads: args.n_heads,
            n_kv_heads: args.n_kv_heads,
            head_dim: args.head_dim,
            sliding_window,
            causal: args.causal,
            slopes,
        })
    }

    /// `x` shape `[B, T, D]`. Returns `[B, T, D]`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;

        let mut xq = self.wq.forward(x)?;
        let mut xk = self.wk.forward(x)?;
        let xv = self.wv.forward(x)?;

        if let (Some(qn), Some(kn)) = (&self.q_norm, &self.k_norm) {
            xq = qn.forward(&xq)?;
            xk = kn.forward(&xk)?;
        }

        let xq = xq.reshape((b, t, self.n_heads, self.head_dim))?;
        let xk = xk.reshape((b, t, self.n_kv_heads, self.head_dim))?;
        let xv = xv.reshape((b, t, self.n_kv_heads, self.head_dim))?;

        // Repeat KV along the heads dim if GQA. Voxtral codec has equal
        // n_heads / n_kv_heads, so this is a no-op in practice.
        let repeats = self.n_heads / self.n_kv_heads;
        let xk = repeat_kv(&xk, repeats)?;
        let xv = repeat_kv(&xv, repeats)?;

        // [B, H, T, D] for matmul.
        let xq = xq.transpose(1, 2)?.contiguous()?;
        let xk = xk.transpose(1, 2)?.contiguous()?;
        let xv = xv.transpose(1, 2)?.contiguous()?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let xq_scaled = (xq * scale)?;
        let scores = xq_scaled.matmul(&xk.transpose(D::Minus2, D::Minus1)?.contiguous()?)?;

        let bias = build_attention_bias(
            &self.slopes,
            t,
            self.sliding_window,
            self.causal,
            x.device(),
            x.dtype(),
        )?;
        let scores = scores.broadcast_add(&bias)?;
        let attn = candle_nn::ops::softmax_last_dim(&scores)?;
        let out = attn.matmul(&xv)?; // [B, H, T, D]
        let out = out
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, t, self.n_heads * self.head_dim))?;
        Ok(self.wo.forward(&out)?)
    }
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

    #[test]
    fn alibi_slopes_power_of_two() {
        let s = alibi_slopes(8);
        // r = 2^(-8/8) = 0.5; slopes = [1, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125]
        let expected = [
            1.0_f32, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125,
        ];
        for i in 0..8 {
            assert!(
                (s[i] - expected[i]).abs() < 1e-7,
                "slope {i}: {} vs {}",
                s[i],
                expected[i]
            );
        }
    }

    #[test]
    fn alibi_slopes_non_power_of_two() {
        // For n=6: largest pow2 prefix m=4 → slopes_pow2(4) = [1, r^1, r^2, r^3] with r=2^(-8/4)=2^-2=0.25
        // Then take 2 from slopes_pow2(8)[::2][:2] = [1, 2^(-2)] = [1.0, 0.25]
        // Result: [1, 0.25, 0.0625, 0.015625, 1.0, 0.25]
        let s = alibi_slopes(6);
        assert_eq!(s.len(), 6);
        // Power-of-2 base part
        assert!((s[0] - 1.0).abs() < 1e-7);
        assert!((s[1] - 0.25).abs() < 1e-7);
        assert!((s[2] - 0.0625).abs() < 1e-7);
        assert!((s[3] - 0.015625).abs() < 1e-7);
        // Tail from slopes_pow2(8)[::2]
        assert!((s[4] - 1.0).abs() < 1e-7);
        assert!((s[5] - 0.25).abs() < 1e-7);
    }

    #[test]
    fn attention_bias_window_mask() {
        let device = Device::Cpu;
        let dtype = DType::F32;
        let slopes_vec = alibi_slopes(2);
        let slopes = Tensor::from_vec(slopes_vec, 2, &device).unwrap().to_dtype(dtype).unwrap();
        // T=4, window=1 (one back), causal
        let bias = build_attention_bias(&slopes, 4, 1, true, &device, dtype).unwrap();
        // bias [1, H=2, T=4, T=4]
        assert_eq!(bias.dims(), &[1, 2, 4, 4]);
        let v: Vec<f32> = bias.flatten_all().unwrap().to_vec1().unwrap();
        // For head 0 (slope=1.0), causal+window=1:
        //   row i, col j: allowed if j ∈ [i-1, i].
        //   bias = 1.0 * (j - i) at allowed pos, -inf otherwise.
        //   (0,0)=0; (0,1..3)=-inf
        //   (1,0)=-1; (1,1)=0; (1,2..3)=-inf
        //   (2,0)=-inf; (2,1)=-1; (2,2)=0; (2,3)=-inf
        //   (3,0)=-inf; (3,1)=-inf; (3,2)=-1; (3,3)=0
        let inf = f32::NEG_INFINITY;
        let expected_h0: Vec<f32> = vec![
            0.0, inf, inf, inf,
            -1.0, 0.0, inf, inf,
            inf, -1.0, 0.0, inf,
            inf, inf, -1.0, 0.0,
        ];
        for (i, e) in expected_h0.iter().enumerate() {
            let got = v[i];
            if e.is_infinite() {
                assert!(got.is_infinite() && got.is_sign_negative(), "[h0,{i}] expected -inf, got {got}");
            } else {
                assert!((got - e).abs() < 1e-7, "[h0,{i}] {got} vs {e}");
            }
        }
    }
}
