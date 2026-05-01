//! Weight-norm convolutions for the codec decoder.
//!
//! The codec ships its conv weights in PyTorch's `parametrizations`
//! form: a magnitude vector `g` (`original0`, shape `[N, 1, 1]`) and a
//! direction tensor `v` (`original1`, same shape as the underlying
//! conv weight). Materialised weight is `w = g · v / ‖v‖₂` where the
//! norm is taken per-row of the leading axis (`dim=0`).
//!
//! Two conv variants are needed by the decoder:
//! - [`CausalConv1d`]: pads the input on the LEFT with `reflect` mode by
//!   `effective_kernel_size - stride` plus a small `extra_padding` to
//!   make the output length exactly `ceil(n_frames)`. Used for the
//!   first decoder block (292→1024) and the final `output_proj`
//!   (1024→240, kernel 7).
//! - [`CausalConvTranspose1d`]: runs `nn.ConvTranspose1d` then trims
//!   `kernel - stride` samples from the RIGHT (`trim_ratio = 1.0`).
//!   Used for the three ×2 upsample blocks.
//!
//! Reference: `voxtral_tts_audio_tokenizer.CausalConv1d`,
//! `CausalConvTranspose1d`, `pad1d`.

use anyhow::{anyhow, Result};
use candle_core::{IndexOp, Module, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, VarBuilder};

/// Materialise `w = g · v / ‖v‖₂` where the norm is per-row of `v`'s
/// leading axis, matching `torch.nn.utils.parametrizations.weight_norm`
/// with the default `dim=0`.
///
/// Inputs:
/// - `g` (`original0`): broadcastable scalars `[N, 1, 1, ...]`.
/// - `v` (`original1`): full weight, shape `[N, ...]`.
pub fn materialize_weight_norm(g: &Tensor, v: &Tensor) -> Result<Tensor> {
    let n = v.dim(0)?;
    // Flatten everything past axis 0 so we can sum-square along one axis.
    let trailing: usize = v.dims().iter().skip(1).product();
    let v_flat = v.reshape((n, trailing))?;
    let v_sq = v_flat.sqr()?;
    let v_sum_sq = v_sq.sum_keepdim(1)?; // [N, 1]
    let v_norm = v_sum_sq.sqrt()?; // [N, 1]
    // Reshape v_norm back to broadcast against `v` (same rank as v).
    let mut shape = vec![n];
    for _ in 1..v.rank() {
        shape.push(1);
    }
    let v_norm = v_norm.reshape(shape)?;
    let v_normalized = v.broadcast_div(&v_norm)?;
    Ok(g.broadcast_mul(&v_normalized)?)
}

/// Construct a candle `Conv1d` with weight-norm-materialised weight.
///
/// `vb` should already be at the `parametrizations.weight` prefix; this
/// reads `original0` (g) and `original1` (v) under it.
pub fn load_weight_norm_conv1d(
    vb: VarBuilder,
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    stride: usize,
    bias: Option<Tensor>,
) -> Result<Conv1d> {
    let g = vb.get((out_ch, 1, 1), "original0")?;
    let v = vb.get((out_ch, in_ch, kernel), "original1")?;
    let weight = materialize_weight_norm(&g, &v)?;
    let cfg = Conv1dConfig {
        padding: 0,
        stride,
        dilation: 1,
        groups: 1,
        ..Default::default()
    };
    Ok(Conv1d::new(weight, bias, cfg))
}

/// Same, but for transposed convolutions. Note: PyTorch
/// `ConvTranspose1d` weight shape is `[in, out, kernel]` (NOT `[out, in,
/// kernel]`), so the leading-axis norm is per INPUT channel.
pub fn load_weight_norm_conv_transpose1d(
    vb: VarBuilder,
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    stride: usize,
    bias: Option<Tensor>,
) -> Result<ConvTranspose1d> {
    let g = vb.get((in_ch, 1, 1), "original0")?;
    let v = vb.get((in_ch, out_ch, kernel), "original1")?;
    let weight = materialize_weight_norm(&g, &v)?;
    let cfg = ConvTranspose1dConfig {
        padding: 0,
        output_padding: 0,
        stride,
        dilation: 1,
        groups: 1,
    };
    Ok(ConvTranspose1d::new(weight, bias, cfg))
}

/// 1-D reflection padding along the last axis. Mirrors
/// `torch.nn.functional.pad(x, (left, right), mode='reflect')`.
///
/// Reflect padding **excludes** the boundary itself, e.g. for
/// `[a, b, c, d, e]`:
/// - `left=2`: `[c, b, a, b, c, d, e]`
/// - `right=2`: `[a, b, c, d, e, d, c]`
///
/// We do not implement upstream's `extra_pad` workaround for short
/// sequences (`length <= max_pad`) because the decoder always operates
/// on `T >> kernel_size` tensors.
pub fn pad1d_reflect(x: &Tensor, left: usize, right: usize) -> Result<Tensor> {
    if left == 0 && right == 0 {
        return Ok(x.clone());
    }
    let last_dim = x.rank() - 1;
    let t = x.dim(last_dim)?;
    if t < 2 && (left > 0 || right > 0) {
        return Err(anyhow!(
            "pad1d_reflect: need T >= 2 for reflect padding (got T={t})"
        ));
    }

    let mut parts: Vec<Tensor> = Vec::with_capacity(3);
    if left > 0 {
        let indices: Vec<u32> = (1..=left as u32).rev().collect();
        let idx = Tensor::from_vec(indices, left, x.device())?;
        parts.push(x.index_select(&idx, last_dim)?);
    }
    parts.push(x.clone());
    if right > 0 {
        // i goes from t-2 down to t-1-right (inclusive both ends).
        let start = (t as i64) - 2;
        let end_inclusive = (t as i64) - 1 - (right as i64);
        if end_inclusive < 0 {
            return Err(anyhow!(
                "pad1d_reflect: right={right} exceeds tail capacity for T={t}"
            ));
        }
        let mut indices: Vec<u32> = Vec::with_capacity(right);
        let mut i = start;
        while i >= end_inclusive {
            indices.push(i as u32);
            i -= 1;
        }
        let idx = Tensor::from_vec(indices, right, x.device())?;
        parts.push(x.index_select(&idx, last_dim)?);
    }
    let parts_refs: Vec<&Tensor> = parts.iter().collect();
    Ok(Tensor::cat(&parts_refs, last_dim)?)
}

/// `CausalConv1d` from upstream — left-pad-then-conv with computed
/// extra-right-pad to match the upstream's "output length is
/// `ceil(n_frames)`" rule.
pub struct CausalConv1d {
    conv: Conv1d,
    stride: usize,
    /// `(kernel - 1) * dilation + 1`. Dilation is always 1 in the codec.
    effective_kernel_size: usize,
    /// `effective_kernel_size - stride`.
    padding_total: usize,
    /// Padding mode — currently only "reflect" is exercised by the codec.
    pad_mode: PadMode,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PadMode {
    Reflect,
    /// `replicate` mode (used by the decoder's first conv per upstream).
    Replicate,
}

impl PadMode {
    pub fn from_str(s: &str) -> Result<Self> {
        match s {
            "reflect" => Ok(PadMode::Reflect),
            "replicate" => Ok(PadMode::Replicate),
            other => Err(anyhow!("unsupported pad_mode {other:?}")),
        }
    }
}

impl CausalConv1d {
    pub fn load(
        vb: VarBuilder,
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        stride: usize,
        pad_mode: PadMode,
    ) -> Result<Self> {
        let conv = load_weight_norm_conv1d(vb.pp("conv.parametrizations.weight"), in_ch, out_ch, kernel, stride, None)?;
        let effective_kernel_size = kernel; // dilation=1
        let padding_total = effective_kernel_size.saturating_sub(stride);
        Ok(Self {
            conv,
            stride,
            effective_kernel_size,
            padding_total,
            pad_mode,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Compute extra right padding to make the output length match
        // `ceil(n_frames)`. Mirrors upstream's CausalConv1d.forward.
        let last_dim = x.rank() - 1;
        let t = x.dim(last_dim)? as f64;
        let n_frames =
            (t - self.effective_kernel_size as f64 + self.padding_total as f64) / self.stride as f64
                + 1.0;
        let target_length = (n_frames.ceil() as i64 - 1) * self.stride as i64
            + (self.effective_kernel_size as i64 - self.padding_total as i64);
        let extra_padding = (target_length - t as i64).max(0) as usize;

        let padded = match self.pad_mode {
            PadMode::Reflect => pad1d_reflect(x, self.padding_total, extra_padding)?,
            PadMode::Replicate => {
                // candle has pad_with_same for replicate-style padding.
                let mut t = x.clone();
                if self.padding_total > 0 {
                    t = t.pad_with_same(last_dim, self.padding_total, 0)?;
                }
                if extra_padding > 0 {
                    t = t.pad_with_same(last_dim, 0, extra_padding)?;
                }
                t
            }
        };
        Ok(self.conv.forward(&padded)?)
    }
}

/// `CausalConvTranspose1d` — runs ConvTranspose1d, then trims
/// `kernel - stride` samples from the right (`trim_ratio=1.0` upstream).
pub struct CausalConvTranspose1d {
    conv: ConvTranspose1d,
    kernel: usize,
    stride: usize,
}

impl CausalConvTranspose1d {
    pub fn load(
        vb: VarBuilder,
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        stride: usize,
    ) -> Result<Self> {
        let conv = load_weight_norm_conv_transpose1d(
            vb.pp("conv.parametrizations.weight"),
            in_ch,
            out_ch,
            kernel,
            stride,
            None,
        )?;
        Ok(Self {
            conv,
            kernel,
            stride,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let total_padding = self.kernel.saturating_sub(self.stride);
        // trim_ratio=1.0 → all padding goes to the right.
        let right = total_padding;
        let left = 0;
        let out = self.conv.forward(x)?;
        let last_dim = out.rank() - 1;
        let t_out = out.dim(last_dim)?;
        Ok(out.i((.., .., left..t_out - right))?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    fn cpu() -> Device {
        Device::Cpu
    }

    #[test]
    fn weight_norm_materialization_matches_formula() {
        // Construct a tiny v=[2, 3, 2] with known values, g=[2, 1, 1].
        // Each row of v: ‖v[0]‖ = sqrt(1+4+9+16+25+36) = sqrt(91)
        //               ‖v[1]‖ = sqrt(49+64+81+100+121+144) = sqrt(559)
        let v_data: Vec<f32> = (1..=12).map(|i| i as f32).collect();
        let v = Tensor::from_vec(v_data.clone(), (2, 3, 2), &cpu()).unwrap();
        let g = Tensor::from_vec(vec![3.0f32, 5.0], (2, 1, 1), &cpu()).unwrap();
        let w = materialize_weight_norm(&g, &v).unwrap();

        // Expected w[i, j, k] = g[i] * v[i, j, k] / ‖v[i]‖
        let v0_norm = ((1.0f32.powi(2) + 4.0 + 9.0 + 16.0 + 25.0 + 36.0) as f32).sqrt();
        let v1_norm = ((49.0f32 + 64.0 + 81.0 + 100.0 + 121.0 + 144.0) as f32).sqrt();
        let w_v: Vec<f32> = w.flatten_all().unwrap().to_vec1().unwrap();
        for i in 0..6 {
            let expected = 3.0 * (i + 1) as f32 / v0_norm;
            assert!(
                (w_v[i] - expected).abs() < 1e-5,
                "row 0 idx {i}: got {} vs {}",
                w_v[i],
                expected
            );
        }
        for i in 0..6 {
            let expected = 5.0 * (i + 7) as f32 / v1_norm;
            assert!(
                (w_v[6 + i] - expected).abs() < 1e-4,
                "row 1 idx {i}: got {} vs {}",
                w_v[6 + i],
                expected
            );
        }
    }

    #[test]
    fn pad1d_reflect_left() {
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0], (1, 1, 5), &cpu()).unwrap();
        let p = pad1d_reflect(&x, 2, 0).unwrap();
        let v: Vec<f32> = p.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(v, vec![3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn pad1d_reflect_right() {
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0], (1, 1, 5), &cpu()).unwrap();
        let p = pad1d_reflect(&x, 0, 2).unwrap();
        let v: Vec<f32> = p.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(v, vec![1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0]);
    }

    #[test]
    fn pad1d_reflect_both() {
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0], (1, 1, 5), &cpu()).unwrap();
        let p = pad1d_reflect(&x, 2, 2).unwrap();
        let v: Vec<f32> = p.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(v, vec![3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0]);
    }

    #[test]
    fn pad1d_reflect_zero_zero() {
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (1, 1, 3), &cpu()).unwrap();
        let p = pad1d_reflect(&x, 0, 0).unwrap();
        assert_eq!(p.dims(), x.dims());
    }
}
