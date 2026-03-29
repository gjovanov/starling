//! RMSNorm and ADA RMSNorm layers.
//!
//! Standard RMSNorm for the LLM, and adaptive RMSNorm (t-conditioned) for
//! both encoder and LLM layers in Voxtral.

use burn::config::Config;
use burn::module::{Module, Param, ParamId};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::gelu;
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};

/// Standard RMSNorm configuration.
#[derive(Config, Debug)]
pub struct RmsNormConfig {
    /// Hidden dimension.
    pub d_model: usize,
    /// Epsilon for numerical stability.
    #[config(default = 1e-5)]
    pub eps: f64,
}

/// Standard RMSNorm layer using matmul-based mean computation.
///
/// Applies: `x * gamma / sqrt(mean(x^2) + eps)`
///
/// Uses `x² @ ones / d_model` instead of `mean_dim` to compute
/// the mean of squares. This avoids a burn/CubeCL reduction kernel
/// bug on DZN (Vulkan-over-DX12) where `mean_dim` produces incorrect
/// results (1.68× scale error).
#[derive(Module, Debug)]
pub struct RmsNorm<B: Backend> {
    /// Learnable scale parameter (gamma).
    pub gamma: Param<Tensor<B, 1>>,
    /// Epsilon for numerical stability.
    pub epsilon: f64,
}

impl RmsNormConfig {
    /// Initialize the RmsNorm layer.
    pub fn init<B: Backend>(&self, device: &B::Device) -> RmsNorm<B> {
        let gamma = Tensor::ones([self.d_model], device);
        RmsNorm {
            gamma: Param::initialized(ParamId::new(), gamma),
            epsilon: self.eps,
        }
    }
}

impl<B: Backend> RmsNorm<B> {
    /// Forward pass using matmul-based mean(x²) computation.
    ///
    /// For input [B, S, D]: computes `x² @ ones_col / D` to get mean(x²)
    /// per position, avoiding the buggy `mean_dim` reduction kernel.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq, d_model] = x.dims();
        let device = x.device();

        // x² [B, S, D]
        let x_sq = x.clone() * x.clone();

        // mean(x²) via matmul: [B, S, D] @ [D, 1] / D → [B, S, 1]
        let ones = Tensor::<B, 2>::ones([d_model, 1], &device);
        let ones_3d = ones.unsqueeze::<3>(); // [1, D, 1]
        let sum_sq = x_sq.matmul(ones_3d); // [B, S, 1]
        let mean_sq = sum_sq / (d_model as f32);

        // rms = sqrt(mean(x²) + eps)
        let rms = (mean_sq + self.epsilon).sqrt();

        // normalize and scale
        (x / rms) * self.gamma.val().unsqueeze::<3>()
    }
}

/// ADA RMSNorm configuration (t-conditioned normalization).
#[derive(Config, Debug)]
pub struct AdaRmsNormConfig {
    /// Hidden dimension.
    pub d_model: usize,
    /// Temporal conditioning dimension.
    pub t_cond_dim: usize,
    /// Epsilon for numerical stability.
    #[config(default = 1e-5)]
    pub eps: f64,
}

/// Adaptive modulation layer with temporal conditioning.
///
/// Architecture: Linear(d_model -> t_cond_dim) -> GELU -> Linear(t_cond_dim -> d_model)
/// Then applies: `x * (1 + scale)`
///
/// Note: This is NOT a normalization layer - it only applies modulation.
/// The actual RMSNorm happens separately in attention_norm/ffn_norm.
#[derive(Module, Debug)]
pub struct AdaRmsNorm<B: Backend> {
    /// First projection: d_model -> t_cond_dim
    pub(crate) w0: Linear<B>,
    /// Second projection: t_cond_dim -> d_model
    pub(crate) w2: Linear<B>,
    /// Epsilon for numerical stability.
    pub(crate) eps: f64,
}

impl AdaRmsNormConfig {
    /// Initialize the ADA RMSNorm layer.
    pub fn init<B: Backend>(&self, device: &B::Device) -> AdaRmsNorm<B> {
        let w0 = LinearConfig::new(self.d_model, self.t_cond_dim)
            .with_bias(false)
            .init(device);
        let w2 = LinearConfig::new(self.t_cond_dim, self.d_model)
            .with_bias(false)
            .init(device);
        AdaRmsNorm {
            w0,
            w2,
            eps: self.eps,
        }
    }
}

impl<B: Backend> AdaRmsNorm<B> {
    /// Create ADA RMSNorm from linear layers (for weight loading).
    pub fn new(w0: Linear<B>, w2: Linear<B>, eps: f64) -> Self {
        Self { w0, w2, eps }
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// * `x` - Input tensor [batch, seq, d_model]
    /// * `t_embed` - Temporal embedding [batch, 1, d_model]
    ///
    /// # Returns
    /// Modulated tensor [batch, seq, d_model] (not normalized - just scaled)
    pub fn forward(&self, x: Tensor<B, 3>, t_embed: Tensor<B, 3>) -> Tensor<B, 3> {
        // Compute adaptive scale: Linear -> GELU -> Linear
        // t_embed: [batch, 1, d_model] -> w0 -> [batch, 1, t_cond_dim]
        let scale = self.w0.forward(t_embed);
        let scale = gelu(scale);
        let scale = self.w2.forward(scale); // [batch, 1, d_model]

        // Apply adaptive modulation: x * (1 + scale)
        // Note: This is NOT normalization - the actual RMSNorm happens separately
        x * (scale + 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Wgpu;

    type TestBackend = Wgpu;

    #[test]
    fn test_rms_norm_shape() {
        let device = Default::default();
        let config = RmsNormConfig::new(64);
        let norm = config.init::<TestBackend>(&device);

        let x = Tensor::<TestBackend, 3>::zeros([2, 10, 64], &device);
        let out = norm.forward(x);

        assert_eq!(out.dims(), [2, 10, 64]);
    }

    #[test]
    fn test_ada_rms_norm_shape() {
        let device = Default::default();
        let config = AdaRmsNormConfig::new(64, 8);
        let norm = config.init::<TestBackend>(&device);

        let x = Tensor::<TestBackend, 3>::zeros([2, 10, 64], &device);
        let t_embed = Tensor::<TestBackend, 3>::zeros([2, 1, 64], &device);
        let out = norm.forward(x, t_embed);

        assert_eq!(out.dims(), [2, 10, 64]);
    }
}
