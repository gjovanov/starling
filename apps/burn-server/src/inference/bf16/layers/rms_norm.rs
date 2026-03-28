//! RMSNorm and ADA RMSNorm layers.
//!
//! Standard RMSNorm for the LLM, and adaptive RMSNorm (t-conditioned) for
//! both encoder and LLM layers in Voxtral.

use burn::config::Config;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::gelu;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

/// Standard RMSNorm configuration.
#[derive(Config, Debug)]
pub struct RmsNormConfig {
    /// Hidden dimension.
    pub d_model: usize,
    /// Epsilon for numerical stability.
    #[config(default = 1e-5)]
    pub eps: f64,
}

/// Standard RMSNorm layer.
///
/// Applies: `x * weight / sqrt(mean(x^2) + eps)`
#[derive(Module, Debug)]
pub struct RmsNorm<B: Backend> {
    /// Learnable scale parameter.
    pub weight: burn::nn::RmsNorm<B>,
}

impl RmsNormConfig {
    /// Initialize the RmsNorm layer.
    pub fn init<B: Backend>(&self, device: &B::Device) -> RmsNorm<B> {
        let weight = burn::nn::RmsNormConfig::new(self.d_model)
            .with_epsilon(self.eps)
            .init(device);
        RmsNorm { weight }
    }
}

impl<B: Backend> RmsNorm<B> {
    /// Forward pass.
    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        self.weight.forward(x)
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
