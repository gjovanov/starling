//! SwiGLU MLP layer.
//!
//! Used in both audio encoder and language model.

use burn::config::Config;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::silu;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

/// SwiGLU configuration.
#[derive(Config, Debug)]
pub struct SwiGLUConfig {
    /// Input/output dimension.
    pub d_model: usize,
    /// Hidden dimension (typically 4x d_model for transformers).
    pub hidden_dim: usize,
    /// Whether to use bias (encoder=false, LLM=false).
    #[config(default = false)]
    pub bias: bool,
}

/// SwiGLU MLP layer.
///
/// Computes: `w2(silu(w1(x)) * w3(x))`
///
/// Named w1/w2/w3 to match Voxtral weight names:
/// - w1: gate projection
/// - w2: down projection
/// - w3: up projection
#[derive(Module, Debug)]
pub struct SwiGLU<B: Backend> {
    /// Gate projection: d_model -> hidden_dim
    w1: Linear<B>,
    /// Down projection: hidden_dim -> d_model
    w2: Linear<B>,
    /// Up projection: d_model -> hidden_dim
    w3: Linear<B>,
}

impl SwiGLUConfig {
    /// Initialize the SwiGLU layer.
    pub fn init<B: Backend>(&self, device: &B::Device) -> SwiGLU<B> {
        let w1 = LinearConfig::new(self.d_model, self.hidden_dim)
            .with_bias(self.bias)
            .init(device);
        let w2 = LinearConfig::new(self.hidden_dim, self.d_model)
            .with_bias(self.bias)
            .init(device);
        let w3 = LinearConfig::new(self.d_model, self.hidden_dim)
            .with_bias(self.bias)
            .init(device);

        SwiGLU { w1, w2, w3 }
    }
}

impl<B: Backend> SwiGLU<B> {
    /// Create SwiGLU from linear layers (for weight loading).
    pub fn new(w1: Linear<B>, w2: Linear<B>, w3: Linear<B>) -> Self {
        Self { w1, w2, w3 }
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// * `x` - Input tensor [batch, seq, d_model]
    ///
    /// # Returns
    /// Output tensor [batch, seq, d_model]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let gate = self.w1.forward(x.clone());
        let gate = silu(gate);
        let up = self.w3.forward(x);
        self.w2.forward(gate * up)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Wgpu;

    type TestBackend = Wgpu;

    #[test]
    fn test_swiglu_shape() {
        let device = Default::default();
        let config = SwiGLUConfig::new(64, 256);
        let mlp = config.init::<TestBackend>(&device);

        let x = Tensor::<TestBackend, 3>::zeros([2, 10, 64], &device);
        let out = mlp.forward(x);

        assert_eq!(out.dims(), [2, 10, 64]);
    }
}
