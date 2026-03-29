//! Convolutional downsampler for the audio encoder.
//!
//! Two Conv1d layers with GELU activation that downsample mel spectrograms
//! from 128 channels to 1280 channels with 4x temporal downsampling.

use burn::config::Config;
use burn::module::Module;
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::nn::PaddingConfig1d;
use burn::tensor::activation::gelu;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

/// Conv downsampler configuration.
#[derive(Config, Debug)]
pub struct ConvDownsamplerConfig {
    /// Input channels (mel bins).
    pub in_channels: usize,
    /// Hidden channels after first conv.
    pub hidden_channels: usize,
    /// Output channels.
    pub out_channels: usize,
    /// Kernel size for both convolutions.
    #[config(default = 3)]
    pub kernel_size: usize,
    /// Stride for both convolutions (total downsample = stride^2).
    #[config(default = 2)]
    pub stride: usize,
}

/// Convolutional downsampler.
///
/// Architecture:
/// - Conv1d(in_channels -> hidden_channels, kernel=3, stride=2, pad=1) + GELU
/// - Conv1d(hidden_channels -> out_channels, kernel=3, stride=2, pad=1) + GELU
///
/// Total temporal downsampling: 4x (stride 2 x stride 2)
#[derive(Module, Debug)]
pub struct ConvDownsampler<B: Backend> {
    pub(crate) conv1: Conv1d<B>,
    pub(crate) conv2: Conv1d<B>,
}

impl ConvDownsamplerConfig {
    /// Initialize the ConvDownsampler.
    pub fn init<B: Backend>(&self, device: &B::Device) -> ConvDownsampler<B> {
        // Padding of 1 with kernel 3 and stride 2 gives: (L + 2*1 - 3) / 2 + 1 = (L + 1) / 2
        // For L=100: (100 + 1) / 2 = 50
        let conv1 = Conv1dConfig::new(self.in_channels, self.hidden_channels, self.kernel_size)
            .with_stride(self.stride)
            .with_padding(PaddingConfig1d::Explicit(1))
            .with_bias(true)
            .init(device);

        let conv2 = Conv1dConfig::new(self.hidden_channels, self.out_channels, self.kernel_size)
            .with_stride(self.stride)
            .with_padding(PaddingConfig1d::Explicit(1))
            .with_bias(true)
            .init(device);

        ConvDownsampler { conv1, conv2 }
    }
}

impl<B: Backend> ConvDownsampler<B> {
    /// Create downsampler from conv layers (for weight loading).
    pub fn new(conv1: Conv1d<B>, conv2: Conv1d<B>) -> Self {
        Self { conv1, conv2 }
    }

    /// Forward pass with causal (left-only) padding.
    ///
    /// Voxtral uses causal convolutions: pad = kernel_size - stride, left side only.
    /// burn's Conv1d uses symmetric padding, so we manually left-pad and set Conv1d
    /// padding to 0.
    ///
    /// Conv0: stride=1, kernel=3 → causal pad=2 (left)
    /// Conv1: stride=2, kernel=3 → causal pad=1 (left)
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // Conv0: causal left-pad by (kernel - stride) = 3 - 1 = 2
        let x = causal_pad(x, 2);
        let x = self.conv1.forward(x);
        let x = gelu(x);

        // Conv1: causal left-pad by (kernel - stride) = 3 - 2 = 1
        let x = causal_pad(x, 1);
        let x = self.conv2.forward(x);
        gelu(x)
    }
}

/// Causal (left-only) padding for Conv1d.
/// Pads `pad` zeros on the left of the time dimension (dim 2).
/// Input: [batch, channels, time] → Output: [batch, channels, time + pad]
fn causal_pad<B: Backend>(x: Tensor<B, 3>, pad: usize) -> Tensor<B, 3> {
    if pad == 0 {
        return x;
    }
    let [batch, channels, time] = x.dims();
    let zeros = Tensor::<B, 3>::zeros([batch, channels, pad], &x.device());
    Tensor::cat(vec![zeros, x], 2) // concat along time dimension
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Wgpu;

    type TestBackend = Wgpu;

    #[test]
    fn test_conv_downsampler_shape() {
        let device = Default::default();
        let config = ConvDownsamplerConfig::new(128, 1280, 1280);
        let conv = config.init::<TestBackend>(&device);

        // Input: [batch=1, channels=128, time=100]
        let x = Tensor::<TestBackend, 3>::zeros([1, 128, 100], &device);
        let out = conv.forward(x);

        // Output should be [1, 1280, 25] (4x downsample)
        assert_eq!(out.dims()[0], 1);
        assert_eq!(out.dims()[1], 1280);
        // With padding=1, kernel=3, stride=2: (100 + 2 - 3) / 2 + 1 = 50
        // Then again: (50 + 2 - 3) / 2 + 1 = 25
        assert_eq!(out.dims()[2], 25);
    }
}
