//! BF16 neural network layer modules for Voxtral-Mini-4B-Realtime.
//!
//! Each layer is a Burn `Module` with `#[derive(Module, Debug)]` and full
//! forward implementations for both standard and KV-cached inference.
//!
//! Layer hierarchy:
//! - `masking` - Causal and sliding window attention masks
//! - `rope` - Rotary position embeddings (RoPE)
//! - `kv_cache` - KV cache for autoregressive generation
//! - `rms_norm` - RMSNorm and adaptive (ADA) RMSNorm
//! - `swiglu` - SwiGLU MLP (gate/up/down projections)
//! - `conv` - Conv1d downsampler (4x temporal reduction)
//! - `attention` - Multi-head attention with MHA/GQA, RoPE, causal masking
//! - `encoder_layer` - Pre-LN transformer block for audio encoder
//! - `decoder_layer` - Pre-LN transformer block with ADA modulation for LLM decoder
//! - `weights` - Weight name constants for SafeTensors/GGUF loading

pub mod attention;
pub mod conv;
pub mod decoder_layer;
pub mod encoder_layer;
pub mod kv_cache;
pub mod masking;
pub mod rms_norm;
pub mod rope;
pub mod swiglu;
pub mod weights;

// Re-export key types for convenient access.
pub use attention::{Attention, AttentionConfig};
pub use conv::{ConvDownsampler, ConvDownsamplerConfig};
pub use decoder_layer::{DecoderLayer, DecoderLayerConfig};
pub use encoder_layer::{EncoderLayer, EncoderLayerConfig};
pub use kv_cache::{KVCache, LayerCaches};
pub use masking::{
    apply_causal_mask, apply_causal_mask_with_offset, apply_sliding_window_mask,
    apply_sliding_window_mask_with_offset,
};
pub use rms_norm::{AdaRmsNorm, AdaRmsNormConfig, RmsNorm, RmsNormConfig};
pub use rope::{RoPE, RoPEConfig};
pub use swiglu::{SwiGLU, SwiGLUConfig};
