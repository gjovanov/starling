//! BF16 inference path using Burn tensors and SafeTensors weight loading.
//!
//! Architecture (4B params):
//!   - Audio Encoder: 32-layer causal Whisper, 1280 dim, sliding window 750
//!   - Adapter: 2-layer MLP projecting 5120 -> 3072 with GELU
//!   - Decoder: 26-layer Ministral-3B, 3072 dim, GQA (32Q/8KV), sliding window 8192
//!
//! Left-padding: 32 silence tokens (standard, upstream default)

pub mod layers;
pub mod loader;
pub mod model;
pub mod weights;
