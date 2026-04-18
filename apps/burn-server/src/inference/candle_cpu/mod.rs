//! CPU inference with Q4 GGUF quantization — candle 0.9 CPU backend.
//!
//! Uses candle-core's quantized module (QTensor, QMatMul) for Q4 dequant-on-the-fly
//! matmul with AVX2/AVX-512 SIMD. Targets real-time ASR on high-end x86 CPUs
//! (e.g., AMD Ryzen 9 9955HX3D with 96MB L3 V-Cache).
//!
//! Architecture is identical to candle_native but:
//! - Device::Cpu instead of CUDA
//! - Manual attention (no FlashAttention)
//! - QMatMul for all weight-heavy layers (loaded from GGUF)
//! - F32 activations and KV caches

pub mod engine;
pub mod model;
