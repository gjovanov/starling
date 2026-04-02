//! Pure candle CUDA inference — bypasses burn for direct cuBLAS + FlashAttention v2.
//!
//! Uses candle-core 0.10 + candle-nn + candle-flash-attn for:
//! - FlashAttention v2 (single kernel per attention layer, GQA native)
//! - Direct cuBLAS bf16 matmuls (no burn abstraction overhead)
//! - candle-nn RmsNorm (single fused kernel)
//! - ~130 kernels/step vs ~320 with burn-candle

pub mod model;
pub mod engine;
