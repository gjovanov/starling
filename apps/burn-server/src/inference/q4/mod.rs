//! Q4_0 GGUF inference path with fused dequant+matmul WGSL shaders.
//!
//! Q4_0 quantization uses 18-byte blocks encoding 32 elements each:
//!   - 2-byte f16 scale
//!   - 16 bytes of packed 4-bit nibbles
//!   - Dequantization: (nibble - 8) * scale
//!
//! Weights are never materialized to f32 — the WGSL shaders read packed
//! 4-bit nibbles and dequantize on-the-fly during matrix multiplication.
//!
//! Left-padding: 76 silence tokens (extended from upstream's 32)
//!
//! Q4 Padding Workaround:
//!   The upstream left-pads with 32 tokens, covering only 16 of 38 decoder prefix
//!   positions with silence. Q4_0 makes the decoder sensitive to speech in the prefix:
//!   audio starting immediately with speech produces all-pad tokens. Left-padding is
//!   increased to 76 tokens → exactly 38 decoder positions of silence → full streaming
//!   prefix covered. See audio/pad.rs for details.

// TODO: Port from voxtral-mini-realtime-rs src/gguf/
// Files to port: reader.rs, q4_tensor.rs, q4_linear.rs, q4_model.rs
// WGSL shaders: shader.wgsl, shader_naive.wgsl
