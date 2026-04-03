//! Q4_0 GGUF inference path with fused dequant+matmul WGSL shaders.
//!
//! Q4_0 quantization uses 18-byte blocks encoding 32 elements each:
//!   - 2-byte f16 scale
//!   - 16 bytes of packed 4-bit nibbles
//!   - Dequantization: (nibble - 8) * scale
//!
//! Weights are never materialized to f32 -- the WGSL shaders read packed
//! 4-bit nibbles and dequantize on-the-fly during matrix multiplication.
//!
//! Left-padding: 76 silence tokens (extended from upstream's 32)
//!
//! Q4 Padding Workaround:
//!   The upstream left-pads with 32 tokens, covering only 16 of 38 decoder prefix
//!   positions with silence. Q4_0 makes the decoder sensitive to speech in the prefix:
//!   audio starting immediately with speech produces all-pad tokens. Left-padding is
//!   increased to 76 tokens -> exactly 38 decoder positions of silence -> full streaming
//!   prefix covered. See audio/pad.rs for details.

pub mod reader;
pub mod tensor;
pub mod linear;
pub mod op;
pub mod model;
pub mod loader;
pub mod engine;

pub use reader::{GgmlDtype, GgufReader, GgufTensorInfo, ShardedCursor};
pub use tensor::Q4Tensor;
pub use linear::Q4Linear;
pub use model::Q4VoxtralModel;
pub use loader::{Q4ModelLoader, Q4ModelParts};

/// Non-fused Wgpu backend for Q4 custom kernel dispatch.
///
/// The default `Wgpu` type includes fusion (`Fusion<CubeBackend<WgpuRuntime>>`),
/// which wraps tensors in `FusionTensor` rather than raw `CubeTensor`. Custom WGSL
/// kernels need direct access to GPU handles, so the Q4 module uses the non-fused
/// `CubeBackend<WgpuRuntime>` directly.
///
/// The shared BF16 layers (RoPE, RmsNorm, KVCache, masking, ConvDownsampler) are
/// generic over `B: Backend` and work with either fused or non-fused backends.
pub type WgpuBackend = burn::backend::wgpu::CubeBackend<burn::backend::wgpu::WgpuRuntime, f32, i32, u32>;

/// The device type used by the Q4 backend (same as `WgpuDevice`).
pub type Q4Device = burn::backend::wgpu::WgpuDevice;
