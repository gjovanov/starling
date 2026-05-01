//! Voxtral-4B-TTS audio-tokenizer codec (decoder side only).
//!
//! Phase 2-D scope: port `vllm_omni.model_executor.models.voxtral_tts
//! .voxtral_tts_audio_tokenizer.VoxtralTTSAudioTokenizer.decode()` to
//! candle. The shipped checkpoint contains only decoder + quantizer +
//! token-embedding weights — encoder support is out of scope.
//!
//! Architecture (from `params.json`):
//! - Quantizer (`MistralAudioCodebook`): semantic codebook (8192 entries
//!   × 256-dim, Euclidean lookup) + acoustic codebook (21 levels × 36
//!   dims, finite-scalar quantisation rescale).
//! - 8 alternating decoder blocks:
//!   - block 0: `CausalConv1d` 292→1024, kernel 3, stride 1.
//!   - block 1: `Transformer` with 2 sublayers, dim=1024.
//!   - block 2: `CausalConvTranspose1d` 1024→1024, kernel 4, stride 2 (×2 upsample).
//!   - block 3: `Transformer` × 2.
//!   - block 4: `CausalConvTranspose1d` ×2 upsample.
//!   - block 5: `Transformer` × 2.
//!   - block 6: `CausalConvTranspose1d` ×2 upsample.
//!   - block 7: `Transformer` × 2.
//! - `output_proj`: `CausalConv1d` 1024→240, kernel 7. Then a
//!   `b (c h) t -> b c (t h)` rearrange yields [B, C=1, T*240] = 24 kHz PCM.
//!
//! Novel features (vs the AR LLM port):
//! - **Weight-norm parametrisation**: convs store `(original0=g [out, 1, 1],
//!   original1=v [out, in, k])` and materialise `w = g * v / ‖v‖` at runtime.
//! - **ALiBi positional bias** instead of RoPE, with per-head geometric
//!   slopes `r = 2^(-8/H)` and bias `slope * (j - i)`.
//! - **Sliding window** attention (16 keys back, no future for causal).
//! - **qk_norm**: RmsNorm on q and k before the attention scaling.
//! - **Layer-scale**: per-layer learnable `attention_scale` and
//!   `ffn_scale` parameter vectors multiply each residual branch.

pub mod args;
pub mod conv;

pub use args::AudioTokenizerArgs;
pub use conv::{
    materialize_weight_norm, pad1d_reflect, CausalConv1d, CausalConvTranspose1d, PadMode,
};
