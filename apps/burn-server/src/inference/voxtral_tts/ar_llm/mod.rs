//! Autoregressive language model — the Voxtral-4B-TTS decoder that
//! emits one audio frame at a time, conditioning the flow-matching head.
//!
//! Architecture (params.json top-level): 26-layer GQA transformer,
//! `dim=3072, n_heads=32, n_kv_heads=8, head_dim=128, hidden_dim=9216,
//! vocab_size=131072, rope_theta=1_000_000, tied_embeddings=true,
//! norm_eps=1e-5`. **No ADA** — unlike the Voxtral-Mini-Realtime ASR
//! decoder, the TTS AR LLM has no audio-feature conditioning path.
//! Identical layer SHAPE to ASR (we can borrow most of
//! `candle_native_flash`'s implementation), but two differences in the
//! input + halt logic:
//!
//! 1. **Voice embedding prefix.** A `[163, 3072]` embedding loaded from
//!    `voice_embedding/<name>.pt` is summed into the input embeddings
//!    at the audio-token positions before the first forward.
//!    `input_embedding_concat_type` is `"sum"` upstream — additive
//!    combination, not concatenate.
//!
//! 2. **Halt criterion.** The AR LLM doesn't emit a textual EOS; it
//!    drives the flow-matching head, which signals end via
//!    `semantic_code == END_AUDIO`. The autoregressive loop checks
//!    that on every step.
//!
//! Phase 2-F.1 scope: args, module skeleton, and a safetensors-key
//! validator that asserts the AR LLM's 234 + 2 + 1 = 237 expected
//! tensors are all present. No forward pass yet — that's Phase 2-F.2.

pub mod args;
pub mod model;

pub use args::ArLlmArgs;
pub use model::{ArLlmAttention, ArLlmBlock, ArLlmFeedForward, ArLlmModel};
