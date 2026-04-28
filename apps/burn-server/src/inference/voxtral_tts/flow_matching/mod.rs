//! Flow-matching acoustic transformer for Voxtral-4B-TTS.
//!
//! Velocity-field network for ODE integration over the acoustic-codebook
//! continuous space. Per AR-LLM step, the network is queried 7 times
//! (8 timesteps, 7 dt intervals) with Classifier-Free Guidance to draw
//! a single acoustic-frame from noise.
//!
//! Pipeline (per [`FlowMatchingAudioTransformer::forward`] call):
//! 1. `semantic_logit = semantic_codebook_output(llm_hidden)` →
//!    argmax → `semantic_code`.
//! 2. If `semantic_code == END_AUDIO`, emit `[EMPTY_AUDIO]` × 36 acoustic codes.
//! 3. Otherwise: `x_0 ~ N(0, 1)` (shape `[B, 36]`), then 7 Euler steps:
//!    - run [`predict_velocity`] with batched `[cond, uncond]`.
//!    - mix `v = α·v_cond + (1-α)·v_uncond` with `α=1.2`.
//!    - `x_{t+1} = x_t + v · dt`.
//! 4. Quantize `clamp(x_7, -1, 1)` → 21-level acoustic codes + special-token offset.
//! 5. Concatenate `[semantic_code, *acoustic_codes]` → `audio_codes [B, 37]`.
//!
//! Reference: `vllm_omni.model_executor.models.voxtral_tts
//! .voxtral_tts_audio_generation.FlowMatchingAudioTransformer`.

pub mod args;
pub mod model;

#[cfg(test)]
mod golden_tests;

pub use args::{
    AcousticTransformerArgs, AudioSpecialTokens, FlowMatchingDecodeArgs, MultimodalAudioModelArgs,
};
pub use model::{
    AcousticTransformerBlock, BidirectionalAttention, FeedForward, FlowMatchingAudioTransformer,
    TimeEmbedding,
};
