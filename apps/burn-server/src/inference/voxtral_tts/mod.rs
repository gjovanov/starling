//! Native Rust/Candle port of Voxtral-4B-TTS (Phase 2 — scaffolding only).
//!
//! Phase status: weight-inventory + safetensors-loader stub. No forward
//! pass yet; that lands in Phase 2-C onwards.
//!
//! Architecture (from the upstream `vllm-omni` reference):
//! - **AR LLM** (26 layers, dim 3072, GQA 32h/8kv, hidden 9216, RoPE θ=1M):
//!   identical-shape to the Voxtral-Mini-Realtime ASR decoder minus ADA.
//!   Reuses 90% of `apps/burn-server/src/inference/candle_native` once
//!   wired up.
//! - **Acoustic Transformer / Flow Matching** (3 layers, bidirectional,
//!   RoPE θ=10k, 8-step Euler integration with CFG α=1.2): predicts
//!   continuous acoustic-codebook outputs from the AR LLM hidden state.
//! - **Codec decoder** (8 alternating conv ↔ transformer blocks, dim
//!   1024, ALiBi sliding-window 16, qk_norm, layer-scale, weight-norm
//!   convs): turns 37-codebook integer codes into 24 kHz PCM.
//!
//! See `apps/burn-server/test_data/tts_golden/` for bit-exact regression
//! fixtures captured from the upstream Python implementation, and the
//! `project_voxtral_tts_model_inventory` memory note for tensor-level
//! ground truth.

pub mod codec;
pub mod flow_matching;
pub mod weights;

pub use codec::AudioTokenizerArgs;
pub use flow_matching::{AcousticTransformerArgs, MultimodalAudioModelArgs};
pub use weights::{ExpectedGroup, ModuleGroup, WeightInventory, EXPECTED_GROUPS, EXPECTED_TOTAL};
