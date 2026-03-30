pub mod bf16;
pub mod bf16_engine;
pub mod config;
#[cfg(feature = "cuda")]
pub mod cuda_engine;
pub mod engine;
pub mod q4;
pub mod tokenizer;

/// Streaming inference session — mirrors the vLLM Realtime API protocol.
///
/// The engine maintains internal state (KV cache, accumulated audio tokens)
/// across multiple `send_audio` + `commit` calls, just like vLLM does.
///
/// Flow (same as vllm-server's VLLMClient):
/// 1. `create_session()` — initialize model, KV cache, streaming prefix
/// 2. Loop:
///    a. `send_audio(samples)` — append raw PCM audio to internal buffer
///    b. `commit()` — trigger inference on buffered audio, return text delta
/// 3. `reset()` — rotate session (clear KV cache) to avoid context overflow
///
/// Each commit adds ~50-80 audio tokens. At 16384 max context, rotate
/// after ~200 commits (~100s of audio).
pub trait InferenceSession: Send {
    /// Append raw 16kHz mono f32 PCM audio to the internal buffer.
    fn send_audio(&mut self, samples: &[f32]);

    /// Trigger inference on buffered audio. Returns the text delta
    /// (new words transcribed since last commit).
    fn commit(&mut self) -> Result<String, Box<dyn std::error::Error + Send + Sync>>;

    /// Reset the session (clear KV cache, audio buffer).
    /// Used to rotate context before hitting max_model_len.
    fn reset(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;

    /// Number of commits since last reset.
    fn commit_count(&self) -> usize;
}

/// Factory for creating streaming inference sessions.
///
/// One engine instance (loaded model weights) can create multiple sessions.
pub trait InferenceEngine: Send + Sync {
    /// Create a new streaming session with the given language.
    fn create_session(
        &self,
        language: &str,
    ) -> Result<Box<dyn InferenceSession>, Box<dyn std::error::Error + Send + Sync>>;
}

/// Maximum commits before rotating the session to avoid context overflow.
/// Matches vllm-server's MAX_COMMITS_BEFORE_ROTATE = 200.
/// At 0.5s batches, this is ~100s of audio.
pub const MAX_COMMITS_BEFORE_ROTATE: usize = 200;
