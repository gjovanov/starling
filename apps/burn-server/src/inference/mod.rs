pub mod bf16;
pub mod config;
pub mod q4;

/// Inference engine trait — abstracts over Q4 and BF16 model implementations.
///
/// Each backend handles its own:
/// - Weight loading (SafeTensors for BF16, GGUF for Q4)
/// - Left-padding (32 tokens for BF16, 76 tokens for Q4)
/// - GPU compute dispatch
pub trait InferenceEngine: Send + Sync {
    /// Transcribe a chunk of 16kHz mono f32 PCM audio to text.
    fn transcribe(
        &self,
        audio: &[f32],
        language: &str,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>>;
}
