//! Audio padding for Voxtral streaming inference.
//!
//! Voxtral streaming mode requires left-padding audio with silence to align
//! with the 38-token decoder prefix (BOS + 37 streaming pad tokens).
//!
//! ## Q4 Padding Workaround
//!
//! The upstream mistral-common library left-pads with 32 silence tokens
//! (at 12.5 Hz), which covers only 16 of 38 decoder prefix positions with
//! silence. The BF16 model tolerates speech content in the remaining 22
//! prefix positions, but Q4_0 quantization makes the decoder sensitive to it:
//! audio that starts immediately with speech produces all-pad tokens.
//!
//! We use 76 left-pad tokens instead, which maps to exactly 38 decoder
//! positions of silence, covering the full streaming prefix.
//!
//! ### Math
//!
//! ```text
//! Token rate: 12.5 Hz (16000 / 1280 samples per token)
//! Left-pad tokens: 76
//! Left-pad samples: 76 * 1280 = 97,280
//!
//! Pipeline: mel → conv(4× downsample) → reshape(4× group) = 16× total
//! Decoder tokens from padding: 76 / 2 = 38 (after conv+reshape)
//!
//! 38 decoder tokens = full streaming prefix (BOS + 37 pad tokens)
//! ```

use super::AudioBuffer;

/// Padding configuration for Voxtral streaming.
#[derive(Debug, Clone)]
pub struct PadConfig {
    /// Audio sample rate
    pub sample_rate: u32,
    /// Number of left-pad tokens (76 for Q4, 32 for BF16)
    pub n_left_pad_tokens: usize,
    /// Token rate in Hz (16000 / 1280 = 12.5)
    pub frame_rate: f32,
    /// Extra right-pad tokens for encoder alignment
    pub extra_right_pad_tokens: usize,
}

impl PadConfig {
    /// Q4-safe padding (76 left-pad tokens — full prefix coverage).
    pub fn q4() -> Self {
        Self {
            sample_rate: 16000,
            n_left_pad_tokens: 76,
            frame_rate: 12.5,
            extra_right_pad_tokens: 17,
        }
    }

    /// BF16 padding (32 left-pad tokens — upstream default).
    pub fn bf16() -> Self {
        Self {
            sample_rate: 16000,
            n_left_pad_tokens: 32,
            frame_rate: 12.5,
            extra_right_pad_tokens: 17,
        }
    }

    /// Samples per token at the given frame rate.
    pub fn samples_per_token(&self) -> usize {
        (self.sample_rate as f32 / self.frame_rate) as usize
    }

    /// Number of silence samples to prepend.
    pub fn left_pad_samples(&self) -> usize {
        self.n_left_pad_tokens * self.samples_per_token()
    }
}

impl Default for PadConfig {
    fn default() -> Self {
        Self::q4()
    }
}

/// Compute the number of audio tokens for a given number of samples.
pub fn num_audio_tokens(num_samples: usize, config: &PadConfig) -> usize {
    let spt = config.samples_per_token();
    if spt == 0 {
        return 0;
    }
    (num_samples + spt - 1) / spt
}

/// Pad audio for Voxtral streaming inference.
///
/// - Left-pads with silence (76 tokens for Q4, 32 for BF16)
/// - Right-pads to align to token boundary + extra tokens for encoder
///
/// Returns a new AudioBuffer with padding applied.
pub fn pad_audio(audio: &AudioBuffer, config: &PadConfig) -> AudioBuffer {
    let spt = config.samples_per_token();
    let left_pad = config.left_pad_samples();

    // Total samples after left-padding
    let total_with_left = left_pad + audio.samples.len();

    // Align to token boundary
    let remainder = total_with_left % spt;
    let align_pad = if remainder > 0 { spt - remainder } else { 0 };

    // Extra right padding for encoder alignment
    let right_extra = config.extra_right_pad_tokens * spt;

    let total_len = total_with_left + align_pad + right_extra;

    let mut padded = vec![0.0f32; total_len];
    padded[left_pad..left_pad + audio.samples.len()].copy_from_slice(&audio.samples);

    AudioBuffer::new(padded, audio.sample_rate)
}

/// Right-only padding: align to token boundary + extra right tokens.
/// Assumes audio already contains left-padding (silence prepended by caller).
pub fn pad_audio_right_only(audio: &AudioBuffer, config: &PadConfig) -> AudioBuffer {
    let spt = config.samples_per_token();

    // Align to token boundary
    let remainder = audio.samples.len() % spt;
    let align_pad = if remainder > 0 { spt - remainder } else { 0 };

    // Extra right padding
    let right_extra = config.extra_right_pad_tokens * spt;
    let total_len = audio.samples.len() + align_pad + right_extra;

    let mut padded = vec![0.0f32; total_len];
    padded[..audio.samples.len()].copy_from_slice(&audio.samples);

    AudioBuffer::new(padded, audio.sample_rate)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q4_padding() {
        let config = PadConfig::q4();
        assert_eq!(config.samples_per_token(), 1280);
        assert_eq!(config.left_pad_samples(), 76 * 1280); // 97,280

        let audio = AudioBuffer::new(vec![1.0; 16000], 16000); // 1 second
        let padded = pad_audio(&audio, &config);

        // First 97,280 samples should be silence
        for &s in &padded.samples[..config.left_pad_samples()] {
            assert_eq!(s, 0.0);
        }
        // Original audio should follow
        assert_eq!(padded.samples[config.left_pad_samples()], 1.0);
    }

    #[test]
    fn test_bf16_padding() {
        let config = PadConfig::bf16();
        assert_eq!(config.left_pad_samples(), 32 * 1280); // 40,960
    }

    #[test]
    fn test_token_count() {
        let config = PadConfig::q4();
        assert_eq!(num_audio_tokens(16000, &config), 13); // 16000/1280 = 12.5 → 13
        assert_eq!(num_audio_tokens(1280, &config), 1);
    }
}
