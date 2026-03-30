//! Frame-by-frame streaming for Voxtral realtime transcription.
//!
//! Matches vllm's VoxtralRealtimeBuffer: processes audio in small overlapping
//! frames (~135ms each) through the full encoder, with decoder KV cache
//! persisting across frames.
//!
//! Frame parameters (from tekken.json audio config):
//!   frame_rate: 12.5 Hz (80ms per token)
//!   look_back: 52.5ms (840 samples)
//!   look_ahead: 2.5ms (40 samples)
//!   step_size: 1280 samples (80ms)
//!   window: 840 + 1280 + 40 = 2160 samples (135ms)

/// Streaming frame parameters for Voxtral realtime.
pub struct StreamConfig {
    pub sample_rate: usize,
    pub frame_rate: f32,
    pub step_samples: usize,
    pub look_back_samples: usize,
    pub look_ahead_samples: usize,
    pub n_left_pad_tokens: usize,
    pub audio_length_per_tok: usize,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            frame_rate: 12.5,
            step_samples: 1280,       // 80ms at 16kHz
            look_back_samples: 840,    // 52.5ms
            look_ahead_samples: 40,    // 2.5ms
            n_left_pad_tokens: 32,
            audio_length_per_tok: 8,
        }
    }
}

impl StreamConfig {
    pub fn window_samples(&self) -> usize {
        self.look_back_samples + self.step_samples + self.look_ahead_samples
    }
}

/// Frame specification: which audio to encode and how many tokens it produces.
pub struct FrameSpec {
    /// Start index in the accumulated audio buffer (after left-padding).
    pub audio_start: usize,
    /// End index (exclusive).
    pub audio_end: usize,
    /// Number of new decoder tokens this frame produces.
    pub num_tokens: usize,
}

/// Generates overlapping audio frames for streaming transcription.
/// Matches vllm's `_generate_frame_size_and_num_tokens`.
pub struct FrameGenerator {
    config: StreamConfig,
    /// Current position in the audio stream (in raw samples, excluding padding).
    /// This is the START of the next unprocessed frame's new audio region.
    cursor: usize,
    /// Initial end position (accounts for prefix tokens' audio).
    initial_end: usize,
    /// Left padding length in samples.
    left_pad_samples: usize,
}

impl FrameGenerator {
    pub fn new(config: StreamConfig) -> Self {
        // Initial end = n_left_pad_tokens * audio_length_per_tok samples at mel level
        // But we work in raw samples. audio_length_per_tok=8 means 8 samples at mel level.
        // At hop=160, 8 mel samples = 8 * 160 = 1280 raw samples per token.
        // Initial prefix: n_left_pad_tokens * step_samples
        let initial_end = config.n_left_pad_tokens * config.step_samples;
        let left_pad_samples = initial_end; // left padding covers the prefix
        Self {
            config,
            cursor: 0,
            initial_end,
            left_pad_samples,
        }
    }

    /// Generate frames for any new audio available.
    /// `total_audio_samples` is the length of accumulated audio (excluding left-padding).
    /// Returns frames that haven't been processed yet.
    pub fn next_frames(&mut self, total_audio_samples: usize) -> Vec<FrameSpec> {
        let mut frames = Vec::new();
        let cfg = &self.config;

        // The "end" pointer advances by step_samples each frame
        let mut start = self.cursor;
        let mut end = if start == 0 { self.initial_end } else { start + cfg.step_samples };

        // Total available (including left-padding space)
        let total_with_pad = self.left_pad_samples + total_audio_samples;

        while end + cfg.look_ahead_samples <= total_with_pad {
            let frame_start = if start > cfg.look_back_samples {
                start - cfg.look_back_samples
            } else {
                0
            };
            let frame_end = end + cfg.look_ahead_samples;

            // num_tokens for this frame
            let num_tokens = (end - start) / cfg.step_samples;

            frames.push(FrameSpec {
                audio_start: frame_start,
                audio_end: frame_end,
                num_tokens,
            });

            start = end;
            end = start + cfg.step_samples;
        }

        // Update cursor to where we stopped
        if !frames.is_empty() {
            self.cursor = start;
        }

        frames
    }

    /// Reset for session rotation.
    pub fn reset(&mut self) {
        self.cursor = 0;
    }

    /// Total frames emitted so far.
    pub fn frames_processed(&self) -> usize {
        if self.cursor == 0 { 0 } else {
            (self.cursor - self.initial_end) / self.config.step_samples + self.config.n_left_pad_tokens
        }
    }
}
