//! Audio chunking for processing long audio files.
//!
//! Voxtral has a max_source_positions limit that constrains how many mel
//! frames can be processed at once. Long audio is split into chunks that
//! fit within this limit.

use super::AudioBuffer;

/// Chunking configuration.
#[derive(Debug, Clone)]
pub struct ChunkConfig {
    /// Maximum mel frames per chunk (default: 1500)
    pub max_mel_frames: usize,
    /// Hop length used by mel spectrogram (default: 160)
    pub hop_length: usize,
    /// Sample rate (default: 16000)
    pub sample_rate: u32,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            max_mel_frames: 1500,
            hop_length: 160,
            sample_rate: 16000,
        }
    }
}

impl ChunkConfig {
    /// Maximum samples per chunk.
    pub fn max_samples(&self) -> usize {
        self.max_mel_frames * self.hop_length
    }

    /// Maximum duration per chunk in seconds.
    pub fn max_duration_secs(&self) -> f32 {
        self.max_samples() as f32 / self.sample_rate as f32
    }
}

/// A single audio chunk with its position in the original audio.
#[derive(Debug, Clone)]
pub struct AudioChunk {
    pub samples: Vec<f32>,
    pub offset_samples: usize,
    pub chunk_index: usize,
    pub total_chunks: usize,
}

/// Check if audio needs chunking.
pub fn needs_chunking(num_samples: usize, config: &ChunkConfig) -> bool {
    num_samples > config.max_samples()
}

/// Split audio into chunks that fit within the mel frame limit.
pub fn chunk_audio(audio: &AudioBuffer, config: &ChunkConfig) -> Vec<AudioChunk> {
    let max_samples = config.max_samples();

    if audio.samples.len() <= max_samples {
        return vec![AudioChunk {
            samples: audio.samples.clone(),
            offset_samples: 0,
            chunk_index: 0,
            total_chunks: 1,
        }];
    }

    let total_chunks = (audio.samples.len() + max_samples - 1) / max_samples;
    let mut chunks = Vec::with_capacity(total_chunks);

    for i in 0..total_chunks {
        let start = i * max_samples;
        let end = (start + max_samples).min(audio.samples.len());

        chunks.push(AudioChunk {
            samples: audio.samples[start..end].to_vec(),
            offset_samples: start,
            chunk_index: i,
            total_chunks,
        });
    }

    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_config_defaults() {
        let config = ChunkConfig::default();
        assert_eq!(config.max_samples(), 240000); // 1500 * 160
        assert!((config.max_duration_secs() - 15.0).abs() < 0.01); // 240000 / 16000
    }

    #[test]
    fn test_no_chunking_needed() {
        let config = ChunkConfig::default();
        let audio = AudioBuffer::new(vec![0.0; 16000], 16000); // 1 second
        assert!(!needs_chunking(audio.len(), &config));

        let chunks = chunk_audio(&audio, &config);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].total_chunks, 1);
    }

    #[test]
    fn test_chunking() {
        let config = ChunkConfig::default();
        // 30 seconds of audio = 480000 samples, needs 2 chunks
        let audio = AudioBuffer::new(vec![0.0; 480000], 16000);
        assert!(needs_chunking(audio.len(), &config));

        let chunks = chunk_audio(&audio, &config);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].samples.len(), 240000);
        assert_eq!(chunks[1].samples.len(), 240000);
        assert_eq!(chunks[0].offset_samples, 0);
        assert_eq!(chunks[1].offset_samples, 240000);
    }
}
