pub mod chunk;
pub mod ffmpeg;
pub mod mel;
pub mod opus;
pub mod pad;
pub mod resample;

/// Mono audio buffer with sample rate metadata.
#[derive(Debug, Clone)]
pub struct AudioBuffer {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

impl AudioBuffer {
    pub fn new(samples: Vec<f32>, sample_rate: u32) -> Self {
        Self {
            samples,
            sample_rate,
        }
    }

    /// Load a WAV file into an AudioBuffer (mono, normalized to [-1.0, 1.0]).
    pub fn load_wav(path: &std::path::Path) -> Result<Self, Box<dyn std::error::Error>> {
        let reader = hound::WavReader::open(path)?;
        let spec = reader.spec();
        let sample_rate = spec.sample_rate;

        let samples: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Float => reader
                .into_samples::<f32>()
                .filter_map(|s| s.ok())
                .collect(),
            hound::SampleFormat::Int => {
                let max_val = (1u32 << (spec.bits_per_sample - 1)) as f32;
                reader
                    .into_samples::<i32>()
                    .filter_map(|s| s.ok())
                    .map(|s| s as f32 / max_val)
                    .collect()
            }
        };

        // Mix to mono if stereo
        let mono = if spec.channels > 1 {
            samples
                .chunks(spec.channels as usize)
                .map(|ch| ch.iter().sum::<f32>() / spec.channels as f32)
                .collect()
        } else {
            samples
        };

        Ok(Self::new(mono, sample_rate))
    }

    /// Peak normalize to target amplitude.
    pub fn peak_normalize(&mut self, target: f32) {
        let peak = self
            .samples
            .iter()
            .map(|s| s.abs())
            .fold(0.0f32, f32::max);
        if peak > 0.0 {
            let scale = target / peak;
            for s in &mut self.samples {
                *s *= scale;
            }
        }
    }

    /// Duration in seconds.
    pub fn duration_secs(&self) -> f32 {
        self.samples.len() as f32 / self.sample_rate as f32
    }

    /// Number of samples.
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}
