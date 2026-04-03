//! Mel-spectrogram computation for Voxtral audio preprocessing.
//!
//! Computes log mel spectrograms from audio samples using the Voxtral audio
//! input specifications (16kHz, 128 mel bins, hop=160, window=400).
//!
//! Normalization follows vLLM's Voxtral implementation:
//! 1. log10(mel) with floor at 1e-10
//! 2. Dynamic range limit using global log_mel_max (1.5 for Voxtral Realtime)
//! 3. Linear scale: (log_spec + 4.0) / 4.0

use rustfft::num_complex::Complex;
use rustfft::FftPlanner;
use std::f32::consts::PI;

/// Configuration for mel spectrogram computation.
#[derive(Debug, Clone)]
pub struct MelConfig {
    pub sample_rate: u32,
    pub n_fft: usize,
    pub hop_length: usize,
    pub win_length: Option<usize>,
    pub n_mels: usize,
    pub fmin: f32,
    pub fmax: Option<f32>,
    /// Global log mel maximum for normalization (default: 1.5 for Voxtral)
    pub log_mel_max: f32,
}

impl Default for MelConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            n_fft: 400,
            hop_length: 160,
            win_length: Some(400),
            n_mels: 128,
            fmin: 0.0,
            fmax: None,
            log_mel_max: 1.5,
        }
    }
}

/// Mel-spectrogram extractor.
pub struct MelSpectrogram {
    config: MelConfig,
    mel_basis: Vec<Vec<f32>>,
    window: Vec<f32>,
}

impl MelSpectrogram {
    pub fn new(config: MelConfig) -> Self {
        let win_length = config.win_length.unwrap_or(config.n_fft);
        let fmax = config.fmax.unwrap_or(config.sample_rate as f32 / 2.0);

        let mel_basis =
            create_mel_filterbank(config.sample_rate, config.n_fft, config.n_mels, config.fmin, fmax);
        let window = hann_window(win_length);

        Self {
            config,
            mel_basis,
            window,
        }
    }

    /// Number of mel frames for a given number of samples.
    pub fn num_frames(&self, num_samples: usize) -> usize {
        let pad_length = self.config.n_fft / 2;
        let padded_len = num_samples + 2 * pad_length;
        // Drop last frame to match Python reference: magnitudes = stft[..., :-1]
        (padded_len - self.config.n_fft) / self.config.hop_length
    }

    /// Compute log mel spectrogram with Voxtral normalization.
    ///
    /// Returns a 2D vector of shape `[n_frames, n_mels]` with values in roughly [-1, 1].
    pub fn compute_log(&self, samples: &[f32]) -> Vec<Vec<f32>> {
        let stft = self.stft(samples);

        // Power spectrogram
        let power_spec: Vec<Vec<f32>> = stft
            .iter()
            .map(|frame| frame.iter().map(|c| c.re * c.re + c.im * c.im).collect())
            .collect();

        // Apply mel filterbank
        let mel: Vec<Vec<f32>> = power_spec
            .iter()
            .map(|frame| {
                self.mel_basis
                    .iter()
                    .map(|filter| filter.iter().zip(frame.iter()).map(|(f, p)| f * p).sum())
                    .collect()
            })
            .collect();

        // Log10 with floor
        let mut log_mel: Vec<Vec<f32>> = mel
            .into_iter()
            .map(|frame| {
                frame
                    .into_iter()
                    .map(|v| v.max(1e-10).log10())
                    .collect()
            })
            .collect();

        // Dynamic range limit
        let log_spec_max = if self.config.log_mel_max > 0.0 {
            self.config.log_mel_max
        } else {
            log_mel
                .iter()
                .flat_map(|frame| frame.iter())
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max)
        };
        let min_val = log_spec_max - 8.0;

        for frame in &mut log_mel {
            for v in frame.iter_mut() {
                *v = v.max(min_val);
            }
        }

        // Linear scale to roughly [-1, 1]
        for frame in &mut log_mel {
            for v in frame.iter_mut() {
                *v = (*v + 4.0) / 4.0;
            }
        }

        log_mel
    }

    /// Compute log mel spectrogram as a flat vector (row-major: [n_frames * n_mels]).
    pub fn compute_log_flat(&self, samples: &[f32]) -> Vec<f32> {
        self.compute_log(samples).into_iter().flatten().collect()
    }

    /// Short-time Fourier transform with reflect padding and Hann window.
    fn stft(&self, samples: &[f32]) -> Vec<Vec<Complex<f32>>> {
        let n_fft = self.config.n_fft;
        let hop_length = self.config.hop_length;
        let win_length = self.window.len();

        // Reflect-pad signal (center=True, matching torch.stft)
        let pad_length = n_fft / 2;
        let mut padded = Vec::with_capacity(pad_length + samples.len() + pad_length);

        // Left reflect padding
        for i in (1..=pad_length).rev() {
            let idx = i.min(samples.len().saturating_sub(1));
            padded.push(samples.get(idx).copied().unwrap_or(0.0));
        }
        padded.extend_from_slice(samples);
        // Right reflect padding
        for i in 0..pad_length {
            let idx = samples.len().saturating_sub(2).saturating_sub(i);
            padded.push(samples.get(idx).copied().unwrap_or(0.0));
        }

        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(n_fft);

        // Drop last frame to match Python: magnitudes = stft[..., :-1]
        let n_frames = (padded.len() - n_fft) / hop_length;
        let mut result = Vec::with_capacity(n_frames);

        for i in 0..n_frames {
            let start = i * hop_length;

            let mut buffer: Vec<Complex<f32>> = (0..n_fft)
                .map(|j| {
                    let sample = if j < win_length && start + j < padded.len() {
                        padded[start + j] * self.window[j]
                    } else {
                        0.0
                    };
                    Complex::new(sample, 0.0)
                })
                .collect();

            fft.process(&mut buffer);

            // Positive frequencies only (n_fft/2 + 1)
            let frame: Vec<Complex<f32>> = buffer.iter().take(n_fft / 2 + 1).copied().collect();
            result.push(frame);
        }

        result
    }
}

/// Convert Hz to mel scale (Slaney / O'Shaughnessy).
fn hz_to_mel(f: f32) -> f32 {
    const F_SP: f32 = 200.0 / 3.0;
    const MIN_LOG_HZ: f32 = 1000.0;
    const MIN_LOG_MEL: f32 = MIN_LOG_HZ / F_SP; // 15.0
    const LOGSTEP: f32 = 0.068_751_74; // ln(6.4) / 27

    if f < MIN_LOG_HZ {
        f / F_SP
    } else {
        MIN_LOG_MEL + (f / MIN_LOG_HZ).ln() / LOGSTEP
    }
}

/// Convert mel to Hz (Slaney / O'Shaughnessy).
fn mel_to_hz(m: f32) -> f32 {
    const F_SP: f32 = 200.0 / 3.0;
    const MIN_LOG_HZ: f32 = 1000.0;
    const MIN_LOG_MEL: f32 = MIN_LOG_HZ / F_SP;
    const LOGSTEP: f32 = 0.068_751_74;

    if m < MIN_LOG_MEL {
        m * F_SP
    } else {
        MIN_LOG_HZ * ((m - MIN_LOG_MEL) * LOGSTEP).exp()
    }
}

/// Create Slaney-normalized triangular mel filterbank (matches librosa.filters.mel).
fn create_mel_filterbank(
    sample_rate: u32,
    n_fft: usize,
    n_mels: usize,
    fmin: f32,
    fmax: f32,
) -> Vec<Vec<f32>> {
    let n_freqs = n_fft / 2 + 1;

    let mel_min = hz_to_mel(fmin);
    let mel_max = hz_to_mel(fmax);
    let mel_points: Vec<f32> = (0..=n_mels + 1)
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
        .collect();

    let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();

    let fft_freqs: Vec<f32> = (0..n_freqs)
        .map(|i| i as f32 * sample_rate as f32 / n_fft as f32)
        .collect();

    let mut filterbank = vec![vec![0.0f32; n_freqs]; n_mels];

    for i in 0..n_mels {
        let f_lower = hz_points[i];
        let f_center = hz_points[i + 1];
        let f_upper = hz_points[i + 2];

        for (j, &freq) in fft_freqs.iter().enumerate() {
            if freq >= f_lower && freq <= f_center && f_center > f_lower {
                filterbank[i][j] = (freq - f_lower) / (f_center - f_lower);
            } else if freq > f_center && freq <= f_upper && f_upper > f_center {
                filterbank[i][j] = (f_upper - freq) / (f_upper - f_center);
            }
        }

        // Slaney area-normalization
        let band_width = hz_points[i + 2] - hz_points[i];
        if band_width > 0.0 {
            let enorm = 2.0 / band_width;
            for val in &mut filterbank[i] {
                *val *= enorm;
            }
        }
    }

    filterbank
}

/// Periodic Hann window (matches torch.hann_window default).
fn hann_window(length: usize) -> Vec<f32> {
    (0..length)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / length as f32).cos()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mel_config_defaults() {
        let config = MelConfig::default();
        assert_eq!(config.n_mels, 128);
        assert_eq!(config.n_fft, 400);
        assert_eq!(config.hop_length, 160);
        assert_eq!(config.sample_rate, 16000);
    }

    #[test]
    fn test_num_frames() {
        let mel = MelSpectrogram::new(MelConfig::default());
        // 1 second of 16kHz audio = 16000 samples
        // Padded: 16000 + 2*200 = 16400
        // Frames: (16400 - 400) / 160 = 100
        assert_eq!(mel.num_frames(16000), 100);
    }

    #[test]
    fn test_silence_produces_low_values() {
        let mel = MelSpectrogram::new(MelConfig::default());
        let silence = vec![0.0f32; 16000];
        let log_mel = mel.compute_log(&silence);
        // All values should be at the floor
        for frame in &log_mel {
            for &v in frame {
                assert!(v < 0.0, "Silence should produce negative log-mel values");
            }
        }
    }

    #[test]
    fn test_hz_mel_roundtrip() {
        for freq in [100.0, 500.0, 1000.0, 4000.0, 8000.0] {
            let mel = hz_to_mel(freq);
            let hz = mel_to_hz(mel);
            assert!(
                (hz - freq).abs() < 0.1,
                "Hz-Mel roundtrip failed for {}: got {}",
                freq,
                hz
            );
        }
    }
}
