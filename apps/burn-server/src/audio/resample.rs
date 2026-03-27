//! Audio resampling to 16kHz mono using rubato.

use super::AudioBuffer;
use rubato::{Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction};

/// Resample an AudioBuffer to 16kHz. Returns unchanged if already 16kHz.
pub fn resample_to_16k(audio: &AudioBuffer) -> Result<AudioBuffer, Box<dyn std::error::Error>> {
    if audio.sample_rate == 16000 {
        return Ok(audio.clone());
    }
    resample(audio, 16000)
}

/// Resample an AudioBuffer to a target sample rate.
pub fn resample(
    audio: &AudioBuffer,
    target_rate: u32,
) -> Result<AudioBuffer, Box<dyn std::error::Error>> {
    if audio.sample_rate == target_rate {
        return Ok(audio.clone());
    }

    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };

    let ratio = target_rate as f64 / audio.sample_rate as f64;
    let chunk_size = 1024;

    let mut resampler = SincFixedIn::<f64>::new(
        ratio,
        2.0, // max relative ratio
        params,
        chunk_size,
        1, // mono
    )?;

    let samples_f64: Vec<f64> = audio.samples.iter().map(|&s| s as f64).collect();

    let mut output = Vec::with_capacity((samples_f64.len() as f64 * ratio) as usize + chunk_size);

    // Process in chunks
    let mut pos = 0;
    while pos + chunk_size <= samples_f64.len() {
        let chunk = vec![&samples_f64[pos..pos + chunk_size]];
        let result = resampler.process(&chunk, None)?;
        output.extend(result[0].iter().map(|&s| s as f32));
        pos += chunk_size;
    }

    // Process remaining samples (pad with zeros)
    if pos < samples_f64.len() {
        let mut last_chunk = vec![0.0f64; chunk_size];
        let remaining = &samples_f64[pos..];
        last_chunk[..remaining.len()].copy_from_slice(remaining);

        let chunk = vec![last_chunk.as_slice()];
        let result = resampler.process(&chunk, None)?;

        // Only take proportional output
        let expected = ((remaining.len() as f64) * ratio).ceil() as usize;
        output.extend(result[0].iter().take(expected).map(|&s| s as f32));
    }

    Ok(AudioBuffer::new(output, target_rate))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resample_noop() {
        let buf = AudioBuffer::new(vec![0.0; 16000], 16000);
        let result = resample_to_16k(&buf).unwrap();
        assert_eq!(result.sample_rate, 16000);
        assert_eq!(result.len(), 16000);
    }

    #[test]
    fn test_resample_downsample() {
        // 48kHz to 16kHz should produce ~1/3 the samples
        let buf = AudioBuffer::new(vec![0.0; 48000], 48000);
        let result = resample_to_16k(&buf).unwrap();
        assert_eq!(result.sample_rate, 16000);
        // Allow some tolerance for edge effects
        assert!(result.len() > 15000 && result.len() < 17000);
    }
}
