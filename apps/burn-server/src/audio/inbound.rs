//! Inbound WebRTC audio pipeline: Opus RTP → PCM 16 kHz mono f32.
//!
//! The browser sends Opus at 48 kHz. We decode each packet to i16 PCM
//! and then downsample to 16 kHz using 3:1 box averaging (cheap, latency-free,
//! adequate for speech ASR where content is below 8 kHz anyway).

/// Downsample 48 kHz f32 samples to 16 kHz using 3-sample box averaging.
///
/// Any trailing 1-2 samples are dropped (caller should accumulate them and
/// prepend on the next call if sample-accurate streaming is required).
pub fn downsample_48k_to_16k(input: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(input.len() / 3);
    for chunk in input.chunks_exact(3) {
        out.push((chunk[0] + chunk[1] + chunk[2]) / 3.0);
    }
    out
}

/// Stateful downsampler that preserves sub-triplet samples across calls.
pub struct Downsampler48to16 {
    carry: Vec<f32>,
}

impl Downsampler48to16 {
    pub fn new() -> Self {
        Self { carry: Vec::with_capacity(2) }
    }

    /// Feed more 48 kHz samples, emit corresponding 16 kHz samples.
    pub fn feed(&mut self, input: &[f32]) -> Vec<f32> {
        let mut buf = std::mem::take(&mut self.carry);
        buf.extend_from_slice(input);

        let trip_count = buf.len() / 3;
        let consumed = trip_count * 3;

        let mut out = Vec::with_capacity(trip_count);
        for chunk in buf[..consumed].chunks_exact(3) {
            out.push((chunk[0] + chunk[1] + chunk[2]) / 3.0);
        }

        self.carry = buf[consumed..].to_vec();
        out
    }
}

impl Default for Downsampler48to16 {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_downsample_basic() {
        let input: Vec<f32> = (0..9).map(|i| i as f32).collect();
        let out = downsample_48k_to_16k(&input);
        assert_eq!(out.len(), 3);
        assert!((out[0] - 1.0).abs() < 1e-6); // (0+1+2)/3
        assert!((out[1] - 4.0).abs() < 1e-6); // (3+4+5)/3
        assert!((out[2] - 7.0).abs() < 1e-6); // (6+7+8)/3
    }

    #[test]
    fn test_stateful_carry() {
        let mut ds = Downsampler48to16::new();

        // 4 samples: yields 1 output (triplet 0,1,2), carry [3]
        let a = ds.feed(&[0.0, 1.0, 2.0, 3.0]);
        assert_eq!(a.len(), 1);
        assert!((a[0] - 1.0).abs() < 1e-6);

        // 2 more samples: [3,4,5] → 4.0, no carry
        let b = ds.feed(&[4.0, 5.0]);
        assert_eq!(b.len(), 1);
        assert!((b[0] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_stateful_preserves_sample_count() {
        // Feeding 48000 samples in random-sized chunks should yield 16000 output
        let mut ds = Downsampler48to16::new();
        let mut total_out = 0;
        let chunk_sizes = [960usize, 960, 960, 1, 2, 100, 44_017]; // = 48000 total
        let chunks: Vec<Vec<f32>> = chunk_sizes
            .iter()
            .map(|n| vec![0.5f32; *n])
            .collect();

        for chunk in &chunks {
            let out = ds.feed(chunk);
            total_out += out.len();
        }
        assert_eq!(total_out, 16000);
    }
}
