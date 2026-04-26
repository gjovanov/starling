//! Dump mel spectrogram to binary file for comparison with llama.cpp.
//! Usage: dump_mel <wav_file> <output.bin>

use burn_server::audio::mel::{MelConfig, MelSpectrogram};
use std::io::Write;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: dump_mel <wav_file> <output.bin>");
        std::process::exit(1);
    }

    let reader = hound::WavReader::open(&args[1]).expect("WAV");
    let spec = reader.spec();
    let mut samples: Vec<f32> = if spec.bits_per_sample == 16 {
        reader.into_samples::<i16>().map(|s| s.unwrap() as f32 / 32768.0).collect()
    } else {
        reader.into_samples::<f32>().map(|s| s.unwrap()).collect()
    };

    // Truncate to 5s for quick comparison
    let max_samples = 5 * spec.sample_rate as usize;
    if samples.len() > max_samples {
        samples.truncate(max_samples);
    }
    eprintln!("Audio: {} samples, {}Hz, {:.1}s", samples.len(), spec.sample_rate, samples.len() as f32 / spec.sample_rate as f32);

    let mel_spec = MelSpectrogram::new(MelConfig::default());
    let log_mel = mel_spec.compute_log(&samples);
    let n_frames = log_mel.len();
    let n_mels = if n_frames > 0 { log_mel[0].len() } else { 128 };

    eprintln!("Mel: {} frames x {} mels", n_frames, n_mels);

    // Dump as binary: header(n_mels:u32, n_frames:u32) + data[n_mels][n_frames] f32
    let mut out = std::fs::File::create(&args[2]).expect("create output");
    out.write_all(&(n_mels as u32).to_le_bytes()).unwrap();
    out.write_all(&(n_frames as u32).to_le_bytes()).unwrap();
    // Write in [n_mels, n_frames] layout (mel-major, matching llama.cpp)
    for m in 0..n_mels {
        for t in 0..n_frames {
            out.write_all(&log_mel[t][m].to_le_bytes()).unwrap();
        }
    }

    // Print stats
    let all_vals: Vec<f32> = log_mel.iter().flat_map(|f| f.iter().copied()).collect();
    let min = all_vals.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = all_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean = all_vals.iter().sum::<f32>() / all_vals.len() as f32;
    eprintln!("Stats: min={:.4} max={:.4} mean={:.4}", min, max, mean);
    eprintln!("Frame 0, mel 0..4: [{:.4}, {:.4}, {:.4}, {:.4}]",
        log_mel[0][0], log_mel[0][1], log_mel[0][2], log_mel[0][3]);
    eprintln!("Frame 100, mel 0..4: [{:.4}, {:.4}, {:.4}, {:.4}]",
        log_mel[100.min(n_frames-1)][0], log_mel[100.min(n_frames-1)][1],
        log_mel[100.min(n_frames-1)][2], log_mel[100.min(n_frames-1)][3]);
    eprintln!("Saved to {}", args[2]);
}
