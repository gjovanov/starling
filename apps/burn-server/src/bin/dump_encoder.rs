//! Dump encoder adapter output to binary file for comparison with llama.cpp.
//! Usage: CANDLE_STREAMING=1 dump_encoder <wav_file> <output.bin>
//! Requires candle-native feature.

#[cfg(feature = "candle-native")]
fn main() {
    use burn_server::audio::mel::{MelConfig, MelSpectrogram};
    use burn_server::audio::pad::{pad_audio, PadConfig};
    use burn_server::audio::AudioBuffer;
    use burn_server::inference::candle_native::model::{self, VoxtralModel};
    use std::io::Write;

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: dump_encoder <wav_file> <output.bin>");
        std::process::exit(1);
    }

    let device = candle_core_native::Device::new_cuda(0).expect("CUDA");
    let st_path = std::path::PathBuf::from(
        std::env::var("MODELS_DIR").unwrap_or_else(|_| "models/cache".to_string())
    ).join("bf16/consolidated.safetensors");
    eprintln!("Loading model from {}", st_path.display());
    let vox_model = VoxtralModel::load(&st_path, &device).expect("model");

    // Load audio
    let reader = hound::WavReader::open(&args[1]).expect("WAV");
    let spec = reader.spec();
    let mut samples: Vec<f32> = if spec.bits_per_sample == 16 {
        reader.into_samples::<i16>().map(|s| s.unwrap() as f32 / 32768.0).collect()
    } else {
        reader.into_samples::<f32>().map(|s| s.unwrap()).collect()
    };
    // Truncate to 15s
    let max = 15 * spec.sample_rate as usize;
    if samples.len() > max { samples.truncate(max); }
    eprintln!("Audio: {} samples, {:.1}s", samples.len(), samples.len() as f32 / spec.sample_rate as f32);

    // Pad with silence (matching streaming engine)
    let pad_config = PadConfig::bf16();
    let audio_buf = AudioBuffer::new(samples, spec.sample_rate);
    let padded = pad_audio(&audio_buf, &pad_config);
    eprintln!("Padded: {} samples", padded.samples.len());

    // Compute mel
    let mel_spec = MelSpectrogram::new(MelConfig::default());
    let log_mel = mel_spec.compute_log(&padded.samples);
    let n_frames = log_mel.len();
    let n_mels = if n_frames > 0 { log_mel[0].len() } else { 128 };
    eprintln!("Mel: {} frames x {} mels", n_frames, n_mels);

    // Build mel tensor [1, n_mels, n_frames]
    let mut flat = vec![0.0f32; n_mels * n_frames];
    for (t, frame) in log_mel.iter().enumerate() {
        for (m, &val) in frame.iter().enumerate() {
            flat[m * n_frames + t] = val;
        }
    }
    let mel = candle_core_native::Tensor::new(flat, &device)
        .and_then(|t| t.to_dtype(candle_core_native::DType::BF16))
        .and_then(|t| t.reshape((1, n_mels, n_frames)))
        .expect("mel tensor");

    // Encode: conv → encoder layers → norm → 4x reshape → adapter
    let mut enc_caches = VoxtralModel::new_encoder_caches();
    let adapter_out = vox_model.encode_audio(&mel, &mut enc_caches).expect("encode");

    let seq_len = adapter_out.dim(1).expect("dim1");
    let d_model = adapter_out.dim(2).expect("dim2");
    eprintln!("Adapter output: [{}, {}]", seq_len, d_model);

    // Dump to binary
    let adapter_f32: Vec<f32> = adapter_out
        .to_dtype(candle_core_native::DType::F32).expect("f32")
        .flatten_all().expect("flatten")
        .to_vec1().expect("vec");

    let mut out = std::fs::File::create(&args[2]).expect("create");
    out.write_all(&(seq_len as u32).to_le_bytes()).unwrap();
    out.write_all(&(d_model as u32).to_le_bytes()).unwrap();
    for v in &adapter_f32 {
        out.write_all(&v.to_le_bytes()).unwrap();
    }

    // Print stats
    let min = adapter_f32.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_v = adapter_f32.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean = adapter_f32.iter().sum::<f32>() / adapter_f32.len() as f32;
    eprintln!("Stats: min={:.4} max={:.4} mean={:.6}", min, max_v, mean);
    eprintln!("Token 0, dim 0..4: [{:.4}, {:.4}, {:.4}, {:.4}]",
        adapter_f32[0], adapter_f32[1], adapter_f32[2], adapter_f32[3]);
    eprintln!("Token 39, dim 0..4: [{:.4}, {:.4}, {:.4}, {:.4}]",
        adapter_f32[39*d_model], adapter_f32[39*d_model+1], adapter_f32[39*d_model+2], adapter_f32[39*d_model+3]);
    eprintln!("Saved to {}", args[2]);
}

#[cfg(not(feature = "candle-native"))]
fn main() {
    eprintln!("Requires --features candle-native");
    std::process::exit(1);
}
