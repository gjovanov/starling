//! Dump encoder intermediate stages for comparison with llama.cpp.
//! Dumps: mel → conv → layer0 → layer31 → adapter
//! Usage: dump_encoder_stages <wav_file>

#[cfg(feature = "candle-native")]
fn main() {
    use burn_server::audio::mel::{MelConfig, MelSpectrogram};
    use burn_server::audio::pad::{pad_audio, PadConfig};
    use burn_server::audio::AudioBuffer;
    use burn_server::inference::candle_native::model::{self, VoxtralModel, KVCache};
    use std::io::Write;

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: dump_encoder_stages <wav_file>");
        std::process::exit(1);
    }

    let device = candle_core_native::Device::new_cuda(0).expect("CUDA");
    let st_path = std::path::PathBuf::from(
        std::env::var("MODELS_DIR").unwrap_or_else(|_| "models/cache".to_string())
    ).join("bf16/consolidated.safetensors");
    let vox_model = VoxtralModel::load(&st_path, &device).expect("model");

    // Load and pad audio
    let reader = hound::WavReader::open(&args[1]).expect("WAV");
    let spec = reader.spec();
    let mut samples: Vec<f32> = if spec.bits_per_sample == 16 {
        reader.into_samples::<i16>().map(|s| s.unwrap() as f32 / 32768.0).collect()
    } else {
        reader.into_samples::<f32>().map(|s| s.unwrap()).collect()
    };
    let max = 15 * spec.sample_rate as usize;
    if samples.len() > max { samples.truncate(max); }

    let pad_config = PadConfig::bf16();
    let audio_buf = AudioBuffer::new(samples, spec.sample_rate);
    let padded = pad_audio(&audio_buf, &pad_config);
    eprintln!("Padded: {} samples", padded.samples.len());

    // Compute mel
    let mel_spec = MelSpectrogram::new(MelConfig::default());
    let log_mel = mel_spec.compute_log(&padded.samples);
    let n_frames = log_mel.len();
    let n_mels = 128;
    eprintln!("Mel: {} frames x {} mels", n_frames, n_mels);

    let mut flat = vec![0.0f32; n_mels * n_frames];
    for (t, frame) in log_mel.iter().enumerate() {
        for (m, &val) in frame.iter().enumerate() {
            flat[m * n_frames + t] = val;
        }
    }
    // Dump mel for comparison
    {
        let mut f = std::fs::File::create("/tmp/candle_mel.bin").expect("create mel");
        f.write_all(&(n_mels as u32).to_le_bytes()).unwrap();
        f.write_all(&(n_frames as u32).to_le_bytes()).unwrap();
        for v in &flat { f.write_all(&v.to_le_bytes()).unwrap(); }
        eprintln!("Saved mel to /tmp/candle_mel.bin ({} floats)", flat.len());
    }

    let mel = candle_core_native::Tensor::new(flat, &device)
        .and_then(|t| t.to_dtype(candle_core_native::DType::BF16))
        .and_then(|t| t.reshape((1, n_mels, n_frames)))
        .expect("mel tensor");

    // Stage 1: Conv
    let conv_out = vox_model.encoder.conv.forward(&mel).expect("conv");
    let conv_f32: Vec<f32> = conv_out.to_dtype(candle_core_native::DType::F32)
        .and_then(|t| t.flatten_all()).and_then(|t| t.to_vec1()).expect("conv f32");
    let conv_seq = conv_out.dim(1).expect("dim1");
    let conv_dim = conv_out.dim(2).expect("dim2");
    eprintln!("Conv: [{}, {}]", conv_seq, conv_dim);
    eprintln!("  first4: [{:.4}, {:.4}, {:.4}, {:.4}]", conv_f32[0], conv_f32[1], conv_f32[2], conv_f32[3]);

    // Stage 2: Encoder layer 0
    let n_aligned = (conv_seq / 4) * 4;
    let x = if n_aligned < conv_seq {
        conv_out.narrow(1, 0, n_aligned).expect("narrow")
    } else { conv_out };

    let mut enc_caches: Vec<KVCache> = (0..32).map(|_| KVCache::new()).collect();
    let mut layer0_out = vox_model.encoder.layers[0]
        .forward(&x, &vox_model.encoder.rope, &mut enc_caches[0]).expect("layer0");
    let l0_f32: Vec<f32> = layer0_out.to_dtype(candle_core_native::DType::F32)
        .and_then(|t| t.flatten_all()).and_then(|t| t.to_vec1()).expect("l0 f32");
    let l0_seq = layer0_out.dim(1).expect("dim1");
    eprintln!("Layer 0: [{}, {}]", l0_seq, conv_dim);
    eprintln!("  first4: [{:.4}, {:.4}, {:.4}, {:.4}]", l0_f32[0], l0_f32[1], l0_f32[2], l0_f32[3]);

    // Stage 3: All remaining layers
    let mut cur = layer0_out;
    for i in 1..32 {
        cur = vox_model.encoder.layers[i]
            .forward(&cur, &vox_model.encoder.rope, &mut enc_caches[i]).expect("layer");
    }
    use candle_core_native::Module;
    let enc_norm = vox_model.encoder.norm.forward(&cur).expect("norm");
    let norm_f32: Vec<f32> = enc_norm.to_dtype(candle_core_native::DType::F32)
        .and_then(|t| t.flatten_all()).and_then(|t| t.to_vec1()).expect("norm f32");
    eprintln!("Encoder out (after norm): first4: [{:.4}, {:.4}, {:.4}, {:.4}]",
        norm_f32[0], norm_f32[1], norm_f32[2], norm_f32[3]);

    // Stage 4: Adapter (4x reshape + projection)
    let reshaped = model::reshape_encoder_output(&enc_norm, 4).expect("reshape");
    let adapter = vox_model.adapter.forward(&reshaped).expect("adapter");
    let ada_f32: Vec<f32> = adapter.to_dtype(candle_core_native::DType::F32)
        .and_then(|t| t.flatten_all()).and_then(|t| t.to_vec1()).expect("ada f32");
    let ada_seq = adapter.dim(1).expect("dim1");
    let ada_dim = adapter.dim(2).expect("dim2");
    eprintln!("Adapter: [{}, {}]", ada_seq, ada_dim);
    eprintln!("  token0: [{:.4}, {:.4}, {:.4}, {:.4}]", ada_f32[0], ada_f32[1], ada_f32[2], ada_f32[3]);
    eprintln!("  token39: [{:.4}, {:.4}, {:.4}, {:.4}]",
        ada_f32[39*ada_dim], ada_f32[39*ada_dim+1], ada_f32[39*ada_dim+2], ada_f32[39*ada_dim+3]);

    // Dump conv output for comparison
    let mut f = std::fs::File::create("/tmp/candle_conv.bin").expect("create");
    f.write_all(&(conv_seq as u32).to_le_bytes()).unwrap();
    f.write_all(&(conv_dim as u32).to_le_bytes()).unwrap();
    for v in &conv_f32 { f.write_all(&v.to_le_bytes()).unwrap(); }
    eprintln!("Saved conv to /tmp/candle_conv.bin");

    // Dump adapter output for comparison
    let mut f = std::fs::File::create("/tmp/candle_adapter.bin").expect("create");
    f.write_all(&(ada_seq as u32).to_le_bytes()).unwrap();
    f.write_all(&(ada_dim as u32).to_le_bytes()).unwrap();
    for v in &ada_f32 { f.write_all(&v.to_le_bytes()).unwrap(); }
    eprintln!("Saved adapter to /tmp/candle_adapter.bin");
}

#[cfg(not(feature = "candle-native"))]
fn main() { eprintln!("Requires --features candle-native"); std::process::exit(1); }
