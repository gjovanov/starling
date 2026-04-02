//! Benchmark binary for BF16 inference: WGPU vs CUDA.
//!
//! Usage:
//!   # WGPU (DZN):
//!   cargo run --release --bin benchmark -- --audio ../../media/broadcast_30s.wav
//!
//!   # CUDA:
//!   cargo run --release --features cuda --bin benchmark -- --backend cuda --audio ../../media/broadcast_30s.wav

use burn_server::audio::mel::{MelConfig, MelSpectrogram};
use burn_server::audio::pad::{pad_audio, PadConfig};
use burn_server::audio::AudioBuffer;
use burn_server::inference::bf16::weights::OwnedSafeTensors;
use burn_server::inference::tokenizer::TekkenDecoder;

use burn::tensor::{Tensor, TensorData};
use clap::Parser;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "benchmark", about = "BF16 inference benchmark")]
struct Args {
    /// GPU backend: wgpu or cuda
    #[arg(long, default_value = "wgpu")]
    backend: String,

    /// Path to audio file (WAV, 16kHz mono)
    #[arg(long)]
    audio: PathBuf,

    /// Path to models directory
    #[arg(long, default_value = "../../models/cache")]
    models_dir: PathBuf,

    /// Maximum decode steps (0 = no limit)
    #[arg(long, default_value = "0")]
    max_steps: usize,

    /// Maximum audio duration in seconds (0 = full file)
    #[arg(long, default_value = "0")]
    duration: usize,
}

fn compute_time_embedding<B: burn::tensor::backend::Backend>(
    t: f32, dim: usize, device: &B::Device,
) -> Tensor<B, 3> {
    let half_dim = dim / 2;
    let log_theta = 10000.0f32.ln();
    let mut embedding = Vec::with_capacity(dim);
    for i in 0..half_dim {
        let freq = (-log_theta * (i as f32) / (half_dim as f32)).exp();
        embedding.push((t * freq).cos());
    }
    for i in 0..half_dim {
        let freq = (-log_theta * (i as f32) / (half_dim as f32)).exp();
        embedding.push((t * freq).sin());
    }
    Tensor::from_data(TensorData::new(embedding, [1, 1, dim]), device)
}

fn prepare_mel(audio_path: &std::path::Path, max_duration_secs: usize) -> (Vec<f32>, usize, usize) {
    let reader = hound::WavReader::open(audio_path).expect("Failed to open WAV");
    let spec = reader.spec();
    let mut samples: Vec<f32> = if spec.bits_per_sample == 16 {
        reader.into_samples::<i16>()
            .map(|s| s.unwrap() as f32 / 32768.0)
            .collect()
    } else {
        reader.into_samples::<f32>()
            .map(|s| s.unwrap())
            .collect()
    };

    if max_duration_secs > 0 {
        let max_samples = max_duration_secs * spec.sample_rate as usize;
        if samples.len() > max_samples {
            samples.truncate(max_samples);
        }
    }

    let audio_secs = samples.len() as f32 / spec.sample_rate as f32;
    eprintln!("Audio: {:.1}s ({} samples, {} Hz)", audio_secs, samples.len(), spec.sample_rate);

    let audio_buf = AudioBuffer::new(samples, spec.sample_rate);
    let pad_config = PadConfig::bf16();
    let padded = pad_audio(&audio_buf, &pad_config);

    let mel_spec = MelSpectrogram::new(MelConfig::default());
    let log_mel = mel_spec.compute_log(&padded.samples);
    let n_frames = log_mel.len();
    let n_mels = if n_frames > 0 { log_mel[0].len() } else { 128 };

    let mut flat = vec![0.0f32; n_mels * n_frames];
    for (t, frame) in log_mel.iter().enumerate() {
        for (m, &val) in frame.iter().enumerate() {
            flat[m * n_frames + t] = val;
        }
    }
    (flat, n_mels, n_frames)
}

/// Prepare mel without silence padding (matching vllm's raw audio processing).
fn prepare_mel_raw(audio_path: &std::path::Path, max_duration_secs: usize) -> (Vec<f32>, usize, usize) {
    let reader = hound::WavReader::open(audio_path).expect("Failed to open WAV");
    let spec = reader.spec();
    let mut samples: Vec<f32> = if spec.bits_per_sample == 16 {
        reader.into_samples::<i16>()
            .map(|s| s.unwrap() as f32 / 32768.0)
            .collect()
    } else {
        reader.into_samples::<f32>()
            .map(|s| s.unwrap())
            .collect()
    };

    if max_duration_secs > 0 {
        let max_samples = max_duration_secs * spec.sample_rate as usize;
        if samples.len() > max_samples {
            samples.truncate(max_samples);
            eprintln!("Limiting to {}s ({} samples)", max_duration_secs, max_samples);
        }
    }

    let audio_secs = samples.len() as f32 / spec.sample_rate as f32;
    eprintln!("Audio: {:.1}s ({} samples, {} Hz)", audio_secs, samples.len(), spec.sample_rate);

    // No padding — compute mel directly from raw audio (matching vllm)
    let mel_spec = MelSpectrogram::new(MelConfig::default());
    let log_mel = mel_spec.compute_log(&samples);
    let n_frames = log_mel.len();
    let n_mels = if n_frames > 0 { log_mel[0].len() } else { 128 };

    let mut flat = vec![0.0f32; n_mels * n_frames];
    for (t, frame) in log_mel.iter().enumerate() {
        for (m, &val) in frame.iter().enumerate() {
            flat[m * n_frames + t] = val;
        }
    }
    (flat, n_mels, n_frames)
}

fn run_wgpu(args: &Args) {
    use burn_server::inference::q4::WgpuBackend;

    let device = burn::backend::wgpu::WgpuDevice::default();
    let st_path = args.models_dir.join("bf16").join("consolidated.safetensors");
    let tokenizer_path = args.models_dir.join("tokenizer").join("tekken.json");

    eprintln!("\n=== WGPU/DZN Benchmark ===");

    let t_load = Instant::now();
    let owned = OwnedSafeTensors::from_file(&st_path).expect("SafeTensors");
    let (encoder, adapter) =
        burn_server::inference::bf16::loader::load_encoder_and_adapter::<WgpuBackend>(&owned, &device)
            .expect("encoder");
    eprintln!("Encoder loaded: {:.1}s", t_load.elapsed().as_secs_f32());

    let tokenizer = TekkenDecoder::from_file(&tokenizer_path).expect("tokenizer");

    let (flat, n_mels, n_frames) = prepare_mel(&args.audio, args.duration);
    let mel: Tensor<WgpuBackend, 3> =
        Tensor::from_data(TensorData::new(flat, [1, n_mels, n_frames]), &device);
    let t_embed = compute_time_embedding::<WgpuBackend>(6.0, 3072, &device);

    let t_infer = Instant::now();
    let token_ids = burn_server::inference::bf16::loader::transcribe::<WgpuBackend>(
        &owned, &encoder, &adapter, &device, mel, t_embed,
    ).expect("transcribe");
    let infer_secs = t_infer.elapsed().as_secs_f32();

    let text = tokenizer.decode(&token_ids);
    let audio_secs = n_frames as f32 / 200.0; // 200 frames per second (before conv)

    eprintln!("\n=== WGPU Results ===");
    eprintln!("Load:       {:.1}s", t_load.elapsed().as_secs_f32());
    eprintln!("Inference:  {:.1}s", infer_secs);
    eprintln!("Tokens:     {}", token_ids.len());
    eprintln!("Realtime:   {:.1}×", infer_secs / audio_secs);
    eprintln!("Text (200c): {:?}", &text[..text.char_indices().take(200).last().map_or(0, |(i, c)| i + c.len_utf8())]);
    println!("{}", text);
}

#[cfg(feature = "cuda")]
fn run_cuda(args: &Args) {
    use burn_server::inference::q4::CudaBackend;

    let device = burn::backend::cuda::CudaDevice::default();
    type CudaDevice = burn::backend::cuda::CudaDevice;
    let st_path = args.models_dir.join("bf16").join("consolidated.safetensors");
    let tokenizer_path = args.models_dir.join("tokenizer").join("tekken.json");

    eprintln!("\n=== CUDA Benchmark ===");

    let t_load = Instant::now();
    let owned = OwnedSafeTensors::from_file(&st_path).expect("SafeTensors");

    fn cuda_sync() {
        use cubecl::Runtime;
        let device = cubecl::cuda::CudaDevice::default();
        let client = cubecl::cuda::CudaRuntime::client(&device);
        client.flush();
        std::thread::sleep(std::time::Duration::from_millis(5));
    }

    let model = burn_server::inference::bf16::loader::load_full_model::<CudaBackend>(
        &owned, &device, cuda_sync,
    ).expect("full model");
    let load_secs = t_load.elapsed().as_secs_f32();
    eprintln!("Full model loaded: {:.1}s", load_secs);

    let tokenizer = TekkenDecoder::from_file(&tokenizer_path).expect("tokenizer");

    let (flat, n_mels, n_frames) = prepare_mel(&args.audio, args.duration);
    let mel: Tensor<CudaBackend, 3> =
        Tensor::from_data(TensorData::new(flat, [1, n_mels, n_frames]), &device);
    let t_embed = compute_time_embedding::<CudaBackend>(6.0, 3072, &device);

    let t_infer = Instant::now();
    let token_ids = burn_server::inference::bf16::loader::transcribe_resident::<CudaBackend>(
        &model, &device, mel, t_embed,
    ).expect("transcribe_resident");
    let infer_secs = t_infer.elapsed().as_secs_f32();

    let text = tokenizer.decode(&token_ids);
    let audio_secs = n_frames as f32 / 200.0;

    eprintln!("\n=== CUDA Results ===");
    eprintln!("Load:       {:.1}s", load_secs);
    eprintln!("Inference:  {:.1}s", infer_secs);
    eprintln!("Tokens:     {}", token_ids.len());
    eprintln!("Realtime:   {:.1}×", infer_secs / audio_secs);
    eprintln!("Text (200c): {:?}", &text[..text.char_indices().take(200).last().map_or(0, |(i, c)| i + c.len_utf8())]);
    println!("{}", text);
}

#[cfg(feature = "candle")]
fn run_candle(args: &Args) {
    use burn_server::inference::q4::CandleBackend;

    let device = burn::backend::candle::CandleDevice::cuda(0);
    let st_path = args.models_dir.join("bf16").join("consolidated.safetensors");
    let tokenizer_path = args.models_dir.join("tokenizer").join("tekken.json");

    eprintln!("\n=== Candle CUDA (bf16 + cuBLAS) Benchmark ===");

    let t_load = Instant::now();
    let owned = OwnedSafeTensors::from_file(&st_path).expect("SafeTensors");

    fn candle_sync() {
        std::thread::sleep(std::time::Duration::from_millis(1));
    }

    let model = burn_server::inference::bf16::loader::load_full_model::<CandleBackend>(
        &owned, &device, candle_sync,
    ).expect("full model");
    let load_secs = t_load.elapsed().as_secs_f32();
    eprintln!("Full model loaded: {:.1}s", load_secs);

    let tokenizer = TekkenDecoder::from_file(&tokenizer_path).expect("tokenizer");

    let (flat, n_mels, n_frames) = prepare_mel(&args.audio, args.duration);
    let mel: Tensor<CandleBackend, 3> =
        Tensor::from_data(TensorData::new(flat, [1, n_mels, n_frames]), &device);
    let t_embed = compute_time_embedding::<CandleBackend>(6.0, 3072, &device);

    let t_infer = Instant::now();
    let token_ids = burn_server::inference::bf16::loader::transcribe_resident::<CandleBackend>(
        &model, &device, mel, t_embed,
    ).expect("transcribe_resident");
    let infer_secs = t_infer.elapsed().as_secs_f32();

    let text = tokenizer.decode(&token_ids);
    let audio_secs = n_frames as f32 / 200.0;

    eprintln!("\n=== Candle Results ===");
    eprintln!("Load:       {:.1}s", load_secs);
    eprintln!("Inference:  {:.1}s", infer_secs);
    eprintln!("Tokens:     {}", token_ids.len());
    eprintln!("Realtime:   {:.1}×", infer_secs / audio_secs);
    eprintln!("Text (200c): {:?}", &text[..text.char_indices().take(200).last().map_or(0, |(i, c)| i + c.len_utf8())]);
    println!("{}", text);
}

#[cfg(feature = "candle-native")]
fn run_candle_native(args: &Args) {
    use burn_server::inference::candle_native::model;

    let device = if std::env::var("CANDLE_CPU").is_ok() {
        eprintln!("Using CPU device (matching voxtral.c)");
        candle_core_native::Device::Cpu
    } else {
        candle_core_native::Device::new_cuda(0).expect("CUDA device")
    };
    let st_path = args.models_dir.join("bf16").join("consolidated.safetensors");
    let tokenizer_path = args.models_dir.join("tokenizer").join("tekken.json");

    eprintln!("\n=== CandleNative (bf16 + FlashAttention v2) Benchmark ===");

    let t_load = Instant::now();
    let vox_model = model::VoxtralModel::load(&st_path, &device).expect("model load");
    let load_secs = t_load.elapsed().as_secs_f32();
    eprintln!("Full model loaded: {:.1}s", load_secs);

    let tokenizer = TekkenDecoder::from_file(&tokenizer_path).expect("tokenizer");

    let mel_dtype = if std::env::var("CANDLE_NATIVE_F32").is_ok() {
        candle_core_native::DType::F32
    } else {
        candle_core_native::DType::BF16
    };

    // Check for external mel binary (from Python/vllm validation)
    let mel_override = std::env::var("CANDLE_MEL_FILE").ok();
    let (n_mels, n_frames, mel) = if let Some(mel_path) = &mel_override {
        let mel_bytes = std::fs::read(mel_path).expect("read mel file");
        let n_floats = mel_bytes.len() / 4;
        let n_mels = 128;
        let n_frames = n_floats / n_mels;
        eprintln!("Loading external mel from {}: [{}, {}]", mel_path, n_mels, n_frames);
        let flat: Vec<f32> = mel_bytes.chunks(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let mel = candle_core_native::Tensor::new(flat, &device)
            .and_then(|t| t.to_dtype(mel_dtype))
            .and_then(|t| t.reshape((1, n_mels, n_frames)))
            .expect("mel tensor");
        (n_mels, n_frames, mel)
    } else {
        // Use prepare_mel WITH padding (matching voxtral.c's left/right silence padding)
        let (flat, n_mels, n_frames) = prepare_mel(&args.audio, args.duration);
        let mel = candle_core_native::Tensor::new(&flat[..n_mels * n_frames], &device)
            .and_then(|t| t.to_dtype(mel_dtype))
            .and_then(|t| t.reshape((1, n_mels, n_frames)))
            .expect("mel tensor");
        (n_mels, n_frames, mel)
    };
    let t_embed = model::compute_time_embedding(6.0, 3072, &device).expect("t_embed");

    let t_infer = Instant::now();
    let token_ids = model::transcribe(&vox_model, &mel, &t_embed).expect("transcribe");
    let infer_secs = t_infer.elapsed().as_secs_f32();

    let text = tokenizer.decode(&token_ids.iter().map(|&t| t as i32).collect::<Vec<_>>());
    let audio_secs = n_frames as f32 / 200.0;

    eprintln!("\n=== CandleNative Results ===");
    eprintln!("Load:       {:.1}s", load_secs);
    eprintln!("Inference:  {:.1}s", infer_secs);
    eprintln!("Tokens:     {}", token_ids.len());
    eprintln!("Realtime:   {:.1}×", infer_secs / audio_secs);
    eprintln!("Text (200c): {:?}", &text[..text.char_indices().take(200).last().map_or(0, |(i, c)| i + c.len_utf8())]);
    println!("{}", text);
}

fn main() {
    let args = Args::parse();

    match args.backend.as_str() {
        "wgpu" => run_wgpu(&args),
        #[cfg(feature = "cuda")]
        "cuda" => run_cuda(&args),
        #[cfg(feature = "candle")]
        "candle" => run_candle(&args),
        #[cfg(feature = "candle-native")]
        "candle-native" => run_candle_native(&args),
        other => {
            eprintln!("Unknown backend: {}. Use 'wgpu', 'cuda', 'candle', or 'candle-native'.", other);
            std::process::exit(1);
        }
    }
}
