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

fn prepare_mel(audio_path: &std::path::Path) -> (Vec<f32>, usize, usize) {
    let reader = hound::WavReader::open(audio_path).expect("Failed to open WAV");
    let spec = reader.spec();
    let samples: Vec<f32> = if spec.bits_per_sample == 16 {
        reader.into_samples::<i16>()
            .map(|s| s.unwrap() as f32 / 32768.0)
            .collect()
    } else {
        reader.into_samples::<f32>()
            .map(|s| s.unwrap())
            .collect()
    };

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

    let (flat, n_mels, n_frames) = prepare_mel(&args.audio);
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

    let (flat, n_mels, n_frames) = prepare_mel(&args.audio);
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

fn main() {
    let args = Args::parse();

    match args.backend.as_str() {
        "wgpu" => run_wgpu(&args),
        #[cfg(feature = "cuda")]
        "cuda" => run_cuda(&args),
        other => {
            eprintln!("Unknown backend: {}. Use 'wgpu' or 'cuda'.", other);
            std::process::exit(1);
        }
    }
}
