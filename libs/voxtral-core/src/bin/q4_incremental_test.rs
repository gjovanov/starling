//! Native test harness for Q4IncrementalSession — same code path as wasm-engine,
//! but on a native Linux/Vulkan/dzn build so we can iterate without browser
//! round-trips. Reads a WAV, streams through the engine in 1-second commits,
//! prints transcription deltas + final text.
//!
//! Usage:
//!   cargo run --bin q4_incremental_test --release --features native -- \
//!     --models-dir <path> --audio <wav> [--commit-ms 1000] [--duration 30]

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use burn::backend::wgpu::{init_setup_async, RuntimeOptions, WgpuDevice};
use burn::backend::wgpu::graphics::Vulkan;

use voxtral_core::q4::loader::Q4ModelLoader;
use voxtral_core::q4::incremental_engine::Q4IncrementalSession;
use voxtral_core::tokenizer::TekkenDecoder;

fn parse_args() -> Result<Args> {
    let mut models_dir = PathBuf::from("models/cache");
    let mut audio = PathBuf::new();
    let mut commit_ms: u32 = 1000;
    let mut duration_s: u32 = 0;

    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--models-dir" => models_dir = it.next().context("--models-dir needs value")?.into(),
            "--audio" => audio = it.next().context("--audio needs value")?.into(),
            "--commit-ms" => commit_ms = it.next().context("--commit-ms")?.parse()?,
            "--duration" => duration_s = it.next().context("--duration")?.parse()?,
            "-h" | "--help" => {
                eprintln!(
                    "Usage: q4_incremental_test --models-dir <path> --audio <wav> \
                     [--commit-ms 1000] [--duration 0]\n\
                     duration=0 means full file."
                );
                std::process::exit(0);
            }
            other => anyhow::bail!("unknown arg: {other}"),
        }
    }
    if audio.as_os_str().is_empty() {
        anyhow::bail!("--audio is required");
    }
    Ok(Args { models_dir, audio, commit_ms, duration_s })
}

struct Args {
    models_dir: PathBuf,
    audio: PathBuf,
    commit_ms: u32,
    duration_s: u32,
}

/// Read a 16-bit / 16 kHz / mono WAV (matches what the browser dump button writes).
fn read_wav_mono16k(path: &std::path::Path) -> Result<Vec<f32>> {
    let reader = hound::WavReader::open(path)
        .with_context(|| format!("opening {}", path.display()))?;
    let spec = reader.spec();
    anyhow::ensure!(
        spec.sample_rate == 16000,
        "expected 16 kHz sample rate, got {}",
        spec.sample_rate
    );
    anyhow::ensure!(spec.channels == 1, "expected mono, got {} channels", spec.channels);

    let samples: Vec<f32> = if spec.bits_per_sample == 16 {
        reader
            .into_samples::<i16>()
            .collect::<std::result::Result<Vec<_>, _>>()?
            .into_iter()
            .map(|s| s as f32 / 32768.0)
            .collect()
    } else {
        reader
            .into_samples::<f32>()
            .collect::<std::result::Result<Vec<_>, _>>()?
    };
    Ok(samples)
}

/// Walk the q4/chunks dir and concatenate chunk_0..chunk_N into a single Vec.
fn read_q4_gguf(models_dir: &std::path::Path) -> Result<Vec<u8>> {
    let single = models_dir.join("q4").join("voxtral-q4.gguf");
    if single.exists() {
        return Ok(fs::read(&single).with_context(|| format!("reading {}", single.display()))?);
    }
    let chunks_dir = models_dir.join("q4").join("chunks");
    if !chunks_dir.is_dir() {
        anyhow::bail!(
            "no GGUF found at {} or chunks at {}",
            single.display(),
            chunks_dir.display()
        );
    }
    let mut bytes = Vec::new();
    let mut i = 0usize;
    loop {
        let p = chunks_dir.join(format!("chunk_{}", i));
        if !p.exists() {
            break;
        }
        bytes.extend(fs::read(&p)?);
        i += 1;
    }
    anyhow::ensure!(i > 0, "no chunk_* files in {}", chunks_dir.display());
    eprintln!("loaded {} chunks ({} MB) from {}", i, bytes.len() / 1_000_000, chunks_dir.display());
    Ok(bytes)
}

fn main() -> Result<()> {
    let args = parse_args()?;

    eprintln!("[q4-test] audio={} models-dir={} commit-ms={} duration={}s",
        args.audio.display(), args.models_dir.display(), args.commit_ms, args.duration_s);

    let pcm = read_wav_mono16k(&args.audio)?;
    let total_secs = pcm.len() as f32 / 16000.0;
    let max_samples = if args.duration_s == 0 {
        pcm.len()
    } else {
        (args.duration_s as usize * 16000).min(pcm.len())
    };
    eprintln!("[q4-test] loaded {:.2}s ({} samples), processing first {:.2}s",
        total_secs, pcm.len(), max_samples as f32 / 16000.0);

    // Init Burn/wgpu device — same as wasm-engine but with Vulkan backend on Linux/dzn
    let device = WgpuDevice::default();
    let rt = pollster::block_on(init_setup_async::<Vulkan>(&device, RuntimeOptions::default()));
    eprintln!("[q4-test] adapter: {} ({:?})",
        rt.adapter.get_info().name, rt.adapter.get_info().backend);

    // Load Q4 GGUF (single shard for simplicity)
    let t0 = Instant::now();
    let gguf = read_q4_gguf(&args.models_dir)?;
    let mut loader = Q4ModelLoader::from_shards(vec![gguf])
        .map_err(|e| anyhow::anyhow!("from_shards: {e:?}"))?;
    let model = loader.load(&device)
        .map_err(|e| anyhow::anyhow!("loader.load: {e:?}"))?;
    eprintln!("[q4-test] model loaded in {:.1}s", t0.elapsed().as_secs_f32());

    // Tokenizer
    let tokenizer_json = fs::read_to_string(args.models_dir.join("tokenizer/tekken.json"))
        .context("reading tokenizer/tekken.json")?;
    let tokenizer = TekkenDecoder::from_json(&tokenizer_json)
        .map_err(|e| anyhow::anyhow!("tokenizer: {e:?}"))?;

    // Session — exactly the same constructor wasm-engine uses
    let mut session = Q4IncrementalSession::new(model, tokenizer, device);

    // Stream commit_ms-sized chunks (default 1000ms = mirrors browser COMMIT_INTERVAL_MS)
    let chunk_samples = (16000u32 * args.commit_ms / 1000) as usize;
    let mut offset = 0usize;
    let mut full_text = String::new();
    let stream_t0 = Instant::now();

    while offset < max_samples {
        let end = (offset + chunk_samples).min(max_samples);
        let chunk = &pcm[offset..end];
        session.send_audio(chunk);

        let commit_t0 = Instant::now();
        let delta = pollster::block_on(session.commit())
            .map_err(|e| anyhow::anyhow!("commit: {e}"))?;
        let commit_ms = commit_t0.elapsed().as_millis();

        if !delta.is_empty() {
            full_text.push_str(&delta);
            eprintln!("[q4-test] commit #{} ({}ms, audio_so_far={:.1}s): {:?}",
                session.commit_count(), commit_ms, end as f32 / 16000.0, delta);
        } else {
            eprintln!("[q4-test] commit #{} ({}ms, audio_so_far={:.1}s): (no text)",
                session.commit_count(), commit_ms, end as f32 / 16000.0);
        }
        offset = end;
    }

    let total_audio = max_samples as f32 / 16000.0;
    let wall = stream_t0.elapsed().as_secs_f32();
    eprintln!("\n=== Final ===");
    eprintln!("audio = {:.1}s, wall = {:.1}s, RT = {:.2}x", total_audio, wall, wall / total_audio);
    eprintln!("text  = {:?}", full_text);

    Ok(())
}
