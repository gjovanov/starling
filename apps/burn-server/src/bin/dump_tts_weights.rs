//! Sanity binary for the voxtral-tts weight inventory.
//!
//! Opens `consolidated.safetensors`, validates all 386 expected tensors
//! are present with the right shapes, and prints a per-group summary.
//!
//! Usage:
//!     cargo run --release --features voxtral-tts --bin dump-tts-weights
//!     cargo run --release --features voxtral-tts --bin dump-tts-weights -- \
//!         --path /custom/path/to/consolidated.safetensors --keys 5

use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;

use burn_server::inference::voxtral_tts::{
    ExpectedGroup, ModuleGroup, WeightInventory, EXPECTED_GROUPS, EXPECTED_TOTAL,
};

#[derive(Parser)]
#[command(about = "Validate Voxtral-4B-TTS safetensors weights against the expected inventory")]
struct Args {
    /// Path to consolidated.safetensors. Defaults to the shared cache.
    #[arg(long, default_value = "../../models/cache/tts/consolidated.safetensors")]
    path: PathBuf,

    /// Print the first N keys per group (0 = none).
    #[arg(long, default_value_t = 0)]
    keys: usize,

    /// Print shape + dtype for the first N keys per group.
    #[arg(long, default_value_t = 0)]
    verbose: usize,
}

fn fmt_bytes(n: usize) -> String {
    const UNITS: &[&str] = &["B", "KiB", "MiB", "GiB"];
    let mut x = n as f64;
    let mut u = 0;
    while x >= 1024.0 && u < UNITS.len() - 1 {
        x /= 1024.0;
        u += 1;
    }
    format!("{x:.2} {}", UNITS[u])
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("Opening: {}", args.path.display());
    let inv = WeightInventory::open(&args.path)?;
    println!(
        "OK — all {EXPECTED_TOTAL} expected tensors present + sentinel shapes match.\n"
    );

    println!("Per-group breakdown:");
    println!("  {:<38} {:>5}  status", "module", "count");
    println!("  {}", "-".repeat(54));

    let mut total = 0usize;
    for &ExpectedGroup { group, count } in EXPECTED_GROUPS {
        let keys = inv.keys_in(group);
        total += keys.len();
        let status = if keys.len() == count { "OK" } else { "MISMATCH" };
        println!("  {:<38} {:>5}  {}", group.label(), keys.len(), status);
        if args.keys > 0 {
            for k in keys.iter().take(args.keys) {
                println!("    - {k}");
            }
            if keys.len() > args.keys {
                println!("    … {} more", keys.len() - args.keys);
            }
        }
        if args.verbose > 0 {
            for k in keys.iter().take(args.verbose) {
                let v = inv.tensor_view(k)?;
                println!(
                    "    - {k}  shape={:?} dtype={:?}",
                    v.shape(),
                    v.dtype()
                );
            }
        }
    }

    let total_bytes = inv.total_payload_bytes();
    println!();
    println!("Total tensors: {total}");
    println!(
        "Total payload size: {} ({} bytes)",
        fmt_bytes(total_bytes),
        total_bytes
    );
    println!("Path: {}", inv.path().display());

    // Voice-cloning capability check (Phase 3-E2).
    //
    // Voxtral's voice-cloning path requires the codec ENCODER weights
    // (`audio_tokenizer.input_proj.*` + `audio_tokenizer.encoder_blocks.*`).
    // These are NOT in the public Voxtral-4B-TTS-2603 checkpoint;
    // vllm-omni's own source confirms `encode_waveforms requires
    // encoder weights which are not available in the open-source
    // checkpoint`. The reference voice embeddings ship pre-computed
    // under `voice_embedding/`, but the encoder you'd use to embed a
    // user-uploaded clip is missing — for both burn-server and
    // vllm-server.
    println!();
    println!("Voice-cloning capability:");
    let raw_keys: Vec<&str> = ModuleGroup::ALL
        .iter()
        .flat_map(|g| inv.keys_in(*g).iter().map(|s| s.as_str()))
        .collect();
    let has_input_proj = raw_keys.iter().any(|k| k.starts_with("audio_tokenizer.input_proj."));
    let has_encoder_blocks = raw_keys
        .iter()
        .any(|k| k.starts_with("audio_tokenizer.encoder_blocks."));
    if has_input_proj && has_encoder_blocks {
        println!("  [OK] codec encoder weights present — voice cloning synthesis works");
    } else {
        println!("  [---] codec encoder weights ABSENT — voice cloning synthesis blocked");
        println!(
            "        (input_proj present: {}, encoder_blocks present: {})",
            has_input_proj, has_encoder_blocks
        );
        println!(
            "        Storage + listing under /api/tts/voices/upload still work; only the"
        );
        println!("        synthesis half is blocked. Affects burn-server AND vllm-server.");
    }

    Ok(())
}
