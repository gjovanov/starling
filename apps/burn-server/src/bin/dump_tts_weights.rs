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

    // Touch ModuleGroup so unused-import lint stays quiet on minimal CLIs.
    let _ = ModuleGroup::ALL.len();

    Ok(())
}
