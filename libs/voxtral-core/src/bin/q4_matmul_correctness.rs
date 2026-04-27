//! Q4 matmul correctness test.
//!
//! Pulls a single Q4_0 weight tensor out of the GGUF, runs both
//!   (1) the wgpu fused dequant+matmul kernel
//!   (2) a CPU reference: dequant the same bytes manually, then dense matmul
//! on the same deterministic input. If the kernel is correct, the two outputs
//! match within float-precision noise. If the kernel has a bias bug, this is
//! where it shows up.
//!
//! Usage:
//!   cargo run --release --bin q4_matmul_correctness --features native -- \
//!     --models-dir models/cache [--tensor blk.0.attn_q.weight] [--num-tensors 5]

use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result};
use burn::backend::wgpu::{init_setup_async, RuntimeOptions, WgpuDevice};
use burn::backend::wgpu::graphics::Vulkan;
use burn::tensor::{Tensor, TensorData};

use voxtral_core::q4::reader::GgufReader;
use voxtral_core::q4::tensor::Q4Tensor;
use voxtral_core::q4::op::q4_matmul;
use voxtral_core::q4::WgpuBackend;

type B = WgpuBackend;

fn parse_args() -> Result<Args> {
    let mut models_dir = PathBuf::from("models/cache");
    let mut tensor_names: Vec<String> = Vec::new();
    let mut num_tensors: usize = 5;

    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--models-dir" => models_dir = it.next().context("--models-dir")?.into(),
            "--tensor" => tensor_names.push(it.next().context("--tensor")?),
            "--num-tensors" => num_tensors = it.next().context("--num-tensors")?.parse()?,
            "-h" | "--help" => {
                eprintln!(
                    "Usage: q4_matmul_correctness --models-dir <dir> [--tensor NAME] [--num-tensors N]\n\
                     If no --tensor specified, picks the N smallest Q4 tensors automatically."
                );
                std::process::exit(0);
            }
            other => anyhow::bail!("unknown arg: {other}"),
        }
    }
    Ok(Args { models_dir, tensor_names, num_tensors })
}

struct Args {
    models_dir: PathBuf,
    tensor_names: Vec<String>,
    num_tensors: usize,
}

fn read_q4_gguf_bytes(models_dir: &std::path::Path) -> Result<Vec<u8>> {
    let single = models_dir.join("q4").join("voxtral-q4.gguf");
    if single.exists() {
        return Ok(fs::read(&single)?);
    }
    let chunks_dir = models_dir.join("q4").join("chunks");
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
    anyhow::ensure!(i > 0, "no GGUF found in {}", models_dir.display());
    Ok(bytes)
}

/// Dequant Q4_0 to f32, matching loader.rs::dequantize_q4_0_cpu byte-for-byte.
fn dequantize_q4_cpu(raw: &[u8], num_elements: usize) -> Vec<f32> {
    let num_blocks = num_elements / 32;
    let mut output = vec![0.0f32; num_elements];
    for block_idx in 0..num_blocks {
        let offset = block_idx * 18;
        let d = half::f16::from_bits(u16::from_le_bytes([raw[offset], raw[offset + 1]])).to_f32();
        let base = block_idx * 32;
        for i in 0..16 {
            let byte = raw[offset + 2 + i];
            let lo = (byte & 0x0F) as f32 - 8.0;
            let hi = ((byte >> 4) & 0x0F) as f32 - 8.0;
            output[base + i] = lo * d;
            output[base + i + 16] = hi * d;
        }
    }
    output
}

/// Reference matmul on CPU: y[n] = sum_k input[k] * W[n,k].
/// W is [N, K] dense f32 (already dequantized).
fn matmul_cpu_ref(input: &[f32], w: &[f32], n: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    for ni in 0..n {
        let row = &w[ni * k..(ni + 1) * k];
        let mut acc = 0.0f64; // f64 accumulator to avoid order-dependent rounding
        for ki in 0..k {
            acc += (row[ki] as f64) * (input[ki] as f64);
        }
        out[ni] = acc as f32;
    }
    out
}

/// Deterministic input vector: a small mixed-sign signal that exercises
/// both positive and negative weight directions.
fn build_test_input(k: usize) -> Vec<f32> {
    (0..k)
        .map(|i| {
            let phase = (i as f32) * 0.013;
            (phase.sin() * 0.5) + ((i as f32 / k as f32) - 0.5) * 0.3
        })
        .collect()
}

fn run_one(
    name: &str,
    raw: &[u8],
    shape: [usize; 2],
    device: &WgpuDevice,
) -> Result<()> {
    let [n, k] = shape;
    let num_elements = n * k;

    // ---- Reference (CPU) ----
    let dequant = dequantize_q4_cpu(raw, num_elements);
    let input = build_test_input(k);
    let ref_out = matmul_cpu_ref(&input, &dequant, n, k);

    // ---- GPU kernel ----
    let q4 = Q4Tensor::from_q4_bytes(raw, shape, device)
        .map_err(|e| anyhow::anyhow!("Q4Tensor: {e:?}"))?;
    let input_t: Tensor<B, 3> = Tensor::from_data(
        TensorData::new(input.clone(), [1, 1, k]),
        device,
    );
    let out_t = q4_matmul(input_t, &q4);
    let out_data = out_t.into_data();
    let gpu_out: Vec<f32> = out_data
        .to_vec::<f32>()
        .map_err(|e| anyhow::anyhow!("readback: {e:?}"))?;

    // ---- Stats ----
    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f64;
    let mut max_rel = 0.0f32;
    let mut idx_of_max_abs = 0usize;
    for i in 0..n {
        let diff = (gpu_out[i] - ref_out[i]).abs();
        sum_abs += diff as f64;
        if diff > max_abs {
            max_abs = diff;
            idx_of_max_abs = i;
        }
        let denom = ref_out[i].abs().max(1e-6);
        let rel = diff / denom;
        if rel > max_rel {
            max_rel = rel;
        }
    }
    let mean_abs = (sum_abs / n as f64) as f32;

    let l2_ref: f32 = ref_out.iter().map(|v| v * v).sum::<f32>().sqrt();
    let l2_gpu: f32 = gpu_out.iter().map(|v| v * v).sum::<f32>().sqrt();
    let mean_ref: f32 = ref_out.iter().sum::<f32>() / n as f32;
    let mean_gpu: f32 = gpu_out.iter().sum::<f32>() / n as f32;

    println!(
        "{name}  shape=[{n}, {k}]\n  \
         max_abs_err = {max_abs:.6} (idx {idx_of_max_abs}, ref={:.4} gpu={:.4})\n  \
         mean_abs_err= {mean_abs:.6}\n  \
         max_rel_err = {max_rel:.4}\n  \
         L2(ref) = {l2_ref:.4}, L2(gpu) = {l2_gpu:.4}\n  \
         mean(ref) = {mean_ref:+.6}, mean(gpu) = {mean_gpu:+.6}",
        ref_out[idx_of_max_abs], gpu_out[idx_of_max_abs]
    );

    // First few side-by-side
    print!("  ref[..6]= [");
    for v in &ref_out[..6.min(n)] { print!("{:+.4} ", v); }
    println!("]");
    print!("  gpu[..6]= [");
    for v in &gpu_out[..6.min(n)] { print!("{:+.4} ", v); }
    println!("]");

    // PASS/FAIL: max relative error should be tiny if kernel is correct.
    // 5% rel-err tolerance covers f32 rounding-order divergence; anything
    // above that on a real Q4 weight is very likely a bug.
    if max_rel > 0.05 || max_abs > 1.0 {
        println!("  -> FAIL (max_rel > 5% or max_abs > 1.0)");
    } else if max_rel > 0.001 {
        println!("  -> SUSPECT (max_rel > 0.1%)");
    } else {
        println!("  -> OK");
    }
    println!();
    Ok(())
}

fn main() -> Result<()> {
    let args = parse_args()?;

    eprintln!("[q4-mm-test] models-dir={}", args.models_dir.display());

    // Open GGUF reader (in-memory, single shard)
    let bytes = read_q4_gguf_bytes(&args.models_dir)?;
    eprintln!("[q4-mm-test] GGUF: {:.1} MB", bytes.len() as f64 / 1e6);
    let cursor = std::io::Cursor::new(bytes);
    let mut reader = GgufReader::open(cursor)
        .map_err(|e| anyhow::anyhow!("open GGUF: {e:?}"))?;

    // Pick tensors to test
    let names: Vec<String> = if !args.tensor_names.is_empty() {
        args.tensor_names.clone()
    } else {
        // Find smallest Q4_0 tensors (filter out F32 norms/biases first)
        let all = reader.tensor_names();
        let mut q4s: Vec<(String, usize)> = all
            .iter()
            .filter_map(|name| {
                let info = reader.tensor_info(name)?;
                let dtype_str = format!("{:?}", info.dtype());
                if !dtype_str.contains("Q4_0") {
                    return None;
                }
                let shape = info.shape();
                if shape.len() != 2 {
                    return None;
                }
                let n_elem: usize = shape.iter().map(|&d| d as usize).product();
                Some((name.to_string(), n_elem))
            })
            .collect();
        q4s.sort_by_key(|(_, n)| *n);
        q4s.into_iter().take(args.num_tensors).map(|(n, _)| n).collect()
    };

    // GPU init
    let device = WgpuDevice::default();
    let rt = pollster::block_on(init_setup_async::<Vulkan>(&device, RuntimeOptions::default()));
    eprintln!("[q4-mm-test] adapter: {} ({:?})\n",
        rt.adapter.get_info().name, rt.adapter.get_info().backend);

    for name in &names {
        let info = reader.tensor_info(name)
            .with_context(|| format!("tensor '{name}' not found"))?;
        let dtype = info.dtype();
        // GGUF stores [in, out]; q4_matmul wants [N=out, K=in]
        let gguf_shape: Vec<usize> = info.shape().iter().map(|&d| d as usize).collect();
        if gguf_shape.len() != 2 {
            eprintln!("[skip] {name}: dtype={:?} non-2D shape {gguf_shape:?}", dtype);
            continue;
        }
        let shape = [gguf_shape[1], gguf_shape[0]]; // [out, in] = [N, K]
        let dtype_str = format!("{dtype:?}");
        if !dtype_str.contains("Q4_0") {
            eprintln!("[skip] {name}: dtype={dtype_str}");
            continue;
        }
        let raw = reader.tensor_data(name)
            .map_err(|e| anyhow::anyhow!("tensor_data({name}): {e:?}"))?;
        run_one(name, &raw, shape, &device)?;
    }

    Ok(())
}
