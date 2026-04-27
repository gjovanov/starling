//! Burn/wgpu fp32 matmul precision test on dzn.
//!
//! Tests the SPECIFIC matmul shapes that voxtral-core's Q4 path uses (besides
//! our own Q4 fused-dequant kernel, which is already verified bit-exact). If
//! any of these diverge from the CPU reference, we've found a Burn/wgpu
//! precision bug specific to dzn.
//!
//! Shapes tested:
//!   1. RmsNorm-shaped sum     : [1, S, D] @ [1, D, 1] → [1, S, 1]
//!   2. lm_head-shaped final   : [1, 1, D] @ [V, D]ᵀ  → [1, 1, V]
//!   3. Attention QK^T (M=1)   : [1, H, 1, dh] @ [1, H, kv, dh]ᵀ → [1, H, 1, kv]
//!   4. Attention output proj  : [1, H, 1, kv] @ [1, H, kv, dh] → [1, H, 1, dh]

use anyhow::Result;
use burn::backend::wgpu::{init_setup_async, RuntimeOptions, WgpuDevice};
use burn::backend::wgpu::graphics::Vulkan;
use burn::backend::Wgpu;
use burn::tensor::{Tensor, TensorData};

type B = Wgpu;

fn deterministic(n: usize, seed: u64) -> Vec<f32> {
    // Fast hash-style PRNG so each test run uses identical inputs.
    let mut s = seed;
    (0..n)
        .map(|_| {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            ((s as i64 as f64) / (i64::MAX as f64) * 0.5) as f32
        })
        .collect()
}

fn cpu_matmul_2d(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f64;
            for kk in 0..k {
                acc += (a[i * k + kk] as f64) * (b[kk * n + j] as f64);
            }
            out[i * n + j] = acc as f32;
        }
    }
    out
}

fn diff_stats(name: &str, gpu: &[f32], cpu: &[f32]) {
    let n = gpu.len().min(cpu.len());
    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f64;
    let mut max_rel = 0.0f32;
    let mut idx_of_max = 0usize;
    for i in 0..n {
        let d = (gpu[i] - cpu[i]).abs();
        sum_abs += d as f64;
        if d > max_abs { max_abs = d; idx_of_max = i; }
        let denom = cpu[i].abs().max(1e-6);
        let r = d / denom;
        if r > max_rel { max_rel = r; }
    }
    let mean_abs = (sum_abs / n as f64) as f32;
    let l2_gpu: f32 = gpu.iter().map(|v| v*v).sum::<f32>().sqrt();
    let l2_cpu: f32 = cpu.iter().map(|v| v*v).sum::<f32>().sqrt();
    let max_gpu = gpu.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min_gpu = gpu.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_cpu = cpu.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min_cpu = cpu.iter().cloned().fold(f32::INFINITY, f32::min);

    let verdict = if max_rel > 0.05 { "FAIL" }
                  else if max_rel > 0.001 { "SUSPECT" }
                  else { "OK" };

    println!("{name}");
    println!("  max_abs = {max_abs:.6} (idx {idx_of_max}, gpu={:.4} cpu={:.4})", gpu[idx_of_max], cpu[idx_of_max]);
    println!("  mean_abs= {mean_abs:.6}");
    println!("  max_rel = {max_rel:.4}");
    println!("  L2(gpu)={l2_gpu:.4}  L2(cpu)={l2_cpu:.4}");
    println!("  range(gpu)=[{min_gpu:+.4}, {max_gpu:+.4}]");
    println!("  range(cpu)=[{min_cpu:+.4}, {max_cpu:+.4}]");
    println!("  -> {verdict}\n");
}

fn extract<const D: usize>(t: Tensor<B, D>) -> Vec<f32> {
    let data = t.into_data();
    data.to_vec().expect("readback")
}

fn main() -> Result<()> {
    println!("Burn/wgpu fp32 matmul precision test (dzn)\n");

    let device = WgpuDevice::default();
    let rt = pollster::block_on(init_setup_async::<Vulkan>(&device, RuntimeOptions::default()));
    println!("adapter: {} ({:?})\n", rt.adapter.get_info().name, rt.adapter.get_info().backend);

    // ============================================================
    // Test 1: RmsNorm-shape sum-via-matmul
    //         x_sq [1, S, D] @ ones [1, D, 1] = sum_sq [1, S, 1]
    // S=1 (sequential M=1 decode), D=3072 (Voxtral d_model)
    // ============================================================
    {
        let s = 1usize;
        let d = 3072usize;
        let x_sq_data = deterministic(s * d, 0x1234);
        // Make values closer to actual squared activations (positive, modest)
        let x_sq_data: Vec<f32> = x_sq_data.iter().map(|v| v.abs() * 10.0).collect();
        let ones = vec![1.0f32; d];

        let cpu = cpu_matmul_2d(&x_sq_data, &ones, s, d, 1);

        let xt = Tensor::<B, 3>::from_data(TensorData::new(x_sq_data, [1, s, d]), &device);
        let ot = Tensor::<B, 2>::ones([d, 1], &device).unsqueeze::<3>();
        let gpu_t = xt.matmul(ot);
        let gpu = extract(gpu_t);
        diff_stats("Test 1 RmsNorm sum: [1, 1, 3072] @ [1, 3072, 1]", &gpu, &cpu);
    }

    // ============================================================
    // Test 2: lm_head-shape — large output projection
    //         hidden [1, 1, D] @ embed_T [D, V]  → [1, 1, V]
    // D=3072, V=131072 (large)
    // ============================================================
    {
        let d = 3072usize;
        let v = 131072usize;
        // Smaller test — still 400 MB worth of weights, manageable
        let v_test = 1024usize;  // truncate vocab for speed
        let hidden = deterministic(d, 0x5678);
        let embed_t_data = deterministic(d * v_test, 0x9abc); // [D, V_test]

        let cpu = cpu_matmul_2d(&hidden, &embed_t_data, 1, d, v_test);

        let ht = Tensor::<B, 3>::from_data(TensorData::new(hidden, [1, 1, d]), &device);
        let et = Tensor::<B, 2>::from_data(TensorData::new(embed_t_data, [d, v_test]), &device).unsqueeze::<3>();
        let gpu_t = ht.matmul(et);
        let gpu = extract(gpu_t);
        diff_stats(&format!("Test 2 lm_head: [1, 1, 3072] @ [1, 3072, {}]", v_test), &gpu, &cpu);
        let _ = v;
    }

    // ============================================================
    // Test 3: Attention QK^T at M=1 (early decode)
    //   Q [1, 32, 1, 96] @ K [1, 32, 4, 96]^T → [1, 32, 1, 4]
    //   simulating cache_len=4
    // (full GQA expand to 32 KV heads after expand_kv)
    // ============================================================
    {
        let h = 32usize;
        let dh = 96usize;
        let kv = 4usize;
        let q = deterministic(h * dh, 0xdead);
        let k_data = deterministic(h * kv * dh, 0xbeef);

        // CPU reference per head: [1, dh] @ [kv, dh]^T = [1, kv]
        let mut cpu = vec![0.0f32; h * kv];
        for hi in 0..h {
            let q_row = &q[hi * dh..(hi + 1) * dh];
            for kvi in 0..kv {
                let k_row = &k_data[hi * kv * dh + kvi * dh..hi * kv * dh + (kvi + 1) * dh];
                let mut acc = 0.0f64;
                for dhi in 0..dh {
                    acc += (q_row[dhi] as f64) * (k_row[dhi] as f64);
                }
                cpu[hi * kv + kvi] = acc as f32;
            }
        }

        let qt = Tensor::<B, 4>::from_data(TensorData::new(q, [1, h, 1, dh]), &device);
        let kt = Tensor::<B, 4>::from_data(TensorData::new(k_data, [1, h, kv, dh]), &device);
        let kt_t = kt.swap_dims(2, 3); // [1, h, dh, kv]
        let scores = qt.matmul(kt_t); // [1, h, 1, kv]
        let gpu = extract(scores);
        diff_stats("Test 3 Attention QK^T: [1, 32, 1, 96] @ [1, 32, 4, 96]^T", &gpu, &cpu);
    }

    // ============================================================
    // Test 4: Attention output projection
    //   attn [1, 32, 1, kv] @ V [1, 32, kv, dh] → [1, 32, 1, dh]
    // ============================================================
    {
        let h = 32usize;
        let dh = 96usize;
        let kv = 4usize;
        let attn = deterministic(h * kv, 0xc0ffee);
        let v_data = deterministic(h * kv * dh, 0xfade);

        let mut cpu = vec![0.0f32; h * dh];
        for hi in 0..h {
            let a_row = &attn[hi * kv..(hi + 1) * kv];
            let v_block = &v_data[hi * kv * dh..(hi + 1) * kv * dh];
            for dhi in 0..dh {
                let mut acc = 0.0f64;
                for kvi in 0..kv {
                    acc += (a_row[kvi] as f64) * (v_block[kvi * dh + dhi] as f64);
                }
                cpu[hi * dh + dhi] = acc as f32;
            }
        }

        let at = Tensor::<B, 4>::from_data(TensorData::new(attn, [1, h, 1, kv]), &device);
        let vt = Tensor::<B, 4>::from_data(TensorData::new(v_data, [1, h, kv, dh]), &device);
        let out = at.matmul(vt); // [1, h, 1, dh]
        let gpu = extract(out);
        diff_stats("Test 4 Attention attn@V: [1, 32, 1, 4] @ [1, 32, 4, 96]", &gpu, &cpu);
    }

    // ============================================================
    // Test 5: Full matmul-based RmsNorm composition
    //   x_sq = x*x; sum=x_sq@ones; mean=sum/D; rms=sqrt(mean+eps);
    //   out = (x/rms) * gamma
    // Tests the chain: elementwise mul, matmul, scalar div, sqrt,
    // broadcast div [1,S,D]/[1,S,1], broadcast mul [1,S,D]*[D]
    // ============================================================
    {
        let s = 1usize;
        let d = 3072usize;
        let eps = 1e-5f32;
        let x = deterministic(s * d, 0x11);
        let gamma = deterministic(d, 0x22);
        let gamma: Vec<f32> = gamma.iter().map(|v| 1.0 + v.abs()).collect(); // positive ~1-2

        // CPU reference
        let mut cpu_out = vec![0.0f32; s * d];
        for si in 0..s {
            let mut sum_sq = 0.0f64;
            for di in 0..d {
                let v = x[si * d + di] as f64;
                sum_sq += v * v;
            }
            let mean = sum_sq / d as f64;
            let rms = (mean + eps as f64).sqrt() as f32;
            for di in 0..d {
                cpu_out[si * d + di] = (x[si * d + di] / rms) * gamma[di];
            }
        }

        // GPU via matmul-based path (mimics layers/rms_norm.rs)
        let xt = Tensor::<B, 3>::from_data(TensorData::new(x.clone(), [1, s, d]), &device);
        let x_sq = xt.clone() * xt.clone();
        let ones = Tensor::<B, 2>::ones([d, 1], &device).unsqueeze::<3>();
        let sum_sq = x_sq.matmul(ones);
        let mean_sq = sum_sq / (d as f32);
        let rms = (mean_sq + eps as f64).sqrt();
        let gamma_t = Tensor::<B, 1>::from_data(TensorData::new(gamma, [d]), &device).unsqueeze::<3>();
        let out_t = (xt / rms) * gamma_t;
        let gpu = extract(out_t);
        diff_stats("Test 5 Full RmsNorm composition: [1, 1, 3072]", &gpu, &cpu_out);
    }

    // ============================================================
    // Test 6: GQA expand_kv via reshape (the "repeat" pattern)
    //   K [1, 8, kv, dh] → expand to [1, 32, kv, dh] (4× repeat of each head)
    // The mechanism in voxtral-core uses unsqueeze + broadcast-into-shape +
    // reshape to expand 8 KV heads to 32 query heads (factor 4).
    // ============================================================
    {
        let n_kv = 8usize;
        let kv_seq = 4usize;
        let dh = 96usize;
        let factor = 4usize;
        let n_q = n_kv * factor;
        let k = deterministic(n_kv * kv_seq * dh, 0x33);

        // CPU reference: each KV head repeated `factor` times consecutively
        let mut cpu = vec![0.0f32; n_q * kv_seq * dh];
        for qi in 0..n_q {
            let kv_idx = qi / factor;
            for sj in 0..kv_seq {
                for dj in 0..dh {
                    cpu[qi * kv_seq * dh + sj * dh + dj] =
                        k[kv_idx * kv_seq * dh + sj * dh + dj];
                }
            }
        }

        // GPU expand using the same pattern voxtral-core's q4/model.rs uses
        let kt = Tensor::<B, 4>::from_data(TensorData::new(k, [1, n_kv, kv_seq, dh]), &device);
        // [1, n_kv, kv_seq, dh] -> [1, n_kv, 1, kv_seq, dh] -> [1, n_kv, factor, kv_seq, dh] -> [1, n_q, kv_seq, dh]
        let expanded = kt
            .reshape([1, n_kv, 1, kv_seq, dh])
            .expand([1, n_kv, factor, kv_seq, dh])
            .reshape([1, n_q, kv_seq, dh]);
        let gpu = extract(expanded);
        diff_stats("Test 6 GQA expand_kv: [1, 8, 4, 96] -> [1, 32, 4, 96]", &gpu, &cpu);
    }

    Ok(())
}
