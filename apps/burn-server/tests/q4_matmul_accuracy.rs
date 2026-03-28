//! Q4 matmul accuracy tests — validates WGSL shader dequant+matmul against f32 reference.
//!
//! Ported from voxtral-mini-realtime-rs/src/gguf/tests.rs

use burn::backend::wgpu::WgpuDevice;
use burn::tensor::{Tensor, TensorData};
use burn_server::inference::q4::{self, tensor::Q4Tensor, WgpuBackend};

/// Convert f32 weights to Q4_0 GGML format bytes.
/// Each block of 32 elements → 18 bytes (2 f16 scale + 16 packed nibbles).
fn quantize_f32_to_q4_0(data: &[f32]) -> Vec<u8> {
    assert!(data.len() % 32 == 0, "Data length must be multiple of 32");
    let num_blocks = data.len() / 32;
    let mut bytes = Vec::with_capacity(num_blocks * 18);

    for block_idx in 0..num_blocks {
        let block = &data[block_idx * 32..(block_idx + 1) * 32];

        // Find absmax
        let amax = block.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let d = amax / 7.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };

        // Write scale as f16
        let d_f16 = half::f16::from_f32(d);
        bytes.extend_from_slice(&d_f16.to_le_bytes());

        // Pack nibbles: lower 16 elements in low nibble, upper 16 in high nibble
        for j in 0..16 {
            let v0 = block[j];
            let v1 = block[j + 16];
            let q0 = ((v0 * id + 8.5).clamp(0.0, 15.0)) as u8;
            let q1 = ((v1 * id + 8.5).clamp(0.0, 15.0)) as u8;
            bytes.push(q0 | (q1 << 4));
        }
    }

    bytes
}

/// Reference f32 matmul: a[M,K] × b_t[N,K]^T → out[M,N]
fn reference_matmul(a: &[f32], b_t: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for l in 0..k {
                sum += a[i * k + l] * b_t[j * k + l];
            }
            out[i * n + j] = sum;
        }
    }
    out
}

/// Dequantize Q4_0 bytes back to f32 (for reference comparison).
fn dequantize_q4_0_to_f32(bytes: &[u8], num_elements: usize) -> Vec<f32> {
    let num_blocks = num_elements / 32;
    let mut output = vec![0.0f32; num_elements];

    for block_idx in 0..num_blocks {
        let bo = block_idx * 18;
        let d = half::f16::from_le_bytes([bytes[bo], bytes[bo + 1]]).to_f32();
        let base = block_idx * 32;

        for j in 0..16 {
            let byte = bytes[bo + 2 + j];
            output[base + j] = ((byte & 0x0F) as f32 - 8.0) * d;
            output[base + j + 16] = (((byte >> 4) & 0x0F) as f32 - 8.0) * d;
        }
    }
    output
}

#[test]
fn test_q4_block_roundtrip() {
    // Quantize → dequantize round-trip should be within Q4 tolerance
    let data: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.1).collect();
    let q4_bytes = quantize_f32_to_q4_0(&data);
    let deq = dequantize_q4_0_to_f32(&q4_bytes, 32);

    let max_diff = data.iter().zip(deq.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    eprintln!("Q4 block round-trip max_diff: {:.6}", max_diff);
    assert!(max_diff < 0.15, "Q4 round-trip error too large: {}", max_diff);
}

#[test]
fn test_q4_matmul_small() {
    let device = WgpuDevice::default();

    let m = 1;
    let k = 32; // Must be multiple of 32 for Q4
    let n = 32;

    // Create random-ish weights and input
    let weights_f32: Vec<f32> = (0..n * k).map(|i| ((i * 7 + 3) % 19) as f32 / 19.0 - 0.5).collect();
    let input_f32: Vec<f32> = (0..m * k).map(|i| ((i * 13 + 5) % 23) as f32 / 23.0 - 0.5).collect();

    // Q4 quantize weights
    let q4_bytes = quantize_f32_to_q4_0(&weights_f32);

    // Dequantize for reference (this is what the GPU shader does)
    let weights_deq = dequantize_q4_0_to_f32(&q4_bytes, n * k);
    let reference = reference_matmul(&input_f32, &weights_deq, m, k, n);

    // GPU Q4 matmul
    let q4_tensor = Q4Tensor::from_q4_bytes(&q4_bytes, [n, k], &device).unwrap();
    let input_tensor: Tensor<WgpuBackend, 3> = Tensor::from_data(
        TensorData::new(input_f32, [1, m, k]),
        &device,
    );
    let output = q4::op::q4_matmul(input_tensor, &q4_tensor);
    let output_data = output.into_data();
    let output_vals = output_data.as_slice::<f32>().unwrap();

    let max_diff = reference.iter().zip(output_vals.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    eprintln!("Q4 small matmul max_diff: {:.6}", max_diff);
    assert!(max_diff < 1e-3, "Q4 small matmul error: {}", max_diff);
}

#[test]
fn test_q4_matmul_decoder_shapes() {
    let device = WgpuDevice::default();

    // Test with actual decoder dimensions
    let test_cases = [
        ("wq", 1, 3072, 4096),   // Q projection
        ("wk", 1, 3072, 1024),   // K projection (GQA)
        ("wo", 1, 4096, 3072),   // O projection
        ("w1", 1, 3072, 9216),   // FFN gate
        ("w2", 1, 9216, 3072),   // FFN down
    ];

    for (name, m, k, n) in test_cases {
        let weights_f32: Vec<f32> = (0..n * k).map(|i| {
            let x = ((i * 7 + 13) % 37) as f32 / 37.0 - 0.5;
            x * 0.1 // Small values typical of trained weights
        }).collect();
        let input_f32: Vec<f32> = (0..m * k).map(|i| {
            ((i * 11 + 3) % 29) as f32 / 29.0 - 0.5
        }).collect();

        let q4_bytes = quantize_f32_to_q4_0(&weights_f32);
        let weights_deq = dequantize_q4_0_to_f32(&q4_bytes, n * k);
        let reference = reference_matmul(&input_f32, &weights_deq, m, k, n);

        let q4_tensor = Q4Tensor::from_q4_bytes(&q4_bytes, [n, k], &device).unwrap();
        let input_tensor: Tensor<WgpuBackend, 3> = Tensor::from_data(
            TensorData::new(input_f32, [1, m, k]),
            &device,
        );
        let output = q4::op::q4_matmul(input_tensor, &q4_tensor);
        let output_data = output.into_data();
        let output_vals = output_data.as_slice::<f32>().unwrap();

        let max_diff = reference.iter().zip(output_vals.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        eprintln!("Q4 matmul '{}' [{m}x{k}x{n}] max_diff: {:.6}", name, max_diff);
        assert!(max_diff < 1e-2, "Q4 matmul '{}' error too large: {}", name, max_diff);
    }
}

#[test]
fn test_q4_matmul_prefill() {
    let device = WgpuDevice::default();

    // Prefill: M=38 (prefix length), K=3072, N=4096
    let m = 38;
    let k = 3072;
    let n = 4096;

    let weights_f32: Vec<f32> = (0..n * k).map(|i| {
        ((i * 7 + 13) % 37) as f32 / 37.0 * 0.02 - 0.01
    }).collect();
    let input_f32: Vec<f32> = (0..m * k).map(|i| {
        ((i * 11 + 3) % 29) as f32 / 29.0 - 0.5
    }).collect();

    let q4_bytes = quantize_f32_to_q4_0(&weights_f32);
    let weights_deq = dequantize_q4_0_to_f32(&q4_bytes, n * k);
    let reference = reference_matmul(&input_f32, &weights_deq, m, k, n);

    let q4_tensor = Q4Tensor::from_q4_bytes(&q4_bytes, [n, k], &device).unwrap();
    let input_tensor: Tensor<WgpuBackend, 3> = Tensor::from_data(
        TensorData::new(input_f32, [1, m, k]),
        &device,
    );
    let output = q4::op::q4_matmul(input_tensor, &q4_tensor);
    let output_data = output.into_data();
    let output_vals = output_data.as_slice::<f32>().unwrap();

    let max_diff = reference.iter().zip(output_vals.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    eprintln!("Q4 prefill matmul [{}x{}x{}] max_diff: {:.6}", m, k, n, max_diff);
    assert!(max_diff < 1e-2, "Q4 prefill matmul error: {}", max_diff);
}
