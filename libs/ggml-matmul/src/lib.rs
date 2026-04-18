//! FFI wrapper for ggml's Q4 quantized matmul via full computation graph.
//!
//! Uses ggml_mul_mat + ggml_graph_compute — the same code path as llama.cpp.
//! Handles threading, cache tiling, AVX-512 dispatch internally.

use std::os::raw::c_int;

extern "C" {
    fn ggml_matmul_set_threads(n: c_int);
    fn ggml_matmul_get_threads() -> c_int;

    /// Q4_0 × F32 matmul via ggml computation graph.
    fn ggml_q4_mul_mat(
        m: c_int,
        k: c_int,
        n: c_int,
        input: *const f32,
        weight: *const u8,
        w_nbytes: usize,
        output: *mut f32,
        n_threads: c_int,
    );
}

/// Q4_0 block size: 32 elements per block, 18 bytes per block
const QK4_0: usize = 32;
const BLOCK_Q4_0_SIZE: usize = 18;

/// Set the number of threads for ggml matmul. 0 = auto (4 threads).
pub fn set_threads(n: i32) {
    unsafe { ggml_matmul_set_threads(n) }
}

/// Q4_0 × F32 matrix multiply via ggml's computation graph.
///
/// Uses ggml_mul_mat internally — same code path as llama.cpp.
/// Handles AVX-512, cache tiling, and multi-threading automatically.
///
/// `m`: number of input rows (1 for decode, >1 for prefill)
/// `k`: inner dimension (must be multiple of 32)
/// `n`: number of output features
/// `lhs`: f32 input [m, k] row-major
/// `rhs_q4`: Q4_0 weight [n, k] as raw blocks, row-major
/// `dst`: output [m, n] row-major
pub fn q4_matmul(m: usize, k: usize, n: usize, lhs: &[f32], rhs_q4: &[u8], dst: &mut [f32]) {
    assert!(k % QK4_0 == 0, "k must be multiple of {}", QK4_0);
    let k_blocks = k / QK4_0;
    let q4_row_bytes = k_blocks * BLOCK_Q4_0_SIZE;

    assert_eq!(lhs.len(), m * k, "lhs size mismatch");
    assert_eq!(rhs_q4.len(), n * q4_row_bytes, "weight size mismatch");
    assert_eq!(dst.len(), m * n, "dst size mismatch");

    unsafe {
        ggml_q4_mul_mat(
            m as c_int,
            k as c_int,
            n as c_int,
            lhs.as_ptr(),
            rhs_q4.as_ptr(),
            rhs_q4.len(),
            dst.as_mut_ptr(),
            0, // use default thread count
        );
    }
}

/// Get the Q4_0 block byte size for a given number of elements.
pub fn q4_byte_size(num_elements: usize) -> usize {
    assert!(num_elements % QK4_0 == 0);
    (num_elements / QK4_0) * BLOCK_Q4_0_SIZE
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q4_matmul_shape() {
        let m = 1;
        let k = 64;
        let n = 32;
        let lhs = vec![0.1f32; m * k];
        let rhs = vec![0u8; n * q4_byte_size(k)];
        let mut dst = vec![0.0f32; m * n];
        q4_matmul(m, k, n, &lhs, &rhs, &mut dst);
        assert_eq!(dst.len(), m * n);
        assert!(dst.iter().all(|x| x.is_finite()));
    }
}
