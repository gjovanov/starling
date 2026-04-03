//! Phase 0 POC: Burn/wgpu on WASM with async GPU readback.
//!
//! Tests:
//! 1. Burn tensor creation on wgpu backend (WebGPU in browser)
//! 2. Matmul operation
//! 3. Async GPU→CPU readback (into_data_async)
//! 4. Argmax (decoder token selection pattern)

use wasm_bindgen::prelude::*;

use burn::backend::wgpu::{
    CubeBackend, RuntimeOptions, WgpuDevice, WgpuRuntime,
    graphics::WebGpu, init_setup_async,
};
use burn::tensor::{Tensor, TensorData};

/// Non-fused wgpu backend (same as Q4 module uses).
type WgpuBackend = CubeBackend<WgpuRuntime, f32, i32, u32>;

#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
    web_sys::console::log_1(&"[wasm-poc] WASM module initialized".into());
}

/// Initialize the wgpu runtime asynchronously.
///
/// On WASM, the default lazy init calls `block_on()` which panics because
/// GPU adapter/device creation is inherently async in WebGPU.
/// This must be called BEFORE any tensor operations.
#[wasm_bindgen]
pub async fn init_runtime() -> Result<String, JsValue> {
    web_sys::console::log_1(&"[wasm-poc] Initializing wgpu runtime async...".into());

    let device = WgpuDevice::default();
    let setup = init_setup_async::<WebGpu>(&device, RuntimeOptions::default()).await;

    let adapter_info = setup.adapter.get_info();
    let msg = format!(
        "Runtime initialized: {} ({:?})",
        adapter_info.name,
        adapter_info.backend,
    );
    web_sys::console::log_1(&format!("[wasm-poc] {}", msg).into());
    Ok(msg)
}

/// Test 1: Create tensors, run matmul, read back result via async GPU readback.
///
/// This is the critical test — `into_data_async().await` must work on WASM.
/// The sync `into_data()` / `into_scalar()` would panic.
#[wasm_bindgen]
pub async fn test_matmul_async_readback() -> Result<String, JsValue> {
    web_sys::console::log_1(&"[wasm-poc] Starting matmul + async readback test...".into());

    let device = WgpuDevice::default();

    // A: [1, 2, 3]  B: [1, 4]
    //    [4, 5, 6]     [2, 5]
    //                   [3, 6]
    // A × B = [14, 32]
    //         [32, 77]
    let a_data = TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
    let b_data = TensorData::new(vec![1.0f32, 4.0, 2.0, 5.0, 3.0, 6.0], [3, 2]);

    let a: Tensor<WgpuBackend, 2> = Tensor::from_data(a_data, &device);
    let b: Tensor<WgpuBackend, 2> = Tensor::from_data(b_data, &device);

    web_sys::console::log_1(&"[wasm-poc] Tensors created on WebGPU device".into());

    let c = a.matmul(b);

    web_sys::console::log_1(&"[wasm-poc] Matmul dispatched, reading back async...".into());

    let result = c.into_data_async().await
        .map_err(|e| JsValue::from_str(&format!("GPU readback failed: {:?}", e)))?;
    let values: Vec<f32> = result.to_vec().unwrap();

    let expected = vec![14.0f32, 32.0, 32.0, 77.0];
    let pass = values == expected;

    let msg = format!(
        "Matmul result: {:?} (expected {:?}) — {}",
        values,
        expected,
        if pass { "PASS" } else { "FAIL" }
    );
    web_sys::console::log_1(&format!("[wasm-poc] {}", msg).into());

    Ok(msg)
}

/// Test 2: Async argmax — this is what the decoder loop needs.
///
/// The Q4 model does `pred.into_scalar().elem()` to get the next token ID.
/// On WASM this must be `pred.into_data_async().await` + extract scalar.
#[wasm_bindgen]
pub async fn test_argmax_async() -> Result<String, JsValue> {
    web_sys::console::log_1(&"[wasm-poc] Starting argmax async test...".into());

    let device = WgpuDevice::default();

    // Simulate logits: [1, 1, 8] (batch=1, seq=1, vocab=8)
    // Token 5 has highest logit
    let logits_data = TensorData::new(
        vec![0.1f32, 0.2, 0.05, 0.3, 0.15, 0.9, 0.1, 0.05],
        [1, 1, 8],
    );
    let logits: Tensor<WgpuBackend, 3> = Tensor::from_data(logits_data, &device);

    let pred = logits.argmax(2);

    let data = pred.into_data_async().await
        .map_err(|e| JsValue::from_str(&format!("GPU readback failed: {:?}", e)))?;
    let token_id: i32 = data.to_vec::<i32>().unwrap()[0];

    let pass = token_id == 5;
    let msg = format!(
        "Argmax result: {} (expected 5) — {}",
        token_id,
        if pass { "PASS" } else { "FAIL" }
    );
    web_sys::console::log_1(&format!("[wasm-poc] {}", msg).into());

    Ok(msg)
}

/// Test 3: Larger matmul simulating a decoder layer projection.
///
/// Tests that WebGPU can handle realistic tensor sizes without OOM.
/// hidden [1, 3072] × weight [3072, 8192] → [1, 8192]
#[wasm_bindgen]
pub async fn test_large_matmul() -> Result<String, JsValue> {
    web_sys::console::log_1(&"[wasm-poc] Starting large matmul test (3072 × 8192)...".into());

    let device = WgpuDevice::default();

    let m = 1;
    let k = 3072;
    let n = 8192;

    let a_vals: Vec<f32> = (0..m * k).map(|i| (i % 17) as f32 * 0.01).collect();
    let b_vals: Vec<f32> = (0..k * n).map(|i| (i % 13) as f32 * 0.01).collect();

    let a: Tensor<WgpuBackend, 2> =
        Tensor::from_data(TensorData::new(a_vals, [m, k]), &device);
    let b: Tensor<WgpuBackend, 2> =
        Tensor::from_data(TensorData::new(b_vals, [k, n]), &device);

    let start = js_sys::Date::now();
    let c = a.matmul(b);
    let result = c.into_data_async().await
        .map_err(|e| JsValue::from_str(&format!("GPU readback failed: {:?}", e)))?;
    let elapsed = js_sys::Date::now() - start;

    let values: Vec<f32> = result.to_vec().unwrap();
    let msg = format!(
        "Large matmul [1,3072]×[3072,8192]: {} elements, first={:.4}, last={:.4}, took {:.1}ms — PASS",
        values.len(),
        values[0],
        values[values.len() - 1],
        elapsed
    );
    web_sys::console::log_1(&format!("[wasm-poc] {}", msg).into());

    Ok(msg)
}

/// Run all tests sequentially.
/// Assumes init_runtime() was already called during page load.
#[wasm_bindgen]
pub async fn run_all_tests() -> Result<String, JsValue> {
    let mut results = Vec::new();

    results.push(test_matmul_async_readback().await?);
    results.push(test_argmax_async().await?);
    results.push(test_large_matmul().await?);

    let summary = results.join("\n");
    web_sys::console::log_1(&format!("[wasm-poc] All tests complete:\n{}", summary).into());
    Ok(summary)
}
