//! Voxtral WASM inference engine.
//!
//! Wraps voxtral-core's Q4 incremental inference pipeline with wasm-bindgen
//! exports for running Voxtral-Mini-4B-Realtime directly in the browser via
//! WebGPU.
//!
//! Uses the incremental encoder KV cache architecture: each commit() processes
//! only NEW audio through the encoder, avoiding re-encoding all accumulated
//! audio. This is the single biggest performance optimization for streaming.
//!
//! ## Usage from JavaScript
//!
//! ```js
//! import init, { WasmEngine } from './pkg/wasm_engine.js';
//!
//! await init();
//! const engine = await WasmEngine.create(modelShards, tokenizerJson);
//! engine.send_audio(new Float32Array(samples));
//! const text = await engine.commit();
//! ```

use wasm_bindgen::prelude::*;

use burn::backend::wgpu::{
    graphics::WebGpu, init_setup_async, RuntimeOptions, WgpuDevice,
};

use voxtral_core::q4::loader::Q4ModelLoader;
use voxtral_core::q4::incremental_engine::Q4IncrementalSession;
use voxtral_core::tokenizer::TekkenDecoder;
use voxtral_core::transcription::split_sentences;

/// Initialize the WebGPU runtime. Must be called before creating a WasmEngine.
///
/// Returns a string describing the GPU adapter.
#[wasm_bindgen]
pub async fn init_runtime() -> Result<String, JsValue> {
    console_error_panic_hook::set_once();
    web_sys::console::log_1(&"[wasm-engine] Initializing WebGPU runtime...".into());

    let device = WgpuDevice::default();
    let setup = init_setup_async::<WebGpu>(&device, RuntimeOptions::default()).await;

    let info = setup.adapter.get_info();
    let msg = format!("{} ({:?})", info.name, info.backend);
    web_sys::console::log_1(&format!("[wasm-engine] GPU: {}", msg).into());
    Ok(msg)
}

/// WASM inference engine — wraps the incremental Q4 session.
///
/// Uses incremental encoder KV caching: each commit processes only new audio,
/// rather than re-encoding everything from scratch.
#[wasm_bindgen]
pub struct WasmEngine {
    session: Q4IncrementalSession,
}

#[wasm_bindgen]
impl WasmEngine {
    /// Create a new engine from pre-loaded model shards and tokenizer JSON.
    ///
    /// `shard_arrays` -- Array of Uint8Array, each a 64MB GGUF shard
    /// `tokenizer_json` -- Contents of tekken.json
    #[wasm_bindgen]
    pub async fn create(
        shard_arrays: Vec<js_sys::Uint8Array>,
        tokenizer_json: &str,
    ) -> Result<WasmEngine, JsValue> {
        web_sys::console::log_1(
            &format!("[wasm-engine] Loading model from {} shards...", shard_arrays.len()).into(),
        );

        let device = WgpuDevice::default();

        // Convert JS Uint8Array shards to Vec<Vec<u8>>
        let shards: Vec<Vec<u8>> = shard_arrays.iter().map(|arr| arr.to_vec()).collect();

        let total_bytes: usize = shards.iter().map(|s| s.len()).sum();
        web_sys::console::log_1(
            &format!("[wasm-engine] Total model size: {:.1} MB", total_bytes as f64 / 1e6).into(),
        );

        // Load model via ShardedCursor (handles >2GB across multiple shards)
        let mut loader = Q4ModelLoader::from_shards(shards)
            .map_err(|e| JsValue::from_str(&format!("Failed to parse GGUF: {:?}", e)))?;

        let model = loader
            .load(&device)
            .map_err(|e| JsValue::from_str(&format!("Failed to load model: {:?}", e)))?;

        web_sys::console::log_1(&"[wasm-engine] Model loaded to WebGPU".into());

        // Load tokenizer
        let tokenizer = TekkenDecoder::from_json(tokenizer_json)
            .map_err(|e| JsValue::from_str(&format!("Failed to load tokenizer: {:?}", e)))?;

        web_sys::console::log_1(
            &format!("[wasm-engine] Tokenizer loaded ({} vocab)", tokenizer.vocab_size()).into(),
        );

        // Create incremental session
        let session = Q4IncrementalSession::new(model, tokenizer, device);

        web_sys::console::log_1(
            &"[wasm-engine] Incremental session created (encoder KV cached)".into(),
        );

        Ok(WasmEngine { session })
    }

    /// Append 16kHz mono f32 PCM audio samples.
    #[wasm_bindgen]
    pub fn send_audio(&mut self, samples: &[f32]) {
        self.session.send_audio(samples);
    }

    /// Process buffered audio incrementally and return the text delta.
    ///
    /// Only NEW audio since the last commit is encoded through the attention
    /// layers. Conv and mel are re-run on all audio (cheap), but the expensive
    /// 32-layer encoder attention uses KV caches to skip already-seen frames.
    ///
    /// This is async because WebGPU readback requires async on WASM.
    #[wasm_bindgen]
    pub async fn commit(&mut self) -> Result<String, JsValue> {
        let delta = self.session
            .commit()
            .await
            .map_err(|e| JsValue::from_str(&format!("Inference failed: {}", e)))?;

        let tokens = self.session.generated_tokens();
        let last_20: Vec<i32> = tokens.iter().rev().take(20).rev().cloned().collect();
        web_sys::console::log_1(
            &format!(
                "[wasm-engine] commit #{}: adapter={} decoder_started={} total_tokens={} last20={:?} delta={:?}",
                self.session.commit_count(),
                self.session.total_adapter_tokens(),
                self.session.decoder_started(),
                tokens.len(),
                last_20,
                &delta[..delta.len().min(80)]
            )
            .into(),
        );

        Ok(delta)
    }

    /// Reset session state (clear audio buffer, KV caches, text history).
    #[wasm_bindgen]
    pub fn reset(&mut self) {
        self.session.reset();
        web_sys::console::log_1(&"[wasm-engine] Session reset".into());
    }

    /// Number of commits since last reset.
    #[wasm_bindgen]
    pub fn commit_count(&self) -> usize {
        self.session.commit_count()
    }

    /// Split text into sentences (for subtitle display).
    #[wasm_bindgen]
    pub fn split_sentences_js(text: &str) -> Vec<String> {
        split_sentences(text)
    }
}
