//! Voxtral WASM inference engine.
//!
//! Wraps voxtral-core's Q4 inference pipeline with wasm-bindgen exports
//! for running Voxtral-Mini-4B-Realtime directly in the browser via WebGPU.
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
use burn::tensor::{Tensor, TensorData};

use voxtral_core::audio::mel::{MelConfig, MelSpectrogram};
use voxtral_core::audio::pad::{pad_audio, PadConfig};
use voxtral_core::audio::AudioBuffer;
use voxtral_core::q4::loader::Q4ModelLoader;
use voxtral_core::q4::model::Q4VoxtralModel;
use voxtral_core::q4::WgpuBackend;
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

/// Compute sinusoidal time embedding for transcription delay.
/// Matches voxtral-mini-realtime-rs TimeEmbedding::embed().
fn compute_time_embedding(t: f32, dim: usize, device: &WgpuDevice) -> Tensor<WgpuBackend, 3> {
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

/// WASM inference engine — holds loaded model weights and session state.
#[wasm_bindgen]
pub struct WasmEngine {
    model: Q4VoxtralModel,
    tokenizer: TekkenDecoder,
    device: WgpuDevice,
    mel_spec: MelSpectrogram,
    pad_config: PadConfig,
    audio_buffer: Vec<f32>,
    commit_count: usize,
    prev_text: String,
}

#[wasm_bindgen]
impl WasmEngine {
    /// Create a new engine from pre-loaded model shards and tokenizer JSON.
    ///
    /// `shard_arrays` — Array of Uint8Array, each a 64MB GGUF shard
    /// `tokenizer_json` — Contents of tekken.json
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

        Ok(WasmEngine {
            model,
            tokenizer,
            device,
            mel_spec: MelSpectrogram::new(MelConfig::default()),
            pad_config: PadConfig::q4(),
            audio_buffer: Vec::new(),
            commit_count: 0,
            prev_text: String::new(),
        })
    }

    /// Append 16kHz mono f32 PCM audio samples.
    #[wasm_bindgen]
    pub fn send_audio(&mut self, samples: &[f32]) {
        self.audio_buffer.extend_from_slice(samples);
    }

    /// Process all buffered audio and return the text delta (new text since last commit).
    ///
    /// This is async because WebGPU readback requires async on WASM.
    #[wasm_bindgen]
    pub async fn commit(&mut self) -> Result<String, JsValue> {
        if self.audio_buffer.is_empty() {
            web_sys::console::log_1(&"[wasm-engine] commit: audio buffer empty, skipping".into());
            return Ok(String::new());
        }

        self.commit_count += 1;

        web_sys::console::log_1(
            &format!(
                "[wasm-engine] commit #{}: {} samples ({:.2}s audio)",
                self.commit_count,
                self.audio_buffer.len(),
                self.audio_buffer.len() as f64 / 16000.0
            )
            .into(),
        );

        // Pad audio
        let audio_buf = AudioBuffer::new(self.audio_buffer.clone(), 16000);
        let padded = pad_audio(&audio_buf, &self.pad_config);

        // Compute mel spectrogram
        let log_mel = self.mel_spec.compute_log(&padded.samples);
        let n_frames = log_mel.len();
        let n_mels = if n_frames > 0 { log_mel[0].len() } else { 128 };

        if n_frames == 0 {
            web_sys::console::log_1(&"[wasm-engine] commit: 0 mel frames, skipping".into());
            return Ok(String::new());
        }

        web_sys::console::log_1(
            &format!(
                "[wasm-engine] commit #{}: padded {}→{} samples, {} mel frames",
                self.commit_count,
                self.audio_buffer.len(),
                padded.samples.len(),
                n_frames
            )
            .into(),
        );

        // Build tensor [1, n_mels, n_frames] (column-major: mel bins are rows)
        let mut flat = vec![0.0f32; n_mels * n_frames];
        for (t, frame) in log_mel.iter().enumerate() {
            for (m, &val) in frame.iter().enumerate() {
                flat[m * n_frames + t] = val;
            }
        }

        let mel_tensor: Tensor<WgpuBackend, 3> =
            Tensor::from_data(TensorData::new(flat, [1, n_mels, n_frames]), &self.device);

        // Time embedding (6 tokens = 480ms transcription delay)
        let t_embed = compute_time_embedding(6.0, 3072, &self.device);

        web_sys::console::log_1(
            &"[wasm-engine] commit: starting transcribe_streaming_async...".into(),
        );

        // Run inference (async version for WASM — WebGPU requires async GPU readback)
        let token_ids = self
            .model
            .transcribe_streaming_async(mel_tensor, t_embed)
            .await
            .map_err(|e| JsValue::from_str(&format!("Inference failed: {}", e)))?;

        // Decode tokens to text
        let full_text = self.tokenizer.decode(&token_ids);

        web_sys::console::log_1(
            &format!(
                "[wasm-engine] Commit #{}: {}samples → {}frames → {} tokens ids={:?} text={:?}",
                self.commit_count,
                self.audio_buffer.len(),
                n_frames,
                token_ids.len(),
                &token_ids[..token_ids.len().min(30)],
                &full_text[..full_text.len().min(80)]
            )
            .into(),
        );

        // Compute delta (new text since last commit)
        let delta = if full_text.len() > self.prev_text.len()
            && full_text.starts_with(&self.prev_text)
        {
            full_text[self.prev_text.len()..].to_string()
        } else {
            full_text.clone()
        };

        self.prev_text = full_text;

        // Clear audio buffer after processing (match native Q4 engine behavior).
        // Each commit processes only its own chunk independently.
        self.audio_buffer.clear();

        Ok(delta)
    }

    /// Reset session state (clear audio buffer, text history).
    #[wasm_bindgen]
    pub fn reset(&mut self) {
        self.audio_buffer.clear();
        self.prev_text.clear();
        self.commit_count = 0;
        web_sys::console::log_1(&"[wasm-engine] Session reset".into());
    }

    /// Number of commits since last reset.
    #[wasm_bindgen]
    pub fn commit_count(&self) -> usize {
        self.commit_count
    }

    /// Split text into sentences (for subtitle display).
    #[wasm_bindgen]
    pub fn split_sentences_js(text: &str) -> Vec<String> {
        split_sentences(text)
    }
}
