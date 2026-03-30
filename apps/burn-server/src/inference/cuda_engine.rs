//! CUDA inference engine — frame-by-frame streaming matching vllm architecture.
//!
//! Each 0.5s commit processes ~6 small frames (~135ms each) through the full
//! encoder independently. The decoder KV cache persists across frames/commits.

#[cfg(feature = "cuda")]
mod inner {
    use std::sync::{Arc, Mutex};
    use std::time::Instant;

    use crate::audio::mel::{MelConfig, MelSpectrogram};
    use crate::audio::pad::{pad_audio, PadConfig};
    use crate::audio::AudioBuffer;
    use crate::inference::bf16::layers::*;
    use crate::inference::bf16::model::VoxtralModel;
    use crate::inference::bf16::streaming::{FrameGenerator, StreamConfig};
    use crate::inference::bf16::weights::OwnedSafeTensors;
    use crate::inference::tokenizer::TekkenDecoder;
    use crate::inference::{InferenceEngine, InferenceSession};

    use burn::tensor::{Tensor, TensorData};

    pub type CudaBackendType = crate::inference::q4::CudaBackend;
    pub type CudaDevice = burn::backend::cuda::CudaDevice;

    fn compute_time_embedding(t: f32, dim: usize, device: &CudaDevice) -> Tensor<CudaBackendType, 3> {
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

    fn gpu_sync_cuda() {
        use cubecl::Runtime;
        let device = cubecl::cuda::CudaDevice::default();
        let client = cubecl::cuda::CudaRuntime::client(&device);
        client.flush();
        std::thread::sleep(std::time::Duration::from_millis(5));
    }

    pub struct CudaEngine {
        model: Arc<Mutex<VoxtralModel<CudaBackendType>>>,
        tokenizer: Arc<TekkenDecoder>,
        device: CudaDevice,
    }

    impl CudaEngine {
        pub fn load(
            model_dir: &std::path::Path,
            tokenizer_path: &std::path::Path,
        ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
            let device = CudaDevice::default();
            let st_path = model_dir.join("consolidated.safetensors");

            eprintln!("[CudaEngine] Loading SafeTensors from {}", st_path.display());
            let owned = crate::inference::bf16::weights::OwnedSafeTensors::from_file(&st_path)
                .map_err(|e| format!("SafeTensors load: {}", e))?;

            eprintln!("[CudaEngine] Loading full model (all layers resident on GPU)...");
            let model = crate::inference::bf16::loader::load_full_model(
                &owned, &device, gpu_sync_cuda,
            ).map_err(|e| format!("Model load: {}", e))?;
            eprintln!("[CudaEngine] Full model loaded (~16 GB VRAM)");

            let tokenizer = TekkenDecoder::from_file(tokenizer_path)
                .map_err(|e| format!("Tokenizer: {}", e))?;
            eprintln!("[CudaEngine] Tokenizer loaded ({} vocab)", tokenizer.vocab_size());

            Ok(Self {
                model: Arc::new(Mutex::new(model)),
                tokenizer: Arc::new(tokenizer),
                device,
            })
        }
    }

    impl InferenceEngine for CudaEngine {
        fn create_session(
            &self,
            language: &str,
        ) -> Result<Box<dyn InferenceSession>, Box<dyn std::error::Error + Send + Sync>> {
            Ok(Box::new(CudaSession {
                model: self.model.clone(),
                tokenizer: self.tokenizer.clone(),
                device: self.device.clone(),
                _language: language.to_string(),
                mel_spec: MelSpectrogram::new(MelConfig::default()),
                audio_buffer: Vec::new(),
                commit_count: 0,
                prev_text: String::new(),
                // Streaming state
                frame_gen: FrameGenerator::new(StreamConfig::default()),
                left_pad: None,
                decoder_caches: None,
                ada_scales: None,
                t_embed: None,
                generated_tokens: Vec::new(),
                prev_token: 1, // BOS
                prefix_fed: false,
                total_decoder_positions: 0,
            }))
        }
    }

    pub struct CudaSession {
        model: Arc<Mutex<VoxtralModel<CudaBackendType>>>,
        tokenizer: Arc<TekkenDecoder>,
        device: CudaDevice,
        _language: String,
        mel_spec: MelSpectrogram,
        audio_buffer: Vec<f32>,
        commit_count: usize,
        prev_text: String,

        // Frame-by-frame streaming state
        frame_gen: FrameGenerator,
        left_pad: Option<Vec<f32>>,
        decoder_caches: Option<LayerCaches<CudaBackendType>>,
        ada_scales: Option<Vec<Tensor<CudaBackendType, 3>>>,
        t_embed: Option<Tensor<CudaBackendType, 3>>,
        generated_tokens: Vec<i32>,
        prev_token: i32,
        prefix_fed: bool,
        total_decoder_positions: usize,
    }

    impl CudaSession {
        fn ensure_initialized(&mut self) {
            if self.t_embed.is_none() {
                self.t_embed = Some(compute_time_embedding(6.0, 3072, &self.device));
            }
            if self.ada_scales.is_none() {
                let model = self.model.lock().unwrap();
                let t = self.t_embed.as_ref().unwrap();
                self.ada_scales = Some(
                    model.decoder.layers.iter()
                        .map(|l| l.precompute_ada_scale(t.clone()))
                        .collect()
                );
            }
            if self.decoder_caches.is_none() {
                self.decoder_caches = Some(
                    LayerCaches::new_preallocated(26, 1, 8, 1024, 128, &self.device)
                );
            }
            if self.left_pad.is_none() {
                // Left padding: 32 tokens × 1280 samples/token = 40960 samples of silence
                let cfg = StreamConfig::default();
                let pad_samples = cfg.n_left_pad_tokens * cfg.step_samples;
                self.left_pad = Some(vec![0.0f32; pad_samples]);
            }
        }

        /// Get the audio with left-padding prepended.
        fn padded_audio(&self) -> Vec<f32> {
            let pad = self.left_pad.as_ref().unwrap();
            let mut out = Vec::with_capacity(pad.len() + self.audio_buffer.len());
            out.extend_from_slice(pad);
            out.extend_from_slice(&self.audio_buffer);
            out
        }
    }

    impl InferenceSession for CudaSession {
        fn send_audio(&mut self, samples: &[f32]) {
            self.audio_buffer.extend_from_slice(samples);
        }

        fn commit(&mut self) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
            if self.audio_buffer.is_empty() {
                return Ok(String::new());
            }
            self.commit_count += 1;
            self.ensure_initialized();

            let t0 = Instant::now();

            // Get new frames to process
            let padded = self.padded_audio();
            let frames = self.frame_gen.next_frames(self.audio_buffer.len());
            if frames.is_empty() {
                return Ok(String::new());
            }

            let model = self.model.lock().map_err(|e| format!("lock: {}", e))?;
            let dec = &model.decoder;
            let ada_scales = self.ada_scales.as_ref().unwrap();
            let caches = self.decoder_caches.as_mut().unwrap();
            let d_model = dec.d_model;

            let mut new_tokens: Vec<i32> = Vec::new();

            for frame in &frames {
                // Get the audio window for this frame
                let audio_window = &padded[frame.audio_start..frame.audio_end];

                // Encode frame (small one-shot: mel → conv → 32 layers → norm → reshape → adapter)
                let frame_embeds = match model.encode_frame(audio_window, &self.mel_spec, &self.device) {
                    Some(e) => e,
                    None => continue,
                };
                let [_, num_dec_pos, _] = frame_embeds.dims();

                // Process each decoder position from this frame
                for pos in 0..num_dec_pos {
                    let audio_pos = frame_embeds.clone().slice([0..1, pos..pos + 1, 0..d_model]);

                    if !self.prefix_fed && self.total_decoder_positions < 39 {
                        // Prefix phase: use streaming pad tokens (id=32) as text input
                        let text_embed = dec.embed_token_gpu(
                            if self.total_decoder_positions == 0 { 1 } else { 32 }, // BOS then pad
                            &self.device,
                        );
                        let x = audio_pos + text_embed;
                        let _hidden = dec.forward_hidden_with_cache_fast(x, ada_scales, caches);
                        self.total_decoder_positions += 1;

                        // At position 39, do the first real decode
                        if self.total_decoder_positions == 39 {
                            self.prefix_fed = true;
                            // Get first token from last hidden state
                            let last = _hidden.slice([0..1, 0..1, 0..d_model]);
                            let logits = dec.lm_head_gpu(last);
                            let argmax_data = logits.argmax(2).into_data();
                            let token: i32 = argmax_data.as_slice::<i32>()
                                .map(|v| v[0])
                                .or_else(|_| argmax_data.as_slice::<i64>().map(|v| v[0] as i32))
                                .unwrap_or(0);
                            self.prev_token = token;
                            new_tokens.push(token);
                            self.generated_tokens.push(token);
                        }
                    } else {
                        // Normal decode: use previous token's embedding
                        let text_embed = dec.embed_token_gpu(self.prev_token, &self.device);
                        let x = audio_pos + text_embed;
                        let hidden = dec.forward_hidden_with_cache_fast(x, ada_scales, caches);

                        // GPU lm_head + argmax
                        let last = hidden.slice([0..1, 0..1, 0..d_model]);
                        let logits = dec.lm_head_gpu(last);
                        let argmax_data = logits.argmax(2).into_data();
                        let token: i32 = argmax_data.as_slice::<i32>()
                            .map(|v| v[0])
                            .or_else(|_| argmax_data.as_slice::<i64>().map(|v| v[0] as i32))
                            .unwrap_or(0);
                        self.prev_token = token;
                        new_tokens.push(token);
                        self.generated_tokens.push(token);
                        self.total_decoder_positions += 1;
                    }
                }
            }

            let infer_ms = t0.elapsed().as_secs_f32() * 1000.0;
            let audio_secs = self.audio_buffer.len() as f32 / 16000.0;
            eprintln!(
                "[CUDA-Stream] #{} {:.1}s audio, {} frames, {} new tokens, {:.0}ms",
                self.commit_count, audio_secs, frames.len(), new_tokens.len(), infer_ms,
            );

            // Decode new tokens to text
            if new_tokens.is_empty() {
                return Ok(String::new());
            }
            let new_text = self.tokenizer.decode(&new_tokens);

            // Don't clear audio_buffer — it accumulates for frame generator
            Ok(new_text)
        }

        fn reset(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            self.audio_buffer.clear();
            self.prev_text.clear();
            self.commit_count = 0;
            self.frame_gen.reset();
            self.decoder_caches = None;
            self.ada_scales = None;
            self.generated_tokens.clear();
            self.prev_token = 1;
            self.prefix_fed = false;
            self.total_decoder_positions = 0;
            Ok(())
        }

        fn commit_count(&self) -> usize {
            self.commit_count
        }
    }
}

#[cfg(feature = "cuda")]
pub use inner::*;
