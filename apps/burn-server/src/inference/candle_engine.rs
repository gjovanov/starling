//! Candle CUDA inference engine — bf16 with cuBLAS tensor cores, eager execution.
//!
//! Uses burn's Candle backend which routes matmul to cuBLAS (automatic tensor cores)
//! and supports native bf16 storage. Eager execution eliminates lazy sync overhead.

#[cfg(feature = "candle")]
mod inner {
    use std::sync::{Arc, Mutex};
    use std::time::Instant;

    use crate::audio::mel::{MelConfig, MelSpectrogram};
    use crate::audio::pad::{pad_audio, PadConfig};
    use crate::audio::AudioBuffer;
    use crate::inference::bf16::model::VoxtralModel;
    use crate::inference::bf16::weights::OwnedSafeTensors;
    use crate::inference::tokenizer::TekkenDecoder;
    use crate::inference::{InferenceEngine, InferenceSession};

    use burn::tensor::{Tensor, TensorData};

    pub type CandleBackendType = crate::inference::q4::CandleBackend;
    type CandleDevice = burn::backend::candle::CandleDevice;

    fn compute_time_embedding(t: f32, dim: usize, device: &CandleDevice) -> Tensor<CandleBackendType, 3> {
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

    fn gpu_sync_candle() {
        // Candle is eager — no lazy graph to flush. Light sync only.
        std::thread::sleep(std::time::Duration::from_millis(1));
    }

    pub struct CandleEngine {
        model: Arc<Mutex<VoxtralModel<CandleBackendType>>>,
        tokenizer: Arc<TekkenDecoder>,
        device: CandleDevice,
    }

    impl CandleEngine {
        pub fn load(
            model_dir: &std::path::Path,
            tokenizer_path: &std::path::Path,
        ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
            let device = CandleDevice::cuda(0);

            let st_path = model_dir.join("consolidated.safetensors");
            eprintln!("[CandleEngine] Loading SafeTensors from {}", st_path.display());
            let owned = OwnedSafeTensors::from_file(&st_path)
                .map_err(|e| format!("SafeTensors load: {}", e))?;

            eprintln!("[CandleEngine] Loading full model (bf16, cuBLAS tensor cores)...");
            let model = crate::inference::bf16::loader::load_full_model(
                &owned, &device, gpu_sync_candle,
            ).map_err(|e| format!("Model load: {}", e))?;
            eprintln!("[CandleEngine] Full model loaded (~8 GB VRAM, bf16)");

            let tokenizer = TekkenDecoder::from_file(tokenizer_path)
                .map_err(|e| format!("Tokenizer: {}", e))?;
            eprintln!("[CandleEngine] Tokenizer loaded ({} vocab)", tokenizer.vocab_size());

            Ok(Self {
                model: Arc::new(Mutex::new(model)),
                tokenizer: Arc::new(tokenizer),
                device,
            })
        }
    }

    impl InferenceEngine for CandleEngine {
        fn create_session(
            &self,
            language: &str,
        ) -> Result<Box<dyn InferenceSession>, Box<dyn std::error::Error + Send + Sync>> {
            Ok(Box::new(CandleSession {
                model: self.model.clone(),
                tokenizer: self.tokenizer.clone(),
                device: self.device.clone(),
                _language: language.to_string(),
                mel_spec: MelSpectrogram::new(MelConfig::default()),
                pad_config: PadConfig::bf16(),
                audio_buffer: Vec::new(),
                commit_count: 0,
                prev_text: String::new(),
            }))
        }
    }

    pub struct CandleSession {
        model: Arc<Mutex<VoxtralModel<CandleBackendType>>>,
        tokenizer: Arc<TekkenDecoder>,
        device: CandleDevice,
        _language: String,
        mel_spec: MelSpectrogram,
        pad_config: PadConfig,
        audio_buffer: Vec<f32>,
        commit_count: usize,
        prev_text: String,
    }

    impl InferenceSession for CandleSession {
        fn send_audio(&mut self, samples: &[f32]) {
            self.audio_buffer.extend_from_slice(samples);
        }

        fn commit(&mut self) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
            if self.audio_buffer.is_empty() {
                return Ok(String::new());
            }
            self.commit_count += 1;

            const MAX_AUDIO_SECS: f32 = 15.0;
            const MAX_AUDIO_SAMPLES: usize = (16000.0 * MAX_AUDIO_SECS) as usize;
            let min_samples = 50000;
            if self.audio_buffer.len() < min_samples {
                return Ok(String::new());
            }

            let t0 = Instant::now();
            let start = if self.audio_buffer.len() > MAX_AUDIO_SAMPLES {
                self.audio_buffer.len() - MAX_AUDIO_SAMPLES
            } else { 0 };
            let window = &self.audio_buffer[start..];

            let audio_buf = AudioBuffer::new(window.to_vec(), 16000);
            let padded = pad_audio(&audio_buf, &self.pad_config);
            let log_mel = self.mel_spec.compute_log(&padded.samples);
            let n_frames = log_mel.len();
            let n_mels = if n_frames > 0 { log_mel[0].len() } else { 128 };
            if n_frames == 0 { return Ok(String::new()); }

            let mut flat = vec![0.0f32; n_mels * n_frames];
            for (t, frame) in log_mel.iter().enumerate() {
                for (m, &val) in frame.iter().enumerate() {
                    flat[m * n_frames + t] = val;
                }
            }
            let mel_tensor: Tensor<CandleBackendType, 3> =
                Tensor::from_data(TensorData::new(flat, [1, n_mels, n_frames]), &self.device);
            let t_embed = compute_time_embedding(6.0, 3072, &self.device);

            let model = self.model.lock().map_err(|e| format!("lock: {}", e))?;
            let token_ids = crate::inference::bf16::loader::transcribe_resident::<CandleBackendType>(
                &model, &self.device, mel_tensor, t_embed,
            ).map_err(|e| format!("Candle transcribe: {}", e))?;

            let full_text = self.tokenizer.decode(&token_ids);
            let audio_secs = window.len() as f32 / 16000.0;
            let total_secs = self.audio_buffer.len() as f32 / 16000.0;
            let infer_ms = t0.elapsed().as_secs_f32() * 1000.0;
            eprintln!(
                "[Candle] #{} {:.1}s/{:.1}s audio → {:.0}ms ({} tok)",
                self.commit_count, audio_secs, total_secs, infer_ms, token_ids.len(),
            );

            let delta = if full_text.len() > self.prev_text.len()
                && full_text.starts_with(&self.prev_text)
            {
                full_text[self.prev_text.len()..].to_string()
            } else if !full_text.is_empty() {
                full_text.clone()
            } else {
                String::new()
            };
            self.prev_text = full_text;
            Ok(delta)
        }

        fn reset(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            self.audio_buffer.clear();
            self.prev_text.clear();
            self.commit_count = 0;
            Ok(())
        }

        fn commit_count(&self) -> usize {
            self.commit_count
        }
    }
}

#[cfg(feature = "candle")]
pub use inner::*;
