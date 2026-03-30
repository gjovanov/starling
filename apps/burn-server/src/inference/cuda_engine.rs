//! CUDA inference engine — full model resident on GPU.
//!
//! All 58 layers (32 encoder + 26 decoder) stay on GPU (~16 GB f32).
//! No per-layer streaming needed — CUDA has full 24 GB VRAM access.

#[cfg(feature = "cuda")]
mod inner {
    use std::sync::{Arc, Mutex};

    use crate::audio::mel::{MelConfig, MelSpectrogram};
    use crate::audio::pad::{pad_audio, PadConfig};
    use crate::audio::AudioBuffer;
    use crate::inference::bf16::model::VoxtralModel;
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
            let owned = OwnedSafeTensors::from_file(&st_path)
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
                pad_config: PadConfig::bf16(),
                audio_buffer: Vec::new(),
                commit_count: 0,
                prev_text: String::new(),
            }))
        }
    }

    pub struct CudaSession {
        model: Arc<Mutex<VoxtralModel<CudaBackendType>>>,
        tokenizer: Arc<TekkenDecoder>,
        device: CudaDevice,
        _language: String,
        mel_spec: MelSpectrogram,
        pad_config: PadConfig,
        audio_buffer: Vec<f32>,
        commit_count: usize,
        prev_text: String,
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

            let audio_buf = AudioBuffer::new(self.audio_buffer.clone(), 16000);
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
            let mel_tensor: Tensor<CudaBackendType, 3> =
                Tensor::from_data(TensorData::new(flat, [1, n_mels, n_frames]), &self.device);
            let t_embed = compute_time_embedding(6.0, 3072, &self.device);

            let model = self.model.lock().map_err(|e| format!("lock: {}", e))?;
            let token_ids = crate::inference::bf16::loader::transcribe_resident::<CudaBackendType>(
                &model, &self.device, mel_tensor, t_embed,
            ).map_err(|e| format!("CUDA transcribe: {}", e))?;

            let full_text = self.tokenizer.decode(&token_ids);
            eprintln!(
                "[CUDA] #{} tokens={} text={:?}",
                self.commit_count, token_ids.len(),
                &full_text[..full_text.char_indices().take(80).last().map_or(0, |(i, c)| i + c.len_utf8())]
            );

            let delta = if full_text.len() > self.prev_text.len()
                && full_text.starts_with(&self.prev_text)
            {
                full_text[self.prev_text.len()..].to_string()
            } else {
                full_text.clone()
            };
            self.prev_text = full_text;
            self.audio_buffer.clear();
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

#[cfg(feature = "cuda")]
pub use inner::*;
