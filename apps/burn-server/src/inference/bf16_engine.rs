//! BF16 inference engine — loads SafeTensors, runs full-precision inference.

use std::sync::{Arc, Mutex};

use crate::audio::mel::{MelConfig, MelSpectrogram};
use crate::audio::pad::{pad_audio, PadConfig};
use crate::audio::AudioBuffer;
use crate::inference::bf16::model::VoxtralModel;
use crate::inference::tokenizer::TekkenDecoder;
use crate::inference::{InferenceEngine, InferenceSession};

use burn::backend::wgpu::WgpuDevice;
use burn::tensor::{Tensor, TensorData};

/// Backend type for BF16 inference — non-fused wgpu (same as Q4).
/// The fused `Wgpu` backend causes trait solver overflow with `Send + Sync`.
pub type Bf16Backend = crate::inference::q4::WgpuBackend;

/// Compute sinusoidal time embedding (same as Q4 engine).
fn compute_time_embedding(t: f32, dim: usize, device: &WgpuDevice) -> Tensor<Bf16Backend, 3> {
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

/// BF16 inference engine — loads the full model from SafeTensors.
pub struct Bf16Engine {
    model: Arc<Mutex<VoxtralModel<Bf16Backend>>>,
    tokenizer: Arc<TekkenDecoder>,
    device: WgpuDevice,
}

impl Bf16Engine {
    pub fn load(
        model_dir: &std::path::Path,
        tokenizer_path: &std::path::Path,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let device = WgpuDevice::default();

        let model = crate::inference::bf16::loader::load_model::<Bf16Backend>(model_dir, &device)
            .map_err(|e| format!("BF16 model load failed: {}", e))?;

        eprintln!("[BF16Engine] Model loaded");

        let tokenizer = TekkenDecoder::from_file(tokenizer_path)
            .map_err(|e| format!("Tokenizer load failed: {}", e))?;

        eprintln!(
            "[BF16Engine] Tokenizer loaded ({} vocab)",
            tokenizer.vocab_size()
        );

        Ok(Self {
            model: Arc::new(Mutex::new(model)),
            tokenizer: Arc::new(tokenizer),
            device,
        })
    }
}

impl InferenceEngine for Bf16Engine {
    fn create_session(
        &self,
        language: &str,
    ) -> Result<Box<dyn InferenceSession>, Box<dyn std::error::Error + Send + Sync>> {
        Ok(Box::new(Bf16Session {
            model: self.model.clone(),
            tokenizer: self.tokenizer.clone(),
            device: self.device.clone(),
            _language: language.to_string(),
            mel_spec: MelSpectrogram::new(MelConfig::default()),
            pad_config: PadConfig::bf16(), // BF16 uses standard 32-token padding
            audio_buffer: Vec::new(),
            commit_count: 0,
            prev_text: String::new(),
        }))
    }
}

pub struct Bf16Session {
    model: Arc<Mutex<VoxtralModel<Bf16Backend>>>,
    tokenizer: Arc<TekkenDecoder>,
    device: WgpuDevice,
    _language: String,
    mel_spec: MelSpectrogram,
    pad_config: PadConfig,
    audio_buffer: Vec<f32>,
    commit_count: usize,
    prev_text: String,
}

impl InferenceSession for Bf16Session {
    fn send_audio(&mut self, samples: &[f32]) {
        self.audio_buffer.extend_from_slice(samples);
    }

    fn commit(&mut self) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        if self.audio_buffer.is_empty() {
            return Ok(String::new());
        }

        self.commit_count += 1;

        // Pad audio
        let audio_buf = AudioBuffer::new(self.audio_buffer.clone(), 16000);
        let padded = pad_audio(&audio_buf, &self.pad_config);

        // Compute mel spectrogram
        let log_mel = self.mel_spec.compute_log(&padded.samples);
        let n_frames = log_mel.len();
        let n_mels = if n_frames > 0 { log_mel[0].len() } else { 128 };

        if n_frames == 0 {
            return Ok(String::new());
        }

        // Build tensor [1, n_mels, n_frames]
        let mut flat = vec![0.0f32; n_mels * n_frames];
        for (t, frame) in log_mel.iter().enumerate() {
            for (m, &val) in frame.iter().enumerate() {
                flat[m * n_frames + t] = val;
            }
        }

        let mel_tensor: Tensor<Bf16Backend, 3> =
            Tensor::from_data(TensorData::new(flat, [1, n_mels, n_frames]), &self.device);

        // Time embedding (delay=6)
        let t_embed = compute_time_embedding(6.0, 3072, &self.device);

        // Run inference
        let token_ids = {
            let model = self.model.lock().map_err(|e| format!("Model lock: {}", e))?;
            model.transcribe_streaming(mel_tensor, t_embed)
        };

        // Decode tokens (offset by 1000)
        let full_text = self.tokenizer.decode(&token_ids);

        eprintln!(
            "[BF16Commit] #{} tokens={} ids={:?} text={:?}",
            self.commit_count,
            token_ids.len(),
            &token_ids[..token_ids.len().min(20)],
            &full_text[..full_text.len().min(80)]
        );

        // Compute delta
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
