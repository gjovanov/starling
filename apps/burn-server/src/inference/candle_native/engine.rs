//! Pure candle CUDA inference engine — implements InferenceSession/InferenceEngine traits.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use candle_core_native::{DType, Device, Tensor};

use crate::audio::mel::{MelConfig, MelSpectrogram};
use crate::audio::pad::{pad_audio, PadConfig};
use crate::audio::AudioBuffer;
use crate::inference::tokenizer::TekkenDecoder;
use crate::inference::{InferenceEngine, InferenceSession};

use super::model::{self, VoxtralModel};

pub struct CandleNativeEngine {
    model: Arc<Mutex<VoxtralModel>>,
    tokenizer: Arc<TekkenDecoder>,
    device: Device,
}

impl CandleNativeEngine {
    pub fn load(
        model_dir: &std::path::Path,
        tokenizer_path: &std::path::Path,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let device = Device::new_cuda(0).map_err(|e| format!("CUDA device: {}", e))?;

        let st_path = model_dir.join("consolidated.safetensors");
        eprintln!("[CandleNativeEngine] Loading from {}", st_path.display());
        let model = VoxtralModel::load(&st_path, &device)
            .map_err(|e| format!("Model load: {}", e))?;

        let tokenizer = TekkenDecoder::from_file(tokenizer_path)
            .map_err(|e| format!("Tokenizer: {}", e))?;
        eprintln!("[CandleNativeEngine] Tokenizer loaded ({} vocab)", tokenizer.vocab_size());

        Ok(Self {
            model: Arc::new(Mutex::new(model)),
            tokenizer: Arc::new(tokenizer),
            device,
        })
    }
}

impl InferenceEngine for CandleNativeEngine {
    fn create_session(
        &self,
        language: &str,
    ) -> Result<Box<dyn InferenceSession>, Box<dyn std::error::Error + Send + Sync>> {
        Ok(Box::new(CandleNativeSession {
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

pub struct CandleNativeSession {
    model: Arc<Mutex<VoxtralModel>>,
    tokenizer: Arc<TekkenDecoder>,
    device: Device,
    _language: String,
    mel_spec: MelSpectrogram,
    pad_config: PadConfig,
    audio_buffer: Vec<f32>,
    commit_count: usize,
    prev_text: String,
}

impl InferenceSession for CandleNativeSession {
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
        } else {
            0
        };
        let window = &self.audio_buffer[start..];

        // Pad + mel
        let audio_buf = AudioBuffer::new(window.to_vec(), 16000);
        let padded = pad_audio(&audio_buf, &self.pad_config);
        let log_mel = self.mel_spec.compute_log(&padded.samples);
        let n_frames = log_mel.len();
        let n_mels = if n_frames > 0 { log_mel[0].len() } else { 128 };
        if n_frames == 0 {
            return Ok(String::new());
        }

        // Build mel tensor [1, n_mels, n_frames] as bf16
        let mut flat = vec![0.0f32; n_mels * n_frames];
        for (t, frame) in log_mel.iter().enumerate() {
            for (m, &val) in frame.iter().enumerate() {
                flat[m * n_frames + t] = val;
            }
        }
        // F32 — matching burn-candle which operates entirely in f32
        let mel_tensor = Tensor::new(flat, &self.device)
            .and_then(|t| t.reshape((1, n_mels, n_frames)))
            .map_err(|e| format!("Mel tensor: {}", e))?;

        let t_embed = model::compute_time_embedding(6.0, 3072, &self.device)
            .map_err(|e| format!("t_embed: {}", e))?;

        // Run inference
        let model = self.model.lock().map_err(|e| format!("lock: {}", e))?;
        let token_ids = model::transcribe(&model, &mel_tensor, &t_embed)
            .map_err(|e| format!("CandleNative transcribe: {}", e))?;

        let full_text = self.tokenizer.decode(
            &token_ids.iter().map(|&t| t as i32).collect::<Vec<_>>(),
        );
        let audio_secs = window.len() as f32 / 16000.0;
        let total_secs = self.audio_buffer.len() as f32 / 16000.0;
        let infer_ms = t0.elapsed().as_secs_f32() * 1000.0;
        eprintln!(
            "[CandleNative] #{} {:.1}s/{:.1}s audio → {:.0}ms ({} tok)",
            self.commit_count, audio_secs, total_secs, infer_ms, token_ids.len(),
        );

        // Compute delta
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
