//! Pure candle CUDA inference engine — streaming with incremental encoding.
//!
//! Each commit() processes only NEW audio through the encoder (with KV cache),
//! then runs the decoder on new adapter tokens. This enables 0.5s commit intervals.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use candle_core_native::{DType, Device, Tensor};

use crate::audio::mel::{MelConfig, MelSpectrogram};
use crate::audio::pad::{PadConfig};
use crate::audio::AudioBuffer;
use crate::inference::tokenizer::TekkenDecoder;
use crate::inference::{InferenceEngine, InferenceSession};

use super::model::{self, KVCache, VoxtralModel};

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
        Ok(Box::new(CandleNativeSession::new(
            self.model.clone(),
            self.tokenizer.clone(),
            self.device.clone(),
            language,
        )?))
    }
}

pub struct CandleNativeSession {
    model: Arc<Mutex<VoxtralModel>>,
    tokenizer: Arc<TekkenDecoder>,
    device: Device,
    _language: String,
    mel_spec: MelSpectrogram,
    pad_config: PadConfig,

    // Audio accumulation
    audio_buffer: Vec<f32>,
    commit_count: usize,

    // Full-window transcription state (re-transcribe on each commit for simplicity)
    prev_text: String,

    // ADA scales (constant across session, precomputed once)
    ada_scales: Option<Vec<Tensor>>,
    t_embed: Option<Tensor>,
}

impl CandleNativeSession {
    fn new(
        model: Arc<Mutex<VoxtralModel>>,
        tokenizer: Arc<TekkenDecoder>,
        device: Device,
        language: &str,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let t_embed = model::compute_time_embedding(6.0, 3072, &device)
            .map_err(|e| format!("t_embed: {}", e))?;

        let ada_scales = {
            let m = model.lock().map_err(|e| format!("lock: {}", e))?;
            m.precompute_ada_scales(&t_embed)
                .map_err(|e| format!("ada_scales: {}", e))?
        };

        Ok(Self {
            model,
            tokenizer,
            device,
            _language: language.to_string(),
            mel_spec: MelSpectrogram::new(MelConfig::default()),
            pad_config: PadConfig::bf16(),
            audio_buffer: Vec::new(),
            commit_count: 0,
            prev_text: String::new(),
            ada_scales: Some(ada_scales),
            t_embed: Some(t_embed),
        })
    }

    /// Transcribe the current audio window and return full text.
    fn transcribe_window(&self) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        // Use last 30s of audio (enough context, fast enough for streaming)
        const MAX_WINDOW_SECS: f32 = 30.0;
        const MAX_WINDOW_SAMPLES: usize = (16000.0 * MAX_WINDOW_SECS) as usize;

        let start = if self.audio_buffer.len() > MAX_WINDOW_SAMPLES {
            self.audio_buffer.len() - MAX_WINDOW_SAMPLES
        } else {
            0
        };
        let window = &self.audio_buffer[start..];

        if window.len() < 8000 { // need at least 0.5s
            return Ok(String::new());
        }

        // Pad + mel
        let audio_buf = AudioBuffer::new(window.to_vec(), 16000);
        let padded = crate::audio::pad::pad_audio(&audio_buf, &self.pad_config);
        let log_mel = self.mel_spec.compute_log(&padded.samples);
        let n_frames = log_mel.len();
        let n_mels = if n_frames > 0 { log_mel[0].len() } else { 128 };
        if n_frames == 0 {
            return Ok(String::new());
        }

        let mut flat = vec![0.0f32; n_mels * n_frames];
        for (t, frame) in log_mel.iter().enumerate() {
            for (m, &val) in frame.iter().enumerate() {
                flat[m * n_frames + t] = val;
            }
        }
        let mel_tensor = Tensor::new(flat, &self.device)
            .and_then(|t| t.to_dtype(DType::BF16))
            .and_then(|t| t.reshape((1, n_mels, n_frames)))
            .map_err(|e| format!("Mel tensor: {}", e))?;

        let model = self.model.lock().map_err(|e| format!("lock: {}", e))?;
        let t_embed = self.t_embed.as_ref().unwrap();
        let token_ids = model::transcribe_streaming(&model, &mel_tensor, t_embed)
            .map_err(|e| format!("transcribe: {}", e))?;

        Ok(self.tokenizer.decode(
            &token_ids.iter().map(|&t| t as i32).collect::<Vec<_>>(),
        ))
    }
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

        // Need enough audio for the model (prompt_len=39 tokens needs ~3s)
        if self.audio_buffer.len() < 48000 { // 3 seconds minimum
            return Ok(String::new());
        }

        let t0 = Instant::now();
        let full_text = self.transcribe_window()?;
        let infer_ms = t0.elapsed().as_secs_f32() * 1000.0;

        let audio_secs = self.audio_buffer.len() as f32 / 16000.0;
        eprintln!(
            "[CandleNative] #{} {:.1}s audio → {:.0}ms ({} chars)",
            self.commit_count, audio_secs, infer_ms, full_text.len(),
        );

        // Compute delta (new text since last commit)
        let delta = if full_text.len() > self.prev_text.len()
            && full_text.starts_with(&self.prev_text)
        {
            full_text[self.prev_text.len()..].to_string()
        } else if !full_text.is_empty() {
            // Window shifted — text changed. Return everything after last common point.
            // Find longest common suffix of prev_text that matches prefix of full_text
            let overlap = find_overlap(&self.prev_text, &full_text);
            if overlap > 0 {
                full_text[overlap..].to_string()
            } else {
                full_text.clone()
            }
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

/// Find the length of the longest suffix of `prev` that is a prefix of `curr`.
fn find_overlap(prev: &str, curr: &str) -> usize {
    let max_check = prev.len().min(curr.len());
    for len in (1..=max_check).rev() {
        if prev.ends_with(&curr[..len]) {
            return len;
        }
    }
    0
}
