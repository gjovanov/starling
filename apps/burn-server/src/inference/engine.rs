//! Concrete InferenceEngine/InferenceSession for Q4 Voxtral.
//!
//! Bridges Q4VoxtralModel (Burn) to the streaming InferenceSession trait.

use std::sync::{Arc, Mutex};

use crate::audio::mel::{MelConfig, MelSpectrogram};
use crate::audio::pad::{pad_audio, PadConfig};
use crate::audio::AudioBuffer;

use super::q4::loader::Q4ModelLoader;
use super::q4::model::Q4VoxtralModel;
use super::q4::WgpuBackend;
use super::tokenizer::TekkenDecoder;
use super::{InferenceEngine, InferenceSession};

use burn::backend::wgpu::WgpuDevice;
use burn::tensor::{Tensor, TensorData};

/// Q4 inference engine — holds the loaded model weights.
pub struct Q4Engine {
    model: Arc<Mutex<Q4VoxtralModel>>,
    tokenizer: Arc<TekkenDecoder>,
    device: WgpuDevice,
}

unsafe impl Sync for Q4Engine {}

impl Q4Engine {
    pub fn load(
        gguf_path: &std::path::Path,
        tokenizer_path: &std::path::Path,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let device = WgpuDevice::default();

        eprintln!("[Q4Engine] Loading model from {}...", gguf_path.display());
        let mut loader = Q4ModelLoader::from_file(gguf_path)
            .map_err(|e| format!("Failed to open GGUF: {}", e))?;
        let model = loader
            .load(&device)
            .map_err(|e| format!("Failed to load Q4 model: {}", e))?;
        eprintln!("[Q4Engine] Model loaded");

        eprintln!(
            "[Q4Engine] Loading tokenizer from {}...",
            tokenizer_path.display()
        );
        let tokenizer = TekkenDecoder::from_file(tokenizer_path)?;
        eprintln!(
            "[Q4Engine] Tokenizer loaded ({} vocab)",
            tokenizer.vocab_size()
        );

        Ok(Self {
            model: Arc::new(Mutex::new(model)),
            tokenizer: Arc::new(tokenizer),
            device,
        })
    }
}

impl InferenceEngine for Q4Engine {
    fn create_session(
        &self,
        language: &str,
    ) -> Result<Box<dyn InferenceSession>, Box<dyn std::error::Error + Send + Sync>> {
        Ok(Box::new(Q4Session {
            model: self.model.clone(),
            tokenizer: self.tokenizer.clone(),
            device: self.device.clone(),
            _language: language.to_string(),
            mel_spec: MelSpectrogram::new(MelConfig::default()),
            pad_config: PadConfig::q4(),
            audio_buffer: Vec::new(),
            commit_count: 0,
            prev_text: String::new(),
        }))
    }
}

/// Q4 streaming inference session.
///
/// Each `commit()` processes accumulated audio: PCM → pad → mel → model → tokens → text.
pub struct Q4Session {
    model: Arc<Mutex<Q4VoxtralModel>>,
    tokenizer: Arc<TekkenDecoder>,
    device: WgpuDevice,
    _language: String,
    mel_spec: MelSpectrogram,
    pad_config: PadConfig,
    audio_buffer: Vec<f32>,
    commit_count: usize,
    prev_text: String,
}

impl Q4Session {
    fn decode_tokens(&self, token_ids: &[i32]) -> String {
        self.tokenizer.decode(token_ids)
    }
}

impl InferenceSession for Q4Session {
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

        let mel_tensor: Tensor<WgpuBackend, 3> =
            Tensor::from_data(TensorData::new(flat, [1, n_mels, n_frames]), &self.device);

        // t-embedding (zeros — t-conditioning value is 0.0 for streaming)
        // The TimeEmbedding produces a d_model-dimensional vector (3072)
        let t_embed: Tensor<WgpuBackend, 3> = Tensor::zeros([1, 1, 3072], &self.device);

        // Run inference (lock the model)
        let token_ids = {
            let model = self.model.lock().map_err(|e| format!("Model lock: {}", e))?;
            model.transcribe_streaming(mel_tensor, t_embed)
        };

        // Decode tokens
        let full_text = self.decode_tokens(&token_ids);

        eprintln!(
            "[Q4Commit] #{} audio={}samples padded={}samples mel={}frames tokens={} ids={:?} text={:?}",
            self.commit_count,
            audio_buf.samples.len(),
            padded.samples.len(),
            n_frames,
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
