//! Incremental CPU inference engine with Q4 GGUF quantization.
//!
//! Each commit() processes ONLY new audio through the encoder (KV cached),
//! then decodes new adapter tokens. Same architecture as candle_native/engine.rs
//! but targeting CPU with Q4 weights.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use candle_core_cpu::{DType, Device, Module, Tensor};

use crate::audio::mel::{MelConfig, MelSpectrogram};
use crate::audio::pad::PadConfig;
use crate::audio::AudioBuffer;
use crate::inference::tokenizer::TekkenDecoder;
use crate::inference::{InferenceEngine, InferenceSession};

use super::model::{self, KVCache, VoxtralModel, reshape_encoder_output};

pub struct CandleCpuEngine {
    model: Arc<Mutex<VoxtralModel>>,
    tokenizer: Arc<TekkenDecoder>,
}

impl CandleCpuEngine {
    pub fn load(
        gguf_path: &std::path::Path,
        tokenizer_path: &std::path::Path,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let device = Device::Cpu;

        // Set ggml thread count from GGML_THREADS env var or use physical core count
        #[cfg(feature = "candle-cpu-ggml")]
        {
            let threads = std::env::var("GGML_THREADS")
                .ok()
                .and_then(|v| v.parse::<i32>().ok())
                .unwrap_or(16); // default to 16 physical cores (no SMT)
            ggml_matmul::set_threads(threads);
            eprintln!("[CandleCpuEngine] ggml threads: {}", threads);
        }

        eprintln!("[CandleCpuEngine] Loading Q4 GGUF from {}", gguf_path.display());
        let model = VoxtralModel::load(gguf_path, &device)
            .map_err(|e| format!("Model load: {}", e))?;

        let tokenizer = TekkenDecoder::from_file(tokenizer_path)
            .map_err(|e| format!("Tokenizer: {}", e))?;
        eprintln!(
            "[CandleCpuEngine] Tokenizer loaded ({} vocab)",
            tokenizer.vocab_size()
        );

        Ok(Self {
            model: Arc::new(Mutex::new(model)),
            tokenizer: Arc::new(tokenizer),
        })
    }
}

impl InferenceEngine for CandleCpuEngine {
    fn create_session(
        &self,
        language: &str,
    ) -> Result<Box<dyn InferenceSession>, Box<dyn std::error::Error + Send + Sync>> {
        Ok(Box::new(CandleCpuSession::new(
            self.model.clone(),
            self.tokenizer.clone(),
            language,
        )?))
    }
}

/// Incremental streaming session for CPU — keeps encoder + decoder KV caches across commits.
pub struct CandleCpuSession {
    model: Arc<Mutex<VoxtralModel>>,
    tokenizer: Arc<TekkenDecoder>,
    _language: String,
    mel_spec: MelSpectrogram,

    // Audio state
    audio_samples: Vec<f32>,
    commit_count: usize,

    // Incremental encoding state
    processed_conv_frames: usize,
    enc_caches: Vec<KVCache>,
    enc_residual: Vec<f32>,
    enc_residual_count: usize,

    // Decoder state (persistent across commits)
    dec_caches: Vec<KVCache>,
    ada_scales: Vec<Tensor>,
    prev_token: u32,
    decoder_started: bool,
    total_adapter_tokens: usize,

    // Text accumulation
    generated_tokens: Vec<u32>,
}

impl CandleCpuSession {
    fn new(
        model: Arc<Mutex<VoxtralModel>>,
        tokenizer: Arc<TekkenDecoder>,
        language: &str,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let device = Device::Cpu;
        let t_embed = model::compute_time_embedding(6.0, 3072, &device)
            .map_err(|e| format!("t_embed: {}", e))?;

        let ada_scales = {
            let m = model.lock().map_err(|e| format!("lock: {}", e))?;
            m.precompute_ada_scales(&t_embed)
                .map_err(|e| format!("ada_scales: {}", e))?
        };

        // Pre-fill audio with left-pad silence (32 * 1280 = 40960 zeros)
        let left_pad = 32 * 1280;
        let audio_samples = vec![0.0f32; left_pad];

        Ok(Self {
            model,
            tokenizer,
            _language: language.to_string(),
            mel_spec: MelSpectrogram::new(MelConfig::default()),
            audio_samples,
            commit_count: 0,
            processed_conv_frames: 0,
            enc_caches: VoxtralModel::new_encoder_caches(),
            enc_residual: Vec::new(),
            enc_residual_count: 0,
            dec_caches: VoxtralModel::new_decoder_caches(),
            ada_scales,
            prev_token: 0,
            decoder_started: false,
            total_adapter_tokens: 0,
            generated_tokens: Vec::new(),
        })
    }
}

impl InferenceSession for CandleCpuSession {
    fn send_audio(&mut self, samples: &[f32]) {
        self.audio_samples.extend_from_slice(samples);
    }

    fn commit(&mut self) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        self.commit_count += 1;
        let t0 = Instant::now();

        if self.audio_samples.len() < 50000 {
            if self.commit_count <= 3 || self.commit_count % 20 == 0 {
                eprintln!(
                    "[CandleCpu] #{} waiting ({} samples < 50000)",
                    self.commit_count,
                    self.audio_samples.len()
                );
            }
            return Ok(String::new());
        }

        let model = self.model.lock().map_err(|e| format!("lock: {}", e))?;
        let device = Device::Cpu;

        // Step 1: Compute mel
        let log_mel = self.mel_spec.compute_log(&self.audio_samples);
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
        let mel = Tensor::new(flat, &device)
            .and_then(|t| t.reshape((1, n_mels, n_frames)))
            .map_err(|e| format!("mel: {}", e))?;

        // Step 2: Conv on full mel
        let conv_out = model
            .encoder
            .conv
            .forward(&mel)
            .map_err(|e| format!("conv: {}", e))?;
        let total_conv = conv_out.dim(1).map_err(|e| format!("dim: {}", e))?;
        let total_aligned = (total_conv / 4) * 4;

        let safe_boundary = 8;
        let safe_end = if total_aligned > safe_boundary {
            total_aligned - safe_boundary
        } else {
            0
        };
        let safe_aligned = (safe_end / 4) * 4;

        let new_start = self.processed_conv_frames;
        if new_start >= safe_aligned {
            return Ok(String::new());
        }
        let new_conv = conv_out
            .narrow(1, new_start, safe_aligned - new_start)
            .map_err(|e| format!("narrow: {}", e))?;
        self.processed_conv_frames = safe_aligned;

        // Step 3: Encoder on NEW conv frames only
        let new_len = safe_aligned - new_start;
        let sliding_window = 750;
        if self.enc_caches[0].seq_len + new_len > sliding_window {
            let keep = sliding_window / 2;
            for cache in &mut self.enc_caches {
                cache.compact(keep).map_err(|e| format!("compact: {}", e))?;
            }
        }

        let mut x = new_conv;
        for (i, layer) in model.encoder.layers.iter().enumerate() {
            x = layer
                .forward(&x, &model.encoder.rope, &mut self.enc_caches[i])
                .map_err(|e| format!("enc layer {}: {}", i, e))?;
        }
        let enc_out = model
            .encoder
            .norm
            .forward(&x)
            .map_err(|e| format!("enc norm: {}", e))?;

        // Step 4: Adapter (4x reshape) with residual handling
        let to_adapt = if self.enc_residual_count > 0 {
            let residual = Tensor::new(
                self.enc_residual[..self.enc_residual_count * 1280].to_vec(),
                &device,
            )
            .and_then(|t| t.reshape((1, self.enc_residual_count, 1280)))
            .map_err(|e| format!("residual: {}", e))?;
            Tensor::cat(&[&residual, &enc_out], 1).map_err(|e| format!("cat: {}", e))?
        } else {
            enc_out
        };

        let total_enc = to_adapt.dim(1).map_err(|e| format!("dim: {}", e))?;
        let usable = (total_enc / 4) * 4;
        let leftover = total_enc - usable;

        let mut new_adapter_tokens = 0;
        let mut adapter_embeds: Option<Tensor> = None;
        if usable > 0 {
            let adapter_input = to_adapt
                .narrow(1, 0, usable)
                .map_err(|e| format!("narrow: {}", e))?;
            let reshaped =
                reshape_encoder_output(&adapter_input, 4).map_err(|e| format!("reshape: {}", e))?;
            let adapted = model
                .adapter
                .forward(&reshaped)
                .map_err(|e| format!("adapter: {}", e))?;
            new_adapter_tokens = adapted.dim(1).map_err(|e| format!("dim: {}", e))?;
            adapter_embeds = Some(adapted);
        }

        // Save leftover
        if leftover > 0 {
            let leftover_t: Tensor = to_adapt
                .narrow(1, usable, leftover)
                .map_err(|e| format!("leftover: {}", e))?;
            let flat: Tensor = leftover_t
                .reshape(leftover * 1280)
                .map_err(|e| format!("reshape: {}", e))?;
            self.enc_residual = flat
                .to_vec1::<f32>()
                .map_err(|e| format!("vec: {}", e))?;
            self.enc_residual_count = leftover;
        } else {
            self.enc_residual_count = 0;
        }

        if new_adapter_tokens == 0 {
            return Ok(String::new());
        }
        let adapter_embeds = adapter_embeds.unwrap();

        // Step 5: Decoder
        let prompt_len = 39usize;
        let eos_token = 2u32;
        let mut new_text_tokens = Vec::new();

        if !self.decoder_started && self.total_adapter_tokens + new_adapter_tokens >= prompt_len {
            let prompt_ids: Vec<u32> = std::iter::once(1u32)
                .chain(std::iter::repeat(32u32).take(prompt_len - 1))
                .collect();
            let prompt_text = model
                .embed_tokens(&prompt_ids)
                .map_err(|e| format!("embed: {}", e))?;
            let prompt_audio = adapter_embeds
                .narrow(1, 0, prompt_len.min(new_adapter_tokens))
                .map_err(|e| format!("narrow: {}", e))?;
            let pl = prompt_audio.dim(1).map_err(|e| format!("dim: {}", e))?;
            let prompt_text_slice = prompt_text
                .narrow(1, 0, pl)
                .map_err(|e| format!("narrow: {}", e))?;
            let prompt_input = prompt_audio
                .broadcast_add(&prompt_text_slice)
                .map_err(|e| format!("add: {}", e))?;

            if pl > 1 {
                let prefill = prompt_input
                    .narrow(1, 0, pl - 1)
                    .map_err(|e| format!("narrow: {}", e))?;
                let _ = model
                    .decoder_forward(prefill, &self.ada_scales, &mut self.dec_caches)
                    .map_err(|e| format!("prefill: {}", e))?;
            }
            let last = prompt_input
                .narrow(1, pl - 1, 1)
                .map_err(|e| format!("narrow: {}", e))?;
            let hidden = model
                .decoder_forward(last, &self.ada_scales, &mut self.dec_caches)
                .map_err(|e| format!("forward: {}", e))?;
            self.prev_token = model
                .lm_head_argmax(&hidden)
                .map_err(|e| format!("lm_head: {}", e))?;
            self.generated_tokens.push(self.prev_token);
            if self.prev_token >= 1000 {
                new_text_tokens.push(self.prev_token);
            }

            self.decoder_started = true;

            let decode_start = pl;
            for j in decode_start..new_adapter_tokens {
                let audio_pos = adapter_embeds
                    .narrow(1, j, 1)
                    .map_err(|e| format!("narrow: {}", e))?;
                let text_embed = model
                    .embed_token(self.prev_token)
                    .map_err(|e| format!("embed: {}", e))?;
                let x = audio_pos
                    .broadcast_add(&text_embed)
                    .map_err(|e| format!("add: {}", e))?;
                let hidden = model
                    .decoder_forward(x, &self.ada_scales, &mut self.dec_caches)
                    .map_err(|e| format!("forward: {}", e))?;
                let token = model
                    .lm_head_argmax(&hidden)
                    .map_err(|e| format!("lm_head: {}", e))?;
                self.generated_tokens.push(token);
                self.prev_token = token;
                if token >= 1000 {
                    new_text_tokens.push(token);
                }
                if token == eos_token {
                    break;
                }
            }
        } else if self.decoder_started {
            for j in 0..new_adapter_tokens {
                let audio_pos = adapter_embeds
                    .narrow(1, j, 1)
                    .map_err(|e| format!("narrow: {}", e))?;
                let text_embed = model
                    .embed_token(self.prev_token)
                    .map_err(|e| format!("embed: {}", e))?;
                let x = audio_pos
                    .broadcast_add(&text_embed)
                    .map_err(|e| format!("add: {}", e))?;
                let hidden = model
                    .decoder_forward(x, &self.ada_scales, &mut self.dec_caches)
                    .map_err(|e| format!("forward: {}", e))?;
                let token = model
                    .lm_head_argmax(&hidden)
                    .map_err(|e| format!("lm_head: {}", e))?;
                self.generated_tokens.push(token);
                self.prev_token = token;
                if token >= 1000 {
                    new_text_tokens.push(token);
                }
                if token == eos_token {
                    break;
                }
            }
        }

        self.total_adapter_tokens += new_adapter_tokens;

        let infer_ms = t0.elapsed().as_secs_f32() * 1000.0;
        let audio_secs = self.audio_samples.len() as f32 / 16000.0;

        let delta = if !new_text_tokens.is_empty() {
            self.tokenizer.decode(
                &new_text_tokens
                    .iter()
                    .map(|&t| t as i32)
                    .collect::<Vec<_>>(),
            )
        } else {
            String::new()
        };

        eprintln!(
            "[CandleCpu] #{} {:.1}s audio, {} new adapter, {} new text → {:.0}ms \"{}\"",
            self.commit_count,
            audio_secs,
            new_adapter_tokens,
            new_text_tokens.len(),
            infer_ms,
            if delta.len() > 60 { &delta[..60] } else { &delta }
        );

        Ok(delta)
    }

    fn reset(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.processed_conv_frames = 0;
        self.enc_caches = VoxtralModel::new_encoder_caches();
        self.enc_residual.clear();
        self.enc_residual_count = 0;
        self.dec_caches = VoxtralModel::new_decoder_caches();
        self.prev_token = 0;
        self.decoder_started = false;
        self.total_adapter_tokens = 0;
        self.generated_tokens.clear();
        self.commit_count = 0;
        let left_pad = 32 * 1280;
        self.audio_samples = vec![0.0f32; left_pad];
        Ok(())
    }

    fn commit_count(&self) -> usize {
        self.commit_count
    }
}
