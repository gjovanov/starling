//! Incremental candle CUDA inference engine.
//!
//! Each commit() processes ONLY new audio through the encoder (KV cached),
//! then decodes new adapter tokens. This enables real-time 0.5s commits.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use candle_core::{DType, Device, Module, Tensor};

use crate::audio::mel::{MelConfig, MelSpectrogram};
use crate::audio::pad::PadConfig;
use crate::audio::AudioBuffer;
use crate::inference::tokenizer::TekkenDecoder;
use crate::inference::{InferenceEngine, InferenceSession};

use super::model::{self, KVCache, VoxtralModel, reshape_encoder_output};

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

/// Incremental streaming session — keeps encoder + decoder KV caches across commits.
pub struct CandleNativeSession {
    model: Arc<Mutex<VoxtralModel>>,
    tokenizer: Arc<TekkenDecoder>,
    device: Device,
    _language: String,
    mel_spec: MelSpectrogram,
    pad_config: PadConfig,

    // Audio state
    audio_samples: Vec<f32>,
    commit_count: usize,

    // Incremental encoding state
    processed_conv_frames: usize,  // how many conv frames already encoded
    enc_caches: Vec<KVCache>,      // 32 encoder layer KV caches
    enc_residual: Vec<f32>,        // leftover encoder frames for 4x alignment (flat [n, 1280])
    enc_residual_count: usize,

    // Decoder state (persistent across commits)
    dec_caches: Vec<KVCache>,      // 26 decoder layer KV caches
    ada_scales: Vec<Tensor>,
    prev_token: u32,
    decoder_started: bool,
    total_adapter_tokens: usize,

    // Text accumulation
    generated_tokens: Vec<u32>,
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

        // Pre-fill audio with left-pad silence (32 * 1280 = 40960 zeros)
        let left_pad = 32 * 1280;
        let audio_samples = vec![0.0f32; left_pad];

        Ok(Self {
            model,
            tokenizer,
            device,
            _language: language.to_string(),
            mel_spec: MelSpectrogram::new(MelConfig::default()),
            pad_config: PadConfig::bf16(),
            audio_samples,
            commit_count: 0,
            processed_conv_frames: 0,
            enc_caches: VoxtralModel::new_encoder_caches(),
            enc_residual: Vec::new(),
            enc_residual_count: 0,
            dec_caches: (0..26).map(|_| KVCache::new()).collect(),
            ada_scales,
            prev_token: 0,
            decoder_started: false,
            total_adapter_tokens: 0,
            generated_tokens: Vec::new(),
        })
    }
}

impl InferenceSession for CandleNativeSession {
    fn send_audio(&mut self, samples: &[f32]) {
        self.audio_samples.extend_from_slice(samples);
    }

    fn commit(&mut self) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        self.commit_count += 1;
        let t0 = Instant::now();

        // Need minimum audio for encoder to produce anything useful
        if self.audio_samples.len() < 50000 {
            if self.commit_count <= 3 || self.commit_count % 20 == 0 {
                eprintln!("[CandleNative] #{} waiting ({} samples < 50000)", self.commit_count, self.audio_samples.len());
            }
            return Ok(String::new());
        }

        let model = self.model.lock().map_err(|e| format!("lock: {}", e))?;

        // Step 1: Compute mel for accumulated audio (no right-pad on intermediate commits)
        // Right-pad would contaminate boundary mel frames. The mel STFT handles edges
        // with reflect padding. New frames arriving later fix any boundary artifacts.
        let log_mel = self.mel_spec.compute_log(&self.audio_samples);
        let n_frames = log_mel.len();
        let n_mels = if n_frames > 0 { log_mel[0].len() } else { 128 };
        if n_frames == 0 { return Ok(String::new()); }

        // Build mel tensor
        let mut flat = vec![0.0f32; n_mels * n_frames];
        for (t, frame) in log_mel.iter().enumerate() {
            for (m, &val) in frame.iter().enumerate() {
                flat[m * n_frames + t] = val;
            }
        }
        let mel = Tensor::new(flat, &self.device)
            .and_then(|t| t.to_dtype(DType::BF16))
            .and_then(|t| t.reshape((1, n_mels, n_frames)))
            .map_err(|e| format!("mel: {}", e))?;

        // Step 2: Conv on full mel (fast, no attention)
        let conv_out = model.encoder.conv.forward(&mel)
            .map_err(|e| format!("conv: {}", e))?;
        let total_conv = conv_out.dim(1).map_err(|e| format!("dim: {}", e))?;
        let total_aligned = (total_conv / 4) * 4;

        // How many NEW conv frames since last commit?
        // Leave a margin at the edge — boundary frames may change when more audio arrives.
        // Conv kernel=3 + mel STFT window extends ~4 conv frames. Use margin of 8.
        let safe_boundary = 8;
        let safe_end = if total_aligned > safe_boundary { total_aligned - safe_boundary } else { 0 };
        let safe_aligned = (safe_end / 4) * 4; // re-align to 4

        let new_start = self.processed_conv_frames;
        if new_start >= safe_aligned {
            return Ok(String::new()); // no stable new frames yet
        }
        let new_conv = conv_out.narrow(1, new_start, safe_aligned - new_start)
            .map_err(|e| format!("narrow: {}", e))?;
        let new_len = safe_aligned - new_start;
        self.processed_conv_frames = safe_aligned;

        // Step 3: Encoder on NEW conv frames only (with KV cache)
        let sliding_window = 750;
        if self.enc_caches[0].seq_len + new_len > sliding_window {
            let keep = sliding_window / 2;
            for cache in &mut self.enc_caches {
                cache.compact(keep).map_err(|e| format!("compact: {}", e))?;
            }
        }

        let mut x = new_conv;
        for (i, layer) in model.encoder.layers.iter().enumerate() {
            x = layer.forward(&x, &model.encoder.rope, &mut self.enc_caches[i])
                .map_err(|e| format!("enc layer {}: {}", i, e))?;
        }
        let enc_out = model.encoder.norm.forward(&x)
            .map_err(|e| format!("enc norm: {}", e))?;

        // Step 4: Adapter (4x reshape) with residual handling
        let enc_frames = enc_out.dim(1).map_err(|e| format!("dim: {}", e))?;
        // Combine with leftover from previous commit
        let to_adapt = if self.enc_residual_count > 0 {
            let residual = Tensor::new(self.enc_residual[..self.enc_residual_count * 1280].to_vec(), &self.device)
                .and_then(|t| t.to_dtype(DType::BF16))
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
            let adapter_input = to_adapt.narrow(1, 0, usable).map_err(|e| format!("narrow: {}", e))?;
            let reshaped = reshape_encoder_output(&adapter_input, 4).map_err(|e| format!("reshape: {}", e))?;
            let adapted = model.adapter.forward(&reshaped).map_err(|e| format!("adapter: {}", e))?;
            new_adapter_tokens = adapted.dim(1).map_err(|e| format!("dim: {}", e))?;
            adapter_embeds = Some(adapted);
        }

        // Save leftover for next commit
        if leftover > 0 {
            let leftover_t: Tensor = to_adapt.narrow(1, usable, leftover)
                .and_then(|t| t.to_dtype(DType::F32))
                .map_err(|e| format!("leftover: {}", e))?;
            let flat: Tensor = leftover_t.reshape(leftover * 1280)
                .map_err(|e| format!("reshape: {}", e))?;
            self.enc_residual = flat.to_vec1::<f32>()
                .map_err(|e| format!("vec: {}", e))?;
            self.enc_residual_count = leftover;
        } else {
            self.enc_residual_count = 0;
        }

        if new_adapter_tokens == 0 {
            return Ok(String::new());
        }
        let adapter_embeds = adapter_embeds.unwrap();

        // Step 5: Decoder — prefill if not started, then decode new tokens
        let prompt_len = 39usize;
        let eos_token = 2u32;
        let mut new_text_tokens = Vec::new();

        if !self.decoder_started && self.total_adapter_tokens + new_adapter_tokens >= prompt_len {
            // Prefill with streaming prefix (BOS + STREAMING_PAD)
            let prompt_ids: Vec<u32> = std::iter::once(1u32)
                .chain(std::iter::repeat(32u32).take(prompt_len - 1))
                .collect();
            let prompt_text = model.embed_tokens(&prompt_ids)
                .map_err(|e| format!("embed: {}", e))?;
            // Use first prompt_len adapter tokens + text
            let prompt_audio = adapter_embeds.narrow(1, 0, prompt_len.min(new_adapter_tokens))
                .map_err(|e| format!("narrow: {}", e))?;
            let pl = prompt_audio.dim(1).map_err(|e| format!("dim: {}", e))?;
            let prompt_text_slice = prompt_text.narrow(1, 0, pl)
                .map_err(|e| format!("narrow: {}", e))?;
            let prompt_input = prompt_audio.broadcast_add(&prompt_text_slice)
                .map_err(|e| format!("add: {}", e))?;

            // Prefill (all but last position)
            if pl > 1 {
                let prefill = prompt_input.narrow(1, 0, pl - 1)
                    .map_err(|e| format!("narrow: {}", e))?;
                let _ = model.decoder_forward(prefill, &self.ada_scales, &mut self.dec_caches)
                    .map_err(|e| format!("prefill: {}", e))?;
            }
            // Last position → first token
            let last = prompt_input.narrow(1, pl - 1, 1)
                .map_err(|e| format!("narrow: {}", e))?;
            let hidden = model.decoder_forward(last, &self.ada_scales, &mut self.dec_caches)
                .map_err(|e| format!("forward: {}", e))?;
            let logits = model.lm_head(&hidden).map_err(|e| format!("lm_head: {}", e))?;
            self.prev_token = argmax_token(&logits)?;
            self.generated_tokens.push(self.prev_token);
            if self.prev_token >= 1000 { new_text_tokens.push(self.prev_token); }

            self.decoder_started = true;

            // Decode remaining adapter tokens from this commit
            let decode_start = pl;
            for j in decode_start..new_adapter_tokens {
                let audio_pos = adapter_embeds.narrow(1, j, 1)
                    .map_err(|e| format!("narrow: {}", e))?;
                let text_embed = model.embed_token(self.prev_token)
                    .map_err(|e| format!("embed: {}", e))?;
                let x = audio_pos.broadcast_add(&text_embed)
                    .map_err(|e| format!("add: {}", e))?;
                let hidden = model.decoder_forward(x, &self.ada_scales, &mut self.dec_caches)
                    .map_err(|e| format!("forward: {}", e))?;
                let logits = model.lm_head(&hidden).map_err(|e| format!("lm_head: {}", e))?;
                let token = argmax_token(&logits)?;
                self.generated_tokens.push(token);
                self.prev_token = token;
                if token >= 1000 { new_text_tokens.push(token); }
                if token == eos_token { break; }
            }
        } else if self.decoder_started {
            // Already running — just decode new adapter tokens
            for j in 0..new_adapter_tokens {
                let audio_pos = adapter_embeds.narrow(1, j, 1)
                    .map_err(|e| format!("narrow: {}", e))?;
                let text_embed = model.embed_token(self.prev_token)
                    .map_err(|e| format!("embed: {}", e))?;
                let x = audio_pos.broadcast_add(&text_embed)
                    .map_err(|e| format!("add: {}", e))?;
                let hidden = model.decoder_forward(x, &self.ada_scales, &mut self.dec_caches)
                    .map_err(|e| format!("forward: {}", e))?;
                let logits = model.lm_head(&hidden).map_err(|e| format!("lm_head: {}", e))?;
                let token = argmax_token(&logits)?;
                self.generated_tokens.push(token);
                self.prev_token = token;
                if token >= 1000 { new_text_tokens.push(token); }
                if token == eos_token { break; }
            }
        }

        self.total_adapter_tokens += new_adapter_tokens;

        let infer_ms = t0.elapsed().as_secs_f32() * 1000.0;
        let audio_secs = self.audio_samples.len() as f32 / 16000.0;

        // Debug: show ALL generated tokens, not just text
        {
            let recent: Vec<u32> = self.generated_tokens.iter().rev().take(10).copied().collect();
            eprintln!("[CandleNative] recent tokens (last 10): {:?}", recent);
        }

        // Decode only new text tokens to string
        let delta = if !new_text_tokens.is_empty() {
            self.tokenizer.decode(
                &new_text_tokens.iter().map(|&t| t as i32).collect::<Vec<_>>(),
            )
        } else {
            String::new()
        };

        eprintln!(
            "[CandleNative] #{} {:.1}s audio, {} new adapter, {} new text → {:.0}ms \"{}\"",
            self.commit_count, audio_secs, new_adapter_tokens,
            new_text_tokens.len(), infer_ms,
            if delta.len() > 60 { &delta[..60] } else { &delta }
        );

        Ok(delta)
    }

    fn reset(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Keep audio, reset encoder+decoder state (context rotation)
        self.processed_conv_frames = 0;
        self.enc_caches = VoxtralModel::new_encoder_caches();
        self.enc_residual.clear();
        self.enc_residual_count = 0;
        self.dec_caches = (0..26).map(|_| KVCache::new()).collect();
        self.prev_token = 0;
        self.decoder_started = false;
        self.total_adapter_tokens = 0;
        self.generated_tokens.clear();
        self.commit_count = 0;
        // Re-start with left-pad silence
        let left_pad = 32 * 1280;
        self.audio_samples = vec![0.0f32; left_pad];
        Ok(())
    }

    fn commit_count(&self) -> usize {
        self.commit_count
    }
}

fn argmax_token(logits: &Tensor) -> Result<u32, Box<dyn std::error::Error + Send + Sync>> {
    let logits_f32: Vec<f32> = logits.to_dtype(DType::F32)
        .and_then(|t| t.reshape(131072))
        .and_then(|t| t.to_vec1())
        .map_err(|e| format!("argmax: {}", e))?;
    Ok(logits_f32.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32).unwrap_or(0))
}
