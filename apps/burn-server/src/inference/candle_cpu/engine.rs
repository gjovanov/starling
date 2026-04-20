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

    // Mel + conv caching — avoid recomputing on full audio each commit
    mel_samples_processed: usize,
    cached_mel_flat: Vec<f32>,    // [n_mels × n_frames] column-major
    cached_mel_frames: usize,
    cached_conv_flat: Vec<f32>,   // [1, T/2, 1280] conv output flattened
    cached_conv_frames: usize,   // number of conv output frames in cache

    // Incremental encoding state
    processed_conv_frames: usize, // how many conv frames already encoded
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
            mel_samples_processed: 0,
            cached_mel_flat: Vec::new(),
            cached_mel_frames: 0,
            cached_conv_flat: Vec::new(),
            cached_conv_frames: 0,
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
        let profile = std::env::var("CANDLE_PROFILE").is_ok();
        let t_enc_start = Instant::now();

        // Step 1: Incremental mel — only compute mel for new audio samples.
        // STFT uses hop_length=160, window_size=400. Recompute the last few
        // boundary frames to handle window overlap with new audio.
        let n_mels = 128;
        let hop = 160;
        let window_overlap_frames = 3; // recompute last 3 frames for boundary correctness
        let total_samples = self.audio_samples.len();

        if self.mel_samples_processed == 0 {
            // First commit: compute mel on all audio
            let log_mel = self.mel_spec.compute_log(&self.audio_samples);
            let n_frames = log_mel.len();
            if n_frames == 0 { return Ok(String::new()); }
            self.cached_mel_flat = vec![0.0f32; n_mels * n_frames];
            for (t, frame) in log_mel.iter().enumerate() {
                for (m, &val) in frame.iter().enumerate() {
                    self.cached_mel_flat[m * n_frames + t] = val;
                }
            }
            self.cached_mel_frames = n_frames;
            self.mel_samples_processed = total_samples;
        } else {
            // Incremental: compute mel ONLY on new audio samples + small overlap window.
            // The overlap ensures boundary mel frames are computed correctly.
            let overlap_samples = window_overlap_frames * hop + 400; // 400 = window_size
            let start_sample = if self.mel_samples_processed > overlap_samples {
                self.mel_samples_processed - overlap_samples
            } else {
                0
            };
            // Only process from start_sample to end of current audio (NOT from 0)
            let new_mel = self.mel_spec.compute_log(&self.audio_samples[start_sample..total_samples]);
            let new_n_frames = new_mel.len();
            if new_n_frames == 0 { return Ok(String::new()); }

            // How many cached frames to keep (before the overlap zone)
            let keep_frames = if self.cached_mel_frames > window_overlap_frames {
                self.cached_mel_frames - window_overlap_frames
            } else {
                0
            };
            let total_frames = keep_frames + new_n_frames;

            // Rebuild flat mel: keep old frames + new frames
            let mut new_flat = vec![0.0f32; n_mels * total_frames];
            // Copy kept frames from cache
            if keep_frames > 0 {
                for m in 0..n_mels {
                    let src_offset = m * self.cached_mel_frames;
                    let dst_offset = m * total_frames;
                    new_flat[dst_offset..dst_offset + keep_frames]
                        .copy_from_slice(&self.cached_mel_flat[src_offset..src_offset + keep_frames]);
                }
            }
            // Copy new frames
            for (t, frame) in new_mel.iter().enumerate() {
                for (m, &val) in frame.iter().enumerate() {
                    new_flat[m * total_frames + keep_frames + t] = val;
                }
            }
            self.cached_mel_flat = new_flat;
            self.cached_mel_frames = total_frames;
            self.mel_samples_processed = total_samples;
        }

        // Step 2: Incremental conv — only compute conv on new mel frames.
        // Conv1 (kernel=3, stride=1, causal) and Conv2 (kernel=3, stride=2, causal)
        // are causal: output[T] depends only on input[≤T].
        // We recompute from a few frames before new content to handle kernel overlap.
        let n_frames = self.cached_mel_frames;
        let conv_kernel_overlap = 4; // frames of mel context needed for conv boundary

        if self.cached_conv_frames == 0 {
            // First time: run conv on full mel
            let mel = Tensor::new(self.cached_mel_flat.clone(), &device)
                .and_then(|t| t.reshape((1, n_mels, n_frames)))
                .map_err(|e| format!("mel: {}", e))?;
            let conv_out = model.encoder.conv.forward(&mel).map_err(|e| format!("conv: {}", e))?;
            let total_conv = conv_out.dim(1).map_err(|e| format!("dim: {}", e))?;
            // Cache conv output as flat f32
            let conv_flat: Vec<f32> = conv_out.flatten_all()
                .and_then(|t| t.to_vec1())
                .map_err(|e| format!("conv flatten: {}", e))?;
            self.cached_conv_flat = conv_flat;
            self.cached_conv_frames = total_conv;
        } else {
            // Incremental: run conv only on recent mel frames (with overlap for kernels)
            let keep_conv = self.cached_conv_frames;
            // Conv2 has stride=2, so each conv output frame corresponds to ~2 mel frames.
            // We need conv_kernel_overlap mel frames of context.
            let mel_start = if n_frames > conv_kernel_overlap * 4 {
                // Map back from conv frames to mel frames: conv output T corresponds to mel frame ~2*T
                let recompute_mel_start = (keep_conv.saturating_sub(conv_kernel_overlap)) * 2;
                recompute_mel_start.min(n_frames)
            } else {
                0
            };

            // Extract mel slice [n_mels, mel_start..n_frames]
            let slice_len = n_frames - mel_start;
            let mut mel_slice = vec![0.0f32; n_mels * slice_len];
            for m in 0..n_mels {
                let src_offset = m * n_frames + mel_start;
                let dst_offset = m * slice_len;
                mel_slice[dst_offset..dst_offset + slice_len]
                    .copy_from_slice(&self.cached_mel_flat[src_offset..src_offset + slice_len]);
            }
            let mel_partial = Tensor::new(mel_slice, &device)
                .and_then(|t| t.reshape((1, n_mels, slice_len)))
                .map_err(|e| format!("mel partial: {}", e))?;

            let conv_partial = model.encoder.conv.forward(&mel_partial)
                .map_err(|e| format!("conv partial: {}", e))?;
            let partial_conv_frames = conv_partial.dim(1).map_err(|e| format!("dim: {}", e))?;
            let partial_flat: Vec<f32> = conv_partial.flatten_all()
                .and_then(|t| t.to_vec1())
                .map_err(|e| format!("conv flatten: {}", e))?;

            // Keep old conv frames up to the recompute point, append new
            let conv_keep = (mel_start / 2).min(keep_conv); // approximate: conv stride=2
            let new_total = conv_keep + partial_conv_frames;
            let d = 1280; // conv output dim
            let mut new_conv_flat = vec![0.0f32; new_total * d];
            // Copy kept portion
            if conv_keep > 0 {
                new_conv_flat[..conv_keep * d].copy_from_slice(&self.cached_conv_flat[..conv_keep * d]);
            }
            // Copy new portion
            new_conv_flat[conv_keep * d..].copy_from_slice(&partial_flat[..partial_conv_frames * d]);
            self.cached_conv_flat = new_conv_flat;
            self.cached_conv_frames = new_total;
        }

        // Create conv_out tensor from cache
        let total_conv = self.cached_conv_frames;
        if profile {
            let mel_conv_ms = t_enc_start.elapsed().as_secs_f32() * 1000.0;
            eprintln!("[CandleCpu DETAIL] #{} mel+conv={:.0}ms mel_frames={} conv_frames={} mel_samples={} audio_samples={}",
                self.commit_count, mel_conv_ms, self.cached_mel_frames, total_conv,
                self.mel_samples_processed, self.audio_samples.len());
        }
        let conv_out = Tensor::new(self.cached_conv_flat.clone(), &device)
            .and_then(|t| t.reshape((1, total_conv, 1280)))
            .map_err(|e| format!("conv tensor: {}", e))?;
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
        let new_len = safe_aligned - new_start;
        if profile {
            eprintln!("[CandleCpu DETAIL] #{} encoder: new_start={} safe_aligned={} new_len={} kv_cache_seq={}",
                self.commit_count, new_start, safe_aligned, new_len, self.enc_caches[0].seq_len);
        }
        let new_conv = conv_out
            .narrow(1, new_start, new_len)
            .map_err(|e| format!("narrow: {}", e))?;
        self.processed_conv_frames = safe_aligned;

        // Step 3: Encoder on NEW conv frames only
        let new_len = safe_aligned - new_start;
        // Periodic full reset (vllm-style): drop encoder KV cache when it gets too big.
        // On CPU, each encoder layer's attention costs O(new_len × kv_seq) per layer.
        // With 32 layers × 12 heads × 64 head_dim, kv_seq=150 already dominates decode.
        // Reset threshold 150 = ~3s audio context (plenty for ASR).
        // vllm does similar periodic resets to bound context.
        let max_kv_before_reset = std::env::var("VOXTRAL_ENC_RESET")
            .ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(150);
        if self.enc_caches[0].seq_len + new_len > max_kv_before_reset {
            self.enc_caches = VoxtralModel::new_encoder_caches();
        }

        let t_enc_layers = Instant::now();
        let mut x = new_conv;

        // Profile first encoder layer in detail for slow commits
        if profile && self.enc_caches[0].seq_len > 250 {
            let t_l0 = Instant::now();
            x = model.encoder.layers[0]
                .forward(&x, &model.encoder.rope, &mut self.enc_caches[0])
                .map_err(|e| format!("enc layer 0: {}", e))?;
            let l0_ms = t_l0.elapsed().as_secs_f32() * 1000.0;

            let t_rest = Instant::now();
            for (i, layer) in model.encoder.layers.iter().enumerate().skip(1) {
                x = layer
                    .forward(&x, &model.encoder.rope, &mut self.enc_caches[i])
                    .map_err(|e| format!("enc layer {}: {}", i, e))?;
            }
            let rest_ms = t_rest.elapsed().as_secs_f32() * 1000.0;
            eprintln!("[CandleCpu DETAIL] #{} enc layer0={:.0}ms rest={:.0}ms",
                self.commit_count, l0_ms, rest_ms);
        } else {
            for (i, layer) in model.encoder.layers.iter().enumerate() {
                x = layer
                    .forward(&x, &model.encoder.rope, &mut self.enc_caches[i])
                    .map_err(|e| format!("enc layer {}: {}", i, e))?;
            }
        }

        let enc_out = model
            .encoder
            .norm
            .forward(&x)
            .map_err(|e| format!("enc norm: {}", e))?;
        if profile {
            eprintln!("[CandleCpu DETAIL] #{} enc_layers={:.0}ms for {} frames, kv_seq={}",
                self.commit_count, t_enc_layers.elapsed().as_secs_f32() * 1000.0,
                new_len, self.enc_caches[0].seq_len);
        }

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
        let enc_ms = t_enc_start.elapsed().as_secs_f32() * 1000.0;
        let t_dec_start = Instant::now();

        // Step 5: Decoder
        let prompt_len = 39usize;
        let eos_token = 2u32;
        let mut new_text_tokens = Vec::new();

        // Periodic decoder KV cache reset (like vllm).
        // Decoder attention scales linearly with kv_seq. On CPU, 800+ positions
        // become slow (~1800ms/commit). Reset drops context but bounds compute.
        // Default 300 = ~25s of audio context (decoder has sliding_window=8192 but
        // CPU can't afford to keep it full).
        let max_dec_kv_before_reset = std::env::var("VOXTRAL_DEC_RESET")
            .ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(300);
        if self.decoder_started && self.dec_caches[0].seq_len > max_dec_kv_before_reset {
            self.dec_caches = VoxtralModel::new_decoder_caches();
            self.decoder_started = false; // will re-prefill with current prefix
            self.prev_token = 0;
        }

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

            // Decode mode: VOXTRAL_BATCH=1 (default) = sequential (correct quality).
            // VOXTRAL_BATCH>1 = batched (faster but duplicated tokens).
            let batch_mode: bool = std::env::var("VOXTRAL_BATCH")
                .ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(1) > 1;
            let decode_start = pl;

            if batch_mode {
                // BATCHED: one forward pass for all tokens (all positions share prev_token).
                // Faster but produces duplicated tokens within a commit.
                let remaining = new_adapter_tokens - decode_start;
                if remaining > 0 {
                    let audio_chunk = adapter_embeds.narrow(1, decode_start, remaining)
                        .map_err(|e| format!("narrow: {}", e))?;
                    let text_embed = model.embed_token(self.prev_token)
                        .map_err(|e| format!("embed: {}", e))?;
                    let x = audio_chunk.broadcast_add(&text_embed)
                        .map_err(|e| format!("add: {}", e))?;
                    let hidden = model.decoder_forward(x, &self.ada_scales, &mut self.dec_caches)
                        .map_err(|e| format!("forward: {}", e))?;
                    let logits = model.lm_head(&hidden).map_err(|e| format!("lm_head: {}", e))?;
                    for pos in 0..remaining {
                        let pos_logits = logits.narrow(1, pos, 1).map_err(|e| format!("narrow: {}", e))?;
                        let logits_f32: Vec<f32> = pos_logits
                            .to_dtype(candle_core_cpu::DType::F32)
                            .and_then(|t| t.reshape(131072))
                            .and_then(|t| t.to_vec1())
                            .map_err(|e| format!("logits: {}", e))?;
                        let token = logits_f32.iter().enumerate()
                            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                            .map(|(idx, _)| idx as u32).unwrap_or(0);
                        self.generated_tokens.push(token);
                        self.prev_token = token;
                        // Dedupe within batch (positions often produce same token)
                        if token >= 1000 && new_text_tokens.last().copied() != Some(token) {
                            new_text_tokens.push(token);
                        }
                        if token == eos_token { break; }
                    }
                }
            } else {
                // SEQUENTIAL: each position uses previous token as prev_token.
                // Correct autoregressive — proper quality.
                for j in decode_start..new_adapter_tokens {
                    let audio_pos = adapter_embeds.narrow(1, j, 1)
                        .map_err(|e| format!("narrow: {}", e))?;
                    let text_embed = model.embed_token(self.prev_token)
                        .map_err(|e| format!("embed: {}", e))?;
                    let x = audio_pos.broadcast_add(&text_embed)
                        .map_err(|e| format!("add: {}", e))?;
                    let hidden = model.decoder_forward(x, &self.ada_scales, &mut self.dec_caches)
                        .map_err(|e| format!("forward: {}", e))?;
                    let token = model.lm_head_argmax(&hidden)
                        .map_err(|e| format!("lm_head: {}", e))?;
                    self.generated_tokens.push(token);
                    self.prev_token = token;
                    if token >= 1000 { new_text_tokens.push(token); }
                    if token == eos_token { break; }
                }
            }
        } else if self.decoder_started {
            let batch_mode: bool = std::env::var("VOXTRAL_BATCH")
                .ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(1) > 1;

            if batch_mode {
                // BATCHED decode (same approach for steady-state)
                let audio_chunk = adapter_embeds.narrow(1, 0, new_adapter_tokens)
                    .map_err(|e| format!("narrow: {}", e))?;
                let text_embed = model.embed_token(self.prev_token)
                    .map_err(|e| format!("embed: {}", e))?;
                let x = audio_chunk.broadcast_add(&text_embed)
                    .map_err(|e| format!("add: {}", e))?;
                let hidden = model.decoder_forward(x, &self.ada_scales, &mut self.dec_caches)
                    .map_err(|e| format!("forward: {}", e))?;
                let logits = model.lm_head(&hidden).map_err(|e| format!("lm_head: {}", e))?;
                for pos in 0..new_adapter_tokens {
                    let pos_logits = logits.narrow(1, pos, 1).map_err(|e| format!("narrow: {}", e))?;
                    let logits_f32: Vec<f32> = pos_logits
                        .to_dtype(candle_core_cpu::DType::F32)
                        .and_then(|t| t.reshape(131072))
                        .and_then(|t| t.to_vec1())
                        .map_err(|e| format!("logits: {}", e))?;
                    let token = logits_f32.iter().enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                        .map(|(idx, _)| idx as u32).unwrap_or(0);
                    self.generated_tokens.push(token);
                    self.prev_token = token;
                    // Dedupe within batch (positions often produce same token)
                    if token >= 1000 && new_text_tokens.last().copied() != Some(token) {
                        new_text_tokens.push(token);
                    }
                    if token == eos_token { break; }
                }
            } else {
                // SEQUENTIAL decode (steady-state)
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
        }

        self.total_adapter_tokens += new_adapter_tokens;

        let dec_ms = t_dec_start.elapsed().as_secs_f32() * 1000.0;
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

        if profile {
            let new_frames = safe_aligned - new_start;
            eprintln!(
                "[CandleCpu COMMIT] #{} enc={:.0}ms({} frames) dec={:.0}ms({} tok) total={:.0}ms | {:.1}s audio \"{}\"",
                self.commit_count, enc_ms, new_frames, dec_ms, new_adapter_tokens,
                infer_ms, audio_secs,
                if delta.len() > 40 { &delta[..40] } else { &delta }
            );
        } else {
            eprintln!(
                "[CandleCpu] #{} {:.1}s audio, {} adapter, {} text → {:.0}ms \"{}\"",
                self.commit_count, audio_secs, new_adapter_tokens,
                new_text_tokens.len(), infer_ms,
                if delta.len() > 60 { &delta[..60] } else { &delta }
            );
        }

        Ok(delta)
    }

    fn reset(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.mel_samples_processed = 0;
        self.cached_mel_flat.clear();
        self.cached_mel_frames = 0;
        self.cached_conv_flat.clear();
        self.cached_conv_frames = 0;
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
