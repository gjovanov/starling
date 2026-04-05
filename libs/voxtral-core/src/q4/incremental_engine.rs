//! Incremental Q4/Burn wgpu inference engine.
//!
//! Ports the incremental encoder KV cache architecture from the candle-native
//! engine (`apps/burn-server/src/inference/candle_native/engine.rs`) to the
//! Q4/Burn wgpu backend.
//!
//! Each `commit()` processes ONLY new audio through the encoder (KV cached),
//! then decodes new adapter tokens through the decoder (also KV cached).
//! This avoids re-encoding all accumulated audio on every commit, which is
//! the single biggest performance optimization for streaming.
//!
//! ## Architecture
//!
//! 1. Append new audio to accumulated buffer (never cleared)
//! 2. Compute mel on ALL accumulated audio (CPU, fast)
//! 3. Run conv on full mel output (GPU, fast -- no attention)
//! 4. Apply boundary margin of 8 frames (edge frames may shift)
//! 5. Feed only NEW conv frames through encoder with persistent KV cache
//! 6. Handle 4x reshape residual (buffer leftover < 4 frames)
//! 7. Pass new adapter tokens through decoder with persistent KV cache
//! 8. Autoregressive decode new tokens
//! 9. Return text delta

use burn::backend::wgpu::WgpuDevice;
use burn::tensor::{Tensor, TensorData};

use crate::audio::mel::{MelConfig, MelSpectrogram};
use crate::audio::pad::PadConfig;
use crate::layers::kv_cache::LayerCaches;
use crate::tokenizer::TekkenDecoder;

use super::model::{Q4VoxtralModel, reshape_encoder_output};
use super::WgpuBackend;

/// Platform-agnostic logging. On native uses eprintln; on WASM this is a no-op
/// (the caller in wasm-engine handles logging via web_sys::console).
#[allow(unused_variables)]
fn log(args: core::fmt::Arguments<'_>) {
    #[cfg(not(target_arch = "wasm32"))]
    eprintln!("{}", args);
}

/// Boundary margin: don't process conv frames within this distance of the
/// trailing edge. Boundary mel frames may shift when new audio arrives
/// (conv kernel=3 + mel STFT window extends ~4 conv frames).
const BOUNDARY_MARGIN: usize = 8;

/// Streaming prefix length: BOS(1) + STREAMING_PAD(32) x 37 = 38 tokens.
/// The decoder needs this many audio positions before it can start decoding.
const PREFIX_LEN: usize = 38;

const BOS_TOKEN: i32 = 1;
const STREAMING_PAD: i32 = 32;
const EOS_TOKEN: i32 = 2;

/// Encoder sliding window size (matching candle-native).
const ENCODER_SLIDING_WINDOW: usize = 750;

/// Maximum decoder positions before context rotation (~100s at 12.5 tokens/s).
const MAX_DECODER_POSITIONS: usize = 1250;

/// Compute sinusoidal time embedding for transcription delay.
fn compute_time_embedding(t: f32, dim: usize, device: &WgpuDevice) -> Tensor<WgpuBackend, 3> {
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

/// Incremental Q4 streaming session.
///
/// Keeps encoder + decoder KV caches across commits, processing only new
/// audio each time. This is the Q4/Burn-wgpu equivalent of the candle-native
/// `CandleNativeSession`.
pub struct Q4IncrementalSession {
    model: Q4VoxtralModel,
    tokenizer: TekkenDecoder,
    device: WgpuDevice,
    mel_spec: MelSpectrogram,
    pad_config: PadConfig,
    t_embed: Tensor<WgpuBackend, 3>,

    // Audio state
    audio_buffer: Vec<f32>,
    commit_count: usize,

    // Incremental encoding state
    processed_conv_frames: usize,
    encoder_caches: LayerCaches<WgpuBackend>,
    enc_residual: Vec<f32>,      // leftover encoder frames for 4x reshape alignment
    enc_residual_count: usize,

    // Decoder state (persistent across commits)
    decoder_caches: LayerCaches<WgpuBackend>,
    prev_token: i32,
    decoder_started: bool,
    total_adapter_tokens: usize,

    // Text accumulation
    generated_tokens: Vec<i32>,
    generated_text: String,
}

impl Q4IncrementalSession {
    /// Create a new incremental session.
    ///
    /// The model and tokenizer are moved in (not shared via Arc<Mutex>)
    /// because WASM is single-threaded — no contention.
    pub fn new(
        model: Q4VoxtralModel,
        tokenizer: TekkenDecoder,
        device: WgpuDevice,
    ) -> Self {
        let t_embed = compute_time_embedding(6.0, 3072, &device);
        let encoder_caches = model.create_encoder_cache();
        let decoder_caches = model.create_decoder_cache();

        // Pre-fill audio with left-pad silence (Q4: 76 * 1280 = 97,280 zeros)
        let pad_config = PadConfig::q4();
        let left_pad = pad_config.left_pad_samples();
        let audio_buffer = vec![0.0f32; left_pad];

        Self {
            model,
            tokenizer,
            device,
            mel_spec: MelSpectrogram::new(MelConfig::default()),
            pad_config,
            t_embed,
            audio_buffer,
            commit_count: 0,
            processed_conv_frames: 0,
            encoder_caches,
            enc_residual: Vec::new(),
            enc_residual_count: 0,
            decoder_caches,
            prev_token: 0,
            decoder_started: false,
            total_adapter_tokens: 0,
            generated_tokens: Vec::new(),
            generated_text: String::new(),
        }
    }

    /// Append 16kHz mono f32 PCM audio samples.
    pub fn send_audio(&mut self, samples: &[f32]) {
        self.audio_buffer.extend_from_slice(samples);
    }

    /// Process accumulated audio incrementally and return text delta.
    ///
    /// This is async because WebGPU readback requires `into_data_async().await`
    /// on WASM. On native, the futures resolve immediately.
    pub async fn commit(&mut self) -> Result<String, String> {
        self.commit_count += 1;

        // Need minimum audio for encoder to produce anything useful
        if self.audio_buffer.len() < 50000 {
            if self.commit_count <= 3 || self.commit_count % 20 == 0 {
                log(format_args!(
                    "[Q4Incremental] #{} waiting ({} samples < 50000)",
                    self.commit_count, self.audio_buffer.len()
                ));
            }
            return Ok(String::new());
        }

        // Check for context rotation
        if self.total_adapter_tokens > MAX_DECODER_POSITIONS {
            self.context_rotate();
            log(format_args!(
                "[Q4Incremental] Context rotation at {} adapter tokens",
                self.total_adapter_tokens
            ));
        }

        // Step 1: Compute mel on ALL accumulated audio (CPU, fast).
        // No right-pad on intermediate commits — right-pad would contaminate
        // boundary mel frames. New frames arriving later fix boundary artifacts.
        let log_mel = self.mel_spec.compute_log(&self.audio_buffer);
        let n_frames = log_mel.len();
        let n_mels = if n_frames > 0 { log_mel[0].len() } else { 128 };
        if n_frames == 0 {
            return Ok(String::new());
        }

        // Build mel tensor [1, n_mels, n_frames] (column-major: mel bins are rows)
        let mut flat = vec![0.0f32; n_mels * n_frames];
        for (t, frame) in log_mel.iter().enumerate() {
            for (m, &val) in frame.iter().enumerate() {
                flat[m * n_frames + t] = val;
            }
        }
        let mel_tensor: Tensor<WgpuBackend, 3> =
            Tensor::from_data(TensorData::new(flat, [1, n_mels, n_frames]), &self.device);

        // Step 2: Conv on full mel (GPU, fast -- no attention layers)
        let conv_out = self.model.encoder_conv(mel_tensor);
        let total_conv = conv_out.dims()[1]; // [batch, conv_frames, 1280]
        let total_aligned = (total_conv / 4) * 4;

        // How many NEW conv frames since last commit?
        // Leave a margin at the edge -- boundary frames may change when more audio arrives.
        let safe_end = if total_aligned > BOUNDARY_MARGIN {
            total_aligned - BOUNDARY_MARGIN
        } else {
            0
        };
        let safe_aligned = (safe_end / 4) * 4; // re-align to 4

        let new_start = self.processed_conv_frames;
        if new_start >= safe_aligned {
            return Ok(String::new()); // no stable new frames yet
        }

        let new_len = safe_aligned - new_start;
        let d_conv = conv_out.dims()[2]; // should be 1280
        let new_conv = conv_out.slice([0..1, new_start..safe_aligned, 0..d_conv]);
        self.processed_conv_frames = safe_aligned;

        // Step 3: Encoder sliding window compaction
        let enc_seq_len = self.encoder_caches.seq_len();
        if enc_seq_len + new_len > ENCODER_SLIDING_WINDOW {
            let keep = ENCODER_SLIDING_WINDOW / 2;
            self.encoder_caches.apply_sliding_window(keep);
        }

        // Step 4: Encoder attention on NEW conv frames only (with KV cache)
        let enc_out = self.model.encoder_layers_with_cache(
            new_conv,
            &mut self.encoder_caches,
        );

        // Step 5: Adapter (4x reshape) with residual handling
        let d_enc = enc_out.dims()[2]; // 1280

        // Combine with leftover from previous commit
        let to_adapt = if self.enc_residual_count > 0 {
            let residual = Tensor::<WgpuBackend, 3>::from_data(
                TensorData::new(
                    self.enc_residual[..self.enc_residual_count * d_enc].to_vec(),
                    [1, self.enc_residual_count, d_enc],
                ),
                &self.device,
            );
            Tensor::cat(vec![residual, enc_out], 1)
        } else {
            enc_out
        };

        let total_enc = to_adapt.dims()[1];
        let usable = (total_enc / 4) * 4;
        let leftover = total_enc - usable;

        let mut new_adapter_tokens = 0;
        let mut adapter_embeds: Option<Tensor<WgpuBackend, 3>> = None;
        if usable > 0 {
            let adapter_input = to_adapt.clone().slice([0..1, 0..usable, 0..d_enc]);
            let reshaped = reshape_encoder_output(adapter_input, 4);
            let adapted = self.model.adapter_forward(reshaped);
            new_adapter_tokens = adapted.dims()[1];
            adapter_embeds = Some(adapted);
        }

        // Save leftover for next commit
        if leftover > 0 {
            let leftover_t = to_adapt.slice([0..1, usable..usable + leftover, 0..d_enc]);
            let leftover_flat = leftover_t.reshape([leftover * d_enc]);
            let data = leftover_flat
                .into_data_async()
                .await
                .map_err(|e| format!("GPU readback failed (enc residual): {:?}", e))?;
            self.enc_residual = data
                .to_vec::<f32>()
                .map_err(|e| format!("enc residual extraction: {:?}", e))?;
            self.enc_residual_count = leftover;
        } else {
            self.enc_residual_count = 0;
        }

        if new_adapter_tokens == 0 {
            return Ok(String::new());
        }
        let adapter_embeds = adapter_embeds.unwrap();
        let d_model = adapter_embeds.dims()[2]; // 3072

        // Step 6: Decoder -- prefill if not started, then decode new tokens
        let mut new_text_tokens: Vec<i32> = Vec::new();

        if !self.decoder_started
            && self.total_adapter_tokens + new_adapter_tokens >= PREFIX_LEN
        {
            // Prefill with streaming prefix (BOS + STREAMING_PAD)
            let mut prefix_ids: Vec<i32> = vec![BOS_TOKEN];
            prefix_ids.extend(std::iter::repeat_n(STREAMING_PAD, PREFIX_LEN - 1));

            let prefix_text = self.model.decoder().embed_tokens_from_ids(
                &prefix_ids,
                1,
                PREFIX_LEN,
            );

            // Use first PREFIX_LEN adapter tokens + text embeddings
            let pl = PREFIX_LEN.min(new_adapter_tokens);
            let prefix_audio = adapter_embeds.clone().slice([0..1, 0..pl, 0..d_model]);
            let prefix_text_slice = prefix_text.slice([0..1, 0..pl, 0..d_model]);
            let prefix_input = prefix_audio + prefix_text_slice;

            // Prefill (all but last position)
            if pl > 1 {
                let prefill = prefix_input.clone().slice([0..1, 0..pl - 1, 0..d_model]);
                let _ = self.model.decoder().forward_hidden_with_cache(
                    prefill,
                    self.t_embed.clone(),
                    &mut self.decoder_caches,
                );
            }

            // Last position -> first token
            let last = prefix_input.slice([0..1, pl - 1..pl, 0..d_model]);
            let hidden = self.model.decoder().forward_hidden_with_cache(
                last,
                self.t_embed.clone(),
                &mut self.decoder_caches,
            );
            let logits = self.model.decoder().lm_head(hidden);
            let first_token = self.argmax_async(&logits).await?;

            self.prev_token = first_token;
            self.generated_tokens.push(first_token);
            if first_token >= 1000 {
                new_text_tokens.push(first_token);
            }

            self.decoder_started = true;

            // Decode remaining adapter tokens from this commit
            for j in pl..new_adapter_tokens {
                let audio_pos = adapter_embeds.clone().slice([0..1, j..j + 1, 0..d_model]);
                let text_embed = self.model.decoder().embed_tokens_from_ids(
                    &[self.prev_token],
                    1,
                    1,
                );
                let x = audio_pos + text_embed;
                let hidden = self.model.decoder().forward_hidden_with_cache(
                    x,
                    self.t_embed.clone(),
                    &mut self.decoder_caches,
                );
                let logits = self.model.decoder().lm_head(hidden);
                let token = self.argmax_async(&logits).await?;

                self.generated_tokens.push(token);
                self.prev_token = token;
                if token >= 1000 {
                    new_text_tokens.push(token);
                }
                if token == EOS_TOKEN {
                    break;
                }
            }
        } else if self.decoder_started {
            // Already running -- just decode new adapter tokens
            for j in 0..new_adapter_tokens {
                let audio_pos = adapter_embeds.clone().slice([0..1, j..j + 1, 0..d_model]);
                let text_embed = self.model.decoder().embed_tokens_from_ids(
                    &[self.prev_token],
                    1,
                    1,
                );
                let x = audio_pos + text_embed;
                let hidden = self.model.decoder().forward_hidden_with_cache(
                    x,
                    self.t_embed.clone(),
                    &mut self.decoder_caches,
                );
                let logits = self.model.decoder().lm_head(hidden);
                let token = self.argmax_async(&logits).await?;

                self.generated_tokens.push(token);
                self.prev_token = token;
                if token >= 1000 {
                    new_text_tokens.push(token);
                }
                if token == EOS_TOKEN {
                    break;
                }
            }
        }

        self.total_adapter_tokens += new_adapter_tokens;

        // Decode new text tokens to string
        let delta = if !new_text_tokens.is_empty() {
            self.tokenizer.decode(&new_text_tokens)
        } else {
            String::new()
        };

        self.generated_text.push_str(&delta);

        let audio_secs = self.audio_buffer.len() as f32 / 16000.0;

        log(format_args!(
            "[Q4Incremental] #{} {:.1}s audio, {} new adapter, {} new text -> \"{}\"",
            self.commit_count,
            audio_secs,
            new_adapter_tokens,
            new_text_tokens.len(),
            if delta.len() > 60 { &delta[..60] } else { &delta }
        ));

        Ok(delta)
    }

    /// Async argmax: read logits back from GPU and find the maximum.
    async fn argmax_async(&self, logits: &Tensor<WgpuBackend, 3>) -> Result<i32, String> {
        let pred = logits.clone().argmax(2);
        let data = pred
            .into_data_async()
            .await
            .map_err(|e| format!("GPU readback failed (argmax): {:?}", e))?;
        let tokens: Vec<i32> = data
            .to_vec::<i32>()
            .map_err(|e| format!("argmax extraction: {:?}", e))?;
        Ok(tokens.first().copied().unwrap_or(0))
    }

    /// Reset everything for context rotation.
    ///
    /// Keeps audio buffer intact but resets all encoder/decoder state so
    /// we re-encode from scratch. This happens every ~100s of audio.
    fn context_rotate(&mut self) {
        self.processed_conv_frames = 0;
        self.encoder_caches = self.model.create_encoder_cache();
        self.enc_residual.clear();
        self.enc_residual_count = 0;
        self.decoder_caches = self.model.create_decoder_cache();
        self.prev_token = 0;
        self.decoder_started = false;
        self.total_adapter_tokens = 0;
        self.generated_tokens.clear();
        self.commit_count = 0;

        // Re-start with left-pad silence
        let left_pad = self.pad_config.left_pad_samples();
        self.audio_buffer = vec![0.0f32; left_pad];
    }

    /// Full reset: clear everything including generated text.
    pub fn reset(&mut self) {
        self.context_rotate();
        self.generated_text.clear();
    }

    /// Number of commits since last reset/rotation.
    pub fn commit_count(&self) -> usize {
        self.commit_count
    }

    /// Full accumulated text.
    pub fn generated_text(&self) -> &str {
        &self.generated_text
    }
}
