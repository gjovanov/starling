//! BF16 Voxtral model — encoder, decoder, adapter, and full model.
//!
//! Uses the layer modules from `bf16::layers` and composes them into
//! the complete Voxtral architecture. Matches the reference implementation
//! (voxtral-mini-realtime-rs) exactly.

use burn::nn::Linear;
use burn::tensor::activation::gelu;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use super::layers::*;

// ---------------------------------------------------------------------------
// Audio Encoder (32 layers, causal Whisper-style)
// ---------------------------------------------------------------------------

pub struct AudioEncoder<B: Backend> {
    pub conv: ConvDownsampler<B>,
    pub rope: RoPE<B>,
    pub layers: Vec<EncoderLayer<B>>,
    pub norm: RmsNorm<B>,
}

impl<B: Backend> AudioEncoder<B> {
    /// mel [1, 128, T] → encoder output [1, T/4, 1280]
    pub fn forward(&self, mel: Tensor<B, 3>, offset: usize) -> Tensor<B, 3> {
        let [_, _mels, mel_frames] = mel.dims();
        let x = self.conv.forward(mel);
        let x = x.swap_dims(1, 2);

        // Log conv output scale
        {
            let n = x.dims()[1] * x.dims()[2];
            let flat = x.clone().reshape([n]).into_data();
            let v = flat.as_slice::<f32>().unwrap();
            let rms = (v.iter().map(|x| x*x).sum::<f32>() / v.len() as f32).sqrt();
            eprintln!("[BF16 Encoder] mel_frames={} conv_out={:?} RMS={:.4} first5={:.4?}",
                mel_frames, x.dims(), rms, &v[..5.min(v.len())]);
        }

        let mut x = x;
        for layer in &self.layers {
            x = layer.forward(x, &self.rope, offset);
        }
        self.norm.forward(x)
    }

    pub fn forward_with_cache(
        &self,
        mel: Tensor<B, 3>,
        caches: &mut LayerCaches<B>,
    ) -> Tensor<B, 3> {
        let x = self.conv.forward(mel);
        let x = x.swap_dims(1, 2);

        let mut x = x;
        for (i, layer) in self.layers.iter().enumerate() {
            if let Some(cache) = caches.get_mut(i) {
                x = layer.forward_with_cache(x, &self.rope, cache);
            }
        }
        self.norm.forward(x)
    }
}

// ---------------------------------------------------------------------------
// Audio-Language Adapter (5120 → GELU → 3072)
// ---------------------------------------------------------------------------

pub struct AudioLanguageAdapter<B: Backend> {
    pub linear1: Linear<B>, // no bias, 5120 → 3072
    pub linear2: Linear<B>, // no bias, 3072 → 3072
}

impl<B: Backend> AudioLanguageAdapter<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.linear1.forward(x);
        let x = gelu(x);
        self.linear2.forward(x)
    }
}

/// Reshape encoder output by grouping `factor` adjacent frames.
/// [B, T, D] → [B, T/factor, D*factor]
pub fn reshape_encoder_output<B: Backend>(
    x: Tensor<B, 3>,
    factor: usize,
) -> Tensor<B, 3> {
    let [batch, seq, dim] = x.dims();
    let new_seq = seq / factor;
    let truncated = new_seq * factor;
    let x = x.slice([0..batch, 0..truncated, 0..dim]);
    x.reshape([batch, new_seq, dim * factor])
}

// ---------------------------------------------------------------------------
// Language Model Decoder (26 layers, GQA, ADA RMSNorm)
// ---------------------------------------------------------------------------

pub struct LanguageModel<B: Backend> {
    /// Token embeddings — kept as raw f32 data on CPU for sparse lookups.
    /// Avoids the 128MB GPU buffer limit for the full 131K×3072 table.
    pub tok_embed_data: Vec<f32>,
    pub vocab_size: usize,
    pub rope: RoPE<B>,
    pub layers: Vec<DecoderLayer<B>>,
    pub norm: RmsNorm<B>,
    pub d_model: usize,
}

impl<B: Backend> LanguageModel<B> {
    /// Embed token IDs from CPU data (avoids GPU buffer limit).
    pub fn embed_tokens_from_ids(&self, ids: &[i32], batch: usize, seq: usize, device: &B::Device) -> Tensor<B, 3> {
        let mut output = vec![0.0f32; ids.len() * self.d_model];
        for (i, &id) in ids.iter().enumerate() {
            let row_start = (id as usize) * self.d_model;
            let row_end = row_start + self.d_model;
            if row_end <= self.tok_embed_data.len() {
                output[i * self.d_model..(i + 1) * self.d_model]
                    .copy_from_slice(&self.tok_embed_data[row_start..row_end]);
            }
        }
        Tensor::from_data(
            burn::tensor::TensorData::new(output, [batch, seq, self.d_model]),
            device,
        )
    }

    /// Forward through decoder layers + final norm.
    pub fn forward_hidden(
        &self,
        mut x: Tensor<B, 3>,
        t_embed: Tensor<B, 3>,
        offset: usize,
    ) -> Tensor<B, 3> {
        for layer in &self.layers {
            x = layer.forward(x, t_embed.clone(), &self.rope, offset);
        }
        self.norm.forward(x)
    }

    /// Forward through decoder layers + final norm (with KV cache).
    pub fn forward_hidden_with_cache(
        &self,
        mut x: Tensor<B, 3>,
        t_embed: Tensor<B, 3>,
        caches: &mut LayerCaches<B>,
    ) -> Tensor<B, 3> {
        for (i, layer) in self.layers.iter().enumerate() {
            if let Some(cache) = caches.get_mut(i) {
                x = layer.forward_with_cache(x, t_embed.clone(), &self.rope, cache);
            }
        }
        self.norm.forward(x)
    }

    /// LM head: hidden → logits via tied embeddings.
    /// Splits the embedding matrix into chunks that fit in 128MB GPU buffers.
    /// hidden [B, seq, d_model] → logits [B, seq, vocab]
    pub fn lm_head(&self, hidden: Tensor<B, 3>, device: &B::Device) -> Tensor<B, 3> {
        let [batch, seq, _] = hidden.dims();
        let max_rows_per_chunk = 128 * 1024 * 1024 / (self.d_model * 4);

        let mut logit_parts = Vec::new();
        let mut offset = 0;
        while offset < self.vocab_size {
            let chunk_rows = (self.vocab_size - offset).min(max_rows_per_chunk);
            let chunk_start = offset * self.d_model;
            let chunk_end = (offset + chunk_rows) * self.d_model;
            let chunk_data = &self.tok_embed_data[chunk_start..chunk_end];

            let chunk_tensor: Tensor<B, 2> = Tensor::from_data(
                burn::tensor::TensorData::new(chunk_data.to_vec(), [chunk_rows, self.d_model]),
                device,
            );
            let chunk_t = chunk_tensor.transpose().unsqueeze::<3>();
            let part_logits = hidden.clone().matmul(chunk_t);
            logit_parts.push(part_logits);
            offset += chunk_rows;
        }

        if logit_parts.len() == 1 {
            logit_parts.pop().unwrap().reshape([batch, seq, self.vocab_size])
        } else {
            Tensor::cat(logit_parts, 2).reshape([batch, seq, self.vocab_size])
        }
    }

    /// Create pre-allocated KV cache for all layers.
    pub fn create_cache_preallocated(&self, max_seq: usize, device: &B::Device) -> LayerCaches<B> {
        // GQA: 8 KV heads, head_dim = 3072/32 = 96... wait, head_dim = 128 for decoder
        // n_kv_heads=8, head_dim=128
        LayerCaches::new_preallocated(self.layers.len(), 1, 8, max_seq, 128, device)
    }
}

// ---------------------------------------------------------------------------
// Full Voxtral Model
// ---------------------------------------------------------------------------

pub struct VoxtralModel<B: Backend> {
    pub encoder: AudioEncoder<B>,
    pub decoder: LanguageModel<B>,
    pub adapter: AudioLanguageAdapter<B>,
    pub reshape_factor: usize, // 4
}

impl<B: Backend> VoxtralModel<B> {
    /// Encode audio: mel → encoder → reshape(4×) → adapter → [B, T/16, 3072]
    pub fn encode_audio(&self, mel: Tensor<B, 3>) -> Tensor<B, 3> {
        let encoder_out = self.encoder.forward(mel, 0);
        let reshaped = reshape_encoder_output(encoder_out, self.reshape_factor);
        self.adapter.forward(reshaped)
    }

    /// Streaming transcription: audio embeddings → prefix + autoregressive decode.
    pub fn transcribe_streaming(
        &self,
        mel: Tensor<B, 3>,
        t_embed: Tensor<B, 3>,
    ) -> Vec<i32> {
        let audio_embeds = self.encode_audio(mel);
        let [_, seq_len, d_model] = audio_embeds.dims();

        const PREFIX_LEN: usize = 39;
        const BOS_TOKEN: i32 = 1;
        const STREAMING_PAD: i32 = 32;

        if seq_len < PREFIX_LEN {
            return Vec::new();
        }

        // Build prefix tokens
        let mut prefix: Vec<i32> = vec![BOS_TOKEN];
        prefix.extend(std::iter::repeat_n(STREAMING_PAD, PREFIX_LEN - 1));

        let device = audio_embeds.device();

        // Embed prefix tokens (CPU lookup)
        let prefix_text_embeds = self.decoder.embed_tokens_from_ids(&prefix, 1, PREFIX_LEN, &device);

        // Combine audio + text for prefix positions
        let prefix_audio = audio_embeds
            .clone()
            .slice([0..1, 0..PREFIX_LEN, 0..d_model]);
        let prefix_inputs = prefix_audio + prefix_text_embeds;

        // Prefill: process all prefix positions at once
        let mut decoder_cache = self.decoder.create_cache_preallocated(seq_len, &device);

        let hidden = self.decoder.forward_hidden_with_cache(
            prefix_inputs,
            t_embed.clone(),
            &mut decoder_cache,
        );
        let logits = self.decoder.lm_head(hidden, &device);

        // Get prediction from last prefix position
        let vocab_size = logits.dims()[2];
        let last_logits = logits.slice([0..1, (PREFIX_LEN - 1)..PREFIX_LEN, 0..vocab_size]);
        let first_pred = last_logits.argmax(2);
        let first_token: i32 = first_pred.into_data().as_slice::<i32>().unwrap()[0];

        let mut generated = prefix;
        generated.push(first_token);

        // Pre-slice audio positions for decode loop
        let audio_slices: Vec<Tensor<B, 3>> = (PREFIX_LEN..seq_len)
            .map(|pos| audio_embeds.clone().slice([0..1, pos..pos + 1, 0..d_model]))
            .collect();
        drop(audio_embeds);

        // Autoregressive decode
        for pos in (PREFIX_LEN + 1)..seq_len {
            let new_token = generated[pos - 1];
            let text_embed = self.decoder.embed_tokens_from_ids(&[new_token], 1, 1, &device);
            let audio_pos = audio_slices[pos - 1 - PREFIX_LEN].clone();
            let input = audio_pos + text_embed;

            let hidden = self.decoder.forward_hidden_with_cache(
                input,
                t_embed.clone(),
                &mut decoder_cache,
            );
            let logits = self.decoder.lm_head(hidden, &device);
            let pred = logits.argmax(2);
            let next_token: i32 = pred.into_data().as_slice::<i32>().unwrap()[0];
            generated.push(next_token);
        }

        generated.into_iter().skip(PREFIX_LEN).collect()
    }
}
