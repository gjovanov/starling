//! BF16 Voxtral model — encoder, decoder, adapter, and full model.
//!
//! Uses the layer modules from `bf16::layers` and composes them into
//! the complete Voxtral architecture. Matches the reference implementation
//! (voxtral-mini-realtime-rs) exactly.

use burn::nn::{Embedding, Linear};
use burn::tensor::activation::gelu;
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

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
        // Conv downsample: [1, 128, T] ��� [1, 1280, T/4] → transpose → [1, T/4, 1280]
        let x = self.conv.forward(mel);
        let x = x.swap_dims(1, 2); // [B, channels, T] → [B, T, channels]

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
    pub tok_embeddings: Embedding<B>,
    pub rope: RoPE<B>,
    pub layers: Vec<DecoderLayer<B>>,
    pub norm: RmsNorm<B>,
    pub d_model: usize,
}

impl<B: Backend> LanguageModel<B> {
    pub fn embed_tokens(&self, token_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.tok_embeddings.forward(token_ids)
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
    /// hidden [B, seq, d_model] → logits [B, seq, vocab]
    pub fn lm_head(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq, _] = hidden.dims();
        let embed_weights = self.tok_embeddings.weight.val();
        let vocab_size = embed_weights.dims()[0];
        let embed_t = embed_weights.transpose().unsqueeze::<3>();
        let logits = hidden.matmul(embed_t);
        logits.reshape([batch, seq, vocab_size])
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

        const PREFIX_LEN: usize = 38;
        const BOS_TOKEN: i32 = 1;
        const STREAMING_PAD: i32 = 32;

        if seq_len < PREFIX_LEN {
            return Vec::new();
        }

        // Build prefix tokens
        let mut prefix: Vec<i32> = vec![BOS_TOKEN];
        prefix.extend(std::iter::repeat_n(STREAMING_PAD, PREFIX_LEN - 1));

        // Embed prefix tokens
        let prefix_ids = Tensor::<B, 2, Int>::from_data(
            burn::tensor::TensorData::new(prefix.clone(), [1, PREFIX_LEN]),
            &audio_embeds.device(),
        );
        let prefix_text_embeds = self.decoder.embed_tokens(prefix_ids);

        // Combine audio + text for prefix positions
        let prefix_audio = audio_embeds
            .clone()
            .slice([0..1, 0..PREFIX_LEN, 0..d_model]);
        let prefix_inputs = prefix_audio + prefix_text_embeds;

        // Prefill: process all prefix positions at once
        let mut decoder_cache = self.decoder.create_cache_preallocated(seq_len, &audio_embeds.device());

        let hidden = self.decoder.forward_hidden_with_cache(
            prefix_inputs,
            t_embed.clone(),
            &mut decoder_cache,
        );
        let logits = self.decoder.lm_head(hidden);

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
            let token_ids = Tensor::<B, 2, Int>::from_data(
                burn::tensor::TensorData::new(vec![new_token], [1, 1]),
                &audio_slices[0].device(),
            );
            let text_embed = self.decoder.embed_tokens(token_ids);
            let audio_pos = audio_slices[pos - 1 - PREFIX_LEN].clone();
            let input = audio_pos + text_embed;

            let hidden = self.decoder.forward_hidden_with_cache(
                input,
                t_embed.clone(),
                &mut decoder_cache,
            );
            let logits = self.decoder.lm_head(hidden);
            let pred = logits.argmax(2);
            let next_token: i32 = pred.into_data().as_slice::<i32>().unwrap()[0];
            generated.push(next_token);
        }

        generated.into_iter().skip(PREFIX_LEN).collect()
    }
}
