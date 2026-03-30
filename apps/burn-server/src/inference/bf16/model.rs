//! BF16 Voxtral model — encoder, decoder, adapter, and full model.

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
    pub fn forward(&self, mel: Tensor<B, 3>, offset: usize) -> Tensor<B, 3> {
        let x = self.conv.forward(mel);
        let x = x.swap_dims(1, 2);
        let mut x = x;
        for layer in &self.layers {
            x = layer.forward(x, &self.rope, offset);
        }
        self.norm.forward(x)
    }

    pub fn forward_with_cache(
        &self, mel: Tensor<B, 3>, caches: &mut LayerCaches<B>,
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
    pub linear1: Linear<B>,
    pub linear2: Linear<B>,
}

impl<B: Backend> AudioLanguageAdapter<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.linear1.forward(x);
        let x = gelu(x);
        self.linear2.forward(x)
    }
}

pub fn reshape_encoder_output<B: Backend>(x: Tensor<B, 3>, factor: usize) -> Tensor<B, 3> {
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
    pub tok_embed_data: Vec<f32>,
    pub vocab_size: usize,
    pub rope: RoPE<B>,
    pub layers: Vec<DecoderLayer<B>>,
    pub norm: RmsNorm<B>,
    pub d_model: usize,
}

impl<B: Backend> LanguageModel<B> {
    pub fn embed_tokens_from_ids(&self, ids: &[i32], batch: usize, seq: usize, device: &B::Device) -> Tensor<B, 3> {
        let mut output = vec![0.0f32; ids.len() * self.d_model];
        for (i, &id) in ids.iter().enumerate() {
            if id >= 0 && (id as usize) < self.vocab_size {
                let start = (id as usize) * self.d_model;
                let end = start + self.d_model;
                if end <= self.tok_embed_data.len() {
                    output[i * self.d_model..(i + 1) * self.d_model]
                        .copy_from_slice(&self.tok_embed_data[start..end]);
                }
            }
        }
        Tensor::from_data(burn::tensor::TensorData::new(output, [batch, seq, self.d_model]), device)
    }

    pub fn forward_hidden(&self, mut x: Tensor<B, 3>, t_embed: Tensor<B, 3>, offset: usize) -> Tensor<B, 3> {
        for layer in &self.layers {
            x = layer.forward(x, t_embed.clone(), &self.rope, offset);
        }
        self.norm.forward(x)
    }

    pub fn forward_hidden_with_cache(&self, mut x: Tensor<B, 3>, t_embed: Tensor<B, 3>, caches: &mut LayerCaches<B>) -> Tensor<B, 3> {
        for (i, layer) in self.layers.iter().enumerate() {
            if let Some(cache) = caches.get_mut(i) {
                x = layer.forward_with_cache(x, t_embed.clone(), &self.rope, cache);
            }
        }
        self.norm.forward(x)
    }

    pub fn lm_head(&self, hidden: Tensor<B, 3>, device: &B::Device) -> Tensor<B, 3> {
        let [batch, seq, _] = hidden.dims();
        let max_rows = 128 * 1024 * 1024 / (self.d_model * 4);
        let mut parts = Vec::new();
        let mut offset = 0;
        while offset < self.vocab_size {
            let rows = (self.vocab_size - offset).min(max_rows);
            let chunk = &self.tok_embed_data[offset * self.d_model..(offset + rows) * self.d_model];
            let ct: Tensor<B, 2> = Tensor::from_data(
                burn::tensor::TensorData::new(chunk.to_vec(), [rows, self.d_model]), device,
            );
            parts.push(hidden.clone().matmul(ct.transpose().unsqueeze::<3>()));
            offset += rows;
        }
        if parts.len() == 1 {
            parts.pop().unwrap().reshape([batch, seq, self.vocab_size])
        } else {
            Tensor::cat(parts, 2).reshape([batch, seq, self.vocab_size])
        }
    }

    pub fn create_cache_preallocated(&self, max_seq: usize, device: &B::Device) -> LayerCaches<B> {
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
    pub reshape_factor: usize,
}

impl<B: Backend> VoxtralModel<B> {
    pub fn encode_audio(&self, mel: Tensor<B, 3>) -> Tensor<B, 3> {
        let encoder_out = self.encoder.forward(mel, 0);
        let reshaped = reshape_encoder_output(encoder_out, self.reshape_factor);
        self.adapter.forward(reshaped)
    }
}
