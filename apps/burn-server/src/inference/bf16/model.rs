//! BF16 Voxtral model — encoder, decoder, adapter, and full model.

use burn::nn::Linear;
use burn::tensor::activation::gelu;
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};

use super::layers::*;
use crate::audio::mel::{MelConfig, MelSpectrogram};
use crate::audio::pad::{pad_audio, PadConfig};

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

    /// Forward pass using fused flash attention (CUDA). O(N) memory vs O(N²).
    pub fn forward_flash(&self, mel: Tensor<B, 3>, offset: usize) -> Tensor<B, 3> {
        let x = self.conv.forward(mel);
        let x = x.swap_dims(1, 2);
        let mut x = x;
        for layer in &self.layers {
            x = layer.forward_flash(x, &self.rope, offset);
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

    /// Run conv downsampler + transpose (no transformer layers).
    /// Returns [batch, positions, d_model] ready for chunked transformer processing.
    pub fn conv_and_transpose(&self, mel: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.conv.forward(mel);
        x.swap_dims(1, 2)
    }

    /// Run transformer layers with KV cache (no conv, no final norm).
    /// For chunked encoding: call conv_and_transpose() once, then this per chunk.
    pub fn forward_layers_with_cache(
        &self, mut x: Tensor<B, 3>, caches: &mut LayerCaches<B>,
    ) -> Tensor<B, 3> {
        for (i, layer) in self.layers.iter().enumerate() {
            if let Some(cache) = caches.get_mut(i) {
                x = layer.forward_with_cache(x, &self.rope, cache);
            }
        }
        x // no norm — caller applies after all chunks
    }

    /// Create pre-allocated encoder KV caches.
    pub fn create_cache(&self, max_seq: usize, device: &B::Device) -> LayerCaches<B> {
        let n_kv_heads = self.layers[0].attention.n_kv_heads; // 32 (MHA)
        let head_dim = self.layers[0].attention.head_dim; // 64
        LayerCaches::new_preallocated(self.layers.len(), 1, n_kv_heads, max_seq, head_dim, device)
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
    /// Persistent GPU embedding table [vocab_size, d_model]. Populated for resident model.
    pub tok_embed_gpu: Option<Tensor<B, 2>>,
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

    /// Initialize persistent GPU embedding table from CPU tok_embed_data.
    pub fn init_gpu_embedding(&mut self, device: &B::Device) {
        let data = burn::tensor::TensorData::new(
            self.tok_embed_data.clone(), [self.vocab_size, self.d_model],
        );
        self.tok_embed_gpu = Some(Tensor::from_data(data, device));
    }

    /// Fast lm_head using persistent GPU embedding. Single matmul, no upload.
    pub fn lm_head_gpu(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        let embed = self.tok_embed_gpu.as_ref().expect("call init_gpu_embedding first");
        // hidden [B, S, D] @ embed^T [D, V] → [B, S, V]
        hidden.matmul(embed.clone().transpose().unsqueeze::<3>())
    }

    /// Forward through all decoder layers with pre-computed ADA scales.
    pub fn forward_hidden_with_cache_fast(
        &self, mut x: Tensor<B, 3>, ada_scales: &[Tensor<B, 3>], caches: &mut LayerCaches<B>,
    ) -> Tensor<B, 3> {
        for (i, layer) in self.layers.iter().enumerate() {
            if let Some(cache) = caches.get_mut(i) {
                x = layer.forward_with_cache_precomputed(x, ada_scales[i].clone(), &self.rope, cache);
            }
        }
        self.norm.forward(x)
    }

    /// GPU-side token embedding lookup using persistent embedding table.
    pub fn embed_token_gpu(&self, token_id: i32, device: &B::Device) -> Tensor<B, 3> {
        let embed = self.tok_embed_gpu.as_ref().expect("call init_gpu_embedding first");
        if token_id >= 0 && (token_id as usize) < self.vocab_size {
            embed.clone().slice([token_id as usize..token_id as usize + 1, 0..self.d_model])
                .unsqueeze::<3>() // [1, 1, d_model]
        } else {
            Tensor::<B, 3>::zeros([1, 1, self.d_model], device)
        }
    }

    /// GPU-only lm_head → argmax → embedding lookup. No CPU roundtrip.
    /// Returns (next_embedding [1,1,d_model], argmax_idx [1,1,1] on GPU).
    /// Call into_data() on argmax_idx LATER (batched) to get token IDs.
    pub fn lm_head_and_next_embed(&self, hidden: Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 3, burn::tensor::Int>) {
        let embed = self.tok_embed_gpu.as_ref().expect("call init_gpu_embedding first");
        // hidden [1,1,D] @ embed_T [D,V] → logits [1,1,V]
        let logits = hidden.matmul(embed.clone().transpose().unsqueeze::<3>());
        // argmax on GPU → [1,1,1] int tensor (stays on GPU!)
        let argmax_idx = logits.argmax(2);
        // select embedding row using GPU argmax index
        // argmax_idx [1,1,1] → reshape to [1] for select()
        let idx_1d = argmax_idx.clone().reshape([1]);
        // embed [V, D] → select(0, [1]) → [1, D]
        let next_embed = embed.clone().select(0, idx_1d).reshape([1, 1, self.d_model]);
        (next_embed, argmax_idx)
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

    /// Encode audio using fused flash attention (CUDA). O(N) memory.
    pub fn encode_audio_flash(&self, mel: Tensor<B, 3>) -> Tensor<B, 3> {
        let encoder_out = self.encoder.forward_flash(mel, 0);
        let reshaped = reshape_encoder_output(encoder_out, self.reshape_factor);
        self.adapter.forward(reshaped)
    }

    /// Encode audio in chunks using pre-allocated KV cache.
    /// Conv runs once on full mel, then transformer layers process chunks.
    /// Each chunk's attention only computes against cached + current positions.
    pub fn encode_audio_chunked(
        &self, mel: Tensor<B, 3>, chunk_positions: usize, device: &B::Device,
    ) -> Tensor<B, 3> {
        // 1. Conv once on full mel → [1, total_pos, d_model]
        let enc_input = self.encoder.conv_and_transpose(mel);
        let [batch, total_pos, d_model] = enc_input.dims();

        // 2. Pre-allocated encoder caches (no per-chunk allocation)
        let mut enc_caches = self.encoder.create_cache(total_pos, device);

        // 3. Process in chunks through 32 transformer layers
        let mut chunk_outputs: Vec<Tensor<B, 3>> = Vec::new();
        let mut offset = 0;
        let n_chunks = (total_pos + chunk_positions - 1) / chunk_positions;
        while offset < total_pos {
            let end = (offset + chunk_positions).min(total_pos);
            let chunk = enc_input.clone().slice([0..batch, offset..end, 0..d_model]);
            let chunk_out = self.encoder.forward_layers_with_cache(chunk, &mut enc_caches);
            chunk_outputs.push(chunk_out);
            let chunk_idx = offset / chunk_positions + 1;
            if chunk_idx % 2 == 0 || offset + chunk_positions >= total_pos {
                eprintln!("[ChunkedEnc] chunk {}/{} pos={}-{}", chunk_idx, n_chunks, offset, end);
            }
            offset = end;
        }

        // 4. Concatenate + final norm + reshape + adapter
        let full_enc_out = if chunk_outputs.len() == 1 {
            chunk_outputs.pop().unwrap()
        } else {
            Tensor::cat(chunk_outputs, 1)
        };
        let normed = self.encoder.norm.forward(full_enc_out);
        let reshaped = reshape_encoder_output(normed, self.reshape_factor);
        self.adapter.forward(reshaped)
    }

    /// Encode a single audio frame (small window ~135ms) through full encoder.
    /// Matches vllm's per-frame approach: mel → conv → 32 encoder layers → norm → reshape → adapter.
    /// Returns decoder-space embeddings [1, num_tokens, 3072].
    /// Returns None if the frame produces 0 decoder positions (due to reshape alignment).
    pub fn encode_frame(
        &self,
        audio_window: &[f32],
        mel_spec: &MelSpectrogram,
        device: &B::Device,
    ) -> Option<Tensor<B, 3>> {
        // 1. Mel spectrogram on the small window
        let log_mel = mel_spec.compute_log(audio_window);
        let n_frames = log_mel.len();
        let n_mels = if n_frames > 0 { log_mel[0].len() } else { return None };
        if n_frames == 0 { return None; }

        let mut flat = vec![0.0f32; n_mels * n_frames];
        for (t, frame) in log_mel.iter().enumerate() {
            for (m, &val) in frame.iter().enumerate() {
                flat[m * n_frames + t] = val;
            }
        }
        let mel: Tensor<B, 3> = Tensor::from_data(
            TensorData::new(flat, [1, n_mels, n_frames]), device,
        );

        // 2. Conv + full encoder (one-shot, no KV cache — small window)
        let encoder_out = self.encoder.forward(mel, 0);
        let [_, enc_pos, _] = encoder_out.dims();

        // 3. Reshape (4×) — need multiple of 4 encoder positions
        let usable = (enc_pos / self.reshape_factor) * self.reshape_factor;
        if usable == 0 { return None; }

        let encoder_out = if usable < enc_pos {
            let [b, _, d] = encoder_out.dims();
            encoder_out.slice([0..b, 0..usable, 0..d])
        } else {
            encoder_out
        };
        let normed = self.encoder.norm.forward(encoder_out);
        let reshaped = reshape_encoder_output(normed, self.reshape_factor);
        let adapted = self.adapter.forward(reshaped);

        let [_, dec_pos, _] = adapted.dims();
        if dec_pos == 0 { return None; }
        Some(adapted)
    }
}
