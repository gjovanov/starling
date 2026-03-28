//! SafeTensors BF16 model loader for Voxtral.
//!
//! Inspired by antirez/voxtral.c: weights stay as BF16 in mmap'd memory.
//! Per-layer forward: BF16→f32 into GPU scratch buffer, matmul, free.
//! Peak VRAM: one decoder layer (~445 MB) + KV cache + activations.

use anyhow::{Context, Result};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use std::sync::Arc;

use super::layers::*;
use super::model::*;
use super::weights::*;
use crate::inference::config;

fn gpu_sync() {
    use burn::backend::wgpu::WgpuDevice;
    use cubecl::Runtime;
    let device = WgpuDevice::default();
    let client = burn::backend::wgpu::WgpuRuntime::client(&device);
    client.flush();
    std::thread::sleep(std::time::Duration::from_millis(50));
}

// ---------------------------------------------------------------------------
// Encoder (bulk-loaded to GPU — fits in ~4.2 GB, freed after encoding)
// ---------------------------------------------------------------------------

fn load_encoder<B: Backend>(st: &safetensors::SafeTensors, device: &B::Device) -> Result<AudioEncoder<B>> {
    let cfg = config::AudioEncoderConfig::default();

    let cn = conv_names();
    let conv = ConvDownsampler {
        conv1: conv1d_from_weights(load_tensor(st, &cn.conv1_weight, device)?, load_tensor(st, &cn.conv1_bias, device)?),
        conv2: conv1d_from_weights(load_tensor(st, &cn.conv2_weight, device)?, load_tensor(st, &cn.conv2_bias, device)?),
    };

    let rope = RoPEConfig::new(cfg.head_dim, 4096).with_theta(cfg.rope_theta).init(device);

    let mut layers = Vec::with_capacity(cfg.n_layers);
    for i in 0..cfg.n_layers {
        layers.push(load_encoder_layer(st, i, &cfg, device).with_context(|| format!("encoder layer {i}"))?);
        if (i + 1) % 4 == 0 {
            gpu_sync();
            if (i + 1) % 8 == 0 { eprintln!("[BF16]   encoder layer {}/{}", i + 1, cfg.n_layers); }
        }
    }

    let norm_w: Tensor<B, 1> = load_tensor(st, &format!("{}.transformer.norm.weight", prefixes::ENCODER), device)?;
    let norm = create_rms_norm(norm_w, cfg.norm_eps);

    Ok(AudioEncoder { conv, rope, layers, norm })
}

fn create_rms_norm<B: Backend>(weight: Tensor<B, 1>, eps: f64) -> RmsNorm<B> {
    let d = weight.dims()[0];
    let device = weight.device();
    let mut norm = RmsNormConfig::new(d).with_eps(eps).init(&device);
    norm.weight.gamma = burn::module::Param::initialized(burn::module::ParamId::new(), weight);
    norm
}

fn load_encoder_layer<B: Backend>(
    st: &safetensors::SafeTensors, i: usize, cfg: &config::AudioEncoderConfig, device: &B::Device,
) -> Result<EncoderLayer<B>> {
    let n = encoder_layer_names(i);
    let attention = Attention::new(
        load_linear(st, &n.wq_weight, Some(&n.wq_bias), device)?,
        load_linear(st, &n.wk_weight, None, device)?,
        load_linear(st, &n.wv_weight, Some(&n.wv_bias), device)?,
        load_linear(st, &n.wo_weight, Some(&n.wo_bias), device)?,
        cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, Some(cfg.sliding_window),
    );
    Ok(EncoderLayer::new(
        load_tensor(st, &n.attention_norm, device)?,
        attention,
        load_tensor(st, &n.ffn_norm, device)?,
        load_linear(st, &n.w1_weight, None, device)?,
        load_linear(st, &n.w2_weight, Some(&n.w2_bias), device)?,
        load_linear(st, &n.w3_weight, None, device)?,
        cfg.norm_eps,
    ))
}

// ---------------------------------------------------------------------------
// Adapter (small — loaded to GPU, freed after use)
// ---------------------------------------------------------------------------

fn load_adapter<B: Backend>(st: &safetensors::SafeTensors, device: &B::Device) -> Result<AudioLanguageAdapter<B>> {
    let n = adapter_names();
    Ok(AudioLanguageAdapter {
        linear1: load_linear(st, &n.linear1_weight, None, device)?,
        linear2: load_linear(st, &n.linear2_weight, None, device)?,
    })
}

// ---------------------------------------------------------------------------
// Streaming decoder layer — load from mmap, forward, free
// ---------------------------------------------------------------------------

/// Load a single decoder layer from SafeTensors to GPU, run forward, free.
fn load_and_forward_decoder_layer<B: Backend>(
    st: &safetensors::SafeTensors,
    layer_idx: usize,
    cfg: &config::LanguageModelConfig,
    device: &B::Device,
    x: Tensor<B, 3>,
    t_embed: Tensor<B, 3>,
    rope: &RoPE<B>,
    cache: &mut KVCache<B>,
) -> Result<Tensor<B, 3>> {
    let n = decoder_layer_names(layer_idx);

    // Load layer weights from mmap → GPU (~445 MB f32)
    let attention = Attention::new(
        load_linear(st, &n.wq_weight, None, device)?,
        load_linear(st, &n.wk_weight, None, device)?,
        load_linear(st, &n.wv_weight, None, device)?,
        load_linear(st, &n.wo_weight, None, device)?,
        cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, Some(cfg.sliding_window),
    );

    let ada_w0: Tensor<B, 2> = load_tensor(st, &n.ada_norm_down, device)?;
    let ada_w2: Tensor<B, 2> = load_tensor(st, &n.ada_norm_up, device)?;

    let layer = DecoderLayer::new(
        ada_w0, ada_w2,
        load_tensor(st, &n.attention_norm, device)?,
        attention,
        load_tensor(st, &n.ffn_norm, device)?,
        load_linear(st, &n.w1_weight, None, device)?,
        load_linear(st, &n.w2_weight, None, device)?,
        load_linear(st, &n.w3_weight, None, device)?,
        32,
        cfg.norm_eps,
    );

    // Forward pass — layer weights are on GPU
    let out = layer.forward_with_cache(x, t_embed, rope, cache);

    // layer dropped here → GPU memory freed for next layer
    Ok(out)
}

// ---------------------------------------------------------------------------
// Phased transcription: encoder → free → streaming decoder
// ---------------------------------------------------------------------------

/// Phased transcription with per-layer decoder streaming.
/// Peak VRAM: max(encoder ~4.2 GB, one decoder layer ~0.5 GB + KV cache).
pub fn phased_transcribe<B: Backend>(
    model_dir: &std::path::Path,
    device: &B::Device,
    mel: Tensor<B, 3>,
    t_embed: Tensor<B, 3>,
) -> Result<Vec<i32>> {
    let st_path = model_dir.join("consolidated.safetensors");
    let owned = Arc::new(OwnedSafeTensors::from_file(&st_path)?);
    let st = &*owned;

    // ── Phase 1: Encoder + Adapter → audio_embeds ──
    eprintln!("[BF16 Phase 1] Loading encoder + adapter...");
    let encoder = load_encoder(st, device)?;
    let adapter = load_adapter(st, device)?;
    eprintln!("[BF16 Phase 1] Encoding audio...");

    let encoder_out = encoder.forward(mel, 0);
    let reshaped = reshape_encoder_output(encoder_out, 4);
    let audio_embeds = adapter.forward(reshaped);

    let [_, seq_len, d_model] = audio_embeds.dims();
    eprintln!("[BF16 Phase 1] Audio encoded: seq_len={}", seq_len);

    // Free encoder + adapter
    drop(encoder);
    drop(adapter);
    gpu_sync();
    eprintln!("[BF16 Phase 1] Encoder freed");

    // ── Phase 2: Streaming decoder (one layer at a time) ──
    let dec_cfg = config::LanguageModelConfig::default();

    // Load tok_embeddings on CPU (for embed_tokens and lm_head)
    let tok_view = st.tensor(prefixes::TOK_EMBEDDINGS).context("tok_embeddings not found")?;
    let tok_shape: Vec<usize> = tok_view.shape().to_vec();
    let tok_embed_data: Vec<f32> = match tok_view.dtype() {
        safetensors::Dtype::BF16 => tok_view.data().chunks_exact(2)
            .map(|b| half::bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
            .collect(),
        safetensors::Dtype::F32 => tok_view.data().chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        other => anyhow::bail!("Unsupported tok_embeddings dtype: {:?}", other),
    };
    let vocab_size = tok_shape[0];

    // Load final norm (tiny — stays on GPU)
    let norm_w: Tensor<B, 1> = load_tensor(st, prefixes::FINAL_NORM, device)?;
    let norm = create_rms_norm(norm_w, dec_cfg.norm_eps);

    // RoPE (tiny — stays on GPU)
    let rope = RoPEConfig::new(dec_cfg.head_dim, 16384).with_theta(dec_cfg.rope_theta).init(device);

    // KV cache (pre-allocated for all 26 layers)
    let mut caches = LayerCaches::new_preallocated(
        dec_cfg.n_layers, 1, dec_cfg.n_kv_heads, seq_len, dec_cfg.head_dim, device,
    );

    eprintln!("[BF16 Phase 2] Streaming decoder ({} layers × {} positions)...", dec_cfg.n_layers, seq_len);

    // Helper: embed tokens from CPU data
    let embed_tokens = |ids: &[i32], batch: usize, seq: usize| -> Tensor<B, 3> {
        let mut output = vec![0.0f32; ids.len() * d_model];
        for (i, &id) in ids.iter().enumerate() {
            let row_start = (id as usize) * d_model;
            let row_end = row_start + d_model;
            if row_end <= tok_embed_data.len() {
                output[i * d_model..(i + 1) * d_model]
                    .copy_from_slice(&tok_embed_data[row_start..row_end]);
            }
        }
        Tensor::from_data(burn::tensor::TensorData::new(output, [batch, seq, d_model]), device)
    };

    // Helper: chunked lm_head
    let lm_head = |hidden: Tensor<B, 3>| -> Tensor<B, 3> {
        let [batch, seq, _] = hidden.dims();
        let max_rows = 128 * 1024 * 1024 / (d_model * 4);
        let mut parts = Vec::new();
        let mut offset = 0;
        while offset < vocab_size {
            let rows = (vocab_size - offset).min(max_rows);
            let chunk = &tok_embed_data[offset * d_model..(offset + rows) * d_model];
            let ct: Tensor<B, 2> = Tensor::from_data(
                burn::tensor::TensorData::new(chunk.to_vec(), [rows, d_model]), device,
            );
            parts.push(hidden.clone().matmul(ct.transpose().unsqueeze::<3>()));
            offset += rows;
        }
        if parts.len() == 1 {
            parts.pop().unwrap().reshape([batch, seq, vocab_size])
        } else {
            Tensor::cat(parts, 2).reshape([batch, seq, vocab_size])
        }
    };

    // ── Prefill: process all prefix positions through 26 layers ──
    const PREFIX_LEN: usize = 38;
    const BOS_TOKEN: i32 = 1;
    const STREAMING_PAD: i32 = 32;

    if seq_len < PREFIX_LEN {
        return Ok(Vec::new());
    }

    let mut prefix: Vec<i32> = vec![BOS_TOKEN];
    prefix.extend(std::iter::repeat_n(STREAMING_PAD, PREFIX_LEN - 1));

    let prefix_text = embed_tokens(&prefix, 1, PREFIX_LEN);
    let prefix_audio = audio_embeds.clone().slice([0..1, 0..PREFIX_LEN, 0..d_model]);
    let mut x = prefix_audio + prefix_text;

    // Stream through 26 decoder layers (prefill)
    for i in 0..dec_cfg.n_layers {
        let cache = caches.get_mut(i).unwrap();
        x = load_and_forward_decoder_layer(st, i, &dec_cfg, device, x, t_embed.clone(), &rope, cache)?;
        // Layer weights freed automatically (dropped at end of function)
        if (i + 1) % 4 == 0 {
            gpu_sync();
        }
    }
    let hidden = norm.forward(x);
    let logits = lm_head(hidden);

    let vocab = logits.dims()[2];
    let last_logits = logits.slice([0..1, (PREFIX_LEN - 1)..PREFIX_LEN, 0..vocab]);
    let first_token: i32 = last_logits.argmax(2).into_data().as_slice::<i32>().unwrap()[0];
    eprintln!("[BF16 Phase 2] Prefill done, first_token={}", first_token);

    let mut generated = prefix;
    generated.push(first_token);

    // Pre-slice audio
    let audio_slices: Vec<Tensor<B, 3>> = (PREFIX_LEN..seq_len)
        .map(|pos| audio_embeds.clone().slice([0..1, pos..pos + 1, 0..d_model]))
        .collect();
    drop(audio_embeds);

    // ── Autoregressive decode: each position streams through 26 layers ──
    let decode_steps = seq_len - PREFIX_LEN - 1;
    for step in 0..decode_steps {
        let pos = PREFIX_LEN + 1 + step;
        let prev_token = generated[pos - 1];
        let text_embed = embed_tokens(&[prev_token], 1, 1);
        let audio_pos = audio_slices[pos - 1 - PREFIX_LEN].clone();
        let mut x = audio_pos + text_embed;

        // Stream through 26 layers
        for i in 0..dec_cfg.n_layers {
            let cache = caches.get_mut(i).unwrap();
            x = load_and_forward_decoder_layer(st, i, &dec_cfg, device, x, t_embed.clone(), &rope, cache)?;
        }
        let hidden = norm.forward(x);
        let logits = lm_head(hidden);
        let next_token: i32 = logits.argmax(2).into_data().as_slice::<i32>().unwrap()[0];
        generated.push(next_token);

        if (step + 1) % 10 == 0 || step == decode_steps - 1 {
            eprintln!("[BF16 Phase 2] Decode step {}/{}, token={}", step + 1, decode_steps, next_token);
        }
    }

    Ok(generated.into_iter().skip(PREFIX_LEN).collect())
}
