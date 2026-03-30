//! SafeTensors BF16 model loader with per-layer streaming decoder.

use anyhow::{Context, Result};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use std::time::Instant;

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
    std::thread::sleep(std::time::Duration::from_millis(30));
}

// ---------------------------------------------------------------------------
// Encoder + Adapter (resident on GPU)
// ---------------------------------------------------------------------------

pub fn load_encoder_and_adapter<B: Backend>(
    st: &safetensors::SafeTensors, device: &B::Device,
) -> Result<(AudioEncoder<B>, AudioLanguageAdapter<B>)> {
    let cfg = config::AudioEncoderConfig::default();
    let cn = conv_names();
    let conv = ConvDownsampler {
        conv1: conv1d_from_weights(load_tensor(st, &cn.conv1_weight, device)?, load_tensor(st, &cn.conv1_bias, device)?, 1),
        conv2: conv1d_from_weights(load_tensor(st, &cn.conv2_weight, device)?, load_tensor(st, &cn.conv2_bias, device)?, 2),
    };
    let rope = RoPEConfig::new(cfg.head_dim, 4096).with_theta(cfg.rope_theta).init(device);
    let mut layers = Vec::with_capacity(cfg.n_layers);
    for i in 0..cfg.n_layers {
        layers.push(load_encoder_layer(st, i, &cfg, device).with_context(|| format!("encoder layer {i}"))?);
        if (i + 1) % 4 == 0 { gpu_sync(); }
    }
    let norm_w: Tensor<B, 1> = load_tensor(st, &format!("{}.transformer.norm.weight", prefixes::ENCODER), device)?;
    let norm = create_rms_norm(norm_w, cfg.norm_eps);
    let encoder = AudioEncoder { conv, rope, layers, norm };

    let an = adapter_names();
    let adapter = AudioLanguageAdapter {
        linear1: load_linear(st, &an.linear1_weight, None, device)?,
        linear2: load_linear(st, &an.linear2_weight, None, device)?,
    };
    Ok((encoder, adapter))
}

fn create_rms_norm<B: Backend>(weight: Tensor<B, 1>, eps: f64) -> RmsNorm<B> {
    RmsNorm {
        gamma: burn::module::Param::initialized(burn::module::ParamId::new(), weight),
        epsilon: eps,
    }
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
// Decoder helpers
// ---------------------------------------------------------------------------

pub fn load_decoder_metadata(st: &safetensors::SafeTensors) -> Result<(Vec<f32>, usize, usize)> {
    let tok_view = st.tensor(prefixes::TOK_EMBEDDINGS).context("tok_embeddings not found")?;
    let tok_shape: Vec<usize> = tok_view.shape().to_vec();
    let tok_embed_data: Vec<f32> = match tok_view.dtype() {
        safetensors::Dtype::BF16 => tok_view.data().chunks_exact(2)
            .map(|b| half::bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32()).collect(),
        safetensors::Dtype::F32 => tok_view.data().chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect(),
        other => anyhow::bail!("Unsupported tok_embeddings dtype: {:?}", other),
    };
    Ok((tok_embed_data, tok_shape[0], tok_shape[1]))
}

fn load_and_forward_decoder_layer<B: Backend>(
    st: &safetensors::SafeTensors, layer_idx: usize,
    cfg: &config::LanguageModelConfig, device: &B::Device,
    x: Tensor<B, 3>, t_embed: Tensor<B, 3>, rope: &RoPE<B>, cache: &mut KVCache<B>,
) -> Result<Tensor<B, 3>> {
    let n = decoder_layer_names(layer_idx);
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
        32, cfg.norm_eps,
    );
    let out = layer.forward_with_cache(x, t_embed, rope, cache);
    // Force GPU pipeline flush before dropping layer weights
    let _ = out.clone().slice([0..1, 0..1, 0..1]).into_data();
    Ok(out)
}

// ---------------------------------------------------------------------------
// Transcription with streaming decoder
// ---------------------------------------------------------------------------

pub fn transcribe<B: Backend>(
    st: &safetensors::SafeTensors,
    encoder: &AudioEncoder<B>,
    adapter: &AudioLanguageAdapter<B>,
    device: &B::Device,
    mel: Tensor<B, 3>,
    t_embed: Tensor<B, 3>,
) -> Result<Vec<i32>> {
    let t0 = Instant::now();
    let encoder_out = encoder.forward(mel, 0);
    let reshaped = reshape_encoder_output(encoder_out, 4);
    let audio_embeds = adapter.forward(reshaped);
    let [_, seq_len, d_model] = audio_embeds.dims();
    eprintln!("[BF16] Encoded: seq_len={} ({:.1}s)", seq_len, t0.elapsed().as_secs_f32());

    let dec_cfg = config::LanguageModelConfig::default();
    let (tok_embed_data, vocab_size, _) = load_decoder_metadata(st)?;
    let norm_w: Tensor<B, 1> = load_tensor(st, prefixes::FINAL_NORM, device)?;
    let norm = create_rms_norm(norm_w, dec_cfg.norm_eps);
    let rope = RoPEConfig::new(dec_cfg.head_dim, 16384).with_theta(dec_cfg.rope_theta).init(device);
    let mut caches = LayerCaches::new_preallocated(
        dec_cfg.n_layers, 1, dec_cfg.n_kv_heads, seq_len, dec_cfg.head_dim, device,
    );

    let embed_tokens = |ids: &[i32]| -> Tensor<B, 3> {
        let seq = ids.len();
        let mut out = vec![0.0f32; seq * d_model];
        for (i, &id) in ids.iter().enumerate() {
            if id >= 0 && (id as usize) < vocab_size {
                let start = (id as usize) * d_model;
                let end = start + d_model;
                if end <= tok_embed_data.len() {
                    out[i * d_model..(i + 1) * d_model].copy_from_slice(&tok_embed_data[start..end]);
                }
            }
        }
        Tensor::from_data(burn::tensor::TensorData::new(out, [1, seq, d_model]), device)
    };

    let lm_head = |hidden: Tensor<B, 3>| -> Tensor<B, 3> {
        let [batch, seq, _] = hidden.dims();
        let max_rows = 128 * 1024 * 1024 / (d_model * 4);
        let mut parts = Vec::new();
        let mut off = 0;
        while off < vocab_size {
            let rows = (vocab_size - off).min(max_rows);
            let chunk = &tok_embed_data[off * d_model..(off + rows) * d_model];
            let ct: Tensor<B, 2> = Tensor::from_data(
                burn::tensor::TensorData::new(chunk.to_vec(), [rows, d_model]), device,
            );
            parts.push(hidden.clone().matmul(ct.transpose().unsqueeze::<3>()));
            off += rows;
        }
        if parts.len() == 1 { parts.pop().unwrap().reshape([batch, seq, vocab_size]) }
        else { Tensor::cat(parts, 2).reshape([batch, seq, vocab_size]) }
    };

    let forward_decoder = |mut x: Tensor<B, 3>, caches: &mut LayerCaches<B>| -> Result<Tensor<B, 3>> {
        for i in 0..dec_cfg.n_layers {
            let cache = caches.get_mut(i).unwrap();
            x = load_and_forward_decoder_layer(st, i, &dec_cfg, device, x, t_embed.clone(), &rope, cache)?;
            if (i + 1) % 8 == 0 { gpu_sync(); }
        }
        Ok(norm.forward(x))
    };

    // ── Prefill ──
    const PREFIX_LEN: usize = 39;
    const BOS_TOKEN: i32 = 1;
    const STREAMING_PAD: i32 = 32;

    if seq_len < PREFIX_LEN { return Ok(Vec::new()); }

    let mut prefix: Vec<i32> = vec![BOS_TOKEN];
    prefix.extend(std::iter::repeat_n(STREAMING_PAD, PREFIX_LEN - 1));

    let prefix_text = embed_tokens(&prefix);
    let prefix_audio = audio_embeds.clone().slice([0..1, 0..PREFIX_LEN, 0..d_model]);
    let x = prefix_audio + prefix_text;

    let t1 = Instant::now();
    let hidden = forward_decoder(x, &mut caches)?;
    let logits = lm_head(hidden);
    let vocab = logits.dims()[2];
    let last_logits = logits.slice([0..1, (PREFIX_LEN - 1)..PREFIX_LEN, 0..vocab]);
    let argmax_data = last_logits.argmax(2).into_data();
    let first_token: i32 = argmax_data.as_slice::<i32>()
        .map(|v| v[0])
        .or_else(|_| argmax_data.as_slice::<i64>().map(|v| v[0] as i32))
        .unwrap_or(0);
    eprintln!("[BF16] Prefill: {:.1}s, first_token={}", t1.elapsed().as_secs_f32(), first_token);

    let mut generated = prefix;
    generated.push(first_token);

    let audio_slices: Vec<Tensor<B, 3>> = (PREFIX_LEN..seq_len)
        .map(|pos| audio_embeds.clone().slice([0..1, pos..pos + 1, 0..d_model]))
        .collect();
    drop(audio_embeds);

    // ── Autoregressive decode ──
    let decode_steps = seq_len - PREFIX_LEN - 1;
    let t2 = Instant::now();
    for step in 0..decode_steps {
        let pos = PREFIX_LEN + 1 + step;
        let prev_token = generated[pos - 1];
        let text_embed = embed_tokens(&[prev_token]);
        let audio_pos = audio_slices[pos - 1 - PREFIX_LEN].clone();
        let x = audio_pos + text_embed;
        let hidden = forward_decoder(x, &mut caches)?;
        let logits = lm_head(hidden);
        let argmax_data = logits.argmax(2).into_data();
        let next_token: i32 = argmax_data.as_slice::<i32>()
            .map(|v| v[0])
            .or_else(|_| argmax_data.as_slice::<i64>().map(|v| v[0] as i32))
            .unwrap_or(0);
        generated.push(next_token);
        if (step + 1) % 20 == 0 || step == decode_steps - 1 {
            let elapsed = t2.elapsed().as_secs_f32();
            let steps_per_sec = (step + 1) as f32 / elapsed;
            eprintln!("[BF16] Decode {}/{} ({:.1} steps/s)", step + 1, decode_steps, steps_per_sec);
        }
    }
    eprintln!("[BF16] Total: {:.1}s ({} tokens)", t0.elapsed().as_secs_f32(), generated.len() - PREFIX_LEN);
    Ok(generated.into_iter().skip(PREFIX_LEN).collect())
}
