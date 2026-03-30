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
    // 32768 supports up to ~10 min audio (30s = ~424 positions, 5min = ~15k)
    let rope = RoPEConfig::new(cfg.head_dim, 32768).with_theta(cfg.rope_theta).init(device);
    let mut layers = Vec::with_capacity(cfg.n_layers);
    for i in 0..cfg.n_layers {
        layers.push(load_encoder_layer(st, i, &cfg, device).with_context(|| format!("encoder layer {i}"))?);
        if (i + 1) % 4 == 0 { gpu_sync(); }
    }
    let norm_w: Tensor<B, 1> = load_tensor(st, &format!("{}.transformer.norm.weight", prefixes::ENCODER), device)?;
    let norm = create_rms_norm(norm_w, cfg.norm_eps, false);
    let encoder = AudioEncoder { conv, rope, layers, norm };

    let an = adapter_names();
    let adapter = AudioLanguageAdapter {
        linear1: load_linear(st, &an.linear1_weight, None, device)?,
        linear2: load_linear(st, &an.linear2_weight, None, device)?,
    };
    Ok((encoder, adapter))
}

fn create_rms_norm<B: Backend>(weight: Tensor<B, 1>, eps: f64, use_standard: bool) -> RmsNorm<B> {
    RmsNorm {
        gamma: burn::module::Param::initialized(burn::module::ParamId::new(), weight),
        epsilon: eps,
        use_standard,
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

/// Load a single decoder layer (weights only, no forward pass).
/// If `cuda_mode`, enables standard softmax + standard RmsNorm.
pub fn load_decoder_layer<B: Backend>(
    st: &safetensors::SafeTensors, i: usize,
    cfg: &config::LanguageModelConfig, device: &B::Device,
) -> Result<DecoderLayer<B>> {
    load_decoder_layer_impl(st, i, cfg, device, false)
}

fn load_decoder_layer_impl<B: Backend>(
    st: &safetensors::SafeTensors, i: usize,
    cfg: &config::LanguageModelConfig, device: &B::Device,
    cuda_mode: bool,
) -> Result<DecoderLayer<B>> {
    let n = decoder_layer_names(i);
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
    Ok(if cuda_mode { layer.with_standard_ops() } else { layer })
}

fn load_and_forward_decoder_layer<B: Backend>(
    st: &safetensors::SafeTensors, layer_idx: usize,
    cfg: &config::LanguageModelConfig, device: &B::Device,
    x: Tensor<B, 3>, t_embed: Tensor<B, 3>, rope: &RoPE<B>, cache: &mut KVCache<B>,
    timing: &mut (f32, f32), // (load_ms, forward_ms)
) -> Result<Tensor<B, 3>> {
    let tl = Instant::now();
    let layer = load_decoder_layer(st, layer_idx, cfg, device)?;
    timing.0 += tl.elapsed().as_secs_f32() * 1000.0;
    let tf = Instant::now();
    let out = layer.forward_with_cache(x, t_embed, rope, cache);
    if (layer_idx + 1) % 4 == 0 || layer_idx == 25 {
        let _ = out.clone().slice([0..1, 0..1, 0..1]).into_data();
    }
    timing.1 += tf.elapsed().as_secs_f32() * 1000.0;
    Ok(out)
}

/// Load complete model — all layers resident on GPU.
/// Requires ~16 GB VRAM (f32). Only viable on CUDA with full VRAM access.
pub fn load_full_model<B: Backend>(
    st: &safetensors::SafeTensors, device: &B::Device,
    sync_fn: impl Fn(),
) -> Result<VoxtralModel<B>> {
    let (mut encoder, adapter) = load_encoder_and_adapter(st, device)?;
    // Enable standard ops on encoder for CUDA (standard softmax + RmsNorm)
    for layer in &mut encoder.layers {
        layer.attention.use_standard_softmax = true;
        layer.attention_norm.use_standard = true;
        layer.ffn_norm.use_standard = true;
    }
    encoder.norm.use_standard = true;

    let dec_cfg = config::LanguageModelConfig::default();
    let (tok_embed_data, vocab_size, d_model) = load_decoder_metadata(st)?;
    let norm_w: Tensor<B, 1> = load_tensor(st, prefixes::FINAL_NORM, device)?;
    let norm = create_rms_norm(norm_w, dec_cfg.norm_eps, true); // CUDA: standard RmsNorm
    let rope = RoPEConfig::new(dec_cfg.head_dim, 16384).with_theta(dec_cfg.rope_theta).init(device);

    let mut layers = Vec::with_capacity(dec_cfg.n_layers);
    for i in 0..dec_cfg.n_layers {
        layers.push(load_decoder_layer_impl(st, i, &dec_cfg, device, true) // CUDA: standard ops
            .with_context(|| format!("decoder layer {i}"))?);
        if (i + 1) % 4 == 0 { sync_fn(); }
        if (i + 1) % 5 == 0 { eprintln!("[load_full_model] decoder layer {}/{}", i + 1, dec_cfg.n_layers); }
    }

    let mut decoder = LanguageModel {
        tok_embed_data, vocab_size, rope, layers, norm, d_model,
        tok_embed_gpu: None,
    };
    // Pre-load embedding table on GPU for fast lm_head + token lookup
    decoder.init_gpu_embedding(device);
    eprintln!("[load_full_model] GPU embedding table loaded ({} × {} = {:.0} MB)",
        vocab_size, d_model, vocab_size as f64 * d_model as f64 * 4.0 / 1e6);
    Ok(VoxtralModel { encoder, decoder, adapter, reshape_factor: 4 })
}

// ---------------------------------------------------------------------------
// Transcription with resident model (all layers on GPU)
// ---------------------------------------------------------------------------

/// Transcribe using a fully-resident model with all optimizations:
/// - GPU embedding table (no 1.6 GB upload per step)
/// - GPU lm_head + argmax (single matmul, no CPU roundtrip for prefill)
/// - Pre-computed ADA scales (eliminates 78 kernels/step)
/// - Standard softmax/RmsNorm on CUDA (no matmul workarounds)
/// - GPU token embedding lookup (no CPU→GPU per step)
pub fn transcribe_resident<B: Backend>(
    model: &VoxtralModel<B>,
    device: &B::Device,
    mel: Tensor<B, 3>,
    t_embed: Tensor<B, 3>,
) -> Result<Vec<i32>> {
    let t0 = Instant::now();

    // Use chunked encoding if chunk_positions > 0 (set via BURN_ENCODER_CHUNK_POSITIONS)
    let chunk_positions = std::env::var("BURN_ENCODER_CHUNK_POSITIONS")
        .ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(0);

    let audio_embeds = if chunk_positions > 0 {
        let embeds = model.encode_audio_chunked(mel, chunk_positions, device);
        let [_, seq_len, _] = embeds.dims();
        eprintln!("[Resident] Encoded (chunked, {}pos): seq_len={} ({:.1}s)",
            chunk_positions, seq_len, t0.elapsed().as_secs_f32());
        embeds
    } else {
        let embeds = model.encode_audio(mel);
        let [_, seq_len, _] = embeds.dims();
        eprintln!("[Resident] Encoded: seq_len={} ({:.1}s)", seq_len, t0.elapsed().as_secs_f32());
        embeds
    };
    let [_, seq_len, d_model] = audio_embeds.dims();

    let dec = &model.decoder;
    let mut caches = dec.create_cache_preallocated(seq_len, device);

    // Pre-compute ADA scales for all 26 layers (constant across decode steps)
    let ada_scales: Vec<Tensor<B, 3>> = dec.layers.iter()
        .map(|layer| layer.precompute_ada_scale(t_embed.clone()))
        .collect();

    const PREFIX_LEN: usize = 39;
    const BOS_TOKEN: i32 = 1;
    const STREAMING_PAD: i32 = 32;

    if seq_len < PREFIX_LEN { return Ok(Vec::new()); }

    let mut prefix: Vec<i32> = vec![BOS_TOKEN];
    prefix.extend(std::iter::repeat_n(STREAMING_PAD, PREFIX_LEN - 1));

    let prefix_text = dec.embed_tokens_from_ids(&prefix, 1, PREFIX_LEN, device);
    let prefix_audio = audio_embeds.clone().slice([0..1, 0..PREFIX_LEN, 0..d_model]);
    let x = prefix_audio + prefix_text;

    // ── Prefill: use pre-computed ADA scales + GPU lm_head ──
    let t1 = Instant::now();
    let hidden = dec.forward_hidden_with_cache_fast(x, &ada_scales, &mut caches);
    let logits = dec.lm_head_gpu(hidden);
    let vocab = logits.dims()[2];
    let last_logits = logits.slice([0..1, (PREFIX_LEN - 1)..PREFIX_LEN, 0..vocab]);
    let argmax_data = last_logits.argmax(2).into_data();
    let first_token: i32 = argmax_data.as_slice::<i32>()
        .map(|v| v[0])
        .or_else(|_| argmax_data.as_slice::<i64>().map(|v| v[0] as i32))
        .unwrap_or(0);
    eprintln!("[Resident] Prefill: {:.1}s, first_token={}", t1.elapsed().as_secs_f32(), first_token);

    let mut generated = prefix;
    generated.push(first_token);

    // Pre-slice audio embeddings as contiguous tensors
    let audio_slices: Vec<Tensor<B, 3>> = (PREFIX_LEN..seq_len)
        .map(|pos| audio_embeds.clone().slice([0..1, pos..pos + 1, 0..d_model]))
        .collect();
    drop(audio_embeds);

    // ── Autoregressive decode — GPU-only pipeline, deferred sync ──
    // Each step: embed_lookup → forward(26 layers) → lm_head → argmax → select(embed)
    // All on GPU. Only sync every SYNC_INTERVAL steps to download token IDs.
    const SYNC_INTERVAL: usize = 16;
    let decode_steps = seq_len - PREFIX_LEN - 1;
    let t2 = Instant::now();

    // First step uses the prefill's first_token (already on CPU)
    let mut prev_embed = dec.embed_token_gpu(first_token, device);
    let mut pending_argmax: Vec<Tensor<B, 3, burn::tensor::Int>> = Vec::new();

    for step in 0..decode_steps {
        let audio_pos = audio_slices[step].clone();
        let x = audio_pos + prev_embed;

        let hidden = dec.forward_hidden_with_cache_fast(x, &ada_scales, &mut caches);
        let last_hidden = hidden.slice([0..1, 0..1, 0..d_model]);

        // GPU-only: lm_head → argmax → select embedding. No into_data()!
        let (next_embed, argmax_idx) = dec.lm_head_and_next_embed(last_hidden);
        pending_argmax.push(argmax_idx);
        prev_embed = next_embed;

        // Batch-sync every SYNC_INTERVAL steps (or at the end)
        if pending_argmax.len() >= SYNC_INTERVAL || step == decode_steps - 1 {
            for idx_tensor in pending_argmax.drain(..) {
                let data = idx_tensor.into_data();
                let token: i32 = data.as_slice::<i32>()
                    .map(|v| v[0])
                    .or_else(|_| data.as_slice::<i64>().map(|v| v[0] as i32))
                    .unwrap_or(0);
                generated.push(token);
            }
        }

        if (step + 1) % 20 == 0 || step == decode_steps - 1 {
            let elapsed = t2.elapsed().as_secs_f32();
            let ms_per_step = elapsed * 1000.0 / (step + 1) as f32;
            eprintln!("[Resident] Decode {}/{} ({:.0}ms/step)",
                step + 1, decode_steps, ms_per_step);
        }
    }
    let total_s = t0.elapsed().as_secs_f32();
    let n_tokens = generated.len() - PREFIX_LEN;
    let audio_secs = seq_len as f32 / 12.5;
    eprintln!("[Resident] Total: {:.1}s ({} tokens, {:.1}× realtime for {:.0}s audio)",
        total_s, n_tokens, total_s / audio_secs, audio_secs);
    Ok(generated.into_iter().skip(PREFIX_LEN).collect())
}

// ---------------------------------------------------------------------------
// Transcription with streaming decoder (WGPU/DZN path)
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
    let norm = create_rms_norm(norm_w, dec_cfg.norm_eps, false); // WGPU: matmul RmsNorm
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

    /// CPU argmax over embedding table — avoids uploading 1.8 GB per decode step.
    /// For hidden [1, 1, d_model], computes dot product with each vocab row on CPU.
    let cpu_argmax = |hidden: &Tensor<B, 3>| -> i32 {
        let data = hidden.clone().reshape([d_model]).into_data();
        let h = data.as_slice::<f32>().unwrap();
        let mut best_id = 0i32;
        let mut best_score = f32::NEG_INFINITY;
        for id in 0..vocab_size {
            let row = &tok_embed_data[id * d_model..(id + 1) * d_model];
            let score: f32 = h.iter().zip(row.iter()).map(|(a, b)| a * b).sum();
            if score > best_score {
                best_score = score;
                best_id = id as i32;
            }
        }
        best_id
    };

    /// GPU lm_head for prefill (larger seq, worth the GPU upload).
    let lm_head_gpu = |hidden: Tensor<B, 3>| -> Tensor<B, 3> {
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

    let mut timing = (0.0f32, 0.0f32); // (load_ms, forward_ms)

    let forward_decoder = |mut x: Tensor<B, 3>, caches: &mut LayerCaches<B>, timing: &mut (f32, f32)| -> Result<Tensor<B, 3>> {
        for i in 0..dec_cfg.n_layers {
            let cache = caches.get_mut(i).unwrap();
            x = load_and_forward_decoder_layer(st, i, &dec_cfg, device, x, t_embed.clone(), &rope, cache, timing)?;
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
    let hidden = forward_decoder(x, &mut caches, &mut timing)?;
    let logits = lm_head_gpu(hidden);
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

    eprintln!("[BF16] Prefill timing: load={:.0}ms fwd={:.0}ms", timing.0, timing.1);
    timing = (0.0, 0.0);

    // ── Autoregressive decode ──
    let decode_steps = seq_len - PREFIX_LEN - 1;
    let t2 = Instant::now();
    let mut lm_head_ms = 0.0f32;
    for step in 0..decode_steps {
        let pos = PREFIX_LEN + 1 + step;
        let prev_token = generated[pos - 1];
        let text_embed = embed_tokens(&[prev_token]);
        let audio_pos = audio_slices[pos - 1 - PREFIX_LEN].clone();
        let x = audio_pos + text_embed;
        let hidden = forward_decoder(x, &mut caches, &mut timing)?;
        let tl = Instant::now();
        // CPU argmax: download 1×3072 hidden → dot product with 150k vocab rows on CPU
        // Much faster than uploading 1.8 GB embedding table to GPU each step
        let last_hidden = hidden.slice([0..1, 0..1, 0..d_model]);
        let next_token = cpu_argmax(&last_hidden);
        lm_head_ms += tl.elapsed().as_secs_f32() * 1000.0;
        generated.push(next_token);
        if (step + 1) % 50 == 0 || step == decode_steps - 1 {
            let elapsed = t2.elapsed().as_secs_f32();
            let ms_per_step = elapsed * 1000.0 / (step + 1) as f32;
            eprintln!("[BF16] Decode {}/{} ({:.0}ms/step) load={:.0}ms fwd={:.0}ms lm={:.0}ms",
                step + 1, decode_steps, ms_per_step, timing.0, timing.1, lm_head_ms);
        }
    }
    let total_s = t0.elapsed().as_secs_f32();
    let n_tokens = generated.len() - PREFIX_LEN;
    eprintln!("[BF16] Total: {:.1}s ({} tokens, {:.0}× realtime for 30s audio)",
        total_s, n_tokens, total_s / 30.0);
    eprintln!("[BF16] Breakdown: load={:.1}s fwd={:.1}s lm_head={:.1}s",
        timing.0 / 1000.0, timing.1 / 1000.0, lm_head_ms / 1000.0);
    Ok(generated.into_iter().skip(PREFIX_LEN).collect())
}
