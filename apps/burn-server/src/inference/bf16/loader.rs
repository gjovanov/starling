//! SafeTensors BF16 model loader for Voxtral.

use anyhow::{Context, Result};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use super::layers::*;
use super::model::*;
use super::weights::*;
use crate::inference::config;

pub fn load_model<B: Backend>(
    model_dir: &std::path::Path,
    device: &B::Device,
) -> Result<VoxtralModel<B>> {
    let st_path = model_dir.join("consolidated.safetensors");
    eprintln!("[BF16] Loading SafeTensors from {}", st_path.display());

    let owned = OwnedSafeTensors::from_file(&st_path)?;
    let st = &*owned;

    let encoder = load_encoder(st, device)?;
    eprintln!("[BF16] Encoder loaded (32 layers)");

    let adapter = load_adapter(st, device)?;
    eprintln!("[BF16] Adapter loaded");

    let decoder = load_decoder(st, device)?;
    eprintln!("[BF16] Decoder loaded (26 layers)");

    Ok(VoxtralModel { encoder, decoder, adapter, reshape_factor: 4 })
}

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
        if (i + 1) % 8 == 0 { eprintln!("[BF16]   encoder layer {}/{}", i + 1, cfg.n_layers); }
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
        load_linear(st, &n.wk_weight, None, device)?, // K: no bias
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

fn load_decoder<B: Backend>(st: &safetensors::SafeTensors, device: &B::Device) -> Result<LanguageModel<B>> {
    let cfg = config::LanguageModelConfig::default();

    // Load tok_embeddings as raw f32 on CPU (avoids 1.6 GB GPU buffer limit).
    // Sparse lookups (embed_tokens) use CPU data; lm_head chunks on-the-fly.
    let tok_view = st.tensor(prefixes::TOK_EMBEDDINGS)
        .context("tok_embeddings not found")?;
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
    let d_model = tok_shape[1];
    eprintln!("[BF16]   tok_embeddings loaded on CPU ({} × {})", vocab_size, d_model);

    let rope = RoPEConfig::new(cfg.head_dim, 16384).with_theta(cfg.rope_theta).init(device);

    let mut layers = Vec::with_capacity(cfg.n_layers);
    for i in 0..cfg.n_layers {
        layers.push(load_decoder_layer(st, i, &cfg, device).with_context(|| format!("decoder layer {i}"))?);
        if (i + 1) % 5 == 0 { eprintln!("[BF16]   decoder layer {}/{}", i + 1, cfg.n_layers); }
    }

    let norm_w: Tensor<B, 1> = load_tensor(st, prefixes::FINAL_NORM, device)?;
    let norm = create_rms_norm(norm_w, cfg.norm_eps);

    Ok(LanguageModel { tok_embed_data, vocab_size, rope, layers, norm, d_model: cfg.dim })
}

fn load_decoder_layer<B: Backend>(
    st: &safetensors::SafeTensors, i: usize, cfg: &config::LanguageModelConfig, device: &B::Device,
) -> Result<DecoderLayer<B>> {
    let n = decoder_layer_names(i);

    let attention = Attention::new(
        load_linear(st, &n.wq_weight, None, device)?,
        load_linear(st, &n.wk_weight, None, device)?,
        load_linear(st, &n.wv_weight, None, device)?,
        load_linear(st, &n.wo_weight, None, device)?,
        cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, Some(cfg.sliding_window),
    );

    // ADA weights are stored as [out, in] in SafeTensors (like all Linear)
    // load_linear transposes them. But ADA w0/w2 are used as Linear layers
    // in the AdaRmsNorm forward pass, so the transpose is correct.
    let ada_w0: Tensor<B, 2> = load_tensor(st, &n.ada_norm_down, device)?;
    let ada_w2: Tensor<B, 2> = load_tensor(st, &n.ada_norm_up, device)?;

    Ok(DecoderLayer::new(
        ada_w0, ada_w2,
        load_tensor(st, &n.attention_norm, device)?,
        attention,
        load_tensor(st, &n.ffn_norm, device)?,
        load_linear(st, &n.w1_weight, None, device)?,
        load_linear(st, &n.w2_weight, None, device)?,
        load_linear(st, &n.w3_weight, None, device)?,
        32, // t_cond_dim
        cfg.norm_eps,
    ))
}

fn load_adapter<B: Backend>(st: &safetensors::SafeTensors, device: &B::Device) -> Result<AudioLanguageAdapter<B>> {
    let n = adapter_names();
    Ok(AudioLanguageAdapter {
        linear1: load_linear(st, &n.linear1_weight, None, device)?,
        linear2: load_linear(st, &n.linear2_weight, None, device)?,
    })
}
