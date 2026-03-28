//! SafeTensors weight loading utilities for BF16 Voxtral model.
//!
//! Handles BF16→f32 conversion, Linear weight transpose (PyTorch→Burn),
//! and memory-mapped file loading for 9GB model files.

use anyhow::{Context, Result};
use burn::module::{Param, ParamId};
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::nn::{Embedding, EmbeddingConfig, Linear, PaddingConfig1d};
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use safetensors::SafeTensors;
use std::path::Path;

// ---------------------------------------------------------------------------
// OwnedSafeTensors — memory-safe wrapper with mmap support
// ---------------------------------------------------------------------------

enum BytesBacking {
    Mapped(memmap2::Mmap),
}

impl AsRef<[u8]> for BytesBacking {
    fn as_ref(&self) -> &[u8] {
        match self {
            BytesBacking::Mapped(m) => m,
        }
    }
}

/// Owning wrapper for SafeTensors that keeps mmap alive.
pub struct OwnedSafeTensors {
    _backing: BytesBacking,
    safetensors: SafeTensors<'static>,
}

impl OwnedSafeTensors {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = std::fs::File::open(path.as_ref())
            .with_context(|| format!("Failed to open: {}", path.as_ref().display()))?;
        let mmap = unsafe { memmap2::Mmap::map(&file) }
            .with_context(|| format!("Failed to mmap: {}", path.as_ref().display()))?;
        let backing = BytesBacking::Mapped(mmap);

        let safetensors = unsafe {
            let static_ref: &'static [u8] = std::mem::transmute(backing.as_ref());
            SafeTensors::deserialize(static_ref).context("Failed to deserialize SafeTensors")?
        };

        Ok(Self {
            _backing: backing,
            safetensors,
        })
    }
}

impl std::ops::Deref for OwnedSafeTensors {
    type Target = SafeTensors<'static>;
    fn deref(&self) -> &Self::Target {
        &self.safetensors
    }
}

// ---------------------------------------------------------------------------
// Tensor loading with BF16→f32 conversion
// ---------------------------------------------------------------------------

/// Load a tensor from SafeTensors, converting BF16/F16 to f32.
pub fn load_tensor<B: Backend, const D: usize>(
    st: &SafeTensors,
    name: &str,
    device: &B::Device,
) -> Result<Tensor<B, D>> {
    let view = st
        .tensor(name)
        .with_context(|| format!("Tensor '{}' not found", name))?;

    let shape: Vec<usize> = view.shape().to_vec();
    let data: Vec<f32> = match view.dtype() {
        safetensors::Dtype::F32 => view
            .data()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        safetensors::Dtype::BF16 => view
            .data()
            .chunks_exact(2)
            .map(|b| half::bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
            .collect(),
        safetensors::Dtype::F16 => view
            .data()
            .chunks_exact(2)
            .map(|b| half::f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
            .collect(),
        other => anyhow::bail!("Unsupported dtype: {:?}", other),
    };

    Ok(Tensor::from_data(TensorData::new(data, shape), device))
}

// ---------------------------------------------------------------------------
// Linear layer construction (with PyTorch→Burn transpose)
// ---------------------------------------------------------------------------

/// Create Linear from weight (transposed) + optional bias.
pub fn linear_from_weights<B: Backend>(
    weight: Tensor<B, 2>,
    bias: Option<Tensor<B, 1>>,
) -> Linear<B> {
    let weight = weight.transpose(); // PyTorch [out, in] → Burn [in, out]
    Linear {
        weight: Param::initialized(ParamId::new(), weight),
        bias: bias.map(|b| Param::initialized(ParamId::new(), b)),
    }
}

/// Load a Linear layer from SafeTensors with optional bias.
pub fn load_linear<B: Backend>(
    st: &SafeTensors,
    weight_name: &str,
    bias_name: Option<&str>,
    device: &B::Device,
) -> Result<Linear<B>> {
    let weight: Tensor<B, 2> = load_tensor(st, weight_name, device)?;
    let bias = match bias_name {
        Some(name) if st.tensor(name).is_ok() => {
            Some(load_tensor::<B, 1>(st, name, device)?)
        }
        _ => None,
    };
    Ok(linear_from_weights(weight, bias))
}

// ---------------------------------------------------------------------------
// Embedding construction
// ---------------------------------------------------------------------------

/// Create Embedding from a weight tensor [vocab, dim].
pub fn embedding_from_weights<B: Backend>(weight: Tensor<B, 2>) -> Embedding<B> {
    let [vocab, dim] = weight.dims();
    let config = EmbeddingConfig::new(vocab, dim);
    let mut embedding = config.init::<B>(&weight.device());
    embedding.weight = Param::initialized(ParamId::new(), weight);
    embedding
}

// ---------------------------------------------------------------------------
// Conv1d construction
// ---------------------------------------------------------------------------

/// Create Conv1d from weight [out, in, k] + bias [out], stride=2, pad=1.
pub fn conv1d_from_weights<B: Backend>(
    weight: Tensor<B, 3>,
    bias: Tensor<B, 1>,
) -> Conv1d<B> {
    let [out_ch, in_ch, kernel] = weight.dims();
    let config = Conv1dConfig::new(in_ch, out_ch, kernel)
        .with_stride(2)
        .with_padding(PaddingConfig1d::Explicit(1));
    let mut conv = config.init::<B>(&weight.device());
    conv.weight = Param::initialized(ParamId::new(), weight);
    conv.bias = Some(Param::initialized(ParamId::new(), bias));
    conv
}

// ---------------------------------------------------------------------------
// Weight name helpers
// ---------------------------------------------------------------------------

pub mod prefixes {
    pub const ENCODER: &str =
        "mm_streams_embeddings.embedding_module.whisper_encoder";
    pub const TOK_EMBEDDINGS: &str =
        "mm_streams_embeddings.embedding_module.tok_embeddings.weight";
    pub const ADAPTER: &str =
        "mm_streams_embeddings.embedding_module.audio_language_projection";
    pub const FINAL_NORM: &str = "norm.weight";
}

pub struct EncoderLayerNames {
    pub attention_norm: String,
    pub wq_weight: String, pub wq_bias: String,
    pub wk_weight: String,
    pub wv_weight: String, pub wv_bias: String,
    pub wo_weight: String, pub wo_bias: String,
    pub ffn_norm: String,
    pub w1_weight: String,
    pub w2_weight: String, pub w2_bias: String,
    pub w3_weight: String,
}

pub fn encoder_layer_names(i: usize) -> EncoderLayerNames {
    let p = format!("{}.transformer.layers.{}", prefixes::ENCODER, i);
    EncoderLayerNames {
        attention_norm: format!("{p}.attention_norm.weight"),
        wq_weight: format!("{p}.attention.wq.weight"),
        wq_bias: format!("{p}.attention.wq.bias"),
        wk_weight: format!("{p}.attention.wk.weight"),
        wv_weight: format!("{p}.attention.wv.weight"),
        wv_bias: format!("{p}.attention.wv.bias"),
        wo_weight: format!("{p}.attention.wo.weight"),
        wo_bias: format!("{p}.attention.wo.bias"),
        ffn_norm: format!("{p}.ffn_norm.weight"),
        w1_weight: format!("{p}.feed_forward.w1.weight"),
        w2_weight: format!("{p}.feed_forward.w2.weight"),
        w2_bias: format!("{p}.feed_forward.w2.bias"),
        w3_weight: format!("{p}.feed_forward.w3.weight"),
    }
}

pub struct DecoderLayerNames {
    pub ada_norm_down: String,
    pub ada_norm_up: String,
    pub attention_norm: String,
    pub wq_weight: String,
    pub wk_weight: String,
    pub wv_weight: String,
    pub wo_weight: String,
    pub ffn_norm: String,
    pub w1_weight: String,
    pub w2_weight: String,
    pub w3_weight: String,
}

pub fn decoder_layer_names(i: usize) -> DecoderLayerNames {
    let p = format!("layers.{}", i);
    DecoderLayerNames {
        ada_norm_down: format!("{p}.ada_rms_norm_t_cond.0.weight"),
        ada_norm_up: format!("{p}.ada_rms_norm_t_cond.2.weight"),
        attention_norm: format!("{p}.attention_norm.weight"),
        wq_weight: format!("{p}.attention.wq.weight"),
        wk_weight: format!("{p}.attention.wk.weight"),
        wv_weight: format!("{p}.attention.wv.weight"),
        wo_weight: format!("{p}.attention.wo.weight"),
        ffn_norm: format!("{p}.ffn_norm.weight"),
        w1_weight: format!("{p}.feed_forward.w1.weight"),
        w2_weight: format!("{p}.feed_forward.w2.weight"),
        w3_weight: format!("{p}.feed_forward.w3.weight"),
    }
}

pub struct ConvNames {
    pub conv1_weight: String, pub conv1_bias: String,
    pub conv2_weight: String, pub conv2_bias: String,
}

pub fn conv_names() -> ConvNames {
    ConvNames {
        conv1_weight: format!("{}.conv_layers.0.conv.weight", prefixes::ENCODER),
        conv1_bias: format!("{}.conv_layers.0.conv.bias", prefixes::ENCODER),
        conv2_weight: format!("{}.conv_layers.1.conv.weight", prefixes::ENCODER),
        conv2_bias: format!("{}.conv_layers.1.conv.bias", prefixes::ENCODER),
    }
}

pub struct AdapterNames {
    pub linear1_weight: String,
    pub linear2_weight: String,
}

pub fn adapter_names() -> AdapterNames {
    AdapterNames {
        linear1_weight: format!("{}.0.weight", prefixes::ADAPTER),
        linear2_weight: format!("{}.2.weight", prefixes::ADAPTER),
    }
}
