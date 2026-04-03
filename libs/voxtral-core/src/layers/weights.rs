//! Weight name constants for loading Voxtral model weights from TrevorJS GGUF.
//!
//! Names match the TrevorJS/voxtral-mini-realtime-gguf GGUF file format,
//! which uses Mistral's original naming convention.

/// Common weight name prefixes.
pub mod prefixes {
    /// Token embedding table.
    pub const TOK_EMBEDDINGS: &str = "mm_streams_embeddings.embedding_module.tok_embeddings.weight";
    /// Final layer norm before LM head.
    pub const FINAL_NORM: &str = "norm.weight";
    /// Audio encoder prefix.
    pub const ENCODER: &str = "mm_streams_embeddings.embedding_module.whisper_encoder";
}

/// Weight names for a single encoder transformer layer.
pub struct EncoderLayerWeightNames {
    pub attention_norm: String,
    pub wq_weight: String,
    pub wq_bias: String,
    pub wk_weight: String,
    pub wk_bias: String,
    pub wv_weight: String,
    pub wv_bias: String,
    pub wo_weight: String,
    pub wo_bias: String,
    pub ffn_norm: String,
    pub w1_weight: String,
    pub w2_weight: String,
    pub w2_bias: String,
    pub w3_weight: String,
}

/// Get weight names for encoder layer `i`.
pub fn encoder_layer_weight_names(i: usize) -> EncoderLayerWeightNames {
    let prefix = format!("{}.transformer.layers.{i}", prefixes::ENCODER);
    EncoderLayerWeightNames {
        attention_norm: format!("{prefix}.attention_norm.weight"),
        wq_weight: format!("{prefix}.attention.wq.weight"),
        wq_bias: format!("{prefix}.attention.wq.bias"),
        wk_weight: format!("{prefix}.attention.wk.weight"),
        wk_bias: format!("{prefix}.attention.wk.bias"),
        wv_weight: format!("{prefix}.attention.wv.weight"),
        wv_bias: format!("{prefix}.attention.wv.bias"),
        wo_weight: format!("{prefix}.attention.wo.weight"),
        wo_bias: format!("{prefix}.attention.wo.bias"),
        ffn_norm: format!("{prefix}.ffn_norm.weight"),
        w1_weight: format!("{prefix}.feed_forward.w1.weight"),
        w2_weight: format!("{prefix}.feed_forward.w2.weight"),
        w2_bias: format!("{prefix}.feed_forward.w2.bias"),
        w3_weight: format!("{prefix}.feed_forward.w3.weight"),
    }
}

/// Weight names for a single decoder transformer layer.
pub struct DecoderLayerWeightNames {
    pub attention_norm: String,
    pub wq_weight: String,
    pub wk_weight: String,
    pub wv_weight: String,
    pub wo_weight: String,
    pub ffn_norm: String,
    pub w1_weight: String,
    pub w2_weight: String,
    pub w3_weight: String,
    pub ada_norm_down: String,
    pub ada_norm_up: String,
}

/// Get weight names for decoder layer `i`.
pub fn decoder_layer_weight_names(i: usize) -> DecoderLayerWeightNames {
    let prefix = format!("layers.{i}");
    DecoderLayerWeightNames {
        attention_norm: format!("{prefix}.attention_norm.weight"),
        wq_weight: format!("{prefix}.attention.wq.weight"),
        wk_weight: format!("{prefix}.attention.wk.weight"),
        wv_weight: format!("{prefix}.attention.wv.weight"),
        wo_weight: format!("{prefix}.attention.wo.weight"),
        ffn_norm: format!("{prefix}.ffn_norm.weight"),
        w1_weight: format!("{prefix}.feed_forward.w1.weight"),
        w2_weight: format!("{prefix}.feed_forward.w2.weight"),
        w3_weight: format!("{prefix}.feed_forward.w3.weight"),
        ada_norm_down: format!("{prefix}.ada_rms_norm_t_cond.0.weight"),
        ada_norm_up: format!("{prefix}.ada_rms_norm_t_cond.2.weight"),
    }
}

/// Weight names for the conv downsampler.
pub struct ConvWeightNames {
    pub conv1_weight: String,
    pub conv1_bias: String,
    pub conv2_weight: String,
    pub conv2_bias: String,
}

/// Get weight names for the conv downsampler.
pub fn conv_weight_names() -> ConvWeightNames {
    let prefix = format!("{}.conv_layers", prefixes::ENCODER);
    ConvWeightNames {
        conv1_weight: format!("{prefix}.0.conv.weight"),
        conv1_bias: format!("{prefix}.0.conv.bias"),
        conv2_weight: format!("{prefix}.1.conv.weight"),
        conv2_bias: format!("{prefix}.1.conv.bias"),
    }
}

/// Weight names for the audio-language adapter.
pub struct AdapterWeightNames {
    pub linear1_weight: String,
    pub linear2_weight: String,
}

/// Get weight names for the adapter MLP.
pub fn adapter_weight_names() -> AdapterWeightNames {
    let prefix = "mm_streams_embeddings.embedding_module.audio_language_projection";
    AdapterWeightNames {
        linear1_weight: format!("{prefix}.0.weight"),
        linear2_weight: format!("{prefix}.2.weight"),
    }
}
