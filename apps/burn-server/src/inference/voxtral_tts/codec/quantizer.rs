//! Codec quantizer — semantic + acoustic codebooks.
//!
//! Decode-only port; the encoder paths (`encode`) are not implemented
//! since the published checkpoint omits encoder weights.
//!
//! Semantic codebook (Euclidean): the on-disk parameters are
//! `embedding_sum [V, D]` (running sum of cluster contents) and
//! `cluster_usage [V]` (running count). The materialised embedding
//! table is `embedding_sum / clamp(cluster_usage, eps=1e-5)`.
//! Decode is a plain table lookup: `[B, 1, T]` (semantic codes) →
//! `[B, D=256, T]`.
//!
//! Acoustic codebook (Finite-Scalar): pure value rescale —
//! `code * 2 / (n_levels - 1) - 1` maps integer codes ∈ `[0, levels-1]`
//! back to floats ∈ `[-1, 1]`. There are no learned acoustic-codebook
//! parameters; the codes ARE the embeddings (modulo rescale).
//!
//! `MistralAudioCodebook.decode([B, K=37, T]) → [B, D=256+36=292, T]`
//! concatenates the two streams along the channel axis.

use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;

use super::args::AudioTokenizerArgs;

const SEMANTIC_USAGE_EPS: f64 = 1e-5;

pub struct SemanticCodebook {
    /// Materialised embedding table `[V=8192, D=256]`. Stored at the
    /// model's compute dtype so the lookup result matches downstream
    /// operations without a per-call cast.
    embedding: Tensor,
}

impl SemanticCodebook {
    pub fn load(
        vb: VarBuilder,
        codebook_size: usize,
        codebook_dim: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let cluster_usage = vb.get(codebook_size, "cluster_usage")?.to_dtype(DType::F32)?;
        let embedding_sum = vb.get((codebook_size, codebook_dim), "embedding_sum")?.to_dtype(DType::F32)?;
        let denom = cluster_usage.clamp(SEMANTIC_USAGE_EPS, f64::INFINITY)?.unsqueeze(1)?;
        let embedding = embedding_sum.broadcast_div(&denom)?;
        let embedding = embedding.to_dtype(dtype)?.to_device(device)?;
        Ok(Self { embedding })
    }

    /// Decode `codes [B, 1, T]` (u32) → `[B, D, T]`.
    pub fn decode(&self, codes: &Tensor) -> Result<Tensor> {
        let (b, k, t) = codes.dims3()?;
        if k != 1 {
            return Err(anyhow::anyhow!(
                "SemanticCodebook expects exactly 1 codebook (got K={k})"
            ));
        }
        let codes = codes.squeeze(1)?; // [B, T]
        // Flat lookup: reshape to [B*T] for index_select, then back.
        let codes_flat = codes.reshape((b * t,))?.to_dtype(DType::U32)?;
        let looked_up = self.embedding.index_select(&codes_flat, 0)?; // [B*T, D]
        let d = looked_up.dim(1)?;
        let out = looked_up.reshape((b, t, d))?; // [B, T, D]
        Ok(out.transpose(1, 2)?.contiguous()?) // [B, D, T]
    }
}

pub struct AcousticCodebook {
    n_levels: usize,
    n_codebooks: usize,
}

impl AcousticCodebook {
    pub fn new(codebook_size: usize, codebook_dim: usize) -> Self {
        Self {
            n_levels: codebook_size,
            n_codebooks: codebook_dim,
        }
    }

    pub fn n_codebooks(&self) -> usize {
        self.n_codebooks
    }

    /// Decode `codes [B, K=36, T]` (u32) → `[B, K, T]` (float in `[-1, 1]`).
    pub fn decode(&self, codes: &Tensor, dtype: DType) -> Result<Tensor> {
        let codes_f = codes.to_dtype(dtype)?;
        let levels_minus_one = (self.n_levels - 1) as f64;
        // x * 2 / (levels - 1) - 1
        let scaled = (codes_f * (2.0 / levels_minus_one))?;
        Ok((scaled - 1.0)?)
    }
}

pub struct MistralAudioCodebook {
    pub semantic: SemanticCodebook,
    pub acoustic: AcousticCodebook,
}

impl MistralAudioCodebook {
    pub fn load(
        vb: VarBuilder,
        args: &AudioTokenizerArgs,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let semantic = SemanticCodebook::load(
            vb.pp("semantic_codebook"),
            args.semantic_codebook_size,
            args.semantic_dim,
            device,
            dtype,
        )?;
        let acoustic = AcousticCodebook::new(args.acoustic_codebook_size, args.acoustic_dim);
        Ok(Self { semantic, acoustic })
    }

    /// Total decoded channel dimension: `semantic_dim + acoustic_dim`
    /// (256 + 36 = 292 for Voxtral-4B-TTS).
    pub fn total_dim(&self, args: &AudioTokenizerArgs) -> usize {
        args.semantic_dim + args.acoustic_dim
    }

    /// Decode `codes [B, K=37, T]` → `[B, D=292, T]`. Splits the
    /// codebook stream into the leading 1-codebook semantic part and
    /// the trailing 36-codebook acoustic part.
    pub fn decode(&self, codes: &Tensor, dtype: DType) -> Result<Tensor> {
        let semantic_codes = codes.i((.., 0..1, ..))?;
        let acoustic_codes = codes.i((.., 1.., ..))?;

        let semantic_emb = self.semantic.decode(&semantic_codes)?.to_dtype(dtype)?;
        let acoustic_emb = self.acoustic.decode(&acoustic_codes, dtype)?;
        Ok(Tensor::cat(&[&semantic_emb, &acoustic_emb], 1)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn acoustic_decode_rescale_to_minus_one_one() {
        // n_levels=21 → boundaries: 0 → -1, 10 → 0, 20 → 1.
        let codes = Tensor::from_vec(vec![0u32, 5, 10, 15, 20], (1, 5, 1), &Device::Cpu).unwrap();
        // We pass it as [B=1, K=5, T=1] so K matches a hypothetical 5-codebook setup.
        let cb = AcousticCodebook::new(21, 5);
        let out = cb.decode(&codes, DType::F32).unwrap();
        let v: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        let expected = [-1.0_f32, -0.5, 0.0, 0.5, 1.0];
        for (i, e) in expected.iter().enumerate() {
            assert!((v[i] - e).abs() < 1e-6, "[{i}] {} vs {}", v[i], e);
        }
    }
}
