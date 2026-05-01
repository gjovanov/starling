//! Input-embedding helpers for the TTS autoregressive loop.
//!
//! Two entry points used by the pipeline:
//!
//! - [`embed_prompt_with_voice`]: prefill path. Takes a token-ID
//!   sequence + voice embedding rows, produces the per-position input
//!   embeddings that feed the AR LLM. At every audio_token position
//!   the corresponding voice row is summed onto the base
//!   tok_embedding (`input_embedding_concat_type = "sum"` upstream).
//!
//! - [`embed_audio_codes`]: step path. Takes the FMA's freshly-emitted
//!   `[B, 37]` audio_codes, looks them up in the multi-codebook
//!   `audio_codebook_embeddings` table (with per-codebook offsets),
//!   and sums across the codebook dim to produce a single
//!   `[B, 1, dim]` next-step embedding.
//!
//! Multi-codebook offsets mirror upstream `MultiVocabEmbeddings`:
//! `offsets = cumsum([0, 8194, 23, 23, ..., 23][:n_codebooks])`. With
//! `AudioSpecialTokens::COUNT = 2` and the published codebook sizes
//! `[8192, 21, ..., 21]` we get
//! `[0, 8194, 8217, 8240, ..., 8999]` (37 entries).

use anyhow::{anyhow, Result};
use candle_core::{DType, Device, IndexOp, Module, Tensor, D};
use candle_nn::Embedding;

use super::super::flow_matching::args::{AudioSpecialTokens, MultimodalAudioModelArgs};

/// Build the per-codebook offsets table matching upstream's
/// `MultiVocabEmbeddings.offsets`. Length = `num_codebooks`.
pub fn codebook_offsets(args: &MultimodalAudioModelArgs) -> Vec<u32> {
    let semantic = (args.semantic_codebook_size + AudioSpecialTokens::COUNT) as u32;
    let acoustic = (args.acoustic_codebook_size + AudioSpecialTokens::COUNT) as u32;
    let n = args.num_codebooks();
    let mut sizes = vec![0u32; n]; // entries 0..n-1 of `[0] + codebook_sizes[:-1]`
    sizes[0] = 0; // leading 0
    if n >= 2 {
        sizes[1] = semantic;
    }
    for i in 2..n {
        sizes[i] = acoustic;
    }
    let mut offsets = Vec::with_capacity(n);
    let mut acc: u32 = 0;
    for s in &sizes {
        acc += s;
        offsets.push(acc);
    }
    offsets
}

/// Embed a prompt of token IDs, summing voice rows at the FIRST
/// `voice.dim(0)` positions where `input_ids == audio_token_id`.
///
/// `tok_embeddings`: `[vocab, dim]` — the AR LLM's tied embedding table.
/// `input_ids`: `[B, S]` u32 token IDs.
/// `voice`: `[N_voice, dim]` voice embedding rows.
/// `audio_token_id`: usually 24.
///
/// Behaviour: the audio_token positions in `input_ids` (in order) get
/// their tok_embedding ADDED with the voice row at the same index.
/// Positions past `N_voice` audio tokens are passed through with just
/// the tok_embedding (no warning — the caller is responsible for
/// matching prompt audio_token count to voice rows).
pub fn embed_prompt_with_voice(
    tok_embeddings: &Embedding,
    input_ids: &Tensor,
    voice: &Tensor,
    audio_token_id: u32,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let base = tok_embeddings.forward(input_ids)?.to_dtype(dtype)?; // [B, S, D]
    let (b, s, d) = base.dims3()?;
    if b != 1 {
        return Err(anyhow!(
            "embed_prompt_with_voice: only batch=1 supported (got {b})"
        ));
    }
    let voice = voice.to_dtype(dtype)?;
    let (n_voice, voice_dim) = voice.dims2()?;
    if voice_dim != d {
        return Err(anyhow!("voice dim {voice_dim} != embed dim {d}"));
    }

    // Find audio_token positions in CPU memory. Doing this on-host is
    // fine — prompt length is ~hundreds, not millions.
    let ids: Vec<u32> = input_ids.flatten_all()?.to_vec1()?;
    let mut audio_positions: Vec<usize> = Vec::new();
    for (i, id) in ids.iter().enumerate() {
        if *id == audio_token_id {
            audio_positions.push(i);
        }
    }

    // For each voice row, scatter-add into `base[0, pos, :]`.
    //
    // We do this by building a `[1, S, D]` overlay tensor that's zero
    // everywhere except the audio-token rows, then `base + overlay`.
    let n_to_inject = audio_positions.len().min(n_voice);
    if n_to_inject == 0 {
        return Ok(base);
    }
    // Construct overlay: take voice[0..n_to_inject], reshape to
    // [n_to_inject, D], then scatter into a [S, D] tensor at the
    // chosen positions.
    let voice_slice = voice.i((0..n_to_inject, ..))?.contiguous()?; // [n, D]

    // Build a [S, D] zero tensor and scatter rows in.
    let zeros = Tensor::zeros((s, d), dtype, device)?;
    let positions_t: Vec<u32> = audio_positions[..n_to_inject]
        .iter()
        .map(|p| *p as u32)
        .collect();
    let pos_tensor = Tensor::from_vec(positions_t, n_to_inject, device)?;
    let scattered = zeros.index_add(&pos_tensor, &voice_slice, 0)?; // [S, D]
    let overlay = scattered.unsqueeze(0)?; // [1, S, D]
    Ok(base.broadcast_add(&overlay)?)
}

/// Multi-codebook embedding lookup for the autoregressive step.
///
/// `audio_codebook_embeddings`: `[V_total, dim]` — the AR LLM's
/// `mm_audio_embeddings.audio_codebook_embeddings.embeddings` table.
/// `audio_codes`: `[B, n_codebooks]` u32. From the FMA output.
/// `offsets`: `[n_codebooks]` u32 — the per-codebook starting indices.
///
/// Returns `[B, 1, dim]` — the embedding for the next AR LLM step
/// (one position, summed across all 37 codebooks).
pub fn embed_audio_codes(
    audio_codebook_embeddings: &Embedding,
    audio_codes: &Tensor,
    offsets: &Tensor,
) -> Result<Tensor> {
    let (b, n) = audio_codes.dims2()?;
    if offsets.dim(0)? != n {
        return Err(anyhow!(
            "offsets length {} != n_codebooks {n}",
            offsets.dim(0)?
        ));
    }
    // Add per-codebook offsets: [B, n] + [n] broadcast.
    let global = audio_codes.broadcast_add(&offsets.unsqueeze(0)?)?;
    // Lookup → [B, n, dim]
    let embeds = audio_codebook_embeddings.forward(&global)?;
    // Sum across codebook dim → [B, dim], then unsqueeze to [B, 1, dim].
    Ok(embeds.sum(1)?.unsqueeze(1)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::voxtral_tts::flow_matching::args::AcousticTransformerArgs;

    fn dummy_args() -> MultimodalAudioModelArgs {
        MultimodalAudioModelArgs {
            semantic_codebook_size: 8192,
            acoustic_codebook_size: 21,
            n_acoustic_codebook: 36,
            acoustic_transformer_args: AcousticTransformerArgs {
                input_dim: 3072,
                dim: 3072,
                n_layers: 3,
                head_dim: 128,
                hidden_dim: 9216,
                n_heads: 32,
                n_kv_heads: 8,
                use_biases: false,
                rope_theta: 10_000.0,
                norm_eps: 1e-5,
                sigma: 1e-5,
                sigma_max: 1.0,
            },
            p_uncond: 0.0,
            condition_dropped_token_id: 42,
        }
    }

    #[test]
    fn codebook_offsets_match_upstream() {
        let args = dummy_args();
        let off = codebook_offsets(&args);
        // Expected: [0, 8194, 8217, 8240, ..., 8999]
        assert_eq!(off.len(), 37);
        assert_eq!(off[0], 0);
        assert_eq!(off[1], 8194); // 8192 + 2 special tokens
        assert_eq!(off[2], 8194 + 23); // + (21 + 2)
        assert_eq!(off[3], 8194 + 46);
        assert_eq!(off[36], 8194 + 23 * 35);
    }
}
