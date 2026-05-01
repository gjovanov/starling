//! TTS autoregressive driver — wires the AR LLM and the flow-matching
//! head together into a generation loop.
//!
//! Each call to [`TtsAutoregressor::synthesize_codes`] takes a prompt
//! (token IDs + a voice embedding + the FMA noise stream) and returns
//! the `[T_frames, num_codebooks=37]` audio code matrix that's
//! suitable for feeding into the codec decoder.
//!
//! Generation terminates on either:
//!   - `semantic_code == END_AUDIO (= 1)` for a frame, OR
//!   - `max_frames` if the upstream model goes silent without emitting
//!     END_AUDIO (safety bound).
//!
//! The noise stream is supplied by the caller as `[max_frames, 36]`
//! to keep the loop deterministic and side-effect-free w.r.t. RNG.
//! Production callers feed a CSPRNG; tests feed zeros for repeatability.

use anyhow::{anyhow, Result};
use candle_core::{DType, Device, IndexOp, Tensor};

use super::ar_llm::{
    codebook_offsets, embed_audio_codes, embed_prompt_with_voice, ArLlmKvCache, ArLlmModel,
    RopeCache,
};
use super::flow_matching::{
    args::{AudioSpecialTokens, MultimodalAudioModelArgs},
    model::FlowMatchingAudioTransformer,
};
use super::voice::VoiceEmbedding;

/// Per-step inputs for the autoregressor. Construct via
/// [`TtsAutoregressor::synthesize_codes`].
pub struct TtsAutoregressor<'a> {
    pub ar_llm: &'a ArLlmModel,
    pub fma: &'a FlowMatchingAudioTransformer,
    pub rope: &'a RopeCache,
    pub fma_args: &'a MultimodalAudioModelArgs,
    pub audio_token_id: u32,
}

impl<'a> TtsAutoregressor<'a> {
    /// Run the full autoregressive loop. Returns `[T_frames, 37]` u32
    /// audio code matrix (semantic + acoustic), suitable for feeding
    /// into [`super::codec::VoxtralTTSAudioTokenizer::decode`].
    ///
    /// `prompt_ids`: `[1, S_prompt]` token IDs.
    /// `voice`: voice embedding loaded from `voice/<name>.safetensors`.
    /// `noise`: `[max_frames, n_acoustic_codebook]` floats.
    /// `max_frames`: safety bound on generation steps.
    pub fn synthesize_codes(
        &self,
        prompt_ids: &Tensor,
        voice: &VoiceEmbedding,
        noise: &Tensor,
        max_frames: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Tensor> {
        let max_n = noise.dim(0)?;
        if max_n < max_frames {
            return Err(anyhow!(
                "noise has {max_n} rows, max_frames is {max_frames}"
            ));
        }
        if noise.dim(1)? != self.fma_args.n_acoustic_codebook {
            return Err(anyhow!(
                "noise[1]={} != n_acoustic_codebook={}",
                noise.dim(1)?,
                self.fma_args.n_acoustic_codebook
            ));
        }

        // 1. Build prompt embeddings with voice prefix injection.
        let prompt_embeds = embed_prompt_with_voice(
            &self.ar_llm.tok_embeddings,
            prompt_ids,
            &voice.tensor,
            self.audio_token_id,
            device,
            dtype,
        )?;

        // 2. AR LLM prefill.
        let mut kv = ArLlmKvCache::new(self.ar_llm.args.n_layers);
        let prefill_hidden = self.ar_llm.forward_embeds(&prompt_embeds, self.rope, &mut kv)?;
        let last_hidden = prefill_hidden.i((.., prefill_hidden.dim(1)? - 1.., ..))?;

        // 3. Multi-codebook offsets — needed for embedding the codes
        //    that come back from the FMA on each step.
        let offsets_v = codebook_offsets(self.fma_args);
        let offsets = Tensor::from_vec(offsets_v, self.fma_args.num_codebooks(), device)?;

        // 4. Generation loop.
        let mut frame_codes: Vec<Tensor> = Vec::with_capacity(max_frames);
        let mut hidden = last_hidden;
        for step in 0..max_frames {
            // Run FMA on the [B=1, dim] last-hidden of this step.
            let h2d = hidden.squeeze(1)?; // [1, dim]
            let x_0 = noise.i(step..step + 1)?.contiguous()?; // [1, n_acoustic]
            let codes = self.fma.forward_with_noise(&h2d, &x_0)?; // [1, 37] u32-ish
            let codes_u = codes.to_dtype(DType::U32)?;

            // Halt check on semantic_code (codes[:, 0]).
            let sem: Vec<u32> = codes_u.i((.., 0..1))?.flatten_all()?.to_vec1()?;
            if sem[0] == AudioSpecialTokens::END_AUDIO {
                break;
            }
            frame_codes.push(codes_u.clone());

            // Embed back into AR LLM input.
            let next_emb = embed_audio_codes(&self.ar_llm.audio_codebook_embeddings, &codes_u, &offsets)?;
            // Step AR LLM by one position.
            hidden = self.ar_llm.forward_embeds(&next_emb, self.rope, &mut kv)?;
        }

        if frame_codes.is_empty() {
            // No usable frames generated.
            return Tensor::zeros((0, self.fma_args.num_codebooks()), DType::U32, device)
                .map_err(|e| anyhow!(e));
        }

        // Stack [n_frames, 37].
        let refs: Vec<&Tensor> = frame_codes.iter().collect();
        let stacked = Tensor::cat(&refs, 0)?;
        Ok(stacked)
    }
}
