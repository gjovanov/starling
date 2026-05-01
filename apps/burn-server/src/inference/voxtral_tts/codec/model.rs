//! Top-level codec decoder. Wires the 8-block alternating
//! conv ↔ transformer chain with per-block sliding-window tracking,
//! plus the final `output_proj` and patch rearrangement.
//!
//! Reference: `voxtral_tts_audio_tokenizer.VoxtralTTSAudioTokenizer
//! .decode()` and `_forward_decoder()`.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use std::collections::BTreeMap;

use super::args::AudioTokenizerArgs;
use super::conv::{CausalConv1d, CausalConvTranspose1d, PadMode};
use super::quantizer::MistralAudioCodebook;
use super::transformer::CodecTransformer;

enum DecoderBlock {
    InputConv(CausalConv1d),
    Upsample(CausalConvTranspose1d),
    Transformer(CodecTransformer),
}

pub struct VoxtralTTSAudioTokenizer {
    pub quantizer: MistralAudioCodebook,
    blocks: Vec<DecoderBlock>,
    output_proj: CausalConv1d,
    pub args: AudioTokenizerArgs,
    /// Frames-per-codec-frame, baked from `decoder_convs_strides`.
    pub samples_per_frame: usize,
}

impl VoxtralTTSAudioTokenizer {
    pub fn load(
        vb: VarBuilder,
        args: AudioTokenizerArgs,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let quantizer = MistralAudioCodebook::load(vb.pp("quantizer"), &args, device, dtype)?;

        let decoder_kernels = args.decoder_convs_kernels()?;
        let decoder_strides = args.decoder_convs_strides()?;
        let decoder_lens = args.decoder_transformer_lengths()?;

        // The decoder block sequence (reading upstream `__init__`):
        //   1. CausalConv1d(latent_dim → dim, kernel=k0, stride=s0)
        //      where latent_dim = semantic_dim + acoustic_dim = 292.
        //   2. for idx, n_layers in enumerate(decoder_transformer_lengths):
        //        Transformer at the current sliding_window
        //        if not last and (kernel[idx+1] != 1 or stride[idx+1] != 1):
        //           CausalConvTranspose1d(dim, dim, kernel[idx+1], stride[idx+1])
        //           sliding_window *= 2 if stride > 1 and half_attn_window flag
        //
        // For Voxtral-4B-TTS-2603 with kernels=[3,4,4,4] strides=[1,2,2,2]:
        //   blocks = [
        //     CausalConv1d(292→1024, k=3, s=1, pad=replicate),
        //     Transformer(2 layers, window=16),
        //     CausalConvTranspose1d(1024→1024, k=4, s=2),
        //     Transformer(2 layers, window=32),
        //     CausalConvTranspose1d(1024→1024, k=4, s=2),
        //     Transformer(2 layers, window=64),
        //     CausalConvTranspose1d(1024→1024, k=4, s=2),
        //     Transformer(2 layers, window=128),
        //   ]

        let latent_dim = args.semantic_dim + args.acoustic_dim;
        let mut blocks: Vec<DecoderBlock> = Vec::new();

        // First conv (block index 0 in the safetensors ordering).
        let input_conv = CausalConv1d::load(
            vb.pp("decoder_blocks.0"),
            latent_dim,
            args.dim,
            decoder_kernels[0],
            decoder_strides[0],
            PadMode::Replicate,
        )?;
        blocks.push(DecoderBlock::InputConv(input_conv));

        // Initial sliding window. The upstream may double it once on
        // the first block if `stride[0] > 1`, but Voxtral has stride[0]=1.
        let mut cur_window = args.attn_sliding_window_size;
        if args.half_attn_window_upon_downsampling && decoder_strides[0] > 1 {
            cur_window *= 2;
        }

        let mut next_block_idx = 1usize;
        for (idx, n_layers) in decoder_lens.iter().enumerate() {
            // Transformer at safetensors index `next_block_idx`.
            let xfmr = CodecTransformer::load(
                vb.pp(&format!("decoder_blocks.{next_block_idx}")),
                &args,
                *n_layers,
                cur_window,
                device,
                dtype,
            )?;
            blocks.push(DecoderBlock::Transformer(xfmr));
            next_block_idx += 1;

            let is_last = idx + 1 == decoder_lens.len();
            if !is_last {
                let next_kernel = decoder_kernels[idx + 1];
                let next_stride = decoder_strides[idx + 1];
                if next_kernel != 1 || next_stride != 1 {
                    let upsample = CausalConvTranspose1d::load(
                        vb.pp(&format!("decoder_blocks.{next_block_idx}")),
                        args.dim,
                        args.dim,
                        next_kernel,
                        next_stride,
                    )?;
                    blocks.push(DecoderBlock::Upsample(upsample));
                    next_block_idx += 1;
                    if args.half_attn_window_upon_downsampling && next_stride > 1 {
                        cur_window *= 2;
                    }
                }
            }
        }

        let output_proj = CausalConv1d::load(
            vb.pp("output_proj"),
            args.dim,
            args.pretransform_patch_size,
            args.patch_proj_kernel_size,
            1, // stride
            PadMode::Reflect,
        )?;

        let samples_per_frame = args.samples_per_frame()?;

        Ok(Self {
            quantizer,
            blocks,
            output_proj,
            args,
            samples_per_frame,
        })
    }

    /// Decode `codes [B, K=37, T]` (u32) → `[B, C=1, T_pcm]` (float).
    pub fn decode(&self, codes: &Tensor, dtype: DType) -> Result<Tensor> {
        // 1. Codes → embedding stream `[B, latent_dim, T]`.
        let emb = self.quantizer.decode(codes, dtype)?;

        // 2. Rearrange to `[B, T, D]` for transformer-friendly layout.
        let mut h = emb.transpose(1, 2)?.contiguous()?;

        // 3. Walk the alternating block sequence.
        for block in &self.blocks {
            h = match block {
                DecoderBlock::InputConv(c) => {
                    let h_dt = h.transpose(1, 2)?.contiguous()?;
                    let out = c.forward(&h_dt)?;
                    out.transpose(1, 2)?.contiguous()?
                }
                DecoderBlock::Upsample(c) => {
                    let h_dt = h.transpose(1, 2)?.contiguous()?;
                    let out = c.forward(&h_dt)?;
                    out.transpose(1, 2)?.contiguous()?
                }
                DecoderBlock::Transformer(t) => t.forward(&h)?,
            };
        }

        // 4. Back to `[B, D, T]` for the final output projection.
        let h = h.transpose(1, 2)?.contiguous()?;
        let h = self.output_proj.forward(&h)?; // [B, patch_size, T']

        // 5. Patch unfold: `[B, P, T'] → [B, 1, T'*P]`.
        // Equivalent to einops `b (c h) t -> b c (t h)` with c=1, h=P.
        let (b, p, t_out) = h.dims3()?;
        let channels = self.args.channels;
        if p % channels != 0 {
            return Err(anyhow::anyhow!(
                "output_proj produced {p} channels, expected multiple of {channels}"
            ));
        }
        let h_per_c = p / channels;
        // [B, P=c*h, T'] → [B, T', c, h] → [B, c, T', h] → [B, c, T'*h]
        let h = h.transpose(1, 2)?.contiguous()?; // [B, T', P]
        let h = h.reshape((b, t_out, channels, h_per_c))?; // [B, T', c, h]
        let h = h.transpose(1, 2)?.contiguous()?; // [B, c, T', h]
        let out = h.reshape((b, channels, t_out * h_per_c))?;
        Ok(out)
    }

    /// Diagnostic variant of [`Self::decode`] that captures the boundary
    /// tensors named exactly as in the
    /// `apps/burn-server/test_data/tts_golden/codec_*.safetensors`
    /// fixtures, for per-block regression testing:
    ///
    /// - `quantizer_emb` `[B, 292, T]`
    /// - `decoder_block_NN_out` per block `0..n_blocks`. Layout matches
    ///   the upstream forward-hook capture: BDT for conv-style blocks,
    ///   BTD for transformer blocks.
    /// - `output_proj_pre_rearrange` `[B, 240, T_after_upsamples]`.
    ///
    /// Returns `(intermediates_map, final_pcm)`.
    pub fn decode_with_intermediates(
        &self,
        codes: &Tensor,
        dtype: DType,
    ) -> Result<(BTreeMap<String, Tensor>, Tensor)> {
        let mut store: BTreeMap<String, Tensor> = BTreeMap::new();

        let emb = self.quantizer.decode(codes, dtype)?;
        store.insert("quantizer_emb".to_string(), emb.clone());

        let mut h = emb.transpose(1, 2)?.contiguous()?; // BTD
        for (idx, block) in self.blocks.iter().enumerate() {
            let key = format!("decoder_block_{idx:02}_out");
            h = match block {
                DecoderBlock::InputConv(c) => {
                    let h_dt = h.transpose(1, 2)?.contiguous()?;
                    let out = c.forward(&h_dt)?; // BDT
                    store.insert(key, out.clone());
                    out.transpose(1, 2)?.contiguous()?
                }
                DecoderBlock::Upsample(c) => {
                    let h_dt = h.transpose(1, 2)?.contiguous()?;
                    let out = c.forward(&h_dt)?; // BDT
                    store.insert(key, out.clone());
                    out.transpose(1, 2)?.contiguous()?
                }
                DecoderBlock::Transformer(t) => {
                    let out = t.forward(&h)?; // BTD
                    store.insert(key, out.clone());
                    out
                }
            };
        }

        let h_dt = h.transpose(1, 2)?.contiguous()?; // BDT for output_proj
        let proj = self.output_proj.forward(&h_dt)?; // [B, 240, T']
        store.insert("output_proj_pre_rearrange".to_string(), proj.clone());

        let (b, p, t_out) = proj.dims3()?;
        let channels = self.args.channels;
        let h_per_c = p / channels;
        let h = proj.transpose(1, 2)?.contiguous()?;
        let h = h.reshape((b, t_out, channels, h_per_c))?;
        let h = h.transpose(1, 2)?.contiguous()?;
        let pcm = h.reshape((b, channels, t_out * h_per_c))?;
        Ok((store, pcm))
    }
}
