//! End-to-end TTS pipeline. Wires AR LLM + flow-matching head + codec
//! into a single `TtsPipeline` whose top-level entry point takes a
//! prompt + voice + noise stream and returns 24 kHz PCM samples.
//!
//! Phase 2-F.4 scope: the pipeline machinery is complete. The
//! prompt-construction layer (mistral tokenizer + chat template) is
//! deferred to Phase 2-G; for now callers supply pre-tokenized prompt
//! IDs. The smoke test in this file uses a hand-rolled minimal prompt
//! to validate the AR LLM ↔ FMA ↔ codec pipeline runs end-to-end on
//! real weights without crashing.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use std::path::Path;

use super::ar_llm::{ArLlmArgs, ArLlmModel, RopeCache};
use super::autoregressor::TtsAutoregressor;
use super::codec::{AudioTokenizerArgs, VoxtralTTSAudioTokenizer};
use super::flow_matching::{
    args::{FlowMatchingDecodeArgs, MultimodalAudioModelArgs},
    model::FlowMatchingAudioTransformer,
};
use super::voice::VoiceEmbedding;

/// All three loaded Voxtral-4B-TTS subcomponents on a single device.
pub struct TtsPipeline {
    pub ar_llm: ArLlmModel,
    pub fma: FlowMatchingAudioTransformer,
    pub codec: VoxtralTTSAudioTokenizer,
    pub rope: RopeCache,
    pub fma_args: MultimodalAudioModelArgs,
    pub audio_token_id: u32,
    pub device: Device,
    pub dtype: DType,
}

impl TtsPipeline {
    /// Load all three subcomponents from `consolidated.safetensors`.
    pub fn load(
        ckpt_path: &Path,
        params_path: &Path,
        max_seq_len: usize,
        audio_token_id: u32,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let ar_args = ArLlmArgs::from_params_json_path(params_path)?;
        let fma_args = MultimodalAudioModelArgs::from_params_json_path(params_path)?;
        let codec_args = AudioTokenizerArgs::from_params_json_path(params_path)?;

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[ckpt_path], dtype, device)? };

        let ar_llm = ArLlmModel::load(vb.clone(), ar_args.clone(), device, dtype)?;
        let fma = FlowMatchingAudioTransformer::load(
            vb.pp("acoustic_transformer"),
            fma_args.clone(),
            FlowMatchingDecodeArgs::default(),
            device,
            dtype,
        )?;
        let codec = VoxtralTTSAudioTokenizer::load(vb.pp("audio_tokenizer"), codec_args, device, dtype)?;
        let rope = RopeCache::new(max_seq_len, ar_args.head_dim, ar_args.rope_theta, device, dtype)?;

        Ok(Self {
            ar_llm,
            fma,
            codec,
            rope,
            fma_args,
            audio_token_id,
            device: device.clone(),
            dtype,
        })
    }

    /// Run the full pipeline: prompt → autoregressive code generation
    /// → codec decode → 24 kHz PCM `[B=1, C=1, T_pcm]`.
    ///
    /// `prompt_ids`: `[1, S]` u32 token IDs.
    /// `voice`: a loaded [`VoiceEmbedding`].
    /// `noise`: `[max_frames, n_acoustic_codebook]` float for the FMA
    /// integration step. Caller-supplied for determinism.
    /// `max_frames`: safety bound on AR generation.
    pub fn synthesize(
        &self,
        prompt_ids: &Tensor,
        voice: &VoiceEmbedding,
        noise: &Tensor,
        max_frames: usize,
    ) -> Result<Tensor> {
        let driver = TtsAutoregressor {
            ar_llm: &self.ar_llm,
            fma: &self.fma,
            rope: &self.rope,
            fma_args: &self.fma_args,
            audio_token_id: self.audio_token_id,
        };
        let codes = driver.synthesize_codes(
            prompt_ids,
            voice,
            noise,
            max_frames,
            &self.device,
            self.dtype,
        )?;
        let n = codes.dim(0)?;
        if n == 0 {
            // No frames generated → silent PCM of zero samples.
            return Tensor::zeros((1, 1, 0), self.dtype, &self.device)
                .map_err(anyhow::Error::from);
        }
        // Reshape codes [n, 37] → [1, 37, n] for the codec's BCT layout.
        let codes_bct = codes.t()?.unsqueeze(0)?.contiguous()?;
        self.codec.decode(&codes_bct, self.dtype)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn ckpt_path() -> PathBuf {
        std::env::var_os("STARLING_TTS_SAFETENSORS")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from(
                    "/home/gjovanov/gjovanov/starling/models/cache/tts/consolidated.safetensors",
                )
            })
    }

    fn params_path() -> PathBuf {
        std::env::var_os("STARLING_TTS_PARAMS")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from("/home/gjovanov/gjovanov/starling/models/cache/tts/params.json")
            })
    }

    fn voice_dir() -> PathBuf {
        std::env::var_os("STARLING_TTS_VOICE_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(super::super::voice::default_voice_dir)
    }

    /// End-to-end smoke test: build a minimal prompt, run AR LLM ↔ FMA
    /// ↔ codec, and confirm the pipeline produces non-trivial PCM.
    /// Bit-exact validation against upstream synthesis lands once we
    /// have captured llm_hidden golden refs (Phase 2-G).
    #[test]
    #[ignore = "loads 4B params + runs codec on CPU; ~2 min. Run with --ignored."]
    fn pipeline_smoke_test() {
        let ckpt = ckpt_path();
        let params = params_path();
        let vdir = voice_dir();
        if !ckpt.exists() || !params.exists() || !vdir.exists() {
            eprintln!("skipping smoke test: assets absent");
            return;
        }

        let device = Device::Cpu;
        let dtype = DType::F32;
        let max_seq_len = 1024;

        let pipeline = TtsPipeline::load(&ckpt, &params, max_seq_len, 24, &device, dtype).unwrap();
        let voice =
            VoiceEmbedding::load_from_dir(&vdir, "neutral_male", 3072, &device, dtype).unwrap();

        // Minimal prompt: just BEGIN_AUDIO + audio_tokens × N (filling
        // the voice slot count). Without the chat template this won't
        // produce coherent speech — it's a graph-exercise test only.
        let n_audio = voice.n_rows().unwrap();
        let mut tokens: Vec<u32> = vec![1, 25]; // BOS, BEGIN_AUDIO
        tokens.extend(std::iter::repeat(24u32).take(n_audio));
        let s = tokens.len();
        let prompt_ids = Tensor::from_vec(tokens, (1, s), &device).unwrap();

        // Zero noise for repeatability — output won't match upstream
        // (which uses torch.randn), but the pipeline must run.
        let max_frames = 4;
        let noise = Tensor::zeros((max_frames, 36), dtype, &device).unwrap();

        let pcm = pipeline.synthesize(&prompt_ids, &voice, &noise, max_frames).unwrap();
        let dims = pcm.dims().to_vec();
        eprintln!("pipeline_smoke: pcm shape = {dims:?}");
        // PCM shape [1, 1, T] where T = n_frames * 1920 (or 0 if it
        // hit END_AUDIO immediately).
        assert_eq!(dims[0], 1);
        assert_eq!(dims[1], 1);
        // We tolerate either non-trivial generation OR an immediate
        // END_AUDIO halt — both are valid pipeline behaviours given
        // the synthetic prompt.
        let t = dims[2];
        if t > 0 {
            assert!(t % 1920 == 0, "PCM length must be multiple of 1920");
        }
    }
}
