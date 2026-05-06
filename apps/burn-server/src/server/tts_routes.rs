//! TTS HTTP routes for burn-server. Phase 2-G + 2-H.
//!
//! Exposes:
//! - `GET  /api/tts/voices`: list the 20 shipped voices (filename-derived).
//! - `GET  /api/tts/config`: return Voxtral-4B-TTS knobs (frame rate, etc.).
//! - `GET  /api/tts/status`: pipeline lifecycle snapshot for the badge.
//! - `POST /api/tts/synthesize`: text-input synthesis. Tokenizes via the
//!   embedded Tekken BPE + Voxtral-TTS chat template (Phase 2-H), runs
//!   the AR LLM ↔ FMA ↔ codec pipeline, returns 24 kHz mono WAV.
//! - `POST /api/tts/synthesize-codes`: pre-tokenized escape hatch. The
//!   client supplies prompt token-IDs + voice + a noise stream and gets
//!   24 kHz mono WAV back. Useful for upstream-fixture replays and tests.
//!
//! Pipeline lifecycle: load lazily on the first /api/tts/* call, then
//! hold for the process lifetime (no idle-unload yet — that's the
//! vllm-omni lifecycle's job, and burn-server's pipeline is small
//! enough on CPU that holding it is fine for now).

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use axum::{
    extract::State,
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use candle_core::{DType, Device, Tensor};
use serde::{Deserialize, Serialize};
use tokio::sync::OnceCell;

use crate::server::routes::{error_response, ApiResponse};
use crate::server::state::AppState;

// ── Lazy pipeline holder ────────────────────────────────────────────

/// Holds the loaded `TtsPipeline` and `TekkenTokenizer` once they've
/// been initialised. Stored inside an `Arc<OnceCell>` on the AppState
/// so all routes share a single instance.
pub struct TtsLifecycleState {
    pipeline: OnceCell<Arc<crate::inference::voxtral_tts::TtsPipeline>>,
    tokenizer: OnceCell<Arc<crate::inference::voxtral_tts::TekkenTokenizer>>,
    /// Track when the pipeline was first loaded for the /status
    /// endpoint and (eventually) idle-unload.
    loaded_at: tokio::sync::Mutex<Option<Instant>>,
}

impl TtsLifecycleState {
    pub fn new() -> Self {
        Self {
            pipeline: OnceCell::new(),
            tokenizer: OnceCell::new(),
            loaded_at: tokio::sync::Mutex::new(None),
        }
    }

    pub async fn get_or_load(
        &self,
        models_dir: &std::path::Path,
        device: &Device,
        dtype: DType,
        max_seq_len: usize,
    ) -> Result<Arc<crate::inference::voxtral_tts::TtsPipeline>, String> {
        let pipeline = self
            .pipeline
            .get_or_try_init(|| async {
                let ckpt = models_dir.join("tts").join("consolidated.safetensors");
                let params = models_dir.join("tts").join("params.json");
                let pipeline =
                    crate::inference::voxtral_tts::TtsPipeline::load(
                        &ckpt,
                        &params,
                        max_seq_len,
                        AUDIO_TOKEN_ID,
                        device,
                        dtype,
                    )
                    .map_err(|e| format!("loading TTS pipeline: {e}"))?;
                let mut at = self.loaded_at.lock().await;
                *at = Some(Instant::now());
                Ok::<_, String>(Arc::new(pipeline))
            })
            .await?;
        Ok(pipeline.clone())
    }

    pub async fn get_or_load_tokenizer(
        &self,
        models_dir: &std::path::Path,
    ) -> Result<Arc<crate::inference::voxtral_tts::TekkenTokenizer>, String> {
        let tk = self
            .tokenizer
            .get_or_try_init(|| async {
                // tekken.json ships in two locations on disk; prefer the
                // TTS-bundled copy because it carries the Voxtral-TTS
                // `audio.voice_num_audio_tokens` map. The shared
                // `tokenizer/tekken.json` is the same BPE but without
                // the audio config block — fine as a fallback.
                let primary = models_dir.join("tts").join("tekken.json");
                let fallback = models_dir.join("tokenizer").join("tekken.json");
                let path = if primary.exists() {
                    primary
                } else if fallback.exists() {
                    fallback
                } else {
                    return Err(format!(
                        "tekken.json not found at {} or {}",
                        primary.display(),
                        fallback.display()
                    ));
                };
                let tk = crate::inference::voxtral_tts::TekkenTokenizer::load(&path)
                    .map_err(|e| format!("loading TekkenTokenizer: {e}"))?;
                Ok::<_, String>(Arc::new(tk))
            })
            .await?;
        Ok(tk.clone())
    }

    pub async fn loaded_secs(&self) -> Option<f64> {
        self.loaded_at
            .lock()
            .await
            .map(|t| t.elapsed().as_secs_f64())
    }
}

impl Default for TtsLifecycleState {
    fn default() -> Self {
        Self::new()
    }
}

const AUDIO_TOKEN_ID: u32 = 24;

// ── Voice catalog ───────────────────────────────────────────────────

/// Canonical voice list (matches the upstream `voice` map in
/// `params.json` and the file-system layout under
/// `models/cache/tts/voice_embedding/`).
pub const CANONICAL_VOICES: &[(&str, &str)] = &[
    ("ar_male", "ar"),
    ("casual_female", "en"),
    ("casual_male", "en"),
    ("cheerful_female", "en"),
    ("de_female", "de"),
    ("de_male", "de"),
    ("es_female", "es"),
    ("es_male", "es"),
    ("fr_female", "fr"),
    ("fr_male", "fr"),
    ("hi_female", "hi"),
    ("hi_male", "hi"),
    ("it_female", "it"),
    ("it_male", "it"),
    ("neutral_female", "en"),
    ("neutral_male", "en"),
    ("nl_female", "nl"),
    ("nl_male", "nl"),
    ("pt_female", "pt"),
    ("pt_male", "pt"),
];

#[derive(Serialize)]
pub struct VoiceEntry {
    pub id: String,
    pub language: String,
    pub available: bool,
}

#[derive(Serialize)]
pub struct VoicesResponse {
    pub voices: Vec<VoiceEntry>,
    pub voice_dir: String,
}

pub async fn list_voices(State(state): State<Arc<AppState>>) -> Response {
    let voice_dir = state.config.models_dir.join("tts").join("voice_embedding");
    let voices: Vec<VoiceEntry> = CANONICAL_VOICES
        .iter()
        .map(|(id, lang)| VoiceEntry {
            id: (*id).to_string(),
            language: (*lang).to_string(),
            available: voice_dir.join(format!("{id}.safetensors")).exists(),
        })
        .collect();
    let resp = VoicesResponse {
        voices,
        voice_dir: voice_dir.display().to_string(),
    };
    ApiResponse::ok(resp).into_response()
}

// ── Config ──────────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct TtsConfig {
    pub sampling_rate: u32,
    pub frame_rate: f32,
    pub samples_per_frame: u32,
    pub num_codebooks: u32,
    pub audio_token_id: u32,
}

pub async fn get_tts_config(State(_state): State<Arc<AppState>>) -> Response {
    // Constants from Voxtral-4B-TTS-2603. Static — no need to load
    // the pipeline.
    let cfg = TtsConfig {
        sampling_rate: 24000,
        frame_rate: 12.5,
        samples_per_frame: 1920,
        num_codebooks: 37,
        audio_token_id: AUDIO_TOKEN_ID,
    };
    ApiResponse::ok(cfg).into_response()
}

// ── Status ──────────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct TtsStatus {
    pub loaded: bool,
    pub uptime_secs: Option<f64>,
}

pub async fn get_tts_status(State(state): State<Arc<AppState>>) -> Response {
    let lifecycle = state.tts.clone();
    let loaded = lifecycle.pipeline.initialized();
    let uptime = lifecycle.loaded_secs().await;
    ApiResponse::ok(TtsStatus {
        loaded,
        uptime_secs: uptime,
    })
    .into_response()
}

// ── Synthesize (pre-tokenized) ──────────────────────────────────────

#[derive(Deserialize)]
pub struct SynthesizeCodesRequest {
    /// Prompt token IDs as produced by an upstream tokenizer +
    /// chat-template formatter. burn-server doesn't include a Mistral
    /// tokenizer yet, so callers tokenize themselves.
    pub prompt_ids: Vec<u32>,
    /// One of [`CANONICAL_VOICES`].
    pub voice: String,
    /// Maximum AR generation steps (frames). Each frame ≈ 80 ms.
    #[serde(default = "default_max_frames")]
    pub max_frames: u32,
    /// Optional RNG seed for the FMA noise stream. Default: 42.
    #[serde(default = "default_seed")]
    pub seed: u64,
}

fn default_max_frames() -> u32 {
    250
}
fn default_seed() -> u64 {
    42
}

#[derive(Serialize)]
pub struct SynthesizeCodesMeta {
    pub frames_generated: u32,
    pub samples: u32,
    pub elapsed_secs: f64,
    pub voice: String,
    pub seed: u64,
}

/// Produce a deterministic Vec<f32> noise stream for the FMA. We use a
/// linear-congruential generator (not torch.randn) — outputs are
/// reproducible across runs but won't match upstream's torch.randn
/// for the same seed. Bit-exact upstream-match needs replaying the
/// captured noise; that's a follow-up.
fn deterministic_noise(seed: u64, n_frames: usize, n_dim: usize) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
    let mut out = Vec::with_capacity(n_frames * n_dim);
    for _ in 0..(n_frames * n_dim) {
        // xorshift64
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        // map u64 → roughly N(0, 1) via Box-Muller on uniform pair (cheap approx).
        let u = (state as f64) / (u64::MAX as f64); // [0, 1)
        let u = u.max(1e-9);
        // Single-sample standard-normal approx via the inverse-erf-of-uniform shortcut:
        let v = (2.0 * u - 1.0).clamp(-0.999_999, 0.999_999);
        let n = std::f32::consts::SQRT_2 * inv_erf(v as f32);
        out.push(n);
    }
    out
}

/// Crude inverse-error-function (Winitzki approximation, max abs err
/// ≈ 4e-3). Sufficient for our noise stream — the FMA isn't sensitive
/// to noise distribution beyond "roughly Gaussian".
fn inv_erf(x: f32) -> f32 {
    let a = 0.147f32;
    let ln = (1.0 - x * x).max(1e-9).ln();
    let t = 2.0 / (std::f32::consts::PI * a) + ln / 2.0;
    let inner = (t * t - ln / a).max(0.0).sqrt() - t;
    inner.sqrt() * x.signum()
}

/// Outcome of a successful pipeline run — WAV bytes plus a meta blob
/// suitable for the `X-TTS-Meta` response header.
struct SynthesisResult {
    wav: Vec<u8>,
    meta: serde_json::Value,
}

/// Shared core for the two TTS routes: validate inputs, lazy-load the
/// pipeline, load the voice, build the prompt tensor, generate the
/// noise stream, run the AR LLM ↔ FMA ↔ codec pipeline, and produce
/// 24 kHz mono 16-bit WAV bytes.
///
/// Returns Err with a human-readable message on any failure; callers
/// translate that into an `error_response` HTTP body.
async fn synthesize_to_wav(
    state: &Arc<AppState>,
    prompt_ids_vec: Vec<u32>,
    voice_id: &str,
    max_frames: u32,
    seed: u64,
) -> Result<SynthesisResult, String> {
    let start = Instant::now();
    let device = Device::Cpu;
    let dtype = DType::F32;
    let max_seq_len = (prompt_ids_vec.len() + max_frames as usize + 8).max(2048);

    let pipeline = state
        .tts
        .get_or_load(&state.config.models_dir, &device, dtype, max_seq_len)
        .await
        .map_err(|e| format!("pipeline load failed: {e}"))?;

    let voice = crate::inference::voxtral_tts::VoiceEmbedding::load_from_dir(
        &state.config.models_dir.join("tts").join("voice_embedding"),
        voice_id,
        pipeline.ar_llm.args.dim,
        &device,
        dtype,
    )
    .map_err(|e| format!("voice load failed: {e}"))?;

    let s = prompt_ids_vec.len();
    let prompt_ids = Tensor::from_vec(prompt_ids_vec, (1, s), &device)
        .map_err(|e| format!("prompt tensor: {e}"))?;

    let n_acoustic = pipeline.fma_args.n_acoustic_codebook;
    let noise_v = deterministic_noise(seed, max_frames as usize, n_acoustic);
    let noise = Tensor::from_vec(noise_v, (max_frames as usize, n_acoustic), &device)
        .map_err(|e| format!("noise tensor: {e}"))?;

    let pipeline_for_blocking = pipeline.clone();
    let max_frames_usize = max_frames as usize;
    let pcm = tokio::task::spawn_blocking(move || {
        pipeline_for_blocking.synthesize(&prompt_ids, &voice, &noise, max_frames_usize)
    })
    .await
    .map_err(|e| format!("blocking task: {e}"))?
    .map_err(|e| format!("synthesis: {e}"))?;

    let samples_f32: Vec<f32> = pcm
        .flatten_all()
        .and_then(|t| t.to_vec1::<f32>())
        .map_err(|e| format!("pcm extract: {e}"))?;
    let n_samples = samples_f32.len();
    let wav = pcm_f32_to_wav(&samples_f32, 24000);
    let elapsed = start.elapsed().as_secs_f64();

    let meta = serde_json::json!({
        "frames_generated": (n_samples / 1920) as u32,
        "samples": n_samples as u32,
        "elapsed_secs": elapsed,
        "voice": voice_id,
        "seed": seed,
    });
    Ok(SynthesisResult { wav, meta })
}

fn wav_response(result: SynthesisResult) -> Response {
    let meta_str = result.meta.to_string();
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "audio/wav")
        .header(
            header::CONTENT_DISPOSITION,
            "attachment; filename=\"synth.wav\"",
        )
        .header("X-TTS-Meta", meta_str)
        .body(axum::body::Body::from(result.wav))
        .unwrap()
        .into_response()
}

pub async fn synthesize_codes(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SynthesizeCodesRequest>,
) -> Response {
    if req.prompt_ids.is_empty() {
        return error_response("prompt_ids must not be empty").into_response();
    }
    if req.max_frames == 0 || req.max_frames > 1000 {
        return error_response("max_frames must be in [1, 1000]").into_response();
    }
    if !CANONICAL_VOICES.iter().any(|(id, _)| *id == req.voice) {
        return error_response(&format!("unknown voice {:?}", req.voice)).into_response();
    }
    match synthesize_to_wav(&state, req.prompt_ids, &req.voice, req.max_frames, req.seed).await {
        Ok(r) => wav_response(r),
        Err(e) => error_response(&e).into_response(),
    }
}

// ── Synthesize (text input — Phase 2-H) ─────────────────────────────

#[derive(Deserialize)]
pub struct SynthesizeRequest {
    /// UTF-8 text to synthesize. Tokenized via the embedded Tekken BPE
    /// + Voxtral-TTS chat template (Phase 2-H).
    pub text: String,
    /// One of [`CANONICAL_VOICES`].
    pub voice: String,
    #[serde(default = "default_max_frames")]
    pub max_frames: u32,
    #[serde(default = "default_seed")]
    pub seed: u64,
}

pub async fn synthesize(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SynthesizeRequest>,
) -> Response {
    if req.text.trim().is_empty() {
        return error_response("text must not be empty").into_response();
    }
    if req.max_frames == 0 || req.max_frames > 1000 {
        return error_response("max_frames must be in [1, 1000]").into_response();
    }
    if !CANONICAL_VOICES.iter().any(|(id, _)| *id == req.voice) {
        return error_response(&format!("unknown voice {:?}", req.voice)).into_response();
    }

    // Lazy-load tokenizer + voice. Voice is loaded twice (once here for
    // n_rows, once inside synthesize_to_wav for the actual tensor) —
    // safetensors mmap makes the second load nearly free, and keeping
    // the helper signature voice-id-only avoids a leaky parameter.
    let tk = match state
        .tts
        .get_or_load_tokenizer(&state.config.models_dir)
        .await
    {
        Ok(tk) => tk,
        Err(e) => return error_response(&format!("tokenizer load failed: {e}")).into_response(),
    };

    let voice_dir = state.config.models_dir.join("tts").join("voice_embedding");
    let voice_n_rows = match crate::inference::voxtral_tts::VoiceEmbedding::load_from_dir(
        &voice_dir,
        &req.voice,
        // Use any nonzero dim — VoiceEmbedding rejects shape mismatches
        // but we only need n_rows here, so pass the canonical 3072.
        3072,
        &Device::Cpu,
        DType::F32,
    ) {
        Ok(v) => match v.n_rows() {
            Ok(n) => n,
            Err(e) => return error_response(&format!("voice n_rows: {e}")).into_response(),
        },
        Err(e) => return error_response(&format!("voice load failed: {e}")).into_response(),
    };

    let prompt_ids = match crate::inference::voxtral_tts::build_voice_prompt(
        &tk,
        &req.text,
        voice_n_rows,
    ) {
        Ok(ids) => ids,
        Err(e) => {
            return error_response(&format!("prompt build failed: {e}")).into_response();
        }
    };

    match synthesize_to_wav(&state, prompt_ids, &req.voice, req.max_frames, req.seed).await {
        Ok(r) => wav_response(r),
        Err(e) => error_response(&e).into_response(),
    }
}

/// Encode a `f32 [-1, 1]` PCM stream as a 24 kHz mono 16-bit WAV.
fn pcm_f32_to_wav(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    let n_samples = samples.len() as u32;
    let bits_per_sample = 16u16;
    let byte_rate = sample_rate * (bits_per_sample as u32) / 8;
    let block_align = bits_per_sample / 8;
    let data_size = n_samples * (block_align as u32);
    let chunk_size = 36 + data_size;

    let mut out = Vec::with_capacity(44 + data_size as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&chunk_size.to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes()); // fmt chunk size
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM format
    out.extend_from_slice(&1u16.to_le_bytes()); // mono
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&bits_per_sample.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_size.to_le_bytes());
    for s in samples {
        let clipped = s.clamp(-1.0, 1.0);
        let v = (clipped * (i16::MAX as f32)) as i16;
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}
