//! TTS HTTP routes for burn-server. Phase 2-G + 2-H + 2-I.
//!
//! Contract: shape-compatible with `apps/vllm-server/voxtral_server/
//! api/tts_routes.py` so the shared frontend works against either
//! backend without branching. Cloning paths are accepted but stubbed
//! with 501 — burn-server doesn't yet implement the
//! `--task-type Base` upstream variant.
//!
//! Exposes:
//! - `GET  /api/tts/voices`: catalog (id, display_name, language,
//!   language_code, gender, kind="builtin"). Cloned voices not yet.
//! - `GET  /api/tts/config`: output_dir, max_chars, default_voice,
//!   sample_rate, supported_formats, long_max_secs, plus burn extras.
//! - `GET  /api/tts/status`: pipeline lifecycle snapshot for the badge.
//! - `POST /api/tts/synthesize`: text→speech. `save=true` writes to
//!   `tts_output_dir` and returns JSON metadata; `save=false` returns
//!   the WAV directly. `voice_ref_id` → 501.
//! - `POST /api/tts/synthesize-codes`: pre-tokenized escape hatch.
//! - `GET  /api/tts/output`: list saved WAVs.
//! - `GET  /api/tts/output/{name}`: download a saved WAV.
//! - `DELETE /api/tts/output/{name}`: remove a saved WAV.
//! - `POST /api/tts/voices/upload`: 501 (cloning not implemented).
//! - `DELETE /api/tts/voices/{ref_id}`: 501.
//!
//! Pipeline lifecycle: load lazily on the first /api/tts/* call, then
//! hold for the process lifetime (no idle-unload yet — that's the
//! vllm-omni lifecycle's job, and burn-server's pipeline is small
//! enough on CPU that holding it is fine for now).

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use axum::{
    extract::{Path as PathParam, State},
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

/// Canonical voice catalog. Mirrors `apps/vllm-server/voxtral_server/
/// tts/voices.py::VOICES` so the burn-server's `/api/tts/voices`
/// response is shape-compatible with the frontend's expectations.
struct CanonicalVoice {
    id: &'static str,
    display_name: &'static str,
    language: &'static str,
    language_code: &'static str,
    gender: &'static str,
}

const CANONICAL_VOICES: &[CanonicalVoice] = &[
    CanonicalVoice { id: "casual_female",   display_name: "Casual (Female)",     language: "English",    language_code: "en", gender: "female" },
    CanonicalVoice { id: "casual_male",     display_name: "Casual (Male)",       language: "English",    language_code: "en", gender: "male"   },
    CanonicalVoice { id: "cheerful_female", display_name: "Cheerful (Female)",   language: "English",    language_code: "en", gender: "female" },
    CanonicalVoice { id: "neutral_female",  display_name: "Neutral (Female)",    language: "English",    language_code: "en", gender: "female" },
    CanonicalVoice { id: "neutral_male",    display_name: "Neutral (Male)",      language: "English",    language_code: "en", gender: "male"   },
    CanonicalVoice { id: "de_female",       display_name: "German (Female)",     language: "German",     language_code: "de", gender: "female" },
    CanonicalVoice { id: "de_male",         display_name: "German (Male)",       language: "German",     language_code: "de", gender: "male"   },
    CanonicalVoice { id: "es_female",       display_name: "Spanish (Female)",    language: "Spanish",    language_code: "es", gender: "female" },
    CanonicalVoice { id: "es_male",         display_name: "Spanish (Male)",      language: "Spanish",    language_code: "es", gender: "male"   },
    CanonicalVoice { id: "fr_female",       display_name: "French (Female)",     language: "French",     language_code: "fr", gender: "female" },
    CanonicalVoice { id: "fr_male",         display_name: "French (Male)",       language: "French",     language_code: "fr", gender: "male"   },
    CanonicalVoice { id: "it_female",       display_name: "Italian (Female)",    language: "Italian",    language_code: "it", gender: "female" },
    CanonicalVoice { id: "it_male",         display_name: "Italian (Male)",      language: "Italian",    language_code: "it", gender: "male"   },
    CanonicalVoice { id: "nl_female",       display_name: "Dutch (Female)",      language: "Dutch",      language_code: "nl", gender: "female" },
    CanonicalVoice { id: "nl_male",         display_name: "Dutch (Male)",        language: "Dutch",      language_code: "nl", gender: "male"   },
    CanonicalVoice { id: "pt_female",       display_name: "Portuguese (Female)", language: "Portuguese", language_code: "pt", gender: "female" },
    CanonicalVoice { id: "pt_male",         display_name: "Portuguese (Male)",   language: "Portuguese", language_code: "pt", gender: "male"   },
    CanonicalVoice { id: "ar_male",         display_name: "Arabic (Male)",       language: "Arabic",     language_code: "ar", gender: "male"   },
    CanonicalVoice { id: "hi_female",       display_name: "Hindi (Female)",      language: "Hindi",      language_code: "hi", gender: "female" },
    CanonicalVoice { id: "hi_male",         display_name: "Hindi (Male)",        language: "Hindi",      language_code: "hi", gender: "male"   },
];

const DEFAULT_VOICE: &str = "casual_male";
const TTS_MAX_CHARS: usize = 20_000;
const TTS_LONG_MAX_SECS: u64 = 180;

fn is_known_voice(voice_id: &str) -> bool {
    CANONICAL_VOICES.iter().any(|v| v.id == voice_id)
}

#[derive(Serialize)]
pub struct VoiceEntry {
    pub id: String,
    pub display_name: String,
    pub language: String,
    pub language_code: String,
    pub gender: String,
    pub kind: &'static str,
}

pub async fn list_voices(State(_state): State<Arc<AppState>>) -> Response {
    // Frontend expects a flat list (the voice catalog is a separate
    // "data" array, not nested inside `{voices: [...]}`).
    let voices: Vec<VoiceEntry> = CANONICAL_VOICES
        .iter()
        .map(|v| VoiceEntry {
            id: v.id.to_string(),
            display_name: v.display_name.to_string(),
            language: v.language.to_string(),
            language_code: v.language_code.to_string(),
            gender: v.gender.to_string(),
            kind: "builtin",
        })
        .collect();
    // (No cloned voices yet on burn-server — Phase 2-J.)
    ApiResponse::ok(voices).into_response()
}

// ── Config ──────────────────────────────────────────────────────────

/// Frontend-facing TTS configuration. Field names match
/// `apps/vllm-server/voxtral_server/models.py::TtsConfig` so the
/// shared frontend reads burn-server and vllm-server identically. The
/// burn-specific Voxtral knobs (`frame_rate`, `samples_per_frame`,
/// `num_codebooks`, `audio_token_id`) are kept for tooling/diagnostics.
#[derive(Serialize)]
pub struct TtsConfig {
    pub output_dir: String,
    pub max_chars: usize,
    pub default_voice: String,
    pub sample_rate: u32,
    pub supported_formats: Vec<String>,
    pub long_max_secs: u64,
    // Burn-specific extras (no equivalent in vllm-server). Kept for
    // diagnostic/probing purposes; the frontend ignores them.
    pub frame_rate: f32,
    pub samples_per_frame: u32,
    pub num_codebooks: u32,
    pub audio_token_id: u32,
}

pub async fn get_tts_config(State(state): State<Arc<AppState>>) -> Response {
    let cfg = TtsConfig {
        output_dir: state.config.tts_output_dir.display().to_string(),
        max_chars: TTS_MAX_CHARS,
        default_voice: DEFAULT_VOICE.to_string(),
        sample_rate: 24000,
        supported_formats: vec!["wav".to_string()],
        long_max_secs: TTS_LONG_MAX_SECS,
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

/// Outcome of a successful pipeline run — raw 16-bit LE PCM payload
/// (no WAV header) plus a meta blob suitable for the `X-TTS-Meta`
/// response header. Callers wrap this in either:
/// - a streaming WAV response (header-with-placeholder + PCM chunks)
///   for the `save=false` path, or
/// - a complete WAV file (header-with-real-sizes + PCM) for the
///   `save=true` save-to-disk path.
struct SynthesisResult {
    pcm16: Vec<u8>,
    meta: serde_json::Value,
}

/// Shared core for the two TTS routes: validate inputs, lazy-load the
/// pipeline, load the voice, build the prompt tensor, generate the
/// noise stream, run the AR LLM ↔ FMA ↔ codec pipeline, and produce
/// raw 16-bit LE PCM bytes (24 kHz mono).
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
    let pcm16 = pcm_f32_to_pcm16_bytes(&samples_f32);
    let n_samples = samples_f32.len();
    let elapsed = start.elapsed().as_secs_f64();

    let meta = serde_json::json!({
        "frames_generated": (n_samples / 1920) as u32,
        "samples": n_samples as u32,
        "elapsed_secs": elapsed,
        "voice": voice_id,
        "seed": seed,
    });
    Ok(SynthesisResult { pcm16, meta })
}

/// Default chunk size for streaming the PCM body (~64 KB ≈ 700 ms of
/// 24 kHz mono 16-bit). Lower means more chunks/syscalls; higher means
/// less responsive to backpressure. 64 KB is a common sweet spot.
const STREAM_CHUNK_BYTES: usize = 64 * 1024;

/// Build a chunked `audio/wav` streaming response: 44-byte streaming
/// header (placeholder sizes) + PCM payload split into ~64 KB chunks.
/// Wire format matches `apps/vllm-server/voxtral_server/tts/wav.py::
/// streaming_header` exactly so the shared `frontend/js/modules/
/// tts-player.js` decodes either backend identically.
///
/// NOTE: The current burn-server pipeline produces the entire PCM
/// buffer in one shot before this helper runs, so chunked transfer
/// here is purely for wire-format parity — no real TTFB win until the
/// codec supports incremental decoding (Phase 2-K).
fn wav_response(result: SynthesisResult) -> Response {
    let meta_str = result.meta.to_string();
    let header = streaming_wav_header(24000, 1, 16);
    let pcm = result.pcm16;

    let mut chunks: Vec<Result<axum::body::Bytes, std::io::Error>> = Vec::new();
    chunks.push(Ok(axum::body::Bytes::from(header)));
    let mut offset = 0;
    while offset < pcm.len() {
        let end = (offset + STREAM_CHUNK_BYTES).min(pcm.len());
        chunks.push(Ok(axum::body::Bytes::copy_from_slice(&pcm[offset..end])));
        offset = end;
    }

    let stream = futures_util::stream::iter(chunks);
    let body = axum::body::Body::from_stream(stream);

    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "audio/wav")
        .header(
            header::CONTENT_DISPOSITION,
            "attachment; filename=\"synth.wav\"",
        )
        .header("X-TTS-Meta", meta_str)
        .body(body)
        .unwrap()
        .into_response()
}

/// 44-byte RIFF/WAVE header with `0xFFFFFFFF` placeholders for the
/// two unknown-size fields. Matches `vllm-server`'s wav.py exactly.
fn streaming_wav_header(sample_rate: u32, channels: u16, bits_per_sample: u16) -> Vec<u8> {
    assert!(bits_per_sample % 8 == 0, "bits_per_sample must be multiple of 8");
    let byte_rate = sample_rate * (channels as u32) * (bits_per_sample as u32) / 8;
    let block_align = channels * bits_per_sample / 8;
    const PLACEHOLDER: u32 = 0xFFFF_FFFF;

    let mut out = Vec::with_capacity(44);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&PLACEHOLDER.to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&bits_per_sample.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&PLACEHOLDER.to_le_bytes());
    out
}

/// Convert `f32 [-1, 1]` PCM to 16-bit signed-LE bytes (no header).
fn pcm_f32_to_pcm16_bytes(samples: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 2);
    for s in samples {
        let clipped = s.clamp(-1.0, 1.0);
        let v = (clipped * (i16::MAX as f32)) as i16;
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
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
    if !is_known_voice(&req.voice) {
        return error_response(&format!("unknown voice {:?}", req.voice)).into_response();
    }
    match synthesize_to_wav(&state, req.prompt_ids, &req.voice, req.max_frames, req.seed).await {
        Ok(r) => wav_response(r),
        Err(e) => error_response(&e).into_response(),
    }
}

// ── Synthesize (text input — Phase 2-H + 2-I) ───────────────────────
//
// Request shape mirrors `apps/vllm-server/voxtral_server/models.py::
// TtsSynthesizeRequest` so the shared frontend works against either
// backend without branching. The `max_frames`/`seed` knobs are
// burn-server-specific extras the frontend doesn't send; they default
// safely.

#[derive(Deserialize)]
pub struct SynthesizeRequest {
    /// UTF-8 text to synthesize. Tokenized via the embedded Tekken BPE
    /// + Voxtral-TTS chat template (Phase 2-H).
    pub text: String,
    /// One of [`CANONICAL_VOICES`]. Empty defaults to [`DEFAULT_VOICE`].
    #[serde(default)]
    pub voice: String,
    /// Voice cloning reference id — burn-server does not yet support
    /// the `--task-type Base` upstream variant. Returns 501 when set.
    #[serde(default)]
    pub voice_ref_id: Option<String>,
    /// Persist the result under `Config::tts_output_dir` and return
    /// `TtsSynthesizeResponse` JSON. Defaults to `true` for parity
    /// with vllm-server.
    #[serde(default = "default_save")]
    pub save: bool,
    /// Optional filename (must end in .wav, sanitized).
    #[serde(default)]
    pub save_filename: Option<String>,
    /// Allow replacing an existing file with the same name.
    #[serde(default)]
    pub overwrite: bool,
    /// Burn-specific safety bound on AR generation (frames). Default 250.
    #[serde(default = "default_max_frames")]
    pub max_frames: u32,
    /// Burn-specific RNG seed for the FMA noise stream. Default 42.
    #[serde(default = "default_seed")]
    pub seed: u64,
}

fn default_save() -> bool {
    true
}

#[derive(Serialize)]
pub struct TtsSynthesizeResponse {
    pub filename: String,
    pub path: String,
    pub bytes: usize,
    pub voice: String,
    pub sample_rate: u32,
    pub duration_secs: Option<f64>,
    pub elapsed_secs: f64,
}

pub async fn synthesize(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SynthesizeRequest>,
) -> Response {
    if req.text.trim().is_empty() {
        return error_response("text is empty").into_response();
    }
    if req.text.len() > TTS_MAX_CHARS {
        return error_response(&format!(
            "text exceeds max length ({} > {})",
            req.text.len(),
            TTS_MAX_CHARS
        ))
        .into_response();
    }
    if req.max_frames == 0 || req.max_frames > 1000 {
        return error_response("max_frames must be in [1, 1000]").into_response();
    }

    if req.voice_ref_id.is_some() {
        return (
            StatusCode::NOT_IMPLEMENTED,
            Json(crate::server::routes::ApiResponse::<()> {
                success: false,
                data: None,
                error: Some(
                    "voice cloning (voice_ref_id) is not supported on burn-server yet"
                        .to_string(),
                ),
            }),
        )
            .into_response();
    }
    let voice_id = if req.voice.is_empty() {
        DEFAULT_VOICE.to_string()
    } else {
        req.voice.clone()
    };
    if !is_known_voice(&voice_id) {
        return error_response(&format!("unknown voice {voice_id:?}")).into_response();
    }

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
        &voice_id,
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

    let synth = match synthesize_to_wav(&state, prompt_ids, &voice_id, req.max_frames, req.seed)
        .await
    {
        Ok(r) => r,
        Err(e) => return error_response(&e).into_response(),
    };

    if !req.save {
        return wav_response(synth);
    }

    // ── Save-to-disk branch ─────────────────────────────────────────
    let out_dir = &state.config.tts_output_dir;
    if let Err(e) = std::fs::create_dir_all(out_dir) {
        return error_response(&format!("could not create output dir: {e}"))
            .into_response();
    }
    let filename = match req.save_filename.as_deref() {
        Some(name) if !name.is_empty() => match sanitize_filename(name) {
            Ok(n) => n,
            Err(e) => return error_response(&e).into_response(),
        },
        _ => auto_filename(&voice_id),
    };
    let target = match safe_join(out_dir, &filename) {
        Ok(p) => p,
        Err(e) => return error_response(&e).into_response(),
    };
    if target.exists() && !req.overwrite {
        return error_response(&format!(
            "file already exists: {} (pass overwrite=true to replace)",
            filename
        ))
        .into_response();
    }
    // Save-to-disk wants a real WAV file (proper RIFF size fields, not
    // the streaming `0xFFFFFFFF` placeholders). Assemble from PCM
    // bytes here using the per-file header encoder.
    let wav_bytes = wrap_pcm16_in_wav_file(&synth.pcm16, 24000, 1, 16);
    let bytes_written = match std::fs::write(&target, &wav_bytes) {
        Ok(()) => wav_bytes.len(),
        Err(e) => return error_response(&format!("could not write file: {e}")).into_response(),
    };

    let elapsed_secs = synth
        .meta
        .get("elapsed_secs")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    let n_samples = synth
        .meta
        .get("samples")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let duration_secs = if n_samples > 0 {
        Some(n_samples as f64 / 24000.0)
    } else {
        None
    };

    ApiResponse::ok(TtsSynthesizeResponse {
        filename,
        path: target.display().to_string(),
        bytes: bytes_written,
        voice: voice_id,
        sample_rate: 24000,
        duration_secs,
        elapsed_secs,
    })
    .into_response()
}

fn sanitize_filename(name: &str) -> Result<String, String> {
    let basename = std::path::Path::new(name)
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or_else(|| format!("invalid filename: {name:?}"))?;
    let mut out = String::with_capacity(basename.len());
    for c in basename.chars() {
        if c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '.' {
            out.push(c);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        return Err("filename is empty after sanitization".to_string());
    }
    if !out.to_lowercase().ends_with(".wav") {
        out.push_str(".wav");
    }
    if out.len() > 128 {
        return Err(format!(
            "filename too long after sanitization ({} > 128)",
            out.len()
        ));
    }
    Ok(out)
}

fn auto_filename(voice: &str) -> String {
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("{voice}_{ts}.wav")
}

fn safe_join(dir: &std::path::Path, filename: &str) -> Result<PathBuf, String> {
    std::fs::create_dir_all(dir).ok();
    let dir_canon = dir
        .canonicalize()
        .map_err(|e| format!("resolving output dir: {e}"))?;
    let candidate = dir_canon.join(filename);
    let parent = candidate
        .parent()
        .ok_or_else(|| "join produced no parent".to_string())?;
    if parent != dir_canon {
        return Err("filename escapes output directory".to_string());
    }
    Ok(candidate)
}

/// Wrap pre-computed 16-bit LE PCM bytes in a complete (non-streaming)
/// WAV file with proper RIFF + data size fields. Used by the
/// save-to-disk path so the resulting `.wav` file plays correctly in
/// any downstream tool that validates header sizes.
fn wrap_pcm16_in_wav_file(
    pcm: &[u8],
    sample_rate: u32,
    channels: u16,
    bits_per_sample: u16,
) -> Vec<u8> {
    let byte_rate = sample_rate * (channels as u32) * (bits_per_sample as u32) / 8;
    let block_align = channels * bits_per_sample / 8;
    let data_size = pcm.len() as u32;
    let chunk_size = 36 + data_size;

    let mut out = Vec::with_capacity(44 + pcm.len());
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&chunk_size.to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM format
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&bits_per_sample.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_size.to_le_bytes());
    out.extend_from_slice(pcm);
    out
}

// ── Saved files (save=true outputs) ─────────────────────────────────

#[derive(Serialize)]
pub struct TtsOutputFile {
    pub name: String,
    pub bytes: u64,
    pub created_at: f64,
}

pub async fn list_outputs(State(state): State<Arc<AppState>>) -> Response {
    let dir = &state.config.tts_output_dir;
    let mut files: Vec<TtsOutputFile> = Vec::new();
    if dir.exists() {
        match std::fs::read_dir(dir) {
            Ok(rd) => {
                for entry in rd.flatten() {
                    let path = entry.path();
                    if !path.is_file() {
                        continue;
                    }
                    let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
                        continue;
                    };
                    if !name.to_lowercase().ends_with(".wav") {
                        continue;
                    }
                    let meta = match entry.metadata() {
                        Ok(m) => m,
                        Err(_) => continue,
                    };
                    let created_at = meta
                        .modified()
                        .ok()
                        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                        .map(|d| d.as_secs_f64())
                        .unwrap_or(0.0);
                    files.push(TtsOutputFile {
                        name: name.to_string(),
                        bytes: meta.len(),
                        created_at,
                    });
                }
            }
            Err(_) => {} // Empty list when the dir is unreadable.
        }
    }
    // Newest first (matches vllm-server behaviour).
    files.sort_by(|a, b| b.created_at.partial_cmp(&a.created_at).unwrap_or(std::cmp::Ordering::Equal));
    ApiResponse::ok(files).into_response()
}

pub async fn get_output(
    State(state): State<Arc<AppState>>,
    PathParam(name): PathParam<String>,
) -> Response {
    let sanitized = match sanitize_filename(&name) {
        Ok(n) => n,
        Err(e) => return error_response(&e).into_response(),
    };
    let target = match safe_join(&state.config.tts_output_dir, &sanitized) {
        Ok(p) => p,
        Err(e) => return error_response(&e).into_response(),
    };
    if !target.exists() {
        return (
            StatusCode::NOT_FOUND,
            Json(crate::server::routes::ApiResponse::<()> {
                success: false,
                data: None,
                error: Some(format!("file not found: {sanitized}")),
            }),
        )
            .into_response();
    }
    let bytes = match std::fs::read(&target) {
        Ok(b) => b,
        Err(e) => return error_response(&format!("could not read file: {e}")).into_response(),
    };
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "audio/wav")
        .header(
            header::CONTENT_DISPOSITION,
            format!("attachment; filename=\"{sanitized}\""),
        )
        .body(axum::body::Body::from(bytes))
        .unwrap()
        .into_response()
}

pub async fn delete_output(
    State(state): State<Arc<AppState>>,
    PathParam(name): PathParam<String>,
) -> Response {
    let sanitized = match sanitize_filename(&name) {
        Ok(n) => n,
        Err(e) => return error_response(&e).into_response(),
    };
    let target = match safe_join(&state.config.tts_output_dir, &sanitized) {
        Ok(p) => p,
        Err(e) => return error_response(&e).into_response(),
    };
    if !target.exists() {
        return (
            StatusCode::NOT_FOUND,
            Json(crate::server::routes::ApiResponse::<()> {
                success: false,
                data: None,
                error: Some(format!("file not found: {sanitized}")),
            }),
        )
            .into_response();
    }
    match std::fs::remove_file(&target) {
        Ok(()) => ApiResponse::ok(serde_json::json!({"deleted": sanitized})).into_response(),
        Err(e) => error_response(&format!("could not delete file: {e}")).into_response(),
    }
}

// ── Voice cloning stubs (501 until burn-server implements --task-type Base)

pub async fn upload_voice_ref_stub() -> Response {
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(crate::server::routes::ApiResponse::<()> {
            success: false,
            data: None,
            error: Some(
                "voice cloning is not supported on burn-server yet — use vllm-server for cloning"
                    .to_string(),
            ),
        }),
    )
        .into_response()
}

pub async fn delete_voice_ref_stub(PathParam(_ref_id): PathParam<String>) -> Response {
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(crate::server::routes::ApiResponse::<()> {
            success: false,
            data: None,
            error: Some(
                "voice cloning is not supported on burn-server yet — use vllm-server for cloning"
                    .to_string(),
            ),
        }),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Anchors the streaming-WAV header byte layout against the
    /// upstream Python helper in
    /// `apps/vllm-server/voxtral_server/tts/wav.py::streaming_header`.
    ///
    /// Re-derive the expected bytes with:
    ///
    /// ```
    /// cd apps/vllm-server && source .venv/bin/activate && python3 -c \
    ///   "from voxtral_server.tts.wav import streaming_header; \
    ///    import sys; sys.stdout.buffer.write(streaming_header())"
    /// ```
    #[test]
    fn streaming_wav_header_matches_upstream_python_output() {
        // 24 kHz mono 16-bit PCM (Voxtral-4B-TTS defaults).
        let got = streaming_wav_header(24000, 1, 16);
        assert_eq!(got.len(), 44);
        // Field-by-field assertion makes regression failures
        // pinpoint the offset rather than dumping the whole buffer.
        assert_eq!(&got[0..4], b"RIFF");
        assert_eq!(&got[4..8], &[0xFF; 4], "RIFF chunk-size placeholder");
        assert_eq!(&got[8..12], b"WAVE");
        assert_eq!(&got[12..16], b"fmt ");
        assert_eq!(&got[16..20], &16u32.to_le_bytes(), "fmt chunk size");
        assert_eq!(&got[20..22], &1u16.to_le_bytes(), "PCM format");
        assert_eq!(&got[22..24], &1u16.to_le_bytes(), "channels");
        assert_eq!(&got[24..28], &24000u32.to_le_bytes(), "sample rate");
        assert_eq!(&got[28..32], &48000u32.to_le_bytes(), "byte rate");
        assert_eq!(&got[32..34], &2u16.to_le_bytes(), "block align");
        assert_eq!(&got[34..36], &16u16.to_le_bytes(), "bits per sample");
        assert_eq!(&got[36..40], b"data");
        assert_eq!(&got[40..44], &[0xFF; 4], "data chunk-size placeholder");
    }

    #[test]
    fn pcm_f32_to_pcm16_clips_and_packs_le() {
        // Values at, beyond, and inside the [-1, 1] range.
        let samples = vec![0.0f32, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0];
        let bytes = pcm_f32_to_pcm16_bytes(&samples);
        assert_eq!(bytes.len(), samples.len() * 2);
        // 0.0 → 0
        assert_eq!(&bytes[0..2], &0i16.to_le_bytes());
        // 1.0 → 32767
        assert_eq!(&bytes[2..4], &i16::MAX.to_le_bytes());
        // -1.0 → -32767 (since i16::MAX is 32767 and we multiply, not -32768)
        assert_eq!(&bytes[4..6], &(-i16::MAX).to_le_bytes());
        // 2.0 clipped → 32767
        assert_eq!(&bytes[10..12], &i16::MAX.to_le_bytes());
        // -2.0 clipped → -32767
        assert_eq!(&bytes[12..14], &(-i16::MAX).to_le_bytes());
    }

    #[test]
    fn wrap_pcm16_in_wav_file_has_real_sizes() {
        // 100 samples × 2 bytes = 200 PCM bytes.
        let pcm = vec![0u8; 200];
        let wav = wrap_pcm16_in_wav_file(&pcm, 24000, 1, 16);
        assert_eq!(wav.len(), 244, "44-byte header + 200 PCM bytes");
        // RIFF chunk size = 36 + data_size = 36 + 200 = 236
        assert_eq!(&wav[4..8], &236u32.to_le_bytes(), "RIFF chunk size");
        // data chunk size = 200
        assert_eq!(&wav[40..44], &200u32.to_le_bytes(), "data chunk size");
        // Same field offsets as streaming header for the rest.
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(&wav[12..16], b"fmt ");
        assert_eq!(&wav[36..40], b"data");
    }
}
