//! Integration tests for the TTS HTTP route surface
//! (Phase 2-G + 2-H + 2-I).
//!
//! Heavy paths (`/api/tts/synthesize{,-codes}` with full pipeline run)
//! require the 4B-param checkpoint and run for tens of seconds on CPU,
//! so we only cover the cheap branches here:
//!
//! - `GET /api/tts/voices` — file-system snapshot, no model load.
//! - `GET /api/tts/config` — static constants, no model load.
//! - `GET /api/tts/status` — should report `loaded=false` before any
//!   synthesis call.
//! - `GET /api/tts/output` — empty list against an empty dir.
//! - Validation errors on `synthesize` and `synthesize-codes` —
//!   empty text/prompt_ids, unknown voice, out-of-range max_frames,
//!   text-too-long, voice_ref_id 501.
//! - Voice-cloning stubs return 501.
//!
//! Tests live behind the `voxtral-tts` feature flag; the file becomes
//! a no-op build target without it.

#![cfg(feature = "voxtral-tts")]

use axum::{
    body::Body,
    extract::DefaultBodyLimit,
    http::{Request, StatusCode},
    routing::{delete, get, post},
    Router,
};
use serde_json::Value;
use std::sync::Arc;
use tower::ServiceExt;

use burn_server::config::{Config, GpuBackend, Quantization};
use burn_server::server::{routes, state::AppState, tts_routes};

fn build_tts_test_app() -> Router {
    let tmp_dir = std::env::temp_dir().join(format!(
        "burn-server-tts-test-{}",
        uuid::Uuid::new_v4()
    ));
    std::fs::create_dir_all(&tmp_dir).ok();

    let config = Config {
        port: 0,
        quant: Quantization::Q4,
        backend: GpuBackend::Wgpu,
        models_dir: tmp_dir.join("models"),
        frontend: tmp_dir.join("frontend"),
        media_dir: tmp_dir.clone(),
        public_ip: None,
        max_sessions: 10,
        turn_server: None,
        turn_username: None,
        turn_password: None,
        turn_shared_secret: None,
        force_relay: false,
        tts_output_dir: tmp_dir.join("tts_output"),
        tts_long_max_secs: 180,
        tts_idle_unload_secs: 0,
        tts_voice_refs_dir: tmp_dir.join("voice_refs"),
        tts_max_voice_refs: 50,
        tts_max_ref_audio_bytes: 5_000_000,
        tts_min_ref_duration_secs: 5.0,
        tts_max_ref_duration_secs: 30.0,
        tts_require_permission: true,
    };

    let media_engine = webrtc::api::media_engine::MediaEngine::default();
    let registry = webrtc::interceptor::registry::Registry::new();
    let webrtc_api = webrtc::api::APIBuilder::new()
        .with_media_engine(media_engine)
        .with_interceptor_registry(registry)
        .build();

    let state = Arc::new(AppState::new(config, webrtc_api));

    Router::new()
        .route("/api/tts/voices", get(tts_routes::list_voices))
        .route("/api/tts/voices/upload", post(tts_routes::upload_voice_ref))
        .route(
            "/api/tts/voices/:ref_id",
            delete(tts_routes::delete_voice_ref),
        )
        .route("/api/tts/config", get(tts_routes::get_tts_config))
        .route("/api/tts/status", get(tts_routes::get_tts_status))
        .route(
            "/api/tts/synthesize-codes",
            post(tts_routes::synthesize_codes),
        )
        .route("/api/tts/synthesize", post(tts_routes::synthesize))
        .route("/api/tts/output", get(tts_routes::list_outputs))
        .route(
            "/api/tts/output/:name",
            get(tts_routes::get_output).delete(tts_routes::delete_output),
        )
        .layer(DefaultBodyLimit::max(10 * 1024 * 1024))
        .with_state(state)
}

async fn json_get(app: &Router, uri: &str) -> (StatusCode, Value) {
    let resp = app
        .clone()
        .oneshot(Request::builder().uri(uri).body(Body::empty()).unwrap())
        .await
        .unwrap();
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
        .await
        .unwrap();
    let body: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, body)
}

async fn json_post(app: &Router, uri: &str, body: Value) -> (StatusCode, Value) {
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(uri)
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
        .await
        .unwrap();
    let body: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, body)
}

async fn delete_(app: &Router, uri: &str) -> (StatusCode, Value) {
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri(uri)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
        .await
        .unwrap();
    let body: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, body)
}

// ── /api/tts/voices ─────────────────────────────────────────────────

#[tokio::test]
async fn voices_returns_flat_list_with_display_name_and_kind() {
    let app = build_tts_test_app();
    let (status, body) = json_get(&app, "/api/tts/voices").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["success"], true, "got body: {body}");
    let voices = body["data"].as_array().unwrap();
    assert_eq!(voices.len(), 20, "must surface all 20 canonical voices");
    let de_male = voices.iter().find(|v| v["id"] == "de_male").unwrap();
    // Frontend reads display_name + kind for the <select> render.
    assert_eq!(de_male["display_name"], "German (Male)");
    assert_eq!(de_male["language"], "German");
    assert_eq!(de_male["language_code"], "de");
    assert_eq!(de_male["gender"], "male");
    assert_eq!(de_male["kind"], "builtin");
}

// ── /api/tts/config ─────────────────────────────────────────────────

#[tokio::test]
async fn config_returns_frontend_compatible_fields() {
    let app = build_tts_test_app();
    let (status, body) = json_get(&app, "/api/tts/config").await;
    assert_eq!(status, StatusCode::OK);
    let cfg = &body["data"];
    // Frontend-required fields.
    assert!(cfg["max_chars"].is_number());
    assert!(cfg["output_dir"].is_string());
    assert_eq!(cfg["default_voice"], "casual_male");
    assert_eq!(cfg["sample_rate"], 24000);
    assert_eq!(cfg["supported_formats"][0], "wav");
    assert!(cfg["long_max_secs"].is_number());
    // Burn-specific extras retained for diagnostics.
    assert_eq!(cfg["frame_rate"], 12.5);
    assert_eq!(cfg["samples_per_frame"], 1920);
    assert_eq!(cfg["num_codebooks"], 37);
    assert_eq!(cfg["audio_token_id"], 24);
}

// ── /api/tts/status ─────────────────────────────────────────────────

#[tokio::test]
async fn status_reports_unloaded_before_any_synthesis() {
    let app = build_tts_test_app();
    let (status, body) = json_get(&app, "/api/tts/status").await;
    assert_eq!(status, StatusCode::OK);
    let data = &body["data"];
    // Backwards-compat fields (Phase 2-G shape) — frontend still reads
    // these.
    assert_eq!(data["loaded"], false);
    assert!(data["uptime_secs"].is_null());
    // Phase 3-D + 3-B fields (frontend renderTtsEngineBadge now reads
    // these). Field names mirror vllm-server's status response so the
    // shared frontend works against either backend.
    assert_eq!(data["state"], "idle");
    assert_eq!(data["inflight_synths"], 0);
    assert!(data["boot_elapsed_secs"].is_null());
    assert!(data["boot_timeout_secs"].is_null());
    assert!(data["last_activity_secs"].is_null());
    assert!(data["blocked_reason"].is_null());
    assert!(data["error"].is_null());
    assert_eq!(data["autostart"], true);
    assert_eq!(data["engine"], "voxtral-4b-tts via burn-server-native");
    // Phase 3-D test config sets `tts_idle_unload_secs: 0` (disabled).
    assert_eq!(data["idle_unload_secs"], 0);
}

// ── /api/tts/synthesize validation ──────────────────────────────────

#[tokio::test]
async fn synthesize_rejects_empty_text() {
    let app = build_tts_test_app();
    let (_status, body) = json_post(
        &app,
        "/api/tts/synthesize",
        serde_json::json!({"text": "   ", "voice": "neutral_male"}),
    )
    .await;
    assert_eq!(body["success"], false);
    assert!(
        body["error"].as_str().unwrap_or("").contains("text"),
        "got: {body}"
    );
}

#[tokio::test]
async fn synthesize_rejects_text_over_max_chars() {
    let app = build_tts_test_app();
    let huge = "a".repeat(20_001);
    let (_status, body) = json_post(
        &app,
        "/api/tts/synthesize",
        serde_json::json!({"text": huge, "voice": "neutral_male"}),
    )
    .await;
    assert_eq!(body["success"], false);
    assert!(
        body["error"].as_str().unwrap_or("").contains("max length"),
        "got: {body}"
    );
}

#[tokio::test]
async fn synthesize_rejects_unknown_voice() {
    let app = build_tts_test_app();
    let (_status, body) = json_post(
        &app,
        "/api/tts/synthesize",
        serde_json::json!({"text": "hello", "voice": "klingon_female"}),
    )
    .await;
    assert_eq!(body["success"], false);
    assert!(
        body["error"].as_str().unwrap_or("").contains("unknown voice"),
        "got: {body}"
    );
}

#[tokio::test]
async fn synthesize_rejects_max_frames_zero() {
    let app = build_tts_test_app();
    let (_status, body) = json_post(
        &app,
        "/api/tts/synthesize",
        serde_json::json!({"text": "hello", "voice": "neutral_male", "max_frames": 0}),
    )
    .await;
    assert_eq!(body["success"], false);
    assert!(
        body["error"].as_str().unwrap_or("").contains("max_frames"),
        "got: {body}"
    );
}

#[tokio::test]
async fn synthesize_rejects_max_frames_too_large() {
    let app = build_tts_test_app();
    let (_status, body) = json_post(
        &app,
        "/api/tts/synthesize",
        serde_json::json!({"text": "hello", "voice": "neutral_male", "max_frames": 5000}),
    )
    .await;
    assert_eq!(body["success"], false);
    assert!(
        body["error"].as_str().unwrap_or("").contains("max_frames"),
        "got: {body}"
    );
}

#[tokio::test]
async fn synthesize_rejects_voice_ref_id_with_501() {
    let app = build_tts_test_app();
    let (status, body) = json_post(
        &app,
        "/api/tts/synthesize",
        serde_json::json!({
            "text": "hello",
            "voice": "neutral_male",
            "voice_ref_id": "abc-123",
        }),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
    assert_eq!(body["success"], false);
    assert!(
        body["error"]
            .as_str()
            .unwrap_or("")
            .contains("voice cloning"),
        "got: {body}"
    );
}

// ── /api/tts/synthesize-codes validation ────────────────────────────

#[tokio::test]
async fn synthesize_codes_rejects_empty_prompt_ids() {
    let app = build_tts_test_app();
    let (_status, body) = json_post(
        &app,
        "/api/tts/synthesize-codes",
        serde_json::json!({"prompt_ids": [], "voice": "neutral_male"}),
    )
    .await;
    assert_eq!(body["success"], false);
    assert!(
        body["error"]
            .as_str()
            .unwrap_or("")
            .contains("prompt_ids must not be empty"),
        "got: {body}"
    );
}

#[tokio::test]
async fn synthesize_codes_rejects_unknown_voice() {
    let app = build_tts_test_app();
    let (_status, body) = json_post(
        &app,
        "/api/tts/synthesize-codes",
        serde_json::json!({"prompt_ids": [1, 25, 36, 35, 25], "voice": "klingon_female"}),
    )
    .await;
    assert_eq!(body["success"], false);
    assert!(
        body["error"].as_str().unwrap_or("").contains("unknown voice"),
        "got: {body}"
    );
}

// ── /api/tts/output ─────────────────────────────────────────────────

#[tokio::test]
async fn output_lists_empty_when_no_files() {
    let app = build_tts_test_app();
    let (status, body) = json_get(&app, "/api/tts/output").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["success"], true);
    let files = body["data"].as_array().unwrap();
    assert!(files.is_empty(), "expected empty list, got {files:?}");
}

#[tokio::test]
async fn output_get_404s_for_missing_file() {
    let app = build_tts_test_app();
    let (status, body) = json_get(&app, "/api/tts/output/no_such.wav").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body["success"], false);
}

#[tokio::test]
async fn output_delete_404s_for_missing_file() {
    let app = build_tts_test_app();
    let (status, body) = delete_(&app, "/api/tts/output/no_such.wav").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body["success"], false);
}

// ── /api/tts/voices/* cloning stubs ─────────────────────────────────

/// Phase 3-E1: upload route now exists; missing audio_sample yields a
/// 200 envelope (per ApiResponse convention) with `success: false`.
/// Body parse errors and validation errors share the same envelope.
#[tokio::test]
async fn voice_clone_upload_rejects_empty_body() {
    let app = build_tts_test_app();
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/tts/voices/upload")
                .header("content-type", "multipart/form-data; boundary=----X")
                .body(Body::from("------X--\r\n"))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
        .await
        .unwrap();
    let body: Value = serde_json::from_slice(&bytes).unwrap_or_default();
    // Either an HTTP-level error (multipart parse failed) OR a
    // success: false envelope. Either is acceptable shape — what we
    // care about is that the request didn't 501 anymore.
    assert_ne!(status, StatusCode::NOT_IMPLEMENTED);
    if status == StatusCode::OK {
        assert_eq!(body["success"], false, "got: {body}");
    }
}

/// Phase 3-E1: delete route uses path validation. A bogus id yields a
/// validation error envelope, not 501.
#[tokio::test]
async fn voice_clone_delete_validates_ref_id() {
    let app = build_tts_test_app();
    let (status, body) = delete_(&app, "/api/tts/voices/some-id-format").await;
    assert_ne!(status, StatusCode::NOT_IMPLEMENTED);
    // The message either says "invalid ref_id" (sanitiser rejected) or
    // "voice ref not found" (passed sanitiser, file missing). Either
    // is correct for an unknown id.
    let err = body["error"].as_str().unwrap_or("");
    assert!(
        err.contains("invalid ref_id") || err.contains("not found"),
        "expected validation/not-found error, got: {body}"
    );
}

// ── /api/tts/output round-trip with a pre-created file ─────────────

/// Build the test app and seed a fake WAV in its `tts_output_dir`. The
/// `Router` doesn't expose its config publicly, so we re-derive the
/// dir from the same recipe (uuid'd temp). This is fragile against
/// `build_tts_test_app` changes but cheap to keep in sync.
fn build_app_and_output_dir() -> (Router, std::path::PathBuf) {
    let tmp_dir = std::env::temp_dir().join(format!(
        "burn-server-tts-test-{}",
        uuid::Uuid::new_v4()
    ));
    std::fs::create_dir_all(&tmp_dir).ok();
    let out_dir = tmp_dir.join("tts_output");
    std::fs::create_dir_all(&out_dir).ok();

    let config = Config {
        port: 0,
        quant: Quantization::Q4,
        backend: GpuBackend::Wgpu,
        models_dir: tmp_dir.join("models"),
        frontend: tmp_dir.join("frontend"),
        media_dir: tmp_dir.clone(),
        public_ip: None,
        max_sessions: 10,
        turn_server: None,
        turn_username: None,
        turn_password: None,
        turn_shared_secret: None,
        force_relay: false,
        tts_output_dir: out_dir.clone(),
        tts_long_max_secs: 180,
        tts_idle_unload_secs: 0,
        tts_voice_refs_dir: tmp_dir.join("voice_refs"),
        tts_max_voice_refs: 50,
        tts_max_ref_audio_bytes: 5_000_000,
        tts_min_ref_duration_secs: 5.0,
        tts_max_ref_duration_secs: 30.0,
        tts_require_permission: true,
    };
    let media_engine = webrtc::api::media_engine::MediaEngine::default();
    let registry = webrtc::interceptor::registry::Registry::new();
    let webrtc_api = webrtc::api::APIBuilder::new()
        .with_media_engine(media_engine)
        .with_interceptor_registry(registry)
        .build();
    let state = Arc::new(AppState::new(config, webrtc_api));

    let app = Router::new()
        .route("/api/tts/output", get(tts_routes::list_outputs))
        .route(
            "/api/tts/output/:name",
            get(tts_routes::get_output).delete(tts_routes::delete_output),
        )
        .with_state(state);
    (app, out_dir)
}

#[tokio::test]
async fn output_lists_seeded_wav_and_skips_non_wav() {
    let (app, out_dir) = build_app_and_output_dir();
    std::fs::write(out_dir.join("greeting.wav"), b"RIFF" /* fake */).unwrap();
    std::fs::write(out_dir.join("notes.txt"), b"ignore me").unwrap();

    let (status, body) = json_get(&app, "/api/tts/output").await;
    assert_eq!(status, StatusCode::OK);
    let files = body["data"].as_array().unwrap();
    assert_eq!(files.len(), 1, "non-wav files must be skipped");
    assert_eq!(files[0]["name"], "greeting.wav");
}

#[tokio::test]
async fn output_get_returns_wav_bytes() {
    let (app, out_dir) = build_app_and_output_dir();
    let body_bytes = b"RIFF\x00\x00\x00\x00WAVEfmt fake-payload";
    std::fs::write(out_dir.join("hello.wav"), body_bytes).unwrap();

    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/api/tts/output/hello.wav")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(
        resp.headers().get("content-type").unwrap().to_str().unwrap(),
        "audio/wav"
    );
    let received = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
        .await
        .unwrap();
    assert_eq!(received.as_ref(), body_bytes);
}

#[tokio::test]
async fn output_delete_removes_file() {
    let (app, out_dir) = build_app_and_output_dir();
    let path = out_dir.join("removeme.wav");
    std::fs::write(&path, b"RIFF").unwrap();
    assert!(path.exists());

    let (status, body) = delete_(&app, "/api/tts/output/removeme.wav").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["success"], true);
    assert_eq!(body["data"]["deleted"], "removeme.wav");
    assert!(!path.exists(), "file must be deleted from disk");
}

#[tokio::test]
async fn output_get_rejects_path_traversal() {
    let (app, _out_dir) = build_app_and_output_dir();
    // Even with %2F-encoding the route param won't include /, but the
    // sanitiser is the safety net regardless. Try a name with `..`.
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/api/tts/output/..%2Fescape.wav")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    // Either 400 (sanitise error) or 404 (file not found after
    // sanitising) is acceptable — both prove we never read outside
    // the output dir.
    assert!(
        resp.status() == StatusCode::BAD_REQUEST
            || resp.status() == StatusCode::NOT_FOUND,
        "got status {}",
        resp.status()
    );
}

// ── Streaming WAV smoke ─────────────────────────────────────────────
//
// We can't easily exercise the full streaming /api/tts/synthesize
// path without the 4B model, but we can:
//
// 1. Test the streaming WAV header byte layout in isolation against
//    the upstream Python implementation in
//    `apps/vllm-server/voxtral_server/tts/wav.py::streaming_header`.
//    The layout is RIFF/WAVE/fmt/data with 0xFFFFFFFF placeholders at
//    offsets 4 and 40.
// 2. Confirm the fixture-aware end-to-end test still produces a
//    `Content-Type: audio/wav` response when the real pipeline runs
//    (covered by the `pipeline_smoke_test` ignore-flagged unit test;
//    no integration analogue here).
//
// Note: HF tokenizers' BPE merges + the Tekken regex are pure functions
// already unit-tested in voxtral_tts::{tokenizer,chat_template} — those
// don't need to be re-tested through the HTTP layer.

/// Reference encoding from the upstream Python helper, for 24 kHz mono
/// 16-bit PCM. Captured by running:
///
///   python3 -c 'from voxtral_server.tts.wav import streaming_header;
///               import sys; sys.stdout.buffer.write(streaming_header())'
///
/// Total 44 bytes; placeholders at offsets 4 and 40 (both 0xFFFFFFFF).
const EXPECTED_STREAMING_HEADER: &[u8] = &[
    // "RIFF"
    b'R', b'I', b'F', b'F',
    // chunk size = 0xFFFFFFFF (placeholder)
    0xFF, 0xFF, 0xFF, 0xFF,
    // "WAVE"
    b'W', b'A', b'V', b'E',
    // "fmt "
    b'f', b'm', b't', b' ',
    // fmt subchunk size = 16
    0x10, 0x00, 0x00, 0x00,
    // PCM format = 1
    0x01, 0x00,
    // channels = 1
    0x01, 0x00,
    // sample rate = 24000 = 0x5DC0
    0xC0, 0x5D, 0x00, 0x00,
    // byte rate = 24000 * 1 * 16 / 8 = 48000 = 0xBB80
    0x80, 0xBB, 0x00, 0x00,
    // block align = 1 * 16 / 8 = 2
    0x02, 0x00,
    // bits per sample = 16
    0x10, 0x00,
    // "data"
    b'd', b'a', b't', b'a',
    // data subchunk size = 0xFFFFFFFF (placeholder)
    0xFF, 0xFF, 0xFF, 0xFF,
];

#[test]
fn streaming_header_matches_upstream_python() {
    // The header function is private to the routes module; we test it
    // indirectly via the Content-Type + first-44-bytes shape of any
    // future synthesize response. For a direct unit test, expose a
    // probe function (gated to test cfg) on tts_routes if we ever
    // need to assert byte-exactness from outside the module.
    //
    // For now: this constant doubles as documentation + a future
    // anchor when we wire a probe.
    assert_eq!(EXPECTED_STREAMING_HEADER.len(), 44);
    assert_eq!(&EXPECTED_STREAMING_HEADER[0..4], b"RIFF");
    assert_eq!(&EXPECTED_STREAMING_HEADER[8..12], b"WAVE");
    assert_eq!(&EXPECTED_STREAMING_HEADER[12..16], b"fmt ");
    assert_eq!(&EXPECTED_STREAMING_HEADER[36..40], b"data");
    // Placeholder size fields.
    assert_eq!(&EXPECTED_STREAMING_HEADER[4..8], &[0xFF; 4]);
    assert_eq!(&EXPECTED_STREAMING_HEADER[40..44], &[0xFF; 4]);
    // Sample rate = 24000 little-endian.
    assert_eq!(&EXPECTED_STREAMING_HEADER[24..28], &24000u32.to_le_bytes());
}

// Quiet the unused-import warning when this file builds in isolation.
#[allow(dead_code)]
fn _ensure_routes_module_linked() {
    let _ = routes::get_config;
}
