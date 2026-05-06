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
        .route("/api/tts/voices/upload", post(tts_routes::upload_voice_ref_stub))
        .route(
            "/api/tts/voices/:ref_id",
            delete(tts_routes::delete_voice_ref_stub),
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
    assert_eq!(body["data"]["loaded"], false);
    assert!(body["data"]["uptime_secs"].is_null());
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

#[tokio::test]
async fn voice_clone_upload_returns_501() {
    let app = build_tts_test_app();
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/tts/voices/upload")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_IMPLEMENTED);
}

#[tokio::test]
async fn voice_clone_delete_returns_501() {
    let app = build_tts_test_app();
    let (status, body) = delete_(&app, "/api/tts/voices/some-uuid").await;
    assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
    assert!(
        body["error"].as_str().unwrap_or("").contains("voice cloning"),
        "got: {body}"
    );
}

// Quiet the unused-import warning when this file builds in isolation.
#[allow(dead_code)]
fn _ensure_routes_module_linked() {
    let _ = routes::get_config;
}
