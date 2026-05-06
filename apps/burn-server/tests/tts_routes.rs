//! Integration tests for the TTS HTTP route surface (Phase 2-G + 2-H).
//!
//! Heavy paths (`/api/tts/synthesize{,-codes}`) require the 4B-param
//! checkpoint and run for tens of seconds on CPU, so we only cover the
//! cheap branches here:
//!
//! - `GET /api/tts/voices` — file-system snapshot, no model load.
//! - `GET /api/tts/config` — static constants, no model load.
//! - `GET /api/tts/status` — should report `loaded=false` before any
//!   synthesis call.
//! - Input validation on `synthesize` and `synthesize-codes` — empty
//!   text/prompt_ids, unknown voice, out-of-range max_frames.
//!
//! All tests live behind the `voxtral-tts` feature flag; the file
//! becomes a no-op build target without it.

#![cfg(feature = "voxtral-tts")]

use axum::{
    body::Body,
    extract::DefaultBodyLimit,
    http::{Request, StatusCode},
    routing::{get, post},
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
        .route("/api/tts/config", get(tts_routes::get_tts_config))
        .route("/api/tts/status", get(tts_routes::get_tts_status))
        .route(
            "/api/tts/synthesize-codes",
            post(tts_routes::synthesize_codes),
        )
        .route("/api/tts/synthesize", post(tts_routes::synthesize))
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
    let body: Value = serde_json::from_slice(&bytes).unwrap();
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
    let body: Value = serde_json::from_slice(&bytes).unwrap();
    (status, body)
}

#[tokio::test]
async fn lists_voices_with_availability_flag() {
    let app = build_tts_test_app();
    let (status, body) = json_get(&app, "/api/tts/voices").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["success"], true, "got body: {body}");
    let voices = body["data"]["voices"].as_array().unwrap();
    assert_eq!(voices.len(), 20, "must surface all 20 canonical voices");
    let de_male = voices.iter().find(|v| v["id"] == "de_male").unwrap();
    assert_eq!(de_male["language"], "de");
    // The temp models_dir has no voice files, so available must be false.
    assert_eq!(
        de_male["available"], false,
        "expected false in scratch test fixture"
    );
}

#[tokio::test]
async fn returns_static_config() {
    let app = build_tts_test_app();
    let (status, body) = json_get(&app, "/api/tts/config").await;
    assert_eq!(status, StatusCode::OK);
    let cfg = &body["data"];
    assert_eq!(cfg["sampling_rate"], 24000);
    assert_eq!(cfg["frame_rate"], 12.5);
    assert_eq!(cfg["samples_per_frame"], 1920);
    assert_eq!(cfg["num_codebooks"], 37);
    assert_eq!(cfg["audio_token_id"], 24);
}

#[tokio::test]
async fn status_reports_unloaded_before_any_synthesis() {
    let app = build_tts_test_app();
    let (status, body) = json_get(&app, "/api/tts/status").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["data"]["loaded"], false);
    assert!(body["data"]["uptime_secs"].is_null());
}

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

// Quiet the unused-import warning on routes when this file builds in
// isolation. `routes` is currently only used for type-side-effects.
#[allow(dead_code)]
fn _ensure_routes_module_linked() {
    let _ = routes::get_config;
}
