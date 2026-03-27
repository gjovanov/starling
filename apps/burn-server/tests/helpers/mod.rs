//! Test helpers for building the burn-server app without GPU dependencies.

use axum::{
    body::Body,
    extract::DefaultBodyLimit,
    http::{Request, StatusCode},
    routing::{delete, get, post},
    Router,
};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};
use tower::ServiceExt;

// We can't import from the binary crate directly in integration tests,
// so we rebuild the router using the library types.
// For now, we test against the routes module directly.

use burn_server::server::routes;
use burn_server::server::state::AppState;
use burn_server::config::{Config, Quantization};

/// Build a test app with no model loaded and a temp media dir.
pub async fn build_test_app() -> Router {
    let tmp_dir = std::env::temp_dir().join(format!("burn-server-test-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&tmp_dir).ok();

    let config = Config {
        port: 0,
        quant: Quantization::Q4,
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

    // Create a minimal WebRTC API (no media engine needed for API tests)
    let media_engine = webrtc::api::media_engine::MediaEngine::default();
    let registry = webrtc::interceptor::registry::Registry::new();
    let webrtc_api = webrtc::api::APIBuilder::new()
        .with_media_engine(media_engine)
        .with_interceptor_registry(registry)
        .build();

    let state = Arc::new(AppState::new(config, webrtc_api));

    Router::new()
        .route("/health", get(|| async { "OK" }))
        .route("/api/config", get(routes::get_config))
        .route("/api/models", get(routes::list_models))
        .route("/api/media", get(routes::list_media))
        .route("/api/media/upload", post(routes::upload_media))
        .route("/api/media/:id", delete(routes::delete_media))
        .route("/api/modes", get(routes::list_modes))
        .route("/api/noise-cancellation", get(routes::list_noise_cancellation))
        .route("/api/diarization", get(routes::list_diarization))
        .route("/api/srt-streams", get(routes::list_srt_streams))
        .route("/api/sessions", get(routes::list_sessions))
        .route("/api/sessions", post(routes::create_session))
        .route("/api/sessions/:id", get(routes::get_session))
        .route("/api/sessions/:id", delete(routes::stop_session))
        .route("/api/sessions/:id/start", post(routes::start_session))
        .layer(axum::extract::DefaultBodyLimit::max(10 * 1024 * 1024))
        .layer(tower_http::cors::CorsLayer::permissive())
        .with_state(state)
}

/// Send GET request and parse JSON response.
pub async fn json_get(app: &Router, uri: &str) -> Value {
    let resp = app
        .clone()
        .oneshot(Request::builder().uri(uri).body(Body::empty()).unwrap())
        .await
        .unwrap();

    let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
        .await
        .unwrap();
    serde_json::from_slice(&body).unwrap()
}

/// Send POST request with JSON body and parse response.
pub async fn json_post(app: &Router, uri: &str, body: Value) -> Value {
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

    let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
        .await
        .unwrap();
    serde_json::from_slice(&body).unwrap()
}

/// Parse response body as JSON.
pub async fn json_body(resp: axum::http::Response<Body>) -> Value {
    let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
        .await
        .unwrap();
    serde_json::from_slice(&body).unwrap()
}
