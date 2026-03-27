//! Integration tests for the burn-server HTTP API.
//!
//! Tests the REST API endpoints without requiring model weights or GPU.
//! Uses a mock InferenceEngine that returns fixed text.

use axum::{
    body::Body,
    http::{Request, StatusCode},
    Router,
};
use burn_server::{self};
use serde_json::Value;
use std::sync::Arc;
use tower::ServiceExt;

mod helpers;
use helpers::{build_test_app, json_body, json_get, json_post};

#[tokio::test]
async fn test_health_endpoint() {
    let app = build_test_app().await;
    let resp = app
        .oneshot(Request::builder().uri("/health").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_list_models() {
    let app = build_test_app().await;
    let body = json_get(&app, "/api/models").await;
    assert_eq!(body["success"], true);
    let models = body["data"].as_array().unwrap();
    assert!(!models.is_empty());
    assert_eq!(models[0]["id"], "voxtral-mini-4b");
}

#[tokio::test]
async fn test_list_modes() {
    let app = build_test_app().await;
    let body = json_get(&app, "/api/modes").await;
    assert_eq!(body["success"], true);
    let modes = body["data"].as_array().unwrap();
    assert!(modes.len() >= 3);

    let mode_ids: Vec<&str> = modes.iter().map(|m| m["id"].as_str().unwrap()).collect();
    assert!(mode_ids.contains(&"speedy"));
    assert!(mode_ids.contains(&"growing_segments"));
    assert!(mode_ids.contains(&"pause_segmented"));
}

#[tokio::test]
async fn test_list_media_empty() {
    let app = build_test_app().await;
    let body = json_get(&app, "/api/media").await;
    assert_eq!(body["success"], true);
    // Media dir might be empty in test environment
    assert!(body["data"].is_array());
}

#[tokio::test]
async fn test_create_session() {
    let app = build_test_app().await;
    let body = json_post(
        &app,
        "/api/sessions",
        serde_json::json!({
            "model_id": "voxtral-mini-4b",
            "language": "en",
            "mode": "speedy",
            "media_id": "test.wav"
        }),
    )
    .await;

    assert_eq!(body["success"], true);
    let session = &body["data"];
    assert!(session["id"].is_string());
    assert_eq!(session["state"], "created");
    assert_eq!(session["model_id"], "voxtral-mini-4b");
    assert_eq!(session["language"], "en");
    assert_eq!(session["mode"], "speedy");
}

#[tokio::test]
async fn test_list_sessions_after_create() {
    let app = build_test_app().await;

    // Create a session
    json_post(
        &app,
        "/api/sessions",
        serde_json::json!({
            "model_id": "voxtral-mini-4b",
            "language": "de",
            "mode": "growing_segments"
        }),
    )
    .await;

    // List sessions
    let body = json_get(&app, "/api/sessions").await;
    assert_eq!(body["success"], true);
    let sessions = body["data"].as_array().unwrap();
    assert_eq!(sessions.len(), 1);
    assert_eq!(sessions[0]["language"], "de");
}

#[tokio::test]
async fn test_get_session() {
    let app = build_test_app().await;

    // Create
    let create_body = json_post(
        &app,
        "/api/sessions",
        serde_json::json!({
            "model_id": "voxtral-mini-4b",
            "language": "en",
            "mode": "speedy"
        }),
    )
    .await;
    assert_eq!(create_body["success"], true);
    let session_id = create_body["data"]["id"].as_str().unwrap();

    // Get — parse response manually to handle both success and error cases
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .uri(&format!("/api/sessions/{}", session_id))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let status = resp.status();
    let body_bytes = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
        .await
        .unwrap();

    if !body_bytes.is_empty() {
        let body: Value = serde_json::from_slice(&body_bytes).unwrap();
        assert_eq!(body["success"], true, "Get session failed: {:?}", body);
        assert_eq!(body["data"]["id"], session_id);
    } else {
        panic!("Empty response body, status: {}", status);
    }
}

#[tokio::test]
async fn test_stop_session() {
    let app = build_test_app().await;

    // Create
    let create_body = json_post(
        &app,
        "/api/sessions",
        serde_json::json!({
            "model_id": "voxtral-mini-4b",
            "mode": "speedy"
        }),
    )
    .await;
    let session_id = create_body["data"]["id"].as_str().unwrap();

    // Stop (DELETE)
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri(&format!("/api/sessions/{}", session_id))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    // Verify session list is empty
    let body = json_get(&app, "/api/sessions").await;
    let sessions = body["data"].as_array().unwrap();
    assert!(sessions.is_empty(), "Session should be removed after DELETE");
}

#[tokio::test]
async fn test_get_nonexistent_session() {
    let app = build_test_app().await;
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/api/sessions/nonexistent-id")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    // error_response returns 400 BAD_REQUEST
    let status = resp.status();
    assert!(
        status == StatusCode::BAD_REQUEST || status == StatusCode::NOT_FOUND,
        "Expected 400 or 404 for nonexistent session, got {}",
        status
    );
}

#[tokio::test]
async fn test_config_endpoint() {
    let app = build_test_app().await;
    let body = json_get(&app, "/api/config").await;
    assert_eq!(body["success"], true);
    assert!(body["data"]["ws_url"].is_string());
}

#[tokio::test]
async fn test_noise_cancellation_endpoint() {
    let app = build_test_app().await;
    let body = json_get(&app, "/api/noise-cancellation").await;
    assert_eq!(body["success"], true);
}

#[tokio::test]
async fn test_diarization_endpoint() {
    let app = build_test_app().await;
    let body = json_get(&app, "/api/diarization").await;
    assert_eq!(body["success"], true);
}

#[tokio::test]
async fn test_srt_streams_endpoint() {
    let app = build_test_app().await;
    let body = json_get(&app, "/api/srt-streams").await;
    assert_eq!(body["success"], true);
}

#[tokio::test]
async fn test_create_session_with_quant() {
    let app = build_test_app().await;
    let body = json_post(
        &app,
        "/api/sessions",
        serde_json::json!({
            "model_id": "voxtral-mini-4b",
            "language": "en",
            "mode": "speedy",
            "quant": "bf16"
        }),
    )
    .await;
    assert_eq!(body["success"], true);
    assert_eq!(body["data"]["quant"], "bf16");
}
