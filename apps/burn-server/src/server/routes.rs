use axum::{
    extract::{Multipart, Path, State},
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use super::state::{AppState, SessionInfo, SessionState};
use crate::config::Quantization;

/// Standard API response envelope
#[derive(Serialize)]
pub struct ApiResponse<T: Serialize> {
    pub success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<T>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl<T: Serialize> ApiResponse<T> {
    pub fn ok(data: T) -> Json<ApiResponse<T>> {
        Json(ApiResponse {
            success: true,
            data: Some(data),
            error: None,
        })
    }
}

pub fn error_response(msg: &str) -> impl IntoResponse {
    (
        StatusCode::BAD_REQUEST,
        Json(ApiResponse::<()> {
            success: false,
            data: None,
            error: Some(msg.to_string()),
        }),
    )
}

// ── Model info ──────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub display_name: String,
    pub languages: Vec<String>,
    pub quant_options: Vec<String>,
}

pub async fn list_models(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let mut quant_options = vec![];
    if state.config.q4_model_path().exists() {
        quant_options.push("q4".to_string());
    }
    if state.config.bf16_model_path().exists() {
        quant_options.push("bf16".to_string());
    }

    let models = vec![ModelInfo {
        id: "voxtral-mini-4b".to_string(),
        display_name: "Voxtral Mini 4B (Burn)".to_string(),
        languages: vec![
            "en", "fr", "es", "de", "ru", "zh", "ja", "it", "pt", "nl", "ar", "hi", "ko",
        ]
        .into_iter()
        .map(String::from)
        .collect(),
        quant_options,
    }];

    ApiResponse::ok(models)
}

// ── Modes ───────────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct ModeInfo {
    pub id: String,
    pub display_name: String,
    pub description: String,
}

pub async fn list_modes() -> impl IntoResponse {
    let modes = vec![
        ModeInfo {
            id: "speedy".to_string(),
            display_name: "Speedy".to_string(),
            description: "Low-latency streaming with pause-based word confirmation".to_string(),
        },
        ModeInfo {
            id: "growing_segments".to_string(),
            display_name: "Growing Segments".to_string(),
            description: "Word-by-word PARTIAL updates building toward FINAL sentences".to_string(),
        },
        ModeInfo {
            id: "pause_segmented".to_string(),
            display_name: "Pause-Segmented".to_string(),
            description: "Segment audio by acoustic pauses, transcribe each chunk once".to_string(),
        },
    ];
    ApiResponse::ok(modes)
}

pub async fn list_noise_cancellation() -> impl IntoResponse {
    ApiResponse::ok(vec![serde_json::json!({"id": "none", "display_name": "None"})])
}

pub async fn list_diarization() -> impl IntoResponse {
    ApiResponse::ok(vec![serde_json::json!({"id": "none", "display_name": "None"})])
}

pub async fn list_srt_streams() -> impl IntoResponse {
    ApiResponse::ok(Vec::<serde_json::Value>::new())
}

// ── Media ───────────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct MediaFile {
    pub id: String,
    pub filename: String,
    pub size_bytes: u64,
}

pub async fn list_media(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let media_dir = &state.config.media_dir;
    let mut files = vec![];

    if let Ok(entries) = std::fs::read_dir(media_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                let audio_exts = ["wav", "mp3", "flac", "ogg", "m4a", "aac", "opus", "wma"];
                if audio_exts.contains(&ext.to_lowercase().as_str()) {
                    let filename = path.file_name().unwrap_or_default().to_string_lossy().to_string();
                    let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
                    files.push(MediaFile {
                        id: filename.clone(),
                        filename,
                        size_bytes: size,
                    });
                }
            }
        }
    }

    ApiResponse::ok(files)
}

pub async fn upload_media(
    State(state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> impl IntoResponse {
    while let Ok(Some(field)) = multipart.next_field().await {
        let filename = match field.file_name() {
            Some(name) => name.to_string(),
            None => continue,
        };

        let data = match field.bytes().await {
            Ok(data) => data,
            Err(e) => return error_response(&format!("Failed to read upload: {}", e)).into_response(),
        };

        let path = state.config.media_dir.join(&filename);
        if let Err(e) = tokio::fs::write(&path, &data).await {
            return error_response(&format!("Failed to save file: {}", e)).into_response();
        }

        return ApiResponse::ok(MediaFile {
            id: filename.clone(),
            filename,
            size_bytes: data.len() as u64,
        })
        .into_response();
    }

    error_response("No file in upload").into_response()
}

pub async fn delete_media(
    State(state): State<Arc<AppState>>,
    Path(media_id): Path<String>,
) -> impl IntoResponse {
    let path = state.config.media_dir.join(&media_id);
    if path.exists() {
        let _ = tokio::fs::remove_file(&path).await;
        ApiResponse::ok("deleted")
    } else {
        ApiResponse::ok("not found")
    }
}

// ── Sessions ────────────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct CreateSessionRequest {
    pub model_id: String,
    #[serde(default = "default_language")]
    pub language: String,
    #[serde(default = "default_mode")]
    pub mode: String,
    pub media_id: Option<String>,
    pub quant: Option<String>,
}

fn default_language() -> String {
    "en".to_string()
}
fn default_mode() -> String {
    "speedy".to_string()
}

pub async fn list_sessions(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let sessions = state.sessions.read().await;
    let infos: Vec<SessionInfo> = sessions.values().map(|s| s.info.clone()).collect();
    ApiResponse::ok(infos)
}

pub async fn create_session(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateSessionRequest>,
) -> impl IntoResponse {
    let session_id = uuid::Uuid::new_v4().to_string();

    let quant_str = req.quant.unwrap_or_else(|| state.config.quant.to_string().to_lowercase());

    let info = SessionInfo {
        id: session_id.clone(),
        state: SessionState::Created,
        model_id: req.model_id,
        quant: quant_str,
        media_id: req.media_id,
        language: req.language,
        mode: req.mode,
    };

    let ctx = super::state::SessionContext {
        info: info.clone(),
        subscribers: vec![],
        audio_state: None,
    };

    state.sessions.write().await.insert(session_id, ctx);
    ApiResponse::ok(info)
}

pub async fn get_session(
    State(state): State<Arc<AppState>>,
    Path(session_id): Path<String>,
) -> impl IntoResponse {
    let sessions = state.sessions.read().await;
    match sessions.get(&session_id) {
        Some(ctx) => ApiResponse::ok(ctx.info.clone()).into_response(),
        None => error_response("Session not found").into_response(),
    }
}

pub async fn stop_session(
    State(state): State<Arc<AppState>>,
    Path(session_id): Path<String>,
) -> impl IntoResponse {
    let mut sessions = state.sessions.write().await;
    if sessions.remove(&session_id).is_some() {
        ApiResponse::ok("stopped")
    } else {
        ApiResponse::ok("not found")
    }
}

pub async fn start_session(
    State(state): State<Arc<AppState>>,
    Path(session_id): Path<String>,
) -> impl IntoResponse {
    // Validate session exists and get its info
    let session_info = {
        let mut sessions = state.sessions.write().await;
        match sessions.get_mut(&session_id) {
            Some(ctx) => {
                if ctx.info.state != SessionState::Created {
                    return error_response("Session already started").into_response();
                }
                ctx.info.state = SessionState::Starting;
                ctx.info.clone()
            }
            None => return error_response("Session not found").into_response(),
        }
    };

    // Check that we have an inference engine
    let engine = match &state.engine {
        Some(e) => e.clone(),
        None => return error_response("Inference engine not loaded").into_response(),
    };

    // Resolve media file path
    let media_path = match &session_info.media_id {
        Some(media_id) => state.config.media_dir.join(media_id),
        None => return error_response("No media_id specified").into_response(),
    };

    if !media_path.exists() {
        return error_response(&format!("Media file not found: {}", media_path.display()))
            .into_response();
    }

    // Create subtitle broadcast channel
    let (subtitle_tx, mut subtitle_rx) =
        tokio::sync::mpsc::channel::<crate::server::state::SubtitleMessage>(256);

    // Create state update channel
    let (state_tx, mut state_rx) =
        tokio::sync::mpsc::channel::<(String, SessionState)>(16);

    // Spawn task to forward subtitles to session subscribers
    let state_clone = state.clone();
    let sid = session_id.clone();
    tokio::spawn(async move {
        while let Some(msg) = subtitle_rx.recv().await {
            let sessions = state_clone.sessions.read().await;
            if let Some(ctx) = sessions.get(&sid) {
                for sub in &ctx.subscribers {
                    let _ = sub.send(msg.clone()).await;
                }
            }
        }
    });

    // Spawn task to update session state
    let state_clone2 = state.clone();
    tokio::spawn(async move {
        while let Some((sid, new_state)) = state_rx.recv().await {
            let mut sessions = state_clone2.sessions.write().await;
            if let Some(ctx) = sessions.get_mut(&sid) {
                ctx.info.state = new_state;
            }
        }
    });

    // Spawn the transcription session
    let runner_config = crate::transcription::session::SessionRunnerConfig {
        session_id: session_id.clone(),
        media_path,
        language: session_info.language.clone(),
        quant: session_info.quant.clone(),
    };

    tokio::spawn(async move {
        crate::transcription::session::run_session(runner_config, engine, subtitle_tx, state_tx)
            .await;
    });

    ApiResponse::ok(session_info).into_response()
}

// ── Config ──────────────────────────────────────────────────────────

pub async fn get_config(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let detected_ip = std::process::Command::new("hostname")
        .arg("-I")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.split_whitespace().next().map(String::from));

    let ws_host = state
        .config
        .public_ip
        .clone()
        .or(detected_ip)
        .unwrap_or_else(|| "localhost".to_string());

    let ws_url = format!("ws://{}:{}/ws", ws_host, state.config.port);

    let mut config = serde_json::json!({
        "ws_url": ws_url,
    });

    if let Some(ref turn) = state.config.turn_server {
        config["turn_server"] = serde_json::json!(turn);
    }
    if let Some(ref user) = state.config.turn_username {
        config["turn_username"] = serde_json::json!(user);
    }
    if let Some(ref pass) = state.config.turn_password {
        config["turn_password"] = serde_json::json!(pass);
    }

    ApiResponse::ok(config)
}
