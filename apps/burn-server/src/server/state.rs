use crate::config::Config;
use crate::inference::InferenceEngine;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64};
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex, RwLock};
use webrtc::peer_connection::RTCPeerConnection;
use webrtc::track::track_local::track_local_static_rtp::TrackLocalStaticRTP;

/// Subtitle message broadcast to WebSocket subscribers
#[derive(Debug, Clone, serde::Serialize)]
pub struct SubtitleMessage {
    /// Always "subtitle" — frontend dispatches on this field
    #[serde(rename = "type")]
    pub msg_type: String,
    pub session_id: String,
    pub text: String,
    pub is_final: bool,
    pub segment_index: u32,
    pub timestamp_ms: u64,
    /// Model inference time in milliseconds (per commit)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inference_time_ms: Option<f32>,
}

/// Session state enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum SessionState {
    Created,
    Starting,
    Running,
    Completed,
    Stopped,
    Error,
}

/// Per-session information (matches vllm-server's SessionInfo for frontend compat)
#[derive(Debug, Clone, serde::Serialize)]
pub struct SessionInfo {
    pub id: String,
    pub state: SessionState,
    pub model_id: String,
    pub model_name: String,
    pub quant: String,
    pub media_id: Option<String>,
    pub media_filename: String,
    pub language: String,
    pub mode: String,
    pub duration_secs: f64,
    pub progress_secs: f64,
    pub created_at: f64,
    pub client_count: u32,
    pub source_type: String,
    pub noise_cancellation: String,
    pub diarization: bool,
    pub sentence_completion: String,
}

/// Client connection with WebRTC peer
pub struct ClientConnection {
    pub id: String,
    pub session_id: String,
    pub peer_connection: Arc<RTCPeerConnection>,
    pub ice_tx: mpsc::Sender<String>,
}

/// Per-session audio track state
pub struct SessionAudioState {
    pub audio_track: Arc<TrackLocalStaticRTP>,
    pub running: Arc<AtomicBool>,
    pub ffmpeg_pid: Arc<std::sync::atomic::AtomicU32>,
}

/// Per-session runtime context
pub struct SessionContext {
    pub info: SessionInfo,
    pub subscribers: Vec<mpsc::Sender<SubtitleMessage>>,
    pub audio_state: Option<SessionAudioState>,
    /// Audio track for WebRTC RTP writing (set by WS handler on "ready")
    pub rtp_track: Option<Arc<TrackLocalStaticRTP>>,
    /// Inbound audio channel for speakers sessions. When a WS client
    /// connects with role=uplink, it feeds decoded 16 kHz f32 PCM here.
    /// The session runner reads from this receiver.
    pub audio_in_tx: Option<mpsc::Sender<Vec<f32>>>,
}

/// Shared application state
pub struct AppState {
    pub config: Config,
    pub webrtc_api: webrtc::api::API,
    pub sessions: RwLock<HashMap<String, SessionContext>>,
    pub clients: Mutex<HashMap<String, ClientConnection>>,
    pub client_count: AtomicU64,
    /// Shared inference engine (Q4 or BF16, loaded at startup)
    pub engine: Option<Arc<dyn InferenceEngine>>,
    /// Lazy-loaded TTS pipeline. Initialised on the first
    /// `/api/tts/*` HTTP call. Always present even when the
    /// `voxtral-tts` Cargo feature isn't enabled — the routes that
    /// need it are gated separately.
    #[cfg(feature = "voxtral-tts")]
    pub tts: Arc<crate::server::tts_routes::TtsLifecycleState>,
}

impl AppState {
    pub fn new(config: Config, webrtc_api: webrtc::api::API) -> Self {
        Self {
            config,
            webrtc_api,
            sessions: RwLock::new(HashMap::new()),
            clients: Mutex::new(HashMap::new()),
            client_count: AtomicU64::new(0),
            engine: None,
            #[cfg(feature = "voxtral-tts")]
            tts: Arc::new(crate::server::tts_routes::TtsLifecycleState::new()),
        }
    }

    pub fn with_engine(mut self, engine: Arc<dyn InferenceEngine>) -> Self {
        self.engine = Some(engine);
        self
    }
}
