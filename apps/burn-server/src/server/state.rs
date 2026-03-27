use crate::config::Config;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64};
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex, RwLock};
use webrtc::peer_connection::RTCPeerConnection;
use webrtc::track::track_local::track_local_static_rtp::TrackLocalStaticRTP;

/// Subtitle message broadcast to WebSocket subscribers
#[derive(Debug, Clone, serde::Serialize)]
pub struct SubtitleMessage {
    pub session_id: String,
    pub text: String,
    pub is_final: bool,
    pub segment_index: u32,
    pub timestamp_ms: u64,
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

/// Per-session information
#[derive(Debug, Clone, serde::Serialize)]
pub struct SessionInfo {
    pub id: String,
    pub state: SessionState,
    pub model_id: String,
    pub quant: String,
    pub media_id: Option<String>,
    pub language: String,
    pub mode: String,
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
}

/// Shared application state
pub struct AppState {
    pub config: Config,
    pub webrtc_api: webrtc::api::API,
    pub sessions: RwLock<HashMap<String, SessionContext>>,
    pub clients: Mutex<HashMap<String, ClientConnection>>,
    pub client_count: AtomicU64,
}

impl AppState {
    pub fn new(config: Config, webrtc_api: webrtc::api::API) -> Self {
        Self {
            config,
            webrtc_api,
            sessions: RwLock::new(HashMap::new()),
            clients: Mutex::new(HashMap::new()),
            client_count: AtomicU64::new(0),
        }
    }
}
