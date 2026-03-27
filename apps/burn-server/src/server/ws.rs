use axum::{
    extract::{
        ws::{Message, WebSocket},
        Path, State, WebSocketUpgrade,
    },
    response::IntoResponse,
};
use futures_util::{SinkExt, StreamExt};
use std::sync::Arc;
use tokio::sync::mpsc;

use super::state::{AppState, SubtitleMessage};

/// WebSocket upgrade handler — same protocol as parakeet-rs and vllm-server
pub async fn ws_handler(
    ws: WebSocketUpgrade,
    Path(session_id): Path<String>,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_ws(socket, session_id, state))
}

async fn handle_ws(socket: WebSocket, session_id: String, state: Arc<AppState>) {
    let (mut ws_tx, mut ws_rx) = socket.split();

    // Create a channel for subtitle messages
    let (sub_tx, mut sub_rx) = mpsc::channel::<SubtitleMessage>(256);

    // Register subscriber for this session
    {
        let mut sessions = state.sessions.write().await;
        if let Some(ctx) = sessions.get_mut(&session_id) {
            ctx.subscribers.push(sub_tx);
        } else {
            let _ = ws_tx
                .send(Message::Text(
                    serde_json::json!({"error": "Session not found"}).to_string(),
                ))
                .await;
            return;
        }
    }

    // Send welcome message (matches vllm-server protocol)
    let _ = ws_tx
        .send(Message::Text(
            serde_json::json!({"type": "welcome", "session_id": session_id}).to_string(),
        ))
        .await;

    // Forward subtitle messages to WebSocket
    let send_task = tokio::spawn(async move {
        while let Some(msg) = sub_rx.recv().await {
            let json = serde_json::to_string(&msg).unwrap_or_default();
            if ws_tx.send(Message::Text(json)).await.is_err() {
                break;
            }
        }
    });

    // Handle incoming WebSocket messages (ICE candidates, SDP, ready signal)
    let state_clone = state.clone();
    let session_id_clone = session_id.clone();
    let recv_task = tokio::spawn(async move {
        while let Some(Ok(msg)) = ws_rx.next().await {
            match msg {
                Message::Text(text) => {
                    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&text) {
                        if let Some(msg_type) = parsed.get("type").and_then(|t| t.as_str()) {
                            match msg_type {
                                "ready" => {
                                    eprintln!(
                                        "[WS] Client ready for session {}",
                                        session_id_clone
                                    );
                                    // TODO: signal session runner to start
                                }
                                "ice_candidate" => {
                                    eprintln!("[WS] ICE candidate for {}", session_id_clone);
                                    // TODO: handle ICE candidate exchange
                                }
                                "offer" => {
                                    eprintln!("[WS] SDP offer for {}", session_id_clone);
                                    // TODO: handle WebRTC SDP offer/answer
                                }
                                _ => {
                                    eprintln!(
                                        "[WS] Unknown message type '{}' for {}",
                                        msg_type, session_id_clone
                                    );
                                }
                            }
                        }
                    }
                }
                Message::Close(_) => break,
                _ => {}
            }
        }
    });

    // Wait for either task to finish
    tokio::select! {
        _ = send_task => {},
        _ = recv_task => {},
    }

    eprintln!("[WS] Client disconnected from session {}", session_id);
}
