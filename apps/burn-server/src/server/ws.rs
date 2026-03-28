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
use webrtc::api::media_engine::MediaEngine;
use webrtc::ice_transport::ice_candidate::RTCIceCandidateInit;
use webrtc::ice_transport::ice_credential_type::RTCIceCredentialType;
use webrtc::ice_transport::ice_server::RTCIceServer;
use webrtc::peer_connection::configuration::RTCConfiguration;
use webrtc::peer_connection::policy::ice_transport_policy::RTCIceTransportPolicy;
use webrtc::peer_connection::sdp::session_description::RTCSessionDescription;
use webrtc::track::track_local::track_local_static_rtp::TrackLocalStaticRTP;

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
                    serde_json::json!({"type": "error", "message": "Session not found"})
                        .to_string(),
                ))
                .await;
            return;
        }
    }

    let client_id = uuid::Uuid::new_v4().to_string()[..8].to_string();

    // Send welcome message (matches vllm-server protocol)
    let welcome = {
        let sessions = state.sessions.read().await;
        if let Some(ctx) = sessions.get(&session_id) {
            serde_json::json!({
                "type": "welcome",
                "message": "Connected to burn-server",
                "client_id": client_id,
                "session": {
                    "id": ctx.info.id,
                    "state": ctx.info.state,
                    "model_id": ctx.info.model_id,
                    "media_id": ctx.info.media_id,
                    "language": ctx.info.language,
                    "mode": ctx.info.mode,
                }
            })
        } else {
            serde_json::json!({"type": "welcome", "session_id": session_id})
        }
    };
    let _ = ws_tx
        .send(Message::Text(welcome.to_string()))
        .await;

    eprintln!("[WS {}] Connected to session {}", client_id, session_id);

    // Channel for sending messages to the WebSocket from multiple tasks
    let (ws_out_tx, mut ws_out_rx) = mpsc::channel::<String>(64);

    // Forward subtitle messages to the outbound channel
    let ws_out_sub = ws_out_tx.clone();
    let sub_forward = tokio::spawn(async move {
        while let Some(msg) = sub_rx.recv().await {
            let json = serde_json::to_string(&msg).unwrap_or_default();
            if ws_out_sub.send(json).await.is_err() {
                break;
            }
        }
    });

    // WebSocket writer task — single owner of ws_tx
    let send_task = tokio::spawn(async move {
        while let Some(json) = ws_out_rx.recv().await {
            if ws_tx.send(Message::Text(json)).await.is_err() {
                break;
            }
        }
    });

    // Handle incoming WebSocket messages (ICE candidates, SDP, ready signal)
    let state_clone = state.clone();
    let session_id_clone = session_id.clone();
    let ws_out_for_recv = ws_out_tx.clone();
    let client_id_clone = client_id.clone();
    let recv_task = tokio::spawn(async move {
        let mut pc = None;

        while let Some(Ok(msg)) = ws_rx.next().await {
            match msg {
                Message::Text(text) => {
                    let parsed = match serde_json::from_str::<serde_json::Value>(&text) {
                        Ok(v) => v,
                        Err(_) => continue,
                    };
                    let msg_type = match parsed.get("type").and_then(|t| t.as_str()) {
                        Some(t) => t,
                        None => continue,
                    };

                    match msg_type {
                        "ready" => {
                            eprintln!(
                                "[WS {}] Client ready for session {}",
                                client_id_clone, session_id_clone
                            );

                            // Build ICE servers (matches parakeet-rs)
                            let force_relay = state_clone.config.force_relay;
                            let mut ice_servers = vec![];

                            // Only add STUN when not in relay-only mode
                            if !force_relay {
                                ice_servers.push(RTCIceServer {
                                    urls: vec!["stun:stun.l.google.com:19302".to_string()],
                                    ..Default::default()
                                });
                            }

                            if let Some(ref turn) = state_clone.config.turn_server {
                                let (username, credential) =
                                    if let Some(ref secret) = state_clone.config.turn_shared_secret
                                    {
                                        crate::server::routes::generate_turn_credentials(secret)
                                    } else {
                                        (
                                            state_clone
                                                .config
                                                .turn_username
                                                .clone()
                                                .unwrap_or_default(),
                                            state_clone
                                                .config
                                                .turn_password
                                                .clone()
                                                .unwrap_or_default(),
                                        )
                                    };

                                let mut turn_urls = vec![turn.clone()];
                                if !turn.contains("?transport=") {
                                    turn_urls.push(format!("{}?transport=tcp", turn));
                                }

                                ice_servers.push(RTCIceServer {
                                    urls: turn_urls,
                                    username,
                                    credential,
                                    credential_type: RTCIceCredentialType::Password,
                                });
                            }

                            let rtc_config = RTCConfiguration {
                                ice_servers,
                                ice_transport_policy: if force_relay {
                                    RTCIceTransportPolicy::Relay
                                } else {
                                    RTCIceTransportPolicy::All
                                },
                                ..Default::default()
                            };

                            // Create peer connection using the shared WebRTC API
                            let peer_connection = match state_clone
                                .webrtc_api
                                .new_peer_connection(rtc_config)
                                .await
                            {
                                Ok(pc) => Arc::new(pc),
                                Err(e) => {
                                    eprintln!(
                                        "[WS {}] Failed to create PeerConnection: {}",
                                        client_id_clone, e
                                    );
                                    let _ = ws_out_for_recv
                                        .send(
                                            serde_json::json!({
                                                "type": "error",
                                                "message": format!("WebRTC setup failed: {}", e)
                                            })
                                            .to_string(),
                                        )
                                        .await;
                                    continue;
                                }
                            };

                            // Add audio track (Opus via RTP, matching parakeet-rs)
                            let audio_track = Arc::new(TrackLocalStaticRTP::new(
                                webrtc::rtp_transceiver::rtp_codec::RTCRtpCodecCapability {
                                    mime_type: webrtc::api::media_engine::MIME_TYPE_OPUS
                                        .to_owned(),
                                    clock_rate: 48000,
                                    channels: 1,
                                    sdp_fmtp_line:
                                        "minptime=10;useinbandfec=1".to_owned(),
                                    rtcp_feedback: vec![],
                                },
                                "audio".to_string(),
                                "burn-server".to_string(),
                            ));

                            if let Err(e) = peer_connection
                                .add_track(audio_track.clone() as Arc<dyn webrtc::track::track_local::TrackLocal + Send + Sync>)
                                .await
                            {
                                eprintln!(
                                    "[WS {}] Failed to add audio track: {}",
                                    client_id_clone, e
                                );
                                continue;
                            }
                            eprintln!("[WS {}] Audio track added", client_id_clone);

                            // Store audio track in session for the session runner to write to
                            {
                                let mut sessions = state_clone.sessions.write().await;
                                if let Some(ctx) = sessions.get_mut(&session_id_clone) {
                                    ctx.rtp_track = Some(audio_track);
                                }
                            }

                            // Forward server ICE candidates to client via WebSocket
                            let ice_ws_tx = ws_out_for_recv.clone();
                            let ice_client_id = client_id_clone.clone();
                            peer_connection.on_ice_candidate(Box::new(
                                move |candidate| {
                                    let tx = ice_ws_tx.clone();
                                    let cid = ice_client_id.clone();
                                    Box::pin(async move {
                                        if let Some(c) = candidate {
                                            let json_val = match c.to_json() {
                                                Ok(j) => j,
                                                Err(e) => {
                                                    eprintln!(
                                                        "[WS {}] Failed to serialize ICE candidate: {}",
                                                        cid, e
                                                    );
                                                    return;
                                                }
                                            };
                                            let msg = serde_json::json!({
                                                "type": "ice-candidate",
                                                "candidate": {
                                                    "candidate": json_val.candidate,
                                                    "sdpMid": json_val.sdp_mid,
                                                    "sdpMLineIndex": json_val.sdp_mline_index,
                                                }
                                            });
                                            let _ = tx.send(msg.to_string()).await;
                                        }
                                    })
                                },
                            ));

                            // Create and send SDP offer
                            let offer = match peer_connection.create_offer(None).await {
                                Ok(o) => o,
                                Err(e) => {
                                    eprintln!(
                                        "[WS {}] Failed to create offer: {}",
                                        client_id_clone, e
                                    );
                                    continue;
                                }
                            };

                            if let Err(e) =
                                peer_connection.set_local_description(offer.clone()).await
                            {
                                eprintln!(
                                    "[WS {}] Failed to set local description: {}",
                                    client_id_clone, e
                                );
                                continue;
                            }

                            let offer_msg = serde_json::json!({
                                "type": "offer",
                                "sdp": offer.sdp,
                            });
                            let _ = ws_out_for_recv.send(offer_msg.to_string()).await;
                            eprintln!("[WS {}] SDP offer sent", client_id_clone);

                            pc = Some(peer_connection);
                        }

                        "answer" => {
                            if let Some(ref peer_connection) = pc {
                                let sdp = parsed
                                    .get("sdp")
                                    .and_then(|s| s.as_str())
                                    .unwrap_or_default()
                                    .to_string();

                                let answer = RTCSessionDescription::answer(sdp).unwrap();
                                if let Err(e) =
                                    peer_connection.set_remote_description(answer).await
                                {
                                    eprintln!(
                                        "[WS {}] Failed to set remote description: {}",
                                        client_id_clone, e
                                    );
                                } else {
                                    eprintln!("[WS {}] SDP answer set", client_id_clone);
                                }
                            }
                        }

                        "ice-candidate" | "ice_candidate" => {
                            if let Some(ref peer_connection) = pc {
                                if let Some(candidate_obj) = parsed.get("candidate") {
                                    let candidate_str = candidate_obj
                                        .get("candidate")
                                        .and_then(|c| c.as_str())
                                        .unwrap_or_default()
                                        .to_string();

                                    if candidate_str.is_empty() {
                                        continue;
                                    }

                                    let sdp_mid = candidate_obj
                                        .get("sdpMid")
                                        .and_then(|s| s.as_str())
                                        .map(String::from);

                                    let sdp_mline_index = candidate_obj
                                        .get("sdpMLineIndex")
                                        .and_then(|s| s.as_u64())
                                        .map(|n| n as u16);

                                    let init = RTCIceCandidateInit {
                                        candidate: candidate_str,
                                        sdp_mid: Some(sdp_mid.unwrap_or_default()),
                                        sdp_mline_index: Some(sdp_mline_index.unwrap_or(0)),
                                        username_fragment: Some(String::new()),
                                    };

                                    if let Err(e) =
                                        peer_connection.add_ice_candidate(init).await
                                    {
                                        eprintln!(
                                            "[WS {}] Failed to add ICE candidate: {}",
                                            client_id_clone, e
                                        );
                                    }
                                }
                            }
                        }

                        _ => {
                            eprintln!(
                                "[WS {}] Unknown message type '{}' for {}",
                                client_id_clone, msg_type, session_id_clone
                            );
                        }
                    }
                }
                Message::Close(_) => break,
                _ => {}
            }
        }

        // Cleanup peer connection on disconnect
        if let Some(peer_connection) = pc {
            let _ = peer_connection.close().await;
        }
    });

    // Wait for any task to finish
    tokio::select! {
        _ = send_task => {},
        _ = recv_task => {},
        _ = sub_forward => {},
    }

    eprintln!("[WS {}] Client disconnected from session {}", client_id, session_id);
}
