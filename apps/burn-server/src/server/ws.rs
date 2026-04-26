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
use webrtc::ice_transport::ice_candidate::RTCIceCandidateInit;
use webrtc::ice_transport::ice_credential_type::RTCIceCredentialType;
use webrtc::ice_transport::ice_server::RTCIceServer;
use webrtc::peer_connection::configuration::RTCConfiguration;
use webrtc::peer_connection::policy::ice_transport_policy::RTCIceTransportPolicy;
use webrtc::peer_connection::sdp::session_description::RTCSessionDescription;
use webrtc::track::track_local::track_local_static_rtp::TrackLocalStaticRTP;
use webrtc::track::track_remote::TrackRemote;

use super::state::{AppState, SubtitleMessage};
use crate::audio::inbound::Downsampler48to16;

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
        let mut pc: Option<Arc<webrtc::peer_connection::RTCPeerConnection>> = None;
        let mut is_uplink = false;

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
                            let role = parsed.get("role").and_then(|r| r.as_str()).unwrap_or("");
                            is_uplink = role == "uplink";
                            eprintln!(
                                "[WS {}] Client ready for session {} (role={})",
                                client_id_clone,
                                session_id_clone,
                                if is_uplink { "uplink" } else { "receiver" }
                            );

                            let rtc_config =
                                build_rtc_config(&state_clone.config);

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

                            // Forward server ICE candidates to client via WebSocket.
                            // (Shared by both flows.)
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

                            if is_uplink {
                                // Speakers / uplink flow: wait for the client's SDP offer.
                                //
                                // Attach an on_track handler that will fire after the offer
                                // is accepted; it pipes decoded PCM into the session's
                                // audio_in_tx channel for the inference runner.
                                let audio_in_tx = {
                                    let sessions = state_clone.sessions.read().await;
                                    sessions
                                        .get(&session_id_clone)
                                        .and_then(|ctx| ctx.audio_in_tx.clone())
                                };

                                let Some(audio_in_tx) = audio_in_tx else {
                                    eprintln!(
                                        "[WS {}] No audio_in_tx on session {} — is it a 'speakers' session that has been started?",
                                        client_id_clone, session_id_clone
                                    );
                                    let _ = ws_out_for_recv.send(
                                        serde_json::json!({
                                            "type": "error",
                                            "message": "Session does not expect an uplink. Make sure you created it with source='speakers' and called /api/sessions/<id>/start first."
                                        }).to_string()
                                    ).await;
                                    continue;
                                };

                                let on_track_client = client_id_clone.clone();
                                peer_connection.on_track(Box::new(
                                    move |track: Arc<TrackRemote>, _, _| {
                                        let audio_tx = audio_in_tx.clone();
                                        let cid = on_track_client.clone();
                                        Box::pin(async move {
                                            eprintln!(
                                                "[WS {}] on_track fired: codec={} kind={}",
                                                cid,
                                                track.codec().capability.mime_type,
                                                track.kind()
                                            );
                                            tokio::spawn(async move {
                                                pump_inbound_opus(track, audio_tx, cid).await;
                                            });
                                        })
                                    },
                                ));

                                pc = Some(peer_connection);
                                eprintln!(
                                    "[WS {}] Uplink PC ready — waiting for client SDP offer",
                                    client_id_clone
                                );
                                // Note: the client already received the initial welcome
                                // sent on WS open and used that to fire its startOffer().
                                // We deliberately do NOT emit a second welcome here.
                            } else {
                                // Receiver (media / SRT) flow: create outbound audio track
                                // and send an SDP offer to the client.
                                let audio_track = Arc::new(TrackLocalStaticRTP::new(
                                    webrtc::rtp_transceiver::rtp_codec::RTCRtpCodecCapability {
                                        mime_type: webrtc::api::media_engine::MIME_TYPE_OPUS
                                            .to_owned(),
                                        clock_rate: 48000,
                                        channels: 1,
                                        sdp_fmtp_line: "minptime=10;useinbandfec=1".to_owned(),
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

                                {
                                    let mut sessions = state_clone.sessions.write().await;
                                    if let Some(ctx) = sessions.get_mut(&session_id_clone) {
                                        ctx.rtp_track = Some(audio_track);
                                    }
                                }

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

                        "offer" => {
                            // Uplink flow: client sent us an SDP offer.
                            let peer_connection = match pc.as_ref() {
                                Some(p) => p.clone(),
                                None => {
                                    eprintln!(
                                        "[WS {}] Offer received before ready — ignoring",
                                        client_id_clone
                                    );
                                    continue;
                                }
                            };

                            let sdp = parsed
                                .get("sdp")
                                .and_then(|s| s.as_str())
                                .unwrap_or_default()
                                .to_string();
                            let offer = match RTCSessionDescription::offer(sdp) {
                                Ok(o) => o,
                                Err(e) => {
                                    eprintln!(
                                        "[WS {}] Invalid SDP offer: {}",
                                        client_id_clone, e
                                    );
                                    continue;
                                }
                            };

                            if let Err(e) = peer_connection.set_remote_description(offer).await {
                                eprintln!(
                                    "[WS {}] set_remote_description(offer) failed: {}",
                                    client_id_clone, e
                                );
                                continue;
                            }

                            let answer = match peer_connection.create_answer(None).await {
                                Ok(a) => a,
                                Err(e) => {
                                    eprintln!(
                                        "[WS {}] Failed to create answer: {}",
                                        client_id_clone, e
                                    );
                                    continue;
                                }
                            };

                            if let Err(e) =
                                peer_connection.set_local_description(answer.clone()).await
                            {
                                eprintln!(
                                    "[WS {}] Failed to set local description (answer): {}",
                                    client_id_clone, e
                                );
                                continue;
                            }

                            let answer_msg = serde_json::json!({
                                "type": "answer",
                                "sdp": answer.sdp,
                            });
                            let _ = ws_out_for_recv.send(answer_msg.to_string()).await;
                            eprintln!("[WS {}] SDP answer sent", client_id_clone);
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

/// Build an RTCConfiguration from the server Config (ICE servers + transport policy).
fn build_rtc_config(config: &crate::config::Config) -> RTCConfiguration {
    let force_relay = config.force_relay;
    let mut ice_servers = vec![];

    if !force_relay {
        ice_servers.push(RTCIceServer {
            urls: vec!["stun:stun.l.google.com:19302".to_string()],
            ..Default::default()
        });
    }

    if let Some(ref turn) = config.turn_server {
        let (username, credential) = if let Some(ref secret) = config.turn_shared_secret {
            crate::server::routes::generate_turn_credentials(secret)
        } else {
            (
                config.turn_username.clone().unwrap_or_default(),
                config.turn_password.clone().unwrap_or_default(),
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

    RTCConfiguration {
        ice_servers,
        ice_transport_policy: if force_relay {
            RTCIceTransportPolicy::Relay
        } else {
            RTCIceTransportPolicy::All
        },
        ..Default::default()
    }
}

/// Consume an inbound WebRTC Opus audio track. Reads RTP packets, decodes the
/// Opus payload to 48 kHz i16 PCM, converts to f32, downsamples to 16 kHz,
/// and forwards the samples to the session's inference runner.
///
/// Runs until the remote track ends (read_rtp returns an error).
async fn pump_inbound_opus(
    track: Arc<TrackRemote>,
    audio_tx: mpsc::Sender<Vec<f32>>,
    client_id: String,
) {
    let codec = track.codec();
    let mime = codec.capability.mime_type;
    let clock_rate = codec.capability.clock_rate;
    let channels = codec.capability.channels;
    eprintln!(
        "[WS {}] pump_inbound_opus start: mime={} clock={} channels={} ssrc={}",
        client_id,
        mime,
        clock_rate,
        channels,
        track.ssrc()
    );

    if !mime.eq_ignore_ascii_case("audio/opus") {
        eprintln!(
            "[WS {}] WARN: expected audio/opus, got '{}'. Decoding may fail.",
            client_id, mime
        );
    }

    // Opus decoder for mono 48 kHz audio.
    // The decoder configured as Mono will mix-down stereo Opus packets internally,
    // so this works regardless of how many channels the browser negotiated.
    let mut decoder = match audiopus::coder::Decoder::new(
        audiopus::SampleRate::Hz48000,
        audiopus::Channels::Mono,
    ) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("[WS {}] Failed to create Opus decoder: {}", client_id, e);
            return;
        }
    };

    // Max Opus frame is 120 ms @ 48 kHz mono = 5760 samples.
    let mut pcm_i16 = vec![0i16; 5760];
    let mut downsampler = Downsampler48to16::new();
    let mut total_samples_16k = 0u64;
    let mut packets_seen = 0u64;
    let mut decode_errors = 0u64;
    let mut send_errors = 0u64;
    let mut last_log = std::time::Instant::now();
    let started_at = std::time::Instant::now();

    loop {
        let read_result = track.read_rtp().await;
        let packet = match read_result {
            Ok((p, _attr)) => p,
            Err(e) => {
                eprintln!(
                    "[WS {}] read_rtp ended after {} packets ({:.1}s audio): {}",
                    client_id,
                    packets_seen,
                    total_samples_16k as f32 / 16000.0,
                    e
                );
                break;
            }
        };

        packets_seen += 1;

        if packets_seen == 1 {
            eprintln!(
                "[WS {}] First RTP packet: payload={} bytes, seq={}, ts={}, marker={}",
                client_id,
                packet.payload.len(),
                packet.header.sequence_number,
                packet.header.timestamp,
                packet.header.marker
            );
        }

        if packet.payload.is_empty() {
            continue;
        }

        let payload_bytes: &[u8] = packet.payload.as_ref();
        let input_packet = match audiopus::packet::Packet::try_from(payload_bytes) {
            Ok(p) => p,
            Err(e) => {
                decode_errors += 1;
                if decode_errors <= 5 {
                    eprintln!(
                        "[WS {}] Invalid Opus packet (#{}): {} (payload={} bytes)",
                        client_id,
                        decode_errors,
                        e,
                        payload_bytes.len()
                    );
                }
                continue;
            }
        };
        let output = match audiopus::MutSignals::try_from(&mut pcm_i16[..]) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("[WS {}] Failed to build MutSignals: {}", client_id, e);
                continue;
            }
        };

        match decoder.decode(Some(input_packet), output, false) {
            Ok(n) => {
                if n == 0 {
                    continue;
                }
                let f32_samples: Vec<f32> =
                    pcm_i16[..n].iter().map(|s| *s as f32 / 32768.0).collect();
                let samples_16k = downsampler.feed(&f32_samples);
                total_samples_16k += samples_16k.len() as u64;

                if !samples_16k.is_empty() {
                    if let Err(e) = audio_tx.send(samples_16k).await {
                        send_errors += 1;
                        eprintln!(
                            "[WS {}] Inference sink closed: {} — stopping inbound pump",
                            client_id, e
                        );
                        break;
                    }
                }

                if packets_seen == 1 {
                    eprintln!(
                        "[WS {}] First Opus decode: {} samples @ 48k → {} samples @ 16k",
                        client_id,
                        n,
                        (n + 2) / 3
                    );
                }
                if last_log.elapsed().as_secs() >= 5 {
                    eprintln!(
                        "[WS {}] Uplink: {} pkts, {:.1}s audio, {} decode errors, {} send errors",
                        client_id,
                        packets_seen,
                        total_samples_16k as f32 / 16000.0,
                        decode_errors,
                        send_errors
                    );
                    last_log = std::time::Instant::now();
                }
            }
            Err(e) => {
                decode_errors += 1;
                if decode_errors <= 5 {
                    eprintln!("[WS {}] Opus decode error (#{}): {}", client_id, decode_errors, e);
                }
            }
        }
    }

    eprintln!(
        "[WS {}] Inbound pump finished: {} pkts in {:.1}s wall, {:.1}s audio decoded, {} decode errors",
        client_id,
        packets_seen,
        started_at.elapsed().as_secs_f32(),
        total_samples_16k as f32 / 16000.0,
        decode_errors
    );
}
