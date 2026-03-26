"""WebSocket handler — WebRTC signaling + subtitle streaming."""

from __future__ import annotations

import asyncio
import json
import sys
import uuid

from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription, RTCIceCandidate
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..config import settings, generate_turn_credentials
from ..state import app_state

router = APIRouter()


def _parse_ice_candidate(data: dict) -> RTCIceCandidate | None:
    """Parse ICE candidate from frontend JSON into aiortc RTCIceCandidate.

    The browser sends a raw candidate string like:
      candidate:188735573 1 udp 50340095 94.130.141.98 11576 typ relay ...
    aiortc requires parsed fields (component, foundation, ip, port, etc.).
    """
    candidate_str = data.get("candidate", "")
    if not candidate_str:
        return None

    # Strip "candidate:" prefix if present
    if candidate_str.startswith("candidate:"):
        candidate_str = candidate_str[len("candidate:"):]

    bits = candidate_str.split()
    if len(bits) < 8:
        return None

    # Parse optional attributes
    related_address = None
    related_port = None
    tcp_type = None
    for i in range(8, len(bits) - 1, 2):
        if bits[i] == "raddr":
            related_address = bits[i + 1]
        elif bits[i] == "rport":
            related_port = int(bits[i + 1])
        elif bits[i] == "tcptype":
            tcp_type = bits[i + 1]

    return RTCIceCandidate(
        foundation=bits[0],
        component=int(bits[1]),
        protocol=bits[2],
        priority=int(bits[3]),
        ip=bits[4],
        port=int(bits[5]),
        type=bits[7],
        relatedAddress=related_address,
        relatedPort=related_port,
        tcpType=tcp_type,
        sdpMid=data.get("sdpMid", "0"),
        sdpMLineIndex=data.get("sdpMLineIndex", 0),
    )


@router.websocket("/ws/{session_id}")
async def websocket_handler(ws: WebSocket, session_id: str):
    await ws.accept()
    client_id = str(uuid.uuid4())[:8]

    ctx = app_state.get_session(session_id)
    if ctx is None:
        await ws.send_json({"type": "error", "message": f"Session '{session_id}' not found"})
        await ws.close()
        return

    ctx.info.client_count += 1
    print(f"[WS {client_id}] Connected to session {session_id}", file=sys.stderr)

    # Send welcome
    await ws.send_json({
        "type": "welcome",
        "message": "Connected to voxtral-server",
        "client_id": client_id,
        "session": ctx.info.model_dump(),
    })

    # Subscribe to subtitle broadcasts
    sub_queue = ctx.subscribe()
    pc: RTCPeerConnection | None = None

    try:
        # Wait for ready message, handle signaling, and forward subtitles concurrently
        ready_received = asyncio.Event()

        async def handle_client_messages():
            nonlocal pc
            while True:
                try:
                    raw = await ws.receive_text()
                except WebSocketDisconnect:
                    break

                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    # Plain text "ready" message
                    if raw.strip().lower() == "ready":
                        ready_received.set()
                        ctx.client_ready.set()
                        await setup_webrtc()
                    continue

                msg_type = msg.get("type", "")

                if msg_type == "ready":
                    ready_received.set()
                    # Signal client_ready so session runner starts audio
                    # (even without full WebRTC handshake — subtitles work over WS)
                    ctx.client_ready.set()
                    await setup_webrtc()

                elif msg_type == "answer" and pc:
                    sdp = msg.get("sdp", "")
                    desc = RTCSessionDescription(sdp=sdp, type="answer")
                    await pc.setRemoteDescription(desc)
                    print(f"[WS {client_id}] SDP answer set", file=sys.stderr)
                    # Signal that a client is ready for audio
                    ctx.client_ready.set()

                elif msg_type == "ice-candidate" and pc:
                    candidate_data = msg.get("candidate", {})
                    if isinstance(candidate_data, dict):
                        candidate = _parse_ice_candidate(candidate_data)
                        if candidate:
                            await pc.addIceCandidate(candidate)

        async def setup_webrtc():
            nonlocal pc

            # Build ICE servers config and pass to RTCPeerConnection
            ice_servers = [RTCIceServer(urls="stun:stun.l.google.com:19302")]
            if settings.turn_server:
                if settings.turn_shared_secret:
                    username, credential = generate_turn_credentials(
                        settings.turn_shared_secret, settings.turn_credential_ttl
                    )
                else:
                    username = settings.turn_username
                    credential = settings.turn_password
                turn_urls = [settings.turn_server]
                if "?transport=" not in settings.turn_server:
                    turn_urls.append(f"{settings.turn_server}?transport=tcp")
                ice_servers.append(RTCIceServer(
                    urls=turn_urls,
                    username=username,
                    credential=credential,
                ))
            config = RTCConfiguration(iceServers=ice_servers)
            pc = RTCPeerConnection(configuration=config)

            # Wait for audio track to be created by session runner
            for _ in range(50):  # up to 5 seconds
                audio_track = getattr(ctx, "audio_track", None)
                if audio_track is not None:
                    break
                await asyncio.sleep(0.1)

            audio_track = getattr(ctx, "audio_track", None)
            if audio_track:
                pc.addTrack(audio_track)
                print(f"[WS {client_id}] Audio track added", file=sys.stderr)
            else:
                print(f"[WS {client_id}] WARNING: No audio track available", file=sys.stderr)

            # Handle ICE candidates
            @pc.on("icecandidate")
            async def on_ice_candidate(candidate):
                if candidate:
                    await ws.send_json({
                        "type": "ice-candidate",
                        "candidate": {
                            "candidate": candidate.candidate,
                            "sdpMid": candidate.sdpMid,
                            "sdpMLineIndex": candidate.sdpMLineIndex,
                        },
                    })

            # Create and send offer
            offer = await pc.createOffer()
            await pc.setLocalDescription(offer)
            await ws.send_json({
                "type": "offer",
                "sdp": pc.localDescription.sdp,
            })
            print(f"[WS {client_id}] SDP offer sent", file=sys.stderr)

        async def forward_subtitles():
            while True:
                try:
                    msg = await asyncio.wait_for(sub_queue.get(), timeout=10.0)
                    await ws.send_json(msg)
                except asyncio.TimeoutError:
                    # Send ping to keep alive
                    try:
                        await ws.send_json({"type": "ping"})
                    except Exception:
                        break
                except WebSocketDisconnect:
                    break
                except Exception as e:
                    print(f"[WS {client_id}] Forward error: {e}", file=sys.stderr)
                    break

        # Run both tasks concurrently
        await asyncio.gather(
            handle_client_messages(),
            forward_subtitles(),
            return_exceptions=True,
        )

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WS {client_id}] Error: {e}", file=sys.stderr)
    finally:
        ctx.info.client_count = max(0, ctx.info.client_count - 1)
        ctx.unsubscribe(sub_queue)
        if pc:
            await pc.close()
        print(f"[WS {client_id}] Disconnected from session {session_id}", file=sys.stderr)
