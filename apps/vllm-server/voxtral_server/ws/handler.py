"""WebSocket handler — WebRTC signaling + subtitle streaming."""

from __future__ import annotations

import asyncio
import json
import sys
import uuid

from aiortc import (
    MediaStreamTrack,
    RTCConfiguration,
    RTCIceCandidate,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
)
from av.audio.resampler import AudioResampler
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..config import settings, generate_turn_credentials
from ..state import app_state

import numpy as np

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


def _build_ice_servers() -> list[RTCIceServer]:
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
    return ice_servers


async def _pump_inbound_audio(track: MediaStreamTrack, out_queue: asyncio.Queue, client_id: str) -> None:
    """Read frames from an inbound WebRTC audio track, convert to 16 kHz mono float
    PCM and push them into ``out_queue`` for the session runner.

    Uses PyAV's ``AudioResampler`` so we don't have to hand-roll channel mix-down
    + sample-rate conversion. The browser typically sends Opus 48 kHz **stereo**;
    aiortc decodes that to a packed s16 frame whose ``ndarray`` shape is
    ``(1, samples * channels)`` (interleaved L/R), which is *not* what the naive
    ``axis=0`` mix-down expects. Letting libavfilter do the work avoids that class
    of bug entirely.
    """
    import time

    # Resample to 16 kHz mono float ('flt' → numpy float32 in [-1.0, 1.0]).
    resampler = AudioResampler(format="flt", layout="mono", rate=16000)

    total = 0
    frames_seen = 0
    decode_errors = 0
    nonzero_frames = 0
    max_abs_seen = 0.0
    started_at = time.monotonic()
    last_log = started_at

    print(f"[WS {client_id}] _pump_inbound_audio started (kind={track.kind})", file=sys.stderr)

    while True:
        try:
            frame = await track.recv()
        except Exception as e:
            print(
                f"[WS {client_id}] Inbound track ended after {frames_seen} frames "
                f"({total / 16000:.1f}s audio): {e}",
                file=sys.stderr,
            )
            break

        frames_seen += 1
        if frames_seen == 1:
            print(
                f"[WS {client_id}] First inbound frame: format={frame.format.name}, "
                f"layout={frame.layout.name}, rate={frame.sample_rate}, "
                f"samples={frame.samples}",
                file=sys.stderr,
            )

        # AudioResampler.resample yields a list of frames at the target format.
        try:
            out_frames = resampler.resample(frame)
        except Exception as e:
            decode_errors += 1
            if decode_errors <= 5:
                print(
                    f"[WS {client_id}] Resample error on frame #{frames_seen}: {e}",
                    file=sys.stderr,
                )
            continue

        for of in out_frames:
            try:
                arr = of.to_ndarray()
            except Exception as e:
                decode_errors += 1
                if decode_errors <= 5:
                    print(f"[WS {client_id}] to_ndarray error: {e}", file=sys.stderr)
                continue

            # 'flt' mono is shape (1, n) — flatten to 1-D
            samples = arr.flatten().astype(np.float32, copy=False)
            if samples.size == 0:
                continue

            # Track audio activity so we can spot silent / corrupt streams.
            local_max = float(np.abs(samples).max())
            if local_max > max_abs_seen:
                max_abs_seen = local_max
            if local_max > 0.005:  # ~ -46 dBFS, quieter than mic noise floor
                nonzero_frames += 1

            samples_16k = samples.tolist()
            total += len(samples_16k)

            try:
                out_queue.put_nowait(samples_16k)
            except asyncio.QueueFull:
                try:
                    out_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    out_queue.put_nowait(samples_16k)
                except asyncio.QueueFull:
                    pass

        now = time.monotonic()
        if now - last_log >= 5.0:
            print(
                f"[WS {client_id}] Uplink: {frames_seen} frames, {total / 16000:.1f}s audio, "
                f"max|sample|={max_abs_seen:.3f}, "
                f"{nonzero_frames}/{frames_seen} frames above noise floor, "
                f"{decode_errors} resample errors",
                file=sys.stderr,
            )
            last_log = now

    try:
        out_queue.put_nowait(None)
    except Exception:
        pass
    print(
        f"[WS {client_id}] Inbound pump done: {frames_seen} frames in "
        f"{time.monotonic() - started_at:.1f}s wall, {total / 16000:.1f}s audio decoded, "
        f"max|sample|={max_abs_seen:.3f}",
        file=sys.stderr,
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
    is_uplink = False

    try:
        # Wait for ready message, handle signaling, and forward subtitles concurrently
        ready_received = asyncio.Event()

        async def handle_client_messages():
            nonlocal pc, is_uplink
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
                        await setup_webrtc_receiver()
                    continue

                msg_type = msg.get("type", "")

                if msg_type == "ready":
                    ready_received.set()
                    ctx.client_ready.set()
                    role = msg.get("role", "")
                    if role == "uplink":
                        is_uplink = True
                        await setup_webrtc_uplink()
                    else:
                        await setup_webrtc_receiver()

                elif msg_type == "offer" and is_uplink and pc:
                    sdp = msg.get("sdp", "")
                    desc = RTCSessionDescription(sdp=sdp, type="offer")
                    await pc.setRemoteDescription(desc)
                    answer = await pc.createAnswer()
                    await pc.setLocalDescription(answer)
                    await ws.send_json({
                        "type": "answer",
                        "sdp": pc.localDescription.sdp,
                    })
                    print(f"[WS {client_id}] SDP answer sent (uplink)", file=sys.stderr)

                elif msg_type == "answer" and pc and not is_uplink:
                    sdp = msg.get("sdp", "")
                    desc = RTCSessionDescription(sdp=sdp, type="answer")
                    await pc.setRemoteDescription(desc)
                    print(f"[WS {client_id}] SDP answer set", file=sys.stderr)
                    ctx.client_ready.set()

                elif msg_type == "ice-candidate" and pc:
                    candidate_data = msg.get("candidate", {})
                    if isinstance(candidate_data, dict):
                        candidate = _parse_ice_candidate(candidate_data)
                        if candidate:
                            await pc.addIceCandidate(candidate)

        async def setup_webrtc_uplink():
            """Speakers: client will send an offer with its local track."""
            nonlocal pc
            config = RTCConfiguration(iceServers=_build_ice_servers())
            pc = RTCPeerConnection(configuration=config)

            audio_queue = ctx.audio_in_queue
            if audio_queue is None:
                await ws.send_json({
                    "type": "error",
                    "message": (
                        "Session does not expect an uplink. Create it with source='speakers' "
                        "and call /api/sessions/<id>/start before connecting."
                    ),
                })
                return

            @pc.on("track")
            def on_track(track: MediaStreamTrack):
                print(
                    f"[WS {client_id}] Inbound track: kind={track.kind}",
                    file=sys.stderr,
                )
                if track.kind == "audio":
                    asyncio.create_task(_pump_inbound_audio(track, audio_queue, client_id))

                @track.on("ended")
                async def _on_ended():
                    print(f"[WS {client_id}] Track ended", file=sys.stderr)

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

            # Note: the client already received the initial welcome on WS open and
            # used that to fire its startOffer(). Sending a second welcome would
            # cause the client to build a second PC and send a second offer.
            print(
                f"[WS {client_id}] Uplink PC ready — waiting for client SDP offer",
                file=sys.stderr,
            )

        async def setup_webrtc_receiver():
            """Media / SRT: server creates outbound track, sends offer to client."""
            nonlocal pc
            config = RTCConfiguration(iceServers=_build_ice_servers())
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
