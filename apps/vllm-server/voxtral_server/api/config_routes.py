"""GET /api/config — server configuration for the frontend."""

from fastapi import APIRouter

from ..config import settings, generate_turn_credentials

router = APIRouter()


@router.get("/api/config")
async def get_config():
    host = settings.public_ip or "localhost"
    port = settings.port

    ice_servers = [{"urls": "stun:stun.l.google.com:19302"}]
    if settings.turn_server:
        # Generate ephemeral credentials if shared secret is configured
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

        ice_servers.append({
            "urls": turn_urls,
            "username": username,
            "credential": credential,
        })

    return {
        "wsUrl": f"ws://{host}:{port}/ws",
        "iceServers": ice_servers,
        "iceTransportPolicy": "relay" if settings.force_relay else "all",
        "audio": {"sampleRate": 16000, "channels": 1, "bufferSize": 4096},
        "subtitles": {"maxSegments": 1000, "autoScroll": True, "showTimestamps": True},
        "speakerColors": [
            "#4A90D9", "#50C878", "#E9967A", "#DDA0DD",
            "#F0E68C", "#87CEEB", "#FFB6C1", "#98FB98",
        ],
        "reconnect": {
            "enabled": True,
            "delay": 2000,
            "maxDelay": 30000,
            "maxAttempts": 10,
        },
    }
