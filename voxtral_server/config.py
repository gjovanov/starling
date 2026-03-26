"""Server configuration via environment variables."""

import base64
import hashlib
import hmac
import time

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # vLLM server WebSocket URL
    vllm_url: str = "ws://localhost:8001/v1/realtime"

    # Server port
    port: int = 8090

    # Media directory
    media_dir: str = "./media"

    # Frontend directory
    frontend_path: str = "./frontend"

    # Public IP for WebRTC (auto-detected if not set)
    public_ip: str = ""

    # TURN server (optional)
    turn_server: str = ""
    turn_username: str = ""
    turn_password: str = ""

    # COTURN shared-secret auth (overrides static username/password)
    turn_shared_secret: str = ""
    turn_credential_ttl: int = 86400  # 24 hours

    # Force TURN relay mode
    force_relay: bool = False

    model_config = {
        "env_prefix": "VOXTRAL_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


def generate_turn_credentials(shared_secret: str, ttl: int) -> tuple[str, str]:
    """Generate ephemeral TURN credentials using HMAC-SHA1 (RFC 5389).

    Matches the Rust parakeet-server implementation exactly:
    - username = "<unix_expiry_timestamp>:parakeet"
    - credential = base64(HMAC-SHA1(shared_secret, username))
    """
    expiry = int(time.time()) + ttl
    username = f"{expiry}:parakeet"
    mac = hmac.new(shared_secret.encode(), username.encode(), hashlib.sha1)
    credential = base64.b64encode(mac.digest()).decode()
    return username, credential


settings = Settings()
