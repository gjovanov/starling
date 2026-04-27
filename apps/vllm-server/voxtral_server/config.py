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

    # ───────── Text-to-speech (Voxtral-4B-TTS-2603 via vllm-omni) ─────────

    # Base URL of the local vllm-omni TTS server (started by start-vllm-tts.sh).
    # Path-suffix `/v1` is appended to match the OpenAI-compatible endpoint.
    tts_vllm_url: str = "http://127.0.0.1:8002/v1"

    # Model "name" sent in the synth payload. vllm-omni accepts the on-disk
    # path it was started with (matches what `GET /v1/models` returns).
    tts_model_path: str = "./../../models/cache/tts"

    # Output directory for save-to-disk synthesis. Created on demand.
    tts_output_dir: str = "./tts_output"

    # Hard cap on input text length (chars). 20k = ~20 minutes of speech.
    # Long-form requests are split sentence-by-sentence; see
    # voxtral_server.tts.text.split_sentences and tts_long_max_secs.
    tts_max_chars: int = 20000

    # Wall-clock cap for a single long-form synthesize call (seconds).
    # Aborts the upstream stream cleanly if hit; partial audio is delivered.
    tts_long_max_secs: float = 300.0

    # Sentence-pipeline depth. Phase 5 v1 ships sequential (1) — no
    # overlapping upstream calls — until we measure stage-0 VRAM under
    # parallel load. Bumping to 2 enables prefetching the next sentence
    # while the current one is still streaming.
    tts_long_max_concurrency: int = 1

    # ───── Lifecycle (Phase 6) ─────
    # Auto-spawn the TTS subprocess on first use; auto-unload after idle.
    # Set to false if you manage start-vllm-tts.sh manually.
    tts_autostart: bool = True

    # Idle seconds before the TTS subprocess is sent SIGTERM. 0 disables
    # auto-unload (leaves it running once started).
    tts_idle_unload_secs: float = 600.0

    # Hard wall-clock cap for the warm-up (model load + CUDA-graph capture).
    # Voxtral-4B-TTS warmed up in ~75 s on our reference RTX 5090 Laptop;
    # 180 s is the comfortable ceiling.
    tts_boot_timeout_secs: float = 180.0

    # Refuse to start TTS if free GPU memory is below this threshold —
    # avoids the OOM that the spike documented (Phase 1.1). With ASR
    # resident at util 0.45 the GPU has ~12 GiB free; TTS needs 11.4 GiB.
    tts_min_free_vram_gib: float = 12.0

    # Port the ASR vLLM listens on. Used by the lifecycle's
    # `is_asr_running` probe to surface a clearer "blocked" reason.
    tts_asr_port: int = 8001

    # ───── Voice cloning (Phase 7) ─────
    # Local cache of uploaded voice references. One subdir per uploaded
    # voice: <id>.wav + <id>.json + an _audit.log of upload/delete events.
    tts_voice_refs_dir: str = "./voice_refs"
    tts_max_ref_audio_bytes: int = 5_000_000      # 5 MB
    tts_min_ref_duration_secs: float = 5.0
    tts_max_ref_duration_secs: float = 30.0
    tts_max_voice_refs: int = 50
    # Server-side enforcement of the "I have permission" checkbox. Set to
    # False only for local dev — the audit log still records the flag value.
    tts_require_permission: bool = True

    # Default voice when a request omits the field. Must be a valid voice id
    # from voxtral_server.tts.voices.VOICES.
    tts_default_voice: str = "casual_male"

    # HTTP timeout for the upstream synth call. The 30s benchmark in the
    # spike memo finished in 6s — 300s is a generous ceiling.
    tts_request_timeout_secs: float = 300.0

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
