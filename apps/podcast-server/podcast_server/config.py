"""podcast-server configuration.

All env vars are prefixed PODCAST_ (e.g. PODCAST_PORT, PODCAST_ENGINE),
mirroring the VOXTRAL_/BURN_/OCR_ convention of the sibling apps.

The model is pinned by revision: the checkpoint is a key-remapped
conversion of microsoft/VibeVoice-1.5B published by the (unofficial)
VibeVoice community org, which is what transformers' own docs and tests
point at. scripts/verify_provenance.py proves the tensors are
bit-identical to Microsoft's release at MS_REFERENCE_REVISION.
"""

from pathlib import Path

from pydantic_settings import BaseSettings

_APP_DIR = Path(__file__).resolve().parent.parent          # apps/podcast-server

# Microsoft's original checkpoint the served weights are verified against.
MS_REFERENCE_REPO = "microsoft/VibeVoice-1.5B"
MS_REFERENCE_REVISION = "c00898d257e6b46004e3e2866a47534085fb685a"


class Settings(BaseSettings):
    port: int = 8093

    # "vibevoice" (GPU worker process) or "mock" (deterministic tones, no
    # torch — tests/CI and API demos without a GPU).
    engine: str = "vibevoice"

    model_id: str = "vibevoice/VibeVoice-1.5B-hf"
    model_revision: str = "d4cd82eb4371bf3e3d088e1ea7fa4eefb60449bf"
    device: str = "cuda"
    dtype: str = "bfloat16"

    voices_dir: str = str(_APP_DIR / "voices")

    # Lifecycle (same shape as vllm-server's TTS lifecycle: lazy start on the
    # first request, unload after idle so the VRAM goes back to the ASR/TTS
    # apps). The model lives in a worker PROCESS, so an unload frees all of
    # it, CUDA context included. 0 = never idle-unload.
    idle_unload_secs: float = 600.0
    load_timeout_secs: float = 300.0
    synth_timeout_secs: float = 1800.0
    # Refuse to load (state "blocked") below this much free VRAM instead of
    # OOMing a card another starling app is using. 0 = skip the check.
    min_free_vram_gib: float = 8.0

    # Generation defaults (the checkpoint's own generation_config values).
    default_seed: int = 42
    default_cfg_scale: float = 1.3
    default_diffusion_steps: int = 10

    # Request limits.
    max_speakers: int = 4                 # model limit
    max_turns: int = 400
    max_chars_per_turn: int = 3000
    max_total_chars: int = 20000          # ≈ 20+ min of speech

    # Runaway guard for generate(): budget max_new_tokens from the text
    # length. The model emits 7.5 audio tokens per second of speech (one
    # per 3200 samples at 24 kHz); natural speech is ~14 chars/s ≈ 0.54
    # tokens/char, so 1.1/char + 150 (20 s) is ~2x headroom.
    tokens_per_char: float = 1.1
    token_slack: int = 150
    max_new_tokens_cap: int = 40500       # the checkpoint's 90-minute ceiling

    # Mock engine
    mock_delay_secs: float = 0.0

    model_config = {
        "env_prefix": "PODCAST_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


settings = Settings()
