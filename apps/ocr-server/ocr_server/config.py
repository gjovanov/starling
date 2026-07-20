"""ocr-server configuration.

All env vars are prefixed OCR_ (e.g. OCR_PORT, OCR_ENGINE), mirroring the
VOXTRAL_/BURN_ convention of the sibling apps. Unprefixed shared vars
(STARLING_MODELS_DIR) are mapped by start.sh.
"""

from pathlib import Path

from pydantic_settings import BaseSettings

_APP_DIR = Path(__file__).resolve().parent.parent          # apps/ocr-server
_ROOT_DIR = _APP_DIR.parent.parent                         # starling/


class Settings(BaseSettings):
    port: int = 8092
    frontend_path: str = str(_ROOT_DIR / "frontend")

    # Storage roots. DB stores paths relative to these so the tree can move.
    media_dir: str = str(_ROOT_DIR / "media" / "ocr")
    results_dir: str = str(_ROOT_DIR / "results" / "ocr")
    db_path: str = str(_APP_DIR / "ocr.sqlite3")
    models_dir: str = str(_ROOT_DIR / "models" / "cache")

    # Pipeline defaults
    engine: str = "glm-ocr"                 # default OCR engine; "mock" for tests/CI
    extract_auto: bool = True               # run stage 2 automatically after OCR
    extract_model: str = "qwen3-8b-fp8"     # default extraction model
    language_hint: str = "bg,en"
    raster_dpi: int = 200
    text_layer_min_chars: int = 32          # PDF text-layer fast-path threshold
    max_upload_bytes: int = 100_000_000

    # Engine subprocesses (vLLM)
    vllm_ocr_port: int = 8003
    vllm_extract_port: int = 8004
    # 0 = never idle-unload (keep engines warm — avoids the ~35 s (OCR) / ~66 s
    # (extractor) vLLM cold boot that otherwise gets charged to the next request
    # after an idle gap). Set >0 to release VRAM (~5 GiB OCR / ~13.5 GiB extractor)
    # after that many idle seconds. Both resident + GPU layout ≈ 20.6/24.5 GiB.
    engine_idle_unload_secs: float = 0.0
    extract_idle_unload_secs: float = 0.0
    engine_boot_timeout_secs: float = 300.0
    gpu_util_ocr: float = 0.20
    # 0.45 OOMs: Qwen3-8B-FP8 weights (~9.2 GiB) leave <1 GiB KV. 0.55 gives
    # ~3.2 GiB KV (16k ctx needs 2.25). Combined with OCR's 0.20 → 0.75 total.
    gpu_util_extract: float = 0.55
    engine_autostart: bool = True           # lazy-spawn vLLM subprocesses on demand
    # Pre-warm the default OCR engine at startup (background, non-blocking) so
    # the first upload skips the cold boot. Only fires for GPU (vllm-subprocess)
    # engines — mock/tesseract have no boot cost.
    engine_prewarm: bool = True
    # PP-DocLayout-V3 layout stage device: 'cuda' ≈ 0.87 s/page vs 'cpu' ≈ 1.4 s
    # (measured, 8-region invoice). 'cuda' costs ~1.5 GiB VRAM; 'cpu' reserves
    # the GPU for the OCR model + extractor coexistence.
    layout_device: str = "cuda"

    # Stage 2 request knobs
    extract_max_tokens: int = 4096
    extract_temperature: float = 0.1
    extract_timeout_secs: float = 300.0
    ocr_page_timeout_secs: float = 120.0

    # tesseract CPU baseline (Phase 4); empty = auto-resolve (PATH, then the
    # user-space ~/.local/tesseract install from prerequisites.sh)
    tesseract_bin: str = ""

    # Mock engine (tests / demo without GPU)
    mock_delay_secs: float = 0.15

    model_config = {
        "env_prefix": "OCR_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


settings = Settings()
