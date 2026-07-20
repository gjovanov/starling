"""OCR engine interface.

Engines convert a rasterized page image into Markdown. GPU-backed engines
are lazy-loaded by the EngineManager which also enforces the VRAM swap
policy ('heavy' engines never coexist).
"""

import abc
from pathlib import Path


class EngineError(Exception):
    """Raised when an engine cannot load or fails to OCR a page."""


class OcrEngine(abc.ABC):
    name: str
    version: str
    kind: str            # 'vllm-subprocess' | 'cli' | 'http' | 'mock'
    vram_class: str      # 'light' | 'heavy' — manager swap policy

    @abc.abstractmethod
    async def ensure_ready(self) -> None:
        """Block until the engine can serve ocr_page calls (spawn/load lazily)."""

    @abc.abstractmethod
    async def ocr_page(self, png_path: Path, language_hint: str) -> str:
        """OCR one page image; returns Markdown."""

    @abc.abstractmethod
    async def unload(self) -> None:
        """Release GPU/process resources. ensure_ready() may be called again."""

    def status(self) -> str:
        """'ready' | 'idle' | 'loading' — for /api/ocr/engines and the UI."""
        return "ready"

    def note_activity(self) -> None:
        """Reset the idle-unload timer (no-op for engines without one)."""
