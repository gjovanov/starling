"""Deterministic mock engine for tests/CI — no GPU, no subprocess.

Emits stable Markdown derived from the page filename so unit/e2e tests can
assert on content. Maintains a loaded/idle flag so the manager's load/unload
paths and WS engine events are exercised without real weights.
"""

import asyncio
from pathlib import Path

from .base import OcrEngine

MOCK_PAGE_TEMPLATE = """# Mock OCR — {page_name}

Engine: mock (deterministic test engine)
Language hint: {language_hint}

| Item | Qty | Price |
|------|-----|-------|
| Widget A | 2 | 10.00 |
| Widget B | 1 | 5.50 |

Total: 25.50
"""


class MockEngine(OcrEngine):
    name = "mock"
    version = "0.1"
    kind = "mock"
    vram_class = "light"

    def __init__(self, delay_secs: float = 0.0) -> None:
        self._delay = delay_secs
        self._loaded = False

    def status(self) -> str:
        return "ready" if self._loaded else "idle"

    async def ensure_ready(self) -> None:
        if self._delay:
            await asyncio.sleep(min(self._delay, 0.2))
        self._loaded = True

    async def ocr_page(self, png_path: Path, language_hint: str) -> str:
        if self._delay:
            await asyncio.sleep(self._delay)
        return MOCK_PAGE_TEMPLATE.format(page_name=png_path.name, language_hint=language_hint)

    async def unload(self) -> None:
        self._loaded = False
