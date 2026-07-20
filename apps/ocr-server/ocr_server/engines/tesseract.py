"""tesseract-bul — CPU baseline engine (zero VRAM, character-level sanity).

Runs the tesseract CLI per page with languages mapped from the run's
language hint (bg,en → bul+eng). Binary resolution order:
  1. settings.tesseract_bin (OCR_TESSERACT_BIN)
  2. `tesseract` on PATH
  3. ~/.local/tesseract/usr/bin/tesseract — the user-space dpkg -x install
     documented in prerequisites.sh (needs LD_LIBRARY_PATH + TESSDATA_PREFIX,
     set automatically here).
"""

import asyncio
import logging
import os
import shutil
from pathlib import Path

from ..config import Settings
from .base import EngineError, OcrEngine

logger = logging.getLogger(__name__)

_LOCAL_PREFIX = Path.home() / ".local" / "tesseract"

# ISO 639-1 hint codes → tesseract traineddata names
_LANG_MAP = {
    "bg": "bul", "en": "eng", "de": "deu", "fr": "fra", "es": "spa",
    "it": "ita", "ru": "rus", "ro": "ron", "el": "ell", "tr": "tur",
    "pt": "por", "nl": "nld", "pl": "pol", "cs": "ces", "sr": "srp",
    "mk": "mkd", "uk": "ukr",
}


def resolve_tesseract(settings: Settings) -> tuple[str, dict[str, str]] | None:
    """Return (binary, extra_env) or None when tesseract is unavailable."""
    if settings.tesseract_bin:
        return settings.tesseract_bin, {}
    on_path = shutil.which("tesseract")
    if on_path:
        return on_path, {}
    local_bin = _LOCAL_PREFIX / "usr" / "bin" / "tesseract"
    if local_bin.is_file():
        return str(local_bin), {
            "LD_LIBRARY_PATH": str(_LOCAL_PREFIX / "usr" / "lib" / "x86_64-linux-gnu"),
            "TESSDATA_PREFIX": str(_LOCAL_PREFIX / "usr" / "share" / "tesseract-ocr"
                                   / "5" / "tessdata"),
        }
    return None


def hint_to_langs(language_hint: str) -> str:
    langs = []
    for code in (c.strip().lower() for c in language_hint.split(",")):
        if code in _LANG_MAP:
            langs.append(_LANG_MAP[code])
        elif len(code) == 3 and code.isalpha():
            langs.append(code)  # already a tesseract code
    return "+".join(dict.fromkeys(langs)) or "eng"


class TesseractEngine(OcrEngine):
    name = "tesseract-bul"
    kind = "cli"
    vram_class = "light"
    version = "unknown"

    def __init__(self, binary: str, extra_env: dict[str, str],
                 timeout_secs: float = 120.0) -> None:
        self._binary = binary
        self._env = {**os.environ, **extra_env}
        self._timeout = timeout_secs
        self._checked = False

    def status(self) -> str:
        return "ready" if self._checked else "idle"

    async def ensure_ready(self) -> None:
        if self._checked:
            return
        proc = await asyncio.create_subprocess_exec(
            self._binary, "--version",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            env=self._env)
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=15.0)
        if proc.returncode != 0:
            raise EngineError(f"tesseract --version failed: {out.decode()[:200]}")
        first = out.decode().splitlines()[0].strip() if out else ""
        if first:
            self.version = first
        self._checked = True

    async def ocr_page(self, png_path: Path, language_hint: str) -> str:
        await self.ensure_ready()
        langs = hint_to_langs(language_hint)
        proc = await asyncio.create_subprocess_exec(
            self._binary, str(png_path), "-", "-l", langs,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            env=self._env)
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=self._timeout)
        except asyncio.TimeoutError:
            proc.kill()
            raise EngineError(f"tesseract timed out on {png_path.name}") from None
        if proc.returncode != 0:
            raise EngineError(
                f"tesseract failed on {png_path.name}: {err.decode()[:300]}")
        # Plain text is valid Markdown; normalize trailing whitespace only.
        return out.decode("utf-8", errors="replace").strip() + "\n"

    async def unload(self) -> None:
        self._checked = False
