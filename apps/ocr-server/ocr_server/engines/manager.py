"""Engine registry + VRAM swap policy.

- 'light' engines (glm-ocr, tesseract, mock) may coexist with the extractor.
- 'heavy' engines (e.g. qwen3-vl-8b later) never coexist with another heavy
  engine: requesting one unloads the resident one first.
- Every load/unload transition is broadcast as a WS 'engine' event so the
  UI stays honest.
"""

import asyncio
import logging

from ..ws import Broadcaster
from .base import EngineError, OcrEngine

logger = logging.getLogger(__name__)


class EngineManager:
    def __init__(self, broadcaster: Broadcaster) -> None:
        self._bc = broadcaster
        self._engines: dict[str, OcrEngine] = {}
        self._lock = asyncio.Lock()
        self._extract_process = None  # VllmProcess of the stage-2 extractor

    def set_extract_process(self, process) -> None:
        """Heavy OCR engines cannot coexist with the extractor on 24 GiB —
        the swap policy needs a handle to stop it."""
        self._extract_process = process

    def register(self, engine: OcrEngine) -> None:
        self._engines[engine.name] = engine

    def names(self) -> list[str]:
        return list(self._engines.keys())

    def get(self, name: str) -> OcrEngine:
        try:
            return self._engines[name]
        except KeyError:
            raise EngineError(f"unknown engine: {name!r} (registered: {self.names()})") from None

    def statuses(self) -> list[dict]:
        return [
            {
                "name": e.name,
                "version": e.version,
                "kind": e.kind,
                "vram_class": e.vram_class,
                "status": e.status(),
            }
            for e in self._engines.values()
        ]

    async def ensure_ready(self, name: str) -> OcrEngine:
        """Load `name`, unloading any other resident 'heavy' engine first.
        Serialized: one GPU, one boot at a time."""
        async with self._lock:
            engine = self.get(name)
            if engine.status() == "ready":
                engine.note_activity()
                return engine

            if engine.vram_class == "heavy":
                for other in self._engines.values():
                    if other.name != name and other.vram_class == "heavy" \
                            and other.status() != "idle":
                        logger.info("[EngineManager] swap: unloading heavy engine %s for %s",
                                    other.name, name)
                        await other.unload()
                        await self._bc.broadcast(
                            {"type": "engine", "name": other.name, "status": "unloaded"})
                await self._stop_extractor_for(name)

            await self._bc.broadcast({"type": "engine", "name": name, "status": "loading"})
            try:
                await engine.ensure_ready()
            except Exception:
                await self._bc.broadcast({"type": "engine", "name": name, "status": "unloaded"})
                raise
            await self._bc.broadcast({"type": "engine", "name": name, "status": "ready"})
            return engine

    async def _stop_extractor_for(self, engine_name: str) -> None:
        from .vllm_process import ProcState

        if self._extract_process is None:
            return
        if self._extract_process.state in (ProcState.READY, ProcState.STARTING):
            logger.info("[EngineManager] swap: stopping extractor for heavy engine %s",
                        engine_name)
            await self._extract_process.ensure_stopped()
            await self._bc.broadcast(
                {"type": "engine", "name": "extractor", "status": "unloaded"})

    async def unload_heavy_ready(self) -> None:
        """Extractor-side swap: called before the extractor boots so a
        resident heavy OCR engine frees the VRAM first."""
        async with self._lock:
            for engine in self._engines.values():
                if engine.vram_class == "heavy" and engine.status() != "idle":
                    logger.info("[EngineManager] swap: unloading heavy engine %s "
                                "for the extractor", engine.name)
                    await engine.unload()
                    await self._bc.broadcast(
                        {"type": "engine", "name": engine.name, "status": "unloaded"})

    async def unload(self, name: str) -> None:
        async with self._lock:
            engine = self.get(name)
            await engine.unload()
            await self._bc.broadcast({"type": "engine", "name": name, "status": "unloaded"})

    async def unload_all(self) -> None:
        """Shutdown hook: keep the GPU clean for the next boot."""
        async with self._lock:
            for engine in self._engines.values():
                try:
                    await engine.unload()
                except Exception as exc:
                    logger.warning("[EngineManager] unload %s failed: %s", engine.name, exc)
