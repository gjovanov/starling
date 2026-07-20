"""Process-wide application state (mirrors voxtral_server.state.app_state).

Routes import the module-level singleton; main.py drives startup/shutdown
via FastAPI events; tests drive them directly with tmp-dir settings.
"""

import asyncio
import logging
from pathlib import Path

from .config import settings
from .db import Database
from .engines.manager import EngineManager
from .engines.mock import MockEngine
from .engines.vllm_process import VllmProcess
from .extraction.extractor import create_extractor
from .queue import JobQueue
from .ws import Broadcaster

logger = logging.getLogger(__name__)

_APP_DIR = Path(__file__).resolve().parent.parent  # apps/ocr-server


class OcrAppState:
    def __init__(self) -> None:
        self.ws = Broadcaster()
        self.db: Database | None = None
        self.engines: EngineManager | None = None
        self.queue: JobQueue | None = None
        self.extract_process: VllmProcess | None = None
        self.prewarm_task: asyncio.Task | None = None
        self.started = False

    async def startup(self) -> None:
        if self.started:
            return
        Path(settings.media_dir).mkdir(parents=True, exist_ok=True)
        Path(settings.results_dir).mkdir(parents=True, exist_ok=True)
        self.db = Database(settings.db_path)
        await self.db.connect()
        self.engines = EngineManager(self.ws)
        self._register_engines()
        if settings.engine not in self.engines.names():
            logger.warning("[State] default engine %r is not registered (available: %s)",
                           settings.engine, self.engines.names())

        self.extract_process = VllmProcess(
            name="extract",
            start_command=["bash", str(_APP_DIR / "start-vllm-extract.sh")],
            cwd=str(_APP_DIR),
            base_url=f"http://127.0.0.1:{settings.vllm_extract_port}",
            boot_timeout_secs=settings.engine_boot_timeout_secs,
            idle_unload_secs=settings.extract_idle_unload_secs,
        )

        async def _extract_idle_unload() -> None:
            await self.extract_process.ensure_stopped()
            await self.ws.broadcast(
                {"type": "engine", "name": "extractor", "status": "unloaded"})

        self.extract_process.set_idle_unload_callback(_extract_idle_unload)
        self.engines.set_extract_process(self.extract_process)

        self.queue = JobQueue(db=self.db, ws=self.ws, engines=self.engines,
                              settings=settings,
                              extractor_factory=self._make_extractor)
        await self.queue.start()

        # Pre-warm the default OCR engine so the first upload skips the ~35 s
        # vLLM cold boot. Non-blocking: the server is up immediately and the
        # engine loads in the background (WS 'engine' events report progress).
        # Only GPU engines pay a boot cost — mock/tesseract are skipped so
        # tests and CPU-only runs stay instant.
        if settings.engine_prewarm and settings.engine in self.engines.names():
            default = self.engines.get(settings.engine)
            if default.kind == "vllm-subprocess":
                self.prewarm_task = asyncio.create_task(self._prewarm(settings.engine))

        self.started = True

    async def _prewarm(self, name: str) -> None:
        try:
            await self.engines.ensure_ready(name)
            logger.info("[State] pre-warmed OCR engine %r", name)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("[State] pre-warm of %r failed (will lazy-load on "
                           "demand): %s", name, exc)

    def _make_extractor(self, model: str, settings_):
        return create_extractor(model, settings_, process=self.extract_process,
                                broadcaster=self.ws,
                                pre_start=self.engines.unload_heavy_ready)

    def _register_engines(self) -> None:
        assert self.engines is not None
        self.engines.register(MockEngine(delay_secs=settings.mock_delay_secs))

        from .engines.glm_ocr import GlmOcrEngine

        glm = GlmOcrEngine(settings)
        self.engines.register(glm)

        # The engine's own idle timer bypasses the manager — broadcast the
        # WS 'engine' event from the unload callback so the UI stays honest.
        async def _glm_idle_unload() -> None:
            await glm.unload()
            await self.ws.broadcast(
                {"type": "engine", "name": glm.name, "status": "unloaded"})

        glm.process.set_idle_unload_callback(_glm_idle_unload)

        from .engines.tesseract import TesseractEngine, resolve_tesseract

        resolved = resolve_tesseract(settings)
        if resolved is not None:
            binary, extra_env = resolved
            self.engines.register(TesseractEngine(
                binary, extra_env, timeout_secs=settings.ocr_page_timeout_secs))
        else:
            logger.info("[State] tesseract not found — CPU baseline engine "
                        "not registered (see prerequisites.sh)")

    async def shutdown(self) -> None:
        if not self.started:
            return
        if self.prewarm_task and not self.prewarm_task.done():
            self.prewarm_task.cancel()
            try:
                await self.prewarm_task
            except (asyncio.CancelledError, Exception):
                pass
        self.prewarm_task = None
        if self.queue:
            await self.queue.stop()
        if self.engines:
            await self.engines.unload_all()
        if self.extract_process:
            await self.extract_process.ensure_stopped()
        if self.db:
            await self.db.close()
        self.queue = None
        self.engines = None
        self.extract_process = None
        self.db = None
        self.started = False


app_state = OcrAppState()
