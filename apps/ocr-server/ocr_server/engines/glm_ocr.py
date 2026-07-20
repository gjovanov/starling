"""GLM-OCR engine: lazy vLLM subprocess (:8003) + glmocr SDK client.

The SDK runs the canonical GLM-OCR pipeline: PP-DocLayout-V3 layout
detection (CPU, transformers port), parallel region recognition against our
vLLM server, and markdown assembly. The vLLM process is spawned on first
use and idle-unloaded via VllmProcess.
"""

import asyncio
import logging
from pathlib import Path

from ..config import Settings
from .base import EngineError, OcrEngine
from .vllm_process import Probe, ProcState, VllmProcess

logger = logging.getLogger(__name__)

_APP_DIR = Path(__file__).resolve().parent.parent.parent  # apps/ocr-server


class GlmOcrEngine(OcrEngine):
    name = "glm-ocr"
    version = "zai-org/GLM-OCR"
    kind = "vllm-subprocess"
    vram_class = "light"

    def __init__(self, settings: Settings, *, probe: Probe | None = None) -> None:
        self._settings = settings
        self.process = VllmProcess(
            name="glm-ocr",
            start_command=["bash", str(_APP_DIR / "start-vllm-ocr.sh")],
            cwd=str(_APP_DIR),
            base_url=f"http://127.0.0.1:{settings.vllm_ocr_port}",
            boot_timeout_secs=settings.engine_boot_timeout_secs,
            idle_unload_secs=settings.engine_idle_unload_secs,
            probe=probe,
        )
        self._parser = None

    def status(self) -> str:
        if self.process.state is ProcState.READY:
            return "ready"
        if self.process.state is ProcState.STARTING:
            return "loading"
        return "idle"

    def note_activity(self) -> None:
        self.process.note_activity()

    async def ensure_ready(self) -> None:
        await self.process.ensure_started()
        if self._parser is None:
            # Loads the CPU layout model (~1 GB RAM) + SDK worker pool.
            self._parser = await asyncio.to_thread(self._make_parser)

    def _make_parser(self):
        from glmocr import GlmOcr

        # No config_path: passing one REPLACES the SDK's bundled config.yaml
        # wholesale (losing label_task_mapping, prompts, id2label, …). The
        # bundled defaults are correct for vLLM (model name 'glm-ocr',
        # /v1/chat/completions); the constructor overrides cover the rest.
        # NOTE: GlmOcr() probes the OCR server during construction — the vLLM
        # process must already be healthy (ensure_ready orders this).
        return GlmOcr(
            mode="selfhosted",
            ocr_api_host="127.0.0.1",
            ocr_api_port=self._settings.vllm_ocr_port,
            layout_device=self._settings.layout_device,
            log_level="WARNING",
        )

    async def ocr_page(self, png_path: Path, language_hint: str) -> str:
        # GLM-OCR takes no language hint — recognition is multilingual; the
        # hint is used by stage 2 and by engines that do accept one.
        if self._parser is None:
            raise EngineError("glm-ocr engine not ready (ensure_ready not called)")
        try:
            result = await asyncio.to_thread(
                self._parser.parse, str(png_path),
                save_layout_visualization=False,
            )
            if isinstance(result, list):
                result = result[0]
            md = getattr(result, "markdown_result", None)
            if md is None:
                d = result.to_dict()
                md = d.get("markdown_result") or d.get("markdown") or ""
        except EngineError:
            raise
        except Exception as exc:
            raise EngineError(f"glm-ocr failed on {png_path.name}: {exc}") from exc

        if not (md or "").strip():
            # The SDK swallows per-region request failures and returns an
            # empty result when the upstream dies mid-parse — distinguish a
            # genuinely blank page from a dead server.
            if not await self.process.is_healthy():
                raise EngineError(
                    f"glm-ocr upstream unreachable while parsing {png_path.name} "
                    "(empty result, health check failed)")
        return md or ""

    async def unload(self) -> None:
        if self._parser is not None:
            try:
                await asyncio.to_thread(self._parser.close)
            except Exception as exc:
                logger.warning("[glm-ocr] parser close failed: %s", exc)
            self._parser = None
        await self.process.ensure_stopped()
