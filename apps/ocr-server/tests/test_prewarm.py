"""Pre-warm at startup: fires for GPU (vllm-subprocess) engines, skips
mock/tesseract; idle-unload=0 keeps the process resident."""

import asyncio

import pytest

from ocr_server.engines.base import OcrEngine
from ocr_server.engines.vllm_process import Probe, ProcState, VllmProcess


async def test_mock_default_is_not_prewarmed(started_state):
    # conftest sets OCR_ENGINE=mock; mock has no boot cost, so pre-warm must
    # skip it and leave it idle (guards test_text_layer_pdf_skips_engine).
    assert started_state.prewarm_task is None
    assert started_state.engines.get("mock").status() == "idle"


async def test_prewarm_fires_for_vllm_subprocess_engine(ocr_settings, monkeypatch):
    """A vllm-subprocess-kind default engine gets ensure_ready()'d in the
    background at startup."""
    loaded = asyncio.Event()

    class FakeGpuEngine(OcrEngine):
        name = "fake-gpu"
        version = "1"
        kind = "vllm-subprocess"
        vram_class = "light"

        def __init__(self):
            self._ready = False

        def status(self):
            return "ready" if self._ready else "idle"

        async def ensure_ready(self):
            self._ready = True
            loaded.set()

        async def ocr_page(self, png_path, language_hint):
            return "x"

        async def unload(self):
            self._ready = False

    monkeypatch.setattr(ocr_settings, "engine", "fake-gpu")
    from ocr_server.state import OcrAppState

    state = OcrAppState()
    # Register our fake as the sole engine.
    orig = OcrAppState._register_engines

    def only_fake(self):
        self.engines.register(FakeGpuEngine())

    monkeypatch.setattr(OcrAppState, "_register_engines", only_fake)
    await state.startup()
    try:
        assert state.prewarm_task is not None
        await asyncio.wait_for(loaded.wait(), timeout=2.0)
        assert state.engines.get("fake-gpu").status() == "ready"
    finally:
        monkeypatch.setattr(OcrAppState, "_register_engines", orig)
        await state.shutdown()


async def test_idle_unload_zero_never_schedules():
    async def healthy():
        return True

    proc = VllmProcess(
        name="warm", start_command=["true"], base_url="http://127.0.0.1:9",
        idle_unload_secs=0.0,
        probe=Probe(health_check=healthy, sleep=lambda s: asyncio.sleep(0)))
    await proc.ensure_started()
    assert proc.state is ProcState.READY
    # idle watcher disabled → no task scheduled, engine stays resident.
    assert proc._idle_task is None
    proc._last_activity_at -= 10_000
    for _ in range(20):
        await asyncio.sleep(0)
    assert proc.state is ProcState.READY
    await proc.ensure_stopped()
