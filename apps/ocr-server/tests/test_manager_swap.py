"""VRAM swap policy: heavy engines never coexist — with each other or with
the extractor. Verified with stubbed heavy engines (no GPU)."""

import asyncio
from pathlib import Path

from ocr_server.engines.base import OcrEngine
from ocr_server.engines.manager import EngineManager
from ocr_server.engines.vllm_process import Probe, ProcState, VllmProcess
from ocr_server.ws import Broadcaster


class FakeHeavyEngine(OcrEngine):
    kind = "vllm-subprocess"
    vram_class = "heavy"
    version = "fake"

    def __init__(self, name: str) -> None:
        self.name = name
        self.loaded = False
        self.load_count = 0
        self.unload_count = 0

    def status(self) -> str:
        return "ready" if self.loaded else "idle"

    async def ensure_ready(self) -> None:
        self.loaded = True
        self.load_count += 1

    async def ocr_page(self, png_path: Path, language_hint: str) -> str:
        return "fake"

    async def unload(self) -> None:
        self.loaded = False
        self.unload_count += 1


def _ready_extract_process() -> VllmProcess:
    async def healthy():
        return True

    proc = VllmProcess(name="extract", start_command=["true"],
                       base_url="http://127.0.0.1:9",
                       probe=Probe(health_check=healthy,
                                   sleep=lambda s: asyncio.sleep(0)))
    return proc


async def test_heavy_swap_unloads_other_heavy_and_extractor():
    bc = Broadcaster()
    events = bc.subscribe()
    mgr = EngineManager(bc)
    a, b = FakeHeavyEngine("heavy-a"), FakeHeavyEngine("heavy-b")
    mgr.register(a)
    mgr.register(b)

    extract = _ready_extract_process()
    await extract.ensure_started()
    assert extract.state is ProcState.READY
    mgr.set_extract_process(extract)

    await mgr.ensure_ready("heavy-a")
    assert a.loaded
    # Loading A stopped the extractor (heavy + extractor never coexist).
    assert extract.state is ProcState.IDLE

    await extract.ensure_started()  # extractor comes back (simulated)
    await mgr.ensure_ready("heavy-b")
    assert b.loaded
    assert not a.loaded, "heavy-a must be unloaded when heavy-b is requested"
    assert extract.state is ProcState.IDLE

    seen = []
    while not events.empty():
        seen.append(events.get_nowait())
    engine_events = [(e["name"], e["status"]) for e in seen if e["type"] == "engine"]
    assert ("extractor", "unloaded") in engine_events
    assert ("heavy-a", "unloaded") in engine_events


async def test_light_engine_does_not_touch_extractor_or_heavies():
    bc = Broadcaster()
    mgr = EngineManager(bc)
    heavy = FakeHeavyEngine("heavy-a")
    heavy.loaded = True

    class LightEngine(FakeHeavyEngine):
        vram_class = "light"

    light = LightEngine("light-a")
    mgr.register(heavy)
    mgr.register(light)

    extract = _ready_extract_process()
    await extract.ensure_started()
    mgr.set_extract_process(extract)

    await mgr.ensure_ready("light-a")
    assert light.loaded
    assert heavy.loaded, "light engines must not evict heavy ones"
    assert extract.state is ProcState.READY


async def test_unload_heavy_ready_for_extractor_boot():
    bc = Broadcaster()
    mgr = EngineManager(bc)
    heavy = FakeHeavyEngine("heavy-a")
    heavy.loaded = True
    light = FakeHeavyEngine("light-a")
    light.vram_class = "light"
    light.loaded = True
    mgr.register(heavy)
    mgr.register(light)

    await mgr.unload_heavy_ready()
    assert not heavy.loaded
    assert light.loaded, "extractor-side swap only evicts heavy engines"
