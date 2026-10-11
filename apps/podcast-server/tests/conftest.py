import wave

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

VOICE_NAMES = ("michael", "emma", "alex", "sam", "kim")


def write_voice(path, secs: float = 1.0, freq: float = 200.0, rate: int = 24000) -> None:
    t = np.arange(int(secs * rate)) / rate
    pcm = (0.3 * np.sin(2 * np.pi * freq * t) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


@pytest.fixture
def voices_dir(tmp_path, monkeypatch):
    """Hermetic voice registry: five tiny tone 'voices' in a tmp dir."""
    from podcast_server.config import settings

    d = tmp_path / "voices"
    d.mkdir()
    for i, name in enumerate(VOICE_NAMES):
        write_voice(d / f"{name}.wav", freq=120.0 + 40 * i)
    monkeypatch.setattr(settings, "voices_dir", str(d))
    return d


@pytest.fixture
def mock_engine():
    from podcast_server.engine import PodcastEngine
    from podcast_server.worker import MockWorker

    return PodcastEngine(MockWorker, engine_name="mock", model_label="mock", idle_unload_secs=0)


@pytest.fixture
async def client(voices_dir, mock_engine):
    from podcast_server.main import create_app

    app = create_app(engine=mock_engine)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    await mock_engine.shutdown()
