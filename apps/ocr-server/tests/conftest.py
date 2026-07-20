import pytest
from httpx import ASGITransport, AsyncClient


@pytest.fixture
def ocr_settings(tmp_path, monkeypatch):
    """Point every storage root at tmp dirs and select the mock stack."""
    from ocr_server.config import settings

    monkeypatch.setattr(settings, "media_dir", str(tmp_path / "media"))
    monkeypatch.setattr(settings, "results_dir", str(tmp_path / "results"))
    monkeypatch.setattr(settings, "db_path", str(tmp_path / "ocr.sqlite3"))
    monkeypatch.setattr(settings, "engine", "mock")
    monkeypatch.setattr(settings, "extract_model", "mock-extract")
    monkeypatch.setattr(settings, "extract_auto", True)
    monkeypatch.setattr(settings, "mock_delay_secs", 0.0)
    return settings


@pytest.fixture
async def started_state(ocr_settings):
    """The module singleton, started against tmp dirs; torn down after."""
    from ocr_server.state import app_state

    await app_state.startup()
    yield app_state
    await app_state.shutdown()


@pytest.fixture
async def client(started_state):
    from ocr_server.main import create_app

    app = create_app(serve_frontend=False)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
