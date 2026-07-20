"""tesseract-bul engine tests — language mapping + full pipeline via a
stubbed tesseract CLI (no real binary needed)."""

import stat

from ocr_server.engines.tesseract import hint_to_langs, resolve_tesseract
from tests.helpers import make_image_only_pdf, upload, wait_run

FAKE_TESSERACT = """#!/bin/sh
if [ "$1" = "--version" ]; then
    echo "tesseract 9.9-fake"
    exit 0
fi
# args: <input> - -l <langs>
echo "FAKE-TESSERACT input=$(basename "$1") langs=$4"
"""


def test_hint_to_langs():
    assert hint_to_langs("bg,en") == "bul+eng"
    assert hint_to_langs("en") == "eng"
    assert hint_to_langs("de, fr") == "deu+fra"
    assert hint_to_langs("bul,en") == "bul+eng"     # tesseract code passthrough
    assert hint_to_langs("xx,??") == "eng"          # unknowns fall back
    assert hint_to_langs("bg,bg,en") == "bul+eng"   # deduped


def test_resolve_prefers_configured_binary(ocr_settings, monkeypatch):
    monkeypatch.setattr(ocr_settings, "tesseract_bin", "/opt/custom/tesseract")
    binary, env = resolve_tesseract(ocr_settings)
    assert binary == "/opt/custom/tesseract"
    assert env == {}


async def test_pipeline_with_stubbed_tesseract(tmp_path, ocr_settings, monkeypatch):
    stub = tmp_path / "tesseract"
    stub.write_text(FAKE_TESSERACT)
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setattr(ocr_settings, "tesseract_bin", str(stub))

    from httpx import ASGITransport, AsyncClient

    from ocr_server.main import create_app
    from ocr_server.state import app_state

    await app_state.startup()
    try:
        assert "tesseract-bul" in app_state.engines.names()
        app = create_app(serve_frontend=False)
        async with AsyncClient(transport=ASGITransport(app=app),
                               base_url="http://test") as client:
            r = await upload(client, "scan.pdf", make_image_only_pdf(),
                             engine="tesseract-bul", extract="off",
                             language_hint="bg,en")
            assert r.status_code == 200, r.text
            run_id = r.json()["data"]["documents"][0]["run_id"]
            detail = await wait_run(client, run_id)
            assert detail["run"]["status"] == "done"
            assert detail["run"]["engine_version"] == "tesseract 9.9-fake"
            assert detail["pages"][0]["source"] == "ocr"
            md = await client.get(f"/api/ocr/runs/{run_id}/pages/1.md")
            assert "FAKE-TESSERACT input=page-001.png langs=bul+eng" in md.text
    finally:
        await app_state.shutdown()
