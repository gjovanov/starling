"""Upload path: mime sniffing, 415/400/413, sha256 dedupe."""

from ocr_server.ingest import detect_mime
from tests.helpers import make_png, make_text_pdf, upload, LONG_TEXT


def test_detect_mime_magic_bytes():
    assert detect_mime(b"%PDF-1.7 rest") == "application/pdf"
    assert detect_mime(b"\x89PNG\r\n\x1a\nrest") == "image/png"
    assert detect_mime(b"\xff\xd8\xff\xe0rest") == "image/jpeg"
    assert detect_mime(b"plain text") is None
    assert detect_mime(b"") is None


async def test_unsupported_type_is_415(client):
    r = await upload(client, "notes.txt", b"just text")
    assert r.status_code == 415


async def test_corrupt_pdf_is_400(client):
    r = await upload(client, "broken.pdf", b"%PDF-1.4\nnot really a pdf")
    assert r.status_code == 400
    assert "corrupt" in r.json()["detail"]


async def test_too_large_is_413(client, ocr_settings, monkeypatch):
    monkeypatch.setattr(ocr_settings, "max_upload_bytes", 10)
    r = await upload(client, "big.png", make_png())
    assert r.status_code == 413


async def test_empty_file_is_400(client):
    r = await upload(client, "empty.pdf", b"")
    assert r.status_code == 400


async def test_sha_dedupe_reuses_document(client):
    content = make_text_pdf([LONG_TEXT])
    r1 = await upload(client, "a.pdf", content)
    r2 = await upload(client, "same-bytes.pdf", content)
    d1 = r1.json()["data"]["documents"][0]
    d2 = r2.json()["data"]["documents"][0]
    assert d1["document_id"] == d2["document_id"]
    assert d1["run_id"] != d2["run_id"]          # each upload gets a fresh run
    assert d1["deduped"] is False
    assert d2["deduped"] is True


async def test_multi_file_partial_failure(client):
    files = [
        ("files", ("good.pdf", make_text_pdf([LONG_TEXT]), "application/pdf")),
        ("files", ("bad.txt", b"nope", "text/plain")),
    ]
    r = await client.post("/api/ocr/documents", files=files)
    assert r.status_code == 200
    data = r.json()["data"]
    assert len(data["documents"]) == 1
    assert len(data["errors"]) == 1
    assert data["errors"][0]["filename"] == "bad.txt"
    assert data["errors"][0]["status"] == 415


async def test_unknown_engine_is_400(client):
    r = await upload(client, "x.png", make_png(), engine="does-not-exist")
    assert r.status_code == 400
    assert "unknown engine" in r.json()["detail"]


async def test_invalid_extract_param_is_400(client):
    r = await upload(client, "x.png", make_png(), extract="maybe")
    assert r.status_code == 400
