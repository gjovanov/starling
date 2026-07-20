"""End-to-end pipeline against the mock engine: upload → OCR → extraction,
with WS events observed via a direct Broadcaster subscription."""

from tests.helpers import (
    LONG_TEXT,
    make_mixed_pdf,
    make_png,
    make_text_pdf,
    upload,
    wait_extraction,
    wait_run,
)


async def test_png_full_pipeline(client, started_state):
    events = started_state.ws.subscribe()

    r = await upload(client, "scan.png", make_png())
    assert r.status_code == 200, r.text
    run_id = r.json()["data"]["documents"][0]["run_id"]

    detail = await wait_run(client, run_id)
    assert detail["run"]["status"] == "done"
    assert detail["run"]["ms_total"] is not None
    assert len(detail["pages"]) == 1
    assert detail["pages"][0]["source"] == "ocr"

    # Page artifacts
    md = await client.get(f"/api/ocr/runs/{run_id}/pages/1.md")
    assert md.status_code == 200
    assert "Mock OCR" in md.text
    png = await client.get(f"/api/ocr/runs/{run_id}/pages/1.png")
    assert png.status_code == 200
    assert png.headers["content-type"] == "image/png"
    concat = await client.get(f"/api/ocr/runs/{run_id}/markdown")
    assert "Mock OCR" in concat.text

    # Auto-extraction with the mock extractor
    ex = await wait_extraction(client, run_id)
    assert ex["status"] == "done"
    assert ex["validation_status"] == "valid"
    ex_detail = await client.get(f"/api/ocr/extractions/{ex['id']}")
    payload = ex_detail.json()["data"]
    assert payload["data"]["document_type"] == "other"
    assert payload["validation"]["overall"] == "valid"

    # WS events: run queued→running(pages)→done, extraction queued→…→done
    seen = []
    while not events.empty():
        seen.append(events.get_nowait())
    started_state.ws.unsubscribe(events)
    run_events = [e for e in seen if e["type"] == "run" and e["id"] == run_id]
    statuses = [e["status"] for e in run_events]
    assert statuses[0] == "queued"
    assert "running" in statuses
    assert statuses[-1] == "done"
    assert run_events[-1]["page"] == {"done": 1, "total": 1}
    ex_events = [e for e in seen if e["type"] == "extraction"]
    assert ex_events and ex_events[-1]["status"] == "done"


async def test_text_layer_pdf_skips_engine(client, started_state):
    r = await upload(client, "digital.pdf", make_text_pdf([LONG_TEXT]), extract="off")
    run_id = r.json()["data"]["documents"][0]["run_id"]
    detail = await wait_run(client, run_id)

    assert detail["run"]["status"] == "done"
    assert detail["pages"][0]["source"] == "text-layer"
    md = await client.get(f"/api/ocr/runs/{run_id}/pages/1.md")
    assert "INVOICE No. 2026-001" in md.text
    # The engine was never loaded — the text-layer fast path skips OCR entirely.
    assert started_state.engines.get("mock").status() == "idle"
    # extract=off → no extraction rows
    assert detail["extractions"] == []


async def test_mixed_pdf_per_page_sources(client):
    r = await upload(client, "mixed.pdf", make_mixed_pdf(), extract="off")
    run_id = r.json()["data"]["documents"][0]["run_id"]
    detail = await wait_run(client, run_id)

    assert detail["run"]["status"] == "done"
    sources = [p["source"] for p in detail["pages"]]
    assert sources == ["text-layer", "ocr"]


async def test_multi_page_progress_rows(client):
    content = make_text_pdf([LONG_TEXT, LONG_TEXT, LONG_TEXT])
    r = await upload(client, "three.pdf", content, extract="off")
    run_id = r.json()["data"]["documents"][0]["run_id"]
    detail = await wait_run(client, run_id)
    assert [p["page_no"] for p in detail["pages"]] == [1, 2, 3]
    assert all(p["ms"] is not None for p in detail["pages"])


async def test_documents_listing_summary(client):
    r = await upload(client, "list-me.pdf", make_text_pdf([LONG_TEXT]))
    run_id = r.json()["data"]["documents"][0]["run_id"]
    await wait_run(client, run_id)
    await wait_extraction(client, run_id)

    listing = await client.get("/api/ocr/documents")
    docs = listing.json()["data"]["documents"]
    assert len(docs) >= 1
    top = docs[0]
    assert top["latest_run"]["id"] == run_id
    assert top["latest_run"]["pages_done"] == 1
    assert top["latest_run"]["latest_extraction"]["status"] == "done"
