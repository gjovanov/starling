"""Document deletion: cascade DB rows + on-disk files + WS event."""

from pathlib import Path

from tests.helpers import LONG_TEXT, make_text_pdf, upload, wait_extraction, wait_run


async def test_delete_cascades_db_and_files_and_broadcasts(client, started_state):
    events = started_state.ws.subscribe()
    r = await upload(client, "del-me.pdf", make_text_pdf([LONG_TEXT]))
    doc_id = r.json()["data"]["documents"][0]["document_id"]
    run_id = r.json()["data"]["documents"][0]["run_id"]
    await wait_run(client, run_id)
    await wait_extraction(client, run_id)

    from ocr_server.config import settings
    media_dir = Path(settings.media_dir) / str(doc_id)
    results_dir = Path(settings.results_dir) / str(run_id)
    assert media_dir.exists()
    assert results_dir.exists()

    resp = await client.delete(f"/api/ocr/documents/{doc_id}")
    assert resp.status_code == 200
    assert resp.json()["data"]["deleted"] == doc_id
    assert run_id in resp.json()["data"]["runs"]

    # DB rows gone
    assert await started_state.db.get_document(doc_id) is None
    assert await started_state.db.get_run(run_id) is None
    listing = await client.get("/api/ocr/documents")
    assert all(d["id"] != doc_id for d in listing.json()["data"]["documents"])

    # files gone
    assert not media_dir.exists()
    assert not results_dir.exists()

    # WS 'document deleted' event
    seen = []
    while not events.empty():
        seen.append(events.get_nowait())
    started_state.ws.unsubscribe(events)
    assert any(e["type"] == "document" and e["status"] == "deleted" and e["id"] == doc_id
               for e in seen)


async def test_delete_missing_is_404(client):
    assert (await client.delete("/api/ocr/documents/9999")).status_code == 404


async def test_delete_rejected_while_running(client, started_state):
    doc = await started_state.db.insert_document(
        filename="busy.pdf", sha256="a" * 64, mime="application/pdf", page_count=1)
    run = await started_state.db.create_run(document_id=doc["id"], engine="mock")
    await started_state.db.conn.execute(
        "UPDATE run SET status='running' WHERE id=?", (run["id"],))
    await started_state.db.conn.commit()
    r = await client.delete(f"/api/ocr/documents/{doc['id']}")
    assert r.status_code == 409


async def test_extraction_ms_in_ws_event(client, started_state):
    events = started_state.ws.subscribe()
    r = await upload(client, "ms.pdf", make_text_pdf([LONG_TEXT]))
    run_id = r.json()["data"]["documents"][0]["run_id"]
    await wait_run(client, run_id)
    await wait_extraction(client, run_id)

    seen = []
    while not events.empty():
        seen.append(events.get_nowait())
    started_state.ws.unsubscribe(events)
    done = [e for e in seen if e["type"] == "extraction" and e["status"] == "done"]
    assert done and done[-1].get("ms") is not None and done[-1]["ms"] >= 0
