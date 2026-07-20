"""Run detail / extraction rerun / error paths."""

from tests.helpers import LONG_TEXT, make_text_pdf, upload, wait_extraction, wait_run


async def test_run_not_found(client):
    r = await client.get("/api/ocr/runs/9999")
    assert r.status_code == 404


async def test_extraction_not_found(client):
    r = await client.get("/api/ocr/extractions/9999")
    assert r.status_code == 404


async def test_document_runs_not_found(client):
    r = await client.get("/api/ocr/documents/9999/runs")
    assert r.status_code == 404


async def test_extract_rerun(client):
    r = await upload(client, "rerun.pdf", make_text_pdf([LONG_TEXT]))
    run_id = r.json()["data"]["documents"][0]["run_id"]
    await wait_run(client, run_id)
    first = await wait_extraction(client, run_id)

    r2 = await client.post(f"/api/ocr/runs/{run_id}/extract", json={"model": "mock-extract"})
    assert r2.status_code == 200
    second_id = r2.json()["data"]["id"]
    assert second_id != first["id"]
    second = await wait_extraction(client, run_id)
    assert second["id"] == second_id
    assert second["status"] == "done"


async def test_extract_on_unfinished_run_is_409(client, started_state):
    # Seed a queued run directly — never picked up because the doc is fake.
    doc = await started_state.db.insert_document(
        filename="seed.pdf", sha256="0" * 64, mime="application/pdf", page_count=1)
    run = await started_state.db.create_run(document_id=doc["id"], engine="mock")
    # Freeze it out of the queue so the state stays 'queued' for this test.
    await started_state.db.conn.execute(
        "UPDATE run SET status='running' WHERE id=?", (run["id"],))
    await started_state.db.conn.commit()

    r = await client.post(f"/api/ocr/runs/{run['id']}/extract", json={})
    assert r.status_code == 409


async def test_unknown_extract_model_fails_extraction(client):
    r = await upload(client, "badmodel.pdf", make_text_pdf([LONG_TEXT]),
                     extract="auto", extract_model="no-such-model")
    run_id = r.json()["data"]["documents"][0]["run_id"]
    await wait_run(client, run_id)
    ex = await wait_extraction(client, run_id)
    assert ex["status"] == "failed"
    assert "unknown extraction model" in ex["error"]


async def test_bakeoff_rerun_endpoint(client):
    r = await upload(client, "bake.pdf", make_text_pdf([LONG_TEXT]), extract="off")
    doc_id = r.json()["data"]["documents"][0]["document_id"]
    run1 = r.json()["data"]["documents"][0]["run_id"]
    await wait_run(client, run1)

    r2 = await client.post(f"/api/ocr/documents/{doc_id}/runs", json={"engine": "mock"})
    assert r2.status_code == 200
    run2 = r2.json()["data"]["id"]
    assert run2 != run1
    detail = await wait_run(client, run2)
    assert detail["run"]["status"] == "done"

    runs = await client.get(f"/api/ocr/documents/{doc_id}/runs")
    assert len(runs.json()["data"]["runs"]) == 2
