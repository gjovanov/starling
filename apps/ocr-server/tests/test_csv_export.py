"""CSV export: pure function structure + the /csv endpoint."""

import csv
import io

from ocr_server.extraction.csv_export import extraction_to_csv
from tests.helpers import LONG_TEXT, make_text_pdf, upload, wait_extraction, wait_run

SAMPLE = {
    "document_type": "invoice",
    "document_number": "INV-2026-0451",
    "issue_date": "2026-06-20",
    "due_date": "2026-07-04",
    "currency": "EUR",
    "supplier": {"name": "Starling Software, Ltd", "address": "12 Harbour Rd, Dublin",
                 "vat_id": "IE6388047V", "company_id": "IE0334512", "iban": "IE29AIBK…"},
    "customer": {"name": "Acme Trading GmbH", "address": "Hauptstr. 5, Berlin",
                 "vat_id": "DE811907980", "company_id": None, "iban": None},
    "line_items": [
        {"description": "Dev services, senior", "quantity": 80, "unit": "h",
         "unit_price": 95.0, "vat_rate": 23, "total": 7600.0},
        {"description": "Cloud hosting (June)", "quantity": 1, "unit": None,
         "unit_price": 400.0, "vat_rate": 23, "total": 400.0},
    ],
    "net_total": 8000.0,
    "vat_total": 1840.0,
    "gross_total": 9840.0,
    "payment_terms": "14 days",
    "notes": None,
}


def test_csv_structure_parses_and_carries_meta_and_items():
    text = extraction_to_csv(SAMPLE)
    rows = list(csv.reader(io.StringIO(text)))
    flat = ["|".join(r) for r in rows]

    assert ["INVOICE METADATA"] in rows
    assert ["Document number", "INV-2026-0451"] in rows
    assert ["Currency", "EUR"] in rows
    assert ["Net total", "8000.0"] in rows
    assert ["VAT total", "1840.0"] in rows
    assert ["Gross total", "9840.0"] in rows

    assert ["SUPPLIER"] in rows
    assert ["Name", "Starling Software, Ltd"] in rows   # comma survives quoting
    assert ["VAT ID", "IE6388047V"] in rows
    assert ["CUSTOMER"] in rows
    assert ["Name", "Acme Trading GmbH"] in rows

    assert ["LINE ITEMS"] in rows
    assert ["#", "Description", "Quantity", "Unit", "Unit price", "VAT %", "Line total"] in rows
    assert ["1", "Dev services, senior", "80", "h", "95.0", "23", "7600.0"] in rows
    assert ["2", "Cloud hosting (June)", "1", "", "400.0", "23", "400.0"] in rows
    # None → empty cell, never the literal "None"
    assert "None" not in text


def test_csv_handles_empty_line_items():
    data = {**SAMPLE, "line_items": []}
    rows = list(csv.reader(io.StringIO(extraction_to_csv(data))))
    assert ["LINE ITEMS"] in rows
    # header present, no data rows after it
    idx = rows.index(["LINE ITEMS"])
    assert rows[idx + 1][0] == "#"
    assert idx + 2 == len(rows) or not rows[idx + 2]


async def test_csv_endpoint(client):
    r = await upload(client, "csv-me.pdf", make_text_pdf([LONG_TEXT]))
    run_id = r.json()["data"]["documents"][0]["run_id"]
    await wait_run(client, run_id)
    ex = await wait_extraction(client, run_id)

    resp = await client.get(f"/api/ocr/extractions/{ex['id']}/csv")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/csv")
    assert "attachment" in resp.headers["content-disposition"]
    assert resp.headers["content-disposition"].endswith('.csv"')
    body = resp.text
    assert body.startswith("﻿")                    # UTF-8 BOM for Excel/Cyrillic
    rows = list(csv.reader(io.StringIO(body.lstrip("﻿"))))
    assert ["INVOICE METADATA"] in rows
    assert ["LINE ITEMS"] in rows
    # Mock extractor supplies a known supplier name.
    assert any(r == ["Name", "Mock Supplier"] for r in rows)


async def test_csv_endpoint_404_and_409(client, started_state):
    assert (await client.get("/api/ocr/extractions/9999/csv")).status_code == 404
    # A queued (incomplete) extraction → 409.
    doc = await started_state.db.insert_document(
        filename="x.pdf", sha256="f" * 64, mime="application/pdf", page_count=1)
    run = await started_state.db.create_run(document_id=doc["id"], engine="mock")
    ex = await started_state.db.create_extraction(run_id=run["id"], model="mock-extract")
    r = await client.get(f"/api/ocr/extractions/{ex['id']}/csv")
    assert r.status_code == 409
