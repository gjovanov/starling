"""Shared test helpers: synthetic documents + polling."""

import asyncio
import time

import pymupdf

LONG_TEXT = (
    "INVOICE No. 2026-001\n"
    "Supplier: Test Ltd, Sofia\n"
    "Item: Widget, Qty 2, Price 10.00\n"
    "Total: 20.00 EUR — thank you for your business."
)


def make_text_pdf(pages: list[str]) -> bytes:
    """Born-digital PDF with a real text layer on every page."""
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((72, 72), text, fontsize=12)
    data = doc.tobytes()
    doc.close()
    return data


def make_image_only_pdf(pages: int = 1) -> bytes:
    """PDF whose pages contain only vector drawings — no extractable text."""
    doc = pymupdf.open()
    for _ in range(pages):
        page = doc.new_page()
        page.draw_rect(pymupdf.Rect(72, 72, 300, 200), color=(0, 0, 0), width=2)
        page.draw_line(pymupdf.Point(72, 250), pymupdf.Point(300, 250))
    data = doc.tobytes()
    doc.close()
    return data


def make_mixed_pdf() -> bytes:
    """Page 1 has a text layer; page 2 is drawing-only (needs OCR)."""
    doc = pymupdf.open()
    p1 = doc.new_page()
    p1.insert_text((72, 72), LONG_TEXT, fontsize=12)
    p2 = doc.new_page()
    p2.draw_rect(pymupdf.Rect(72, 72, 300, 200), color=(0, 0, 0), width=2)
    data = doc.tobytes()
    doc.close()
    return data


def make_png() -> bytes:
    doc = pymupdf.open()
    page = doc.new_page(width=200, height=100)
    page.draw_rect(pymupdf.Rect(10, 10, 190, 90), color=(0, 0, 0), width=2)
    png = page.get_pixmap().tobytes("png")
    doc.close()
    return png


async def upload(client, filename: str, content: bytes, **form):
    files = [("files", (filename, content, "application/octet-stream"))]
    return await client.post("/api/ocr/documents", files=files, data=form)


async def wait_run(client, run_id: int, timeout: float = 15.0) -> dict:
    """Poll run detail until done/failed; returns the detail payload."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        r = await client.get(f"/api/ocr/runs/{run_id}")
        assert r.status_code == 200, r.text
        detail = r.json()["data"]
        if detail["run"]["status"] in ("done", "failed"):
            return detail
        await asyncio.sleep(0.02)
    raise TimeoutError(f"run {run_id} did not finish within {timeout}s")


async def wait_extraction(client, run_id: int, timeout: float = 15.0) -> dict:
    """Poll run detail until its newest extraction is done/failed."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        r = await client.get(f"/api/ocr/runs/{run_id}")
        assert r.status_code == 200, r.text
        detail = r.json()["data"]
        exs = detail["extractions"]
        if exs and exs[0]["status"] in ("done", "failed"):
            return exs[0]
        await asyncio.sleep(0.02)
    raise TimeoutError(f"extraction for run {run_id} did not finish within {timeout}s")
