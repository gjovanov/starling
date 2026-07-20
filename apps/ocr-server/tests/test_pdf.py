"""Unit tests for the PDF text-layer heuristic and rasterization."""

from pathlib import Path

import pytest

from ocr_server import pdf
from tests.helpers import make_image_only_pdf, make_mixed_pdf, make_png, make_text_pdf, LONG_TEXT


def test_page_count_pdf():
    assert pdf.page_count(make_text_pdf(["a" * 40, "b" * 40]), "application/pdf") == 2


def test_page_count_png():
    assert pdf.page_count(make_png(), "image/png") == 1


def test_page_count_corrupt():
    with pytest.raises(pdf.PdfError):
        pdf.page_count(b"%PDF-1.4\ngarbage", "application/pdf")


def test_page_count_unsupported_mime():
    with pytest.raises(pdf.PdfError):
        pdf.page_count(b"anything", "text/plain")


def test_text_layer_detected(tmp_path: Path):
    p = tmp_path / "doc.pdf"
    p.write_bytes(make_text_pdf([LONG_TEXT]))
    layers = pdf.extract_text_layers(p, "application/pdf", min_chars=32)
    assert len(layers) == 1
    assert layers[0] is not None
    assert "INVOICE No. 2026-001" in layers[0]


def test_image_only_page_is_none(tmp_path: Path):
    p = tmp_path / "doc.pdf"
    p.write_bytes(make_image_only_pdf(pages=2))
    layers = pdf.extract_text_layers(p, "application/pdf", min_chars=32)
    assert layers == [None, None]


def test_mixed_pdf_per_page(tmp_path: Path):
    p = tmp_path / "doc.pdf"
    p.write_bytes(make_mixed_pdf())
    layers = pdf.extract_text_layers(p, "application/pdf", min_chars=32)
    assert layers[0] is not None
    assert layers[1] is None


def test_short_text_below_threshold_is_none(tmp_path: Path):
    p = tmp_path / "doc.pdf"
    p.write_bytes(make_text_pdf(["hi"]))
    layers = pdf.extract_text_layers(p, "application/pdf", min_chars=32)
    assert layers == [None]


def test_image_upload_has_no_text_layer(tmp_path: Path):
    p = tmp_path / "img.png"
    p.write_bytes(make_png())
    assert pdf.extract_text_layers(p, "image/png", min_chars=32) == [None]


def test_rasterize_pdf(tmp_path: Path):
    p = tmp_path / "doc.pdf"
    p.write_bytes(make_text_pdf([LONG_TEXT, LONG_TEXT]))
    out = pdf.rasterize_pages(p, "application/pdf", tmp_path / "pages", dpi=72)
    assert [x.name for x in out] == ["page-001.png", "page-002.png"]
    assert all(x.is_file() and x.stat().st_size > 0 for x in out)
    # Idempotent — second call keeps existing files.
    mtimes = [x.stat().st_mtime_ns for x in out]
    out2 = pdf.rasterize_pages(p, "application/pdf", tmp_path / "pages", dpi=72)
    assert [x.stat().st_mtime_ns for x in out2] == mtimes


def test_rasterize_png_converts(tmp_path: Path):
    p = tmp_path / "img.png"
    p.write_bytes(make_png())
    out = pdf.rasterize_pages(p, "image/png", tmp_path / "pages", dpi=200)
    assert len(out) == 1 and out[0].name == "page-001.png"
