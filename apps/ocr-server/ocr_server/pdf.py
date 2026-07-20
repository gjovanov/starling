"""PDF / image page handling via PyMuPDF.

Text-layer fast path: a PDF page with a real extractable text layer is
converted straight to Markdown (source='text-layer'), skipping the GPU
entirely. Image-only pages are rasterized at ~raster_dpi for OCR. Mixed
PDFs are handled per page. All functions here are synchronous (PyMuPDF is
sync) — callers run them via asyncio.to_thread.
"""

from pathlib import Path

import pymupdf


class PdfError(ValueError):
    """Raised when a document cannot be parsed (corrupt / unsupported)."""


_FILETYPE_BY_MIME = {
    "application/pdf": "pdf",
    "image/png": "png",
    "image/jpeg": "jpg",
}


def _open(content_or_path: bytes | Path, mime: str) -> pymupdf.Document:
    filetype = _FILETYPE_BY_MIME.get(mime)
    if filetype is None:
        raise PdfError(f"unsupported mime type: {mime}")
    try:
        if isinstance(content_or_path, bytes):
            doc = pymupdf.open(stream=content_or_path, filetype=filetype)
        else:
            doc = pymupdf.open(str(content_or_path), filetype=filetype)
    except Exception as exc:
        raise PdfError(f"cannot parse document: {exc}") from exc
    if doc.page_count < 1:
        doc.close()
        raise PdfError("document has no pages")
    return doc


def page_count(content: bytes, mime: str) -> int:
    """Validate the uploaded bytes and return the page count. Raises PdfError."""
    doc = _open(content, mime)
    try:
        if mime == "application/pdf":
            # Force-load the first page: truncated PDFs often open but fail here.
            doc.load_page(0)
        return doc.page_count
    except PdfError:
        raise
    except Exception as exc:
        raise PdfError(f"cannot read document pages: {exc}") from exc
    finally:
        doc.close()


def extract_text_layers(path: Path, mime: str, *, min_chars: int) -> list[str | None]:
    """Per-page Markdown from the PDF text layer, or None when the page is
    image-only (needs OCR). Images always return [None]."""
    if mime != "application/pdf":
        return [None]
    doc = _open(path, mime)
    try:
        out: list[str | None] = []
        for page in doc:
            raw = page.get_text("text")
            if len("".join(raw.split())) >= min_chars:
                out.append(text_to_markdown(page))
            else:
                out.append(None)
        return out
    finally:
        doc.close()


def text_to_markdown(page: pymupdf.Page) -> str:
    """Markdown from a text-layer page: blocks in reading order, blank-line
    separated. Line structure inside a block is preserved (invoices are
    line-oriented)."""
    blocks = page.get_text("blocks")
    # block tuple: (x0, y0, x1, y1, text, block_no, block_type); type 1 = image
    text_blocks = [b for b in blocks if len(b) < 7 or b[6] == 0]
    text_blocks.sort(key=lambda b: (round(b[1], 1), round(b[0], 1)))
    parts = []
    for b in text_blocks:
        t = b[4].rstrip()
        if t.strip():
            parts.append(t)
    return "\n\n".join(parts) + "\n"


def rasterize_pages(path: Path, mime: str, out_dir: Path, *, dpi: int) -> list[Path]:
    """Render every page to out_dir/page-NNN.png (idempotent — existing files
    are kept). Returns the ordered list of PNG paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = _open(path, mime)
    try:
        paths: list[Path] = []
        for i in range(doc.page_count):
            png = out_dir / f"page-{i + 1:03d}.png"
            if not png.exists():
                page = doc.load_page(i)
                if mime == "application/pdf":
                    pix = page.get_pixmap(dpi=dpi)
                else:
                    # Uploaded images keep their native resolution.
                    pix = page.get_pixmap()
                pix.save(str(png))
            paths.append(png)
        return paths
    except Exception as exc:
        raise PdfError(f"rasterization failed: {exc}") from exc
    finally:
        doc.close()
