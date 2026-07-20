"""Upload handling: mime sniffing, sha256 dedupe, original storage.

Uploaded originals live at media/ocr/{doc_id}/original.{ext}; page rasters
are produced lazily by the run worker (same directory). Filenames on disk
are never derived from user input — the display name only lives in the DB.
"""

import hashlib
from pathlib import Path

from . import pdf
from .db import Database


class IngestError(ValueError):
    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


_EXT_BY_MIME = {
    "application/pdf": ".pdf",
    "image/png": ".png",
    "image/jpeg": ".jpg",
}


def detect_mime(content: bytes) -> str | None:
    """Magic-byte sniffing — never trust the client's content type."""
    if content.startswith(b"%PDF-"):
        return "application/pdf"
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if content.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    return None


def original_path(media_root: Path, doc: dict) -> Path:
    return media_root / str(doc["id"]) / f"original{_EXT_BY_MIME[doc['mime']]}"


async def ingest_upload(
    db: Database,
    media_root: Path,
    *,
    filename: str,
    content: bytes,
    max_bytes: int,
) -> tuple[dict, bool]:
    """Validate + store one uploaded file. Returns (document_row, created).

    Re-uploading bytes with a known sha256 reuses the existing document row
    (and its rasters) — a new run is still created by the caller.
    """
    if len(content) == 0:
        raise IngestError("empty file", status_code=400)
    if len(content) > max_bytes:
        raise IngestError(
            f"file too large ({len(content)} bytes; limit {max_bytes})", status_code=413)

    mime = detect_mime(content)
    if mime is None:
        raise IngestError(
            "unsupported file type — accepted: PDF, PNG, JPEG", status_code=415)

    sha256 = hashlib.sha256(content).hexdigest()
    existing = await db.get_document_by_sha(sha256)
    if existing is not None:
        # Ensure the original still exists on disk (it may have been pruned).
        target = original_path(media_root, existing)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        return existing, False

    # Parse before persisting anything: corrupt files must not leave rows behind.
    try:
        import asyncio

        pages = await asyncio.to_thread(pdf.page_count, content, mime)
    except pdf.PdfError as exc:
        raise IngestError(f"corrupt or unreadable document: {exc}", status_code=400) from exc

    doc = await db.insert_document(
        filename=filename, sha256=sha256, mime=mime, page_count=pages)
    target = original_path(media_root, doc)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    return doc, True
