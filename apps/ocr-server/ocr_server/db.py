"""SQLite (WAL) persistence for documents, runs, page results and extractions.

The unit of comparison is a document; each OCR attempt is a run
(document x engine); each structured-extraction attempt hangs off a run.
All file paths are stored RELATIVE to the configured media/results roots so
the tree can be relocated without breaking the DB.
"""

import datetime as _dt
from pathlib import Path
from typing import Any

import aiosqlite

SCHEMA = """
CREATE TABLE IF NOT EXISTS document(
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  filename    TEXT NOT NULL,
  sha256      TEXT NOT NULL UNIQUE,
  mime        TEXT NOT NULL,
  page_count  INTEGER NOT NULL DEFAULT 0,
  created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS run(
  id             INTEGER PRIMARY KEY AUTOINCREMENT,
  document_id    INTEGER NOT NULL REFERENCES document(id),
  engine         TEXT NOT NULL,
  engine_version TEXT,
  status         TEXT NOT NULL DEFAULT 'queued',  -- queued|running|done|failed
  error          TEXT,
  extract_auto   INTEGER NOT NULL DEFAULT 1,
  extract_model  TEXT,
  language_hint  TEXT,
  ms_total       INTEGER,
  created_at     TEXT NOT NULL,
  started_at     TEXT,
  finished_at    TEXT
);

CREATE TABLE IF NOT EXISTS page_result(
  run_id        INTEGER NOT NULL REFERENCES run(id),
  page_no       INTEGER NOT NULL,                 -- 1-based
  source        TEXT NOT NULL,                    -- 'ocr' | 'text-layer'
  markdown_path TEXT NOT NULL,                    -- relative to results root
  image_path    TEXT,                             -- relative to media root
  ms            INTEGER,
  PRIMARY KEY(run_id, page_no)
);

CREATE TABLE IF NOT EXISTS extraction(
  id                INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id            INTEGER NOT NULL REFERENCES run(id),
  model             TEXT NOT NULL,
  status            TEXT NOT NULL DEFAULT 'queued',
  error             TEXT,
  json_path         TEXT,                          -- relative to results root
  validation_status TEXT,                          -- 'valid' | 'needs_review'
  validation_path   TEXT,                          -- relative to results root
  ms                INTEGER,
  created_at        TEXT NOT NULL,
  finished_at       TEXT
);

CREATE INDEX IF NOT EXISTS idx_run_status ON run(status);
CREATE INDEX IF NOT EXISTS idx_run_document ON run(document_id);
CREATE INDEX IF NOT EXISTS idx_extraction_status ON extraction(status);
CREATE INDEX IF NOT EXISTS idx_extraction_run ON extraction(run_id);
"""


def utcnow() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _row(row: aiosqlite.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


class Database:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._db: aiosqlite.Connection | None = None

    @property
    def conn(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("Database.connect() not called")
        return self._db

    async def connect(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self._path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA foreign_keys=ON")
        await self._db.executescript(SCHEMA)
        await self._db.commit()

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    # ── documents ────────────────────────────────────────────────────

    async def get_document_by_sha(self, sha256: str) -> dict | None:
        cur = await self.conn.execute("SELECT * FROM document WHERE sha256=?", (sha256,))
        return _row(await cur.fetchone())

    async def get_document(self, doc_id: int) -> dict | None:
        cur = await self.conn.execute("SELECT * FROM document WHERE id=?", (doc_id,))
        return _row(await cur.fetchone())

    async def insert_document(self, *, filename: str, sha256: str, mime: str, page_count: int) -> dict:
        cur = await self.conn.execute(
            "INSERT INTO document(filename, sha256, mime, page_count, created_at) VALUES(?,?,?,?,?)",
            (filename, sha256, mime, page_count, utcnow()),
        )
        await self.conn.commit()
        doc = await self.get_document(cur.lastrowid)
        assert doc is not None
        return doc

    async def list_documents(self) -> list[dict]:
        cur = await self.conn.execute("SELECT * FROM document ORDER BY id DESC")
        return [dict(r) for r in await cur.fetchall()]

    async def delete_document(self, doc_id: int) -> list[int]:
        """Cascade-delete a document + its runs, page_results and extractions.
        Returns the deleted run ids so the caller can prune result dirs."""
        cur = await self.conn.execute("SELECT id FROM run WHERE document_id=?", (doc_id,))
        run_ids = [r[0] for r in await cur.fetchall()]
        for rid in run_ids:
            await self.conn.execute("DELETE FROM extraction WHERE run_id=?", (rid,))
            await self.conn.execute("DELETE FROM page_result WHERE run_id=?", (rid,))
        await self.conn.execute("DELETE FROM run WHERE document_id=?", (doc_id,))
        await self.conn.execute("DELETE FROM document WHERE id=?", (doc_id,))
        await self.conn.commit()
        return run_ids

    # ── runs ─────────────────────────────────────────────────────────

    async def create_run(
        self,
        *,
        document_id: int,
        engine: str,
        engine_version: str | None = None,
        extract_auto: bool = True,
        extract_model: str | None = None,
        language_hint: str | None = None,
    ) -> dict:
        cur = await self.conn.execute(
            "INSERT INTO run(document_id, engine, engine_version, extract_auto, extract_model,"
            " language_hint, created_at) VALUES(?,?,?,?,?,?,?)",
            (document_id, engine, engine_version, int(extract_auto), extract_model,
             language_hint, utcnow()),
        )
        await self.conn.commit()
        run = await self.get_run(cur.lastrowid)
        assert run is not None
        return run

    async def get_run(self, run_id: int) -> dict | None:
        cur = await self.conn.execute("SELECT * FROM run WHERE id=?", (run_id,))
        return _row(await cur.fetchone())

    async def runs_for_document(self, doc_id: int) -> list[dict]:
        cur = await self.conn.execute(
            "SELECT * FROM run WHERE document_id=? ORDER BY id DESC", (doc_id,))
        return [dict(r) for r in await cur.fetchall()]

    async def latest_run_for_document(self, doc_id: int) -> dict | None:
        cur = await self.conn.execute(
            "SELECT * FROM run WHERE document_id=? ORDER BY id DESC LIMIT 1", (doc_id,))
        return _row(await cur.fetchone())

    async def next_queued_run(self) -> dict | None:
        cur = await self.conn.execute(
            "SELECT * FROM run WHERE status='queued' ORDER BY id LIMIT 1")
        return _row(await cur.fetchone())

    async def mark_run_running(self, run_id: int, *, engine_version: str | None = None) -> None:
        await self.conn.execute(
            "UPDATE run SET status='running', started_at=?, engine_version=COALESCE(?, engine_version)"
            " WHERE id=?",
            (utcnow(), engine_version, run_id),
        )
        await self.conn.commit()

    async def mark_run_finished(self, run_id: int, *, status: str, error: str | None = None,
                                ms_total: int | None = None,
                                engine_version: str | None = None) -> None:
        # engine_version again here: CLI engines only learn their real
        # version during the run (lazy --version probe).
        await self.conn.execute(
            "UPDATE run SET status=?, error=?, ms_total=?, finished_at=?,"
            " engine_version=COALESCE(?, engine_version) WHERE id=?",
            (status, error, ms_total, utcnow(), engine_version, run_id),
        )
        await self.conn.commit()

    async def reset_inflight(self) -> int:
        """Resume-on-start: anything left 'running' by a dead process re-queues."""
        cur1 = await self.conn.execute("UPDATE run SET status='queued', started_at=NULL WHERE status='running'")
        cur2 = await self.conn.execute("UPDATE extraction SET status='queued' WHERE status='running'")
        await self.conn.commit()
        return (cur1.rowcount or 0) + (cur2.rowcount or 0)

    async def has_queued_work(self) -> bool:
        cur = await self.conn.execute(
            "SELECT EXISTS(SELECT 1 FROM run WHERE status='queued')"
            " OR EXISTS(SELECT 1 FROM extraction WHERE status='queued')")
        row = await cur.fetchone()
        return bool(row[0])

    # ── page results ─────────────────────────────────────────────────

    async def upsert_page(self, *, run_id: int, page_no: int, source: str,
                          markdown_path: str, image_path: str | None, ms: int | None) -> None:
        await self.conn.execute(
            "INSERT INTO page_result(run_id, page_no, source, markdown_path, image_path, ms)"
            " VALUES(?,?,?,?,?,?)"
            " ON CONFLICT(run_id, page_no) DO UPDATE SET"
            " source=excluded.source, markdown_path=excluded.markdown_path,"
            " image_path=excluded.image_path, ms=excluded.ms",
            (run_id, page_no, source, markdown_path, image_path, ms),
        )
        await self.conn.commit()

    async def pages_for_run(self, run_id: int) -> list[dict]:
        cur = await self.conn.execute(
            "SELECT * FROM page_result WHERE run_id=? ORDER BY page_no", (run_id,))
        return [dict(r) for r in await cur.fetchall()]

    async def get_page(self, run_id: int, page_no: int) -> dict | None:
        cur = await self.conn.execute(
            "SELECT * FROM page_result WHERE run_id=? AND page_no=?", (run_id, page_no))
        return _row(await cur.fetchone())

    async def pages_done(self, run_id: int) -> int:
        cur = await self.conn.execute(
            "SELECT COUNT(*) FROM page_result WHERE run_id=?", (run_id,))
        row = await cur.fetchone()
        return int(row[0])

    # ── extractions ──────────────────────────────────────────────────

    async def create_extraction(self, *, run_id: int, model: str) -> dict:
        cur = await self.conn.execute(
            "INSERT INTO extraction(run_id, model, created_at) VALUES(?,?,?)",
            (run_id, model, utcnow()),
        )
        await self.conn.commit()
        ex = await self.get_extraction(cur.lastrowid)
        assert ex is not None
        return ex

    async def get_extraction(self, ex_id: int) -> dict | None:
        cur = await self.conn.execute("SELECT * FROM extraction WHERE id=?", (ex_id,))
        return _row(await cur.fetchone())

    async def extractions_for_run(self, run_id: int) -> list[dict]:
        cur = await self.conn.execute(
            "SELECT * FROM extraction WHERE run_id=? ORDER BY id DESC", (run_id,))
        return [dict(r) for r in await cur.fetchall()]

    async def latest_extraction_for_run(self, run_id: int) -> dict | None:
        cur = await self.conn.execute(
            "SELECT * FROM extraction WHERE run_id=? ORDER BY id DESC LIMIT 1", (run_id,))
        return _row(await cur.fetchone())

    async def next_queued_extraction(self) -> dict | None:
        cur = await self.conn.execute(
            "SELECT * FROM extraction WHERE status='queued' ORDER BY id LIMIT 1")
        return _row(await cur.fetchone())

    async def mark_extraction_running(self, ex_id: int) -> None:
        await self.conn.execute(
            "UPDATE extraction SET status='running' WHERE id=?", (ex_id,))
        await self.conn.commit()

    async def mark_extraction_finished(
        self, ex_id: int, *, status: str, error: str | None = None,
        json_path: str | None = None, validation_status: str | None = None,
        validation_path: str | None = None, ms: int | None = None,
    ) -> None:
        await self.conn.execute(
            "UPDATE extraction SET status=?, error=?, json_path=?, validation_status=?,"
            " validation_path=?, ms=?, finished_at=? WHERE id=?",
            (status, error, json_path, validation_status, validation_path, ms, utcnow(), ex_id),
        )
        await self.conn.commit()
