/**
 * OCR jobs table — one row per run, live-updated from WS events (no polling).
 * Shows per-step durations (OCR + extraction), client-side pagination, and
 * per-document deletion. resync() reconciles rows with the server after a WS
 * reconnect so in-flight rows can't get stuck on a lost event.
 */
const API_BASE = '';
const PAGE_SIZE = 15;

function fmtMs(ms) {
  if (ms === null || ms === undefined) return '';
  return ms < 1000 ? `${ms} ms` : `${(ms / 1000).toFixed(1)} s`;
}

export class OcrJobs {
  constructor(container, { fetchJson, esc, onView, onCompare, onDelete }) {
    this.container = container;
    this.fetchJson = fetchJson;
    this.esc = esc;
    this.onView = onView;
    this.onCompare = onCompare;
    this.onDelete = onDelete;
    this.rows = new Map();       // run_id -> row state
    this._hydrating = new Set(); // run_ids with an in-flight hydrate
    this.page = 0;
  }

  _rowFrom(run, doc, extraction) {
    return {
      run_id: run.id,
      document_id: doc.id,
      filename: doc.filename,
      engine: run.engine,
      status: run.status,
      error: run.error,
      done: run.pages_done ?? 0,
      total: doc.page_count,
      ms_total: run.ms_total,
      extraction: extraction ? {
        id: extraction.id,
        status: extraction.status,
        validation_status: extraction.validation_status,
        ms: extraction.ms ?? null,
      } : null,
    };
  }

  setInitial(documents) {
    for (const d of documents) {
      if (!d.latest_run) continue;
      this.rows.set(d.latest_run.id, this._rowFrom(d.latest_run, d, d.latest_run.latest_extraction));
    }
    this.render();
  }

  /** Reconcile with server truth after a WS reconnect: refresh latest-run
   *  rows (corrects a status stuck on a lost event) and drop rows whose
   *  document no longer exists (deleted elsewhere). */
  resync(documents) {
    const liveDocIds = new Set(documents.map((d) => d.id));
    for (const d of documents) {
      if (!d.latest_run) continue;
      this.rows.set(d.latest_run.id, this._rowFrom(d.latest_run, d, d.latest_run.latest_extraction));
    }
    for (const [runId, row] of this.rows) {
      if (!liveDocIds.has(row.document_id)) this.rows.delete(runId);
    }
    this._clampPage();
    this.render();
  }

  async hydrate(runId) {
    if (this.rows.has(runId) || this._hydrating.has(runId)) return;
    this._hydrating.add(runId);
    try {
      const detail = await this.fetchJson(`${API_BASE}/api/ocr/runs/${runId}`);
      if (!this.rows.has(runId)) {
        this.rows.set(runId, this._rowFrom(detail.run, detail.document, detail.extractions[0]));
        this.render();
      }
    } catch (_) {
      // Row will appear on the next WS event instead.
    } finally {
      this._hydrating.delete(runId);
    }
  }

  handleEvent(msg) {
    if (msg.type === 'run') {
      const row = this.rows.get(msg.id);
      if (!row) { this.hydrate(msg.id); return; }
      row.status = msg.status;
      if (msg.page) { row.done = msg.page.done; row.total = msg.page.total; }
      if (msg.ms_total !== undefined) row.ms_total = msg.ms_total;
      if (msg.error) row.error = msg.error;
      this.render();
    } else if (msg.type === 'extraction') {
      const row = this.rows.get(msg.run_id);
      if (!row) { this.hydrate(msg.run_id); return; }
      row.extraction = {
        id: msg.id,
        status: msg.status,
        validation_status: msg.validation_status || null,
        ms: msg.ms ?? (row.extraction ? row.extraction.ms : null),
      };
      this.render();
    } else if (msg.type === 'document' && msg.status === 'deleted') {
      this.removeDocument(msg.id);
    }
  }

  removeDocument(docId) {
    let changed = false;
    for (const [runId, row] of this.rows) {
      if (row.document_id === docId) { this.rows.delete(runId); changed = true; }
    }
    if (changed) { this._clampPage(); this.render(); }
  }

  _sorted() {
    return Array.from(this.rows.values()).sort((a, b) => b.run_id - a.run_id);
  }

  _pageCount() {
    return Math.max(1, Math.ceil(this.rows.size / PAGE_SIZE));
  }

  _clampPage() {
    this.page = Math.min(Math.max(0, this.page), this._pageCount() - 1);
  }

  render() {
    const esc = this.esc;
    if (this.rows.size === 0) {
      this.container.innerHTML =
        '<div class="empty-state"><p>No documents yet — upload to start.</p></div>';
      return;
    }
    this._clampPage();
    const all = this._sorted();
    const start = this.page * PAGE_SIZE;
    const pageRows = all.slice(start, start + PAGE_SIZE);

    const body = pageRows.map((r) => {
      const pct = r.total ? Math.round((100 * r.done) / r.total) : 0;
      const ex = r.extraction;
      const ocrDur = r.ms_total != null
        ? `<span class="ocr-dur">${fmtMs(r.ms_total)}</span>` : '';
      const exBadge = !ex
        ? '<span class="ocr-badge">—</span>'
        : `<span class="ocr-badge ${esc(ex.status)}">${esc(ex.status)}</span>`;
      const valBadge = ex && ex.validation_status
        ? ` <span class="ocr-badge ${esc(ex.validation_status)}">${esc(ex.validation_status)}</span>`
        : '';
      const exDur = ex && ex.ms != null
        ? ` <span class="ocr-dur">${fmtMs(ex.ms)}</span>` : '';
      const errTitle = r.error ? ` title="${esc(r.error)}"` : '';
      return `<tr data-run="${r.run_id}" data-doc="${r.document_id}">
        <td><span class="ocr-doc-link" data-role="compare" title="Compare runs of this document">${esc(r.filename)}</span> <span style="color:#6c757d;">#${r.run_id}</span></td>
        <td><span class="ocr-badge engine">${esc(r.engine)}</span></td>
        <td>
          <div class="ocr-progress"><div class="ocr-progress-fill" style="width:${pct}%"></div></div>
          <span style="color:#a0a0a0;font-size:0.85em;"> ${r.done}/${r.total}</span>
        </td>
        <td><span class="ocr-badge ${esc(r.status)}"${errTitle}>${esc(r.status)}</span> ${ocrDur}</td>
        <td>${exBadge}${valBadge}${exDur}</td>
        <td style="white-space:nowrap;">
          <button class="btn-small btn-primary" data-role="view">View</button>
          <button class="btn-small btn-danger" data-role="delete" title="Delete document + all its runs">✕</button>
        </td>
      </tr>`;
    }).join('');

    const pages = this._pageCount();
    const pager = pages > 1 ? `
      <div class="ocr-pager">
        <button class="btn-small btn-primary" data-role="prev" ${this.page <= 0 ? 'disabled' : ''}>&larr; Prev</button>
        <span style="color:#a0a0a0;">page ${this.page + 1} / ${pages} · ${this.rows.size} runs</span>
        <button class="btn-small btn-primary" data-role="next" ${this.page >= pages - 1 ? 'disabled' : ''}>Next &rarr;</button>
      </div>` : '';

    this.container.innerHTML = `<table class="ocr-table">
      <thead><tr>
        <th>Document</th><th>Engine</th><th>Pages</th><th>OCR (dur)</th>
        <th>Extract (dur)</th><th></th>
      </tr></thead>
      <tbody>${body}</tbody></table>${pager}`;

    this.container.querySelectorAll('[data-role="view"]').forEach((btn) =>
      btn.addEventListener('click', () =>
        this.onView(parseInt(btn.closest('tr').dataset.run, 10))));
    this.container.querySelectorAll('[data-role="compare"]').forEach((el) =>
      el.addEventListener('click', () => {
        if (this.onCompare) this.onCompare(parseInt(el.closest('tr').dataset.doc, 10));
      }));
    this.container.querySelectorAll('[data-role="delete"]').forEach((btn) =>
      btn.addEventListener('click', () => {
        const tr = btn.closest('tr');
        const docId = parseInt(tr.dataset.doc, 10);
        const name = tr.querySelector('.ocr-doc-link').textContent;
        if (this.onDelete) this.onDelete(docId, name);
      }));
    const prev = this.container.querySelector('[data-role="prev"]');
    const next = this.container.querySelector('[data-role="next"]');
    if (prev) prev.addEventListener('click', () => { this.page--; this.render(); });
    if (next) next.addEventListener('click', () => { this.page++; this.render(); });
  }
}
