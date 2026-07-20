/**
 * OCR viewer — a 4-tab control (Viewer image | Markdown | Extraction JSON |
 * Download CSV). Page image/markdown are per-page; extraction/download are
 * per-run. Live-refreshes while its run progresses.
 */
const API_BASE = '';

const fmt = (v, esc) => (v === null || v === undefined ? '' : esc(v));

export class OcrViewer {
  constructor({ fetchJson, esc }) {
    this.fetchJson = fetchJson;
    this.esc = esc;
    this.runId = null;
    this.page = 1;
    this.detail = null;
    this.activeTab = 'viewer';
    this.extraction = null;    // { status, data, validation } of the latest
    this.extractionId = null;  // id for the CSV download
    this.els = {
      section:    document.getElementById('ocr-viewer-section'),
      title:      document.getElementById('ocr-viewer-title'),
      prev:       document.getElementById('ocr-page-prev'),
      next:       document.getElementById('ocr-page-next'),
      indicator:  document.getElementById('ocr-page-indicator'),
      pagenav:    document.getElementById('ocr-tab-pagenav'),
      close:      document.getElementById('ocr-viewer-close'),
      rerun:      document.getElementById('ocr-extract-rerun'),
      image:      document.getElementById('ocr-viewer-image'),
      markdown:   document.getElementById('ocr-viewer-markdown'),
      source:     document.getElementById('ocr-viewer-source'),
      json:       document.getElementById('ocr-viewer-json'),
      checks:     document.getElementById('ocr-viewer-checks'),
      validation: document.getElementById('ocr-viewer-validation'),
      downloadBtn:     document.getElementById('ocr-download-csv'),
      downloadNote:    document.getElementById('ocr-download-note'),
      downloadPreview: document.getElementById('ocr-download-preview'),
      tabs:  Array.from(document.querySelectorAll('#ocr-viewer-section .ocr-tab')),
      panes: Array.from(document.querySelectorAll('#ocr-viewer-section .ocr-tab-pane')),
    };
    if (!this.els.section) return;
    this.els.prev.addEventListener('click', () => this.showPage(this.page - 1));
    this.els.next.addEventListener('click', () => this.showPage(this.page + 1));
    this.els.close.addEventListener('click', () => this.closeViewer());
    this.els.rerun.addEventListener('click', () => this.rerunExtraction());
    this.els.tabs.forEach((t) => t.addEventListener('click', () => this.setTab(t.dataset.otab)));
    this.els.downloadBtn.addEventListener('click', () => this.downloadCsv());
  }

  setTab(name) {
    this.activeTab = name;
    this.els.tabs.forEach((t) => t.classList.toggle('active', t.dataset.otab === name));
    this.els.panes.forEach((p) => p.classList.toggle('active', p.dataset.otabPane === name));
    // Page prev/next only makes sense for the per-page tabs.
    this.els.pagenav.style.display = (name === 'viewer' || name === 'markdown') ? '' : 'none';
  }

  async open(runId) {
    this.runId = runId;
    this.page = 1;
    this.setTab('viewer');
    await this.refresh();
    this.els.section.style.display = '';
    this.els.section.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  closeViewer() {
    this.runId = null;
    this.els.section.style.display = 'none';
  }

  closeIfDocument(docId) {
    if (this.detail && this.detail.document && this.detail.document.id === docId) {
      this.closeViewer();
    }
  }

  handleEvent(msg) {
    if (msg.type === 'document' && msg.status === 'deleted') {
      this.closeIfDocument(msg.id);
      return;
    }
    if (this.runId === null) return;
    const relevant =
      (msg.type === 'run' && msg.id === this.runId) ||
      (msg.type === 'extraction' && msg.run_id === this.runId);
    if (relevant) this.refresh();
  }

  async refresh() {
    if (this.runId === null) return;
    try {
      this.detail = await this.fetchJson(`${API_BASE}/api/ocr/runs/${this.runId}`);
    } catch (_) {
      return;
    }
    const { run, document: doc } = this.detail;
    this.els.title.textContent = `${doc.filename} — run #${run.id} (${run.engine})`;
    this.els.rerun.style.display = run.status === 'done' ? '' : 'none';
    await this.showPage(this.page);
    await this.loadExtraction();
  }

  async showPage(pageNo) {
    if (!this.detail) return;
    const pages = this.detail.pages;
    const total = this.detail.document.page_count;
    const clamped = Math.max(1, Math.min(pageNo, total));
    this.page = clamped;
    this.els.indicator.textContent = `page ${clamped} / ${total}`;
    this.els.prev.disabled = clamped <= 1;
    this.els.next.disabled = clamped >= total;

    const pageRow = pages.find((p) => p.page_no === clamped);
    if (!pageRow) {
      this.els.image.removeAttribute('src');
      this.els.markdown.textContent = '(page not processed yet)';
      this.els.source.textContent = '';
      return;
    }
    this.els.image.src = `${API_BASE}/api/ocr/runs/${this.runId}/pages/${clamped}.png?v=${Date.now()}`;
    this.els.source.textContent = pageRow.source;
    this.els.source.className = 'ocr-badge ' + (pageRow.source === 'ocr' ? 'engine' : 'done');
    try {
      const r = await fetch(`${API_BASE}/api/ocr/runs/${this.runId}/pages/${clamped}.md`);
      this.els.markdown.textContent = r.ok ? await r.text() : `(markdown unavailable: ${r.status})`;
    } catch (err) {
      this.els.markdown.textContent = `(markdown unavailable: ${err.message})`;
    }
  }

  async loadExtraction() {
    const esc = this.esc;
    const latest = (this.detail.extractions || [])[0];
    this.extraction = null;
    this.extractionId = null;
    this.els.validation.style.display = 'none';
    this.els.checks.innerHTML = '';

    if (!latest) {
      this.els.json.textContent = '(no extraction)';
      this.renderDownloadTab();
      return;
    }
    if (latest.status === 'failed') {
      this.els.json.textContent = `extraction failed: ${latest.error || 'unknown error'}`;
      this.extraction = { status: 'failed' };
      this.renderDownloadTab();
      return;
    }
    if (latest.status !== 'done') {
      this.els.json.textContent = `extraction ${latest.status}…`;
      this.extraction = { status: latest.status };
      this.renderDownloadTab();
      return;
    }
    let payload;
    try {
      payload = await this.fetchJson(`${API_BASE}/api/ocr/extractions/${latest.id}`);
    } catch (err) {
      this.els.json.textContent = `(extraction unavailable: ${err.message})`;
      this.renderDownloadTab();
      return;
    }
    this.extraction = { status: 'done', data: payload.data, validation: payload.validation };
    this.extractionId = latest.id;

    if (payload.validation) {
      const v = payload.validation;
      this.els.validation.textContent = v.overall;
      this.els.validation.className = `ocr-badge ${v.overall}`;
      this.els.validation.style.display = '';
      this.els.checks.innerHTML = (v.checks || []).map((c) => {
        const cls = c.status === 'fail' ? 'ocr-check-fail'
          : c.status === 'pass' ? 'ocr-check-pass'
          : c.status === 'warn' ? 'ocr-check-warn' : 'ocr-check-skip';
        const sym = c.status === 'fail' ? '✗' : c.status === 'pass' ? '✓'
          : c.status === 'warn' ? '!' : '—';
        return `<div class="${cls}">${sym} ` +
          `${esc(c.check)}${c.details ? ` — ${esc(c.details)}` : ''}</div>`;
      }).join('');
    }
    this.els.json.textContent = JSON.stringify(payload.data, null, 2);
    this.renderDownloadTab();
  }

  renderDownloadTab() {
    const esc = this.esc;
    const ex = this.extraction;
    const el = this.els.downloadPreview;
    if (!ex || ex.status !== 'done' || !ex.data) {
      this.els.downloadBtn.disabled = true;
      this.els.downloadNote.textContent =
        ex && ex.status && ex.status !== 'done' ? `extraction ${ex.status}…`
          : 'no extraction to download';
      el.innerHTML = '';
      return;
    }
    const d = ex.data;
    this.els.downloadBtn.disabled = false;
    this.els.downloadNote.textContent = 'metadata + line items — opens in Excel / LibreOffice';

    const kv = (rows) => `<table class="ocr-table"><tbody>` + rows.map(([k, v]) =>
      `<tr><td style="color:#a0a0a0;width:180px;">${esc(k)}</td><td>${fmt(v, esc)}</td></tr>`
    ).join('') + `</tbody></table>`;
    const party = (p) => kv([
      ['Name', p && p.name], ['Address', p && p.address], ['VAT ID', p && p.vat_id],
      ['Company ID', p && p.company_id], ['IBAN', p && p.iban],
    ]);
    const items = d.line_items || [];
    const itemsTable = `<table class="ocr-table"><thead><tr>` +
      ['#', 'Description', 'Qty', 'Unit', 'Unit price', 'VAT %', 'Total']
        .map((h) => `<th>${h}</th>`).join('') + `</tr></thead><tbody>` +
      (items.length ? items.map((it, i) =>
        `<tr><td>${i + 1}</td><td>${fmt(it.description, esc)}</td><td>${fmt(it.quantity, esc)}</td>` +
        `<td>${fmt(it.unit, esc)}</td><td>${fmt(it.unit_price, esc)}</td>` +
        `<td>${fmt(it.vat_rate, esc)}</td><td>${fmt(it.total, esc)}</td></tr>`).join('')
        : `<tr><td colspan="7" style="color:#6c757d;">(no line items)</td></tr>`) +
      `</tbody></table>`;

    el.innerHTML =
      `<h4>Invoice metadata</h4>` + kv([
        ['Document type', d.document_type], ['Document number', d.document_number],
        ['Issue date', d.issue_date], ['Due date', d.due_date], ['Currency', d.currency],
        ['Net total', d.net_total], ['VAT total', d.vat_total], ['Gross total', d.gross_total],
        ['Payment terms', d.payment_terms],
      ]) +
      `<h4>Supplier</h4>` + party(d.supplier) +
      `<h4>Customer</h4>` + party(d.customer) +
      `<h4>Line items</h4>` + itemsTable;
  }

  downloadCsv() {
    if (!this.extractionId) return;
    const a = document.createElement('a');
    a.href = `${API_BASE}/api/ocr/extractions/${this.extractionId}/csv`;
    a.download = '';
    document.body.appendChild(a);
    a.click();
    a.remove();
  }

  async rerunExtraction() {
    if (this.runId === null) return;
    try {
      await this.fetchJson(`${API_BASE}/api/ocr/runs/${this.runId}/extract`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({}),
      });
    } catch (err) {
      this.els.json.textContent = `re-extract failed: ${err.message}`;
    }
  }
}

/**
 * Bake-off compare: per-document grid of runs + two-pane Markdown line diff.
 */
export class OcrCompare {
  constructor({ fetchJson, esc, engines }) {
    this.fetchJson = fetchJson;
    this.esc = esc;
    this.docId = null;
    this.runs = [];
    this.els = {
      section: document.getElementById('ocr-compare-section'),
      title:   document.getElementById('ocr-compare-title'),
      close:   document.getElementById('ocr-compare-close'),
      grid:    document.getElementById('ocr-compare-grid'),
      selA:    document.getElementById('ocr-compare-a'),
      selB:    document.getElementById('ocr-compare-b'),
      labelA:  document.getElementById('ocr-compare-a-label'),
      labelB:  document.getElementById('ocr-compare-b-label'),
      mdA:     document.getElementById('ocr-compare-a-md'),
      mdB:     document.getElementById('ocr-compare-b-md'),
      rerunEngine: document.getElementById('ocr-compare-rerun-engine'),
      rerun:   document.getElementById('ocr-compare-rerun'),
    };
    if (!this.els.section) return;
    this.els.rerunEngine.innerHTML = (engines || []).map(
      (name) => `<option value="${this.esc(name)}">${this.esc(name)}</option>`).join('');
    this.els.close.addEventListener('click', () => this.closeCompare());
    this.els.selA.addEventListener('change', () => this.renderDiff());
    this.els.selB.addEventListener('change', () => this.renderDiff());
    this.els.rerun.addEventListener('click', () => this.rerun());
  }

  async rerun() {
    if (this.docId === null) return;
    try {
      await this.fetchJson(`${API_BASE}/api/ocr/documents/${this.docId}/runs`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ engine: this.els.rerunEngine.value }),
      });
      // WS run events refresh the grid as the new run progresses.
    } catch (err) {
      this.els.title.textContent += ` — re-run failed: ${err.message}`;
    }
  }

  async open(docId) {
    this.docId = docId;
    await this.refresh();
    this.els.section.style.display = '';
    this.els.section.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  closeCompare() {
    this.docId = null;
    this.els.section.style.display = 'none';
  }

  closeIfDocument(docId) {
    if (this.docId === docId) this.closeCompare();
  }

  handleEvent(msg) {
    if (msg.type === 'document' && msg.status === 'deleted') {
      this.closeIfDocument(msg.id);
      return;
    }
    if (this.docId === null) return;
    if (msg.type === 'run' && msg.document_id === this.docId && msg.status !== 'running') {
      this.refresh();
    }
  }

  async refresh() {
    if (this.docId === null) return;
    let data;
    try {
      data = await this.fetchJson(`${API_BASE}/api/ocr/documents/${this.docId}/runs`);
    } catch (_) {
      return;
    }
    const esc = this.esc;
    this.runs = data.runs;
    this.els.title.textContent = `${data.document.filename} — ${data.runs.length} run(s)`;

    this.els.grid.innerHTML = `<table class="ocr-table"><thead><tr>
        <th>Run</th><th>Engine</th><th>Version</th><th>Status</th>
        <th>ms/page</th><th>Validation</th></tr></thead><tbody>` +
      data.runs.map((r) => {
        const pages = r.pages_done || 0;
        const msPage = r.ms_total != null && pages
          ? Math.round(r.ms_total / pages) : null;
        const ex = r.latest_extraction;
        const val = ex && ex.validation_status
          ? `<span class="ocr-badge ${esc(ex.validation_status)}">${esc(ex.validation_status)}</span>`
          : '—';
        return `<tr>
          <td>#${r.id}</td>
          <td><span class="ocr-badge engine">${esc(r.engine)}</span></td>
          <td style="color:#a0a0a0;">${esc(r.engine_version || '')}</td>
          <td><span class="ocr-badge ${esc(r.status)}">${esc(r.status)}</span></td>
          <td style="color:#a0a0a0;">${msPage != null ? msPage : ''}</td>
          <td>${val}</td></tr>`;
      }).join('') + '</tbody></table>';

    const done = data.runs.filter((r) => r.status === 'done');
    for (const sel of [this.els.selA, this.els.selB]) {
      const prev = sel.value;
      sel.innerHTML = done.map((r) =>
        `<option value="${r.id}">#${r.id} ${esc(r.engine)}</option>`).join('');
      if (prev && done.some((r) => String(r.id) === prev)) sel.value = prev;
    }
    if (done.length >= 2 && this.els.selA.value === this.els.selB.value) {
      this.els.selA.value = String(done[1].id);
      this.els.selB.value = String(done[0].id);
    }
    await this.renderDiff();
  }

  async renderDiff() {
    const a = this.els.selA.value;
    const b = this.els.selB.value;
    if (!a || !b) {
      this.els.mdA.textContent = '(need two completed runs)';
      this.els.mdB.textContent = '';
      return;
    }
    const runA = this.runs.find((r) => String(r.id) === a);
    const runB = this.runs.find((r) => String(r.id) === b);
    this.els.labelA.textContent = `Run #${a} (${runA ? runA.engine : '?'})`;
    this.els.labelB.textContent = `Run #${b} (${runB ? runB.engine : '?'})`;
    const [textA, textB] = await Promise.all([
      fetch(`${API_BASE}/api/ocr/runs/${a}/markdown`).then((r) => r.ok ? r.text() : ''),
      fetch(`${API_BASE}/api/ocr/runs/${b}/markdown`).then((r) => r.ok ? r.text() : ''),
    ]);
    const { onlyA, onlyB } = lineDiff(textA, textB);
    this.els.mdA.innerHTML = renderDiffLines(textA, onlyA, 'ocr-diff-del', this.esc);
    this.els.mdB.innerHTML = renderDiffLines(textB, onlyB, 'ocr-diff-add', this.esc);
  }
}

/** LCS line diff: returns the sets of line indices unique to each side. */
function lineDiff(textA, textB) {
  const a = textA.split('\n');
  const b = textB.split('\n');
  const n = a.length;
  const m = b.length;
  // Cap the O(n·m) table for very large documents.
  if (n * m > 4_000_000) {
    return { onlyA: new Set(), onlyB: new Set() };
  }
  const dp = Array.from({ length: n + 1 }, () => new Int32Array(m + 1));
  for (let i = n - 1; i >= 0; i--) {
    for (let j = m - 1; j >= 0; j--) {
      dp[i][j] = a[i] === b[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
    }
  }
  const onlyA = new Set();
  const onlyB = new Set();
  let i = 0, j = 0;
  while (i < n && j < m) {
    if (a[i] === b[j]) { i++; j++; }
    else if (dp[i + 1][j] >= dp[i][j + 1]) { onlyA.add(i); i++; }
    else { onlyB.add(j); j++; }
  }
  while (i < n) { onlyA.add(i); i++; }
  while (j < m) { onlyB.add(j); j++; }
  return { onlyA, onlyB };
}

function renderDiffLines(text, marked, cls, esc) {
  return text.split('\n').map((line, idx) =>
    marked.has(idx)
      ? `<span class="${cls}">${esc(line) || '&nbsp;'}</span>`
      : `${esc(line)}\n`
  ).join('');
}
