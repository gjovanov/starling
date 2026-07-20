/**
 * OCR tab controller: config bootstrap, uploads, WS live updates, engines panel.
 * Served by ocr-server (:8092); the tab hides itself when /api/ocr/config is absent.
 */
import { OcrJobs } from './ocr-jobs.js';
import { OcrCompare, OcrViewer } from './ocr-viewer.js';

const API_BASE = ''; // same origin

async function fetchJson(url, options) {
  const r = await fetch(url, options);
  let body;
  try { body = await r.json(); } catch (_) { body = null; }
  if (!r.ok || (body && body.success === false)) {
    const msg = (body && (body.error || body.detail)) || `${r.status} ${r.statusText}`;
    const err = new Error(msg);
    err.status = r.status;
    throw err;
  }
  return body && 'data' in body ? body.data : body;
}

export function esc(s) {
  return String(s).replace(/[&<>"']/g, c => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
  ));
}

const ocrState = {
  els: null,
  jobs: null,
  viewer: null,
  ws: null,
  wsAttempts: 0,
  wsClosedByUs: false,
};

export async function setupOcrTab() {
  const els = {
    uploadZone:   document.getElementById('ocr-upload-zone'),
    fileInput:    document.getElementById('ocr-file-input'),
    engineSelect: document.getElementById('ocr-engine-select'),
    languageHint: document.getElementById('ocr-language-hint'),
    autoExtract:  document.getElementById('ocr-auto-extract'),
    wsBadge:      document.getElementById('ocr-ws-badge'),
    uploadStatus: document.getElementById('ocr-upload-status'),
    jobs:         document.getElementById('ocr-jobs'),
    engines:      document.getElementById('ocr-engines'),
  };
  if (!els.uploadZone || !els.jobs) return; // pane not present in DOM
  ocrState.els = els;

  // Throws when this backend has no OCR routes — caller hides the tab.
  const cfg = await fetchJson(`${API_BASE}/api/ocr/config`);

  els.engineSelect.innerHTML = '';
  for (const name of cfg.engines) {
    const opt = document.createElement('option');
    opt.value = name;
    opt.textContent = name;
    if (name === cfg.default_engine) opt.selected = true;
    els.engineSelect.appendChild(opt);
  }
  els.languageHint.value = cfg.language_hint || 'bg,en';
  els.autoExtract.checked = !!cfg.extract_auto;

  ocrState.viewer = new OcrViewer({ fetchJson, esc });
  ocrState.compare = new OcrCompare({ fetchJson, esc, engines: cfg.engines });
  ocrState.jobs = new OcrJobs(els.jobs, {
    fetchJson,
    esc,
    onView: (runId) => ocrState.viewer.open(runId),
    onCompare: (docId) => ocrState.compare.open(docId),
    onDelete: (docId, name) => deleteDocument(docId, name),
  });

  const docs = await fetchJson(`${API_BASE}/api/ocr/documents`);
  ocrState.jobs.setInitial(docs.documents);

  setupUpload(els);
  await refreshEngines();
  connectWs();
}

// ── upload ─────────────────────────────────────────────────────────

function setupUpload(els) {
  els.uploadZone.addEventListener('click', () => els.fileInput.click());
  els.uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    e.stopPropagation();
    els.uploadZone.classList.add('dragover');
  });
  els.uploadZone.addEventListener('dragleave', (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (!els.uploadZone.contains(e.relatedTarget)) {
      els.uploadZone.classList.remove('dragover');
    }
  });
  els.uploadZone.addEventListener('drop', async (e) => {
    e.preventDefault();
    e.stopPropagation();
    els.uploadZone.classList.remove('dragover');
    const files = Array.from(e.dataTransfer?.files || []);
    if (files.length) await uploadFiles(files);
  });
  els.fileInput.addEventListener('change', async (e) => {
    const files = Array.from(e.target.files || []);
    if (files.length) await uploadFiles(files);
    e.target.value = '';
  });
}

async function uploadFiles(files) {
  const { els } = ocrState;
  els.uploadStatus.textContent = `Uploading ${files.length} file(s)…`;
  const fd = new FormData();
  for (const f of files) fd.append('files', f, f.name);
  fd.append('engine', els.engineSelect.value);
  fd.append('extract', els.autoExtract.checked ? 'auto' : 'off');
  fd.append('language_hint', els.languageHint.value || 'bg,en');
  try {
    const data = await fetchJson(`${API_BASE}/api/ocr/documents`, { method: 'POST', body: fd });
    const ok = data.documents.length;
    const bad = (data.errors || []).length;
    els.uploadStatus.textContent =
      `Queued ${ok} document(s)` + (bad ? `; ${bad} rejected (${data.errors[0].error})` : '');
    // WS events drive the table, but hydrate immediately for responsiveness.
    for (const d of data.documents) {
      ocrState.jobs.hydrate(d.run_id);
    }
  } catch (err) {
    els.uploadStatus.textContent = `Upload failed: ${err.message}`;
  }
}

async function resyncJobs() {
  try {
    const docs = await fetchJson(`${API_BASE}/api/ocr/documents`);
    ocrState.jobs.resync(docs.documents);
  } catch (_) {
    // Next WS event or a manual refresh will reconcile instead.
  }
}

// ── delete ─────────────────────────────────────────────────────────

async function deleteDocument(docId, name) {
  if (!window.confirm(`Delete "${name}" and all its runs? This cannot be undone.`)) return;
  try {
    await fetchJson(`${API_BASE}/api/ocr/documents/${docId}`, { method: 'DELETE' });
    ocrState.jobs.removeDocument(docId);      // optimistic; WS event is idempotent
    if (ocrState.viewer) ocrState.viewer.closeIfDocument(docId);
    if (ocrState.compare) ocrState.compare.closeIfDocument(docId);
  } catch (err) {
    ocrState.els.uploadStatus.textContent = `Delete failed: ${err.message}`;
  }
}

// ── engines panel ──────────────────────────────────────────────────

async function refreshEngines() {
  const { els } = ocrState;
  let data;
  try {
    data = await fetchJson(`${API_BASE}/api/ocr/engines`);
  } catch (_) {
    return;
  }
  els.engines.innerHTML = '';
  for (const e of data.engines) {
    const row = document.createElement('div');
    row.className = 'ocr-engine-row';
    row.dataset.engine = e.name;
    row.innerHTML = `
      <span class="ocr-badge engine">${esc(e.name)}</span>
      <span class="grow" style="color:#a0a0a0;">${esc(e.kind)} · ${esc(e.vram_class)}</span>
      <span class="ocr-badge ${e.status === 'ready' ? 'done' : ''}" data-role="status">${esc(e.status)}</span>
      <button class="btn-small btn-primary" data-role="load">Load</button>
      <button class="btn-small btn-danger" data-role="unload">Unload</button>`;
    row.querySelector('[data-role="load"]').addEventListener('click', async () => {
      try { await fetchJson(`${API_BASE}/api/ocr/engines/${encodeURIComponent(e.name)}/load`, { method: 'POST' }); } catch (_) {}
    });
    row.querySelector('[data-role="unload"]').addEventListener('click', async () => {
      try { await fetchJson(`${API_BASE}/api/ocr/engines/${encodeURIComponent(e.name)}/unload`, { method: 'POST' }); } catch (_) {}
    });
    els.engines.appendChild(row);
  }
  if (data.extractor) {
    const ex = data.extractor;
    const row = document.createElement('div');
    row.className = 'ocr-engine-row';
    row.dataset.engine = 'extractor';
    row.innerHTML = `
      <span class="ocr-badge engine">extractor</span>
      <span class="grow" style="color:#a0a0a0;">${esc(ex.model)} · stage 2 · :${ex.port}</span>
      <span class="ocr-badge ${ex.status === 'ready' ? 'done' : ''}" data-role="status">${esc(ex.status)}</span>`;
    els.engines.appendChild(row);
  }
}

function updateEngineBadge(name, status) {
  const row = ocrState.els.engines.querySelector(`[data-engine="${CSS.escape(name)}"]`);
  if (!row) { refreshEngines(); return; }
  const badge = row.querySelector('[data-role="status"]');
  badge.textContent = status;
  badge.className = 'ocr-badge ' +
    (status === 'ready' ? 'done' : status === 'loading' ? 'running' : '');
}

// ── WebSocket live updates ─────────────────────────────────────────

function connectWs() {
  const { els } = ocrState;
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WebSocket(`${proto}://${location.host}/api/ocr/ws`);
  ocrState.ws = ws;

  ws.onopen = () => {
    els.wsBadge.textContent = 'live';
    els.wsBadge.className = 'ocr-badge live';
    // Events emitted while the socket was down are lost — reconcile the jobs
    // table with server truth so no row gets stuck (e.g. an OCR run that
    // finished during the gap). Skip on the very first connect (setupOcrTab
    // already seeded the table).
    if (ocrState.wsAttempts > 0 || ocrState.wsConnectedOnce) {
      resyncJobs();
    }
    ocrState.wsConnectedOnce = true;
    ocrState.wsAttempts = 0;
  };
  ws.onmessage = (event) => {
    let msg;
    try { msg = JSON.parse(event.data); } catch (_) { return; }
    if (msg.type === 'ping') return;
    if (msg.type === 'run' || msg.type === 'extraction' || msg.type === 'document') {
      ocrState.jobs.handleEvent(msg);
      ocrState.viewer.handleEvent(msg);
      if (ocrState.compare) ocrState.compare.handleEvent(msg);
    } else if (msg.type === 'engine') {
      updateEngineBadge(msg.name, msg.status);
    }
  };
  ws.onclose = () => {
    els.wsBadge.textContent = 'offline';
    els.wsBadge.className = 'ocr-badge offline';
    if (ocrState.wsClosedByUs) return;
    // Exponential backoff, same shape as modules/websocket.js.
    const delay = Math.min(2000 * Math.pow(1.5, ocrState.wsAttempts), 30000);
    ocrState.wsAttempts += 1;
    setTimeout(() => connectWs(), delay);
  };
  ws.onerror = () => { /* onclose follows and handles reconnect */ };
}
