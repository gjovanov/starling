/**
 * TTS tab manager — fetches voice catalog + TTS config, runs synthesis
 * requests, and renders the saved-files list.
 *
 * The backend exposes `/api/tts/*` (see voxtral_server/api/tts_routes.py).
 * Phase 1: save-to-disk only. Phase 2 will add streaming playback.
 */

const API_BASE = ''; // same origin

async function fetchJson(url, options) {
  const r = await fetch(url, options);
  let body;
  try { body = await r.json(); } catch (_) { body = null; }
  if (!r.ok || (body && body.success === false)) {
    const msg = (body && body.error) || `${r.status} ${r.statusText}`;
    const err = new Error(msg);
    err.status = r.status;
    throw err;
  }
  return body && 'data' in body ? body.data : body;
}

export async function fetchVoices() {
  return fetchJson(`${API_BASE}/api/tts/voices`);
}

export async function fetchTtsConfig() {
  return fetchJson(`${API_BASE}/api/tts/config`);
}

/** Lightweight lifecycle snapshot — used by the TTS engine status badge. */
export async function fetchTtsStatus() {
  return fetchJson(`${API_BASE}/api/tts/status`);
}

export async function fetchSavedFiles() {
  return fetchJson(`${API_BASE}/api/tts/output`);
}

export async function deleteSavedFile(filename) {
  // Caller must already have a sanitizer-passing filename.
  return fetchJson(`${API_BASE}/api/tts/output/${encodeURIComponent(filename)}`, {
    method: 'DELETE',
  });
}

/** Save-to-server synthesis. Returns { filename, path, bytes, ... }. */
export async function synthesizeAndSave({ text, voice, voiceRefId, filename, overwrite }) {
  return fetchJson(`${API_BASE}/api/tts/synthesize`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      text,
      voice,
      voice_ref_id: voiceRefId || null,
      save: true,
      save_filename: filename || null,
      overwrite: !!overwrite,
    }),
  });
}

/** Multipart upload of a reference audio + transcript for voice cloning. */
export async function uploadVoiceRef({ file, name, refText, permissionConfirmed }) {
  const fd = new FormData();
  fd.append('audio_sample', file, file.name);
  fd.append('name', name);
  fd.append('ref_text', refText);
  fd.append('permission_confirmed', permissionConfirmed ? 'true' : 'false');
  const r = await fetch(`${API_BASE}/api/tts/voices/upload`, {
    method: 'POST',
    body: fd,
  });
  let body;
  try { body = await r.json(); } catch (_) { body = null; }
  if (!r.ok || (body && body.success === false)) {
    const msg = (body && body.error) || `${r.status} ${r.statusText}`;
    const err = new Error(msg);
    err.status = r.status;
    throw err;
  }
  return body && 'data' in body ? body.data : body;
}

/** Delete an uploaded voice reference. */
export async function deleteVoiceRef(refId) {
  return fetchJson(`${API_BASE}/api/tts/voices/${encodeURIComponent(refId)}`, {
    method: 'DELETE',
  });
}

/**
 * Begin a play-in-browser synthesis via the streaming endpoint.
 *
 * Sends `save=false` so the server returns a chunked audio/wav response
 * (24 kHz mono PCM wrapped in a streaming WAV header). Returns the raw
 * `Response` so the caller can hand it to `TtsPlayer.start()` for
 * progressive playback (Phase 4 — Web Audio).
 *
 * Throws on transport error or any 4xx/5xx with an `Error.status` field.
 */
export async function synthesizeForPlayback({ text, voice, voiceRefId }) {
  const r = await fetch(`${API_BASE}/api/tts/synthesize`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      text,
      voice,
      voice_ref_id: voiceRefId || null,
      save: false,
    }),
  });
  if (!r.ok) {
    let msg;
    try { msg = (await r.json()).error || `${r.status} ${r.statusText}`; }
    catch (_) { msg = `${r.status} ${r.statusText}`; }
    const err = new Error(msg);
    err.status = r.status;
    throw err;
  }
  return r;
}
