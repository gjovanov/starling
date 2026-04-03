/**
 * WASM UI — adds WASM mode toggle, progress bar, WebGPU status indicator,
 * audio file input, and transcription controls to the session creation form.
 */

import { checkWebGPUSupport } from './loader.js';
import { WasmSession } from './wasm-session.js';
import { decodeToMono16k } from './audio-decode.js';

/** Default URLs — match burn-server's /models nest_service (models_dir = ../../models/cache) */
const DEFAULT_CHUNKS_URL = '/models/q4/chunks';
const DEFAULT_TOKENIZER_URL = '/models/tokenizer/tekken.json';

let progressBarEl = null;
let progressTextEl = null;
let webgpuStatusEl = null;
let wasmCheckboxEl = null;
let wasmSectionEl = null;
let wasmFileInputEl = null;
let wasmStartBtnEl = null;
let wasmStopBtnEl = null;
let wasmTranscriptEl = null;
let wasmStatsEl = null;

/** @type {WasmSession|null} */
let wasmSession = null;

/** @type {Float32Array|null} */
let decodedPcm = null;

/**
 * Initialize the WASM UI elements inside the given container.
 * Inserts a WASM mode section with:
 *   - "Use Browser" checkbox
 *   - WebGPU status indicator
 *   - Progress bar for model download
 *   - Audio file picker for local transcription
 *   - Start/Stop transcription buttons
 *   - Transcription output area
 *
 * @param {HTMLElement} container - The form container to append WASM UI into
 */
export async function initWasmUI(container) {
  if (!container) return;

  // Create the WASM section wrapper
  wasmSectionEl = document.createElement('div');
  wasmSectionEl.id = 'wasm-mode-section';

  // "Use Browser" checkbox
  const checkboxGroup = document.createElement('div');
  checkboxGroup.className = 'form-group';
  checkboxGroup.innerHTML = `
    <label>
      <input type="checkbox" id="wasm-browser-toggle">
      Use Browser (WASM inference)
    </label>
    <small style="color: #666;">Run model locally in the browser via WebAssembly + WebGPU</small>
  `;
  wasmSectionEl.appendChild(checkboxGroup);

  // WebGPU status indicator
  const statusGroup = document.createElement('div');
  statusGroup.className = 'form-group';
  statusGroup.innerHTML = `
    <label>WebGPU Status</label>
    <span id="webgpu-status" style="padding: 4px 10px; border-radius: 12px; font-size: 0.85em; font-weight: 500;">Checking...</span>
  `;
  wasmSectionEl.appendChild(statusGroup);

  // Progress bar (for model download and loading)
  const progressGroup = document.createElement('div');
  progressGroup.className = 'form-group';
  progressGroup.id = 'wasm-progress-group';
  progressGroup.style.display = 'none';
  progressGroup.innerHTML = `
    <label>Model Download</label>
    <div class="session-progress" style="height: 6px;">
      <div id="wasm-progress-bar" class="session-progress-bar" style="width: 0%;"></div>
    </div>
    <small id="wasm-progress-text" style="color: #888;">0%</small>
  `;
  wasmSectionEl.appendChild(progressGroup);

  // WASM controls (hidden until checkbox is checked)
  const controlsGroup = document.createElement('div');
  controlsGroup.id = 'wasm-controls-group';
  controlsGroup.style.display = 'none';
  controlsGroup.innerHTML = `
    <div class="form-group">
      <label for="wasm-audio-file">Audio File (for local transcription)</label>
      <input type="file" id="wasm-audio-file" accept=".wav,.mp3,.ogg,.flac,.m4a,.webm"
             style="padding: 8px; border-radius: 8px; border: 1px solid #333; background: #16213e; color: white; width: 100%;">
      <small style="color: #666;">Select an audio file to transcribe entirely in the browser</small>
    </div>
    <div class="form-group" id="wasm-decode-status" style="display: none;">
      <small id="wasm-decode-text" style="color: #888;">Decoding audio...</small>
    </div>
    <div class="form-actions" style="margin-top: 10px;">
      <button id="wasm-start-btn" class="action-btn" disabled>Start Transcription</button>
      <button id="wasm-stop-btn" class="action-btn small danger" style="display: none;">Stop</button>
    </div>
    <div class="form-group" style="margin-top: 10px;">
      <div id="wasm-stats" style="font-size: 0.85em; color: #888; display: none;">
        Commits: <span id="wasm-commit-count">0</span> &middot;
        Elapsed: <span id="wasm-elapsed-time">0.0s</span>
      </div>
    </div>
    <div class="form-group" style="margin-top: 10px;">
      <label>WASM Transcription Output</label>
      <div id="wasm-transcript"
           style="background: #16213e; border-radius: 8px; padding: 15px; min-height: 80px;
                  max-height: 300px; overflow-y: auto; color: #e0e0e0; font-size: 0.95em;
                  white-space: pre-wrap; word-wrap: break-word; border: 1px solid #333;">
        <span style="color: #666;">Transcription will appear here...</span>
      </div>
    </div>
  `;
  wasmSectionEl.appendChild(controlsGroup);

  container.appendChild(wasmSectionEl);

  // Cache references
  wasmCheckboxEl = document.getElementById('wasm-browser-toggle');
  webgpuStatusEl = document.getElementById('webgpu-status');
  progressBarEl = document.getElementById('wasm-progress-bar');
  progressTextEl = document.getElementById('wasm-progress-text');
  wasmFileInputEl = document.getElementById('wasm-audio-file');
  wasmStartBtnEl = document.getElementById('wasm-start-btn');
  wasmStopBtnEl = document.getElementById('wasm-stop-btn');
  wasmTranscriptEl = document.getElementById('wasm-transcript');
  wasmStatsEl = document.getElementById('wasm-stats');

  // Checkbox toggle — show/hide controls
  wasmCheckboxEl.addEventListener('change', () => {
    const controls = document.getElementById('wasm-controls-group');
    if (controls) {
      controls.style.display = wasmCheckboxEl.checked ? 'block' : 'none';
    }
    console.log(`[wasm] Browser mode ${wasmCheckboxEl.checked ? 'enabled' : 'disabled'}`);
  });

  // File input — decode audio when a file is selected
  wasmFileInputEl.addEventListener('change', async () => {
    const file = wasmFileInputEl.files[0];
    if (!file) return;

    decodedPcm = null;
    wasmStartBtnEl.disabled = true;

    const decodeStatus = document.getElementById('wasm-decode-status');
    const decodeText = document.getElementById('wasm-decode-text');
    decodeStatus.style.display = 'block';
    decodeText.textContent = `Decoding "${file.name}"...`;
    decodeText.style.color = '#888';

    try {
      decodedPcm = await decodeToMono16k(file);
      const durationS = decodedPcm.length / 16000;
      decodeText.textContent = `Decoded: ${durationS.toFixed(1)}s of audio (${(decodedPcm.length).toLocaleString()} samples)`;
      decodeText.style.color = '#28a745';
      wasmStartBtnEl.disabled = false;
      console.log(`[wasm] Audio decoded: ${durationS.toFixed(1)}s`);
    } catch (err) {
      console.error('[wasm] Audio decode failed:', err);
      decodeText.textContent = `Decode failed: ${err.message}`;
      decodeText.style.color = '#dc3545';
    }
  });

  // Start button — load model (if needed) and start transcription
  wasmStartBtnEl.addEventListener('click', async () => {
    if (!decodedPcm) {
      console.warn('[wasm] No audio decoded yet');
      return;
    }
    await startWasmTranscription();
  });

  // Stop button
  wasmStopBtnEl.addEventListener('click', () => {
    stopWasmTranscription();
  });

  // Check WebGPU support and update status
  const gpuInfo = await checkWebGPUSupport();
  if (gpuInfo.supported) {
    webgpuStatusEl.textContent = `Supported (${gpuInfo.adapter})`;
    webgpuStatusEl.style.background = '#28a745';
    webgpuStatusEl.style.color = 'white';
  } else {
    webgpuStatusEl.textContent = `Not available: ${gpuInfo.error}`;
    webgpuStatusEl.style.background = '#dc3545';
    webgpuStatusEl.style.color = 'white';
    if (wasmCheckboxEl) {
      wasmCheckboxEl.disabled = true;
    }
  }
}

/**
 * Start WASM transcription — initializes runtime, loads model, and streams audio.
 */
async function startWasmTranscription() {
  if (!decodedPcm) return;

  wasmStartBtnEl.disabled = true;
  wasmStopBtnEl.style.display = 'inline-block';
  wasmTranscriptEl.innerHTML = '';
  wasmStatsEl.style.display = 'block';
  updateStats(0, 0);

  try {
    // Create session if needed
    if (!wasmSession) {
      wasmSession = new WasmSession();

      wasmSession.on('subtitle', (data) => {
        // Update transcript display
        wasmTranscriptEl.textContent = data.fullText || '';
        wasmTranscriptEl.scrollTop = wasmTranscriptEl.scrollHeight;
        updateStats(data.commitCount, data.elapsedMs);
      });

      wasmSession.on('status', (data) => {
        console.log(`[wasm] Status: ${data.state} — ${data.message}`);
        if (data.state === 'completed') {
          wasmStartBtnEl.disabled = false;
          wasmStopBtnEl.style.display = 'none';
        }
      });

      wasmSession.on('error', (data) => {
        console.error('[wasm] Session error:', data.message);
        wasmTranscriptEl.innerHTML += `\n<span style="color: #dc3545;">[Error: ${data.message}]</span>`;
        wasmStartBtnEl.disabled = false;
        wasmStopBtnEl.style.display = 'none';
      });
    }

    // Initialize runtime if not already done
    if (!wasmSession.runtimeReady) {
      await wasmSession.initRuntime();
    }

    // Load model if not already loaded
    if (!wasmSession.modelLoaded) {
      showProgress(0, 0);
      await wasmSession.loadModel(DEFAULT_CHUNKS_URL, DEFAULT_TOKENIZER_URL, (loaded, total, stage) => {
        if (stage === 'chunks') {
          showProgress(loaded, total);
        }
      });
      setModelLoaded(true);
    }

    // Start streaming
    wasmSession.startTranscription(decodedPcm);
  } catch (err) {
    console.error('[wasm] Start transcription failed:', err);
    wasmTranscriptEl.innerHTML = `<span style="color: #dc3545;">Failed to start: ${err.message}</span>`;
    wasmStartBtnEl.disabled = false;
    wasmStopBtnEl.style.display = 'none';
  }
}

/**
 * Stop the current WASM transcription.
 */
function stopWasmTranscription() {
  if (wasmSession) {
    wasmSession.stop();
  }
  wasmStartBtnEl.disabled = false;
  wasmStopBtnEl.style.display = 'none';
  console.log('[wasm] Transcription stopped by user');
}

/**
 * Update the stats display.
 * @param {number} commitCount
 * @param {number} elapsedMs
 */
function updateStats(commitCount, elapsedMs) {
  const commitEl = document.getElementById('wasm-commit-count');
  const elapsedEl = document.getElementById('wasm-elapsed-time');
  if (commitEl) commitEl.textContent = commitCount;
  if (elapsedEl) elapsedEl.textContent = `${(elapsedMs / 1000).toFixed(1)}s`;
}

/**
 * Update the progress bar with download progress.
 * @param {number} loaded - Bytes loaded so far
 * @param {number} total - Total bytes expected
 */
export function showProgress(loaded, total) {
  const progressGroup = document.getElementById('wasm-progress-group');
  if (progressGroup) {
    progressGroup.style.display = 'block';
  }
  if (progressBarEl) {
    const pct = total > 0 ? Math.round((loaded / total) * 100) : 0;
    progressBarEl.style.width = `${pct}%`;
  }
  if (progressTextEl) {
    const loadedMB = (loaded / (1024 * 1024)).toFixed(1);
    const totalMB = (total / (1024 * 1024)).toFixed(1);
    const pct = total > 0 ? Math.round((loaded / total) * 100) : 0;
    progressTextEl.textContent = `${loadedMB} / ${totalMB} MB (${pct}%)`;
  }
}

/**
 * Toggle the loaded state of the WASM model.
 * @param {boolean} loaded - Whether the model has finished loading
 */
export function setModelLoaded(loaded) {
  const progressGroup = document.getElementById('wasm-progress-group');
  if (loaded) {
    if (progressBarEl) {
      progressBarEl.style.width = '100%';
    }
    if (progressTextEl) {
      progressTextEl.textContent = 'Model loaded and ready';
      progressTextEl.style.color = '#28a745';
    }
  } else {
    if (progressBarEl) {
      progressBarEl.style.width = '0%';
    }
    if (progressTextEl) {
      progressTextEl.textContent = '0%';
      progressTextEl.style.color = '#888';
    }
    if (progressGroup) {
      progressGroup.style.display = 'none';
    }
  }
}

/**
 * Show or hide the WASM mode section.
 * @param {boolean} visible
 */
export function setWasmSectionVisible(visible) {
  if (wasmSectionEl) {
    wasmSectionEl.style.display = visible ? 'block' : 'none';
  }
}

/**
 * @returns {boolean} Whether the "Use Browser" checkbox is checked
 */
export function isWasmModeEnabled() {
  return wasmCheckboxEl ? wasmCheckboxEl.checked : false;
}

/**
 * Get the current WasmSession instance (if any).
 * @returns {WasmSession|null}
 */
export function getWasmSession() {
  return wasmSession;
}
