/**
 * WASM UI — adds WASM mode toggle, progress bar, and WebGPU status indicator
 * to the session creation form.
 */

import { checkWebGPUSupport } from './loader.js';

let progressBarEl = null;
let progressTextEl = null;
let webgpuStatusEl = null;
let wasmCheckboxEl = null;
let wasmSectionEl = null;

/**
 * Initialize the WASM UI elements inside the given container.
 * Inserts a WASM mode section with:
 *   - "Use Browser" checkbox
 *   - WebGPU status indicator
 *   - Progress bar for model download
 *
 * @param {HTMLElement} container - The form container to append WASM UI into
 */
export async function initWasmUI(container) {
  if (!container) return;

  // Create the WASM section wrapper
  wasmSectionEl = document.createElement('div');
  wasmSectionEl.id = 'wasm-mode-section';
  wasmSectionEl.style.display = 'none';

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

  // Progress bar
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

  container.appendChild(wasmSectionEl);

  // Cache references
  wasmCheckboxEl = document.getElementById('wasm-browser-toggle');
  webgpuStatusEl = document.getElementById('webgpu-status');
  progressBarEl = document.getElementById('wasm-progress-bar');
  progressTextEl = document.getElementById('wasm-progress-text');

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
