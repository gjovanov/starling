/**
 * WASM Inference Worker — runs Voxtral model inference in a Web Worker
 * using wasm-bindgen bindings.
 *
 * This must be loaded as a module worker:
 *   new Worker('./js/wasm/worker.js', { type: 'module' })
 *
 * Messages IN:
 *   { type: 'init' }                                          — Initialize WASM runtime (WebGPU)
 *   { type: 'load', shards: Uint8Array[], tokenizerJson: string } — Load model from pre-fetched shards
 *   { type: 'audio', samples: Float32Array }                  — Send audio chunk (16kHz mono f32)
 *   { type: 'commit' }                                        — Trigger inference, returns text delta
 *   { type: 'reset' }                                         — Reset session state
 *
 * Messages OUT:
 *   { type: 'ready' }                                         — Runtime initialized
 *   { type: 'loaded' }                                        — Model loaded and ready
 *   { type: 'text', delta: string, fullText: string }         — Transcription result
 *   { type: 'error', message: string }                        — Error
 */

import init, { init_runtime, WasmEngine } from '../../wasm/pkg/wasm_engine.js';

/** @type {WasmEngine|null} */
let engine = null;

/** Accumulated full text across commits */
let fullText = '';

self.onmessage = async function (e) {
  const { type } = e.data;

  switch (type) {
    case 'init':
      await handleInit();
      break;
    case 'load':
      await handleLoad(e.data.shards, e.data.tokenizerJson);
      break;
    case 'audio':
      handleAudio(e.data.samples);
      break;
    case 'commit':
      await handleCommit();
      break;
    case 'reset':
      handleReset();
      break;
    default:
      self.postMessage({ type: 'error', message: `[wasm] Unknown message type: ${type}` });
  }
};

/**
 * Initialize the WASM module and WebGPU runtime.
 */
async function handleInit() {
  try {
    console.log('[wasm] Initializing WASM module...');
    await init();
    console.log('[wasm] WASM module loaded, initializing WebGPU runtime...');

    const adapterInfo = await init_runtime();
    console.log(`[wasm] WebGPU runtime ready: ${adapterInfo}`);

    self.postMessage({ type: 'ready' });
  } catch (err) {
    console.error('[wasm] Init failed:', err);
    self.postMessage({ type: 'error', message: `Init failed: ${err.message}` });
  }
}

/**
 * Load model from pre-fetched shards and tokenizer JSON.
 * @param {Uint8Array[]} shards - Array of 64MB GGUF shard Uint8Arrays
 * @param {string} tokenizerJson - Contents of tekken.json
 */
async function handleLoad(shards, tokenizerJson) {
  try {
    if (!shards || shards.length === 0) {
      throw new Error('No model shards provided');
    }
    if (!tokenizerJson) {
      throw new Error('No tokenizer JSON provided');
    }

    console.log(`[wasm] Creating engine from ${shards.length} shard(s)...`);
    engine = await WasmEngine.create(shards, tokenizerJson);
    fullText = '';
    console.log('[wasm] Engine created and model loaded');

    self.postMessage({ type: 'loaded' });
  } catch (err) {
    console.error('[wasm] Load failed:', err);
    self.postMessage({ type: 'error', message: `Load failed: ${err.message}` });
  }
}

/**
 * Buffer audio samples for the next commit.
 * @param {Float32Array} samples - 16kHz mono f32 PCM samples
 */
function handleAudio(samples) {
  if (!engine) {
    self.postMessage({ type: 'error', message: 'Engine not loaded — cannot accept audio' });
    return;
  }

  try {
    console.log(`[wasm] send_audio: ${samples.length} samples (${(samples.length / 16000).toFixed(2)}s)`);
    engine.send_audio(samples);
  } catch (err) {
    console.error('[wasm] send_audio failed:', err);
    self.postMessage({ type: 'error', message: `send_audio failed: ${err.message}` });
  }
}

/**
 * Run inference on buffered audio and return the text delta.
 */
async function handleCommit() {
  if (!engine) {
    self.postMessage({ type: 'error', message: 'Engine not loaded — cannot commit' });
    return;
  }

  try {
    const t0 = performance.now();
    console.log('[wasm] commit: starting inference...');
    const delta = await engine.commit();
    const elapsed = (performance.now() - t0).toFixed(0);
    if (delta) {
      fullText += delta;
    }

    console.log(`[wasm] commit: done in ${elapsed}ms, delta="${(delta || '').slice(0, 60)}"`);
    self.postMessage({
      type: 'text',
      delta: delta || '',
      fullText: fullText,
    });
  } catch (err) {
    console.error('[wasm] commit failed:', err);
    self.postMessage({ type: 'error', message: `commit failed: ${err.message}` });
  }
}

/**
 * Reset session state — clears audio buffer and text history.
 */
function handleReset() {
  if (engine) {
    try {
      engine.reset();
    } catch (err) {
      console.error('[wasm] reset failed:', err);
    }
  }
  fullText = '';
  console.log('[wasm] Session reset');
}
