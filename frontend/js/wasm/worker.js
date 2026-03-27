/**
 * WASM Inference Worker — loads a WASM module and runs transcription inference.
 *
 * Messages IN:
 *   { type: 'load', chunksUrl: string }     — Fetch model chunks and initialize WASM
 *   { type: 'transcribe', audio: Float32Array } — Run inference on audio samples
 *
 * Messages OUT:
 *   { type: 'progress', loaded: number, total: number } — Download progress
 *   { type: 'loaded' }                                   — Model ready
 *   { type: 'result', text: string }                     — Transcription result
 *   { type: 'error', message: string }                   — Error
 */

let wasmModule = null;
let modelReady = false;

self.onmessage = async function (e) {
  const { type } = e.data;

  switch (type) {
    case 'load':
      await handleLoad(e.data.chunksUrl);
      break;
    case 'transcribe':
      handleTranscribe(e.data.audio);
      break;
    default:
      self.postMessage({ type: 'error', message: `Unknown message type: ${type}` });
  }
};

async function handleLoad(chunksUrl) {
  try {
    // Dynamically import the loader (works in module workers)
    // For classic workers, the chunks are fetched inline
    const chunks = [];
    let loaded = 0;
    let total = 0;
    let chunkIndex = 0;
    const CHUNK_SIZE = 64 * 1024 * 1024;

    while (true) {
      const url = `${chunksUrl}/chunk_${chunkIndex}`;
      let res;
      try {
        res = await fetch(url);
      } catch (_) {
        break;
      }

      if (!res.ok) break;

      const contentLength = parseInt(res.headers.get('Content-Length') || '0', 10);
      if (chunkIndex === 0) {
        const chunkCount = parseInt(res.headers.get('X-Chunk-Count') || '0', 10);
        if (chunkCount > 0 && contentLength > 0) {
          total = chunkCount * contentLength;
        }
      }

      const buffer = await res.arrayBuffer();
      chunks.push(buffer);
      loaded += buffer.byteLength;
      if (total === 0) total = loaded;

      self.postMessage({ type: 'progress', loaded, total });
      chunkIndex++;
    }

    if (chunks.length === 0) {
      throw new Error(`No model chunks found at ${chunksUrl}`);
    }

    // Assemble model weights
    const assembled = new Uint8Array(loaded);
    let offset = 0;
    for (const chunk of chunks) {
      assembled.set(new Uint8Array(chunk), offset);
      offset += chunk.byteLength;
    }

    // Initialize WASM module with assembled weights
    // The actual WASM binary is expected at the same base path
    const wasmUrl = `${chunksUrl}/model.wasm`;
    const wasmResponse = await fetch(wasmUrl);
    if (!wasmResponse.ok) {
      throw new Error(`Failed to fetch WASM binary at ${wasmUrl}: ${wasmResponse.status}`);
    }
    const wasmBytes = await wasmResponse.arrayBuffer();

    wasmModule = await WebAssembly.instantiate(wasmBytes, {
      env: {
        memory: new WebAssembly.Memory({ initial: 256, maximum: 4096 }),
      },
    });

    // Store model weights in WASM memory if the module exports an init function
    if (wasmModule.instance.exports.init) {
      const weightsPtr = wasmModule.instance.exports.alloc(assembled.byteLength);
      const wasmMemory = new Uint8Array(wasmModule.instance.exports.memory.buffer);
      wasmMemory.set(assembled, weightsPtr);
      wasmModule.instance.exports.init(weightsPtr, assembled.byteLength);
    }

    modelReady = true;
    self.postMessage({ type: 'loaded' });
  } catch (err) {
    self.postMessage({ type: 'error', message: err.message });
  }
}

function handleTranscribe(audio) {
  if (!modelReady || !wasmModule) {
    self.postMessage({ type: 'error', message: 'Model not loaded yet' });
    return;
  }

  try {
    const exports = wasmModule.instance.exports;

    // Allocate space for audio samples in WASM memory
    const audioBytes = audio.byteLength;
    const audioPtr = exports.alloc(audioBytes);
    const wasmMemory = new Float32Array(exports.memory.buffer, audioPtr, audio.length);
    wasmMemory.set(audio);

    // Run inference
    const resultPtr = exports.transcribe(audioPtr, audio.length);

    // Read result string from WASM memory
    const memoryView = new Uint8Array(exports.memory.buffer);
    let end = resultPtr;
    while (memoryView[end] !== 0) end++;
    const textBytes = memoryView.slice(resultPtr, end);
    const text = new TextDecoder().decode(textBytes);

    // Free allocated memory
    if (exports.free) {
      exports.free(audioPtr);
      exports.free(resultPtr);
    }

    self.postMessage({ type: 'result', text });
  } catch (err) {
    self.postMessage({ type: 'error', message: err.message });
  }
}
