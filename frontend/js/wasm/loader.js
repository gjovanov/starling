/**
 * WASM Model Loader — fetches chunked model weights and checks WebGPU support.
 */

const CHUNK_SIZE = 64 * 1024 * 1024; // 64MB per chunk

/**
 * Check whether the browser supports WebGPU.
 * @returns {Promise<{supported: boolean, adapter: string|null, error: string|null}>}
 */
export async function checkWebGPUSupport() {
  if (!navigator.gpu) {
    return { supported: false, adapter: null, error: 'WebGPU API not available in this browser' };
  }
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      return { supported: false, adapter: null, error: 'No WebGPU adapter found' };
    }
    const info = adapter.info || {};
    const adapterName = info.device || info.description || 'unknown';
    return { supported: true, adapter: adapterName, error: null };
  } catch (e) {
    return { supported: false, adapter: null, error: e.message };
  }
}

/**
 * Fetch 64MB model weight chunks sequentially and assemble into a single ArrayBuffer.
 *
 * Chunks are expected at `${baseUrl}/chunk_0`, `${baseUrl}/chunk_1`, etc.
 * The server should return 404 once there are no more chunks.
 *
 * @param {string} baseUrl - Base URL for chunk files (e.g. "/models/q4")
 * @param {function(number, number, number): void} onProgress -
 *   Callback invoked after each chunk: (bytesLoaded, totalBytes, chunkIndex).
 *   `totalBytes` is estimated from Content-Length of the first chunk times expected count,
 *   or grows as chunks are discovered.
 * @returns {Promise<ArrayBuffer>} Assembled model weights
 */
export async function fetchModelChunks(baseUrl, onProgress) {
  const chunks = [];
  let loaded = 0;
  let total = 0;
  let chunkIndex = 0;

  while (true) {
    const url = `${baseUrl}/chunk_${chunkIndex}`;
    let res;
    try {
      res = await fetch(url);
    } catch (e) {
      // Network error — stop fetching
      break;
    }

    if (!res.ok) {
      // 404 or other error — no more chunks
      break;
    }

    const contentLength = parseInt(res.headers.get('Content-Length') || '0', 10);

    // On first chunk, estimate total if the server provides a chunk count header
    if (chunkIndex === 0) {
      const chunkCount = parseInt(res.headers.get('X-Chunk-Count') || '0', 10);
      if (chunkCount > 0 && contentLength > 0) {
        total = chunkCount * contentLength;
      }
    }

    const buffer = await res.arrayBuffer();
    chunks.push(buffer);
    loaded += buffer.byteLength;

    // If we had no total estimate, keep updating it as a running sum
    if (total === 0) {
      total = loaded;
    }

    if (onProgress) {
      onProgress(loaded, total, chunkIndex);
    }

    chunkIndex++;
  }

  if (chunks.length === 0) {
    throw new Error(`No model chunks found at ${baseUrl}`);
  }

  // Assemble into a single ArrayBuffer
  const assembled = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    assembled.set(new Uint8Array(chunk), offset);
    offset += chunk.byteLength;
  }

  return assembled.buffer;
}
