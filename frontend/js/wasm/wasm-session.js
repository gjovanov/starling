/**
 * WasmSession — client-side session manager for browser WASM inference.
 *
 * Manages the Web Worker lifecycle, model loading, and audio streaming.
 * Emits events: 'subtitle', 'status', 'error'
 */

/** Default chunk commit interval in milliseconds (3s of audio for better quality) */
const COMMIT_INTERVAL_MS = 3000;

/** Sample rate expected by the WASM engine */
const SAMPLE_RATE = 16000;

/** Samples per commit interval */
const SAMPLES_PER_COMMIT = (SAMPLE_RATE * COMMIT_INTERVAL_MS) / 1000; // 48000

export class WasmSession {
  constructor() {
    /** @type {Worker|null} */
    this.worker = null;

    /** Whether the WASM runtime has been initialized */
    this.runtimeReady = false;

    /** Whether the model is loaded and ready for inference */
    this.modelLoaded = false;

    /** Whether audio streaming is in progress */
    this.streaming = false;

    /** Timer ID for streaming interval */
    this.streamTimer = null;

    /** PCM data being streamed */
    this.pcmData = null;

    /** Current offset into pcmData (in samples) */
    this.pcmOffset = 0;

    /** Number of commits completed */
    this.commitCount = 0;

    /** Start time of streaming (for elapsed time tracking) */
    this.startTime = 0;

    /** Whether a commit is in-flight (waiting for response) */
    this.commitPending = false;

    /** Event listeners */
    this._listeners = {};
  }

  /**
   * Register an event listener.
   * @param {string} event - Event name ('subtitle', 'status', 'error')
   * @param {Function} callback
   */
  on(event, callback) {
    if (!this._listeners[event]) {
      this._listeners[event] = [];
    }
    this._listeners[event].push(callback);
  }

  /**
   * Remove an event listener.
   * @param {string} event
   * @param {Function} callback
   */
  off(event, callback) {
    if (!this._listeners[event]) return;
    this._listeners[event] = this._listeners[event].filter(cb => cb !== callback);
  }

  /**
   * Emit an event to all listeners.
   * @param {string} event
   * @param {*} data
   */
  _emit(event, data) {
    if (this._listeners[event]) {
      for (const cb of this._listeners[event]) {
        try {
          cb(data);
        } catch (err) {
          console.error(`[wasm] Event handler error for '${event}':`, err);
        }
      }
    }
  }

  /**
   * Initialize the Web Worker and WASM runtime.
   * @returns {Promise<void>} Resolves when runtime is ready.
   */
  async initRuntime() {
    if (this.worker) {
      console.log('[wasm] Runtime already initialized');
      return;
    }

    this._emit('status', { state: 'initializing', message: 'Starting WASM runtime...' });

    this.worker = new Worker('./js/wasm/worker.js', { type: 'module' });
    this._setupWorkerHandlers();

    return new Promise((resolve, reject) => {
      const onReady = () => {
        this._readyResolve = null;
        this.runtimeReady = true;
        this._emit('status', { state: 'runtime_ready', message: 'WASM runtime ready' });
        resolve();
      };
      const onError = (msg) => {
        this._readyResolve = null;
        reject(new Error(msg));
      };
      this._readyResolve = onReady;
      this._readyReject = onError;

      this.worker.postMessage({ type: 'init' });
    });
  }

  /**
   * Set up message handlers for the worker.
   */
  _setupWorkerHandlers() {
    this.worker.onmessage = (e) => {
      const msg = e.data;

      switch (msg.type) {
        case 'ready':
          if (this._readyResolve) this._readyResolve();
          break;

        case 'loaded':
          this.modelLoaded = true;
          if (this._loadResolve) {
            this._loadResolve();
            this._loadResolve = null;
          }
          this._emit('status', { state: 'model_loaded', message: 'Model loaded and ready' });
          break;

        case 'text':
          this.commitPending = false;
          this.commitCount++;
          this._emit('subtitle', {
            delta: msg.delta,
            fullText: msg.fullText,
            commitCount: this.commitCount,
            elapsedMs: Date.now() - this.startTime,
          });
          break;

        case 'error':
          this.commitPending = false;
          console.error('[wasm] Worker error:', msg.message);
          if (this._readyReject) {
            this._readyReject(msg.message);
            this._readyReject = null;
          }
          if (this._loadReject) {
            this._loadReject(new Error(msg.message));
            this._loadReject = null;
          }
          this._emit('error', { message: msg.message });
          break;

        default:
          console.warn('[wasm] Unknown worker message:', msg);
      }
    };

    this.worker.onerror = (err) => {
      console.error('[wasm] Worker uncaught error:', err);
      this._emit('error', { message: `Worker error: ${err.message}` });
    };
  }

  /**
   * Fetch model chunks and tokenizer, then load them into the WASM engine.
   *
   * @param {string} chunksUrl - Base URL for chunk files (e.g. "/models/cache/q4/chunks")
   * @param {string} tokenizerUrl - URL for tokenizer JSON (e.g. "/models/cache/tokenizer/tekken.json")
   * @param {function(number, number, string): void} [onProgress] - Progress callback (loaded, total, stage)
   * @returns {Promise<void>} Resolves when model is loaded.
   */
  async loadModel(chunksUrl, tokenizerUrl, onProgress) {
    if (!this.runtimeReady) {
      await this.initRuntime();
    }

    this._emit('status', { state: 'downloading', message: 'Downloading model chunks...' });

    // Fetch model chunks
    if (onProgress) onProgress(0, 0, 'chunks');
    console.log(`[wasm] Fetching model chunks from ${chunksUrl}...`);

    const shards = await this._fetchChunksAsShards(chunksUrl, (loaded, total) => {
      if (onProgress) onProgress(loaded, total, 'chunks');
    });

    console.log(`[wasm] Downloaded ${shards.length} shard(s)`);

    // Fetch tokenizer
    this._emit('status', { state: 'downloading_tokenizer', message: 'Downloading tokenizer...' });
    if (onProgress) onProgress(0, 0, 'tokenizer');
    console.log(`[wasm] Fetching tokenizer from ${tokenizerUrl}...`);

    const tokenizerRes = await fetch(tokenizerUrl);
    if (!tokenizerRes.ok) {
      throw new Error(`Failed to fetch tokenizer: ${tokenizerRes.status} ${tokenizerRes.statusText}`);
    }
    const tokenizerJson = await tokenizerRes.text();
    console.log(`[wasm] Tokenizer downloaded (${(tokenizerJson.length / 1024).toFixed(0)} KB)`);

    // Send shards and tokenizer to the worker
    this._emit('status', { state: 'loading_model', message: 'Loading model into WASM engine...' });

    return new Promise((resolve, reject) => {
      this._loadResolve = resolve;
      this._loadReject = reject;

      // Transfer the shard ArrayBuffers to the worker for zero-copy
      const transferables = shards.map(s => s.buffer);
      this.worker.postMessage(
        { type: 'load', shards, tokenizerJson },
        transferables
      );
    });
  }

  /**
   * Fetch chunks and return them as an array of Uint8Array shards
   * (one per chunk file), suitable for WasmEngine.create().
   *
   * @param {string} baseUrl - Base URL for chunk files
   * @param {function(number, number): void} onProgress - Progress callback
   * @returns {Promise<Uint8Array[]>}
   */
  async _fetchChunksAsShards(baseUrl, onProgress) {
    const shards = [];
    let loaded = 0;
    let total = 0;
    let chunkIndex = 0;

    while (true) {
      const url = `${baseUrl}/chunk_${chunkIndex}`;
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
      shards.push(new Uint8Array(buffer));
      loaded += buffer.byteLength;
      if (total === 0) total = loaded;

      if (onProgress) onProgress(loaded, total);
      chunkIndex++;
    }

    if (shards.length === 0) {
      throw new Error(`No model chunks found at ${baseUrl}`);
    }

    return shards;
  }

  /**
   * Start streaming pre-decoded PCM data to the WASM engine.
   * Sends 0.5s chunks and commits after each chunk.
   *
   * @param {Float32Array} pcmData - Mono 16kHz f32 PCM samples
   */
  startTranscription(pcmData) {
    if (!this.modelLoaded) {
      this._emit('error', { message: 'Model not loaded — cannot start transcription' });
      return;
    }

    if (this.streaming) {
      console.warn('[wasm] Already streaming, stopping previous stream');
      this.stop();
    }

    this.pcmData = pcmData;
    this.pcmOffset = 0;
    this.commitCount = 0;
    this.streaming = true;
    this.startTime = Date.now();
    this.commitPending = false;

    // Reset the engine for a fresh transcription
    this.worker.postMessage({ type: 'reset' });

    console.log(`[wasm] Starting transcription: ${pcmData.length} samples (${(pcmData.length / SAMPLE_RATE).toFixed(1)}s)`);
    this._emit('status', { state: 'transcribing', message: 'Transcription started' });

    // Stream audio in commit-interval-sized chunks
    this.streamTimer = setInterval(() => {
      this._streamNextChunk();
    }, COMMIT_INTERVAL_MS);

    // Send the first chunk immediately
    this._streamNextChunk();
  }

  /**
   * Send the next 0.5s chunk to the worker and trigger a commit.
   */
  _streamNextChunk() {
    if (!this.streaming || !this.pcmData) return;

    // Skip if a commit is still in-flight to avoid piling up
    if (this.commitPending) {
      console.log('[wasm] _streamNextChunk: skipping, commit still pending');
      return;
    }

    const remaining = this.pcmData.length - this.pcmOffset;
    if (remaining <= 0) {
      // All audio has been sent — do a final commit if we haven't already
      console.log('[wasm] All audio sent, finishing up');
      this.stop();
      this._emit('status', { state: 'completed', message: 'Transcription completed' });
      return;
    }

    const chunkSize = Math.min(SAMPLES_PER_COMMIT, remaining);
    const chunk = this.pcmData.slice(this.pcmOffset, this.pcmOffset + chunkSize);
    this.pcmOffset += chunkSize;

    console.log(`[wasm] _streamNextChunk: sending ${chunkSize} samples (offset ${this.pcmOffset}/${this.pcmData.length})`);

    // Send audio then commit
    this.worker.postMessage(
      { type: 'audio', samples: chunk },
      [chunk.buffer]
    );

    this.commitPending = true;
    this.worker.postMessage({ type: 'commit' });
  }

  /**
   * Stop the current streaming transcription.
   */
  stop() {
    if (this.streamTimer) {
      clearInterval(this.streamTimer);
      this.streamTimer = null;
    }
    this.streaming = false;
    this.commitPending = false;
    console.log('[wasm] Streaming stopped');
  }

  /**
   * Destroy the worker and clean up all resources.
   */
  destroy() {
    this.stop();
    if (this.worker) {
      this.worker.terminate();
      this.worker = null;
    }
    this.runtimeReady = false;
    this.modelLoaded = false;
    this.pcmData = null;
    this._listeners = {};
    console.log('[wasm] Session destroyed');
  }
}
