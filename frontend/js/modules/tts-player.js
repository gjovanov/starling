/**
 * Progressive TTS audio player using the Web Audio API.
 *
 * Why Web Audio and not <audio src=blob:> + MediaSource Extensions:
 *   - Chrome's MediaSource.isTypeSupported('audio/wav') returns false
 *     (MSE only works for fragmented MP4 / WebM). Re-muxing PCM into fMP4
 *     server-side is the only way to use MSE for our stream — too much
 *     plumbing for a Phase-4 deliverable.
 *   - Web Audio gives sample-accurate scheduling for free. Phase 5 needs
 *     this to stitch sentence-by-sentence streams gap-free.
 *
 * The player consumes the chunked `audio/wav` body emitted by
 * `POST /api/tts/synthesize` with `save: false`. The first read includes
 * the 44-byte streaming WAV header (RIFF/WAVE/fmt/data with `0xFFFFFFFF`
 * placeholder sizes). We parse the header once to learn sample-rate +
 * channels + bps, then treat every byte after the `data` marker as raw
 * PCM frames.
 *
 * State machine:
 *
 *   idle ─► buffering ─► playing ─► ended
 *      ▲       │   ▲           │
 *      │       ▼   │           ▼
 *      └──── paused ──── (cancel)
 *
 * Public API:
 *   const p = new TtsPlayer({ onStateChange })
 *   await p.start(response)     // streams + schedules + resolves on EOF
 *   p.pause() / p.resume()
 *   p.cancel()                  // aborts fetch reader, tears down sources
 *   p.getState() → { state, currentTimeSecs, bufferedSecs, sampleRate }
 *   p.events.addEventListener('tts:state', e => …)
 */

const STATES = Object.freeze({
  IDLE: 'idle',
  BUFFERING: 'buffering',
  PLAYING: 'playing',
  PAUSED: 'paused',
  ENDED: 'ended',
});

const HEADER_BYTES = 44;
const TARGET_BUFFER_AHEAD_SECS = 0.3;   // before flipping to "playing"
const STARVE_THRESHOLD_SECS = 0.05;     // schedule head this close to now → buffering

// Aggregate incoming chunks into blocks of at least this many seconds before
// scheduling them as one AudioBufferSourceNode. Reduces the number of source
// boundaries (and hence the number of resampler-startup transients that
// otherwise stack into a sub-bass amplitude tremolo perceived as "fading").
//
// The first chunk is always scheduled immediately so TTFB stays low; the
// aggregation only applies to subsequent chunks.
const AGGREGATE_TARGET_SECS = 0.5;

export class TtsPlayer extends EventTarget {
  constructor({ onStateChange, playbackRate = 1.0 } = {}) {
    super();
    /**
     * Playback-rate multiplier. 1.0 = natural; >1 = faster (also slightly
     * higher pitch — Web Audio doesn't pitch-correct). 0.9–1.25 sounds
     * natural; outside that range the pitch shift is noticeable.
     */
    this._playbackRate = Math.max(0.5, Math.min(2.0, Number(playbackRate) || 1.0));
    /** @type {AudioContext|null} */
    this._ctx = null;
    this._reader = null;
    /** Bytes still to be parsed for the WAV header (44 bytes). */
    this._tail = new Uint8Array(0);
    /**
     * Trailing bytes that didn't form a whole audio frame on their own.
     * Carried forward to the next chunk's prefix so every byte ends up at
     * the correct (LE int16) position. Without this carry, an odd-byte
     * chunk on the wire — which happens whenever the network/HTTP layer
     * splits a packet mid-sample — shifts the entire downstream stream by
     * one byte and decodes as crackling noise.
     */
    this._pcmTail = new Uint8Array(0);
    /** When the next AudioBufferSourceNode should start (in ctx.currentTime). */
    this._nextStartAt = 0;
    /**
     * Aggregation buffer: list of Float32Array fragments waiting to be
     * scheduled as one AudioBufferSourceNode. Flushed once `_aggFrames`
     * exceeds the target.
     */
    this._aggFragments = [];
    this._aggFrames = 0;
    this._firstChunkScheduled = false;
    /** Sources scheduled but not yet ended. We disconnect+null them on stop. */
    this._liveSources = new Set();
    this._sampleRate = 24000;
    this._channels = 1;
    this._bps = 16;
    this._state = STATES.IDLE;
    this._onStateChange = onStateChange;
    this._cancelled = false;
    /** Set when the user clicks Pause; resumed via resume(). */
    this._pausedAt = 0;          // ctx.currentTime snapshot
    this._pauseDelta = 0;        // total time spent paused (subtracted from currentTimeSecs)
    /** Pause-on-blur handler. */
    this._visibilityHandler = null;
  }

  /** @returns {{state: string, currentTimeSecs: number, bufferedSecs: number, sampleRate: number}} */
  getState() {
    const now = this._ctx?.currentTime ?? 0;
    const playStart = this._playStartAt ?? 0;
    const currentTimeSecs = playStart === 0 ? 0 : Math.max(0, now - playStart - this._pauseDelta);
    const bufferedSecs = Math.max(0, this._nextStartAt - now);
    return {
      state: this._state,
      currentTimeSecs,
      bufferedSecs,
      sampleRate: this._sampleRate,
    };
  }

  /**
   * Begin playback from a streaming `Response` (chunked audio/wav).
   * Resolves when EOF is reached. Rejects on transport error or cancel.
   */
  async start(response) {
    if (this._ctx) {
      throw new Error('TtsPlayer.start: already started — use cancel() first');
    }
    if (!response.body) {
      throw new Error('TtsPlayer.start: response has no body stream');
    }

    this._ctx = new (window.AudioContext || window.webkitAudioContext)();
    // Some browsers start AudioContext in 'suspended' state until a user
    // gesture. The Generate-button click that triggered start() satisfies
    // the policy; resume here just in case the gesture path was indirect.
    if (this._ctx.state === 'suspended') {
      try { await this._ctx.resume(); } catch (_) { /* ignore */ }
    }

    this._installVisibilityHandler();
    this._setState(STATES.BUFFERING);

    this._reader = response.body.getReader();
    this._nextStartAt = this._ctx.currentTime + 0.05; // small head-start

    try {
      // Header pump: keep reading until we have ≥ 44 bytes, then parse.
      while (this._tail.length < HEADER_BYTES) {
        const { value, done } = await this._reader.read();
        if (done) {
          // Stream closed before the header arrived — treat as error/cancel.
          if (this._cancelled) {
            this._setState(STATES.ENDED);
            return;
          }
          throw new Error(
            `TtsPlayer: stream ended before WAV header (${this._tail.length}/44 bytes)`,
          );
        }
        this._tail = _concat(this._tail, value);
      }

      const headerBytes = this._tail.slice(0, HEADER_BYTES);
      const fmt = _parseWavHeader(headerBytes);
      this._sampleRate = fmt.sampleRate;
      this._channels = fmt.channels;
      this._bps = fmt.bps;
      this._tail = this._tail.slice(HEADER_BYTES);

      // Body pump: feed bytes through the frame-aligning helper. Any
      // leftover bytes from the header read are the start of the PCM
      // payload.
      if (this._tail.length > 0) {
        this._enqueuePcmBytes(this._tail);
        this._tail = new Uint8Array(0);
      }

      while (true) {
        if (this._cancelled) break;
        const { value, done } = await this._reader.read();
        if (done) break;
        if (value && value.length) this._enqueuePcmBytes(value);
      }

      // Flush whole frames left in the tail (drop any partial frame at EOF).
      const blockAlign = 2 * this._channels;
      if (this._pcmTail.length >= blockAlign) {
        const aligned = Math.floor(this._pcmTail.length / blockAlign) * blockAlign;
        this._scheduleChunk(this._pcmTail.subarray(0, aligned));
        this._pcmTail = new Uint8Array(0);
      }
      // Flush any aggregated samples that haven't reached the target yet.
      this._flushAggregated();

      // Wait for the last scheduled source to finish, then transition to ended.
      await this._waitForDrain();
      if (this._state !== STATES.ENDED) this._setState(STATES.ENDED);
    } catch (err) {
      // We may have already played some audio — leave any live sources
      // running and just transition to ended. Caller decides what to do.
      this._setState(STATES.ENDED);
      throw err;
    }
  }

  pause() {
    if (this._state !== STATES.PLAYING && this._state !== STATES.BUFFERING) return;
    if (!this._ctx) return;
    this._pausedAt = this._ctx.currentTime;
    // suspend() pauses the clock for all scheduled sources atomically.
    this._ctx.suspend().catch(() => { /* ignore */ });
    this._setState(STATES.PAUSED);
  }

  resume() {
    if (this._state !== STATES.PAUSED) return;
    if (!this._ctx) return;
    if (this._pausedAt > 0) {
      this._pauseDelta += this._ctx.currentTime - this._pausedAt;
      this._pausedAt = 0;
    }
    this._ctx.resume().catch(() => { /* ignore */ });
    // Resume back to whichever state we were in pre-pause. We default to
    // playing — getState() will keep its currentTime advancing.
    this._setState(STATES.PLAYING);
  }

  cancel() {
    this._cancelled = true;
    if (this._reader) {
      try { this._reader.cancel(); } catch (_) { /* ignore */ }
      this._reader = null;
    }
    this._tearDownSources();
    this._pcmTail = new Uint8Array(0);
    if (this._ctx) {
      this._ctx.close().catch(() => { /* ignore */ });
      this._ctx = null;
    }
    this._removeVisibilityHandler();
    this._setState(STATES.ENDED);
  }

  // ── internals ────────────────────────────────────────────────────

  _setState(next) {
    if (this._state === next) return;
    this._state = next;
    const detail = this.getState();
    try { this._onStateChange?.(detail); } catch (_) { /* ignore */ }
    this.dispatchEvent(new CustomEvent('tts:state', { detail }));
  }

  /**
   * Append fresh bytes to the PCM tail buffer, then schedule as many whole
   * audio frames as the combined buffer permits. The trailing odd-byte
   * remainder (if any) is carried forward so the next chunk's first byte
   * lands at the correct (LE) position within an int16 sample.
   *
   * Without this carry, an odd-byte network split shifts every subsequent
   * sample by 1 byte and decodes as crackling noise (which compounds the
   * longer the stream runs — exactly the "distant / crunching" symptom).
   */
  _enqueuePcmBytes(bytes) {
    if (!bytes || !bytes.length) return;
    const combined = _concat(this._pcmTail, bytes);
    const blockAlign = 2 * this._channels;     // bytes per audio frame
    const aligned = Math.floor(combined.length / blockAlign) * blockAlign;
    if (aligned > 0) {
      this._scheduleChunk(combined.subarray(0, aligned));
    }
    this._pcmTail = combined.subarray(aligned);
  }

  _scheduleChunk(byteArray) {
    if (!this._ctx) return;
    // 16-bit signed little-endian → Float32 in [-1, 1].
    const sampleCount = Math.floor(byteArray.length / 2 / this._channels);
    if (sampleCount === 0) return;
    const samples = this._channels === 1
      ? _decodeMonoInt16(byteArray, sampleCount)
      : _decodeMultichannelInt16(byteArray, sampleCount, this._channels);

    // First chunk always schedules immediately so the listener hears
    // audio at TTFB, not after `AGGREGATE_TARGET_SECS` of buffering.
    if (!this._firstChunkScheduled) {
      this._firstChunkScheduled = true;
      this._scheduleSamples(samples);
      return;
    }

    // Aggregate into ~AGGREGATE_TARGET_SECS blocks. Fewer source-boundaries
    // → fewer Web Audio resampler-startup transients to stack into a
    // perceptible amplitude tremolo on long sentences.
    this._aggFragments.push(samples);
    this._aggFrames += sampleCount;
    if (this._aggFrames >= this._sampleRate * AGGREGATE_TARGET_SECS) {
      this._flushAggregated();
    }
  }

  _flushAggregated() {
    if (this._aggFrames === 0) return;
    let combined;
    if (this._channels === 1) {
      combined = new Float32Array(this._aggFrames);
      let offset = 0;
      for (const f of this._aggFragments) {
        combined.set(f, offset);
        offset += f.length;
      }
    } else {
      // Per-channel concatenation — each fragment is itself a list of
      // per-channel Float32Arrays (see _decodeMultichannelInt16).
      combined = new Array(this._channels);
      for (let ch = 0; ch < this._channels; ch++) {
        combined[ch] = new Float32Array(this._aggFrames);
        let offset = 0;
        for (const f of this._aggFragments) {
          combined[ch].set(f[ch], offset);
          offset += f[ch].length;
        }
      }
    }
    this._aggFragments = [];
    this._aggFrames = 0;
    this._scheduleSamples(combined);
  }

  /**
   * Schedule a Float32Array (mono) or array-of-Float32Arrays (multichannel)
   * as one AudioBufferSourceNode. The block can be any length; this is the
   * step that actually creates a Web Audio source.
   */
  _scheduleSamples(samples) {
    if (!this._ctx) return;
    const ctx = this._ctx;
    const length = this._channels === 1 ? samples.length : samples[0].length;
    if (length === 0) return;
    const buffer = ctx.createBuffer(this._channels, length, this._sampleRate);
    if (this._channels === 1) {
      buffer.copyToChannel(samples, 0);
    } else {
      for (let ch = 0; ch < this._channels; ch++) {
        buffer.copyToChannel(samples[ch], ch);
      }
    }

    // Decide where this source starts. Two cases:
    //   1. _nextStartAt is already in the future → schedule there (perfect
    //      back-to-back continuation).
    //   2. _nextStartAt has fallen behind ctx.currentTime → start at NOW.
    //      Critically, we use `src.start(now)` not `src.start(<past time>)`
    //      because Web Audio interprets a past-time `when` by SKIPPING
    //      INTO the buffer by (now - when). That sliced off the first
    //      samples of every sentence in long-form mode → "fade-in" effect.
    //      With max(now, ...) the boundary is a brief silence gap rather
    //      than mangled audio, which is far less perceptible.
    const now = ctx.currentTime;
    const startAt = Math.max(now, this._nextStartAt);
    if (startAt > this._nextStartAt + 0.001) {
      // We had to skip forward — flag as buffering so the UI shows the
      // gap accurately. (The schedule itself is still correct.)
      this._setState(STATES.BUFFERING);
    }

    const src = ctx.createBufferSource();
    src.buffer = buffer;
    src.playbackRate.value = this._playbackRate;
    src.connect(ctx.destination);
    src.onended = () => {
      this._liveSources.delete(src);
      try { src.disconnect(); } catch (_) { /* ignore */ }
    };
    src.start(startAt);
    this._liveSources.add(src);

    if (this._playStartAt === undefined) this._playStartAt = startAt;

    // Buffer's *played* duration shrinks/expands by playbackRate.
    this._nextStartAt = startAt + (buffer.duration / this._playbackRate);

    // Once we've buffered enough ahead, switch out of "buffering" into
    // "playing". For very short syntheses we may never accumulate
    // TARGET_BUFFER_AHEAD_SECS — go to playing as soon as audio is queued.
    const ahead = this._nextStartAt - now;
    if (this._state === STATES.BUFFERING && ahead >= 0) {
      this._setState(STATES.PLAYING);
    }
  }

  /** Adjust playback speed mid-stream. Affects future chunks only. */
  setPlaybackRate(rate) {
    const next = Math.max(0.5, Math.min(2.0, Number(rate) || 1.0));
    this._playbackRate = next;
    // Apply to any source that hasn't ended yet so the change is audible
    // immediately rather than waiting for the next chunk.
    for (const src of this._liveSources) {
      try { src.playbackRate.value = next; } catch (_) { /* ignore */ }
    }
  }

  async _waitForDrain() {
    if (!this._ctx) return;
    while (this._liveSources.size > 0) {
      // Sleep for the duration to the next-end. Cheap polling is fine —
      // we hit this only at end-of-stream.
      await new Promise(r => setTimeout(r, 50));
    }
    // Plus the small head-start we added at the front, just to be safe.
    const remaining = this._nextStartAt - this._ctx.currentTime;
    if (remaining > 0) await new Promise(r => setTimeout(r, remaining * 1000));
  }

  _tearDownSources() {
    for (const src of this._liveSources) {
      try { src.stop(); } catch (_) { /* ignore */ }
      try { src.disconnect(); } catch (_) { /* ignore */ }
    }
    this._liveSources.clear();
  }

  _installVisibilityHandler() {
    // Pause when the tab loses focus to avoid AudioContext clock-throttling
    // glitches; resume on visibilitychange back.
    this._visibilityHandler = () => {
      if (document.hidden && this._state === STATES.PLAYING) this.pause();
      else if (!document.hidden && this._state === STATES.PAUSED) this.resume();
    };
    document.addEventListener('visibilitychange', this._visibilityHandler);
  }

  _removeVisibilityHandler() {
    if (this._visibilityHandler) {
      document.removeEventListener('visibilitychange', this._visibilityHandler);
      this._visibilityHandler = null;
    }
  }
}

// ── helpers ────────────────────────────────────────────────────────

function _concat(a, b) {
  const out = new Uint8Array(a.length + b.length);
  out.set(a, 0);
  out.set(b, a.length);
  return out;
}

/** Decode a packed mono int16 LE byte buffer into a Float32Array in [-1, 1). */
function _decodeMonoInt16(byteArray, sampleCount) {
  const view = new DataView(byteArray.buffer, byteArray.byteOffset, byteArray.byteLength);
  const out = new Float32Array(sampleCount);
  for (let i = 0; i < sampleCount; i++) {
    out[i] = view.getInt16(i * 2, true) / 32768;
  }
  return out;
}

/** Decode an interleaved multichannel int16 LE byte buffer into one
 * Float32Array per channel. Currently used only as a fallback — Voxtral
 * outputs mono, so `_decodeMonoInt16` is the hot path. */
function _decodeMultichannelInt16(byteArray, sampleCount, channels) {
  const view = new DataView(byteArray.buffer, byteArray.byteOffset, byteArray.byteLength);
  const stride = 2 * channels;
  const out = new Array(channels);
  for (let ch = 0; ch < channels; ch++) out[ch] = new Float32Array(sampleCount);
  for (let i = 0; i < sampleCount; i++) {
    for (let ch = 0; ch < channels; ch++) {
      out[ch][i] = view.getInt16(i * stride + ch * 2, true) / 32768;
    }
  }
  return out;
}

/**
 * Parse the 44-byte streaming WAV header. Tolerates the `0xFFFFFFFF`
 * placeholder sizes our server emits (we only need rate/channels/bps).
 */
function _parseWavHeader(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const riff = String.fromCharCode(...bytes.slice(0, 4));
  const wave = String.fromCharCode(...bytes.slice(8, 12));
  if (riff !== 'RIFF' || wave !== 'WAVE') {
    throw new Error(`TtsPlayer: bad WAV header (RIFF=${riff} WAVE=${wave})`);
  }
  const fmtTag = String.fromCharCode(...bytes.slice(12, 16));
  if (fmtTag !== 'fmt ') {
    throw new Error(`TtsPlayer: missing fmt chunk (got ${fmtTag})`);
  }
  const channels = view.getUint16(22, true);
  const sampleRate = view.getUint32(24, true);
  const bps = view.getUint16(34, true);
  const dataTag = String.fromCharCode(...bytes.slice(36, 40));
  if (dataTag !== 'data') {
    throw new Error(`TtsPlayer: missing data marker (got ${dataTag})`);
  }
  if (bps !== 16) {
    throw new Error(`TtsPlayer: only 16-bit PCM supported (got ${bps})`);
  }
  return { channels, sampleRate, bps };
}

/** Expose state on `window` for Playwright assertions. */
export function exposePlayerForTests(player) {
  window.__ttsPlayerState = player.getState();
  player.events ??= player; // EventTarget already
  player.addEventListener('tts:state', () => {
    window.__ttsPlayerState = player.getState();
  });
}
