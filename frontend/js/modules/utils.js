/**
 * Utility functions for the transcription frontend
 */

/**
 * Format seconds as MM:SS or HH:MM:SS
 * @param {number} seconds - Time in seconds
 * @returns {string} Formatted time string
 */
export function formatTime(seconds) {
  if (seconds === null || seconds === undefined || isNaN(seconds)) {
    return '00:00';
  }

  const hrs = Math.floor(seconds / 3600);
  const mins = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);

  if (hrs > 0) {
    return `${hrs.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  }
  return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}

/**
 * Decode base64 string to Int16Array (PCM s16le)
 * @param {string} base64 - Base64 encoded PCM data
 * @returns {Int16Array} PCM samples
 */
export function base64ToInt16(base64) {
  const binaryString = atob(base64);
  const len = binaryString.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return new Int16Array(bytes.buffer);
}

/**
 * Convert Int16Array to Float32Array (normalize to -1..1)
 * @param {Int16Array} int16Array - PCM samples as Int16
 * @returns {Float32Array} Normalized samples
 */
export function int16ToFloat32(int16Array) {
  const float32Array = new Float32Array(int16Array.length);
  for (let i = 0; i < int16Array.length; i++) {
    float32Array[i] = int16Array[i] / 32768.0;
  }
  return float32Array;
}

/**
 * Decode base64 PCM to Float32Array
 * @param {string} base64 - Base64 encoded PCM data
 * @returns {Float32Array} Normalized samples
 */
export function base64ToFloat32(base64) {
  return int16ToFloat32(base64ToInt16(base64));
}

/**
 * Debounce a function
 * @param {Function} fn - Function to debounce
 * @param {number} ms - Delay in milliseconds
 * @returns {Function} Debounced function
 */
export function debounce(fn, ms) {
  let timeoutId;
  return function (...args) {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => fn.apply(this, args), ms);
  };
}

/**
 * Throttle a function
 * @param {Function} fn - Function to throttle
 * @param {number} ms - Minimum interval in milliseconds
 * @returns {Function} Throttled function
 */
export function throttle(fn, ms) {
  let lastCall = 0;
  let timeoutId;
  return function (...args) {
    const now = Date.now();
    const remaining = ms - (now - lastCall);

    if (remaining <= 0) {
      clearTimeout(timeoutId);
      lastCall = now;
      fn.apply(this, args);
    } else if (!timeoutId) {
      timeoutId = setTimeout(() => {
        lastCall = Date.now();
        timeoutId = null;
        fn.apply(this, args);
      }, remaining);
    }
  };
}

/**
 * Create an event emitter mixin
 * @returns {Object} Event emitter methods
 */
export function createEventEmitter() {
  const listeners = new Map();

  return {
    on(event, callback) {
      if (!listeners.has(event)) {
        listeners.set(event, new Set());
      }
      listeners.get(event).add(callback);
      return () => this.off(event, callback);
    },

    off(event, callback) {
      if (listeners.has(event)) {
        listeners.get(event).delete(callback);
      }
    },

    emit(event, data) {
      if (listeners.has(event)) {
        listeners.get(event).forEach(callback => callback(data));
      }
    },

    removeAllListeners(event) {
      if (event) {
        listeners.delete(event);
      } else {
        listeners.clear();
      }
    },
  };
}

/**
 * Escape HTML to prevent XSS
 * @param {string} text - Text to escape
 * @returns {string} Escaped text
 */
export function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}
