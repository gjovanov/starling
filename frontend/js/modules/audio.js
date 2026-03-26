/**
 * Audio playback module using Web Audio API
 *
 * Designed for streaming with network jitter:
 * - Large jitter buffer (2s default) for smooth playback
 * - Auto-pause on underrun, auto-resume when buffer recovers
 * - Adaptive scheduling based on buffer health
 * - Continuous buffer with efficient memory management
 */
import { base64ToFloat32, createEventEmitter } from './utils.js';

export class AudioPlayer {
  /**
   * @param {Object} options - Configuration options
   */
  constructor(options = {}) {
    this.options = {
      sampleRate: 16000,
      // Initial jitter buffer: wait for this much before starting (2 seconds)
      jitterBufferMs: 2000,
      // Minimum buffer to maintain - pause if below this (500ms)
      minBufferMs: 500,
      // Resume playback when buffer reaches this level after underrun (1.5s)
      resumeBufferMs: 1500,
      // How far ahead to schedule audio (1 second)
      scheduleAheadMs: 1000,
      // How often to check and schedule (50ms)
      schedulerIntervalMs: 50,
      // Maximum chunk size to schedule at once (500ms)
      maxScheduleChunkMs: 500,
      ...options,
    };

    /** @type {AudioContext|null} */
    this.audioContext = null;

    /** @type {GainNode|null} */
    this.gainNode = null;

    // Continuous audio buffer (grows as chunks arrive)
    /** @type {Float32Array} */
    this.audioBuffer = new Float32Array(0);

    // Active audio sources (for cleanup)
    /** @type {Set<AudioBufferSourceNode>} */
    this.activeSources = new Set();

    // Playback state
    this.isPlaying = false;
    this.isPausedForBuffer = false;  // Paused due to buffer underrun

    // Timing
    this.playbackStartTime = 0;      // AudioContext time when playback started
    this.playbackOffset = 0;         // Sample offset when playback started/resumed
    this.scheduledUntilSample = 0;   // How many samples we've scheduled
    this.lastScheduledEndTime = 0;   // AudioContext time of last scheduled end

    // Buffer state
    this.jitterBufferReady = false;
    this.underrunCount = 0;

    // Pre-calculate sample counts
    this.jitterBufferSamples = Math.floor(this.options.jitterBufferMs * this.options.sampleRate / 1000);
    this.minBufferSamples = Math.floor(this.options.minBufferMs * this.options.sampleRate / 1000);
    this.resumeBufferSamples = Math.floor(this.options.resumeBufferMs * this.options.sampleRate / 1000);
    this.scheduleAheadSamples = Math.floor(this.options.scheduleAheadMs * this.options.sampleRate / 1000);
    this.maxScheduleChunkSamples = Math.floor(this.options.maxScheduleChunkMs * this.options.sampleRate / 1000);

    // Server timestamp tracking for sync
    this.lastServerTimestamp = 0;

    // Scheduler
    this.schedulerInterval = null;

    // Event emitter
    const emitter = createEventEmitter();
    this.on = emitter.on.bind(emitter);
    this.off = emitter.off.bind(emitter);
    this.emit = emitter.emit.bind(emitter);
  }

  /**
   * Initialize audio context (must be called after user interaction)
   */
  async init() {
    if (this.audioContext) {
      return;
    }

    this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
      sampleRate: this.options.sampleRate,
      latencyHint: 'playback',
    });

    // Create gain node for volume control
    this.gainNode = this.audioContext.createGain();
    this.gainNode.connect(this.audioContext.destination);

    // Resume if suspended
    if (this.audioContext.state === 'suspended') {
      await this.audioContext.resume();
    }

    console.log(`AudioContext initialized: ${this.audioContext.sampleRate}Hz`);
  }

  /**
   * Append audio chunk from base64 encoded PCM
   * @param {string} base64Data - Base64 encoded PCM s16le data
   * @param {number} timestamp - Server timestamp (for sync)
   */
  appendChunk(base64Data, timestamp) {
    const newSamples = base64ToFloat32(base64Data);
    this.lastServerTimestamp = timestamp;

    // Grow the buffer
    const newBuffer = new Float32Array(this.audioBuffer.length + newSamples.length);
    newBuffer.set(this.audioBuffer);
    newBuffer.set(newSamples, this.audioBuffer.length);
    this.audioBuffer = newBuffer;

    const bufferedSamples = this.audioBuffer.length - this.scheduledUntilSample;
    const bufferedMs = (bufferedSamples / this.options.sampleRate) * 1000;

    // Check if initial jitter buffer is ready
    if (!this.jitterBufferReady && bufferedSamples >= this.jitterBufferSamples) {
      this.jitterBufferReady = true;
      console.log(`Jitter buffer ready (${bufferedMs.toFixed(0)}ms buffered)`);
      this.emit('bufferReady');
    }

    // Check if we can resume from underrun pause
    if (this.isPausedForBuffer && bufferedSamples >= this.resumeBufferSamples) {
      console.log(`Buffer recovered (${bufferedMs.toFixed(0)}ms), resuming playback`);
      this.isPausedForBuffer = false;
      this.emit('bufferRecovered');
      // Reset scheduling from current position
      this.resetScheduling();
    }

    // Emit buffer update
    this.emit('bufferUpdate', {
      buffered: this.getBufferedDuration(),
      total: this.duration,
    });

    // If playing and ready, schedule more audio
    if (this.isPlaying && this.jitterBufferReady && !this.isPausedForBuffer && this.audioContext) {
      this.scheduleAudio();
    }
  }

  /**
   * Start or resume playback
   */
  async play() {
    if (!this.audioContext) {
      await this.init();
    }

    if (this.audioContext.state === 'suspended') {
      await this.audioContext.resume();
    }

    if (this.isPlaying && !this.isPausedForBuffer) {
      return;
    }

    // Check if we have enough buffer
    const bufferedSamples = this.audioBuffer.length - this.scheduledUntilSample;

    if (!this.jitterBufferReady && bufferedSamples < this.jitterBufferSamples) {
      console.log(`Waiting for jitter buffer (have ${(bufferedSamples / this.options.sampleRate * 1000).toFixed(0)}ms, need ${this.options.jitterBufferMs}ms)`);
      this.emit('buffering');
      this.isPlaying = true;
      this.startScheduler();
      return;
    }

    this.isPlaying = true;
    this.isPausedForBuffer = false;
    this.jitterBufferReady = true;

    // Reset scheduling timing
    this.resetScheduling();

    // Start scheduler
    this.startScheduler();
    this.scheduleAudio();
    this.emit('play');
  }

  /**
   * Reset scheduling state for fresh start
   */
  resetScheduling() {
    if (this.audioContext) {
      this.playbackStartTime = this.audioContext.currentTime;
      this.lastScheduledEndTime = this.audioContext.currentTime;
    }
  }

  /**
   * Pause playback
   */
  pause() {
    if (!this.isPlaying) {
      return;
    }

    this.isPlaying = false;
    this.stopScheduler();
    this.stopAllSources();

    // Save position
    this.playbackOffset = this.scheduledUntilSample;

    this.emit('pause');
  }

  /**
   * Stop playback and reset
   */
  stop() {
    this.pause();
    this.playbackStartTime = 0;
    this.playbackOffset = 0;
    this.scheduledUntilSample = 0;
    this.lastScheduledEndTime = 0;
    this.isPausedForBuffer = false;
    this.emit('stop');
  }

  /**
   * Stop all active audio sources
   */
  stopAllSources() {
    for (const source of this.activeSources) {
      try {
        source.stop();
      } catch (e) {
        // Ignore if already stopped
      }
    }
    this.activeSources.clear();
  }

  /**
   * Seek to time
   * @param {number} time - Time in seconds
   */
  seek(time) {
    const wasPlaying = this.isPlaying && !this.isPausedForBuffer;

    this.stop();

    const targetSample = Math.floor(time * this.options.sampleRate);
    this.playbackOffset = Math.min(targetSample, this.audioBuffer.length);
    this.scheduledUntilSample = this.playbackOffset;

    if (wasPlaying) {
      this.play();
    }

    this.emit('seek', time);
  }

  /**
   * Start the audio scheduler
   */
  startScheduler() {
    this.stopScheduler();
    this.schedulerInterval = setInterval(() => {
      this.tick();
    }, this.options.schedulerIntervalMs);
  }

  /**
   * Stop the audio scheduler
   */
  stopScheduler() {
    if (this.schedulerInterval) {
      clearInterval(this.schedulerInterval);
      this.schedulerInterval = null;
    }
  }

  /**
   * Scheduler tick - called periodically
   */
  tick() {
    // Emit time update
    this.emit('timeUpdate', this.currentTime);

    if (!this.isPlaying || this.isPausedForBuffer || !this.audioContext) {
      return;
    }

    // Check buffer health
    const bufferedSamples = this.audioBuffer.length - this.scheduledUntilSample;

    if (bufferedSamples < this.minBufferSamples) {
      // Buffer underrun!
      this.underrunCount++;
      console.warn(`Buffer underrun #${this.underrunCount} (${(bufferedSamples / this.options.sampleRate * 1000).toFixed(0)}ms remaining)`);
      this.isPausedForBuffer = true;
      this.stopAllSources();
      this.emit('bufferUnderrun', { count: this.underrunCount });
      return;
    }

    // Schedule more audio
    this.scheduleAudio();
  }

  /**
   * Schedule audio playback
   */
  scheduleAudio() {
    if (!this.audioContext || !this.isPlaying || this.isPausedForBuffer) {
      return;
    }

    const now = this.audioContext.currentTime;

    // How far ahead are we scheduled?
    const scheduledAhead = this.lastScheduledEndTime - now;

    // If we're already scheduled far enough ahead, skip
    if (scheduledAhead >= this.options.scheduleAheadMs / 1000) {
      return;
    }

    // How many samples are available?
    const availableSamples = this.audioBuffer.length - this.scheduledUntilSample;

    if (availableSamples <= 0) {
      return;
    }

    // Calculate how many samples to schedule
    // Schedule up to maxScheduleChunkSamples at a time for smoother memory usage
    const samplesToSchedule = Math.min(availableSamples, this.maxScheduleChunkSamples);

    if (samplesToSchedule <= 0) {
      return;
    }

    // Extract samples
    const samples = this.audioBuffer.slice(
      this.scheduledUntilSample,
      this.scheduledUntilSample + samplesToSchedule
    );

    // Create audio buffer
    const audioBuffer = this.audioContext.createBuffer(
      1, // mono
      samples.length,
      this.options.sampleRate
    );
    audioBuffer.getChannelData(0).set(samples);

    // Create source
    const source = this.audioContext.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(this.gainNode);

    // Schedule playback
    const startTime = Math.max(this.lastScheduledEndTime, now);
    source.start(startTime);

    // Update state
    const durationSecs = samples.length / this.options.sampleRate;
    this.scheduledUntilSample += samples.length;
    this.lastScheduledEndTime = startTime + durationSecs;

    // Track source
    this.activeSources.add(source);
    source.onended = () => {
      this.activeSources.delete(source);
    };
  }

  /**
   * Get current playback time
   * @returns {number} Time in seconds
   */
  get currentTime() {
    if (!this.audioContext || !this.isPlaying) {
      return this.playbackOffset / this.options.sampleRate;
    }

    if (this.isPausedForBuffer) {
      return this.scheduledUntilSample / this.options.sampleRate;
    }

    // Estimate current position based on what's been scheduled
    const now = this.audioContext.currentTime;
    const scheduledDuration = this.lastScheduledEndTime - this.playbackStartTime;
    const elapsed = now - this.playbackStartTime;

    // Current position is playbackOffset + elapsed time
    const currentSample = this.playbackOffset + Math.min(elapsed, scheduledDuration) * this.options.sampleRate;
    return currentSample / this.options.sampleRate;
  }

  /**
   * Get total duration of received audio
   * @returns {number} Duration in seconds
   */
  get duration() {
    return this.audioBuffer.length / this.options.sampleRate;
  }

  /**
   * Get buffered duration ahead of current position
   * @returns {number} Buffered duration in seconds
   */
  getBufferedDuration() {
    const bufferedSamples = this.audioBuffer.length - this.scheduledUntilSample;
    return Math.max(0, bufferedSamples / this.options.sampleRate);
  }

  /**
   * Check if currently playing
   * @returns {boolean}
   */
  get playing() {
    return this.isPlaying && !this.isPausedForBuffer;
  }

  /**
   * Check if paused due to buffer underrun
   * @returns {boolean}
   */
  get isBuffering() {
    return this.isPausedForBuffer || (!this.jitterBufferReady && this.isPlaying);
  }

  /**
   * Set volume (0.0 to 1.0)
   * @param {number} volume
   */
  setVolume(volume) {
    if (this.gainNode) {
      this.gainNode.gain.value = Math.max(0, Math.min(1, volume));
    }
  }

  /**
   * Get buffer health info
   * @returns {Object}
   */
  getBufferHealth() {
    const bufferedSamples = this.audioBuffer.length - this.scheduledUntilSample;
    const bufferedMs = (bufferedSamples / this.options.sampleRate) * 1000;
    return {
      bufferedMs,
      minBufferMs: this.options.minBufferMs,
      jitterBufferMs: this.options.jitterBufferMs,
      isHealthy: bufferedMs >= this.options.minBufferMs,
      underrunCount: this.underrunCount,
    };
  }

  /**
   * Clear audio buffer and reset
   */
  clear() {
    this.stop();
    this.audioBuffer = new Float32Array(0);
    this.jitterBufferReady = false;
    this.isPausedForBuffer = false;
    this.lastServerTimestamp = 0;
    this.underrunCount = 0;
    this.emit('clear');
  }

  /**
   * Destroy the audio player
   */
  destroy() {
    this.stopScheduler();
    this.stopAllSources();
    if (this.audioContext) {
      this.audioContext.close();
      this.audioContext = null;
    }
  }
}

/**
 * Browser audio capture for microphone input
 */
export class BrowserAudioCapture {
  /**
   * @param {Object} options - Configuration options
   */
  constructor(options = {}) {
    this.options = {
      sampleRate: 16000,
      ...options,
    };

    /** @type {MediaStream|null} */
    this.stream = null;

    /** @type {AudioContext|null} */
    this.audioContext = null;

    /** @type {ScriptProcessorNode|null} */
    this.processor = null;

    this.isCapturing = false;

    // Event emitter
    const emitter = createEventEmitter();
    this.on = emitter.on.bind(emitter);
    this.off = emitter.off.bind(emitter);
    this.emit = emitter.emit.bind(emitter);
  }

  /**
   * Start capturing from microphone
   */
  async start() {
    if (this.isCapturing) {
      return;
    }

    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: this.options.sampleRate,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
        },
      });

      this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: this.options.sampleRate,
      });

      const source = this.audioContext.createMediaStreamSource(this.stream);

      this.processor = this.audioContext.createScriptProcessor(4096, 1, 1);

      this.processor.onaudioprocess = (e) => {
        const inputData = e.inputBuffer.getChannelData(0);
        const samples = new Float32Array(inputData);
        this.emit('audio', samples);
      };

      source.connect(this.processor);
      this.processor.connect(this.audioContext.destination);

      this.isCapturing = true;
      this.emit('start');

    } catch (error) {
      console.error('Failed to start audio capture:', error);
      this.emit('error', error);
      throw error;
    }
  }

  /**
   * Stop capturing
   */
  stop() {
    if (!this.isCapturing) {
      return;
    }

    if (this.processor) {
      this.processor.disconnect();
      this.processor = null;
    }

    if (this.stream) {
      this.stream.getTracks().forEach(track => track.stop());
      this.stream = null;
    }

    if (this.audioContext) {
      this.audioContext.close();
      this.audioContext = null;
    }

    this.isCapturing = false;
    this.emit('stop');
  }

  /**
   * Check if currently capturing
   * @returns {boolean}
   */
  get capturing() {
    return this.isCapturing;
  }
}
