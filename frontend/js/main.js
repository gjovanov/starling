/**
 * Main entry point - orchestrates all modules
 *
 * Key sync features:
 * - Subtitle buffer: Holds subtitles until audio playback reaches their start time
 * - Time sync: Displays subtitles based on audio playback position
 * - Latency compensation: Optional delay adjustment for perfect sync
 */
import { config } from './config.js';
import { WebSocketClient } from './modules/websocket.js';
import { AudioPlayer } from './modules/audio.js';
import { SubtitleRenderer } from './modules/subtitles.js';
import { formatTime } from './modules/utils.js';

// Application state
const state = {
  connected: false,
  audioEnabled: true,
  totalDuration: 0,
  // Subtitle sync
  pendingSubtitles: [],      // Subtitles waiting to be displayed
  latencyCompensation: 0,    // Manual latency adjustment (seconds)
};

// Initialize modules
let wsClient = null;
let audioPlayer = null;
let subtitleRenderer = null;

// DOM elements
const elements = {};

/**
 * Initialize the application
 */
function init() {
  // Cache DOM elements
  elements.connectionStatus = document.getElementById('connection-status');
  elements.bufferInfo = document.getElementById('buffer-info');
  elements.durationInfo = document.getElementById('duration-info');
  elements.playPauseBtn = document.getElementById('play-pause');
  elements.currentTime = document.getElementById('current-time');
  elements.totalDuration = document.getElementById('total-duration');
  elements.progressBuffered = document.getElementById('progress-buffered');
  elements.progressPlayed = document.getElementById('progress-played');
  elements.progressContainer = document.getElementById('progress-container');
  elements.connectBtn = document.getElementById('connect-btn');
  elements.liveSubtitle = document.getElementById('live-subtitle');
  elements.transcriptContent = document.getElementById('transcript-content');
  elements.autoScrollCheckbox = document.getElementById('auto-scroll');
  elements.showTimestampsCheckbox = document.getElementById('show-timestamps');
  elements.exportBtn = document.getElementById('export-btn');
  elements.clearBtn = document.getElementById('clear-btn');
  elements.exportModal = document.getElementById('export-modal');
  elements.wsUrlInput = document.getElementById('ws-url');

  // Initialize subtitle renderer
  subtitleRenderer = new SubtitleRenderer(
    elements.liveSubtitle,
    elements.transcriptContent,
    {
      maxSegments: config.subtitles.maxSegments,
      autoScroll: config.subtitles.autoScroll,
      showTimestamps: config.subtitles.showTimestamps,
      speakerColors: config.speakerColors,
    }
  );

  // Initialize audio player with large jitter buffer for streaming
  audioPlayer = new AudioPlayer({
    sampleRate: config.audio.sampleRate,
    jitterBufferMs: 2000,      // 2 second initial buffer
    minBufferMs: 500,          // Pause if below 500ms
    resumeBufferMs: 1500,      // Resume when buffer reaches 1.5s
    scheduleAheadMs: 1000,     // Schedule 1 second ahead
    schedulerIntervalMs: 50,   // Check every 50ms
    maxScheduleChunkMs: 500,   // Max chunk size
  });

  // Set up event listeners
  setupEventListeners();

  // Initialize URL from config
  if (elements.wsUrlInput) {
    elements.wsUrlInput.value = config.wsUrl;
  }

  console.log('Application initialized');
}

/**
 * Set up all event listeners
 */
function setupEventListeners() {
  // Connect button
  elements.connectBtn.addEventListener('click', () => {
    if (state.connected) {
      disconnect();
    } else {
      connect();
    }
  });

  // Play/Pause button
  elements.playPauseBtn.addEventListener('click', async () => {
    if (audioPlayer.playing) {
      audioPlayer.pause();
    } else {
      await audioPlayer.play();
    }
  });

  // Progress bar seek
  elements.progressContainer.addEventListener('click', (e) => {
    const rect = elements.progressContainer.getBoundingClientRect();
    const percent = (e.clientX - rect.left) / rect.width;
    const time = percent * state.totalDuration;
    audioPlayer.seek(time);
    subtitleRenderer.updateTime(time);
  });

  // Auto-scroll toggle
  elements.autoScrollCheckbox.addEventListener('change', (e) => {
    subtitleRenderer.setAutoScroll(e.target.checked);
  });

  // Show timestamps toggle
  elements.showTimestampsCheckbox.addEventListener('change', (e) => {
    subtitleRenderer.setShowTimestamps(e.target.checked);
  });

  // Export button
  elements.exportBtn.addEventListener('click', () => {
    elements.exportModal.style.display = 'flex';
  });

  // Export options
  document.getElementById('export-txt').addEventListener('click', () => {
    downloadTranscript('transcript.txt', subtitleRenderer.getTranscript());
    elements.exportModal.style.display = 'none';
  });

  document.getElementById('export-timestamps').addEventListener('click', () => {
    downloadTranscript('transcript_timestamps.txt', subtitleRenderer.getTranscriptWithTimestamps());
    elements.exportModal.style.display = 'none';
  });

  document.getElementById('export-json').addEventListener('click', () => {
    downloadTranscript('transcript.json', subtitleRenderer.exportJSON());
    elements.exportModal.style.display = 'none';
  });

  document.getElementById('close-modal').addEventListener('click', () => {
    elements.exportModal.style.display = 'none';
  });

  // Clear button
  elements.clearBtn.addEventListener('click', () => {
    if (confirm('Clear all transcript data?')) {
      subtitleRenderer.clear();
      audioPlayer.clear();
      state.totalDuration = 0;
      state.pendingSubtitles = [];
      updateUI();
    }
  });

  // Close modal on outside click
  elements.exportModal.addEventListener('click', (e) => {
    if (e.target === elements.exportModal) {
      elements.exportModal.style.display = 'none';
    }
  });

  // Audio player events
  audioPlayer.on('play', () => {
    elements.playPauseBtn.querySelector('.icon-play').style.display = 'none';
    elements.playPauseBtn.querySelector('.icon-pause').style.display = 'inline';
  });

  audioPlayer.on('pause', () => {
    elements.playPauseBtn.querySelector('.icon-play').style.display = 'inline';
    elements.playPauseBtn.querySelector('.icon-pause').style.display = 'none';
  });

  audioPlayer.on('timeUpdate', (time) => {
    elements.currentTime.textContent = formatTime(time);

    // Process pending subtitles that should now be visible
    processPendingSubtitles(time);

    // Update subtitle highlight
    subtitleRenderer.updateTime(time);
    updateProgressBar();
  });

  audioPlayer.on('bufferUpdate', ({ buffered, total }) => {
    updateProgressBar();
  });

  audioPlayer.on('buffering', () => {
    elements.bufferInfo.textContent = 'Buffering...';
  });

  audioPlayer.on('bufferReady', () => {
    elements.bufferInfo.textContent = 'Buffer ready';
  });

  audioPlayer.on('bufferUnderrun', ({ count }) => {
    elements.bufferInfo.textContent = `Buffer underrun #${count} - pausing...`;
    console.warn(`Buffer underrun #${count}, waiting for buffer to recover`);
  });

  audioPlayer.on('bufferRecovered', () => {
    elements.bufferInfo.textContent = 'Buffer recovered - resuming';
    console.log('Buffer recovered, resuming playback');
  });

  // Subtitle renderer events
  subtitleRenderer.on('seek', (time) => {
    audioPlayer.seek(time);
  });
}

/**
 * Process pending subtitles based on current playback time
 * This ensures subtitles appear in sync with audio
 * @param {number} currentTime - Current audio playback time
 */
function processPendingSubtitles(currentTime) {
  // Adjust for latency compensation
  const adjustedTime = currentTime + state.latencyCompensation;

  // Find subtitles that should be displayed now
  const toDisplay = [];
  const stillPending = [];

  for (const subtitle of state.pendingSubtitles) {
    // Display subtitle when audio reaches its START time
    // (with a small margin for smoother appearance)
    if (subtitle.start <= adjustedTime + 0.1) {
      toDisplay.push(subtitle);
    } else {
      stillPending.push(subtitle);
    }
  }

  // Update pending list
  state.pendingSubtitles = stillPending;

  // Display subtitles in order
  for (const subtitle of toDisplay) {
    subtitleRenderer.addSegment(subtitle);
  }
}

/**
 * Connect to WebSocket server
 */
function connect() {
  const url = elements.wsUrlInput?.value || config.wsUrl;

  wsClient = new WebSocketClient(url, {
    reconnect: config.reconnect.enabled,
    reconnectDelay: config.reconnect.delay,
    maxReconnectDelay: config.reconnect.maxDelay,
    maxReconnectAttempts: config.reconnect.maxAttempts,
  });

  // WebSocket events
  wsClient.on('connect', () => {
    state.connected = true;
    updateConnectionStatus('connected');
    elements.connectBtn.textContent = 'Disconnect';
    elements.playPauseBtn.disabled = false;
    console.log('Connected to server');
  });

  wsClient.on('disconnect', ({ code, reason }) => {
    state.connected = false;
    updateConnectionStatus('disconnected');
    elements.connectBtn.textContent = 'Connect';
    console.log('Disconnected:', code, reason);
  });

  wsClient.on('reconnecting', ({ attempt, delay }) => {
    updateConnectionStatus('reconnecting');
    console.log(`Reconnecting (attempt ${attempt})...`);
  });

  wsClient.on('reconnectFailed', () => {
    updateConnectionStatus('disconnected');
    console.error('Failed to reconnect');
  });

  wsClient.on('error', (error) => {
    console.error('WebSocket error:', error);
  });

  wsClient.on('welcome', (message) => {
    console.log('Server welcome:', message);
  });

  wsClient.on('audio', ({ data, timestamp }) => {
    if (state.audioEnabled) {
      audioPlayer.appendChunk(data, timestamp);
    }
  });

  wsClient.on('subtitle', (segment) => {
    // Add to pending queue - will be displayed when audio catches up
    // Or display immediately if we're caught up or not playing audio
    const currentTime = audioPlayer.currentTime;

    if (!state.audioEnabled || !audioPlayer.playing || segment.start <= currentTime + 0.5) {
      // Display immediately if:
      // - Audio is disabled
      // - Not playing yet
      // - Audio has already passed this segment's start time
      subtitleRenderer.addSegment(segment);
    } else {
      // Queue for later display
      state.pendingSubtitles.push(segment);
      // Keep sorted by start time
      state.pendingSubtitles.sort((a, b) => a.start - b.start);
    }
  });

  wsClient.on('status', ({ bufferTime, totalDuration }) => {
    state.totalDuration = totalDuration;
    const bufferedAudio = audioPlayer.getBufferedDuration();
    elements.bufferInfo.textContent = `Buffer: ${bufferedAudio.toFixed(1)}s`;
    elements.durationInfo.textContent = `Duration: ${formatTime(totalDuration)}`;
    elements.totalDuration.textContent = formatTime(totalDuration);
    updateProgressBar();
  });

  wsClient.on('end', ({ totalDuration }) => {
    state.totalDuration = totalDuration;
    elements.totalDuration.textContent = formatTime(totalDuration);
    console.log('Stream ended, total duration:', formatTime(totalDuration));

    // Display any remaining pending subtitles
    for (const subtitle of state.pendingSubtitles) {
      subtitleRenderer.addSegment(subtitle);
    }
    state.pendingSubtitles = [];
  });

  wsClient.on('serverError', ({ message }) => {
    console.error('Server error:', message);
    alert(`Server error: ${message}`);
  });

  // Connect
  updateConnectionStatus('connecting');
  wsClient.connect();
}

/**
 * Disconnect from WebSocket server
 */
function disconnect() {
  if (wsClient) {
    wsClient.disconnect();
    wsClient = null;
  }
  state.connected = false;
  updateConnectionStatus('disconnected');
  elements.connectBtn.textContent = 'Connect';
}

/**
 * Update connection status display
 * @param {string} status - 'connected' | 'disconnected' | 'connecting' | 'reconnecting'
 */
function updateConnectionStatus(status) {
  elements.connectionStatus.className = `status ${status}`;

  switch (status) {
    case 'connected':
      elements.connectionStatus.textContent = 'Connected';
      break;
    case 'disconnected':
      elements.connectionStatus.textContent = 'Disconnected';
      break;
    case 'connecting':
      elements.connectionStatus.textContent = 'Connecting...';
      break;
    case 'reconnecting':
      elements.connectionStatus.textContent = 'Reconnecting...';
      break;
  }
}

/**
 * Update progress bar
 */
function updateProgressBar() {
  if (state.totalDuration === 0) {
    elements.progressBuffered.style.width = '0%';
    elements.progressPlayed.style.width = '0%';
    return;
  }

  const bufferedPercent = (audioPlayer.duration / state.totalDuration) * 100;
  const playedPercent = (audioPlayer.currentTime / state.totalDuration) * 100;

  elements.progressBuffered.style.width = `${Math.min(bufferedPercent, 100)}%`;
  elements.progressPlayed.style.width = `${Math.min(playedPercent, 100)}%`;
}

/**
 * Update UI state
 */
function updateUI() {
  elements.currentTime.textContent = formatTime(audioPlayer.currentTime);
  elements.totalDuration.textContent = formatTime(state.totalDuration);
  updateProgressBar();
}

/**
 * Download text as file
 * @param {string} filename - File name
 * @param {string} content - File content
 */
function downloadTranscript(filename, content) {
  const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

/**
 * Set latency compensation (for manual sync adjustment)
 * Positive values = subtitles appear earlier
 * Negative values = subtitles appear later
 * @param {number} seconds - Compensation in seconds
 */
window.setLatencyCompensation = function(seconds) {
  state.latencyCompensation = seconds;
  console.log(`Latency compensation set to ${seconds}s`);
};

// Initialize when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
