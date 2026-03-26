/**
 * Main entry point for WebRTC mode
 *
 * Uses WebRTC for ultra-low-latency audio streaming.
 * Browser handles all buffering via native jitter buffer.
 * Configuration is loaded dynamically from server (/api/config).
 */
import { loadConfig, getConfig } from './config.js';
import { WebRTCClient } from './modules/webrtc.js';
import { SubtitleRenderer } from './modules/subtitles.js';
import { formatTime } from './modules/utils.js';

// Application state
const state = {
  connected: false,
  totalDuration: 0,
};

// Modules
let webrtcClient = null;
let subtitleRenderer = null;

// DOM elements
const elements = {};

/**
 * Initialize the application
 */
async function init() {
  // Load configuration from server first
  const config = await loadConfig();

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
  elements.audioPlayer = document.getElementById('audio-player');
  elements.latencyInfo = document.getElementById('latency-info');

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

  // Set up event listeners
  setupEventListeners();

  // Initialize URL from config
  if (elements.wsUrlInput) {
    elements.wsUrlInput.value = config.wsUrl;
  }

  // Show WebRTC mode indicator
  elements.bufferInfo.textContent = 'WebRTC Mode';

  console.log('WebRTC Application initialized with config from server');
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
    if (webrtcClient?.playing) {
      webrtcClient.pause();
      elements.playPauseBtn.querySelector('.icon-play').style.display = 'inline';
      elements.playPauseBtn.querySelector('.icon-pause').style.display = 'none';
    } else if (webrtcClient) {
      await webrtcClient.play();
      elements.playPauseBtn.querySelector('.icon-play').style.display = 'none';
      elements.playPauseBtn.querySelector('.icon-pause').style.display = 'inline';
    }
  });

  // Audio element events
  elements.audioPlayer.addEventListener('play', () => {
    elements.playPauseBtn.querySelector('.icon-play').style.display = 'none';
    elements.playPauseBtn.querySelector('.icon-pause').style.display = 'inline';
  });

  elements.audioPlayer.addEventListener('pause', () => {
    elements.playPauseBtn.querySelector('.icon-play').style.display = 'inline';
    elements.playPauseBtn.querySelector('.icon-pause').style.display = 'none';
  });

  elements.audioPlayer.addEventListener('timeupdate', () => {
    const currentTime = elements.audioPlayer.currentTime;
    elements.currentTime.textContent = formatTime(currentTime);
    subtitleRenderer.updateTime(currentTime);
    updateProgressBar();
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
      state.totalDuration = 0;
      updateUI();
    }
  });

  // Close modal on outside click
  elements.exportModal.addEventListener('click', (e) => {
    if (e.target === elements.exportModal) {
      elements.exportModal.style.display = 'none';
    }
  });

  // Subtitle renderer events
  subtitleRenderer.on('seek', (time) => {
    if (elements.audioPlayer) {
      elements.audioPlayer.currentTime = time;
    }
  });

  // Update stats periodically
  setInterval(updateStats, 1000);
}

/**
 * Connect to WebRTC server
 */
async function connect() {
  const config = getConfig();
  const url = elements.wsUrlInput?.value || config.wsUrl;

  // Create WebRTC client with ICE servers and transport policy from server config
  webrtcClient = new WebRTCClient(url, {
    iceServers: config.iceServers,
    iceTransportPolicy: config.iceTransportPolicy || 'all'
  });

  // WebRTC events
  webrtcClient.on('welcome', (msg) => {
    console.log('Server welcome:', msg);
  });

  webrtcClient.on('connected', () => {
    state.connected = true;
    updateConnectionStatus('connected');
    elements.connectBtn.textContent = 'Disconnect';
    elements.playPauseBtn.disabled = false;
    elements.bufferInfo.textContent = 'WebRTC Connected';
    console.log('WebRTC connected');
  });

  webrtcClient.on('trackReceived', (stream) => {
    console.log('Audio track received');
    elements.bufferInfo.textContent = 'Audio streaming';
  });

  webrtcClient.on('autoplayBlocked', () => {
    elements.bufferInfo.textContent = 'Click Play to start';
  });

  webrtcClient.on('disconnect', ({ code, reason }) => {
    state.connected = false;
    updateConnectionStatus('disconnected');
    elements.connectBtn.textContent = 'Connect';
    // Clear any stuck partial subtitle on disconnect
    subtitleRenderer.clearCurrent();
    console.log('Disconnected:', code, reason);
  });

  webrtcClient.on('connectionFailed', () => {
    state.connected = false;
    updateConnectionStatus('reconnecting');
    elements.bufferInfo.textContent = 'ICE failed, reconnecting...';
    // Clear any stuck partial subtitle on connection failure
    subtitleRenderer.clearCurrent();
    console.error('WebRTC connection failed');
  });

  webrtcClient.on('reconnecting', ({ attempt, delay }) => {
    updateConnectionStatus('reconnecting');
    elements.bufferInfo.textContent = `Reconnecting (attempt ${attempt})...`;
  });

  webrtcClient.on('reconnectFailed', () => {
    updateConnectionStatus('disconnected');
    elements.bufferInfo.textContent = 'Reconnection failed. Click Connect to retry.';
  });

  webrtcClient.on('error', (error) => {
    console.error('WebRTC error:', error);
  });

  webrtcClient.on('subtitle', (segment) => {
    subtitleRenderer.addSegment(segment);
  });

  webrtcClient.on('status', ({ bufferTime, totalDuration }) => {
    state.totalDuration = totalDuration;
    elements.durationInfo.textContent = `Duration: ${formatTime(totalDuration)}`;
    elements.totalDuration.textContent = formatTime(totalDuration);
    updateProgressBar();
  });

  webrtcClient.on('end', ({ totalDuration }) => {
    state.totalDuration = totalDuration;
    elements.totalDuration.textContent = formatTime(totalDuration);
    console.log('Stream ended, total duration:', formatTime(totalDuration));
  });

  webrtcClient.on('serverError', ({ message }) => {
    console.error('Server error:', message);
    alert(`Server error: ${message}`);
  });

  // Connect
  updateConnectionStatus('connecting');
  try {
    await webrtcClient.connect(elements.audioPlayer);
  } catch (error) {
    console.error('Failed to connect:', error);
    updateConnectionStatus('disconnected');
    elements.bufferInfo.textContent = 'Connection failed';
  }
}

/**
 * Disconnect from server
 */
function disconnect() {
  if (webrtcClient) {
    webrtcClient.disconnect();
    webrtcClient = null;
  }
  state.connected = false;
  updateConnectionStatus('disconnected');
  elements.connectBtn.textContent = 'Connect';
}

/**
 * Update connection status display
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

  const currentTime = elements.audioPlayer?.currentTime || 0;
  const playedPercent = (currentTime / state.totalDuration) * 100;

  // For live streams, buffered is always ahead
  elements.progressBuffered.style.width = '100%';
  elements.progressPlayed.style.width = `${Math.min(playedPercent, 100)}%`;
}

/**
 * Update UI state
 */
function updateUI() {
  const currentTime = elements.audioPlayer?.currentTime || 0;
  elements.currentTime.textContent = formatTime(currentTime);
  elements.totalDuration.textContent = formatTime(state.totalDuration);
  updateProgressBar();
}

/**
 * Update WebRTC stats display
 */
async function updateStats() {
  if (!webrtcClient || !state.connected) {
    return;
  }

  try {
    const stats = await webrtcClient.getStats();
    if (stats && elements.latencyInfo) {
      const rtt = (stats.roundTripTime * 1000).toFixed(0);
      const jitter = (stats.jitter * 1000).toFixed(1);
      elements.latencyInfo.textContent = `RTT: ${rtt}ms | Jitter: ${jitter}ms`;
    }
  } catch (e) {
    // Ignore stats errors
  }
}

/**
 * Download text as file
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

// Initialize when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
