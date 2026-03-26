/**
 * Session Manager - Handles multi-session API interactions
 */

const API_BASE = '';

export class SessionManager {
  constructor() {
    this.sessions = [];
    this.models = [];
    this.mediaFiles = [];
    this.modes = [];
    this.noiseCancellationOptions = [];
    this.diarizationOptions = [];
    this.srtStreams = [];
    this.srtConfigured = false;
    this.currentSession = null;
    this.listeners = {};
    this.pollInterval = null;
  }

  on(event, callback) {
    if (!this.listeners[event]) {
      this.listeners[event] = [];
    }
    this.listeners[event].push(callback);
  }

  emit(event, data) {
    if (this.listeners[event]) {
      this.listeners[event].forEach(cb => cb(data));
    }
  }

  async fetchModels() {
    try {
      const res = await fetch(`${API_BASE}/api/models`);
      const json = await res.json();
      if (json.success) {
        this.models = json.data;
        this.emit('modelsLoaded', this.models);
      }
      return this.models;
    } catch (e) {
      console.error('Failed to fetch models:', e);
      return [];
    }
  }

  async fetchMedia() {
    try {
      const res = await fetch(`${API_BASE}/api/media`);
      const json = await res.json();
      if (json.success) {
        this.mediaFiles = json.data;
        this.emit('mediaLoaded', this.mediaFiles);
      }
      return this.mediaFiles;
    } catch (e) {
      console.error('Failed to fetch media:', e);
      return [];
    }
  }

  async fetchModes() {
    try {
      const res = await fetch(`${API_BASE}/api/modes`);
      const json = await res.json();
      if (json.success) {
        this.modes = json.data;
        this.emit('modesLoaded', this.modes);
      }
      return this.modes;
    } catch (e) {
      console.error('Failed to fetch modes:', e);
      return [];
    }
  }

  async fetchNoiseCancellation() {
    try {
      const res = await fetch(`${API_BASE}/api/noise-cancellation`);
      const json = await res.json();
      if (json.success) {
        this.noiseCancellationOptions = json.data;
        this.emit('noiseCancellationLoaded', this.noiseCancellationOptions);
      }
      return this.noiseCancellationOptions;
    } catch (e) {
      console.error('Failed to fetch noise cancellation options:', e);
      return [];
    }
  }

  async fetchDiarization() {
    try {
      const res = await fetch(`${API_BASE}/api/diarization`);
      const json = await res.json();
      if (json.success) {
        this.diarizationOptions = json.data;
        this.emit('diarizationLoaded', this.diarizationOptions);
      }
      return this.diarizationOptions;
    } catch (e) {
      console.error('Failed to fetch diarization options:', e);
      return [];
    }
  }

  async fetchSrtStreams() {
    try {
      const res = await fetch(`${API_BASE}/api/srt-streams`);
      const json = await res.json();
      if (json.success) {
        this.srtStreams = json.streams;
        this.srtConfigured = json.configured;
        this.emit('srtStreamsLoaded', { streams: this.srtStreams, configured: this.srtConfigured });
      }
      return this.srtStreams;
    } catch (e) {
      console.error('Failed to fetch SRT streams:', e);
      return [];
    }
  }

  async fetchSessions() {
    try {
      const res = await fetch(`${API_BASE}/api/sessions`);
      const json = await res.json();
      if (json.success) {
        this.sessions = json.data;
        this.emit('sessionsUpdated', this.sessions);
      }
      return this.sessions;
    } catch (e) {
      console.error('Failed to fetch sessions:', e);
      return [];
    }
  }

  /**
   * Create a new session
   * @param {string} modelId - Model ID to use
   * @param {Object} options - Session options
   * @param {string} [options.mediaId] - Media file ID (for file sessions)
   * @param {number} [options.srtChannelId] - SRT channel ID (for SRT stream sessions)
   * @param {string} [options.mode='speedy'] - Latency mode
   * @param {string} [options.language='de'] - Language code
   * @param {string} [options.noiseCancellation='none'] - Noise cancellation type
   * @param {boolean} [options.diarization=false] - Enable diarization
   * @param {Object} [options.pauseConfig] - Pause detection config
   * @param {Object} [options.growingSegmentsConfig] - Growing segments config
   * @param {string} [options.sentenceCompletion='minimal'] - Sentence completion mode
   * @param {string} [options.fabEnabled='default'] - FAB forwarding ("default", "enabled", "disabled")
   * @param {string} [options.fabUrl=''] - FAB endpoint URL override
   * @param {string} [options.fabSendType='default'] - FAB send type ("default", "growing", "confirmed")
   */
  async createSession(modelId, options = {}) {
    const {
      mediaId = null,
      srtChannelId = null,
      mode = 'speedy',
      language = 'de',
      noiseCancellation = 'none',
      diarization = false,
      pauseConfig = null,
      growingSegmentsConfig = null,
      sentenceCompletion = 'minimal',
      fabEnabled = 'default',
      fabUrl = '',
      fabSendType = 'default',
      withoutTranscription = false
    } = options;

    try {
      const body = {
        mode,
        language,
        noise_cancellation: noiseCancellation,
        diarization: diarization,
        sentence_completion: sentenceCompletion
      };

      if (withoutTranscription) {
        body.without_transcription = true;
      } else {
        body.model_id = modelId;
      }

      // Add source - either media file or SRT channel (mutually exclusive)
      if (mediaId) {
        body.media_id = mediaId;
      } else if (srtChannelId !== null && srtChannelId !== undefined) {
        body.srt_channel_id = srtChannelId;
      } else {
        throw new Error('Must specify either mediaId or srtChannelId');
      }

      // Add pause config if provided
      if (pauseConfig) {
        body.pause_config = pauseConfig;
      }

      // Add growing segments config if provided
      if (growingSegmentsConfig) {
        body.growing_segments_config = growingSegmentsConfig;
      }


      // Add FAB config if not default
      if (fabEnabled && fabEnabled !== 'default') {
        body.fab_enabled = fabEnabled;
      }
      if (fabUrl) {
        body.fab_url = fabUrl;
      }
      if (fabSendType && fabSendType !== 'default') {
        body.fab_send_type = fabSendType;
      }

      const res = await fetch(`${API_BASE}/api/sessions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });
      const json = await res.json();
      if (json.success) {
        this.currentSession = json.data;
        this.emit('sessionCreated', this.currentSession);
        await this.fetchSessions();
        return this.currentSession;
      } else {
        throw new Error(json.error || 'Failed to create session');
      }
    } catch (e) {
      console.error('Failed to create session:', e);
      throw e;
    }
  }

  async startSession(sessionId) {
    try {
      const res = await fetch(`${API_BASE}/api/sessions/${sessionId}/start`, {
        method: 'POST'
      });
      const json = await res.json();
      if (json.success) {
        this.currentSession = json.data;
        this.emit('sessionStarted', this.currentSession);
        return this.currentSession;
      } else {
        throw new Error(json.error || 'Failed to start session');
      }
    } catch (e) {
      console.error('Failed to start session:', e);
      throw e;
    }
  }

  async stopSession(sessionId) {
    try {
      const res = await fetch(`${API_BASE}/api/sessions/${sessionId}`, {
        method: 'DELETE'
      });
      const json = await res.json();
      if (json.success) {
        this.emit('sessionStopped', sessionId);
        await this.fetchSessions();
      }
      return json.success;
    } catch (e) {
      console.error('Failed to stop session:', e);
      return false;
    }
  }

  async getSession(sessionId) {
    try {
      const res = await fetch(`${API_BASE}/api/sessions/${sessionId}`);
      const json = await res.json();
      if (json.success) {
        return json.data;
      }
      return null;
    } catch (e) {
      console.error('Failed to get session:', e);
      return null;
    }
  }

  async uploadMedia(file) {
    try {
      const formData = new FormData();
      formData.append('file', file);

      const res = await fetch(`${API_BASE}/api/media/upload`, {
        method: 'POST',
        body: formData
      });
      const json = await res.json();
      if (json.success) {
        await this.fetchMedia();
        this.emit('mediaUploaded', json.data);
        return json.data;
      } else {
        throw new Error(json.error || 'Failed to upload media');
      }
    } catch (e) {
      console.error('Failed to upload media:', e);
      throw e;
    }
  }

  async deleteMedia(mediaId) {
    try {
      const res = await fetch(`${API_BASE}/api/media/${mediaId}`, {
        method: 'DELETE'
      });
      const json = await res.json();
      if (json.success) {
        await this.fetchMedia();
        this.emit('mediaDeleted', mediaId);
      }
      return json.success;
    } catch (e) {
      console.error('Failed to delete media:', e);
      return false;
    }
  }

  startPolling(intervalMs = 2000) {
    this.stopPolling();
    this.pollInterval = setInterval(() => {
      this.fetchSessions();
    }, intervalMs);
  }

  stopPolling() {
    if (this.pollInterval) {
      clearInterval(this.pollInterval);
      this.pollInterval = null;
    }
  }

  getWebSocketUrl(sessionId) {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    return `${protocol}//${host}/ws/${sessionId}`;
  }

}

export function formatDuration(seconds) {
  if (!seconds || isNaN(seconds)) return '0:00';
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}

export function formatFileSize(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}
