/**
 * WebRTC client for ultra-low-latency audio streaming
 *
 * Features:
 * - Automatic jitter buffer (handled by browser)
 * - Opus codec (optimized for speech)
 * - ~100-300ms latency
 * - Native audio playback via <audio> element
 * - Dynamic configuration from server (/api/config)
 */
import { createEventEmitter } from './utils.js';

export class WebRTCClient {
  /**
   * @param {string} wsUrl - WebSocket URL for signaling
   * @param {Object} options - Configuration options
   * @param {Array} options.iceServers - ICE servers for WebRTC (from server config)
   * @param {boolean} options.autoReconnect - Enable auto-reconnection (default: true)
   * @param {number} options.reconnectDelay - Initial reconnect delay in ms (default: 2000)
   * @param {number} options.maxReconnectDelay - Max reconnect delay in ms (default: 30000)
   * @param {number} options.maxReconnectAttempts - Max attempts before giving up (default: 10)
   */
  constructor(wsUrl, options = {}) {
    this.wsUrl = wsUrl;

    // Default ICE servers (fallback if not provided by server)
    const defaultIceServers = [
      { urls: 'stun:stun.l.google.com:19302' }
    ];

    this.options = {
      iceServers: options.iceServers || defaultIceServers,
      // Allow all transport (direct + relay)
      iceTransportPolicy: 'all',
      // Reconnection options
      autoReconnect: options.autoReconnect !== false,
      reconnectDelay: options.reconnectDelay || 2000,
      maxReconnectDelay: options.maxReconnectDelay || 30000,
      maxReconnectAttempts: options.maxReconnectAttempts || 10,
      ...options,
    };

    /** @type {WebSocket|null} */
    this.ws = null;

    /** @type {RTCPeerConnection|null} */
    this.pc = null;

    /** @type {HTMLAudioElement|null} */
    this.audioElement = null;

    /** @type {MediaStream|null} */
    this.remoteStream = null;

    this.connected = false;
    this.clientId = null;

    // Reconnection state
    this.intentionalClose = false;
    this.reconnectAttempts = 0;
    this.reconnectTimeout = null;

    // Event emitter
    const emitter = createEventEmitter();
    this.on = emitter.on.bind(emitter);
    this.off = emitter.off.bind(emitter);
    this.emit = emitter.emit.bind(emitter);
  }

  /**
   * Connect to WebRTC server
   * @param {HTMLAudioElement} audioElement - Audio element for playback
   */
  async connect(audioElement) {
    console.log('[WebRTC] connect() called with audioElement:', audioElement);
    console.log('[WebRTC] Connecting to:', this.wsUrl);

    if (audioElement) {
      this.audioElement = audioElement;
    }

    this.intentionalClose = false;

    return new Promise((resolve, reject) => {
      // Create WebSocket for signaling
      console.log('[WebRTC] Creating WebSocket...');
      this.ws = new WebSocket(this.wsUrl);

      this.ws.onopen = () => {
        console.log('[WebRTC] Signaling connected');
        this.reconnectAttempts = 0; // Reset on successful connection
        this.setupPeerConnection();
      };

      this.ws.onclose = (event) => {
        console.log('[WebRTC] Signaling disconnected:', event.code, event.reason);
        this.connected = false;
        this.emit('disconnect', { code: event.code, reason: event.reason });

        // Auto-reconnect if enabled and not intentional close
        if (!this.intentionalClose && this.options.autoReconnect) {
          this.scheduleReconnect();
        }
      };

      this.ws.onerror = (error) => {
        console.error('[WebRTC] Signaling error:', error);
        this.emit('error', error);
        reject(error);
      };

      this.ws.onmessage = async (event) => {
        try {
          const msg = JSON.parse(event.data);
          await this.handleSignalingMessage(msg, resolve);
        } catch (e) {
          console.error('[WebRTC] Error handling message:', e);
        }
      };
    });
  }

  /**
   * Schedule a reconnection attempt with exponential backoff
   */
  scheduleReconnect() {
    if (this.options.maxReconnectAttempts > 0 &&
        this.reconnectAttempts >= this.options.maxReconnectAttempts) {
      console.error('[WebRTC] Max reconnection attempts reached');
      this.emit('reconnectFailed');
      return;
    }

    // Exponential backoff
    const delay = Math.min(
      this.options.reconnectDelay * Math.pow(1.5, this.reconnectAttempts),
      this.options.maxReconnectDelay
    );

    console.log(`[WebRTC] Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts + 1})`);
    this.emit('reconnecting', { attempt: this.reconnectAttempts + 1, delay });

    this.reconnectTimeout = setTimeout(() => {
      this.reconnectAttempts++;
      this.reconnect();
    }, delay);
  }

  /**
   * Perform reconnection (cleanup old connection and create new one)
   */
  async reconnect() {
    console.log('[WebRTC] Attempting reconnection...');

    // Clear ICE disconnect timer
    if (this.iceDisconnectTimer) {
      clearTimeout(this.iceDisconnectTimer);
      this.iceDisconnectTimer = null;
    }

    // Cleanup old peer connection
    if (this.pc) {
      this.pc.close();
      this.pc = null;
    }

    // Close old WebSocket (prevent duplicate connections)
    if (this.ws) {
      this.intentionalClose = true; // prevent ws.onclose from scheduling another reconnect
      this.ws.close();
      this.ws = null;
      this.intentionalClose = false;
    }

    // Don't close the audio element - we want to resume playback
    if (this.audioElement) {
      this.audioElement.srcObject = null;
    }

    this.remoteStream = null;

    // Reconnect
    try {
      await this.connect();
      console.log('[WebRTC] Reconnected successfully');
    } catch (e) {
      console.error('[WebRTC] Reconnection failed:', e);
      this.scheduleReconnect();
    }
  }

  /**
   * Set up WebRTC peer connection
   */
  setupPeerConnection() {
    // Create peer connection with ICE transport policy from server config
    const rtcConfig = {
      iceServers: this.options.iceServers,
      iceTransportPolicy: this.options.iceTransportPolicy || 'all',
    };
    console.log('[WebRTC] RTCPeerConnection config:', JSON.stringify(rtcConfig, null, 2));
    this.pc = new RTCPeerConnection(rtcConfig);

    // Handle incoming audio track
    this.pc.ontrack = (event) => {
      console.log('[WebRTC] Received audio track');
      this.remoteStream = event.streams[0];

      if (this.audioElement) {
        this.audioElement.srcObject = this.remoteStream;
        this.audioElement.play().catch(e => {
          console.warn('[WebRTC] Autoplay blocked, user interaction required:', e);
          this.emit('autoplayBlocked');
        });
      }

      this.emit('trackReceived', event.streams[0]);
    };

    // Handle ICE candidates
    this.pc.onicecandidate = (event) => {
      if (event.candidate) {
        console.log('[WebRTC] Sending ICE candidate to server:', event.candidate.candidate);
        this.ws.send(JSON.stringify({
          type: 'ice-candidate',
          candidate: event.candidate.toJSON(),
        }));
      } else {
        console.log('[WebRTC] ICE gathering complete');
      }
    };

    // Handle connection state changes
    this.pc.onconnectionstatechange = () => {
      console.log('[WebRTC] Connection state:', this.pc.connectionState);

      switch (this.pc.connectionState) {
        case 'connected':
          this.connected = true;
          this.emit('connected');
          break;
        case 'disconnected':
        case 'failed':
          this.connected = false;
          this.emit('connectionFailed');
          break;
        case 'closed':
          this.connected = false;
          this.emit('closed');
          break;
      }
    };

    // Handle ICE connection state
    this.pc.oniceconnectionstatechange = () => {
      console.log('[WebRTC] ICE connection state:', this.pc.iceConnectionState);
      if (this.pc.iceConnectionState === 'failed') {
        console.error('[WebRTC] ICE connection failed, closing connection to trigger reconnect');
        this.emit('connectionFailed');
        // Close PC and WS to trigger auto-reconnect via ws.onclose
        if (this.pc) { this.pc.close(); this.pc = null; }
        if (this.ws) { this.ws.close(); }
      } else if (this.pc.iceConnectionState === 'connected' || this.pc.iceConnectionState === 'completed') {
        console.log('[WebRTC] ICE connected! Media should flow now.');
      } else if (this.pc.iceConnectionState === 'checking') {
        console.log('[WebRTC] ICE checking connectivity...');
      } else if (this.pc.iceConnectionState === 'disconnected') {
        console.warn('[WebRTC] ICE disconnected - waiting 5s for recovery');
        if (this.iceDisconnectTimer) clearTimeout(this.iceDisconnectTimer);
        this.iceDisconnectTimer = setTimeout(() => {
          if (this.pc && this.pc.iceConnectionState === 'disconnected') {
            console.error('[WebRTC] ICE did not recover, closing connection to trigger reconnect');
            this.emit('connectionFailed');
            // Close the peer connection and WebSocket to trigger auto-reconnect
            if (this.pc) { this.pc.close(); this.pc = null; }
            if (this.ws) { this.ws.close(); } // triggers ws.onclose → scheduleReconnect()
          }
        }, 5000);
      } else if (['connected', 'completed'].includes(this.pc.iceConnectionState)) {
        if (this.iceDisconnectTimer) {
          clearTimeout(this.iceDisconnectTimer);
          this.iceDisconnectTimer = null;
        }
      }
    };

    // Handle ICE gathering state
    this.pc.onicegatheringstatechange = () => {
      console.log('[WebRTC] ICE gathering state:', this.pc.iceGatheringState);
    };

    // Signal ready to receive offer
    this.ws.send(JSON.stringify({ type: 'ready' }));
  }

  /**
   * Handle signaling messages from server
   * @param {Object} msg - Signaling message
   * @param {Function} resolve - Promise resolve function
   */
  async handleSignalingMessage(msg, resolve) {
    switch (msg.type) {
      case 'welcome':
        console.log('[WebRTC] Welcome:', msg.message);
        this.clientId = msg.client_id;
        this.emit('welcome', msg);
        break;

      case 'offer':
        console.log('[WebRTC] Received offer');
        try {
          await this.pc.setRemoteDescription({
            type: 'offer',
            sdp: msg.sdp,
          });

          const answer = await this.pc.createAnswer();
          await this.pc.setLocalDescription(answer);

          this.ws.send(JSON.stringify({
            type: 'answer',
            sdp: answer.sdp,
          }));

          console.log('[WebRTC] Sent answer');
          resolve();
        } catch (e) {
          console.error('[WebRTC] Error handling offer:', e);
          this.emit('error', e);
        }
        break;

      case 'ice-candidate':
        if (msg.candidate) {
          const candidateStr = msg.candidate.candidate || '';

          // Only filter Docker network IPs for HOST candidates (not relay/srflx)
          // Docker bridge typically uses 172.17.x.x
          // WSL2 uses 172.x.x.x which should NOT be filtered
          const isHostCandidate = candidateStr.includes('typ host');
          const dockerBridgePattern = /\s172\.17\.\d+\.\d+\s/;

          if (isHostCandidate && dockerBridgePattern.test(candidateStr)) {
            console.log('[WebRTC] Skipping Docker bridge host candidate:', candidateStr);
            break;
          }

          console.log('[WebRTC] Received ICE candidate from server:', msg.candidate);
          console.log('[WebRTC] Current signaling state:', this.pc.signalingState);
          console.log('[WebRTC] Current ICE gathering state:', this.pc.iceGatheringState);
          try {
            // Construct RTCIceCandidate properly
            const iceCandidate = new RTCIceCandidate({
              candidate: msg.candidate.candidate,
              sdpMid: msg.candidate.sdpMid?.toString() || '0',
              sdpMLineIndex: msg.candidate.sdpMLineIndex ?? 0,
            });
            console.log('[WebRTC] Constructed RTCIceCandidate:', iceCandidate);
            await this.pc.addIceCandidate(iceCandidate);
            console.log('[WebRTC] Added ICE candidate successfully');
          } catch (e) {
            console.error('[WebRTC] Error adding ICE candidate:', e);
            console.error('[WebRTC] Candidate details:', JSON.stringify(msg.candidate));
          }
        }
        break;

      case 'subtitle':
        console.log('[WebRTC] Received subtitle message:', msg);
        this.emit('subtitle', {
          text: msg.text,
          growingText: msg.growing_text,
          delta: msg.delta,
          tailChanged: msg.tail_changed,
          speaker: msg.speaker,
          start: msg.start,
          end: msg.end,
          isFinal: msg.is_final,
          inferenceTimeMs: msg.inference_time_ms,
        });
        break;

      case 'status':
        this.emit('status', {
          bufferTime: msg.buffer_time || msg.progress_secs,
          totalDuration: msg.total_duration,
        });
        break;

      case 'end':
        this.emit('end', {
          totalDuration: msg.total_duration,
          is_live: msg.is_live,
        });
        break;

      case 'error':
        console.error('[WebRTC] Server error:', msg.message);
        this.emit('serverError', { message: msg.message });
        break;

      case 'reconnecting':
        // SRT stream reconnection
        console.log('[WebRTC] SRT reconnecting:', msg);
        this.emit('srtReconnecting', {
          attempt: msg.attempt,
          maxAttempts: msg.max_attempts,
          delayMs: msg.delay_ms,
        });
        break;

      case 'reconnected':
        // SRT stream reconnected
        console.log('[WebRTC] SRT reconnected');
        this.emit('srtReconnected');
        break;

      case 'vod_progress':
        // VoD chunk processing progress
        console.log('[WebRTC] VoD progress:', msg);
        this.emit('vodProgress', {
          totalChunks: msg.total_chunks,
          completedChunks: msg.completed_chunks,
          currentChunk: msg.current_chunk,
          percent: msg.percent,
        });
        break;

      case 'vod_complete':
        // VoD processing complete
        console.log('[WebRTC] VoD complete:', msg);
        this.emit('vodComplete', {
          transcriptAvailable: msg.transcript_available,
          durationSecs: msg.duration_secs,
          segmentCount: msg.segment_count,
        });
        break;

      default:
        console.log('[WebRTC] Unhandled message type:', msg.type, 'Full message:', msg);
        break;
    }
  }

  /**
   * Disconnect from server
   */
  disconnect() {
    this.intentionalClose = true;

    // Cancel any pending reconnection
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }

    // Clear ICE disconnect timer
    if (this.iceDisconnectTimer) {
      clearTimeout(this.iceDisconnectTimer);
      this.iceDisconnectTimer = null;
    }

    if (this.pc) {
      this.pc.close();
      this.pc = null;
    }

    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }

    if (this.audioElement) {
      this.audioElement.srcObject = null;
    }

    this.remoteStream = null;
    this.connected = false;
    this.reconnectAttempts = 0;

    this.emit('disconnect', { code: 1000, reason: 'User disconnect' });
  }

  /**
   * Get current playback time
   * @returns {number} Current time in seconds
   */
  get currentTime() {
    return this.audioElement?.currentTime || 0;
  }

  /**
   * Get total duration
   * @returns {number} Duration in seconds
   */
  get duration() {
    return this.audioElement?.duration || 0;
  }

  /**
   * Check if playing
   * @returns {boolean}
   */
  get playing() {
    return this.audioElement && !this.audioElement.paused;
  }

  /**
   * Play audio
   */
  async play() {
    if (this.audioElement) {
      try {
        await this.audioElement.play();
      } catch (e) {
        console.error('[WebRTC] Play error:', e);
        this.emit('error', e);
      }
    }
  }

  /**
   * Pause audio
   */
  pause() {
    if (this.audioElement) {
      this.audioElement.pause();
    }
  }

  /**
   * Set volume
   * @param {number} volume - Volume level (0.0 to 1.0)
   */
  setVolume(volume) {
    if (this.audioElement) {
      this.audioElement.volume = Math.max(0, Math.min(1, volume));
    }
  }

  /**
   * Get connection stats
   * @returns {Promise<Object>} Connection statistics
   */
  async getStats() {
    if (!this.pc) {
      return null;
    }

    const stats = await this.pc.getStats();
    const result = {
      bytesReceived: 0,
      packetsReceived: 0,
      packetsLost: 0,
      jitter: 0,
      roundTripTime: 0,
    };

    stats.forEach(report => {
      if (report.type === 'inbound-rtp' && report.kind === 'audio') {
        result.bytesReceived = report.bytesReceived || 0;
        result.packetsReceived = report.packetsReceived || 0;
        result.packetsLost = report.packetsLost || 0;
        result.jitter = report.jitter || 0;
      }
      if (report.type === 'candidate-pair' && report.state === 'succeeded') {
        result.roundTripTime = report.currentRoundTripTime || 0;
      }
    });

    return result;
  }

  /**
   * Debug function to log comprehensive connection status
   */
  async debugStatus() {
    console.log('=== WebRTC Debug Status ===');

    // Connection states
    if (this.pc) {
      console.log('ICE Connection State:', this.pc.iceConnectionState);
      console.log('ICE Gathering State:', this.pc.iceGatheringState);
      console.log('Connection State:', this.pc.connectionState);
      console.log('Signaling State:', this.pc.signalingState);
    } else {
      console.log('PeerConnection: null');
    }

    // Audio element status
    if (this.audioElement) {
      console.log('Audio Element:');
      console.log('  - srcObject:', this.audioElement.srcObject ? 'set' : 'null');
      console.log('  - paused:', this.audioElement.paused);
      console.log('  - muted:', this.audioElement.muted);
      console.log('  - volume:', this.audioElement.volume);
      console.log('  - readyState:', this.audioElement.readyState);
      console.log('  - networkState:', this.audioElement.networkState);
      console.log('  - error:', this.audioElement.error);
    } else {
      console.log('Audio Element: null');
    }

    // Remote stream status
    if (this.remoteStream) {
      console.log('Remote Stream:');
      console.log('  - active:', this.remoteStream.active);
      const tracks = this.remoteStream.getTracks();
      console.log('  - tracks:', tracks.length);
      tracks.forEach((track, i) => {
        console.log(`  - track[${i}]: kind=${track.kind}, enabled=${track.enabled}, muted=${track.muted}, readyState=${track.readyState}`);
      });
    } else {
      console.log('Remote Stream: null');
    }

    // WebRTC stats
    const stats = await this.getStats();
    if (stats) {
      console.log('WebRTC Stats:');
      console.log('  - packetsReceived:', stats.packetsReceived);
      console.log('  - bytesReceived:', stats.bytesReceived);
      console.log('  - packetsLost:', stats.packetsLost);
      console.log('  - jitter:', stats.jitter);
      console.log('  - roundTripTime:', stats.roundTripTime);
    }

    console.log('=== End Debug Status ===');
    return {
      iceConnectionState: this.pc?.iceConnectionState,
      connectionState: this.pc?.connectionState,
      audioElementSrcObject: !!this.audioElement?.srcObject,
      audioPaused: this.audioElement?.paused,
      remoteStreamActive: this.remoteStream?.active,
      stats
    };
  }

  /**
   * Start periodic stats logging (for debugging)
   * @param {number} intervalMs - Interval in milliseconds (default 5000)
   */
  startStatsLogging(intervalMs = 5000) {
    if (this._statsInterval) {
      clearInterval(this._statsInterval);
    }
    this._statsInterval = setInterval(async () => {
      const stats = await this.getStats();
      if (stats && stats.packetsReceived > 0) {
        console.log(`[WebRTC Stats] packets=${stats.packetsReceived}, bytes=${stats.bytesReceived}, lost=${stats.packetsLost}`);
      } else if (this.pc) {
        console.log(`[WebRTC Stats] No packets received. ICE: ${this.pc.iceConnectionState}, Conn: ${this.pc.connectionState}`);
      }
    }, intervalMs);
    console.log(`[WebRTC] Stats logging started (every ${intervalMs}ms)`);
  }

  /**
   * Stop periodic stats logging
   */
  stopStatsLogging() {
    if (this._statsInterval) {
      clearInterval(this._statsInterval);
      this._statsInterval = null;
      console.log('[WebRTC] Stats logging stopped');
    }
  }
}
