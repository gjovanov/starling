/**
 * WebRTC Uplink Client — client-initiated audio uplink for "Speakers" sessions.
 *
 * Unlike webrtc.js (which receives server-pushed audio), this module:
 *  1. Captures local audio via getDisplayMedia() or getUserMedia()
 *  2. Creates a PeerConnection and adds the captured track as outbound
 *  3. Sends the SDP *offer* to the server (server replies with an answer)
 *  4. Consumes subtitle messages from the same WebSocket
 *
 * Protocol additions on top of webrtc.js:
 *   Client → Server: { type: "ready", role: "uplink" }
 *   Client → Server: { type: "offer", sdp: "..." }
 *   Server → Client: { type: "answer", sdp: "..." }
 */
import { createEventEmitter } from './utils.js';

export class WebRTCUplinkClient {
  /**
   * @param {string} wsUrl - WebSocket URL for signaling
   * @param {Object} options
   * @param {Array}  [options.iceServers]          ICE servers from server config
   * @param {string} [options.iceTransportPolicy]  'all' or 'relay'
   * @param {MediaStream} options.stream           Local media stream with an audio track
   */
  constructor(wsUrl, options = {}) {
    this.wsUrl = wsUrl;

    const defaultIceServers = [{ urls: 'stun:stun.l.google.com:19302' }];

    this.options = {
      iceServers: options.iceServers || defaultIceServers,
      iceTransportPolicy: options.iceTransportPolicy || 'all',
      stream: options.stream || null,
      autoReconnect: options.autoReconnect !== false,
      reconnectDelay: options.reconnectDelay || 2000,
      maxReconnectDelay: options.maxReconnectDelay || 30000,
      maxReconnectAttempts: options.maxReconnectAttempts || 5,
      ...options,
    };

    this.ws = null;
    this.pc = null;
    this.stream = options.stream || null;
    this.connected = false;
    this.clientId = null;
    /** Set true once we've kicked off a single offer for this WS connection.
     * Some servers may emit more than one welcome — we only act on the first. */
    this._offerStarted = false;

    this.intentionalClose = false;
    this.reconnectAttempts = 0;
    this.reconnectTimeout = null;

    const emitter = createEventEmitter();
    this.on = emitter.on.bind(emitter);
    this.off = emitter.off.bind(emitter);
    this.emit = emitter.emit.bind(emitter);
  }

  /**
   * Connect to the server and start streaming the local track.
   */
  async connect() {
    console.log('[WebRTC-Uplink] connect() ws=', this.wsUrl);
    if (!this.stream || this.stream.getAudioTracks().length === 0) {
      throw new Error('WebRTCUplinkClient requires a MediaStream with at least one audio track');
    }

    this.intentionalClose = false;

    return new Promise((resolve, reject) => {
      this.ws = new WebSocket(this.wsUrl);

      this.ws.onopen = () => {
        console.log('[WebRTC-Uplink] Signaling connected');
        this.reconnectAttempts = 0;
        // Tell server we're an uplink client so it knows to expect an offer
        this.ws.send(JSON.stringify({ type: 'ready', role: 'uplink' }));
      };

      this.ws.onclose = (event) => {
        console.log('[WebRTC-Uplink] Signaling disconnected:', event.code, event.reason);
        this.connected = false;
        this.emit('disconnect', { code: event.code, reason: event.reason });
        if (!this.intentionalClose && this.options.autoReconnect) {
          this.scheduleReconnect();
        }
      };

      this.ws.onerror = (err) => {
        console.error('[WebRTC-Uplink] Signaling error:', err);
        this.emit('error', err);
        reject(err);
      };

      this.ws.onmessage = async (event) => {
        try {
          const msg = JSON.parse(event.data);
          await this.handleSignalingMessage(msg, resolve);
        } catch (e) {
          console.error('[WebRTC-Uplink] Message error:', e);
        }
      };
    });
  }

  /**
   * Set up the peer connection with our outbound audio track, then send an offer.
   * Called after the server acknowledges the ready message (welcome).
   */
  async startOffer() {
    const rtcConfig = {
      iceServers: this.options.iceServers,
      iceTransportPolicy: this.options.iceTransportPolicy || 'all',
    };
    console.log('[WebRTC-Uplink] RTCPeerConnection config:', rtcConfig);
    this.pc = new RTCPeerConnection(rtcConfig);

    // Add the local audio track
    const audioTracks = this.stream.getAudioTracks();
    audioTracks.forEach(track => {
      this.pc.addTrack(track, this.stream);
      console.log('[WebRTC-Uplink] Added local track:', track.label, track.kind);
    });

    // ICE candidate → server
    this.pc.onicecandidate = (event) => {
      if (event.candidate && this.ws?.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify({
          type: 'ice-candidate',
          candidate: event.candidate.toJSON(),
        }));
      }
    };

    this.pc.onconnectionstatechange = () => {
      const s = this.pc.connectionState;
      console.log('[WebRTC-Uplink] Connection state:', s);
      if (s === 'connected') {
        this.connected = true;
        this.emit('connected');
      } else if (s === 'failed' || s === 'disconnected') {
        this.connected = false;
        this.emit('connectionFailed');
      } else if (s === 'closed') {
        this.connected = false;
        this.emit('closed');
      }
    };

    this.pc.oniceconnectionstatechange = () => {
      console.log('[WebRTC-Uplink] ICE:', this.pc.iceConnectionState);
    };

    // Create and send offer
    const offer = await this.pc.createOffer({
      offerToReceiveAudio: false,
      offerToReceiveVideo: false,
    });
    await this.pc.setLocalDescription(offer);
    this.ws.send(JSON.stringify({ type: 'offer', sdp: offer.sdp }));
    console.log('[WebRTC-Uplink] Offer sent');
  }

  async handleSignalingMessage(msg, resolve) {
    switch (msg.type) {
      case 'welcome':
        this.clientId = msg.client_id;
        this.emit('welcome', msg);
        // Server is ready — kick off our offer exactly once per WS connection.
        // If the server emits more than one welcome (back-compat), ignore the rest.
        if (this._offerStarted) {
          console.log('[WebRTC-Uplink] Ignoring duplicate welcome');
          break;
        }
        this._offerStarted = true;
        try {
          await this.startOffer();
        } catch (e) {
          this._offerStarted = false; // allow retry on the next welcome
          console.error('[WebRTC-Uplink] Failed to start offer:', e);
          this.emit('error', e);
        }
        break;

      case 'answer':
        if (this.pc) {
          try {
            await this.pc.setRemoteDescription({ type: 'answer', sdp: msg.sdp });
            console.log('[WebRTC-Uplink] SDP answer set');
            resolve && resolve();
          } catch (e) {
            console.error('[WebRTC-Uplink] setRemoteDescription failed:', e);
            this.emit('error', e);
          }
        }
        break;

      case 'ice-candidate':
        if (this.pc && msg.candidate) {
          try {
            const c = new RTCIceCandidate({
              candidate: msg.candidate.candidate,
              sdpMid: msg.candidate.sdpMid?.toString() || '0',
              sdpMLineIndex: msg.candidate.sdpMLineIndex ?? 0,
            });
            await this.pc.addIceCandidate(c);
          } catch (e) {
            console.warn('[WebRTC-Uplink] addIceCandidate failed:', e);
          }
        }
        break;

      case 'subtitle':
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
        this.emit('end', { totalDuration: msg.total_duration, is_live: msg.is_live });
        break;

      case 'error':
        console.error('[WebRTC-Uplink] Server error:', msg.message);
        this.emit('serverError', { message: msg.message });
        break;

      case 'ping':
        // Keep-alive; ignore
        break;

      default:
        console.log('[WebRTC-Uplink] Unhandled message:', msg.type, msg);
    }
  }

  scheduleReconnect() {
    if (this.options.maxReconnectAttempts > 0 &&
        this.reconnectAttempts >= this.options.maxReconnectAttempts) {
      console.error('[WebRTC-Uplink] Max reconnect attempts reached');
      this.emit('reconnectFailed');
      return;
    }

    const delay = Math.min(
      this.options.reconnectDelay * Math.pow(1.5, this.reconnectAttempts),
      this.options.maxReconnectDelay
    );

    console.log(`[WebRTC-Uplink] Reconnecting in ${delay}ms`);
    this.emit('reconnecting', { attempt: this.reconnectAttempts + 1, delay });

    this.reconnectTimeout = setTimeout(() => {
      this.reconnectAttempts++;
      this.reconnect();
    }, delay);
  }

  async reconnect() {
    if (this.pc) { this.pc.close(); this.pc = null; }
    if (this.ws) {
      this.intentionalClose = true;
      this.ws.close();
      this.ws = null;
      this.intentionalClose = false;
    }
    // Reset per-connection state so the new welcome triggers a fresh offer
    this._offerStarted = false;
    try {
      await this.connect();
      console.log('[WebRTC-Uplink] Reconnected');
    } catch (e) {
      console.error('[WebRTC-Uplink] Reconnect failed:', e);
      this.scheduleReconnect();
    }
  }

  /**
   * Stop capturing and disconnect. Also stops the underlying stream tracks.
   */
  disconnect() {
    this.intentionalClose = true;
    if (this.reconnectTimeout) { clearTimeout(this.reconnectTimeout); this.reconnectTimeout = null; }

    if (this.pc) { this.pc.close(); this.pc = null; }
    if (this.ws) { this.ws.close(); this.ws = null; }

    if (this.stream) {
      this.stream.getTracks().forEach(t => t.stop());
      this.stream = null;
    }

    this.connected = false;
    this.reconnectAttempts = 0;
    this.emit('disconnect', { code: 1000, reason: 'User disconnect' });
  }

  async getStats() {
    if (!this.pc) return null;
    const stats = await this.pc.getStats();
    const out = {
      bytesSent: 0, packetsSent: 0, jitter: 0, roundTripTime: 0,
    };
    stats.forEach(report => {
      if (report.type === 'outbound-rtp' && report.kind === 'audio') {
        out.bytesSent = report.bytesSent || 0;
        out.packetsSent = report.packetsSent || 0;
      }
      if (report.type === 'candidate-pair' && report.state === 'succeeded') {
        out.roundTripTime = report.currentRoundTripTime || 0;
      }
      if (report.type === 'remote-inbound-rtp' && report.kind === 'audio') {
        out.jitter = report.jitter || 0;
      }
    });
    return out;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Audio source helpers

/**
 * Enumerate audio input devices (requires microphone permission for labels).
 * Returns [{ deviceId, label, kind }].
 */
export async function listAudioInputDevices() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
    return [];
  }
  const devices = await navigator.mediaDevices.enumerateDevices();
  return devices
    .filter(d => d.kind === 'audioinput')
    .map(d => ({
      deviceId: d.deviceId,
      label: d.label || 'Unnamed device',
      groupId: d.groupId,
    }));
}

/**
 * Capture a stream based on the selected source.
 *
 *  sourceId === 'display'  → getDisplayMedia, audio-only
 *  sourceId === <deviceId> → getUserMedia with that input device
 */
export async function captureSpeakersStream(sourceId) {
  if (!navigator.mediaDevices) {
    throw new Error('MediaDevices API is not available (use HTTPS or localhost)');
  }

  if (sourceId === 'display' || !sourceId) {
    if (!navigator.mediaDevices.getDisplayMedia) {
      throw new Error(
        'This browser does not support screen-audio capture (getDisplayMedia). ' +
        'Use Chrome, Edge, or Firefox on a desktop OS.'
      );
    }
    // Some browsers require video:true for the audio checkbox to appear in the picker.
    // Request video, then stop it immediately so we only stream audio uplink.
    let stream;
    try {
      stream = await navigator.mediaDevices.getDisplayMedia({
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
        },
        video: true,
      });
    } catch (e) {
      // The user cancelled the picker, or permission was denied.
      const name = e && e.name ? e.name : 'Error';
      if (name === 'NotAllowedError' || name === 'PermissionDeniedError') {
        throw new Error('Screen-share permission denied. Re-try and accept the dialog.');
      }
      throw e;
    }
    // Strip video tracks; we only need audio for transcription
    stream.getVideoTracks().forEach(t => { try { t.stop(); } catch (_) {} stream.removeTrack(t); });
    if (stream.getAudioTracks().length === 0) {
      stream.getTracks().forEach(t => { try { t.stop(); } catch (_) {} });
      throw new Error(
        'No audio in the captured stream. Re-try and tick the "Share audio" checkbox in ' +
        'the picker. On Chrome/Windows, choose "Entire screen" or "Chrome tab"; ' +
        '"Window" capture does not include audio.'
      );
    }
    return stream;
  }

  return navigator.mediaDevices.getUserMedia({
    audio: {
      deviceId: { exact: sourceId },
      echoCancellation: false,
      noiseSuppression: false,
      autoGainControl: false,
    },
    video: false,
  });
}
