/**
 * WebSocket client module for transcription server
 */
import { createEventEmitter } from './utils.js';

export class WebSocketClient {
  /**
   * @param {string} url - WebSocket server URL
   * @param {Object} options - Configuration options
   */
  constructor(url, options = {}) {
    this.url = url;
    this.options = {
      reconnect: true,
      reconnectDelay: 2000,
      maxReconnectDelay: 30000,
      maxReconnectAttempts: 10,
      ...options,
    };

    this.ws = null;
    this.reconnectAttempts = 0;
    this.reconnectTimeout = null;
    this.intentionalClose = false;

    // Event emitter
    const emitter = createEventEmitter();
    this.on = emitter.on.bind(emitter);
    this.off = emitter.off.bind(emitter);
    this.emit = emitter.emit.bind(emitter);
    this.removeAllListeners = emitter.removeAllListeners.bind(emitter);
  }

  /**
   * Connect to WebSocket server
   */
  connect() {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      console.warn('WebSocket already connected');
      return;
    }

    this.intentionalClose = false;

    try {
      this.ws = new WebSocket(this.url);

      this.ws.onopen = () => {
        console.log('WebSocket connected');
        this.reconnectAttempts = 0;
        this.emit('connect');
      };

      this.ws.onclose = (event) => {
        console.log('WebSocket closed:', event.code, event.reason);
        this.emit('disconnect', { code: event.code, reason: event.reason });

        if (!this.intentionalClose && this.options.reconnect) {
          this.scheduleReconnect();
        }
      };

      this.ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        this.emit('error', error);
      };

      this.ws.onmessage = (event) => {
        this.handleMessage(event.data);
      };
    } catch (error) {
      console.error('Failed to create WebSocket:', error);
      this.emit('error', error);
      if (this.options.reconnect) {
        this.scheduleReconnect();
      }
    }
  }

  /**
   * Disconnect from WebSocket server
   */
  disconnect() {
    this.intentionalClose = true;
    clearTimeout(this.reconnectTimeout);

    if (this.ws) {
      this.ws.close(1000, 'Client disconnect');
      this.ws = null;
    }
  }

  /**
   * Schedule reconnection attempt
   */
  scheduleReconnect() {
    if (this.options.maxReconnectAttempts > 0 &&
        this.reconnectAttempts >= this.options.maxReconnectAttempts) {
      console.error('Max reconnection attempts reached');
      this.emit('reconnectFailed');
      return;
    }

    // Exponential backoff
    const delay = Math.min(
      this.options.reconnectDelay * Math.pow(1.5, this.reconnectAttempts),
      this.options.maxReconnectDelay
    );

    console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts + 1})`);
    this.emit('reconnecting', { attempt: this.reconnectAttempts + 1, delay });

    this.reconnectTimeout = setTimeout(() => {
      this.reconnectAttempts++;
      this.connect();
    }, delay);
  }

  /**
   * Handle incoming WebSocket message
   * @param {string} data - Raw message data
   */
  handleMessage(data) {
    try {
      const message = JSON.parse(data);

      switch (message.type) {
        case 'welcome':
          this.emit('welcome', message);
          break;

        case 'audio':
          this.emit('audio', {
            data: message.data,
            timestamp: message.timestamp,
          });
          break;

        case 'subtitle':
          this.emit('subtitle', {
            text: message.text,
            growingText: message.growing_text,
            fullTranscript: message.full_transcript,
            delta: message.delta,
            tailChanged: message.tail_changed,
            speaker: message.speaker,
            start: message.start,
            end: message.end,
            isFinal: message.is_final,
            inferenceTimeMs: message.inference_time_ms,
          });
          break;

        case 'status':
          this.emit('status', {
            bufferTime: message.buffer_time,
            totalDuration: message.total_duration,
          });
          break;

        case 'end':
          this.emit('end', {
            totalDuration: message.total_duration,
          });
          break;

        case 'error':
          this.emit('serverError', { message: message.message });
          break;

        default:
          console.warn('Unknown message type:', message.type);
          this.emit('unknown', message);
      }
    } catch (error) {
      console.error('Failed to parse message:', error, data);
    }
  }

  /**
   * Check if connected
   * @returns {boolean}
   */
  get isConnected() {
    return this.ws && this.ws.readyState === WebSocket.OPEN;
  }

  /**
   * Get connection state
   * @returns {string} 'connecting' | 'connected' | 'disconnected' | 'reconnecting'
   */
  get state() {
    if (this.reconnectTimeout) return 'reconnecting';
    if (!this.ws) return 'disconnected';

    switch (this.ws.readyState) {
      case WebSocket.CONNECTING:
        return 'connecting';
      case WebSocket.OPEN:
        return 'connected';
      default:
        return 'disconnected';
    }
  }
}
