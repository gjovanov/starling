/**
 * Configuration for the transcription frontend
 *
 * Configuration is loaded dynamically from the server at /api/config
 * This allows the same frontend bundle to work with different backend deployments.
 *
 * Environment variables on the server control:
 *   - PUBLIC_IP: Server's public IP address
 *   - PORT: Server port (default: 8080)
 *   - TURN_SERVER: TURN server URL for NAT traversal
 *   - TURN_USERNAME: TURN authentication username
 *   - TURN_PASSWORD: TURN authentication password
 */

// Default configuration (used as fallback if API fetch fails)
const defaultConfig = {
  // WebSocket server URL (will be overridden by server config)
  wsUrl: `ws://${window.location.host}/ws`,

  // ICE servers for WebRTC (will be overridden by server config)
  iceServers: [
    { urls: 'stun:stun.l.google.com:19302' }
  ],

  // Audio settings
  audio: {
    sampleRate: 16000,
    channels: 1,
    bufferSize: 4096,
  },

  // Subtitle display settings
  subtitles: {
    maxSegments: 1000,
    autoScroll: true,
    showTimestamps: true,
  },

  // Speaker colors (up to 8 speakers)
  speakerColors: [
    '#4A90D9', // Blue
    '#50C878', // Green
    '#E9967A', // Salmon
    '#DDA0DD', // Plum
    '#F0E68C', // Khaki
    '#87CEEB', // Sky Blue
    '#FFB6C1', // Light Pink
    '#98FB98', // Pale Green
  ],

  // Reconnection settings
  reconnect: {
    enabled: true,
    delay: 2000,
    maxDelay: 30000,
    maxAttempts: 10,
  },
};

// Mutable config object that will be populated
let config = { ...defaultConfig };

/**
 * Load configuration from the server
 * @returns {Promise<Object>} The configuration object
 */
export async function loadConfig() {
  try {
    const response = await fetch('/api/config');
    if (!response.ok) {
      throw new Error(`Config fetch failed: ${response.status}`);
    }
    const serverConfig = await response.json();

    // Merge server config with defaults (server config takes precedence)
    config = {
      ...defaultConfig,
      ...serverConfig,
      audio: { ...defaultConfig.audio, ...serverConfig.audio },
      subtitles: { ...defaultConfig.subtitles, ...serverConfig.subtitles },
      reconnect: { ...defaultConfig.reconnect, ...serverConfig.reconnect },
    };

    console.log('[Config] Loaded from server:', config.wsUrl);
    return config;
  } catch (error) {
    console.warn('[Config] Failed to load from server, using defaults:', error.message);
    // Use default config with current host
    config = {
      ...defaultConfig,
      wsUrl: `ws://${window.location.host}/ws`,
    };
    return config;
  }
}

/**
 * Get the current configuration
 * @returns {Object} The configuration object
 */
export function getConfig() {
  return config;
}

// Export default config for backwards compatibility
export { config };
