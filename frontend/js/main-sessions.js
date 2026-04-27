/**
 * Main entry point for multi-session mode
 */
import { loadConfig, getConfig } from './config.js';
import { WebRTCClient } from './modules/webrtc.js';
import { WebRTCUplinkClient, listAudioInputDevices, captureSpeakersStream } from './modules/webrtc-uplink.js';
import { SubtitleRenderer } from './modules/subtitles.js';
import { formatTime } from './modules/utils.js';
import { SessionManager, formatDuration, formatFileSize } from './modules/session-manager.js';
import { initWasmUI } from './wasm/wasm-ui.js';
import {
  fetchVoices as ttsFetchVoices,
  fetchTtsConfig,
  fetchTtsStatus,
  fetchSavedFiles as ttsFetchSavedFiles,
  deleteSavedFile as ttsDeleteSavedFile,
  synthesizeAndSave as ttsSynthesizeAndSave,
  synthesizeForPlayback as ttsSynthesizeForPlayback,
  uploadVoiceRef as ttsUploadVoiceRef,
  deleteVoiceRef as ttsDeleteVoiceRef,
} from './modules/tts-manager.js';
import { TtsPlayer, exposePlayerForTests } from './modules/tts-player.js';

// Application state
const state = {
  connected: false,
  totalDuration: 0,
  selectedSessionId: null,
  subtitleCount: 0,  // Counter for received subtitles
  sourceType: 'media', // 'media' or 'srt'
};


// Map slider value (1-5) to actual silence_energy_threshold
function getSilenceEnergyThreshold(sliderValue) {
  const values = [0.003, 0.005, 0.008, 0.012, 0.02];
  return values[sliderValue - 1];
}

// Modules
let webrtcClient = null;       // Receiving (media/SRT sessions)
let uplinkClient = null;       // Sending (speakers sessions)
let subtitleRenderer = null;
let sessionManager = null;

// DOM elements
const elements = {};

/**
 * Initialize the application
 */
async function init() {
  // Load configuration from server
  const config = await loadConfig();

  // Cache DOM elements
  cacheElements();

  // Initialize session manager
  sessionManager = new SessionManager();
  setupSessionManagerListeners();

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

  // Set up UI event listeners
  setupEventListeners();

  // Populate FAB defaults from config
  if (elements.fabEnabledSelect && config.fabEnabled !== undefined) {
    const defaultLabel = config.fabEnabled ? 'Default (enabled)' : 'Default (disabled)';
    elements.fabEnabledSelect.options[0].textContent = defaultLabel;
  }
  if (elements.fabUrlInput && config.fabUrl) {
    elements.fabUrlInput.value = config.fabUrl;
    elements.fabUrlInput.placeholder = config.fabUrl || 'No server default';
  }
  if (elements.fabSendTypeSelect && config.fabSendType) {
    const defaultLabel = `Default (${config.fabSendType})`;
    elements.fabSendTypeSelect.options[0].textContent = defaultLabel;
  }

  // Load initial data
  await Promise.all([
    sessionManager.fetchModels(),
    sessionManager.fetchMedia(),
    sessionManager.fetchModes(),
    sessionManager.fetchNoiseCancellation(),
    sessionManager.fetchDiarization(),
    sessionManager.fetchSrtStreams(),
    sessionManager.fetchSessions(),
  ]);

  // Start polling for session updates
  sessionManager.startPolling(3000);

  // Initialize WASM UI (browser-side inference)
  const wasmContainer = document.getElementById('wasm-mode-container');
  if (wasmContainer) {
    wasmContainer.style.display = 'block';
    await initWasmUI(wasmContainer);
    console.log('WASM UI initialized');
  }

  // Initialize the TTS tab. Failures here must NOT block the rest of the
  // app — the TTS server is optional and may not be running.
  try {
    await setupTtsTab();
  } catch (err) {
    console.warn('TTS tab init failed (server may be down):', err);
  }

  console.log('Multi-session application initialized');
}

function cacheElements() {
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
  elements.audioPlayer = document.getElementById('audio-player');
  elements.latencyInfo = document.getElementById('latency-info');

  // Audio-only toggle
  elements.withoutTranscriptionToggle = document.getElementById('without-transcription-toggle');
  elements.modelSelectGroup = document.getElementById('model-select-group');

  // Session panel elements
  elements.sessionTabs = document.querySelectorAll('.session-tab');
  elements.sessionContents = document.querySelectorAll('.session-content');
  elements.sessionsList = document.getElementById('sessions-list');
  elements.modelSelect = document.getElementById('model-select');
  elements.mediaSelect = document.getElementById('media-select');
  elements.modeSelect = document.getElementById('mode-select');
  elements.languageSelect = document.getElementById('language-select');
  elements.createSessionBtn = document.getElementById('create-session-btn');
  elements.uploadZone = document.getElementById('upload-zone');
  elements.fileInput = document.getElementById('file-input');
  elements.mediaList = document.getElementById('media-list');

  // Quantization select
  elements.quantSelect = document.getElementById('quant-select');

  // Noise cancellation and diarization elements
  elements.noiseSelect = document.getElementById('noise-select');
  elements.diarizationSelect = document.getElementById('diarization-select');
  elements.sentenceCompletionSelect = document.getElementById('sentence-completion-select');

  // Pause-segmented config elements (all pause params consolidated here)
  elements.pauseSegmentedConfig = document.getElementById('pause-segmented-config');
  elements.pauseThreshold = document.getElementById('pause-threshold');
  elements.silenceEnergy = document.getElementById('silence-energy');
  elements.maxSegment = document.getElementById('max-segment');
  elements.pauseThresholdValue = document.getElementById('pause-threshold-value');
  elements.silenceEnergyValue = document.getElementById('silence-energy-value');
  elements.maxSegmentValue = document.getElementById('max-segment-value');
  elements.psContextSegments = document.getElementById('ps-context-segments');
  elements.psContextSegmentsValue = document.getElementById('ps-context-segments-value');
  elements.psMinSegment = document.getElementById('ps-min-segment');
  elements.psMinSegmentValue = document.getElementById('ps-min-segment-value');
  elements.psPartialInterval = document.getElementById('ps-partial-interval');
  elements.psPartialIntervalValue = document.getElementById('ps-partial-interval-value');

  // Growing segments config elements
  elements.growingSegmentsConfig = document.getElementById('growing-segments-config');
  elements.gsBufferSize = document.getElementById('gs-buffer-size');
  elements.gsBufferSizeValue = document.getElementById('gs-buffer-size-value');
  elements.gsProcessInterval = document.getElementById('gs-process-interval');
  elements.gsProcessIntervalValue = document.getElementById('gs-process-interval-value');
  elements.gsPauseThreshold = document.getElementById('gs-pause-threshold');
  elements.gsPauseThresholdValue = document.getElementById('gs-pause-threshold-value');
  elements.gsSilenceSensitivity = document.getElementById('gs-silence-sensitivity');
  elements.gsSilenceSensitivityValue = document.getElementById('gs-silence-sensitivity-value');

  // Growing segments advanced tuning elements
  elements.gsEmitFullText = document.getElementById('gs-emit-full-text');
  elements.gsMinStableCount = document.getElementById('gs-min-stable-count');
  elements.gsMinStableCountValue = document.getElementById('gs-min-stable-count-value');
  elements.gsEchoDedupThreshold = document.getElementById('gs-echo-dedup-threshold');
  elements.gsEchoDedupThresholdValue = document.getElementById('gs-echo-dedup-threshold-value');
  elements.gsEchoDedupWindow = document.getElementById('gs-echo-dedup-window');
  elements.gsEchoDedupWindowValue = document.getElementById('gs-echo-dedup-window-value');
  elements.gsMinFinalWords = document.getElementById('gs-min-final-words');
  elements.gsMinFinalWordsValue = document.getElementById('gs-min-final-words-value');
  elements.gsPromotionEnabled = document.getElementById('gs-promotion-enabled');
  elements.gsPromotionMinWords = document.getElementById('gs-promotion-min-words');
  elements.gsPromotionMinWordsValue = document.getElementById('gs-promotion-min-words-value');
  // FAB config elements
  elements.fabEnabledSelect = document.getElementById('fab-enabled-select');
  elements.fabUrlGroup = document.getElementById('fab-url-group');
  elements.fabUrlInput = document.getElementById('fab-url-input');
  elements.fabSendTypeGroup = document.getElementById('fab-send-type-group');
  elements.fabSendTypeSelect = document.getElementById('fab-send-type-select');

  // Source type tabs (Media Files / Live Streams / Speakers)
  elements.sourceTabs = document.querySelectorAll('.source-tab');
  elements.mediaSourceContent = document.getElementById('media-source-content');
  elements.srtSourceContent = document.getElementById('srt-source-content');
  elements.srtSelect = document.getElementById('srt-select');
  elements.speakersSourceContent = document.getElementById('speakers-source-content');
  elements.speakersSelect = document.getElementById('speakers-select');
  elements.speakersRefreshBtn = document.getElementById('speakers-refresh-btn');
  elements.speakersPermissionHint = document.getElementById('speakers-permission-hint');
  elements.speakersMethodRadios = document.querySelectorAll('input[name="speakers-method"]');
  elements.speakersDeviceRow = document.getElementById('speakers-device-row');
  elements.speakersTestBtn = document.getElementById('speakers-test-btn');
  elements.speakersTestStatus = document.getElementById('speakers-test-status');
}

function setupSessionManagerListeners() {
  sessionManager.on('modelsLoaded', renderModelSelect);
  sessionManager.on('mediaLoaded', () => {
    renderMediaSelect();
    renderMediaList();
  });
  sessionManager.on('modesLoaded', renderModeSelect);
  sessionManager.on('noiseCancellationLoaded', renderNoiseSelect);
  sessionManager.on('diarizationLoaded', renderDiarizationSelect);
  sessionManager.on('srtStreamsLoaded', ({ streams, configured }) => {
    renderSrtSelect(streams, configured);
  });
  sessionManager.on('sessionsUpdated', renderSessionsList);
  sessionManager.on('sessionCreated', (session) => {
    selectSession(session.id);
    showTab('sessions');
  });
  sessionManager.on('mediaUploaded', () => {
    elements.bufferInfo.textContent = 'File uploaded!';
  });
}

function setupEventListeners() {
  // Tab switching
  elements.sessionTabs.forEach(tab => {
    tab.addEventListener('click', () => {
      showTab(tab.dataset.tab);
    });
  });

  // Source type tab switching (Media Files / Live Streams / Speakers)
  if (elements.sourceTabs) {
    elements.sourceTabs.forEach(tab => {
      tab.addEventListener('click', () => {
        const sourceType = tab.dataset.source;
        state.sourceType = sourceType;

        // Update tab active state
        elements.sourceTabs.forEach(t => t.classList.toggle('active', t.dataset.source === sourceType));

        // Show/hide source content
        if (elements.mediaSourceContent) {
          elements.mediaSourceContent.style.display = sourceType === 'media' ? 'block' : 'none';
        }
        if (elements.srtSourceContent) {
          elements.srtSourceContent.style.display = sourceType === 'srt' ? 'block' : 'none';
        }
        if (elements.speakersSourceContent) {
          elements.speakersSourceContent.style.display = sourceType === 'speakers' ? 'flex' : 'none';
        }

        if (sourceType === 'speakers') {
          // Sync the device row visibility against the current radio state
          const method = getSpeakersMethod();
          if (elements.speakersDeviceRow) {
            elements.speakersDeviceRow.style.display = method === 'device' ? 'flex' : 'none';
          }
          if (method === 'device') {
            refreshAudioInputDevices();
          }
        }
      });
    });
  }

  // Speakers: method radio buttons toggle the device row
  if (elements.speakersMethodRadios && elements.speakersMethodRadios.length) {
    elements.speakersMethodRadios.forEach(radio => {
      radio.addEventListener('change', () => {
        const method = getSpeakersMethod();
        if (elements.speakersDeviceRow) {
          elements.speakersDeviceRow.style.display = method === 'device' ? 'flex' : 'none';
        }
        if (method === 'device') {
          refreshAudioInputDevices();
        }
      });
    });
  }

  // Speakers: refresh device list
  if (elements.speakersRefreshBtn) {
    elements.speakersRefreshBtn.addEventListener('click', async () => {
      // Request a transient mic permission so device labels are populated,
      // then enumerate.
      try {
        const tmp = await navigator.mediaDevices.getUserMedia({ audio: true });
        tmp.getTracks().forEach(t => t.stop());
        if (elements.speakersPermissionHint) {
          elements.speakersPermissionHint.style.display = 'none';
        }
      } catch (e) {
        console.warn('[Speakers] Mic permission denied or failed:', e);
        if (elements.speakersPermissionHint) {
          elements.speakersPermissionHint.style.display = 'block';
        }
      }
      await refreshAudioInputDevices();
    });
  }

  // Speakers: "Test capture" verifies the picker without creating a session
  if (elements.speakersTestBtn) {
    elements.speakersTestBtn.addEventListener('click', async () => {
      await testSpeakersCapture();
    });
  }

  // Audio-only toggle - hide/show transcription-related controls
  if (elements.withoutTranscriptionToggle) {
    elements.withoutTranscriptionToggle.addEventListener('change', () => {
      const audioOnly = elements.withoutTranscriptionToggle.checked;
      const displayVal = audioOnly ? 'none' : '';

      // Hide/show transcription-related form groups
      const transcriptionElements = [
        elements.modelSelectGroup,
        elements.modeSelect?.closest('.form-group'),
        elements.languageSelect?.closest('.form-group'),
        elements.noiseSelect?.closest('.form-group'),
        elements.diarizationSelect?.closest('.form-group'),
        elements.sentenceCompletionSelect?.closest('.form-group'),
        elements.fabEnabledSelect?.closest('.form-group'),
        elements.fabUrlGroup,
        elements.fabSendTypeGroup,
        elements.pauseSegmentedConfig,
        elements.growingSegmentsConfig,
      ];

      for (const el of transcriptionElements) {
        if (el) el.style.display = displayVal;
      }
    });
  }

  // Model select - update language dropdown when model changes
  if (elements.modelSelect) {
    elements.modelSelect.addEventListener('change', () => {
      renderLanguageSelect();
    });
  }

  // Mode select - show/hide mode-specific config panels
  if (elements.modeSelect) {
    elements.modeSelect.addEventListener('change', () => {
      const mode = elements.modeSelect.value;

      // Show/hide pause-segmented config (all pause params in one panel)
      if (elements.pauseSegmentedConfig) {
        elements.pauseSegmentedConfig.style.display = mode === 'pause_segmented' ? 'block' : 'none';
      }

      // Show/hide growing segments config
      if (elements.growingSegmentsConfig) {
        elements.growingSegmentsConfig.style.display = mode === 'growing_segments' ? 'block' : 'none';
      }
    });
  }

  // Pause config sliders - update displayed values
  if (elements.pauseThreshold) {
    elements.pauseThreshold.addEventListener('input', () => {
      elements.pauseThresholdValue.textContent = elements.pauseThreshold.value;
    });
  }
  if (elements.silenceEnergy) {
    elements.silenceEnergy.addEventListener('input', () => {
      const labels = ['Very High', 'High', 'Medium', 'Low', 'Very Low'];
      elements.silenceEnergyValue.textContent = labels[elements.silenceEnergy.value - 1];
    });
  }
  if (elements.maxSegment) {
    elements.maxSegment.addEventListener('input', () => {
      elements.maxSegmentValue.textContent = elements.maxSegment.value;
    });
  }
  // Growing segments sliders - update displayed values
  if (elements.gsBufferSize) {
    elements.gsBufferSize.addEventListener('input', () => {
      elements.gsBufferSizeValue.textContent = elements.gsBufferSize.value;
    });
  }
  if (elements.gsProcessInterval) {
    elements.gsProcessInterval.addEventListener('input', () => {
      elements.gsProcessIntervalValue.textContent = elements.gsProcessInterval.value;
    });
  }
  if (elements.gsPauseThreshold) {
    elements.gsPauseThreshold.addEventListener('input', () => {
      elements.gsPauseThresholdValue.textContent = elements.gsPauseThreshold.value;
    });
  }
  if (elements.gsSilenceSensitivity) {
    elements.gsSilenceSensitivity.addEventListener('input', () => {
      const labels = ['Very High', 'High', 'Medium', 'Low', 'Very Low'];
      elements.gsSilenceSensitivityValue.textContent = labels[elements.gsSilenceSensitivity.value - 1];
    });
  }

  // Growing segments advanced sliders
  if (elements.gsMinStableCount) {
    elements.gsMinStableCount.addEventListener('input', () => {
      elements.gsMinStableCountValue.textContent = elements.gsMinStableCount.value;
    });
  }
  if (elements.gsEchoDedupThreshold) {
    elements.gsEchoDedupThreshold.addEventListener('input', () => {
      elements.gsEchoDedupThresholdValue.textContent = parseFloat(elements.gsEchoDedupThreshold.value).toFixed(2);
    });
  }
  if (elements.gsEchoDedupWindow) {
    elements.gsEchoDedupWindow.addEventListener('input', () => {
      elements.gsEchoDedupWindowValue.textContent = elements.gsEchoDedupWindow.value;
    });
  }
  if (elements.gsMinFinalWords) {
    elements.gsMinFinalWords.addEventListener('input', () => {
      elements.gsMinFinalWordsValue.textContent = elements.gsMinFinalWords.value;
    });
  }
  if (elements.gsPromotionMinWords) {
    elements.gsPromotionMinWords.addEventListener('input', () => {
      elements.gsPromotionMinWordsValue.textContent = elements.gsPromotionMinWords.value;
    });
  }

  // Pause-segmented sliders
  if (elements.psContextSegments) {
    elements.psContextSegments.addEventListener('input', () => {
      elements.psContextSegmentsValue.textContent = elements.psContextSegments.value;
    });
  }
  if (elements.psMinSegment) {
    elements.psMinSegment.addEventListener('input', () => {
      elements.psMinSegmentValue.textContent = parseFloat(elements.psMinSegment.value).toFixed(1);
    });
  }
  if (elements.psPartialInterval) {
    elements.psPartialInterval.addEventListener('input', () => {
      elements.psPartialIntervalValue.textContent = parseFloat(elements.psPartialInterval.value).toFixed(1);
    });
  }

  // FAB select - show/hide URL input and send type
  if (elements.fabEnabledSelect) {
    elements.fabEnabledSelect.addEventListener('change', () => {
      const val = elements.fabEnabledSelect.value;
      const show = val === 'enabled';
      if (elements.fabUrlGroup) {
        elements.fabUrlGroup.style.display = show ? 'flex' : 'none';
      }
      if (elements.fabSendTypeGroup) {
        elements.fabSendTypeGroup.style.display = show ? 'flex' : 'none';
      }
    });
  }

  // Create session
  elements.createSessionBtn.addEventListener('click', async () => {
    const withoutTranscription = elements.withoutTranscriptionToggle?.checked || false;
    const modelId = withoutTranscription ? null : elements.modelSelect.value;
    const mode = elements.modeSelect?.value || 'speedy';
    const language = elements.languageSelect?.value || 'de';
    const noiseCancellation = elements.noiseSelect?.value || 'none';
    const diarization = elements.diarizationSelect?.value !== 'none';
    const sentenceCompletion = elements.sentenceCompletionSelect?.value || 'minimal';
    const quant = elements.quantSelect?.value || 'q4';

    // Determine source based on active tab
    let mediaId = null;
    let srtChannelId = null;
    let sourceOverride = null;

    if (state.sourceType === 'srt') {
      const srtValue = elements.srtSelect?.value;
      if (!srtValue || srtValue === '') {
        alert('Please select an SRT stream');
        return;
      }
      srtChannelId = parseInt(srtValue, 10);
      sourceOverride = 'srt';
    } else if (state.sourceType === 'speakers') {
      sourceOverride = 'speakers';
    } else {
      mediaId = elements.mediaSelect.value;
      if (!mediaId) {
        alert('Please select a media file');
        return;
      }
      sourceOverride = 'media';
    }

    if (!withoutTranscription && !modelId) {
      alert('Please select a model');
      return;
    }

    // Get FAB config
    const fabEnabled = elements.fabEnabledSelect?.value || 'default';
    const fabUrl = elements.fabUrlInput?.value?.trim() || '';
    const fabSendType = elements.fabSendTypeSelect?.value || 'default';

    // Get pause config for pause_segmented mode
    let pauseConfig = null;
    if (mode === 'pause_segmented' && elements.pauseThreshold) {
      pauseConfig = {
        pause_threshold_ms: parseInt(elements.pauseThreshold.value, 10),
        silence_energy_threshold: getSilenceEnergyThreshold(parseInt(elements.silenceEnergy?.value || 3, 10)),
        max_segment_secs: parseFloat(elements.maxSegment?.value || 15),
        context_segments: elements.psContextSegments ? parseInt(elements.psContextSegments.value, 10) : 1,
        min_segment_secs: elements.psMinSegment ? parseFloat(elements.psMinSegment.value) : undefined,
        partial_interval_secs: elements.psPartialInterval ? parseFloat(elements.psPartialInterval.value) : undefined,
      };
    }

    // Get growing segments config
    let growingSegmentsConfig = null;
    if (mode === 'growing_segments' && elements.gsBufferSize) {
      growingSegmentsConfig = {
        buffer_size_secs: parseFloat(elements.gsBufferSize.value),
        process_interval_secs: parseFloat(elements.gsProcessInterval.value),
        pause_threshold_ms: parseInt(elements.gsPauseThreshold.value, 10),
        silence_energy_threshold: getSilenceEnergyThreshold(parseInt(elements.gsSilenceSensitivity?.value || 3, 10)),
        emit_full_text: elements.gsEmitFullText?.value === 'true' ? true : undefined,
        min_stable_count: elements.gsMinStableCount ? parseInt(elements.gsMinStableCount.value, 10) : undefined,
        echo_dedup_threshold: elements.gsEchoDedupThreshold ? parseFloat(elements.gsEchoDedupThreshold.value) : undefined,
        echo_dedup_window: elements.gsEchoDedupWindow ? parseInt(elements.gsEchoDedupWindow.value, 10) : undefined,
        min_final_words: elements.gsMinFinalWords ? parseInt(elements.gsMinFinalWords.value, 10) : undefined,
        promotion_enabled: elements.gsPromotionEnabled?.value === 'false' ? false : undefined,
        promotion_min_words: elements.gsPromotionMinWords ? parseInt(elements.gsPromotionMinWords.value, 10) : undefined,
      };
    }

    try {
      elements.createSessionBtn.disabled = true;
      elements.createSessionBtn.textContent = 'Creating...';
      const session = await sessionManager.createSession(modelId, {
        mediaId,
        srtChannelId,
        source: sourceOverride,
        mode,
        language,
        noiseCancellation,
        diarization,
        pauseConfig,
        growingSegmentsConfig,
        sentenceCompletion,
        fabEnabled,
        fabUrl,
        fabSendType,
        withoutTranscription,
        quant
      });
      // Auto-start the session
      await sessionManager.startSession(session.id);

      // For speakers sessions: immediately capture local audio and connect uplink.
      // The user already clicked a button, so the gesture requirement for
      // getDisplayMedia / getUserMedia is satisfied.
      if (sourceOverride === 'speakers') {
        await startSpeakersUplink(session.id);
      }
    } catch (e) {
      alert('Failed to create session: ' + e.message);
    } finally {
      elements.createSessionBtn.disabled = false;
      elements.createSessionBtn.textContent = 'Create Session';
    }
  });

  // File upload
  elements.uploadZone.addEventListener('click', () => {
    elements.fileInput.click();
  });

  elements.uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    elements.uploadZone.classList.add('dragover');
  });

  elements.uploadZone.addEventListener('dragleave', () => {
    elements.uploadZone.classList.remove('dragover');
  });

  elements.uploadZone.addEventListener('drop', async (e) => {
    e.preventDefault();
    elements.uploadZone.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file) await uploadFile(file);
  });

  elements.fileInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (file) await uploadFile(file);
    e.target.value = '';
  });

  // Connect button
  elements.connectBtn.addEventListener('click', () => {
    console.log('[Click] Connect button clicked, connected:', state.connected, 'selectedSessionId:', state.selectedSessionId);
    if (state.connected) {
      disconnect();
    } else if (state.selectedSessionId) {
      // For speakers sessions, we need to re-capture the stream (can't be joined like a media session)
      const sess = sessionManager.sessions.find(s => s.id === state.selectedSessionId);
      if (sess && sess.source_type === 'speakers') {
        startSpeakersUplink(state.selectedSessionId).catch(e => {
          console.error('[Speakers] join failed:', e);
          alert('Failed to start speakers uplink: ' + e.message);
        });
      } else {
        connect(state.selectedSessionId);
      }
    } else {
      console.log('[Click] No session selected!');
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

  // Export buttons
  elements.exportBtn.addEventListener('click', () => {
    elements.exportModal.style.display = 'flex';
  });

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

// ─── TTS tab ────────────────────────────────────────────────────────────

const ttsState = {
  voices: [],
  config: null,
  loaded: false,
  inFlight: false,
  /** @type {TtsPlayer|null} */
  player: null,
};

async function setupTtsTab() {
  const els = {
    text:        document.getElementById('tts-text'),
    charCount:   document.getElementById('tts-char-count'),
    charMax:     document.getElementById('tts-char-max'),
    voice:       document.getElementById('tts-voice'),
    modeRadios:  document.querySelectorAll('input[name="tts-output-mode"]'),
    filenameGrp: document.getElementById('tts-filename-group'),
    filename:    document.getElementById('tts-filename'),
    overwrite:   document.getElementById('tts-overwrite'),
    outputDir:   document.getElementById('tts-output-dir'),
    generate:    document.getElementById('tts-generate-btn'),
    pause:       document.getElementById('tts-pause-btn'),
    resume:      document.getElementById('tts-resume-btn'),
    stop:        document.getElementById('tts-stop-btn'),
    status:      document.getElementById('tts-status'),
    audio:       document.getElementById('tts-audio'),
    savedGroup:  document.getElementById('tts-saved-files-group'),
    savedList:   document.getElementById('tts-saved-files'),
    engineBadge: document.getElementById('tts-engine-badge'),
  };
  if (!els.text || !els.voice || !els.generate) return; // tab not present in DOM

  // Fetch the catalog + static config in parallel; either failing leaves
  // the tab in a clearly-disabled state.
  const [voices, cfg] = await Promise.all([ttsFetchVoices(), fetchTtsConfig()]);
  ttsState.voices = voices;
  ttsState.config = cfg;
  renderVoiceSelect(els);
  els.voice.disabled = false;
  els.generate.disabled = false;

  els.charMax.textContent = cfg.max_chars;
  els.text.maxLength = cfg.max_chars;
  els.outputDir.textContent = cfg.output_dir;
  ttsState.loaded = true;

  // Live char-count
  els.text.addEventListener('input', () => {
    els.charCount.textContent = els.text.value.length;
  });

  // Mode toggle: show filename input only when "save" is selected
  for (const r of els.modeRadios) {
    r.addEventListener('change', () => {
      const mode = document.querySelector('input[name="tts-output-mode"]:checked')?.value;
      els.filenameGrp.style.display = mode === 'save' ? '' : 'none';
    });
  }

  els.generate.addEventListener('click', () => generateTts(els));
  els.pause.addEventListener('click', () => ttsState.player?.pause());
  els.resume.addEventListener('click', () => ttsState.player?.resume());
  els.stop.addEventListener('click', () => {
    ttsState.player?.cancel();
    setStreamingButtons(els, 'idle');
    els.status.textContent = 'Stopped.';
    els.status.style.color = '#a0a0a0';
  });

  // Saved-files list (populated on first render + after each save)
  await refreshSavedFiles(els);

  // Lifecycle status badge — poll while the tab is active.
  await refreshTtsEngineStatus(els);
  setInterval(() => refreshTtsEngineStatus(els), 5000);

  // Voice-cloning panel.
  const cloneEls = {
    name:        document.getElementById('tts-clone-name'),
    text:        document.getElementById('tts-clone-text'),
    file:        document.getElementById('tts-clone-file'),
    permission:  document.getElementById('tts-clone-permission'),
    upload:      document.getElementById('tts-clone-upload-btn'),
    status:      document.getElementById('tts-clone-status'),
    listGroup:   document.getElementById('tts-clone-list-group'),
    list:        document.getElementById('tts-clone-list'),
  };
  if (cloneEls.upload) {
    cloneEls.permission.addEventListener('change', () => updateCloneUploadEnabled(cloneEls));
    [cloneEls.name, cloneEls.text, cloneEls.file].forEach(el =>
      el.addEventListener('input', () => updateCloneUploadEnabled(cloneEls))
    );
    cloneEls.upload.addEventListener('click', () => uploadClone(cloneEls, els));
    refreshCloneList(cloneEls, els);
  }
}

/** Render the voice <select> grouped by language plus a "Custom voices" group. */
function renderVoiceSelect(els) {
  const voices = ttsState.voices ?? [];
  const cfg = ttsState.config;
  const builtin = voices.filter(v => v.kind !== 'cloned');
  const cloned = voices.filter(v => v.kind === 'cloned');

  const byLang = {};
  for (const v of builtin) (byLang[v.language] ||= []).push(v);
  let html = Object.entries(byLang).map(([lang, vs]) => {
    const opts = vs.map(v => {
      const sel = v.id === cfg?.default_voice ? ' selected' : '';
      return `<option value="${v.id}"${sel}>${v.display_name}</option>`;
    }).join('');
    return `<optgroup label="${lang}">${opts}</optgroup>`;
  }).join('');

  if (cloned.length) {
    const opts = cloned.map(v =>
      `<option value="ref:${v.id}">${v.name}</option>`
    ).join('');
    html += `<optgroup label="Custom voices">${opts}</optgroup>`;
  }
  els.voice.innerHTML = html;
}

function updateCloneUploadEnabled(cloneEls) {
  const ok =
    !!cloneEls.name.value.trim() &&
    !!cloneEls.text.value.trim() &&
    !!cloneEls.file.files?.[0] &&
    cloneEls.permission.checked;
  cloneEls.upload.disabled = !ok;
}

async function uploadClone(cloneEls, els) {
  const file = cloneEls.file.files?.[0];
  if (!file) return;
  cloneEls.upload.disabled = true;
  cloneEls.status.textContent = 'Uploading…';
  cloneEls.status.style.color = '#a0a0a0';
  try {
    await ttsUploadVoiceRef({
      file,
      name: cloneEls.name.value.trim(),
      refText: cloneEls.text.value.trim(),
      permissionConfirmed: cloneEls.permission.checked,
    });
    cloneEls.status.textContent = 'Uploaded.';
    cloneEls.status.style.color = '#7fbf7f';
    // Reset form
    cloneEls.name.value = '';
    cloneEls.text.value = '';
    cloneEls.file.value = '';
    cloneEls.permission.checked = false;
    // Re-fetch voices and refresh both the dropdown and the list.
    ttsState.voices = await ttsFetchVoices();
    renderVoiceSelect(els);
    await refreshCloneList(cloneEls, els);
  } catch (err) {
    cloneEls.status.textContent = `Error: ${err.message || err}`;
    cloneEls.status.style.color = '#dc8b8b';
  } finally {
    updateCloneUploadEnabled(cloneEls);
  }
}

async function refreshCloneList(cloneEls, els) {
  const all = ttsState.voices ?? (await ttsFetchVoices());
  const cloned = all.filter(v => v.kind === 'cloned');
  if (!cloned.length) {
    cloneEls.listGroup.style.display = 'none';
    cloneEls.list.innerHTML = '';
    return;
  }
  cloneEls.listGroup.style.display = '';
  cloneEls.list.innerHTML = cloned.map(v => {
    const created = new Date(v.created_at * 1000).toLocaleString();
    return `
      <div class="media-item" style="display:flex;justify-content:space-between;align-items:center;gap:10px;padding:8px 12px;background:#16213e;border-radius:6px;margin-bottom:6px;">
        <div style="flex:1;min-width:0;">
          <div style="color:white;font-weight:500;">${v.name}</div>
          <div style="font-size:0.8em;color:#a0a0a0;">${v.duration_secs.toFixed(1)} s · ${created}</div>
        </div>
        <button class="action-btn small danger" data-clone-delete="${v.id}">Delete</button>
      </div>`;
  }).join('');

  cloneEls.list.querySelectorAll('[data-clone-delete]').forEach(btn => {
    btn.addEventListener('click', async () => {
      try {
        await ttsDeleteVoiceRef(btn.dataset.cloneDelete);
        ttsState.voices = await ttsFetchVoices();
        renderVoiceSelect(els);
        await refreshCloneList(cloneEls, els);
      } catch (err) {
        console.warn('clone delete failed:', err);
      }
    });
  });
}

/** Render the TTS engine status badge from /api/tts/status. */
async function refreshTtsEngineStatus(els) {
  if (!els.engineBadge) return;
  let info;
  try {
    info = await fetchTtsStatus();
  } catch (_) {
    els.engineBadge.textContent = 'unreachable';
    els.engineBadge.style.background = '#5a3a3a';
    els.engineBadge.style.color = '#dc8b8b';
    return;
  }
  const presets = {
    idle:     { label: 'idle',                 bg: '#3a3a4a', fg: '#a0a0a0' },
    starting: { label: 'warming up…',          bg: '#5a4a2a', fg: '#ffd866' },
    ready:    { label: 'ready',                bg: '#2a5a3a', fg: '#7fbf7f' },
    stopping: { label: 'stopping…',            bg: '#5a4a2a', fg: '#ffd866' },
    blocked:  { label: 'blocked by ASR',       bg: '#5a3a3a', fg: '#dc8b8b' },
  };
  const p = presets[info.state] ?? presets.idle;
  let label = p.label;
  if (info.state === 'starting' && info.boot_elapsed_secs != null && info.boot_timeout_secs) {
    const remaining = Math.max(0, info.boot_timeout_secs - info.boot_elapsed_secs);
    label = `warming up · ${remaining.toFixed(0)}s left`;
  } else if (info.state === 'blocked' && info.blocked_reason) {
    label = info.blocked_reason;
  }
  els.engineBadge.textContent = label;
  els.engineBadge.style.background = p.bg;
  els.engineBadge.style.color = p.fg;

  // Disable Generate while blocked.
  if (info.state === 'blocked') {
    els.generate.disabled = true;
    els.generate.title = info.blocked_reason || 'TTS blocked';
  } else if (ttsState.loaded) {
    els.generate.disabled = ttsState.inFlight;
    els.generate.title = '';
  }
}

/**
 * Toggle the visibility of Pause/Resume/Stop buttons based on player state.
 * `idle` hides everything; `playing` shows Pause+Stop; `paused` shows
 * Resume+Stop; `buffering` shows Stop (no pause-while-buffering).
 */
function setStreamingButtons(els, state) {
  els.pause.hidden = state !== 'playing';
  els.resume.hidden = state !== 'paused';
  els.stop.hidden = state === 'idle' || state === 'ended';
}

async function generateTts(els) {
  if (ttsState.inFlight) return;
  const text = els.text.value.trim();
  if (!text) {
    els.status.textContent = 'Enter some text first.';
    els.status.style.color = '#dc8b8b';
    return;
  }
  // Cloned-voice options use the `ref:<uuid>` prefix in the <select> value.
  const raw = els.voice.value;
  const isCloned = raw.startsWith('ref:');
  const voice = isCloned ? '' : raw;
  const voiceRefId = isCloned ? raw.slice(4) : null;
  const mode = document.querySelector('input[name="tts-output-mode"]:checked')?.value || 'play';

  ttsState.inFlight = true;
  els.generate.disabled = true;
  els.status.style.color = '#a0a0a0';
  els.status.textContent = 'Synthesizing…';

  try {
    if (mode === 'save') {
      const filename = (els.filename.value || '').trim() || null;
      const result = await ttsSynthesizeAndSave({
        text, voice, voiceRefId, filename,
        overwrite: !!els.overwrite.checked,
      });
      els.status.textContent =
        `Saved ${result.filename} (${formatFileSize(result.bytes)}, ${result.elapsed_secs.toFixed(2)}s)`;
      els.status.style.color = '#7fbf7f';
      els.audio.hidden = false;
      els.audio.src = `/api/tts/output/${encodeURIComponent(result.filename)}`;
      await refreshSavedFiles(els);
    } else {
      // Phase 4: progressive playback via Web Audio. We hand the streaming
      // Response straight to TtsPlayer.start() — no `await response.blob()`
      // round-trip, so first audio plays as soon as the first PCM chunk
      // lands on the wire.
      els.audio.hidden = true;          // <audio> is unused in stream mode
      ttsState.player?.cancel();        // tear down any previous player

      const player = new TtsPlayer({
        onStateChange: ({ state, currentTimeSecs, bufferedSecs }) => {
          setStreamingButtons(els, state);
          if (state === 'buffering') {
            els.status.textContent = 'Buffering…';
            els.status.style.color = '#a0a0a0';
          } else if (state === 'playing') {
            els.status.textContent =
              `Playing (${currentTimeSecs.toFixed(1)} s, ${bufferedSecs.toFixed(1)} s buffered)`;
            els.status.style.color = '#7fbf7f';
          } else if (state === 'paused') {
            els.status.textContent = 'Paused.';
            els.status.style.color = '#d0d050';
          } else if (state === 'ended') {
            els.status.textContent = `Done (${currentTimeSecs.toFixed(1)} s).`;
            els.status.style.color = '#7fbf7f';
          }
        },
      });
      ttsState.player = player;
      // Make the player observable to Playwright via window.__ttsPlayerState.
      try { exposePlayerForTests(player); } catch (_) { /* ignore in prod */ }

      const t0 = performance.now();
      const response = await ttsSynthesizeForPlayback({ text, voice, voiceRefId });
      try {
        await player.start(response);
      } catch (err) {
        console.warn('TtsPlayer.start failed:', err);
        els.status.textContent = `Player error: ${err.message || err}`;
        els.status.style.color = '#dc8b8b';
      }
      const elapsed = (performance.now() - t0) / 1000;
      // Final status is set by the onStateChange handler; just append timing.
      els.status.textContent += ` · TTFB→done ${elapsed.toFixed(2)} s`;
      setStreamingButtons(els, 'ended');
    }
  } catch (err) {
    console.error('TTS synth failed:', err);
    els.status.textContent = `Error: ${err.message || err}`;
    els.status.style.color = '#dc8b8b';
  } finally {
    ttsState.inFlight = false;
    els.generate.disabled = false;
  }
}

async function refreshSavedFiles(els) {
  let files = [];
  try { files = await ttsFetchSavedFiles(); } catch (_) { return; }
  if (!files.length) {
    els.savedGroup.style.display = 'none';
    els.savedList.innerHTML = '';
    return;
  }
  els.savedGroup.style.display = '';
  els.savedList.innerHTML = files.map(f => {
    const created = new Date(f.created_at * 1000).toLocaleString();
    const url = `/api/tts/output/${encodeURIComponent(f.name)}`;
    return `
      <div class="media-item" style="display:flex;justify-content:space-between;align-items:center;gap:10px;padding:8px 12px;background:#16213e;border-radius:6px;margin-bottom:6px;">
        <div style="flex:1;min-width:0;">
          <div style="color:white;font-weight:500;">${f.name}</div>
          <div style="font-size:0.8em;color:#a0a0a0;">${formatFileSize(f.bytes)} · ${created}</div>
        </div>
        <a href="${url}" download="${f.name}" class="action-btn small">Download</a>
        <button class="action-btn small danger" data-tts-delete="${encodeURIComponent(f.name)}">Delete</button>
      </div>`;
  }).join('');

  els.savedList.querySelectorAll('[data-tts-delete]').forEach(btn => {
    btn.addEventListener('click', async () => {
      const name = decodeURIComponent(btn.dataset.ttsDelete);
      try {
        await ttsDeleteSavedFile(name);
        await refreshSavedFiles(els);
      } catch (err) {
        console.warn('TTS delete failed:', err);
      }
    });
  });
}

function showTab(tabName) {
  elements.sessionTabs.forEach(tab => {
    tab.classList.toggle('active', tab.dataset.tab === tabName);
  });
  elements.sessionContents.forEach(content => {
    content.classList.toggle('active', content.id === `${tabName}-content`);
  });
}

// Language display names for the dropdown
const LANGUAGE_NAMES = {
  en: 'English',
  de: 'German (Deutsch)',
  fr: 'French (Fran\u00e7ais)',
  es: 'Spanish (Espa\u00f1ol)',
  it: 'Italian (Italiano)',
  pt: 'Portuguese (Portugu\u00eas)',
  nl: 'Dutch (Nederlands)',
  pl: 'Polish (Polski)',
  ja: 'Japanese (\u65e5\u672c\u8a9e)',
  zh: 'Chinese (\u4e2d\u6587)',
  ko: 'Korean (\ud55c\uad6d\uc5b4)',
  ru: 'Russian (\u0420\u0443\u0441\u0441\u043a\u0438\u0439)',
};

function renderModelSelect() {
  const models = sessionManager.models;
  const available = models.filter(m => m.is_loaded);
  const unavailable = models.filter(m => !m.is_loaded);
  if (available.length === 0 && unavailable.length === 0) {
    elements.modelSelect.innerHTML = '<option value="">No models available</option>';
  } else {
    elements.modelSelect.innerHTML =
      available.map(m => `<option value="${m.id}">${m.display_name}</option>`).join('') +
      unavailable.map(m => `<option value="${m.id}" disabled>${m.display_name} (not installed)</option>`).join('');
  }
  // Update language dropdown for the initially selected model
  renderLanguageSelect();
}

function renderLanguageSelect() {
  const modelId = elements.modelSelect?.value;
  const model = sessionManager.models?.find(m => m.id === modelId);
  const languages = model?.languages || ['en', 'de', 'fr', 'es'];
  const currentLang = elements.languageSelect?.value;

  elements.languageSelect.innerHTML = languages
    .map(code => {
      const name = LANGUAGE_NAMES[code] || code;
      const selected = code === currentLang ? ' selected' : (code === 'de' && !languages.includes(currentLang) ? ' selected' : '');
      return `<option value="${code}"${selected}>${name}</option>`;
    })
    .join('');

  // Update hint text
  const hint = document.getElementById('language-hint');
  if (hint && model) {
    hint.textContent = `${model.display_name}: ${languages.join('/')}`;
  }
}

function renderMediaSelect() {
  const files = sessionManager.mediaFiles;
  elements.mediaSelect.innerHTML = files.length
    ? files.map(f => `<option value="${f.id}">${f.filename} (${formatDuration(f.duration_secs)})</option>`).join('')
    : '<option value="">No media files</option>';
}

function renderModeSelect() {
  if (!elements.modeSelect) return;
  const modes = sessionManager.modes;
  elements.modeSelect.innerHTML = modes.length
    ? modes.map(m => `<option value="${m.id}" title="${m.description}">${m.name}</option>`).join('')
    : '<option value="speedy">Speedy (~0.3-1.5s)</option>';
}

function renderNoiseSelect() {
  if (!elements.noiseSelect) return;
  const options = sessionManager.noiseCancellationOptions;
  elements.noiseSelect.innerHTML = options.length
    ? options.filter(o => o.available).map(o =>
        `<option value="${o.id}" title="${o.description}">${o.name}</option>`
      ).join('')
    : '<option value="none">None</option>';
}

function renderDiarizationSelect() {
  if (!elements.diarizationSelect) return;
  const options = sessionManager.diarizationOptions;
  elements.diarizationSelect.innerHTML = options.length
    ? options.filter(o => o.available).map(o =>
        `<option value="${o.id}" title="${o.description}">${o.name}</option>`
      ).join('')
    : '<option value="none">None</option>';
}

function renderSrtSelect(streams, configured) {
  if (!elements.srtSelect) return;

  if (!configured || streams.length === 0) {
    elements.srtSelect.innerHTML = '<option value="">SRT not configured</option>';
    elements.srtSelect.disabled = true;
    // Hide SRT tab if not configured
    if (elements.sourceTabs) {
      elements.sourceTabs.forEach(tab => {
        if (tab.dataset.source === 'srt') {
          tab.style.display = 'none';
        }
      });
    }
    return;
  }

  // Show SRT tab if configured
  if (elements.sourceTabs) {
    elements.sourceTabs.forEach(tab => {
      if (tab.dataset.source === 'srt') {
        tab.style.display = '';
      }
    });
  }

  elements.srtSelect.disabled = false;
  elements.srtSelect.innerHTML = streams
    .map(s => `<option value="${s.id}">${s.display}</option>`)
    .join('');
}

function renderMediaList() {
  const files = sessionManager.mediaFiles;
  if (files.length === 0) {
    elements.mediaList.innerHTML = '<div class="empty-state"><p>No media files uploaded</p></div>';
    return;
  }

  elements.mediaList.innerHTML = files.map(f => `
    <div class="media-item">
      <div>
        <div class="media-name">${f.filename}</div>
        <div class="media-meta">${formatDuration(f.duration_secs)} | ${formatFileSize(f.size_bytes)}</div>
      </div>
      <button class="btn-small btn-danger" onclick="window.deleteMedia('${f.id}')">Delete</button>
    </div>
  `).join('');
}

// Expose delete function globally
window.deleteMedia = async (mediaId) => {
  if (confirm('Delete this media file?')) {
    await sessionManager.deleteMedia(mediaId);
  }
};

function renderSessionsList() {
  const sessions = sessionManager.sessions;
  if (sessions.length === 0) {
    elements.sessionsList.innerHTML = `
      <div class="empty-state">
        <h3>No active sessions</h3>
        <p>Create a new session to start transcribing</p>
      </div>
    `;
    return;
  }

  elements.sessionsList.innerHTML = sessions.map(s => {
    const isLive = s.source_type === 'srt_stream';
    const progress = !isLive && s.duration_secs > 0 ? (s.progress_secs / s.duration_secs) * 100 : 0;
    const isSelected = state.selectedSessionId === s.id;
    const modeLabel = s.mode ? s.mode.replace(/_/g, '-') : 'speedy';

    // Build badges
    const noiseLabel = s.noise_cancellation && s.noise_cancellation !== 'none'
      ? ` | <span class="noise-badge">${s.noise_cancellation}</span>`
      : '';
    const diarLabel = s.diarization
      ? ` | <span class="diar-badge">${s.diarization_model || 'Diar'}</span>`
      : '';
    const sentenceLabel = s.sentence_completion && s.sentence_completion !== 'off'
      ? ` | <span class="sentence-badge" title="Sentence completion: ${s.sentence_completion}">${s.sentence_completion}</span>`
      : '';

    const liveLabel = isLive
      ? '<span class="live-badge">LIVE</span> '
      : '';

    const durationDisplay = isLive
      ? `${formatDuration(s.progress_secs)} elapsed`
      : `${formatDuration(s.progress_secs)} / ${formatDuration(s.duration_secs)}`;

    const progressBar = isLive
      ? '<div class="session-progress-bar live-progress" style="width: 100%"></div>'
      : `<div class="session-progress-bar" style="width: ${progress}%"></div>`;

    const actionButtons = s.state === 'running'
      ? `<button class="btn-small btn-danger" onclick="event.stopPropagation(); window.stopSession('${s.id}')">Stop</button>`
      : '';

    return `
      <div class="session-card ${isSelected ? 'selected' : ''}" onclick="window.selectSession('${s.id}')">
        <div class="session-info">
          <div class="session-title">${liveLabel}${s.media_filename}</div>
          <div class="session-meta">${s.model_name} | <span class="mode-badge">${modeLabel}</span>${sentenceLabel}${noiseLabel}${diarLabel} | ${durationDisplay}</div>
          <div class="session-progress">
            ${progressBar}
          </div>
        </div>
        <div class="session-actions">
          <span class="session-status ${s.state}">${s.state}</span>
          ${actionButtons}
        </div>
      </div>
    `;
  }).join('');
}

function selectSession(sessionId) {
  console.log('[SelectSession] Selecting session:', sessionId);
  state.selectedSessionId = sessionId;
  elements.connectBtn.disabled = !sessionId;
  elements.bufferInfo.textContent = sessionId ? 'Ready to join' : 'Select a session';
  renderSessionsList();
}

// Expose functions globally
window.selectSession = selectSession;

window.stopSession = async (sessionId) => {
  if (confirm('Stop this session?')) {
    await sessionManager.stopSession(sessionId);
    if (state.selectedSessionId === sessionId) {
      selectSession(null);
    }
  }
};

async function uploadFile(file) {
  if (!file.name.match(/\.(wav|mp3)$/i)) {
    alert('Please upload a WAV or MP3 file');
    return;
  }

  try {
    elements.bufferInfo.textContent = 'Uploading...';
    await sessionManager.uploadMedia(file);
    elements.bufferInfo.textContent = 'Upload complete!';
  } catch (e) {
    alert('Upload failed: ' + e.message);
    elements.bufferInfo.textContent = 'Upload failed';
  }
}

async function connect(sessionId) {
  console.log('[Connect] Starting connection to session:', sessionId);
  const config = getConfig();
  console.log('[Connect] Config:', config);
  const wsUrl = sessionManager.getWebSocketUrl(sessionId);
  console.log('[Connect] WebSocket URL:', wsUrl);

  // Create WebRTC client with ICE transport policy from server config
  webrtcClient = new WebRTCClient(wsUrl, {
    iceServers: config.iceServers,
    iceTransportPolicy: config.iceTransportPolicy || 'all'
  });

  // WebRTC events
  webrtcClient.on('welcome', (msg) => {
    console.log('Server welcome:', msg);
    if (msg.session) {
      state.totalDuration = msg.session.duration_secs || 0;
      elements.totalDuration.textContent = formatTime(state.totalDuration);
    }
  });

  webrtcClient.on('connected', () => {
    state.connected = true;
    updateConnectionStatus('connected');
    elements.connectBtn.textContent = 'Leave Session';
    elements.playPauseBtn.disabled = false;
    elements.bufferInfo.textContent = 'WebRTC Connected';
    console.log('WebRTC connected');
  });

  webrtcClient.on('trackReceived', async () => {
    console.log('Audio track received');
    elements.bufferInfo.textContent = 'Audio streaming';
    // Start stats logging and run debug after short delay
    webrtcClient.startStatsLogging(3000);
    setTimeout(async () => {
      await webrtcClient.debugStatus();
    }, 2000);
  });

  webrtcClient.on('autoplayBlocked', () => {
    elements.bufferInfo.textContent = 'Click Play to start';
  });

  webrtcClient.on('disconnect', ({ code, reason }) => {
    state.connected = false;
    updateConnectionStatus('disconnected');
    elements.connectBtn.textContent = 'Join Session';
    subtitleRenderer.clearCurrent();
    console.log('Disconnected:', code, reason);
  });

  webrtcClient.on('reconnecting', ({ attempt, delay }) => {
    state.connected = false;
    updateConnectionStatus('reconnecting');
    elements.bufferInfo.textContent = `Reconnecting... (attempt ${attempt})`;
    console.log(`Reconnecting in ${delay}ms (attempt ${attempt})`);
  });

  webrtcClient.on('reconnectFailed', () => {
    state.connected = false;
    updateConnectionStatus('disconnected');
    elements.bufferInfo.textContent = 'Reconnection failed';
    elements.connectBtn.textContent = 'Join Session';
    console.error('Max reconnection attempts reached');
  });

  webrtcClient.on('connectionFailed', () => {
    state.connected = false;
    updateConnectionStatus('reconnecting');
    elements.bufferInfo.textContent = 'ICE failed, reconnecting...';
    subtitleRenderer.clearCurrent();
  });

  webrtcClient.on('error', (error) => {
    console.error('WebRTC error:', error);
  });

  webrtcClient.on('subtitle', (segment) => {
    state.subtitleCount++;
    elements.bufferInfo.textContent = `Subtitles: ${state.subtitleCount}`;
    subtitleRenderer.addSegment(segment);
  });

  webrtcClient.on('status', ({ bufferTime, totalDuration }) => {
    state.totalDuration = totalDuration;
    elements.durationInfo.textContent = `Duration: ${formatTime(totalDuration)}`;
    elements.totalDuration.textContent = formatTime(totalDuration);
    updateProgressBar();
  });

  webrtcClient.on('end', ({ totalDuration, is_live }) => {
    state.totalDuration = totalDuration;
    elements.totalDuration.textContent = formatTime(totalDuration);
    if (is_live) {
      console.log('Live stream stopped');
    } else {
      console.log('Stream ended, total duration:', formatTime(totalDuration));
    }
  });

  webrtcClient.on('serverError', ({ message }) => {
    console.error('Server error:', message);
    alert(`Server error: ${message}`);
  });

  // SRT stream reconnection events
  webrtcClient.on('srtReconnecting', ({ attempt, maxAttempts, delayMs }) => {
    elements.bufferInfo.textContent = `SRT reconnecting... (${attempt}/${maxAttempts})`;
    elements.connectionStatus.textContent = 'Reconnecting...';
    elements.connectionStatus.className = 'status reconnecting';
    console.log(`SRT reconnecting in ${delayMs}ms (attempt ${attempt}/${maxAttempts})`);
  });

  webrtcClient.on('srtReconnected', () => {
    elements.bufferInfo.textContent = 'SRT reconnected';
    elements.connectionStatus.textContent = 'Connected';
    elements.connectionStatus.className = 'status connected';
    console.log('SRT stream reconnected');
  });

  // Reset subtitle counter
  state.subtitleCount = 0;

  // Connect
  updateConnectionStatus('connecting');
  try {
    await webrtcClient.connect(elements.audioPlayer);
    // Expose for debugging (call window._webrtcClient.debugStatus() in console)
    window._webrtcClient = webrtcClient;
  } catch (error) {
    console.error('Failed to connect:', error);
    updateConnectionStatus('disconnected');
    elements.bufferInfo.textContent = 'Connection failed';
  }
}

function disconnect() {
  if (webrtcClient) {
    webrtcClient.stopStatsLogging();
    webrtcClient.disconnect();
    webrtcClient = null;
    window._webrtcClient = null;
  }
  if (uplinkClient) {
    uplinkClient.disconnect();
    uplinkClient = null;
    window._uplinkClient = null;
  }
  state.connected = false;
  updateConnectionStatus('disconnected');
  elements.connectBtn.textContent = 'Join Session';
}

/**
 * Read the current "Speakers" method from the radio group ("display" or "device").
 */
function getSpeakersMethod() {
  if (!elements.speakersMethodRadios) return 'display';
  for (const radio of elements.speakersMethodRadios) {
    if (radio.checked) return radio.value;
  }
  return 'display';
}

/**
 * Resolve the capture source ID to pass to captureSpeakersStream().
 *  - method=display → "display" (browser screen-share picker)
 *  - method=device  → the selected audioinput deviceId
 */
function getSpeakersSourceId() {
  const method = getSpeakersMethod();
  if (method === 'display') return 'display';
  const id = elements.speakersSelect?.value || '';
  if (!id || id === 'display') {
    throw new Error('Pick an audio input device, or switch to the "Pick a tab/window/screen" method.');
  }
  return id;
}

/**
 * "Test capture" — verifies the user can grant capture permissions and that
 * the resulting stream contains an audio track. Stops the stream immediately;
 * the user can then create a real session.
 */
async function testSpeakersCapture() {
  if (!elements.speakersTestStatus) return;
  elements.speakersTestStatus.style.color = '#888';
  elements.speakersTestStatus.textContent = 'Opening picker…';

  let stream;
  try {
    const sourceId = getSpeakersSourceId();
    stream = await captureSpeakersStream(sourceId);
  } catch (e) {
    elements.speakersTestStatus.style.color = '#f88';
    const msg = (e && e.message) || String(e);
    elements.speakersTestStatus.textContent = `Capture failed: ${msg}`;
    return;
  }

  const audioTracks = stream.getAudioTracks();
  if (!audioTracks.length) {
    stream.getTracks().forEach(t => t.stop());
    elements.speakersTestStatus.style.color = '#f88';
    elements.speakersTestStatus.textContent =
      'Capture has no audio. Re-try and tick "Share audio" in the dialog.';
    return;
  }

  const label = audioTracks[0].label || 'audio source';
  elements.speakersTestStatus.style.color = '#7d7';
  elements.speakersTestStatus.textContent = `OK — capturing ${label}. Stopping preview.`;

  // Stop the test stream immediately — the real session will request a fresh one
  setTimeout(() => {
    stream.getTracks().forEach(t => t.stop());
  }, 300);
}

/**
 * Populate the speakers device dropdown from enumerateDevices().
 */
async function refreshAudioInputDevices() {
  if (!elements.speakersSelect) return;
  try {
    const devices = await listAudioInputDevices();
    const prev = elements.speakersSelect.value;
    const opts = [];

    if (devices.length === 0) {
      opts.push('<option value="">No audio input devices found</option>');
    } else {
      // Highlight likely loopback devices (system-audio capture) at the top.
      const loopbackPattern = /(stereo\s*mix|monitor of|loopback|blackhole|soundflower|virtual.*cable|vb-?audio|wasapi.*loopback|what\s*u\s*hear)/i;
      const loopback = devices.filter(d => loopbackPattern.test(d.label || ''));
      const others = devices.filter(d => !loopbackPattern.test(d.label || ''));

      if (loopback.length) {
        opts.push('<optgroup label="Likely speaker loopback (recommended)">');
        for (const d of loopback) {
          opts.push(`<option value="${d.deviceId}">${d.label}</option>`);
        }
        opts.push('</optgroup>');
        opts.push('<optgroup label="Other inputs (microphones, headsets — won\'t capture speakers)">');
      } else {
        opts.push('<optgroup label="Microphones / inputs (won\'t capture speakers — install a loopback driver)">');
      }
      for (const d of others) {
        opts.push(`<option value="${d.deviceId}">${d.label}</option>`);
      }
      opts.push('</optgroup>');
    }

    elements.speakersSelect.innerHTML = opts.join('');
    if (prev && [...elements.speakersSelect.options].some(o => o.value === prev)) {
      elements.speakersSelect.value = prev;
    }

    // If all labels are empty, the browser is masking them — need mic permission.
    if (devices.length > 0 && devices.every(d => !d.label || d.label === 'Unnamed device')) {
      if (elements.speakersPermissionHint) {
        elements.speakersPermissionHint.style.display = 'block';
      }
    }
  } catch (e) {
    console.warn('[Speakers] Failed to enumerate devices:', e);
  }
}

/**
 * Capture local audio and connect as an uplink (client → server) WebRTC peer.
 */
async function startSpeakersUplink(sessionId) {
  if (uplinkClient) {
    uplinkClient.disconnect();
    uplinkClient = null;
  }

  let sourceId;
  try {
    sourceId = getSpeakersSourceId();
  } catch (e) {
    elements.bufferInfo.textContent = e.message;
    throw e;
  }

  elements.bufferInfo.textContent = sourceId === 'display'
    ? 'Pick a tab/window/screen and tick "Share audio"…'
    : 'Requesting capture permission…';

  let stream;
  try {
    stream = await captureSpeakersStream(sourceId);
  } catch (e) {
    console.error('[Speakers] Capture failed:', e);
    elements.bufferInfo.textContent = 'Capture cancelled';
    throw e;
  }

  const audioTracks = stream.getAudioTracks();
  if (!audioTracks.length) {
    stream.getTracks().forEach(t => t.stop());
    throw new Error('No audio track available from the selected source');
  }
  console.log('[Speakers] Capturing from', audioTracks[0].label);

  const config = getConfig();
  const wsUrl = sessionManager.getWebSocketUrl(sessionId);

  uplinkClient = new WebRTCUplinkClient(wsUrl, {
    iceServers: config.iceServers,
    iceTransportPolicy: config.iceTransportPolicy || 'all',
    stream,
  });

  uplinkClient.on('welcome', (msg) => {
    console.log('[Speakers] Welcome:', msg);
  });

  uplinkClient.on('connected', () => {
    state.connected = true;
    updateConnectionStatus('connected');
    elements.connectBtn.disabled = false;
    elements.connectBtn.textContent = 'Leave Session';
    elements.playPauseBtn.disabled = true; // No local playback needed
    elements.bufferInfo.textContent = 'Uplink connected';
  });

  uplinkClient.on('disconnect', ({ code, reason }) => {
    state.connected = false;
    updateConnectionStatus('disconnected');
    elements.connectBtn.textContent = 'Join Session';
    console.log('[Speakers] Disconnected:', code, reason);
  });

  uplinkClient.on('reconnecting', ({ attempt, delay }) => {
    state.connected = false;
    updateConnectionStatus('reconnecting');
    elements.bufferInfo.textContent = `Reconnecting (${attempt})…`;
  });

  uplinkClient.on('reconnectFailed', () => {
    state.connected = false;
    updateConnectionStatus('disconnected');
    elements.bufferInfo.textContent = 'Reconnection failed';
  });

  uplinkClient.on('connectionFailed', () => {
    state.connected = false;
    updateConnectionStatus('reconnecting');
    elements.bufferInfo.textContent = 'ICE failed';
  });

  uplinkClient.on('subtitle', (segment) => {
    state.subtitleCount++;
    elements.bufferInfo.textContent = `Subtitles: ${state.subtitleCount}`;
    subtitleRenderer.addSegment(segment);
  });

  uplinkClient.on('status', ({ bufferTime, totalDuration }) => {
    state.totalDuration = totalDuration;
    elements.durationInfo.textContent = `Duration: ${formatTime(totalDuration)}`;
    elements.totalDuration.textContent = formatTime(totalDuration);
  });

  uplinkClient.on('end', ({ totalDuration }) => {
    console.log('[Speakers] End, total duration:', formatTime(totalDuration));
  });

  uplinkClient.on('serverError', ({ message }) => {
    console.error('[Speakers] Server error:', message);
    alert(`Server error: ${message}`);
  });

  updateConnectionStatus('connecting');
  state.subtitleCount = 0;

  // Auto-stop if the user ends sharing from the browser picker
  audioTracks[0].addEventListener('ended', () => {
    console.log('[Speakers] Capture track ended — disconnecting');
    disconnect();
  });

  try {
    await uplinkClient.connect();
    window._uplinkClient = uplinkClient;
  } catch (e) {
    console.error('[Speakers] Connect failed:', e);
    updateConnectionStatus('disconnected');
    elements.bufferInfo.textContent = 'Connection failed';
    throw e;
  }
}

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

function updateProgressBar() {
  if (state.totalDuration === 0) {
    elements.progressBuffered.style.width = '0%';
    elements.progressPlayed.style.width = '0%';
    return;
  }

  const currentTime = elements.audioPlayer?.currentTime || 0;
  const playedPercent = (currentTime / state.totalDuration) * 100;

  elements.progressBuffered.style.width = '100%';
  elements.progressPlayed.style.width = `${Math.min(playedPercent, 100)}%`;
}

function updateUI() {
  const currentTime = elements.audioPlayer?.currentTime || 0;
  elements.currentTime.textContent = formatTime(currentTime);
  elements.totalDuration.textContent = formatTime(state.totalDuration);
  updateProgressBar();
}

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
