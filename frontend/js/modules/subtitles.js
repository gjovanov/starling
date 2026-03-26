/**
 * Subtitle renderer module — displays backend output without preprocessing.
 */
import { formatTime, escapeHtml, createEventEmitter } from './utils.js';

/**
 * Subtitle segment
 * @typedef {Object} Segment
 * @property {string} text - Segment text
 * @property {string|null} growingText - Full growing transcript
 * @property {string|null} delta - New text added since last update
 * @property {boolean|null} tailChanged - Whether the tail was modified
 * @property {number|null} speaker - Speaker ID
 * @property {number} start - Start time in seconds
 * @property {number} end - End time in seconds
 * @property {boolean} isFinal - Whether segment is finalized
 * @property {number|null} inferenceTimeMs - Model inference time in milliseconds
 */

export class SubtitleRenderer {
  /**
   * @param {HTMLElement} liveElement - Element for current subtitle display
   * @param {HTMLElement} transcriptElement - Element for full transcript
   * @param {Object} options - Configuration options
   */
  constructor(liveElement, transcriptElement, options = {}) {
    this.liveElement = liveElement;
    this.transcriptElement = transcriptElement;

    this.options = {
      maxSegments: 1000,
      autoScroll: true,
      showTimestamps: true,
      speakerColors: [
        '#4A90D9', '#50C878', '#E9967A', '#DDA0DD',
        '#F0E68C', '#87CEEB', '#FFB6C1', '#98FB98',
      ],
      ...options,
    };

    /** @type {Segment[]} */
    this.segments = [];

    /** @type {Segment|null} */
    this.currentSegment = null;

    this.currentTime = 0;

    // Stale subtitle timeout (clear display after 5 seconds of no updates)
    this.staleTimeoutMs = 5000;
    this.staleTimer = null;
    this.isStale = false;

    // Event emitter
    const emitter = createEventEmitter();
    this.on = emitter.on.bind(emitter);
    this.off = emitter.off.bind(emitter);
    this.emit = emitter.emit.bind(emitter);

    this.initializeDisplay();
  }

  initializeDisplay() {
    if (this.liveElement) {
      this.liveElement.innerHTML = `
        <div class="speaker-label" data-speaker="?">Speaker ?</div>
        <div class="subtitle-text"></div>
        <div class="inference-time"></div>
      `;
      this.liveSpeakerEl = this.liveElement.querySelector('.speaker-label');
      this.liveTextEl = this.liveElement.querySelector('.subtitle-text');
      this.liveInferenceTimeEl = this.liveElement.querySelector('.inference-time');
    }

    if (this.transcriptElement) {
      this.transcriptElement.innerHTML = '';
    }
  }

  getSpeakerColor(speaker) {
    if (speaker === null || speaker === undefined) return '#888888';
    return this.options.speakerColors[speaker % this.options.speakerColors.length];
  }

  getSpeakerName(speaker) {
    if (speaker === null || speaker === undefined) return 'Speaker ?';
    return `Speaker ${speaker}`;
  }

  /**
   * Add a new segment — pass-through, no filtering or dedup.
   * @param {Segment} segment
   */
  addSegment(segment) {
    this.resetStaleTimer();

    if (segment.isFinal) {
      this.segments.push(segment);

      while (this.segments.length > this.options.maxSegments) {
        this.segments.shift();
      }

      this.appendToTranscript(segment);
    }

    // Always update live display with the latest segment (PARTIAL or FINAL)
    this.currentSegment = segment;
    this.updateLiveDisplay();
    this.emit('segment', segment);
  }

  resetStaleTimer() {
    if (this.staleTimer) clearTimeout(this.staleTimer);
    this.isStale = false;

    this.staleTimer = setTimeout(() => {
      this.isStale = true;
      this.currentSegment = null;
      this.updateLiveDisplay();
    }, this.staleTimeoutMs);
  }

  clearCurrent() {
    this.currentSegment = null;
    this.isStale = true;
    if (this.staleTimer) {
      clearTimeout(this.staleTimer);
      this.staleTimer = null;
    }
    this.updateLiveDisplay();
  }

  appendToTranscript(segment) {
    if (!this.transcriptElement) return;

    const el = document.createElement('div');
    el.className = 'transcript-segment';
    el.dataset.start = segment.start;
    el.dataset.end = segment.end;
    el.dataset.speaker = segment.speaker ?? '?';

    const color = this.getSpeakerColor(segment.speaker);
    const speakerName = this.getSpeakerName(segment.speaker);

    let html = '';

    if (speakerName) {
      html += `<span class="speaker-label" style="color: ${color}">[${speakerName}]</span>`;
    }

    if (this.options.showTimestamps) {
      html += `<span class="segment-time">${formatTime(segment.start)}</span>`;
    }

    html += `<span class="segment-text">${escapeHtml(segment.text)}</span>`;

    if (segment.inferenceTimeMs != null) {
      const inferenceTime = segment.inferenceTimeMs >= 1000
        ? `${(segment.inferenceTimeMs / 1000).toFixed(1)}s`
        : `${segment.inferenceTimeMs}ms`;
      html += `<span class="segment-inference-time" title="Model inference time">${inferenceTime}</span>`;
    }

    el.innerHTML = html;

    el.addEventListener('click', () => {
      this.emit('seek', segment.start);
    });

    this.transcriptElement.insertBefore(el, this.transcriptElement.firstChild);

    if (this.options.autoScroll) {
      this.scrollToTop();
    }
  }

  updateLiveDisplay() {
    if (!this.liveElement) return;

    let segment = null;
    if (!this.isStale) {
      segment = this.currentSegment;
      if (!segment && this.segments.length > 0) {
        segment = this.segments[this.segments.length - 1];
      }
    }

    if (segment) {
      const color = this.getSpeakerColor(segment.speaker);
      const speakerName = this.getSpeakerName(segment.speaker);

      if (speakerName) {
        this.liveSpeakerEl.textContent = speakerName;
        this.liveSpeakerEl.style.color = color;
        this.liveSpeakerEl.style.display = '';
      } else {
        this.liveSpeakerEl.textContent = '';
        this.liveSpeakerEl.style.display = 'none';
      }
      this.liveSpeakerEl.dataset.speaker = segment.speaker ?? '?';

      const displayText = segment.growingText || segment.text;
      this.liveTextEl.innerHTML = `<p>${escapeHtml(displayText)}</p>`;

      if (this.liveInferenceTimeEl && segment.inferenceTimeMs != null) {
        const inferenceTime = segment.inferenceTimeMs >= 1000
          ? `${(segment.inferenceTimeMs / 1000).toFixed(1)}s`
          : `${segment.inferenceTimeMs}ms`;
        this.liveInferenceTimeEl.textContent = `Inference: ${inferenceTime}`;
        this.liveInferenceTimeEl.style.display = '';
      } else if (this.liveInferenceTimeEl) {
        this.liveInferenceTimeEl.textContent = '';
        this.liveInferenceTimeEl.style.display = 'none';
      }

      this.liveElement.classList.add('active');
    } else {
      this.liveSpeakerEl.textContent = '';
      this.liveSpeakerEl.style.display = 'none';
      this.liveTextEl.textContent = '';
      if (this.liveInferenceTimeEl) {
        this.liveInferenceTimeEl.textContent = '';
        this.liveInferenceTimeEl.style.display = 'none';
      }
      this.liveElement.classList.remove('active');
    }
  }

  updateTime(time) {
    this.currentTime = time;
    this.updateLiveDisplay();
    this.highlightCurrentSegment(time);
  }

  highlightCurrentSegment(time) {
    if (!this.transcriptElement) return;

    this.transcriptElement.querySelectorAll('.current').forEach(el => {
      el.classList.remove('current');
    });

    const segments = this.transcriptElement.querySelectorAll('.transcript-segment');
    for (const el of segments) {
      const start = parseFloat(el.dataset.start);
      const end = parseFloat(el.dataset.end);
      if (time >= start && time <= end) {
        el.classList.add('current');
        if (this.options.autoScroll) {
          el.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
        break;
      }
    }
  }

  getSegmentAtTime(time) {
    if (this.currentSegment &&
        time >= this.currentSegment.start &&
        time <= this.currentSegment.end) {
      return this.currentSegment;
    }
    for (let i = this.segments.length - 1; i >= 0; i--) {
      const seg = this.segments[i];
      if (time >= seg.start && time <= seg.end) return seg;
    }
    return null;
  }

  getTranscript() {
    return this.segments.map(seg => {
      const speaker = this.getSpeakerName(seg.speaker);
      return `[${speaker}] ${seg.text}`;
    }).join('\n');
  }

  getTranscriptWithTimestamps() {
    return this.segments.map(seg => {
      const speaker = this.getSpeakerName(seg.speaker);
      const time = formatTime(seg.start);
      return `[${time}] [${speaker}] ${seg.text}`;
    }).join('\n');
  }

  exportJSON() {
    return JSON.stringify(this.segments, null, 2);
  }

  scrollToTop() {
    if (this.transcriptElement) this.transcriptElement.scrollTop = 0;
  }

  scrollToBottom() {
    if (this.transcriptElement) this.transcriptElement.scrollTop = this.transcriptElement.scrollHeight;
  }

  clear() {
    this.segments = [];
    this.currentSegment = null;
    this.currentTime = 0;
    this.isStale = false;
    if (this.staleTimer) {
      clearTimeout(this.staleTimer);
      this.staleTimer = null;
    }
    if (this.transcriptElement) this.transcriptElement.innerHTML = '';
    this.updateLiveDisplay();
    this.emit('clear');
  }

  setSpeakerColors(colors) {
    this.options.speakerColors = colors;
  }

  setAutoScroll(enabled) {
    this.options.autoScroll = enabled;
  }

  setShowTimestamps(show) {
    this.options.showTimestamps = show;
    this.rerender();
  }

  rerender() {
    if (!this.transcriptElement) return;
    this.transcriptElement.innerHTML = '';
    for (const segment of this.segments) {
      this.appendToTranscript(segment);
    }
  }
}
