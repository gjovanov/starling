/**
 * Browser-side audio decoding — converts audio files to mono 16kHz Float32Array
 * suitable for the WASM inference engine.
 *
 * Uses the Web Audio API (OfflineAudioContext) for decoding and resampling.
 */

/** Target sample rate for the WASM engine */
const TARGET_SAMPLE_RATE = 16000;

/**
 * Decode an audio file to mono 16kHz Float32Array PCM samples.
 *
 * Handles:
 *   - Any format supported by the browser (WAV, MP3, OGG, FLAC, etc.)
 *   - Stereo to mono downmix
 *   - Resampling to 16kHz via OfflineAudioContext
 *
 * @param {File} file - Audio file to decode
 * @returns {Promise<Float32Array>} Mono 16kHz f32 PCM samples
 */
export async function decodeToMono16k(file) {
  if (!file) {
    throw new Error('No file provided');
  }

  console.log(`[wasm] Decoding audio file: ${file.name} (${(file.size / (1024 * 1024)).toFixed(1)} MB)`);

  // Read file as ArrayBuffer
  const arrayBuffer = await file.arrayBuffer();

  // Decode the audio using a temporary AudioContext
  // We use a standard AudioContext for decoding since OfflineAudioContext
  // requires knowing the duration up front.
  const tempCtx = new (window.AudioContext || window.webkitAudioContext)();
  let decodedBuffer;
  try {
    decodedBuffer = await tempCtx.decodeAudioData(arrayBuffer);
  } catch (err) {
    tempCtx.close();
    throw new Error(`Failed to decode audio file "${file.name}": ${err.message}`);
  }
  tempCtx.close();

  const sourceSampleRate = decodedBuffer.sampleRate;
  const sourceChannels = decodedBuffer.numberOfChannels;
  const sourceDuration = decodedBuffer.duration;

  console.log(`[wasm] Decoded: ${sourceSampleRate}Hz, ${sourceChannels}ch, ${sourceDuration.toFixed(1)}s`);

  // Downmix to mono
  let monoSamples;
  if (sourceChannels === 1) {
    monoSamples = decodedBuffer.getChannelData(0);
  } else {
    // Average all channels to mono
    const length = decodedBuffer.length;
    monoSamples = new Float32Array(length);
    for (let ch = 0; ch < sourceChannels; ch++) {
      const channelData = decodedBuffer.getChannelData(ch);
      for (let i = 0; i < length; i++) {
        monoSamples[i] += channelData[i];
      }
    }
    // Normalize by channel count
    const scale = 1.0 / sourceChannels;
    for (let i = 0; i < length; i++) {
      monoSamples[i] *= scale;
    }
    console.log(`[wasm] Downmixed ${sourceChannels} channels to mono`);
  }

  // Resample to 16kHz if needed
  if (sourceSampleRate === TARGET_SAMPLE_RATE) {
    console.log(`[wasm] Already at ${TARGET_SAMPLE_RATE}Hz, no resampling needed`);
    return monoSamples;
  }

  console.log(`[wasm] Resampling from ${sourceSampleRate}Hz to ${TARGET_SAMPLE_RATE}Hz...`);

  // Calculate the output length at the target sample rate
  const outputLength = Math.ceil(monoSamples.length * TARGET_SAMPLE_RATE / sourceSampleRate);

  // Use OfflineAudioContext to resample
  const offlineCtx = new OfflineAudioContext(1, outputLength, TARGET_SAMPLE_RATE);

  // Create a buffer at the source sample rate with our mono data
  const sourceBuffer = offlineCtx.createBuffer(1, monoSamples.length, sourceSampleRate);
  sourceBuffer.getChannelData(0).set(monoSamples);

  // Play the source buffer into the offline context (which resamples to target rate)
  const sourceNode = offlineCtx.createBufferSource();
  sourceNode.buffer = sourceBuffer;
  sourceNode.connect(offlineCtx.destination);
  sourceNode.start(0);

  const renderedBuffer = await offlineCtx.startRendering();
  const resampledData = renderedBuffer.getChannelData(0);

  console.log(`[wasm] Resampled: ${resampledData.length} samples (${(resampledData.length / TARGET_SAMPLE_RATE).toFixed(1)}s at ${TARGET_SAMPLE_RATE}Hz)`);

  return resampledData;
}
