//! Session runner: orchestrates audio → mel → inference → text → broadcast.
//!
//! Lifecycle:
//!   1. Resolve media file path
//!   2. Spawn FFmpeg to decode media file to PCM 16kHz
//!   3. Accumulate audio in batches (0.5s)
//!   4. Apply padding (Q4: 76 tokens, BF16: 32 tokens)
//!   5. Compute mel spectrogram
//!   6. Run inference (Q4 or BF16 via InferenceEngine trait)
//!   7. Split text into sentences
//!   8. Broadcast SubtitleMessage to WebSocket subscribers

use crate::audio::{
    ffmpeg,
    mel::{MelConfig, MelSpectrogram},
    pad::{pad_audio, PadConfig},
    AudioBuffer,
};
use crate::inference::InferenceEngine;
use crate::server::state::{SessionState, SubtitleMessage};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{mpsc, RwLock};

use super::streaming::GrowingTextTracker;

/// Configuration for a transcription session.
#[derive(Debug, Clone)]
pub struct SessionRunnerConfig {
    pub session_id: String,
    pub media_path: PathBuf,
    pub language: String,
    pub mode: String,
    pub quant: String,
    /// Audio batch duration in milliseconds
    pub batch_duration_ms: u32,
}

impl SessionRunnerConfig {
    pub fn batch_duration_ms(&self) -> u32 {
        if self.batch_duration_ms > 0 {
            self.batch_duration_ms
        } else {
            500 // default 0.5s
        }
    }
}

/// Result of running a transcription session.
#[derive(Debug)]
pub struct SessionResult {
    pub session_id: String,
    pub total_audio_secs: f32,
    pub total_segments: u32,
    pub error: Option<String>,
}

/// Run a transcription session.
///
/// This is the main orchestration function that ties together:
/// FFmpeg → PCM chunks → mel spectrogram → inference → text → broadcast
pub async fn run_session(
    config: SessionRunnerConfig,
    engine: Arc<dyn InferenceEngine>,
    subtitle_tx: mpsc::Sender<SubtitleMessage>,
    state_update_tx: mpsc::Sender<(String, SessionState)>,
) -> SessionResult {
    let session_id = config.session_id.clone();

    // Signal that we're starting
    let _ = state_update_tx
        .send((session_id.clone(), SessionState::Running))
        .await;

    // Create mel spectrogram extractor
    let mel_spec = MelSpectrogram::new(MelConfig::default());

    // Choose padding based on quantization
    let pad_config = if config.quant == "q4" {
        PadConfig::q4()
    } else {
        PadConfig::bf16()
    };

    // Spawn FFmpeg
    let media_path = config.media_path.clone();
    let batch_ms = config.batch_duration_ms();

    let (audio_tx, mut audio_rx) = mpsc::channel::<Vec<f32>>(64);

    // FFmpeg runs in a blocking thread
    let ffmpeg_handle = {
        let path = media_path.clone();
        let tx = audio_tx.clone();
        tokio::task::spawn_blocking(move || {
            match ffmpeg::spawn_ffmpeg(&path) {
                Ok(child) => {
                    ffmpeg::read_pcm_chunks(child, batch_ms, tx);
                }
                Err(e) => {
                    eprintln!("[Session] Failed to spawn FFmpeg: {}", e);
                }
            }
        })
    };
    // Drop our copy so the channel closes when FFmpeg thread finishes
    drop(audio_tx);

    // Text tracking
    let mut text_tracker = GrowingTextTracker::new();
    let mut total_samples: usize = 0;
    let mut segment_count: u32 = 0;
    let mut accumulated_audio = Vec::new();
    let start_time = Instant::now();

    // Process audio chunks as they arrive
    while let Some(samples) = audio_rx.recv().await {
        total_samples += samples.len();
        accumulated_audio.extend_from_slice(&samples);

        // Accumulate at least 1 second of audio before first inference
        // (mel spectrogram needs enough frames for meaningful output)
        let min_samples = 16000; // 1 second
        if accumulated_audio.len() < min_samples {
            // Send a progress status
            let progress_secs = total_samples as f32 / 16000.0;
            let _ = subtitle_tx
                .send(SubtitleMessage {
                    session_id: session_id.clone(),
                    text: String::new(),
                    is_final: false,
                    segment_index: segment_count,
                    timestamp_ms: (progress_secs * 1000.0) as u64,
                })
                .await;
            continue;
        }

        // Create AudioBuffer, apply padding, compute mel, run inference
        let audio_buf = AudioBuffer::new(accumulated_audio.clone(), 16000);
        let padded = pad_audio(&audio_buf, &pad_config);

        // Compute mel spectrogram
        let mel_frames = mel_spec.compute_log_flat(&padded.samples);

        // Run inference
        let language = config.language.clone();
        let engine_clone = engine.clone();

        let transcription = tokio::task::spawn_blocking(move || {
            engine_clone.transcribe(&mel_frames, &language)
        })
        .await;

        match transcription {
            Ok(Ok(text)) if !text.is_empty() => {
                let (finals, partial) = text_tracker.update(&text);

                // Emit final segments
                for (sentence, idx) in finals {
                    segment_count = idx + 1;
                    let _ = subtitle_tx
                        .send(SubtitleMessage {
                            session_id: session_id.clone(),
                            text: sentence,
                            is_final: true,
                            segment_index: idx,
                            timestamp_ms: (total_samples as f32 / 16000.0 * 1000.0) as u64,
                        })
                        .await;
                }

                // Emit partial
                if let Some((text, idx)) = partial {
                    let _ = subtitle_tx
                        .send(SubtitleMessage {
                            session_id: session_id.clone(),
                            text,
                            is_final: false,
                            segment_index: idx,
                            timestamp_ms: (total_samples as f32 / 16000.0 * 1000.0) as u64,
                        })
                        .await;
                }

                // Reset accumulated audio after successful inference
                // (keep processing incrementally for streaming modes)
                accumulated_audio.clear();
            }
            Ok(Ok(_)) => {
                // Empty transcription — continue accumulating
            }
            Ok(Err(e)) => {
                eprintln!(
                    "[Session {}] Inference error at {:.1}s: {}",
                    session_id,
                    total_samples as f32 / 16000.0,
                    e
                );
            }
            Err(e) => {
                eprintln!("[Session {}] Task join error: {}", session_id, e);
            }
        }
    }

    // Flush any remaining text
    if let Some((text, idx)) = text_tracker.flush() {
        let _ = subtitle_tx
            .send(SubtitleMessage {
                session_id: session_id.clone(),
                text,
                is_final: true,
                segment_index: idx,
                timestamp_ms: (total_samples as f32 / 16000.0 * 1000.0) as u64,
            })
            .await;
        segment_count = idx + 1;
    }

    // Wait for FFmpeg to finish
    let _ = ffmpeg_handle.await;

    let elapsed = start_time.elapsed();
    let audio_secs = total_samples as f32 / 16000.0;
    eprintln!(
        "[Session {}] Complete: {:.1}s audio in {:.1}s ({:.2}x realtime), {} segments",
        session_id,
        audio_secs,
        elapsed.as_secs_f32(),
        audio_secs / elapsed.as_secs_f32(),
        segment_count,
    );

    let _ = state_update_tx
        .send((session_id.clone(), SessionState::Completed))
        .await;

    SessionResult {
        session_id,
        total_audio_secs: audio_secs,
        total_segments: segment_count,
        error: None,
    }
}
