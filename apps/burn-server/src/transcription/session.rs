//! Session runner — replicates vllm-server's audio processing flow.
//!
//! Architecture: Two decoupled async tasks connected by a channel.
//!
//! Task 1 (Playback): FFmpeg → Opus encode → WebRTC RTP write (real-time, never blocks)
//! Task 2 (Inference): Receives 0.5s batches → send_audio + commit → emit subtitles
//!
//! This prevents GPU inference from starving the audio stream.

use crate::audio::ffmpeg;
use crate::audio::opus::OpusEncoder;
use crate::inference::{InferenceEngine, MAX_COMMITS_BEFORE_ROTATE};
use crate::server::state::{AppState, SessionState, SubtitleMessage};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc;
use webrtc::track::track_local::TrackLocalWriter;

use super::streaming::split_sentences;

/// Audio batch interval in seconds.
/// candle-native: 0.5s batches matching vllm-server for responsive growing segments.
const BATCH_INTERVAL_SECS: f32 = 0.5;
/// Samples per batch at 16kHz
const BATCH_SAMPLES: usize = (16000.0 * BATCH_INTERVAL_SECS) as usize;

/// Configuration for a transcription session.
#[derive(Debug, Clone)]
pub struct SessionRunnerConfig {
    pub session_id: String,
    pub media_path: PathBuf,
    pub language: String,
}

fn subtitle(session_id: &str, text: String, is_final: bool, segment_index: u32, timestamp_ms: u64, inference_time_ms: Option<f32>) -> SubtitleMessage {
    SubtitleMessage {
        msg_type: "subtitle".to_string(),
        session_id: session_id.to_string(),
        text,
        is_final,
        segment_index,
        timestamp_ms,
        inference_time_ms,
    }
}

/// Run a transcription session — same flow as vllm-server's run_session().
pub async fn run_session(
    config: SessionRunnerConfig,
    engine: Arc<dyn InferenceEngine>,
    subtitle_tx: mpsc::Sender<SubtitleMessage>,
    state_update_tx: mpsc::Sender<(String, SessionState)>,
    app_state: Arc<AppState>,
) {
    let session_id = config.session_id.clone();
    let language = config.language.clone();

    // Create inference session
    let session_result = {
        let lang = language.clone();
        let eng = engine.clone();
        tokio::task::spawn_blocking(move || eng.create_session(&lang)).await
    };

    let mut inference_session = match session_result {
        Ok(Ok(session)) => session,
        Ok(Err(e)) => {
            eprintln!("[Session {}] Failed to create inference session: {}", session_id, e);
            let _ = state_update_tx.send((session_id.clone(), SessionState::Error)).await;
            let _ = subtitle_tx.send(subtitle(&session_id, format!("Error: {}", e), true, 0, 0, None)).await;
            return;
        }
        Err(e) => {
            eprintln!("[Session {}] Task join error: {}", session_id, e);
            let _ = state_update_tx.send((session_id, SessionState::Error)).await;
            return;
        }
    };

    let _ = state_update_tx
        .send((session_id.clone(), SessionState::Running))
        .await;

    // Broadcast start
    let _ = subtitle_tx.send(subtitle(&session_id, String::new(), false, 0, 0, None)).await;

    // Spawn FFmpeg
    let (audio_tx, mut audio_rx) = mpsc::channel::<Vec<f32>>(64);
    let media_path = config.media_path.clone();
    let ffmpeg_handle = {
        let tx = audio_tx;
        tokio::task::spawn_blocking(move || {
            match ffmpeg::spawn_ffmpeg(&media_path) {
                Ok(child) => { ffmpeg::read_pcm_chunks(child, 20, tx); }
                Err(e) => eprintln!("[FFmpeg] Failed to spawn: {}", e),
            }
        })
    };

    // Channel: playback task → inference task (std::sync for cross-thread use)
    let (batch_tx, batch_rx) = std::sync::mpsc::sync_channel::<(Vec<f32>, f32)>(64);

    let start_time = Instant::now();

    // === Task 1: Audio playback (real-time, never blocks on inference) ===
    let playback_sid = session_id.clone();
    let playback_state = app_state.clone();
    let playback_handle = tokio::spawn(async move {
        let mut opus_encoder = match OpusEncoder::new() {
            Ok(enc) => Some(enc),
            Err(e) => {
                eprintln!("[Session {}] Opus encoder failed: {} — no audio", playback_sid, e);
                None
            }
        };

        let mut total_samples: usize = 0;
        let mut audio_batch: Vec<f32> = Vec::with_capacity(BATCH_SAMPLES);

        while let Some(samples_16k) = audio_rx.recv().await {
            total_samples += samples_16k.len();

            // Encode and write to WebRTC (real-time, non-blocking)
            if let Some(ref mut encoder) = opus_encoder {
                let rtp_packets = encoder.encode(&samples_16k);

                // Always re-read track — client may reconnect mid-session, replacing the track.
                // Read lock every 20ms is negligible vs stale-track silence on reconnect.
                let track = {
                    let sessions = playback_state.sessions.read().await;
                    sessions.get(&playback_sid).and_then(|ctx| ctx.rtp_track.clone())
                };

                if let Some(ref track) = track {
                    for packet in &rtp_packets {
                        let _ = track.write(packet).await;
                    }
                }
            }

            // Accumulate into 0.5s batches for inference
            audio_batch.extend_from_slice(&samples_16k);
            if audio_batch.len() >= BATCH_SAMPLES {
                let batch = std::mem::replace(&mut audio_batch, Vec::with_capacity(BATCH_SAMPLES));
                let current_time = total_samples as f32 / 16000.0;
                // Non-blocking: if inference is behind, drop the batch
                let _ = batch_tx.try_send((batch, current_time));
            }
        }

        // Flush remaining batch
        if !audio_batch.is_empty() {
            let current_time = total_samples as f32 / 16000.0;
            let _ = batch_tx.send((audio_batch, current_time));
        }

        total_samples
    });

    // === Task 2: Inference (plain thread — completely off the tokio runtime) ===
    let infer_sid = session_id.clone();
    let infer_state = app_state.clone();
    let inference_handle = std::thread::spawn(move || {
        let mut growing_text = String::new();
        let mut segment_count: u32 = 0;

        while let Ok((batch, current_time)) = batch_rx.recv() {
            inference_session.send_audio(&batch);

            let t_commit = Instant::now();
            let batch_delta = match inference_session.commit() {
                Ok(delta) => delta,
                Err(e) => {
                    eprintln!("[Session {}] Inference error at {:.1}s: {}", infer_sid, current_time, e);
                    String::new()
                }
            };
            let infer_ms = (t_commit.elapsed().as_secs_f32() * 10000.0).round() / 10.0; // round to 0.1ms

            if !batch_delta.is_empty() {
                growing_text.push_str(&batch_delta);
                let (sentences, remainder) = extract_complete_sentences(&growing_text);

                for sentence in sentences {
                    segment_count += 1;
                    let _ = subtitle_tx.blocking_send(
                        subtitle(&infer_sid, sentence, true, segment_count - 1, (current_time * 1000.0) as u64, Some(infer_ms)));
                }

                growing_text = remainder;

                if !growing_text.trim().is_empty() {
                    let _ = subtitle_tx.blocking_send(
                        subtitle(&infer_sid, growing_text.trim().to_string(), false, segment_count, (current_time * 1000.0) as u64, Some(infer_ms)));
                }
            }

            // Update progress — use try_write to avoid blocking/starving other tasks
            if let Ok(mut sessions) = infer_state.sessions.try_write() {
                if let Some(ctx) = sessions.get_mut(&infer_sid) {
                    ctx.info.progress_secs = current_time as f64;
                }
            }

            // Rotate session before context overflow
            if inference_session.commit_count() >= MAX_COMMITS_BEFORE_ROTATE {
                eprintln!(
                    "[Session {}] Rotating ({} commits, {:.1}s audio)",
                    infer_sid, inference_session.commit_count(), current_time
                );
                if let Err(e) = inference_session.reset() {
                    eprintln!("[Session {}] Rotation failed: {}", infer_sid, e);
                }
            }
        }

        // Flush remaining text as FINAL
        if !growing_text.trim().is_empty() {
            let (sentences, remainder) = extract_complete_sentences(&growing_text);
            let mut all = sentences;
            if !remainder.trim().is_empty() {
                all.push(remainder.trim().to_string());
            }
            for sentence in all {
                segment_count += 1;
                let _ = subtitle_tx.blocking_send(
                    subtitle(&infer_sid, sentence, true, segment_count - 1, 0, None));
            }
        }

        segment_count
    });

    // Wait for playback (async) and FFmpeg (blocking)
    let total_samples = playback_handle.await.unwrap_or(0);
    let _ = ffmpeg_handle.await;

    // Wait for inference thread (blocking — run in spawn_blocking to not block tokio)
    let segment_count = tokio::task::spawn_blocking(move || {
        inference_handle.join().unwrap_or(0)
    }).await.unwrap_or(0);

    let total_duration = total_samples as f32 / 16000.0;
    let wall_time = start_time.elapsed().as_secs_f32();

    let _ = state_update_tx
        .send((session_id.clone(), SessionState::Completed))
        .await;

    eprintln!(
        "[Session {}] Completed: {:.1}s audio, {} segments, {:.1}s wall time",
        session_id, total_duration, segment_count, wall_time
    );
}

/// Extract complete sentences from text — port of vllm-server's _extract_complete_sentences.
fn extract_complete_sentences(text: &str) -> (Vec<String>, String) {
    let sentences = split_sentences(text);

    if sentences.is_empty() {
        return (vec![], text.to_string());
    }

    if sentences.len() == 1 {
        let s = sentences[0].trim();
        if s.ends_with('.') || s.ends_with('!') || s.ends_with('?') {
            return (vec![s.to_string()], String::new());
        } else {
            return (vec![], s.to_string());
        }
    }

    let last = sentences.last().unwrap().trim().to_string();
    let complete: Vec<String> = sentences[..sentences.len() - 1]
        .iter()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    if last.ends_with('.') || last.ends_with('!') || last.ends_with('?') {
        let mut all = complete;
        all.push(last);
        (all, String::new())
    } else {
        (complete, last)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_complete_sentences() {
        let (sentences, remainder) = extract_complete_sentences("Hello world. How are you?");
        assert_eq!(sentences, vec!["Hello world.", "How are you?"]);
        assert!(remainder.is_empty());
    }

    #[test]
    fn test_extract_with_remainder() {
        let (sentences, remainder) =
            extract_complete_sentences("Hello world. How are you? I am");
        assert_eq!(sentences, vec!["Hello world.", "How are you?"]);
        assert_eq!(remainder, "I am");
    }

    #[test]
    fn test_extract_no_sentence_end() {
        let (sentences, remainder) = extract_complete_sentences("Hello world how are you");
        assert!(sentences.is_empty());
        assert_eq!(remainder, "Hello world how are you");
    }

    #[test]
    fn test_extract_numbered_date() {
        let (sentences, remainder) =
            extract_complete_sentences("Am 19. November war es kalt");
        assert!(sentences.is_empty());
        assert_eq!(remainder, "Am 19. November war es kalt");
    }
}
