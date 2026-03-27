//! Session runner — replicates vllm-server's audio processing flow.
//!
//! This is a direct port of vllm-server's session_runner.py to Rust,
//! using the same streaming approach:
//!
//! 1. FFmpeg decodes media to PCM 16kHz (real-time paced)
//! 2. Accumulate 0.5s audio batches (8000 samples)
//! 3. Send batch to inference engine via send_audio() + commit()
//! 4. Engine returns text deltas (new words since last commit)
//! 5. Accumulate deltas into growing_text
//! 6. Split at sentence boundaries → emit FINAL
//! 7. Keep remainder as PARTIAL
//! 8. After 200 commits (~100s audio), rotate session to avoid context overflow

use crate::audio::ffmpeg;
use crate::inference::{InferenceEngine, InferenceSession, MAX_COMMITS_BEFORE_ROTATE};
use crate::server::state::{SessionState, SubtitleMessage};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc;

use super::streaming::split_sentences;

/// Audio batch interval in seconds (matches vllm-server's BATCH_INTERVAL_SECS = 0.5)
const BATCH_INTERVAL_SECS: f32 = 0.5;
/// Samples per batch at 16kHz
const BATCH_SAMPLES: usize = (16000.0 * BATCH_INTERVAL_SECS) as usize; // 8000

/// Configuration for a transcription session.
#[derive(Debug, Clone)]
pub struct SessionRunnerConfig {
    pub session_id: String,
    pub media_path: PathBuf,
    pub language: String,
    pub quant: String,
}

/// Run a transcription session — same flow as vllm-server's run_session().
pub async fn run_session(
    config: SessionRunnerConfig,
    engine: Arc<dyn InferenceEngine>,
    subtitle_tx: mpsc::Sender<SubtitleMessage>,
    state_update_tx: mpsc::Sender<(String, SessionState)>,
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
            let _ = subtitle_tx
                .send(SubtitleMessage {
                    session_id,
                    text: format!("Error: {}", e),
                    is_final: true,
                    segment_index: 0,
                    timestamp_ms: 0,
                })
                .await;
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
    let _ = subtitle_tx
        .send(SubtitleMessage {
            session_id: session_id.clone(),
            text: String::new(),
            is_final: false,
            segment_index: 0,
            timestamp_ms: 0,
        })
        .await;

    // Spawn FFmpeg
    let (audio_tx, mut audio_rx) = mpsc::channel::<Vec<f32>>(64);
    let media_path = config.media_path.clone();

    // FFmpeg reads in 20ms chunks (320 samples) — we accumulate to 0.5s batches
    let ffmpeg_handle = {
        let tx = audio_tx;
        tokio::task::spawn_blocking(move || {
            match ffmpeg::spawn_ffmpeg(&media_path) {
                Ok(child) => {
                    // Read in 20ms chunks (matching parakeet-rs/vllm-server granularity)
                    ffmpeg::read_pcm_chunks(child, 20, tx);
                }
                Err(e) => {
                    eprintln!("[FFmpeg] Failed to spawn: {}", e);
                }
            }
        })
    };

    // State for subtitle accumulation (matches vllm-server exactly)
    let mut growing_text = String::new();
    let mut full_transcript = String::new();
    let mut segment_count: u32 = 0;
    let mut total_samples: usize = 0;
    let mut audio_batch: Vec<f32> = Vec::with_capacity(BATCH_SAMPLES);
    let start_time = Instant::now();

    // Main audio processing loop
    while let Some(samples_16k) = audio_rx.recv().await {
        total_samples += samples_16k.len();

        // Accumulate into 0.5s batch
        audio_batch.extend_from_slice(&samples_16k);

        if audio_batch.len() >= BATCH_SAMPLES {
            let batch = std::mem::replace(&mut audio_batch, Vec::with_capacity(BATCH_SAMPLES));
            let current_time = total_samples as f32 / 16000.0;

            // Send audio + commit (matches vllm-server: send_audio → commit → drain_deltas)
            inference_session.send_audio(&batch);

            let commit_result = {
                // Run inference in blocking context (GPU work)
                // We can't move the session into spawn_blocking since it's !Send across awaits
                // Instead, commit is synchronous (blocks the current task briefly)
                inference_session.commit()
            };

            let batch_delta = match commit_result {
                Ok(delta) => delta,
                Err(e) => {
                    eprintln!(
                        "[Session {}] Inference error at {:.1}s: {}",
                        session_id, current_time, e
                    );
                    String::new()
                }
            };

            if !batch_delta.is_empty() {
                growing_text.push_str(&batch_delta);

                // Extract complete sentences (matches vllm-server's _extract_complete_sentences)
                let (sentences, remainder) = extract_complete_sentences(&growing_text);

                // Emit each complete sentence as FINAL
                for sentence in sentences {
                    segment_count += 1;
                    full_transcript = if full_transcript.is_empty() {
                        sentence.clone()
                    } else {
                        format!("{} {}", full_transcript, sentence)
                    };

                    let _ = subtitle_tx
                        .send(SubtitleMessage {
                            session_id: session_id.clone(),
                            text: sentence,
                            is_final: true,
                            segment_index: segment_count - 1,
                            timestamp_ms: (current_time * 1000.0) as u64,
                        })
                        .await;
                }

                // Keep remainder as growing_text
                growing_text = remainder;

                // Emit PARTIAL for remainder
                if !growing_text.trim().is_empty() {
                    let _ = subtitle_tx
                        .send(SubtitleMessage {
                            session_id: session_id.clone(),
                            text: growing_text.trim().to_string(),
                            is_final: false,
                            segment_index: segment_count,
                            timestamp_ms: (current_time * 1000.0) as u64,
                        })
                        .await;
                }
            }

            // Rotate session before context overflow (matches vllm-server)
            if inference_session.commit_count() >= MAX_COMMITS_BEFORE_ROTATE {
                eprintln!(
                    "[Session {}] Rotating session (after {} commits, {:.1}s audio)",
                    session_id,
                    inference_session.commit_count(),
                    current_time
                );
                if let Err(e) = inference_session.reset() {
                    eprintln!("[Session {}] Session rotation failed: {}", session_id, e);
                }
            }
        }
    }

    // Flush remaining audio batch (matches vllm-server's flush logic)
    if !audio_batch.is_empty() {
        inference_session.send_audio(&audio_batch);
        if let Ok(final_delta) = inference_session.commit() {
            if !final_delta.is_empty() {
                growing_text.push_str(&final_delta);
            }
        }
    }

    // Emit all remaining text as FINAL
    if !growing_text.trim().is_empty() {
        let (sentences, remainder) = extract_complete_sentences(&growing_text);
        let mut all_sentences = sentences;
        if !remainder.trim().is_empty() {
            all_sentences.push(remainder.trim().to_string());
        }

        let current_time = total_samples as f32 / 16000.0;
        for sentence in all_sentences {
            segment_count += 1;
            full_transcript = if full_transcript.is_empty() {
                sentence.clone()
            } else {
                format!("{} {}", full_transcript, sentence)
            };

            let _ = subtitle_tx
                .send(SubtitleMessage {
                    session_id: session_id.clone(),
                    text: sentence,
                    is_final: true,
                    segment_index: segment_count - 1,
                    timestamp_ms: (current_time * 1000.0) as u64,
                })
                .await;
        }
    }

    // Wait for FFmpeg
    let _ = ffmpeg_handle.await;

    // Broadcast end
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
///
/// Splits at sentence boundaries (. ! ? followed by space + uppercase letter),
/// avoiding false splits on "19. November" or "Dr. Müller".
///
/// Returns (complete_sentences, remainder).
fn extract_complete_sentences(text: &str) -> (Vec<String>, String) {
    let sentences = split_sentences(text);

    if sentences.is_empty() {
        return (vec![], text.to_string());
    }

    if sentences.len() == 1 {
        // Check if the single segment ends with sentence punctuation
        let s = sentences[0].trim();
        if s.ends_with('.') || s.ends_with('!') || s.ends_with('?') {
            return (vec![s.to_string()], String::new());
        } else {
            return (vec![], s.to_string());
        }
    }

    // Multiple sentences: all but last are complete
    let last = sentences.last().unwrap().trim().to_string();
    let complete: Vec<String> = sentences[..sentences.len() - 1]
        .iter()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    // Check if last also ends with punctuation
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
