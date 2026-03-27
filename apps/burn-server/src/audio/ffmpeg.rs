//! FFmpeg subprocess for decoding media files to PCM 16kHz mono.
//!
//! Spawns FFmpeg as a child process, reads stdout as raw PCM s16le at 16kHz.
//! Yields chunks of f32 samples in [-1.0, 1.0].
//!
//! Adapted from parakeet-rs audio_pipeline.rs.

use std::io::Read;
use std::path::Path;
use std::process::{Child, Command, Stdio};
use tokio::sync::mpsc;

/// Spawn FFmpeg to decode a media file to PCM s16le at 16kHz mono.
///
/// Uses `-re` flag for real-time pacing (simulates playback speed).
pub fn spawn_ffmpeg(path: &Path) -> Result<Child, std::io::Error> {
    Command::new("ffmpeg")
        .args([
            "-re",
            "-i",
            path.to_str().unwrap_or(""),
            "-f",
            "s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-loglevel",
            "error",
            "-",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
}

/// Spawn FFmpeg without real-time pacing (process as fast as possible).
pub fn spawn_ffmpeg_fast(path: &Path) -> Result<Child, std::io::Error> {
    Command::new("ffmpeg")
        .args([
            "-i",
            path.to_str().unwrap_or(""),
            "-f",
            "s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-loglevel",
            "error",
            "-",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
}

/// Read PCM from FFmpeg stdout and send f32 sample chunks via channel.
///
/// Reads in `chunk_duration_ms` chunks (default 500ms = 8000 samples at 16kHz).
/// Converts s16le bytes to f32 normalized to [-1.0, 1.0].
pub fn read_pcm_chunks(
    mut ffmpeg: Child,
    chunk_duration_ms: u32,
    tx: mpsc::Sender<Vec<f32>>,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        let samples_per_chunk = (16000 * chunk_duration_ms / 1000) as usize;
        let bytes_per_chunk = samples_per_chunk * 2; // s16le = 2 bytes per sample

        let mut stdout = match ffmpeg.stdout.take() {
            Some(s) => s,
            None => {
                eprintln!("[FFmpeg] Failed to get stdout");
                return;
            }
        };

        let mut byte_buffer = vec![0u8; bytes_per_chunk];

        loop {
            match stdout.read_exact(&mut byte_buffer) {
                Ok(_) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                    // End of audio
                    break;
                }
                Err(e) => {
                    eprintln!("[FFmpeg] Read error: {}", e);
                    break;
                }
            }

            // Convert s16le to f32 normalized
            let samples: Vec<f32> = byte_buffer
                .chunks(2)
                .map(|chunk| {
                    let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
                    sample as f32 / 32768.0
                })
                .collect();

            if tx.blocking_send(samples).is_err() {
                // Receiver dropped
                break;
            }
        }

        // Cleanup
        let _ = ffmpeg.kill();
        let _ = ffmpeg.wait();
    })
}

/// Convenience: decode an entire media file to f32 samples (blocking).
pub fn decode_file_to_samples(path: &Path) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let mut ffmpeg = spawn_ffmpeg_fast(path)?;
    let mut stdout = ffmpeg
        .stdout
        .take()
        .ok_or("Failed to get FFmpeg stdout")?;

    let mut all_bytes = Vec::new();
    stdout.read_to_end(&mut all_bytes)?;

    let _ = ffmpeg.wait();

    let samples: Vec<f32> = all_bytes
        .chunks(2)
        .map(|chunk| {
            if chunk.len() == 2 {
                let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
                sample as f32 / 32768.0
            } else {
                0.0
            }
        })
        .collect();

    Ok(samples)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_s16le_conversion() {
        // Max positive i16
        let bytes = 32767i16.to_le_bytes();
        let sample = i16::from_le_bytes(bytes) as f32 / 32768.0;
        assert!((sample - 0.99997).abs() < 0.001);

        // Zero
        let bytes = 0i16.to_le_bytes();
        let sample = i16::from_le_bytes(bytes) as f32 / 32768.0;
        assert_eq!(sample, 0.0);

        // Max negative i16
        let bytes = (-32768i16).to_le_bytes();
        let sample = i16::from_le_bytes(bytes) as f32 / 32768.0;
        assert_eq!(sample, -1.0);
    }
}
