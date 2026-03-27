//! FFmpeg subprocess for decoding media files to PCM 16kHz mono.
//!
//! Spawns FFmpeg as a child process, reads stdout as raw PCM s16le at 16kHz.
//! Yields 0.5s chunks as f32 samples in [-1.0, 1.0].
//!
//! Adapted from parakeet-rs audio_pipeline.rs and vllm-server ffmpeg_source.py.

// TODO: Implement FFmpeg subprocess audio extraction
// Fork from parakeet-rs src/bin/server/transcription/audio_pipeline.rs
