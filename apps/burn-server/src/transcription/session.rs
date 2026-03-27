//! Session runner: orchestrates audio → mel → inference → text → broadcast.
//!
//! Lifecycle:
//!   1. Wait for client_ready signal
//!   2. Spawn FFmpeg to decode media file to PCM 16kHz
//!   3. Batch audio in 0.5s intervals
//!   4. Compute mel spectrogram
//!   5. Run inference (Q4 or BF16)
//!   6. Split text into sentences
//!   7. Broadcast SubtitleMessage to WebSocket subscribers
//!
//! Adapted from both parakeet-rs transcription/mod.rs and vllm-server session_runner.py.

// TODO: Implement session runner
