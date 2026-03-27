//! WASM bindings for browser-based Voxtral inference.
//!
//! When compiled with `--features wasm`, this module exports:
//!   - `load_model(chunks_url: &str)` — Fetch and assemble Q4 GGUF weight chunks
//!   - `transcribe(audio: Float32Array)` — Run inference on audio samples
//!   - `get_progress()` — Return model loading progress (0.0 - 1.0)
//!
//! Browser constraints handled:
//!   - 2 GB ArrayBuffer limit → sharded cursor reads
//!   - 4 GB address space → two-phase loading
//!   - 1.5 GiB embedding table → Q4 on GPU + CPU byte lookups
//!   - No sync GPU readback → into_data_async().await
//!   - 256 workgroup invocation limit → patched CubeCL

// TODO: Implement WASM bindings
// Port from voxtral-mini-realtime-rs src/web/
