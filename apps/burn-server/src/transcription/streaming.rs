//! Streaming prefix management and greedy token decoding.
//!
//! Streaming Prefix (38 tokens):
//!   - 1 BOS token
//!   - 37 streaming pad tokens
//!   - Pre-allocated KV cache
//!
//! Greedy Decoding:
//!   - Simple argmax token selection (no beam search, no temperature)
//!   - Sentence boundary detection with regex
//!   - Context window rotation after ~200 commits (~100s audio)

// TODO: Implement streaming prefix and decoding logic
// Port from voxtral-mini-realtime-rs
