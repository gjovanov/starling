//! Mistral Tekken BPE loader for the Voxtral-4B-TTS chat template.
//!
//! Phase 2-H.2 scope: convert `models/cache/tts/tekken.json` (Mistral's own
//! BPE format — base64-encoded `token_bytes` per rank, plus a `pattern`
//! regex and named special tokens) into a runtime-equivalent HF
//! `tokenizers::Tokenizer`. Adapter pattern lifted from `mistral.rs`
//! (`mistralrs-core/src/utils/tokenizer.rs::load_tekken_tokenizer`),
//! adapted for our smaller-surface needs:
//!
//! - We expose only `encode(text) -> Vec<u32>` (no special tokens added),
//!   matching the upstream call
//!   `instruct.tokenizer.encode(text, bos=False, eos=False)` in
//!   `mistral_common.tokens.tokenizers.instruct.encode_speech_request`.
//! - The named special-token IDs we need for the chat template are
//!   surfaced via a small constants struct so the chat-template builder
//!   doesn't have to re-parse tekken.json.
//!
//! Verification: `tests::matches_upstream_text_token_ids` round-trips the
//! text portion of every captured fixture in `test_data/tts_golden/
//! prompt_ids_*.json` and asserts byte-exact equality with upstream.
//!
//! See `project_voxtral_tts_prompt_format` memory note for the full
//! prompt structure this tokenizer plugs into.

use anyhow::{anyhow, Context, Result};
use base64::Engine;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

use tokenizers::{
    decoders::byte_level::ByteLevel as ByteLevelDecoder,
    models::bpe::BpeBuilder,
    pre_tokenizers::{
        byte_level::ByteLevel,
        sequence::Sequence,
        split::{Split, SplitPattern},
        PreTokenizerWrapper,
    },
    tokenizer::{self, normalizer::SplitDelimiterBehavior, Tokenizer},
};

// ── tekken.json schema ──────────────────────────────────────────────

#[derive(Deserialize)]
struct TekkenVocabEntry {
    rank: usize,
    token_bytes: String,
}

#[derive(Deserialize)]
struct TekkenSpecialToken {
    rank: usize,
    token_str: String,
}

#[derive(Deserialize)]
struct TekkenConfig {
    pattern: String,
    default_vocab_size: usize,
    default_num_special_tokens: usize,
}

#[derive(Deserialize)]
struct TekkenFile {
    config: TekkenConfig,
    vocab: Vec<TekkenVocabEntry>,
    #[serde(default)]
    special_tokens: Vec<TekkenSpecialToken>,
}

// ── Special token IDs ───────────────────────────────────────────────

/// The five special-token IDs the Voxtral-TTS chat template needs.
/// Discovered at load time from tekken.json's `special_tokens` array.
#[derive(Debug, Clone, Copy)]
pub struct SpecialTokenIds {
    pub bos: u32,
    pub audio: u32,         // [AUDIO]
    pub begin_audio: u32,   // [BEGIN_AUDIO]
    pub text_to_audio: u32, // [NEXT_AUDIO_TEXT] — see prompt-format memo
    pub audio_to_text: u32, // [REPEAT_AUDIO_TEXT]
}

// ── Public Tekken tokenizer wrapper ─────────────────────────────────

pub struct TekkenTokenizer {
    inner: Tokenizer,
    pub specials: SpecialTokenIds,
}

impl TekkenTokenizer {
    /// Load `tekken.json` from disk and build an HF `Tokenizer` with the
    /// Tekken BPE merges, special-token table, and Mistral split pattern.
    pub fn load(path: &Path) -> Result<Self> {
        let raw = std::fs::read(path)
            .with_context(|| format!("reading tekken.json {}", path.display()))?;
        let tekken: TekkenFile = serde_json::from_slice(&raw)
            .with_context(|| format!("parsing tekken.json {}", path.display()))?;
        let inner = build_hf_tokenizer(&tekken)?;
        let specials = extract_specials(&tekken)?;
        Ok(Self { inner, specials })
    }

    /// Encode `text` to BPE token IDs **without** adding BOS/EOS or any
    /// special tokens. Matches upstream
    /// `instruct.tokenizer.encode(text, bos=False, eos=False)`.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let enc = self
            .inner
            .encode(text, false)
            .map_err(|e| anyhow!("BPE encode failed: {e}"))?;
        Ok(enc.get_ids().to_vec())
    }
}

// ── Construction internals ──────────────────────────────────────────

fn extract_specials(tekken: &TekkenFile) -> Result<SpecialTokenIds> {
    let mut by_str: HashMap<&str, u32> = HashMap::new();
    for st in &tekken.special_tokens {
        by_str.insert(st.token_str.as_str(), st.rank as u32);
    }
    let lookup = |name: &str| -> Result<u32> {
        by_str
            .get(name)
            .copied()
            .ok_or_else(|| anyhow!("tekken.json missing special token {name:?}"))
    };
    Ok(SpecialTokenIds {
        bos: lookup("<s>").or_else(|_| lookup("[BOS]")).unwrap_or(1),
        audio: lookup("[AUDIO]")?,
        begin_audio: lookup("[BEGIN_AUDIO]")?,
        // Mistral binds the semantic roles to these literal special-token
        // strings — see `mistral_common/tokens/tokenizers/base.py`
        // SpecialTokens enum: `text_to_audio = "[NEXT_AUDIO_TEXT]"` and
        // `audio_to_text = "[REPEAT_AUDIO_TEXT]"`.
        text_to_audio: lookup("[NEXT_AUDIO_TEXT]")?,
        audio_to_text: lookup("[REPEAT_AUDIO_TEXT]")?,
    })
}

fn build_hf_tokenizer(tekken: &TekkenFile) -> Result<Tokenizer> {
    let num_special = tekken.config.default_num_special_tokens;
    let inner_vocab_size = tekken.config.default_vocab_size - num_special;

    // bytes → rank for the BPE inner vocab.
    let mut bpe_ranks: HashMap<Vec<u8>, u32> = HashMap::new();
    for entry in &tekken.vocab {
        if entry.rank >= inner_vocab_size {
            continue;
        }
        let token_bytes = base64::engine::general_purpose::STANDARD
            .decode(&entry.token_bytes)
            .with_context(|| format!("base64-decoding vocab rank {}", entry.rank))?;
        bpe_ranks.insert(token_bytes, entry.rank as u32);
    }

    let mut vocab: HashMap<String, u32> = HashMap::new();
    for (token_bytes, rank) in &bpe_ranks {
        vocab.insert(token_bytes_to_string(token_bytes), *rank + num_special as u32);
    }

    // Reconstruct merges by finding, for each multi-byte token, the
    // bipartite split into two known sub-tokens, lowest combined rank
    // first. Identical to mistral.rs's loader.
    let mut merges: Vec<(Vec<u8>, Vec<u8>, u32)> = Vec::new();
    for (token, rank) in &bpe_ranks {
        if token.len() == 1 {
            continue;
        }
        let mut local = Vec::new();
        for index in 1..token.len() {
            let l = &token[..index];
            let r = &token[index..];
            if let (Some(&rl), Some(&rr)) = (bpe_ranks.get(l), bpe_ranks.get(r)) {
                local.push((l.to_vec(), r.to_vec(), *rank, rl, rr));
            }
        }
        local.sort_by_key(|(_, _, _, rl, rr)| (*rl, *rr));
        for (l, r, rk, _, _) in local {
            merges.push((l, r, rk));
        }
    }
    merges.sort_by_key(|(_, _, rank)| *rank);
    let merges: Vec<(String, String)> = merges
        .into_iter()
        .map(|(l, r, _)| (token_bytes_to_string(&l), token_bytes_to_string(&r)))
        .collect();

    // Assign IDs 0..num_special to the named special tokens (or
    // synthesised "<SPECIAL_n>" placeholders).
    let mut special_token_map: HashMap<usize, String> = HashMap::new();
    for st in &tekken.special_tokens {
        special_token_map.insert(st.rank, st.token_str.clone());
    }
    let special_token_names: Vec<String> = (0..num_special)
        .map(|id| {
            special_token_map
                .get(&id)
                .cloned()
                .unwrap_or_else(|| format!("<SPECIAL_{id}>"))
        })
        .collect();
    for (id, name) in special_token_names.iter().enumerate() {
        vocab.insert(name.clone(), id as u32);
    }

    let bpe_model = BpeBuilder::new()
        .vocab_and_merges(vocab, merges)
        .build()
        .map_err(|e| anyhow!("Failed to build BPE model: {e}"))?;

    let mut tokenizer = Tokenizer::new(bpe_model);

    let split = Split::new(
        SplitPattern::Regex(tekken.config.pattern.clone()),
        SplitDelimiterBehavior::Isolated,
        false,
    )
    .map_err(|e| anyhow!("split pre-tokenizer build: {e}"))?;
    let byte_level = ByteLevel::new(false, true, true);
    let pre_tokenizer = Sequence::new(vec![
        PreTokenizerWrapper::Split(split),
        PreTokenizerWrapper::ByteLevel(byte_level),
    ]);
    tokenizer.with_pre_tokenizer(Some(pre_tokenizer));

    let decoder = ByteLevelDecoder::new(true, false, false);
    tokenizer.with_decoder(Some(decoder));

    let added: Vec<tokenizer::AddedToken> = special_token_names
        .into_iter()
        .map(|name| tokenizer::AddedToken::from(name, true))
        .collect();
    tokenizer.add_special_tokens(&added);

    Ok(tokenizer)
}

// ── Byte-level encoder (matches Python tiktoken.bytes_to_unicode) ───

fn bytes_to_unicode() -> HashMap<u8, char> {
    let mut bs: Vec<u8> = vec![];
    bs.extend(b'!'..=b'~');
    bs.extend(0xA1u8..=0xACu8);
    bs.extend(0xAEu8..=0xFFu8);
    let mut cs: Vec<u32> = bs.iter().map(|&b| b as u32).collect();
    let mut n: u32 = 0;
    for b in 0u8..=255 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut map = HashMap::new();
    for (b, c) in bs.iter().zip(cs.iter()) {
        map.insert(*b, char::from_u32(*c).unwrap());
    }
    map
}

fn token_bytes_to_string(bytes: &[u8]) -> String {
    let map = bytes_to_unicode();
    bytes.iter().map(|&b| map[&b]).collect()
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn tekken_path() -> PathBuf {
        std::env::var_os("STARLING_TTS_TEKKEN")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from("/home/gjovanov/gjovanov/starling/models/cache/tts/tekken.json")
            })
    }

    fn fixture_dir() -> PathBuf {
        PathBuf::from("/home/gjovanov/gjovanov/starling/apps/burn-server/test_data/tts_golden")
    }

    fn load() -> Option<TekkenTokenizer> {
        let p = tekken_path();
        if !p.exists() {
            eprintln!("skipping: {} absent", p.display());
            return None;
        }
        Some(TekkenTokenizer::load(&p).expect("load tekken"))
    }

    #[test]
    fn loads_special_token_ids() {
        let Some(tk) = load() else { return };
        assert_eq!(tk.specials.bos, 1, "BOS must be 1 (Mistral default)");
        assert_eq!(tk.specials.audio, 24, "[AUDIO] must be 24");
        assert_eq!(tk.specials.begin_audio, 25, "[BEGIN_AUDIO] must be 25");
        assert_eq!(
            tk.specials.text_to_audio, 36,
            "[NEXT_AUDIO_TEXT] must be 36"
        );
        assert_eq!(
            tk.specials.audio_to_text, 35,
            "[REPEAT_AUDIO_TEXT] must be 35"
        );
    }

    /// For each captured prompt_ids_*.json fixture, our `encode(text)`
    /// must match the upstream `text_token_ids` byte-exact.
    #[test]
    fn matches_upstream_text_token_ids() {
        let Some(tk) = load() else { return };
        let fixtures = [
            "prompt_ids_de_short_de_male.json",
            "prompt_ids_en_short_neutral_male.json",
            "prompt_ids_en_short_neutral_female.json",
            "prompt_ids_fr_short_fr_female.json",
            "prompt_ids_en_medium_casual_male.json",
            "prompt_ids_en_long_cheerful_female.json",
        ];
        let dir = fixture_dir();
        if !dir.exists() {
            eprintln!("skipping: fixtures dir absent");
            return;
        }
        for f in fixtures {
            let p = dir.join(f);
            if !p.exists() {
                eprintln!("skipping {f}: not present");
                continue;
            }
            let raw = std::fs::read_to_string(&p).unwrap();
            let v: serde_json::Value = serde_json::from_str(&raw).unwrap();
            let text = v["text"].as_str().unwrap();
            let expected: Vec<u32> = v["text_token_ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as u32)
                .collect();
            let got = tk.encode(text).unwrap();
            assert_eq!(
                got, expected,
                "fixture {f}: text {text:?} → got {got:?}, expected {expected:?}"
            );
        }
    }
}
