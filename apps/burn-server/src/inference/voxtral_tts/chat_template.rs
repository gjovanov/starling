//! Voxtral-4B-TTS chat template (voice-only path).
//!
//! Builds the AR LLM input token sequence from `(text, voice_n_audio_tokens)`,
//! mirroring `mistral_common.tokens.tokenizers.instruct.encode_speech_request`
//! byte-for-byte for the no-`ref_audio` case.
//!
//! ```text
//! [BOS] [BEGIN_AUDIO] [AUDIO]×N_voice
//!     [TEXT_TO_AUDIO] BPE(text, no BOS/EOS) [AUDIO_TO_TEXT] [BEGIN_AUDIO]
//! ```
//!
//! `N_voice` comes from the loaded `VoiceEmbedding`'s row count (matches
//! `tekken.json::audio.voice_num_audio_tokens[voice]`). The voice-cloning
//! variant (`ref_audio` instead of preset) is out of scope for Phase 2-H.

use anyhow::Result;

use super::tokenizer::TekkenTokenizer;

/// Build the full prompt token sequence for a `(text, voice)` request.
///
/// Pure function — no I/O, no allocations beyond the returned `Vec`.
/// `voice_n_audio_tokens` must equal the voice's per-`tekken.json` slot
/// count (use `VoiceEmbedding::n_rows()`).
pub fn build_voice_prompt(
    tk: &TekkenTokenizer,
    text: &str,
    voice_n_audio_tokens: usize,
) -> Result<Vec<u32>> {
    let s = &tk.specials;
    let text_ids = tk.encode(text)?;
    let total = 1 + 1 + voice_n_audio_tokens + 1 + text_ids.len() + 1 + 1;
    let mut out = Vec::with_capacity(total);
    out.push(s.bos);
    out.push(s.begin_audio);
    out.extend(std::iter::repeat(s.audio).take(voice_n_audio_tokens));
    out.push(s.text_to_audio);
    out.extend(text_ids);
    out.push(s.audio_to_text);
    out.push(s.begin_audio);
    Ok(out)
}

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

    fn load_tk() -> Option<TekkenTokenizer> {
        let p = tekken_path();
        if !p.exists() {
            eprintln!("skipping: {} absent", p.display());
            return None;
        }
        Some(TekkenTokenizer::load(&p).expect("load tekken"))
    }

    /// For each captured `prompt_ids_*.json`, building the prompt locally
    /// must match the upstream sequence byte-exact.
    #[test]
    fn matches_upstream_prompt_ids() {
        let Some(tk) = load_tk() else { return };
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
            let voice_n = v["voice_n_audio_tokens"].as_u64().unwrap() as usize;
            let expected: Vec<u32> = v["prompt_ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as u32)
                .collect();
            let got = build_voice_prompt(&tk, text, voice_n).unwrap();
            assert_eq!(
                got.len(),
                expected.len(),
                "fixture {f}: length mismatch (got {}, expected {})",
                got.len(),
                expected.len()
            );
            assert_eq!(got, expected, "fixture {f}: byte-exact mismatch");
        }
    }

    /// Sanity: structural shape is `[BOS, BEGIN_AUDIO, AUDIO×N, TEXT_TO_AUDIO, …, AUDIO_TO_TEXT, BEGIN_AUDIO]`.
    #[test]
    fn structural_shape() {
        let Some(tk) = load_tk() else { return };
        let n = 7;
        let ids = build_voice_prompt(&tk, "hi", n).unwrap();
        let s = &tk.specials;
        assert_eq!(ids[0], s.bos);
        assert_eq!(ids[1], s.begin_audio);
        for i in 0..n {
            assert_eq!(ids[2 + i], s.audio, "position {} should be AUDIO", 2 + i);
        }
        assert_eq!(ids[2 + n], s.text_to_audio);
        // Tail: AUDIO_TO_TEXT, BEGIN_AUDIO.
        assert_eq!(ids[ids.len() - 2], s.audio_to_text);
        assert_eq!(ids[ids.len() - 1], s.begin_audio);
    }
}
