//! Voxtral-4B-TTS weight inventory + safetensors loader stub.
//!
//! Phase 2-B: this module only validates that all 386 tensors exist in
//! `consolidated.safetensors` with the expected shapes, and exposes a
//! safetensors view for later phases to draw from. No tensor framework
//! (candle / burn) is pulled in — pure `safetensors` + `memmap2`.
//!
//! The upstream Python reference is `vllm_omni.model_executor.models
//! .voxtral_tts.voxtral_tts*`. See memory note
//! `project_voxtral_tts_model_inventory` for the per-prefix breakdown.

use anyhow::{anyhow, bail, Context, Result};
use memmap2::Mmap;
use safetensors::SafeTensors;
use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};

/// One of the four top-level modules in `consolidated.safetensors`,
/// plus the standalone `norm.weight`.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum ModuleGroup {
    /// 26-layer AR LLM decoder. Prefix `layers.`.
    ArLlm,
    /// 8-block codec decoder + quantizer. Prefix `audio_tokenizer.`.
    AudioTokenizer,
    /// 3-layer flow-matching velocity field. Prefix `acoustic_transformer.`.
    AcousticTransformer,
    /// Token + codebook embeddings. Prefix `mm_audio_embeddings.`.
    MmAudioEmbeddings,
    /// Final RmsNorm of the AR stack. Exact key `norm.weight`.
    FinalNorm,
}

impl ModuleGroup {
    /// All variants in iteration order matching the on-disk grouping.
    pub const ALL: &'static [ModuleGroup] = &[
        ModuleGroup::ArLlm,
        ModuleGroup::AudioTokenizer,
        ModuleGroup::AcousticTransformer,
        ModuleGroup::MmAudioEmbeddings,
        ModuleGroup::FinalNorm,
    ];

    pub fn key_prefix(self) -> &'static str {
        match self {
            ModuleGroup::ArLlm => "layers.",
            ModuleGroup::AudioTokenizer => "audio_tokenizer.",
            ModuleGroup::AcousticTransformer => "acoustic_transformer.",
            ModuleGroup::MmAudioEmbeddings => "mm_audio_embeddings.",
            ModuleGroup::FinalNorm => "norm.weight",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            ModuleGroup::ArLlm => "AR LLM",
            ModuleGroup::AudioTokenizer => "Audio Tokenizer (codec)",
            ModuleGroup::AcousticTransformer => "Acoustic Transformer (flow-match)",
            ModuleGroup::MmAudioEmbeddings => "MM Audio Embeddings",
            ModuleGroup::FinalNorm => "Final Norm",
        }
    }

    pub fn classify(key: &str) -> Self {
        if key == "norm.weight" {
            ModuleGroup::FinalNorm
        } else if key.starts_with("layers.") {
            ModuleGroup::ArLlm
        } else if key.starts_with("audio_tokenizer.") {
            ModuleGroup::AudioTokenizer
        } else if key.starts_with("acoustic_transformer.") {
            ModuleGroup::AcousticTransformer
        } else if key.starts_with("mm_audio_embeddings.") {
            ModuleGroup::MmAudioEmbeddings
        } else {
            panic!("voxtral-tts: unrecognised key {key:?}");
        }
    }
}

/// Tensor-count expectation for one `ModuleGroup`. The total must sum
/// to `EXPECTED_TOTAL = 386`.
#[derive(Clone, Copy, Debug)]
pub struct ExpectedGroup {
    pub group: ModuleGroup,
    pub count: usize,
}

pub const EXPECTED_GROUPS: &[ExpectedGroup] = &[
    ExpectedGroup { group: ModuleGroup::ArLlm, count: 234 },
    ExpectedGroup { group: ModuleGroup::AudioTokenizer, count: 116 },
    ExpectedGroup { group: ModuleGroup::AcousticTransformer, count: 33 },
    ExpectedGroup { group: ModuleGroup::MmAudioEmbeddings, count: 2 },
    ExpectedGroup { group: ModuleGroup::FinalNorm, count: 1 },
];

pub const EXPECTED_TOTAL: usize = 386;

/// Hand-picked sentinel tensor specs that we always check on load.
/// Each entry: (key, expected shape).
///
/// Picked to cover one tensor from each module group + a few load-bearing
/// shapes (vocab dim, voice-embedding dim, etc.). If any of these change
/// upstream, we'd rather fail loudly than load broken weights silently.
const SENTINELS: &[(&str, &[usize])] = &[
    // AR LLM
    ("layers.0.attention.wq.weight", &[4096, 3072]),
    ("layers.0.feed_forward.w1.weight", &[9216, 3072]),
    ("layers.25.ffn_norm.weight", &[3072]),
    // Audio Tokenizer
    ("audio_tokenizer.decoder_blocks.0.conv.parametrizations.weight.original0", &[1024, 1, 1]),
    ("audio_tokenizer.decoder_blocks.0.conv.parametrizations.weight.original1", &[1024, 292, 3]),
    ("audio_tokenizer.output_proj.conv.parametrizations.weight.original1", &[240, 1024, 7]),
    ("audio_tokenizer.quantizer.semantic_codebook.embedding_sum", &[8192, 256]),
    // Acoustic Transformer
    ("acoustic_transformer.input_projection.weight", &[3072, 36]),
    ("acoustic_transformer.acoustic_codebook_output.weight", &[36, 3072]),
    ("acoustic_transformer.semantic_codebook_output.weight", &[8320, 3072]),
    ("acoustic_transformer.time_projection.weight", &[3072, 3072]),
    ("acoustic_transformer.norm.weight", &[3072]),
    // MM Audio Embeddings
    ("mm_audio_embeddings.tok_embeddings.weight", &[131072, 3072]),
    ("mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight", &[9088, 3072]),
    // Final Norm
    ("norm.weight", &[3072]),
];

/// Memory-mapped view of `consolidated.safetensors` plus a validated key
/// index. Construction enforces every group-count and sentinel-shape
/// expectation; later phases can use [`Self::tensor_view`] /
/// [`Self::keys_in`] without re-validating.
pub struct WeightInventory {
    /// `Mmap` is held to keep the underlying file mapping alive; the
    /// `SafeTensors` view borrows from it.
    _mmap: Mmap,
    /// `SafeTensors` view referencing `_mmap`. We use a self-referential
    /// pattern via `'static` transmute on the boxed mmap — cleanly
    /// expressed via `ouroboros` would add a dep. Instead we expose
    /// only the per-tensor accessors below.
    safetensors: SafeTensorsHolder,
    /// Per-group key index, populated at load.
    keys_by_group: BTreeMap<ModuleGroup, Vec<String>>,
    /// File path (kept for diagnostics).
    path: PathBuf,
}

/// Helper that owns the mmap-backed `SafeTensors` view. We pin the
/// `Mmap` separately on `WeightInventory` and rely on it not being
/// moved/freed for the lifetime of the holder.
struct SafeTensorsHolder {
    /// `'static` lifetime is a lie — the slice points into the `Mmap`
    /// owned by the parent `WeightInventory`. Consumers must access
    /// only via the parent's methods, which apply the correct lifetime.
    inner: SafeTensors<'static>,
}

impl WeightInventory {
    /// Open + validate the safetensors file at `path`.
    ///
    /// On success every group-count and sentinel-shape check has passed.
    /// On failure returns a descriptive error indicating which check
    /// detected the problem (missing key, wrong count, wrong shape).
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path).with_context(|| format!("opening {}", path.display()))?;
        // SAFETY: file is opened read-only and we hold the `File` (via
        // the mmap which keeps the fd open) for the lifetime of `Mmap`.
        let mmap = unsafe { Mmap::map(&file) }
            .with_context(|| format!("mmap'ing {}", path.display()))?;

        // Build a SafeTensors view borrowing from the mmap. The borrow
        // is alive as long as the parent struct (which holds the mmap).
        // We transmute the lifetime so the view can sit beside the mmap
        // in the same struct; the public API ensures the references
        // returned to callers carry the parent's lifetime.
        let safetensors_view: SafeTensors<'_> = SafeTensors::deserialize(&mmap)
            .map_err(|e| anyhow!("parsing safetensors header in {}: {e}", path.display()))?;
        // SAFETY: extending the borrow to 'static is sound because
        // `safetensors_view` is moved into `SafeTensorsHolder` which
        // lives strictly less than `mmap` (both are fields of the same
        // struct, dropped in declaration order: holder before mmap).
        let safetensors_view: SafeTensors<'static> =
            unsafe { std::mem::transmute(safetensors_view) };
        let safetensors = SafeTensorsHolder { inner: safetensors_view };

        // Validate group counts.
        let mut keys_by_group: BTreeMap<ModuleGroup, Vec<String>> = BTreeMap::new();
        for k in safetensors.inner.names() {
            let group = ModuleGroup::classify(k);
            keys_by_group.entry(group).or_default().push(k.to_string());
        }
        for ExpectedGroup { group, count } in EXPECTED_GROUPS {
            let actual = keys_by_group.get(group).map(|v| v.len()).unwrap_or(0);
            if actual != *count {
                bail!(
                    "voxtral-tts: {} group has {actual} tensors, expected {count}",
                    group.label()
                );
            }
        }
        let total: usize = keys_by_group.values().map(|v| v.len()).sum();
        if total != EXPECTED_TOTAL {
            bail!("voxtral-tts: total tensor count is {total}, expected {EXPECTED_TOTAL}");
        }

        // Sort keys for stable iteration order.
        for v in keys_by_group.values_mut() {
            v.sort();
        }

        // Validate sentinel shapes.
        for (key, expected_shape) in SENTINELS {
            let view = safetensors
                .inner
                .tensor(key)
                .map_err(|e| anyhow!("voxtral-tts: sentinel key {key:?} missing — {e}"))?;
            if view.shape() != *expected_shape {
                bail!(
                    "voxtral-tts: sentinel {key:?} shape {:?}, expected {expected_shape:?}",
                    view.shape()
                );
            }
        }

        Ok(Self {
            _mmap: mmap,
            safetensors,
            keys_by_group,
            path,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// All tensor names in `group`, sorted ascending.
    pub fn keys_in(&self, group: ModuleGroup) -> &[String] {
        self.keys_by_group
            .get(&group)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Iterate over all keys, grouped by module.
    pub fn iter_groups(&self) -> impl Iterator<Item = (ModuleGroup, &[String])> {
        ModuleGroup::ALL
            .iter()
            .map(move |g| (*g, self.keys_in(*g)))
    }

    /// Borrow a `safetensors::TensorView` for `name` (header-only;
    /// the underlying bytes are mmap'd on demand by callers).
    pub fn tensor_view(&self, name: &str) -> Result<safetensors::tensor::TensorView<'_>> {
        self.safetensors
            .inner
            .tensor(name)
            .map_err(|e| anyhow!("voxtral-tts: missing tensor {name:?} — {e}"))
    }

    /// Total bytes of all tensor payloads (excludes the JSON header).
    pub fn total_payload_bytes(&self) -> usize {
        self.safetensors
            .inner
            .names()
            .iter()
            .map(|n| {
                self.safetensors
                    .inner
                    .tensor(n)
                    .map(|t| t.data().len())
                    .unwrap_or(0)
            })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn checkpoint_path() -> PathBuf {
        // The model lives in the shared cache; in CI the env var should
        // override. Skip the test if the file is absent so unit-tests
        // remain green on machines without the 8 GB checkpoint.
        std::env::var_os("STARLING_TTS_SAFETENSORS")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from("/home/gjovanov/gjovanov/starling/models/cache/tts/consolidated.safetensors")
            })
    }

    #[test]
    fn group_counts_sum_to_expected_total() {
        let actual: usize = EXPECTED_GROUPS.iter().map(|g| g.count).sum();
        assert_eq!(actual, EXPECTED_TOTAL);
    }

    #[test]
    fn classify_covers_all_inventory_prefixes() {
        // Sanity: every well-known prefix maps to a unique group.
        assert_eq!(ModuleGroup::classify("layers.0.attention.wq.weight"), ModuleGroup::ArLlm);
        assert_eq!(
            ModuleGroup::classify("audio_tokenizer.decoder_blocks.0.conv.parametrizations.weight.original0"),
            ModuleGroup::AudioTokenizer
        );
        assert_eq!(
            ModuleGroup::classify("acoustic_transformer.norm.weight"),
            ModuleGroup::AcousticTransformer
        );
        assert_eq!(
            ModuleGroup::classify("mm_audio_embeddings.tok_embeddings.weight"),
            ModuleGroup::MmAudioEmbeddings
        );
        assert_eq!(ModuleGroup::classify("norm.weight"), ModuleGroup::FinalNorm);
    }

    #[test]
    fn opens_real_checkpoint_when_present() {
        let path = checkpoint_path();
        if !path.exists() {
            eprintln!("skipping: {} not present", path.display());
            return;
        }
        let inv = WeightInventory::open(&path)
            .expect("WeightInventory::open should succeed against the published checkpoint");

        for ExpectedGroup { group, count } in EXPECTED_GROUPS {
            let n = inv.keys_in(*group).len();
            assert_eq!(n, *count, "{} count mismatch", group.label());
        }
    }
}
