//! Voice embedding loader.
//!
//! Each voice ships as a `[N, 3072]` BF16 tensor where `N` varies per
//! voice (67–218 rows in the published Voxtral-4B-TTS-2603 set). Phase
//! 2-A's `convert_voice_embeddings_to_safetensors.py` writes a
//! parallel `.safetensors` for every `.pt`; we mmap the safetensors
//! file (no .pt / pickle parser needed in Rust).

use anyhow::{anyhow, Context, Result};
use candle_core::{DType, Device, Tensor};
use std::path::{Path, PathBuf};

/// One loaded voice — opaque to callers, just hold the tensor + a
/// label for diagnostic output.
pub struct VoiceEmbedding {
    pub name: String,
    pub tensor: Tensor,
}

impl VoiceEmbedding {
    /// Load `<dir>/<voice>.safetensors`. The file must contain a
    /// single tensor named `"voice"` with shape `[N, dim]`.
    pub fn load_from_dir(dir: &Path, voice: &str, dim: usize, device: &Device, dtype: DType) -> Result<Self> {
        let path = dir.join(format!("{voice}.safetensors"));
        Self::load_from_path(&path, dim, device, dtype).map(|mut v| {
            v.name = voice.to_string();
            v
        })
    }

    pub fn load_from_path(path: &Path, dim: usize, device: &Device, dtype: DType) -> Result<Self> {
        let bytes =
            std::fs::read(path).with_context(|| format!("reading voice file {}", path.display()))?;
        let st = safetensors::SafeTensors::deserialize(&bytes)
            .map_err(|e| anyhow!("parsing voice safetensors {}: {e}", path.display()))?;
        let view = st
            .tensor("voice")
            .map_err(|e| anyhow!("voice file {} missing 'voice' tensor: {e}", path.display()))?;
        let dims: Vec<usize> = view.shape().to_vec();
        if dims.len() != 2 || dims[1] != dim {
            return Err(anyhow!(
                "voice file {} has shape {dims:?}, expected [_, {dim}]",
                path.display()
            ));
        }
        let view_dtype = match view.dtype() {
            safetensors::Dtype::BF16 => DType::BF16,
            safetensors::Dtype::F32 => DType::F32,
            safetensors::Dtype::F16 => DType::F16,
            other => return Err(anyhow!("unsupported voice dtype {other:?}")),
        };
        let tensor = Tensor::from_raw_buffer(view.data(), view_dtype, &dims, device)?
            .to_dtype(dtype)?;
        Ok(Self {
            name: path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("voice")
                .to_string(),
            tensor,
        })
    }

    /// Number of audio-token slots this voice occupies in the prompt.
    pub fn n_rows(&self) -> Result<usize> {
        Ok(self.tensor.dim(0)?)
    }

    pub fn dim(&self) -> Result<usize> {
        Ok(self.tensor.dim(1)?)
    }
}

/// Default location for shipped voices: `models/cache/tts/voice_embedding/`.
pub fn default_voice_dir() -> PathBuf {
    PathBuf::from("/home/gjovanov/gjovanov/starling/models/cache/tts/voice_embedding")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn voice_dir() -> PathBuf {
        std::env::var_os("STARLING_TTS_VOICE_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(default_voice_dir)
    }

    #[test]
    fn loads_de_male_when_present() {
        let dir = voice_dir();
        let path = dir.join("de_male.safetensors");
        if !path.exists() {
            eprintln!("skipping: {} not present", path.display());
            return;
        }
        let v = VoiceEmbedding::load_from_dir(&dir, "de_male", 3072, &Device::Cpu, DType::F32).unwrap();
        assert_eq!(v.dim().unwrap(), 3072);
        assert_eq!(v.n_rows().unwrap(), 163);
        assert_eq!(v.name, "de_male");
    }

    #[test]
    fn rejects_wrong_dim() {
        let dir = voice_dir();
        let path = dir.join("de_male.safetensors");
        if !path.exists() {
            return;
        }
        let res =
            VoiceEmbedding::load_from_dir(&dir, "de_male", 1024, &Device::Cpu, DType::F32);
        assert!(res.is_err(), "loading with wrong dim should fail");
    }
}
