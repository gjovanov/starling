use clap::Parser;
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum Quantization {
    Q4,
    Bf16,
}

impl std::fmt::Display for Quantization {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Quantization::Q4 => write!(f, "Q4"),
            Quantization::Bf16 => write!(f, "BF16"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum GpuBackend {
    Wgpu,
    #[cfg(feature = "cuda")]
    Cuda,
    #[cfg(feature = "candle")]
    Candle,
    #[cfg(feature = "candle-native")]
    CandleNative,
    #[cfg(feature = "candle-native-flash")]
    CandleNativeFlash,
    #[cfg(feature = "candle-cpu")]
    CandleCpu,
}

impl std::fmt::Display for GpuBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuBackend::Wgpu => write!(f, "WGPU"),
            #[cfg(feature = "cuda")]
            GpuBackend::Cuda => write!(f, "CUDA"),
            #[cfg(feature = "candle")]
            GpuBackend::Candle => write!(f, "Candle"),
            #[cfg(feature = "candle-native")]
            GpuBackend::CandleNative => write!(f, "CandleNative"),
            #[cfg(feature = "candle-native-flash")]
            GpuBackend::CandleNativeFlash => write!(f, "CandleNativeFlash"),
            #[cfg(feature = "candle-cpu")]
            GpuBackend::CandleCpu => write!(f, "CandleCpu"),
        }
    }
}

#[derive(Parser, Debug)]
#[command(name = "burn-server")]
#[command(about = "Real-time Voxtral ASR server using Burn ML framework")]
pub struct Config {
    /// HTTP/WebSocket server port
    #[arg(long, env = "BURN_PORT", default_value = "8091")]
    pub port: u16,

    /// Model quantization to use
    #[arg(long, env = "BURN_QUANT", value_enum, default_value = "q4")]
    pub quant: Quantization,

    /// GPU backend (wgpu for DZN/WebGPU, cuda for native NVIDIA)
    #[arg(long, env = "BURN_BACKEND", value_enum, default_value = "wgpu")]
    pub backend: GpuBackend,

    /// Path to shared models directory
    #[arg(long, env = "STARLING_MODELS_DIR", default_value = "../../models/cache")]
    pub models_dir: PathBuf,

    /// Path to frontend directory
    #[arg(long, env = "BURN_FRONTEND_PATH", default_value = "../../frontend")]
    pub frontend: PathBuf,

    /// Media directory for audio files
    #[arg(long, env = "BURN_MEDIA_DIR", default_value = "../../media")]
    pub media_dir: PathBuf,

    /// Public IP address for WebRTC ICE candidates
    #[arg(long, env = "PUBLIC_IP")]
    pub public_ip: Option<String>,

    /// Maximum concurrent sessions
    #[arg(long, env = "MAX_CONCURRENT_SESSIONS", default_value = "10")]
    pub max_sessions: usize,

    /// TURN server for NAT traversal
    #[arg(long, env = "TURN_SERVER")]
    pub turn_server: Option<String>,

    /// TURN username
    #[arg(long, env = "TURN_USERNAME")]
    pub turn_username: Option<String>,

    /// TURN password
    #[arg(long, env = "TURN_PASSWORD")]
    pub turn_password: Option<String>,

    /// TURN shared secret for ephemeral credentials
    #[arg(long, env = "TURN_SHARED_SECRET")]
    pub turn_shared_secret: Option<String>,

    /// Force TURN relay mode
    #[arg(long, env = "FORCE_RELAY", default_value = "false")]
    pub force_relay: bool,

    /// Directory where saved TTS WAV files are written.
    /// Used by the `voxtral-tts` feature's save-to-disk path.
    #[arg(long, env = "STARLING_TTS_OUTPUT_DIR", default_value = "./tts_output")]
    pub tts_output_dir: PathBuf,

    /// Maximum wall-clock seconds a single long-form synthesis may run
    /// before the streaming response is closed. Mirrors vllm-server's
    /// `tts_long_max_secs` (default 180 s).
    #[arg(long, env = "BURN_TTS_LONG_MAX_SECS", default_value = "180")]
    pub tts_long_max_secs: u64,

    /// Drop the TTS pipeline after this many seconds of inactivity,
    /// freeing ~12 GiB of resident weights. The next `/api/tts/*`
    /// hit transparently re-loads. Set to `0` to disable. Mirrors
    /// vllm-server's `tts_idle_unload_secs` (default 600 s).
    #[arg(long, env = "BURN_TTS_IDLE_UNLOAD_SECS", default_value = "600")]
    pub tts_idle_unload_secs: u64,

    /// Directory where uploaded reference-audio voice clones live
    /// (`<id>.wav` + `<id>.json` + `_audit.log`). Mirrors vllm-server's
    /// `tts_voice_refs_dir`.
    #[arg(long, env = "BURN_TTS_VOICE_REFS_DIR", default_value = "./tts_voice_refs")]
    pub tts_voice_refs_dir: PathBuf,

    /// Maximum number of voice clones a single server may host. New
    /// uploads beyond this fail with a clear error directing the user
    /// to delete an existing one first.
    #[arg(long, env = "BURN_TTS_MAX_VOICE_REFS", default_value = "50")]
    pub tts_max_voice_refs: usize,

    /// Hard cap on the size of an uploaded reference audio file in
    /// bytes. 5 MB ≈ 30 s of 24 kHz mono PCM16 + extension headroom.
    #[arg(long, env = "BURN_TTS_MAX_REF_AUDIO_BYTES", default_value = "5000000")]
    pub tts_max_ref_audio_bytes: usize,

    /// Minimum reference audio duration in seconds. Below this Voxtral
    /// can't extract a stable speaker embedding.
    #[arg(long, env = "BURN_TTS_MIN_REF_DURATION_SECS", default_value = "5.0")]
    pub tts_min_ref_duration_secs: f64,

    /// Maximum reference audio duration in seconds. Above this is
    /// wasteful and produces no quality gain.
    #[arg(long, env = "BURN_TTS_MAX_REF_DURATION_SECS", default_value = "30.0")]
    pub tts_max_ref_duration_secs: f64,

    /// Require the multipart upload to include
    /// `permission_confirmed=true`. Server-side enforced; the
    /// frontend has its own checkbox but a Playwright test bypasses
    /// the client gate to verify the server gate.
    #[arg(long, env = "BURN_TTS_REQUIRE_PERMISSION", default_value = "true")]
    pub tts_require_permission: bool,
}

impl Config {
    /// Returns the path to BF16 SafeTensors model directory
    pub fn bf16_model_path(&self) -> PathBuf {
        self.models_dir.join("bf16")
    }

    /// Returns the path to Q4 GGUF model file
    pub fn q4_model_path(&self) -> PathBuf {
        self.models_dir.join("q4").join("voxtral-q4.gguf")
    }

    /// Returns the path to the Tekken tokenizer
    pub fn tokenizer_path(&self) -> PathBuf {
        // Try shared tokenizer first, fall back to Q4 dir
        let shared = self.models_dir.join("tokenizer").join("tekken.json");
        if shared.exists() {
            shared
        } else {
            self.models_dir.join("q4").join("tekken.json")
        }
    }

    /// Returns the active model path based on selected quantization
    pub fn active_model_path(&self) -> PathBuf {
        match self.quant {
            Quantization::Q4 => self.q4_model_path(),
            Quantization::Bf16 => self.bf16_model_path(),
        }
    }
}
