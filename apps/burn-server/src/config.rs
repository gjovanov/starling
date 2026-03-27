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
