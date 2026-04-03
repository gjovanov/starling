mod audio;
mod config;
mod inference;
mod server;
mod transcription;
mod web;

use axum::{
    extract::DefaultBodyLimit,
    routing::{delete, get, post},
    Router,
};
use clap::Parser;
use config::Config;
use server::state::AppState;
use std::sync::Arc;
use tower_http::{cors::CorsLayer, services::ServeDir};
use webrtc::api::{
    interceptor_registry::register_default_interceptors, media_engine::MediaEngine, APIBuilder,
};
use webrtc::ice::network_type::NetworkType;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = Config::parse();

    eprintln!("===========================================");
    eprintln!("  Burn Server — Voxtral ASR");
    eprintln!("===========================================");
    eprintln!("Port:        {}", config.port);
    eprintln!("Quant:       {}", config.quant);
    eprintln!("Backend:     {}", config.backend);
    eprintln!("Models:      {}", config.models_dir.display());
    eprintln!("Frontend:    {}", config.frontend.display());
    eprintln!("Media:       {}", config.media_dir.display());
    eprintln!("Max sessions: {}", config.max_sessions);
    eprintln!("===========================================");
    eprintln!();

    // Verify model exists
    let model_path = config.active_model_path();
    if !model_path.exists() {
        eprintln!("ERROR: Model not found at {}", model_path.display());
        eprintln!("       Run ../../models/download.sh first.");
        std::process::exit(1);
    }
    eprintln!("Model: {} ({})", model_path.display(), config.quant);

    // Create WebRTC API
    let mut media_engine = MediaEngine::default();
    media_engine.register_default_codecs()?;

    let mut registry = webrtc::interceptor::registry::Registry::new();
    registry = register_default_interceptors(registry, &mut media_engine)?;

    let mut setting_engine = webrtc::api::setting_engine::SettingEngine::default();

    if !config.force_relay {
        let detected_ip = std::process::Command::new("hostname")
            .arg("-I")
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .and_then(|s| s.split_whitespace().next().map(String::from));

        let nat_ip = config
            .public_ip
            .clone()
            .or(detected_ip)
            .unwrap_or_else(|| "127.0.0.1".to_owned());

        eprintln!("WebRTC NAT 1:1 IP: {}", nat_ip);
        setting_engine.set_nat_1to1_ips(
            vec![nat_ip],
            webrtc::ice_transport::ice_candidate_type::RTCIceCandidateType::Host,
        );
    } else {
        eprintln!("WebRTC: FORCE_RELAY mode");
    }

    setting_engine.set_network_types(vec![
        NetworkType::Udp4,
        NetworkType::Udp6,
        NetworkType::Tcp4,
        NetworkType::Tcp6,
    ]);

    let webrtc_api = APIBuilder::new()
        .with_media_engine(media_engine)
        .with_interceptor_registry(registry)
        .with_setting_engine(setting_engine)
        .build();

    // Load inference engine
    let tokenizer_path = config.tokenizer_path();
    let model_path = config.active_model_path();

    eprintln!("Loading inference engine...");
    let engine: Arc<dyn inference::InferenceEngine> = match (config.quant, config.backend) {
        (config::Quantization::Q4, _) => {
            let q4_engine = inference::engine::Q4Engine::load(&model_path, &tokenizer_path)
                .map_err(|e| -> Box<dyn std::error::Error> { e })?;
            Arc::new(q4_engine)
        }
        (config::Quantization::Bf16, config::GpuBackend::Wgpu) => {
            let bf16_engine =
                inference::bf16_engine::Bf16Engine::load(&model_path, &tokenizer_path)
                    .map_err(|e| -> Box<dyn std::error::Error> { e })?;
            Arc::new(bf16_engine)
        }
        #[cfg(feature = "cuda")]
        (config::Quantization::Bf16, config::GpuBackend::Cuda) => {
            let cuda_engine =
                inference::cuda_engine::CudaEngine::load(&model_path, &tokenizer_path)
                    .map_err(|e| -> Box<dyn std::error::Error> { e })?;
            Arc::new(cuda_engine)
        }
        #[cfg(feature = "candle")]
        (config::Quantization::Bf16, config::GpuBackend::Candle) => {
            let candle_engine =
                inference::candle_engine::CandleEngine::load(&model_path, &tokenizer_path)
                    .map_err(|e| -> Box<dyn std::error::Error> { e })?;
            Arc::new(candle_engine)
        }
        #[cfg(feature = "candle-native")]
        (config::Quantization::Bf16, config::GpuBackend::CandleNative) => {
            let candle_native_engine =
                inference::candle_native::engine::CandleNativeEngine::load(&model_path, &tokenizer_path)
                    .map_err(|e| -> Box<dyn std::error::Error> { e })?;
            Arc::new(candle_native_engine)
        }
    };
    eprintln!("Inference engine ready.");

    let frontend_path = config.frontend.clone();
    let models_dir = config.models_dir.clone();
    let port = config.port;
    let state = Arc::new(AppState::new(config, webrtc_api).with_engine(engine));

    // Build router (same API contract as vllm-server and parakeet-rs)
    let app = Router::new()
        // Health check
        .route("/health", get(|| async { "OK" }))
        // Config
        .route("/api/config", get(server::routes::get_config))
        // Models
        .route("/api/models", get(server::routes::list_models))
        // Media
        .route("/api/media", get(server::routes::list_media))
        .route("/api/media/upload", post(server::routes::upload_media))
        .route("/api/media/:id", delete(server::routes::delete_media))
        // Modes
        .route("/api/modes", get(server::routes::list_modes))
        // Noise cancellation
        .route(
            "/api/noise-cancellation",
            get(server::routes::list_noise_cancellation),
        )
        // Diarization
        .route("/api/diarization", get(server::routes::list_diarization))
        // SRT streams
        .route("/api/srt-streams", get(server::routes::list_srt_streams))
        // Sessions
        .route("/api/sessions", get(server::routes::list_sessions))
        .route("/api/sessions", post(server::routes::create_session))
        .route("/api/sessions/:id", get(server::routes::get_session))
        .route("/api/sessions/:id", delete(server::routes::stop_session))
        .route(
            "/api/sessions/:id/start",
            post(server::routes::start_session),
        )
        // WebSocket
        .route("/ws/:session_id", get(server::ws::ws_handler))
        // Static frontend (models served via symlink: frontend/models → ../models/cache)
        .fallback_service(ServeDir::new(&frontend_path))
        .layer(DefaultBodyLimit::max(2 * 1024 * 1024 * 1024)) // 2GB upload
        .layer(CorsLayer::permissive())
        .with_state(state);

    // Start server
    let addr = format!("0.0.0.0:{}", port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;

    eprintln!("Server listening on http://{}", addr);
    eprintln!("API: http://{}/api/*", addr);
    eprintln!("Frontend: http://{}", addr);
    eprintln!();

    axum::serve(listener, app).await?;

    Ok(())
}
