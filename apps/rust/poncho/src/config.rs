use serde::{Deserialize, Serialize};
use std::env;
use std::fs;

#[derive(Debug, Deserialize, Serialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct VllmProxyConfig {
    pub host: String,
    pub port: u16,
    pub model_name_override: String,
    pub allow_logprobs: bool,
    pub crop_max_tokens: bool,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct RoutingConfig {
    pub timeout_seconds: u16,
    pub max_payload_size_mb: u16,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ModelConfig {
    pub model_public_name: String,
    pub max_position_embeddings: u64,
    pub max_tokens: u64,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Config {
    pub log_level: String,
    pub server: ServerConfig,
    pub vllm_backend: VllmProxyConfig,
    pub routing: RoutingConfig,
    pub model_config_data: ModelConfig,
}

impl Config {
    pub fn load() -> Result<Self, Box<dyn std::error::Error>> {
        let config_path =
            env::var("CONFIG_PATH").unwrap_or_else(|_| "./config/config.yaml".to_string());

        let config_str = fs::read_to_string(&config_path)
            .map_err(|e| format!("Failed to read config file '{}': {}", config_path, e))?;

        let config: Config = serde_yaml::from_str(&config_str)?;
        Ok(config)
    }
}
