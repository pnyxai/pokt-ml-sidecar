use crate::media::MediaConfig;
use crate::policy::EngineRole;
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;

#[derive(Debug, Deserialize, Serialize)]
pub struct EngineConfig {
    pub role: EngineRole,
    #[serde(default)]
    pub serve_completions: bool,
}

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
    pub max_batched_texts_embedding: Option<usize>,
    pub max_rerank_documents: Option<usize>,
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
    pub engine: EngineConfig,
    pub media: Option<MediaConfig>,
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

        let config: Config = serde_yaml::from_str(&config_str)
            .map_err(|e| format!("Failed to parse config file '{}': {}", config_path, e))?;

        config.validate(&config_path)?;
        Ok(config)
    }

    fn validate(&self, config_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        if self.engine.role == EngineRole::Chat && self.media.is_none() {
            return Err(format!(
                "Invalid config '{}': engine role 'chat' requires an explicit 'media' block \
                 (mirroring the engine's --limit-mm-per-prompt flags)",
                config_path
            )
            .into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BASE: &str = r#"
log_level: "info"
engine:
  role: chat
media:
  allow_video: false
  limit_mm_per_prompt:
    image: 4
    audio: 1
    video: 0
server:
  host: "0.0.0.0"
  port: 8080
vllm_backend:
  host: "http://127.0.0.1"
  port: 8000
  model_name_override: "some/model"
  allow_logprobs: false
  crop_max_tokens: true
routing:
  timeout_seconds: 600
  max_payload_size_mb: 10
model_config_data:
  model_public_name: "public"
  max_position_embeddings: 4096
  max_tokens: 4095
"#;

    fn parse(yaml: &str) -> Result<Config, String> {
        let config: Config = serde_yaml::from_str(yaml).map_err(|e| e.to_string())?;
        config.validate("test").map_err(|e| e.to_string())?;
        Ok(config)
    }

    #[test]
    fn chat_with_media_block_parses() {
        let config = parse(BASE).expect("valid config");
        assert_eq!(config.engine.role, EngineRole::Chat);
        assert!(config.media.is_some());
    }

    #[test]
    fn missing_role_fails_to_parse() {
        let yaml = BASE.replace("  role: chat\n", "");
        let err = parse(&yaml).unwrap_err();
        assert!(err.contains("role"), "unexpected error: {}", err);
    }

    #[test]
    fn missing_engine_block_fails_to_parse() {
        let yaml = BASE.replace("engine:\n  role: chat\n", "");
        let err = parse(&yaml).unwrap_err();
        assert!(err.contains("engine"), "unexpected error: {}", err);
    }

    #[test]
    fn unknown_role_fails_to_parse() {
        let yaml = BASE.replace("role: chat", "role: embeddings-extra");
        assert!(parse(&yaml).unwrap_err().contains("role"));
    }

    #[test]
    fn chat_without_media_block_fails() {
        let yaml = BASE.replace("media:\n", "media_disabled:\n");
        let err = parse(&yaml).unwrap_err();
        assert!(err.contains("media"), "unexpected error: {}", err);
    }

    #[test]
    fn non_chat_roles_may_omit_media_block() {
        for role in ["embedding", "reranker"] {
            let yaml = BASE
                .replace("role: chat", &format!("role: {}", role))
                .replace("media:\n", "media_disabled:\n");
            let config = parse(&yaml).unwrap_or_else(|e| panic!("{}: {}", role, e));
            assert_eq!(
                config.engine.role,
                if role == "embedding" {
                    EngineRole::Embedding
                } else {
                    EngineRole::Reranker
                }
            );
            assert!(config.media.is_none());
        }
    }

    #[test]
    fn partial_media_block_uses_field_defaults() {
        let yaml = BASE.replace(
            "media:\n  allow_video: false\n  limit_mm_per_prompt:\n    image: 4\n    audio: 1\n    video: 0\n",
            "media:\n  limit_mm_per_prompt:\n    image: 2\n",
        );
        let config = parse(&yaml).expect("partial media block");
        let media = config.media.unwrap();
        assert_eq!(media.limit_mm_per_prompt.image, 2);
        assert_eq!(media.limit_mm_per_prompt.audio, 0);
        assert!(!media.allow_video);
        assert_eq!(media.max_item_bytes.image, 5 * 1024 * 1024);
    }
}
