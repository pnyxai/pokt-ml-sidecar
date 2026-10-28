use percent_encoding::percent_decode_str;
use serde::{Deserialize, Serialize};

/// The role of the engine behind this sidecar. Each role only exposes the
/// endpoints it is meant to serve.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum EngineRole {
    #[default]
    Chat,
    #[serde(alias = "embeddings")]
    Embedding,
    Reranker,
}

/// Path and method allowlist for a given engine role.
#[derive(Debug, Clone, Copy)]
pub struct RoutePolicy {
    pub role: EngineRole,
    pub serve_completions: bool,
}

impl RoutePolicy {
    pub fn new(role: EngineRole, serve_completions: bool) -> Self {
        Self {
            role,
            serve_completions,
        }
    }

    /// Returns true only when the raw request path can be canonicalized and
    /// the resulting path/method pair is exposed for this engine role.
    pub fn allows(&self, method: &str, raw_path: &str) -> bool {
        let Some(path) = canonicalize_path(raw_path) else {
            return false;
        };
        let method = method.to_ascii_uppercase();

        match self.role {
            EngineRole::Chat => match (method.as_str(), path.as_str()) {
                ("GET", "/v1/models") => true,
                ("POST", "/v1/chat/completions") => true,
                ("POST", "/v1/completions") => self.serve_completions,
                _ => false,
            },
            EngineRole::Embedding => matches!(
                (method.as_str(), path.as_str()),
                ("GET", "/v1/models") | ("POST", "/v1/embeddings")
            ),
            EngineRole::Reranker => matches!(
                (method.as_str(), path.as_str()),
                ("GET", "/v1/models") | ("POST", "/v1/rerank") | ("POST", "/v1/score")
            ),
        }
    }
}

/// Validates and normalizes a raw request path.
///
/// Any percent-encoded path is rejected: endpoints are fixed ASCII paths, so
/// the decoded and normalized form must be byte-for-byte identical to the raw
/// path. This prevents an encoded path from being allowlisted under one form
/// while being routed or proxied under another.
pub fn canonicalize_path(raw_path: &str) -> Option<String> {
    if raw_path.is_empty() || !raw_path.starts_with('/') {
        return None;
    }
    if raw_path.chars().any(|c| c.is_control()) {
        return None;
    }

    let raw_lower = raw_path.to_ascii_lowercase();
    if has_forbidden_pattern(&raw_lower) {
        return None;
    }

    let decoded_cow = percent_decode_str(raw_path).decode_utf8().ok()?;
    let decoded = decoded_cow.as_ref();
    let decoded_lower = decoded.to_ascii_lowercase();
    if has_forbidden_pattern(&decoded_lower) {
        return None;
    }
    if decoded != raw_path {
        return None;
    }

    let mut segments = Vec::new();
    for segment in decoded.split('/').skip(1) {
        if segment.is_empty() || segment == "." || segment == ".." {
            return None;
        }
        segments.push(segment);
    }

    let normalized = format!("/{}", segments.join("/"));
    if normalized != decoded {
        return None;
    }

    Some(normalized)
}

fn has_forbidden_pattern(path: &str) -> bool {
    path.contains("..")
        || path.contains("//")
        || path.contains('\\')
        || path.contains("%2f")
        || path.contains("%5c")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chat() -> RoutePolicy {
        RoutePolicy::new(EngineRole::Chat, false)
    }

    fn chat_with_completions() -> RoutePolicy {
        RoutePolicy::new(EngineRole::Chat, true)
    }

    fn embedding() -> RoutePolicy {
        RoutePolicy::new(EngineRole::Embedding, false)
    }

    fn reranker() -> RoutePolicy {
        RoutePolicy::new(EngineRole::Reranker, false)
    }

    #[test]
    fn chat_allows_chat_completions() {
        assert!(chat().allows("POST", "/v1/chat/completions"));
    }

    #[test]
    fn chat_allows_completions_only_when_sold() {
        assert!(!chat().allows("POST", "/v1/completions"));
        assert!(chat_with_completions().allows("POST", "/v1/completions"));
    }

    #[test]
    fn chat_rejects_other_engine_endpoints() {
        for path in [
            "/v1/embeddings",
            "/v1/rerank",
            "/v1/score",
            "/v1/responses",
            "/v1/messages",
            "/v1/chat/completions/render",
            "/v1/load_lora_adapter",
        ] {
            assert!(!chat().allows("POST", path), "{} should be rejected", path);
        }
    }

    #[test]
    fn embedding_only_allows_embeddings() {
        assert!(embedding().allows("POST", "/v1/embeddings"));
        assert!(!embedding().allows("POST", "/v1/chat/completions"));
        assert!(!embedding().allows("POST", "/v1/completions"));
        assert!(!embedding().allows("POST", "/v1/rerank"));
        assert!(!embedding().allows("POST", "/v1/score"));
    }

    #[test]
    fn reranker_only_allows_rerank_and_score() {
        assert!(reranker().allows("POST", "/v1/rerank"));
        assert!(reranker().allows("POST", "/v1/score"));
        assert!(!reranker().allows("POST", "/v1/chat/completions"));
        assert!(!reranker().allows("POST", "/v1/embeddings"));
    }

    #[test]
    fn models_is_static_for_every_role() {
        assert!(chat().allows("GET", "/v1/models"));
        assert!(embedding().allows("GET", "/v1/models"));
        assert!(reranker().allows("GET", "/v1/models"));
        assert!(!chat().allows("POST", "/v1/models"));
        assert!(!embedding().allows("POST", "/v1/models"));
        assert!(!reranker().allows("POST", "/v1/models"));
    }

    #[test]
    fn wrong_methods_are_rejected() {
        assert!(!chat().allows("GET", "/v1/chat/completions"));
        assert!(!embedding().allows("GET", "/v1/embeddings"));
        assert!(!reranker().allows("GET", "/v1/rerank"));
        assert!(!chat().allows("OPTIONS", "/v1/chat/completions"));
    }

    #[test]
    fn encoded_paths_are_rejected() {
        for path in [
            "/v1/%65mbeddings",
            "/v1/chat%2Fcompletions",
            "/v1/chat%2fcompletions",
            "/v1/chat%5Ccompletions",
            "/v1/chat%5ccompletions",
            "/v1/%2e%2e/models",
            "/v1/chat/completions%252f..%252frender",
        ] {
            assert!(!chat().allows("POST", path), "{} should be rejected", path);
            assert!(
                !embedding().allows("POST", path),
                "{} should be rejected",
                path
            );
            assert!(
                !reranker().allows("POST", path),
                "{} should be rejected",
                path
            );
            assert_eq!(canonicalize_path(path), None, "{} should be rejected", path);
        }
    }

    #[test]
    fn traversal_and_duplicate_separators_are_rejected() {
        for path in [
            "/v1/chat/completions/../render",
            "/v1/../v1/models",
            "/v1/./models",
            "///v1/models",
            "/v1//models",
            "/v1/chat\\completions",
            "/v1/models/",
            "/",
        ] {
            assert_eq!(canonicalize_path(path), None, "{} should be rejected", path);
        }
    }

    #[test]
    fn canonical_paths_are_accepted() {
        assert_eq!(
            canonicalize_path("/v1/chat/completions").as_deref(),
            Some("/v1/chat/completions")
        );
        assert_eq!(
            canonicalize_path("/v1/models").as_deref(),
            Some("/v1/models")
        );
        assert_eq!(
            canonicalize_path("/pokt/config").as_deref(),
            Some("/pokt/config")
        );
        assert_eq!(canonicalize_path("/health").as_deref(), Some("/health"));
    }

    #[test]
    fn control_characters_and_bad_encoding_are_rejected() {
        for path in [
            "/v1/chat/completions%00",
            "/v1/chat/completions%C0%AF",
            "/v1/chat/completions%FF",
        ] {
            assert_eq!(canonicalize_path(path), None, "{} should be rejected", path);
        }
    }
}
