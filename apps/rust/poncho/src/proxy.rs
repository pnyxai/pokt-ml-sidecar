use axum::{
    body::{Body, Bytes},
    extract::{Path, Query, Request, State},
    http::{HeaderMap, HeaderName, HeaderValue, Method, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response as AxumResponse},
};
use log::{debug, error, info, warn};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{collections::HashMap, str::FromStr, sync::Arc, time::Duration};
use tokio_stream::StreamExt;

use crate::media::MediaConfig;
use crate::policy::{canonicalize_path, RoutePolicy};

#[derive(Clone)]
pub struct VllmOverrides {
    pub model_name: String,
    pub allow_logprobs: bool,
    pub crop_max_tokens: bool,
    pub max_tokens: u64,
    pub max_position_embeddings: u64,
    pub overriden_name: String,
    pub max_batched_texts_embedding: Option<usize>,
    pub max_rerank_documents: Option<usize>,
}

// These are to be used to request input token counts
#[derive(Deserialize)]
struct Usage {
    prompt_tokens: u64,
}
#[derive(Deserialize)]
struct Response {
    usage: Usage,
}

// In web servers, each incoming request is handled by a separate async task.
// The ProxyState struct allows you to share the HTTP Client across all these
//  concurrent request handlers without creating a new client for each request.
#[derive(Clone)]
pub struct ProxyState {
    pub client: Client,
    pub backend_url: String,
    pub backend_port: String,
    pub max_payload_size_mb: usize,
    pub timeout: u64,
    pub vllm_overrides: VllmOverrides,
    pub route_policy: RoutePolicy,
    pub media: MediaConfig,
}

#[derive(Debug)]
pub enum ProxyError {
    Internal(String),
    Upstream(String),
    Json(String),
    Validation(String),
}

#[derive(Serialize)]
pub struct ErrorResponse {
    pub code: u32,
    pub message: String,
}

impl IntoResponse for ProxyError {
    fn into_response(self) -> AxumResponse {
        let (status, _code, message) = match self {
            ProxyError::Internal(msg) => {
                error!("[PROXY] ❌ Internal error: {}", msg);
                (StatusCode::INTERNAL_SERVER_ERROR, "INTERNAL_ERROR", msg)
            }
            ProxyError::Upstream(msg) => {
                error!("[PROXY] ❌ Upstream error: {}", msg);
                (StatusCode::BAD_GATEWAY, "UPSTREAM_ERROR", msg)
            }
            ProxyError::Json(msg) => {
                error!("[PROXY] ❌ JSON error: {}", msg);
                (StatusCode::BAD_REQUEST, "JSON_ERROR", msg)
            }
            ProxyError::Validation(msg) => {
                error!("[PROXY] ❌ Validation error: {}", msg);
                (StatusCode::BAD_REQUEST, "VALIDATION_ERROR", msg)
            }
        };

        let error_response = ErrorResponse {
            code: status.as_u16() as u32,
            message,
        };

        (status, axum::Json(error_response)).into_response()
    }
}

impl From<reqwest::Error> for ProxyError {
    fn from(err: reqwest::Error) -> Self {
        ProxyError::Upstream(format!("Request error: {}", err))
    }
}

impl From<serde_json::Error> for ProxyError {
    fn from(err: serde_json::Error) -> Self {
        ProxyError::Json(format!("JSON error: {}", err))
    }
}

/// Rejects requests whose decoded, normalised path is not allowed for the
/// configured engine role. The `/v1` namespace is strictly allowlisted;
/// everything else in that namespace gets a JSON 404.
pub async fn v1_path_guard(
    State(state): State<Arc<ProxyState>>,
    request: Request,
    next: Next,
) -> AxumResponse {
    let path = request.uri().path();
    let method = request.method().as_str();

    if canonicalize_path(path).is_none() {
        warn!("[GUARD] ❌ Rejected non-canonical path: {}", path);
        return not_found_response();
    }

    if (path == "/v1" || path.starts_with("/v1/")) && !state.route_policy.allows(method, path) {
        warn!(
            "[GUARD] ❌ Rejected {} {} for engine role {:?}",
            method, path, state.route_policy.role
        );
        return not_found_response();
    }

    next.run(request).await
}

/// Parses a request body as JSON independently of any Content-Type header.
/// Empty or malformed bodies are rejected with a 400.
fn parse_json_body(body: Bytes) -> Result<Value, ProxyError> {
    if body.is_empty() {
        return Err(ProxyError::Json(
            "Request body must be valid JSON".to_string(),
        ));
    }

    serde_json::from_slice(&body).map_err(|e| ProxyError::Json(format!("Invalid JSON body: {}", e)))
}

/// Serializes the parsed JSON value. Only this re-serialized form is forwarded.
fn serialize_json_body(json: &Value) -> Result<Bytes, ProxyError> {
    let modified_json = serde_json::to_vec(json)
        .map_err(|e| ProxyError::Json(format!("Failed to serialize JSON: {}", e)))?;
    Ok(Bytes::from(modified_json))
}

pub async fn proxy_handler(
    State(state): State<Arc<ProxyState>>,
    Path(path): Path<String>,
    Query(params): Query<HashMap<String, String>>,
    method: Method,
    headers: HeaderMap,
    body: Body,
) -> Result<AxumResponse, ProxyError> {
    // Set the backend target endpoint
    let target_base: String = format!("{}:{}", state.backend_url, state.backend_port);
    // Prepend v1/ because this handler is only mounted under /v1/*path
    let target_url: String = format!("{}/v1/{}", target_base, path);

    debug!("[PROXY] 🎯 Proxying {} {} -> {}", method, path, target_url);

    // Build query string, if any, after the backend path (not really tested)
    let query_string = if params.is_empty() {
        String::new()
    } else {
        let mut query = String::with_capacity(256);
        query.push('?');
        for (i, (k, v)) in params.iter().enumerate() {
            if i > 0 {
                query.push('&');
            }
            query.push_str(k);
            query.push('=');
            query.push_str(v);
        }
        query
    };
    let full_url = format!("{}{}", target_url, query_string);

    // Read body of the request, up to the given number of MBs
    debug!("[PROXY] 📥 Reading request body...");
    let body_bytes = match axum::body::to_bytes(body, state.max_payload_size_mb * 1024 * 1024).await
    {
        Ok(bytes) => {
            debug!("[PROXY] ✅ Body read successfully: {} bytes", bytes.len());
            bytes
        }
        Err(e) => {
            error!("[PROXY] ❌ Failed to read body: {}", e);
            return Err(ProxyError::Internal(format!(
                "Failed to read request body: {}",
                e
            )));
        }
    };

    // Every POST body is parsed as JSON regardless of the inbound Content-Type.
    // Only the re-serialized value is forwarded upstream.
    let processed_body = if method == Method::POST {
        let mut json = parse_json_body(body_bytes)?;

        if should_modify_request(&method, &path) {
            debug!("[MODIFY] 🔧 Modifying request body...");
            if let Err(e) = modify_json_payload(
                &mut json,
                &path,
                &state.media,
                state.vllm_overrides.clone(),
                target_url,
                &state.client,
            )
            .await
            {
                warn!("[MODIFY] ❌ Failed to modify body: {:?}", e);
                return Err(e);
            }
            debug!("[MODIFY] ✅ Body modified successfully");
        }

        serialize_json_body(&json)?
    } else {
        body_bytes
    };

    // Prepare headers
    debug!("[PROXY] 📋 Preparing headers...");
    let mut upstream_headers = reqwest::header::HeaderMap::new();

    // POST bodies are always forwarded as re-serialized JSON.
    if method == Method::POST || !processed_body.is_empty() {
        upstream_headers.insert(
            reqwest::header::CONTENT_TYPE,
            reqwest::header::HeaderValue::from_static("application/json"),
        );
    }

    // Add authorization header if present in original request
    if let Some(auth) = headers.get("authorization") {
        if let Ok(auth_value) = reqwest::header::HeaderValue::from_bytes(auth.as_bytes()) {
            upstream_headers.insert(reqwest::header::AUTHORIZATION, auth_value);
        }
    }

    // Debug: Print all headers being sent
    debug!("[PROXY] 📋 Headers to send:");
    for (name, value) in upstream_headers.iter() {
        debug!(
            "[PROXY]    {}: {}",
            name,
            value.to_str().unwrap_or("[unprintable]")
        );
    }
    debug!("[PROXY] 🚀 Making upstream request...");
    debug!("[PROXY]    URL: {}", full_url);
    debug!("[PROXY]    Method: {}", method);
    debug!("[PROXY]    Body size: {} bytes", processed_body.len());

    // Build request
    let request_builder = state
        // set client
        .client
        // add method and endpoint
        .request(method.clone(), &full_url)
        // add headers
        .headers(upstream_headers)
        // add body
        .body(processed_body);

    debug!("[PROXY] ⏳ Sending request to backend...");
    debug!("[PROXY] 🔍 About to call request_builder.send()...");

    // Add a timeout wrapper to catch hanging requests
    let request_future = request_builder.send();
    let timeout_duration = Duration::from_secs(state.timeout);

    debug!(
        "[PROXY] ⏰ Starting request with {} second timeout...",
        timeout_duration.as_secs()
    );

    // Await the request future
    let upstream_response = match tokio::time::timeout(timeout_duration, request_future).await {
        Ok(Ok(response)) => {
            debug!(
                "[PROXY] ✅ Got response from backend: {} - Headers: {:?}",
                response.status(),
                response.headers()
            );
            response
        }
        Ok(Err(e)) => {
            error!(
                "[PROXY] ❌ Request failed after send(): {:?}\n\
                 [PROXY]    Is timeout: {}\n\
                 [PROXY]    Is connect: {}\n\
                 [PROXY]    Is request: {}\n\
                 [PROXY]    Is decode: {}",
                e,
                e.is_timeout(),
                e.is_connect(),
                e.is_request(),
                e.is_decode()
            );

            if e.is_timeout() {
                return Err(ProxyError::Upstream(format!(
                    "Request timeout to {}: {}",
                    full_url, e
                )));
            } else if e.is_connect() {
                return Err(ProxyError::Upstream(format!(
                    "Connection failed to {}: {} - Check if backend server is running",
                    full_url, e
                )));
            } else if e.is_request() {
                return Err(ProxyError::Upstream(format!(
                    "Request error to {}: {}",
                    full_url, e
                )));
            } else {
                return Err(ProxyError::Upstream(format!(
                    "Network error to {}: {}",
                    full_url, e
                )));
            }
        }
        Err(_) => {
            error!(
                "[PROXY] ❌ Request timed out after {} seconds",
                timeout_duration.as_secs()
            );
            return Err(ProxyError::Upstream(format!(
                "Request hung/timed out after {} seconds to {}",
                timeout_duration.as_secs(),
                full_url
            )));
        }
    };

    // Check if it is a streaming request
    let is_streaming = upstream_response
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .map(|v| {
            v.contains("text/event-stream")
                || v.contains("application/x-ndjson")
                || v.contains("text/plain")
        })
        .unwrap_or(false);

    info!(
        "[PROXY] 📦 Response type: {}",
        if is_streaming { "streaming" } else { "regular" }
    );

    if is_streaming {
        // Handle a streaming response
        handle_streaming_response(
            upstream_response,
            state.vllm_overrides.overriden_name.clone(),
        )
        .await
    } else {
        // Handle a regular response
        handle_regular_response(
            upstream_response,
            state.vllm_overrides.overriden_name.clone(),
        )
        .await
    }
}

/// Checks whether a request body should be modified or not
fn should_modify_request(method: &Method, path: &str) -> bool {
    // If the method is POST and the path is either completions or chat completions
    method == Method::POST && matches!(path, "chat/completions" | "completions")
}

/// Validates an optional numeric field against an inclusive range.
///
/// The field is accepted when absent, or when present as a finite JSON number
/// within `min..=max`. Strings, booleans, `null`, arrays, objects, non-finite
/// values, and out-of-range values are rejected.
fn validate_finite_range(
    obj: &serde_json::Map<String, Value>,
    field: &str,
    min: f64,
    max: f64,
) -> Result<(), ProxyError> {
    if let Some(value) = obj.get(field) {
        let number = value.as_f64().ok_or_else(|| {
            ProxyError::Validation(format!(
                "{} must be a finite number between {} and {}",
                field, min, max
            ))
        })?;

        if !number.is_finite() || number < min || number > max {
            return Err(ProxyError::Validation(format!(
                "{} must be a finite number between {} and {}",
                field, min, max
            )));
        }
    }

    Ok(())
}

/// Enforces the legacy completions prompt shape.
///
/// For `/v1/completions` only, `prompt` must be present and be a single JSON
/// string. Other endpoints are unaffected.
fn validate_completions_prompt(
    path: &str,
    obj: &serde_json::Map<String, Value>,
) -> Result<(), ProxyError> {
    if path != "completions" {
        return Ok(());
    }

    match obj.get("prompt") {
        Some(Value::String(_)) => Ok(()),
        Some(_) => Err(ProxyError::Validation(
            "prompt must be a single string".to_string(),
        )),
        None => Err(ProxyError::Validation(
            "prompt is required and must be a single string".to_string(),
        )),
    }
}

/// Summary of the validated generation-limit fields in a request.
#[derive(Debug)]
struct TokenLimitState {
    /// Names of the token-limit fields present in the request.
    fields: Vec<&'static str>,
    /// Whether at least one requested value exceeded the configured cap.
    limit_exceed: bool,
    /// Whether the request did not specify any token limit.
    no_limit: bool,
    /// Smallest requested limit after capping, or zero when no limit was given.
    tokens_req: u64,
}

/// Validates and caps `max_tokens` / `max_completion_tokens`.
///
/// Each field that is present must be a positive integer. Values above
/// `vllm_overrides.max_tokens` are clamped in place when `crop_max_tokens` is
/// enabled, and rejected otherwise. The returned state tells the caller
/// whether context-aware adjustment is still required.
fn validate_token_limits(
    obj: &mut serde_json::Map<String, Value>,
    vllm_overrides: &VllmOverrides,
) -> Result<TokenLimitState, ProxyError> {
    let fields: Vec<&'static str> = ["max_tokens", "max_completion_tokens"]
        .into_iter()
        .filter(|field| obj.contains_key(*field))
        .collect();

    let mut limit_exceed = false;

    for field in &fields {
        let value = obj
            .get(*field)
            .and_then(|value| value.as_u64())
            .filter(|value| *value > 0)
            .ok_or_else(|| {
                ProxyError::Validation(format!("{} must be a positive integer", field))
            })?;

        if value > vllm_overrides.max_tokens {
            if !vllm_overrides.crop_max_tokens {
                return Err(ProxyError::Validation(format!(
                    "This model's maximum context length is {} tokens. However, you requested {} tokens (plus input tokens).",
                    vllm_overrides.max_tokens, value
                )));
            }

            limit_exceed = true;
            obj.insert((*field).to_string(), Value::from(vllm_overrides.max_tokens));
        }
    }

    let tokens_req = fields
        .iter()
        .filter_map(|field| obj.get(*field).and_then(|value| value.as_u64()))
        .min()
        .unwrap_or(0);

    Ok(TokenLimitState {
        no_limit: fields.is_empty(),
        fields,
        limit_exceed,
        tokens_req,
    })
}

/// Top-level fields allowed on `/v1/chat/completions`. Anything else is
/// rejected with a 400, so poncho does not depend on the engine ignoring or
/// safely handling unknown/experimental parameters.
const CHAT_COMPLETIONS_FIELDS: &[&str] = &[
    "model",
    "messages",
    "max_tokens",
    "max_completion_tokens",
    "temperature",
    "top_p",
    "n",
    "stream",
    "stop",
    "presence_penalty",
    "frequency_penalty",
    "logit_bias",
    "logprobs",
    "top_logprobs",
    "prompt_logprobs",
    "user",
    "seed",
    "response_format",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "modalities",
    "prediction",
    "audio",
    "reasoning_effort",
    "metadata",
    "service_tier",
    "stream_options",
    "store",
];

/// Top-level fields allowed on `/v1/completions`.
const COMPLETIONS_FIELDS: &[&str] = &[
    "model",
    "prompt",
    "suffix",
    "max_tokens",
    "temperature",
    "top_p",
    "n",
    "stream",
    "stop",
    "presence_penalty",
    "frequency_penalty",
    "logit_bias",
    "logprobs",
    "top_logprobs",
    "user",
    "seed",
];

/// Rejects any top-level field that is not in the endpoint's explicit allowlist.
fn validate_allowed_fields(
    path: &str,
    obj: &serde_json::Map<String, Value>,
) -> Result<(), ProxyError> {
    let allowed = match path {
        "chat/completions" => CHAT_COMPLETIONS_FIELDS,
        "completions" => COMPLETIONS_FIELDS,
        _ => return Ok(()),
    };

    for key in obj.keys() {
        if !allowed.iter().any(|&field| field == key) {
            return Err(ProxyError::Validation(format!(
                "Unsupported field '{}' for /v1/{}",
                key, path
            )));
        }
    }

    Ok(())
}

/// Enforces that `n` is exactly 1 when present.
fn validate_n(obj: &serde_json::Map<String, Value>) -> Result<(), ProxyError> {
    if let Some(value) = obj.get("n") {
        let n = value
            .as_u64()
            .ok_or_else(|| ProxyError::Validation("n must be a positive integer".to_string()))?;
        if n != 1 {
            return Err(ProxyError::Validation("n must be 1".to_string()));
        }
    }
    Ok(())
}

/// Returns true when the request needs a backend round-trip to count input
/// tokens before the generation cap can be applied.
///
/// The legacy `/v1/completions` path never counts: it has no `messages` to send
/// to the counting endpoint, so it gets the configured cap directly instead.
fn should_count_input_tokens(
    path: &str,
    token_state: &TokenLimitState,
    crop_max_tokens: bool,
    has_messages: bool,
) -> bool {
    if path == "completions" {
        return false;
    }

    token_state.no_limit || (crop_max_tokens && token_state.limit_exceed && has_messages)
}

/// Validates and rewrites a chat/completions request payload.
///
/// The model name is overridden, disallowed logprobs are rejected, generation
/// parameters are validated, and token limits are capped. The completions path
/// additionally requires a single string `prompt`.
async fn modify_json_payload(
    json: &mut Value,
    path: &str,
    media: &MediaConfig,
    vllm_overrides: VllmOverrides,
    target_url: String,
    client: &Client,
) -> Result<(), ProxyError> {
    let obj = json
        .as_object_mut()
        .ok_or_else(|| ProxyError::Validation("Request body must be a JSON object".to_string()))?;

    validate_allowed_fields(path, obj)?;
    validate_n(obj)?;

    // Model override
    if obj.contains_key("model") {
        obj.insert(
            "model".to_string(),
            Value::String(vllm_overrides.model_name.clone()),
        );
    }

    // Check for logprobs
    if (obj.contains_key("logprobs") || obj.contains_key("prompt_logprobs"))
        && !vllm_overrides.allow_logprobs
    {
        return Err(ProxyError::Validation(
            "logprobs parameter is not allowed must be omitted".to_string(),
        ));
    }

    validate_completions_prompt(path, obj)?;
    validate_finite_range(obj, "temperature", 0.0, 2.0)?;
    validate_finite_range(obj, "top_p", 0.0, 1.0)?;
    crate::media::validate_and_sanitize(obj, media).map_err(ProxyError::Validation)?;

    let token_state = validate_token_limits(obj, &vllm_overrides)?;

    if token_state.tokens_req > 0 {
        debug!(
            "[MODIFY] 🔍 Found requested {} tokens of a max of {}. Is exceding limit? : {}",
            token_state.tokens_req, vllm_overrides.max_tokens, token_state.limit_exceed
        );
    }

    if path == "completions" && token_state.no_limit {
        // Legacy completions has no `messages` to count; apply the configured
        // generation cap directly instead of failing on token counting.
        obj.insert(
            "max_tokens".to_string(),
            Value::from(vllm_overrides.max_tokens),
        );
    } else if should_count_input_tokens(
        path,
        &token_state,
        vllm_overrides.crop_max_tokens,
        obj.contains_key("messages"),
    ) {
        // Get input tokens
        debug!("[MODIFY] 🔍 Requesting input tokens count.");
        let mut input_tokens: u64 = 0;
        if let Some(in_message) = obj.get("messages") {
            // Create a new call, requesting one token, using the incomming request message
            let body = serde_json::json!({
                "messages": in_message.clone(),  // Clone the array directly
                "model": Value::String(vllm_overrides.model_name.clone()),
                // TODO: Add tools field
                "max_tokens": 1
            });
            // Call the model to get input tokens
            let response = client.post(&target_url).json(&body).send().await?;
            // Read
            let data: Response = response.json().await?;
            // Get input tokens
            input_tokens = data.usage.prompt_tokens;

            debug!("[MODIFY]  Counted {} input tokens", input_tokens)
        }
        if input_tokens == 0 {
            return Err(ProxyError::Validation(
                "Unable to get total input tokens.".to_string(),
            ));
        }
        // Add a max completitions field to respect the max tokens override we set
        if input_tokens < vllm_overrides.max_position_embeddings {
            // Get available tokens from model length
            let available_tokens = vllm_overrides.max_position_embeddings - input_tokens;
            // Check if the context length is limmiting the generation
            let max_completion = available_tokens.min(vllm_overrides.max_tokens);
            let target_completion = if token_state.no_limit {
                max_completion
            } else {
                token_state.tokens_req.min(max_completion)
            };

            if token_state.fields.is_empty() {
                obj.insert(
                    "max_completion_tokens".to_string(),
                    Value::from(target_completion),
                );
            } else {
                for field in &token_state.fields {
                    let current = obj
                        .get(*field)
                        .and_then(|value| value.as_u64())
                        .unwrap_or(target_completion);
                    obj.insert(
                        (*field).to_string(),
                        Value::from(current.min(target_completion)),
                    );
                }
            }
        } else {
            return Err(ProxyError::Validation(
                "Number of input tokens exceed the model's maximum allowed tokens.".to_string(),
            ));
        }
    }

    debug!("[MODIFY] 🔧 Modified JSON payload");

    Ok(())
}

/// Stream response handler, will send the proxied request response as it arrives
async fn handle_streaming_response(
    upstream_response: reqwest::Response,
    new_model_name: String,
) -> Result<AxumResponse, ProxyError> {
    let status = upstream_response.status();
    let headers = upstream_response.headers().clone();

    debug!(
        "[STREAM] 🌊 Handling streaming response with status: {}",
        status
    );

    // Copy headers
    let mut response_headers = HeaderMap::with_capacity(headers.len());
    for (name, value) in headers.iter() {
        if let (Ok(name), Ok(value)) = (
            HeaderName::from_str(name.as_str()),
            HeaderValue::from_bytes(value.as_bytes()),
        ) {
            response_headers.insert(name, value);
        }
    }

    // Create stream
    let stream = upstream_response.bytes_stream();
    let body_stream = stream.map(move |chunk_result| {
        match chunk_result {
            Ok(chunk) => {
                debug!("[STREAM] 📦 Streaming chunk: {} bytes", chunk.len());

                // Try to parse and modify the JSON chunk
                match modify_json_chunk(&chunk, &new_model_name) {
                    Ok(modified_chunk) => {
                        debug!(
                            "[STREAM] ✏️ Modified JSON chunk: {} bytes",
                            modified_chunk.len()
                        );
                        Ok(modified_chunk)
                    }
                    Err(e) => {
                        // If parsing fails, pass through original chunk
                        // This handles cases where chunk might not be complete JSON
                        debug!(
                            "[STREAM] ⚠️ Failed to parse chunk as JSON (passing through): {}",
                            e
                        );
                        Ok(chunk)
                    }
                }
            }
            Err(e) => {
                error!("[STREAM] ❌ Stream error: {}", e);
                Err(std::io::Error::new(std::io::ErrorKind::Other, e))
            }
        }
    });

    let body = Body::from_stream(body_stream);

    let mut response = AxumResponse::builder().status(status.as_u16());

    for (name, value) in response_headers.iter() {
        response = response.header(name, value);
    }

    debug!("[STREAM] ✅ Streaming response built successfully");

    response
        .body(body)
        .map_err(|e| ProxyError::Internal(format!("Failed to build streaming response: {}", e)))
}

/// Modify the model field in a JSON chunk
fn modify_json_chunk(
    chunk: &Bytes,
    new_model_name: &str,
) -> Result<Bytes, Box<dyn std::error::Error>> {
    let chunk_str = std::str::from_utf8(chunk)?;

    // Handle SSE format: "data: {json}\n\n"
    if chunk_str.starts_with("data: ") {
        let json_part = &chunk_str[6..]; // Skip "data: "
        let json_part = json_part.trim_end(); // Remove trailing whitespace/newlines

        if json_part == "[DONE]" {
            // Pass through completion marker
            return Ok(chunk.clone());
        }

        // Parse and modify JSON
        let mut json: Value = serde_json::from_str(json_part)?;

        if let Some(obj) = json.as_object_mut() {
            if obj.contains_key("model") {
                obj.insert(
                    "model".to_string(),
                    Value::String(new_model_name.to_string()),
                );
            }
        }

        // Reconstruct SSE format
        let modified_json = serde_json::to_string(&json)?;
        let sse_chunk = format!("data: {}\n\n", modified_json);

        Ok(Bytes::from(sse_chunk))
    } else {
        // Not SSE format, probably not OpenAI, do not modify
        return Ok(chunk.clone());
    }
}

/// Regular response handler, will receive the response and then send it back to
/// the proxied source
async fn handle_regular_response(
    upstream_response: reqwest::Response,
    new_model_name: String,
) -> Result<AxumResponse, ProxyError> {
    let status = upstream_response.status();
    let headers = upstream_response.headers().clone();

    debug!(
        "[RESPONSE] 📄 Handling regular response with status: {}",
        status
    );

    let body_bytes = upstream_response
        .bytes()
        .await
        .map_err(|e| ProxyError::Upstream(format!("Failed to read response body: {}", e)))?;

    debug!("[RESPONSE] 📥 Response body: {} bytes", body_bytes.len());

    // Override model name
    // Read all the body as a json, it should be a json
    let mut json: Value = serde_json::from_slice(&body_bytes)
        .map_err(|e| ProxyError::Json(format!("Failed to parse JSON: {}", e)))?;
    if let Some(obj) = json.as_object_mut() {
        // Model override
        if obj.contains_key("model") {
            obj.insert("model".to_string(), Value::String(new_model_name));
        }
    }
    let modified_json = serde_json::to_vec(&json)
        .map_err(|e| ProxyError::Json(format!("Failed to serialize JSON: {}", e)))?;
    let modified_body_bytes = Bytes::from(modified_json);
    debug!(
        "[RESPONSE] ✏️ Modified JSON regular response: {} bytes",
        modified_body_bytes.len()
    );

    // Copy headers
    let mut response_headers = HeaderMap::with_capacity(headers.len());
    for (name, value) in headers.iter() {
        if let (Ok(name), Ok(value)) = (
            HeaderName::from_str(name.as_str()),
            HeaderValue::from_bytes(value.as_bytes()),
        ) {
            if name.as_str() == "content-length" {
                // Replace context len with valid value
                if let Ok(value) =
                    HeaderValue::from_str(format!("{}", modified_body_bytes.len()).as_str())
                {
                    response_headers.insert(name, value);
                }
            } else {
                response_headers.insert(name, value);
            }
        }
    }

    let mut response = AxumResponse::builder().status(status.as_u16());

    for (name, value) in response_headers.iter() {
        response = response.header(name, value);
    }

    debug!("[RESPONSE] ✅ Regular response built successfully");

    response
        .body(Body::from(modified_body_bytes))
        .map_err(|e| ProxyError::Internal(format!("Failed to build regular response: {}", e)))
}

/// Returns an OpenAI-compatible /v1/models response with the configured model name
pub async fn models_handler(State(state): State<Arc<ProxyState>>) -> impl IntoResponse {
    let model_name = state.vllm_overrides.overriden_name.clone();

    let response = serde_json::json!({
        "object": "list",
        "data": [
            {
                "id": model_name,
                "object": "model",
                "created": 1677610602,
                "owned_by": "pocket-network-sidecar"
            }
        ]
    });

    (StatusCode::OK, axum::Json(response))
}

/// Health check handler: pings the backend /health endpoint.
/// Returns 200 with {"status": "healthy"} if the backend responds with 200.
/// Returns a JSON ProxyError otherwise.
pub async fn health_handler(
    State(state): State<Arc<ProxyState>>,
) -> Result<impl IntoResponse, ProxyError> {
    let target_url = format!("{}:{}/health", state.backend_url, state.backend_port);

    let health_timeout = Duration::from_secs(3);

    let request_future = state.client.get(&target_url).send();

    match tokio::time::timeout(health_timeout, request_future).await {
        Ok(Ok(response)) => {
            if response.status().is_success() {
                Ok((
                    StatusCode::OK,
                    axum::Json(serde_json::json!({"status": "healthy"})),
                ))
            } else {
                error!(
                    "[HEALTH] ❌ Backend health check returned non-2xx status: {}",
                    response.status()
                );
                Err(ProxyError::Upstream(format!(
                    "Backend health check returned {}",
                    response.status()
                )))
            }
        }
        Ok(Err(e)) => {
            error!("[HEALTH] ❌ Backend health check request failed: {}", e);
            Err(ProxyError::Upstream(format!(
                "Backend health check failed: {}",
                e
            )))
        }
        Err(_) => {
            error!(
                "[HEALTH] ❌ Backend health check timed out after {} seconds",
                health_timeout.as_secs()
            );
            Err(ProxyError::Upstream(format!(
                "Backend health check timed out after {} seconds",
                health_timeout.as_secs()
            )))
        }
    }
}

/// Validates an embeddings request body and applies the model override.
///
/// Only text input is accepted: `input` must be a string or an array of
/// strings, and `texts` must be an array of strings. Token-id input (numbers,
/// integer arrays) is rejected, which also blocks the negative-id PoC for
/// GHSA-25q3-v2hm-8vpf.
fn apply_embeddings_overrides(
    json: &mut Value,
    model_name: &str,
    max_batched_texts: Option<usize>,
) -> Result<(), ProxyError> {
    let obj = json
        .as_object_mut()
        .ok_or_else(|| ProxyError::Validation("Request body must be a JSON object".to_string()))?;

    obj.insert("model".to_string(), Value::String(model_name.to_string()));

    let mut count = 0;

    if let Some(texts) = obj.get("texts") {
        let texts = texts.as_array().ok_or_else(|| {
            ProxyError::Validation("texts must be an array of strings".to_string())
        })?;
        if !texts.iter().all(Value::is_string) {
            return Err(ProxyError::Validation(
                "texts must be an array of strings".to_string(),
            ));
        }
        count = count.max(texts.len());
    }

    if let Some(input) = obj.get("input") {
        let input_count = match input {
            Value::String(_) => 1,
            Value::Array(items) => {
                if !items.iter().all(Value::is_string) {
                    return Err(ProxyError::Validation(
                        "input must be a string or an array of strings \
                         (token-id input is not supported)"
                            .to_string(),
                    ));
                }
                items.len()
            }
            _ => {
                return Err(ProxyError::Validation(
                    "input must be a string or an array of strings \
                     (token-id input is not supported)"
                        .to_string(),
                ));
            }
        };
        count = count.max(input_count);
    }

    if let Some(max) = max_batched_texts {
        if count > max {
            return Err(ProxyError::Validation(format!(
                "Too many texts for embedding: {} > maximum allowed {}",
                count, max
            )));
        }
    }

    Ok(())
}

/// Validates a rerank/score request body and applies the model override.
fn apply_rerank_overrides(
    json: &mut Value,
    model_name: &str,
    max_rerank_documents: Option<usize>,
) -> Result<(), ProxyError> {
    let obj = json
        .as_object_mut()
        .ok_or_else(|| ProxyError::Validation("Request body must be a JSON object".to_string()))?;

    obj.insert("model".to_string(), Value::String(model_name.to_string()));

    if let Some(docs) = obj.get("documents") {
        let docs = docs
            .as_array()
            .ok_or_else(|| ProxyError::Validation("documents must be an array".to_string()))?;
        let count = docs.len();
        if let Some(max) = max_rerank_documents {
            if count > max {
                return Err(ProxyError::Validation(format!(
                    "Too many documents for rerank: {} > maximum allowed {}",
                    count, max
                )));
            }
        }
    }

    Ok(())
}

/// Forwards a JSON request to the backend and returns the response body.
/// Used by embeddings and rerank handlers (no streaming).
async fn forward_json_request(
    state: &ProxyState,
    path: &str,
    headers: HeaderMap,
    body_bytes: Bytes,
) -> Result<AxumResponse, ProxyError> {
    let target_url = format!("{}:{}/{}", state.backend_url, state.backend_port, path);

    let mut upstream_headers = reqwest::header::HeaderMap::new();
    upstream_headers.insert(
        reqwest::header::CONTENT_TYPE,
        reqwest::header::HeaderValue::from_static("application/json"),
    );
    if let Some(auth) = headers.get("authorization") {
        if let Ok(auth_value) = reqwest::header::HeaderValue::from_bytes(auth.as_bytes()) {
            upstream_headers.insert(reqwest::header::AUTHORIZATION, auth_value);
        }
    }

    debug!("[PROXY] 📤 Forwarding JSON to {}", target_url);

    let response = state
        .client
        .post(&target_url)
        .headers(upstream_headers)
        .body(body_bytes)
        .send()
        .await
        .map_err(|e| ProxyError::Upstream(format!("Backend request failed: {}", e)))?;

    let status = response.status();
    let resp_headers = response.headers().clone();
    let body = response
        .bytes()
        .await
        .map_err(|e| ProxyError::Upstream(format!("Failed to read backend response: {}", e)))?;

    let mut response_headers = HeaderMap::with_capacity(resp_headers.len());
    for (name, value) in resp_headers.iter() {
        if let (Ok(name), Ok(value)) = (
            HeaderName::from_str(name.as_str()),
            HeaderValue::from_bytes(value.as_bytes()),
        ) {
            if name.as_str() == "content-length" {
                if let Ok(value) = HeaderValue::from_str(format!("{}", body.len()).as_str()) {
                    response_headers.insert(name, value);
                }
            } else {
                response_headers.insert(name, value);
            }
        }
    }

    let mut response = AxumResponse::builder().status(status.as_u16());
    for (name, value) in response_headers.iter() {
        response = response.header(name, value);
    }

    response
        .body(Body::from(body))
        .map_err(|e| ProxyError::Internal(format!("Failed to build response: {}", e)))
}

/// Embeddings handler: validates batch size and proxies to backend.
/// Accepts both OpenAI-style `input` and vLLM-style `texts`.
pub async fn embeddings_handler(
    State(state): State<Arc<ProxyState>>,
    headers: HeaderMap,
    body: Body,
) -> Result<AxumResponse, ProxyError> {
    debug!("[EMBED] 📥 Reading embeddings request body...");
    let body_bytes = match axum::body::to_bytes(body, state.max_payload_size_mb * 1024 * 1024).await
    {
        Ok(bytes) => bytes,
        Err(e) => {
            error!("[EMBED] ❌ Failed to read embeddings body: {}", e);
            return Err(ProxyError::Internal(format!(
                "Failed to read request body: {}",
                e
            )));
        }
    };

    let mut json = parse_json_body(body_bytes)?;

    apply_embeddings_overrides(
        &mut json,
        &state.vllm_overrides.model_name,
        state.vllm_overrides.max_batched_texts_embedding,
    )?;

    let modified_json = serialize_json_body(&json)?;

    forward_json_request(&state, "v1/embeddings", headers, modified_json).await
}

/// Validates a rerank/score request body and applies the model override.
async fn prepare_rerank_request(
    state: &ProxyState,
    headers: HeaderMap,
    body: Body,
) -> Result<(HeaderMap, Bytes), ProxyError> {
    debug!("[RERANK] 📥 Reading rerank request body...");
    let body_bytes = match axum::body::to_bytes(body, state.max_payload_size_mb * 1024 * 1024).await
    {
        Ok(bytes) => bytes,
        Err(e) => {
            error!("[RERANK] ❌ Failed to read rerank body: {}", e);
            return Err(ProxyError::Internal(format!(
                "Failed to read request body: {}",
                e
            )));
        }
    };

    let mut json = parse_json_body(body_bytes)?;

    apply_rerank_overrides(
        &mut json,
        &state.vllm_overrides.model_name,
        state.vllm_overrides.max_rerank_documents,
    )?;

    let modified_json = serialize_json_body(&json)?;

    Ok((headers, modified_json))
}

/// Rerank handler: validates document count and proxies to backend.
pub async fn rerank_handler(
    State(state): State<Arc<ProxyState>>,
    headers: HeaderMap,
    body: Body,
) -> Result<AxumResponse, ProxyError> {
    let (headers, body) = prepare_rerank_request(&state, headers, body).await?;
    forward_json_request(&state, "v1/rerank", headers, body).await
}

/// Score handler: mirrors rerank validation and proxies to the backend.
pub async fn score_handler(
    State(state): State<Arc<ProxyState>>,
    headers: HeaderMap,
    body: Body,
) -> Result<AxumResponse, ProxyError> {
    let (headers, body) = prepare_rerank_request(&state, headers, body).await?;
    forward_json_request(&state, "v1/score", headers, body).await
}

/// Fallback handler for unmatched routes, returns a JSON 404 error
pub async fn not_found_handler() -> AxumResponse {
    not_found_response()
}

/// Builds the JSON 404 response used by the guard and fallback.
pub fn not_found_response() -> AxumResponse {
    let error_response = ErrorResponse {
        code: 404,
        message: "Not found".to_string(),
    };
    (StatusCode::NOT_FOUND, axum::Json(error_response)).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_overrides(max_tokens: u64, crop_max_tokens: bool) -> VllmOverrides {
        VllmOverrides {
            model_name: "backend-model".to_string(),
            allow_logprobs: false,
            crop_max_tokens,
            max_tokens,
            max_position_embeddings: 4096,
            overriden_name: "public-model".to_string(),
            max_batched_texts_embedding: None,
            max_rerank_documents: None,
        }
    }

    fn object(value: Value) -> serde_json::Map<String, Value> {
        value.as_object().unwrap().clone()
    }

    #[test]
    fn max_tokens_must_be_a_positive_integer() {
        for value in [
            serde_json::json!({"max_tokens": 0}),
            serde_json::json!({"max_tokens": -1}),
            serde_json::json!({"max_tokens": 1.5}),
            serde_json::json!({"max_tokens": "10"}),
            serde_json::json!({"max_tokens": null}),
            serde_json::json!({"max_completion_tokens": 0}),
            serde_json::json!({"max_completion_tokens": -1}),
            serde_json::json!({"max_completion_tokens": 1.5}),
            serde_json::json!({"max_completion_tokens": "10"}),
            serde_json::json!({"max_completion_tokens": null}),
        ] {
            let mut obj = object(value);
            assert!(
                validate_token_limits(&mut obj, &test_overrides(100, true)).is_err(),
                "value should be rejected: {:?}",
                obj
            );
        }
    }

    #[test]
    fn oversized_token_limits_are_capped_when_crop_enabled() {
        let mut obj = object(serde_json::json!({
            "max_tokens": 50,
            "max_completion_tokens": 500
        }));

        let state = validate_token_limits(&mut obj, &test_overrides(100, true)).unwrap();

        assert_eq!(obj["max_tokens"], 50);
        assert_eq!(obj["max_completion_tokens"], 100);
        assert!(state.limit_exceed);
        assert!(!state.no_limit);
        assert_eq!(state.tokens_req, 50);
    }

    #[test]
    fn oversized_token_limits_are_rejected_without_crop() {
        for value in [
            serde_json::json!({"max_tokens": 500}),
            serde_json::json!({"max_completion_tokens": 500}),
        ] {
            let mut obj = object(value);
            assert!(
                validate_token_limits(&mut obj, &test_overrides(100, false)).is_err(),
                "value should be rejected: {:?}",
                obj
            );
        }
    }

    #[test]
    fn sampling_params_must_be_finite_and_in_range() {
        assert!(validate_finite_range(
            &object(serde_json::json!({"temperature": 0.0})),
            "temperature",
            0.0,
            2.0
        )
        .is_ok());
        assert!(validate_finite_range(
            &object(serde_json::json!({"temperature": 2})),
            "temperature",
            0.0,
            2.0
        )
        .is_ok());
        assert!(validate_finite_range(
            &object(serde_json::json!({"top_p": 0.0})),
            "top_p",
            0.0,
            1.0
        )
        .is_ok());
        assert!(
            validate_finite_range(&object(serde_json::json!({"top_p": 1})), "top_p", 0.0, 1.0)
                .is_ok()
        );

        for value in [
            serde_json::json!({"temperature": -0.1}),
            serde_json::json!({"temperature": 2.1}),
            serde_json::json!({"temperature": "1"}),
            serde_json::json!({"temperature": null}),
            serde_json::json!({"top_p": -0.1}),
            serde_json::json!({"top_p": 1.1}),
            serde_json::json!({"top_p": "1"}),
            serde_json::json!({"top_p": null}),
            serde_json::json!({"temperature": 1e308}),
        ] {
            let field = if value.get("temperature").is_some() {
                ("temperature", 0.0, 2.0)
            } else {
                ("top_p", 0.0, 1.0)
            };
            let obj = object(value);
            assert!(
                validate_finite_range(&obj, field.0, field.1, field.2).is_err(),
                "value should be rejected: {:?}",
                obj
            );
        }
    }

    #[test]
    fn completions_prompt_must_be_a_single_string() {
        assert!(validate_completions_prompt(
            "completions",
            &object(serde_json::json!({"prompt": "hello"}))
        )
        .is_ok());

        for value in [
            serde_json::json!({}),
            serde_json::json!({"prompt": ["hello"]}),
            serde_json::json!({"prompt": null}),
            serde_json::json!({"prompt": 1}),
            serde_json::json!({"prompt": {"text": "hello"}}),
        ] {
            assert!(
                validate_completions_prompt("completions", &object(value)).is_err(),
                "prompt should be rejected"
            );
        }

        assert!(validate_completions_prompt(
            "chat/completions",
            &object(serde_json::json!({"messages": []}))
        )
        .is_ok());
    }

    #[test]
    fn parses_valid_json_without_content_type() {
        let json = parse_json_body(Bytes::from_static(br#"{"model":"client-value"}"#)).unwrap();
        assert_eq!(json["model"], "client-value");
    }

    #[test]
    fn rejects_empty_and_non_json_bodies() {
        for body in [
            &b""[..],
            b"not json",
            b"{",
            b"{} trailing",
            b"[] []",
            b"\xff\xfe",
        ] {
            assert!(
                parse_json_body(Bytes::copy_from_slice(body)).is_err(),
                "body should be rejected: {:?}",
                body
            );
        }
    }

    #[test]
    fn reserializes_only_the_parsed_object() {
        let json =
            parse_json_body(Bytes::from_static(b"{ \"b\": 2, \"a\": 1, \"a\": 3 }")).unwrap();
        let serialized = serialize_json_body(&json).unwrap();

        assert_eq!(serialized.as_ref(), br#"{"a":3,"b":2}"#);
    }

    #[test]
    fn completion_paths_are_modified_independently_of_headers() {
        assert!(should_modify_request(&Method::POST, "chat/completions"));
        assert!(should_modify_request(&Method::POST, "completions"));
        assert!(!should_modify_request(&Method::GET, "chat/completions"));
        assert!(!should_modify_request(&Method::POST, "embeddings"));
    }

    #[test]
    fn embeddings_rejects_non_object_bodies() {
        for value in [
            serde_json::json!([]),
            serde_json::json!(["text"]),
            serde_json::json!("text"),
            serde_json::json!(1),
            serde_json::json!(null),
        ] {
            let mut json = value.clone();
            assert!(
                matches!(
                    apply_embeddings_overrides(&mut json, "backend-model", None),
                    Err(ProxyError::Validation(_))
                ),
                "body should be rejected: {:?}",
                value
            );
        }
    }

    #[test]
    fn embeddings_rejects_token_id_input() {
        for value in [
            serde_json::json!({"input": [[-1, 1879]]}),
            serde_json::json!({"input": [1, 2, 3]}),
            serde_json::json!({"input": ["ok", 1]}),
            serde_json::json!({"input": 123}),
            serde_json::json!({"input": null}),
            serde_json::json!({"texts": [1, 2]}),
            serde_json::json!({"texts": "text"}),
            serde_json::json!({"texts": [{"text": "x"}]}),
        ] {
            let mut json = value.clone();
            assert!(
                matches!(
                    apply_embeddings_overrides(&mut json, "backend-model", None),
                    Err(ProxyError::Validation(_))
                ),
                "input should be rejected: {:?}",
                value
            );
        }
    }

    #[test]
    fn embeddings_accepts_text_input_and_overrides_model() {
        for value in [
            serde_json::json!({"model": "client-model", "input": ["a", "b"]}),
            serde_json::json!({"model": "client-model", "input": "a"}),
            serde_json::json!({"model": "client-model", "texts": ["a"]}),
        ] {
            let mut json = value.clone();
            apply_embeddings_overrides(&mut json, "backend-model", None).unwrap();
            assert_eq!(json["model"], "backend-model");
        }
    }

    #[test]
    fn embeddings_enforces_batch_limit_for_both_shapes() {
        for value in [
            serde_json::json!({"input": ["a", "b", "c"]}),
            serde_json::json!({"texts": ["a", "b", "c"]}),
            serde_json::json!({"input": ["a"], "texts": ["a", "b", "c"]}),
        ] {
            let mut json = value.clone();
            assert!(
                matches!(
                    apply_embeddings_overrides(&mut json, "backend-model", Some(2)),
                    Err(ProxyError::Validation(_))
                ),
                "batch should exceed limit: {:?}",
                value
            );
            assert!(apply_embeddings_overrides(&mut json, "backend-model", Some(3)).is_ok());
        }
    }

    #[test]
    fn rerank_rejects_non_object_bodies() {
        for value in [
            serde_json::json!([]),
            serde_json::json!("text"),
            serde_json::json!(1),
            serde_json::json!(null),
        ] {
            let mut json = value.clone();
            assert!(
                matches!(
                    apply_rerank_overrides(&mut json, "backend-model", None),
                    Err(ProxyError::Validation(_))
                ),
                "body should be rejected: {:?}",
                value
            );
        }
    }

    #[test]
    fn rerank_overrides_model_and_enforces_document_limit() {
        let mut json = serde_json::json!({
            "model": "client-model",
            "query": "q",
            "documents": ["a", "b", "c"]
        });
        assert!(apply_rerank_overrides(&mut json, "backend-model", Some(2)).is_err());
        apply_rerank_overrides(&mut json, "backend-model", Some(3)).unwrap();
        assert_eq!(json["model"], "backend-model");
    }

    #[test]
    fn rerank_rejects_non_array_documents() {
        let mut json = serde_json::json!({"query": "q", "documents": "not-an-array"});
        assert!(matches!(
            apply_rerank_overrides(&mut json, "backend-model", None),
            Err(ProxyError::Validation(_))
        ));
    }

    #[test]
    fn chat_completions_rejects_unknown_fields() {
        let obj = object(serde_json::json!({
            "model": "m",
            "messages": [],
            "mm_processor_kwargs": {},
            "allowed_token_ids": [1, 2, 3]
        }));
        assert!(matches!(
            validate_allowed_fields("chat/completions", &obj),
            Err(ProxyError::Validation(_))
        ));
    }

    #[test]
    fn chat_completions_accepts_allowed_fields() {
        let obj = object(serde_json::json!({
            "model": "m",
            "messages": [],
            "max_tokens": 10,
            "max_completion_tokens": 10,
            "temperature": 0.5,
            "top_p": 0.9,
            "n": 1,
            "stream": true,
            "stop": ["stop"],
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "logit_bias": {},
            "logprobs": false,
            "top_logprobs": 0,
            "user": "u",
            "seed": 42,
            "response_format": { "type": "json_object" },
            "tools": [],
            "tool_choice": "auto",
            "parallel_tool_calls": true,
            "modalities": ["text"],
            "prediction": { "content": "x" },
            "audio": { "voice": "alloy", "format": "pcm16" },
            "reasoning_effort": "low",
            "metadata": {},
            "service_tier": "auto",
            "stream_options": { "include_usage": true },
            "store": false
        }));
        validate_allowed_fields("chat/completions", &obj).unwrap();
    }

    #[test]
    fn completions_rejects_unknown_fields() {
        let obj = object(serde_json::json!({
            "model": "m",
            "prompt": "p",
            "best_of": 2
        }));
        assert!(matches!(
            validate_allowed_fields("completions", &obj),
            Err(ProxyError::Validation(_))
        ));
    }

    #[test]
    fn completions_accepts_allowed_fields() {
        let obj = object(serde_json::json!({
            "model": "m",
            "prompt": "p",
            "suffix": "s",
            "max_tokens": 10,
            "temperature": 0.5,
            "top_p": 0.9,
            "n": 1,
            "stream": false,
            "stop": ["stop"],
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "logit_bias": {},
            "logprobs": false,
            "top_logprobs": 0,
            "user": "u",
            "seed": 42
        }));
        validate_allowed_fields("completions", &obj).unwrap();
    }

    #[test]
    fn n_must_be_one() {
        for value in [
            serde_json::json!({"n": 2}),
            serde_json::json!({"n": 0}),
            serde_json::json!({"n": -1}),
            serde_json::json!({"n": "1"}),
            serde_json::json!({"n": 1.5}),
        ] {
            let obj = object(value.clone());
            assert!(
                validate_n(&obj).is_err(),
                "n should be rejected: {:?}",
                value
            );
        }
        assert!(validate_n(&object(serde_json::json!({"n": 1}))).is_ok());
        assert!(validate_n(&object(serde_json::json!({}))).is_ok());
    }

    #[tokio::test]
    async fn completions_without_limit_gets_configured_cap() {
        let mut json = serde_json::json!({"model": "client-model", "prompt": "hello"});
        modify_json_payload(
            &mut json,
            "completions",
            &crate::media::MediaConfig::default(),
            test_overrides(100, true),
            "http://127.0.0.1:1/v1/completions".to_string(),
            &reqwest::Client::new(),
        )
        .await
        .unwrap();

        assert_eq!(json["max_tokens"], 100);
    }

    #[tokio::test]
    async fn completions_without_limit_caps_even_without_crop() {
        let mut json = serde_json::json!({"model": "client-model", "prompt": "hello"});
        modify_json_payload(
            &mut json,
            "completions",
            &crate::media::MediaConfig::default(),
            test_overrides(100, false),
            "http://127.0.0.1:1/v1/completions".to_string(),
            &reqwest::Client::new(),
        )
        .await
        .unwrap();

        assert_eq!(json["max_tokens"], 100);
    }

    #[tokio::test]
    async fn completions_explicit_limit_is_not_overwritten() {
        let mut json =
            serde_json::json!({"model": "client-model", "prompt": "hello", "max_tokens": 50});
        modify_json_payload(
            &mut json,
            "completions",
            &crate::media::MediaConfig::default(),
            test_overrides(100, true),
            "http://127.0.0.1:1/v1/completions".to_string(),
            &reqwest::Client::new(),
        )
        .await
        .unwrap();

        assert_eq!(json["max_tokens"], 50);
    }

    #[test]
    fn completions_never_counts_input_tokens() {
        let mut obj = object(serde_json::json!({}));
        let state = validate_token_limits(&mut obj, &test_overrides(100, true)).unwrap();
        assert!(state.no_limit);
        assert!(!should_count_input_tokens(
            "completions",
            &state,
            true,
            false
        ));

        let mut obj = object(serde_json::json!({"max_tokens": 500}));
        let state = validate_token_limits(&mut obj, &test_overrides(100, true)).unwrap();
        assert!(state.limit_exceed);
        assert!(!should_count_input_tokens(
            "completions",
            &state,
            true,
            true
        ));
    }

    #[test]
    fn chat_counts_when_limit_missing_or_exceeded() {
        let mut obj = object(serde_json::json!({}));
        let state = validate_token_limits(&mut obj, &test_overrides(100, true)).unwrap();
        assert!(should_count_input_tokens(
            "chat/completions",
            &state,
            true,
            true
        ));

        let mut obj = object(serde_json::json!({"max_tokens": 500}));
        let state = validate_token_limits(&mut obj, &test_overrides(100, true)).unwrap();
        assert!(should_count_input_tokens(
            "chat/completions",
            &state,
            true,
            true
        ));
        assert!(!should_count_input_tokens(
            "chat/completions",
            &state,
            true,
            false
        ));
        assert!(!should_count_input_tokens(
            "chat/completions",
            &state,
            false,
            true
        ));
    }
}
