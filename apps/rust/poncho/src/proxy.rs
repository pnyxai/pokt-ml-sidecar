use axum::{
    body::{Body, Bytes},
    extract::{Path, Query, State},
    http::{HeaderMap, HeaderName, HeaderValue, Method, StatusCode},
    response::{IntoResponse, Response as AxumResponse},
};
use log::{debug, error, info, warn};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{collections::HashMap, str::FromStr, sync::Arc, time::Duration};
use tokio_stream::StreamExt;

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

    // Check if I should modify the resquest
    let should_modify = should_modify_request(&method, &path, &headers);

    // If so, and there is some body here, modify it
    let processed_body = if should_modify && !body_bytes.is_empty() {
        debug!("[MODIFY] 🔧 Modifying request body...");
        match modify_json_payload(
            body_bytes,
            state.vllm_overrides.clone(),
            target_url,
            &state.client,
        )
        .await
        {
            Ok(modified) => {
                debug!("[MODIFY] ✅ Body modified successfully");
                modified
            }
            Err(e) => {
                warn!("[MODIFY] ❌ Failed to modify body: {:?}", e);
                return Err(e);
            }
        }
    } else {
        body_bytes
    };

    // Prepare headers
    debug!("[PROXY] 📋 Preparing headers...");
    let mut upstream_headers = reqwest::header::HeaderMap::new();

    // Only add content-type if we have a body
    if !processed_body.is_empty() {
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
        debug!("[PROXY]    {}: {}", name, value.to_str().unwrap_or("[unprintable]"));
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
                e, e.is_timeout(), e.is_connect(), e.is_request(), e.is_decode()
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
fn should_modify_request(method: &Method, path: &str, headers: &HeaderMap) -> bool {
    // If the method is POST
    method == Method::POST
        // if the path is either a completions or chat completions
        && path.contains("completions")
        // and if the content type is application/json
        && headers
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .map(|v| v.contains("application/json"))
            .unwrap_or(false)
}

/// Modifies the json payload of a vllm request, overriding the model name and
/// raising an error if logprobs are requested but not allowed
async fn modify_json_payload(
    body: Bytes,
    vllm_overrides: VllmOverrides,
    target_url: String,
    client: &Client,
) -> Result<Bytes, ProxyError> {
    // just in case empty body check
    if body.is_empty() {
        return Ok(body);
    }

    // Read all the body as a json, it should be a json
    let mut json: Value = serde_json::from_slice(&body)
        .map_err(|e| ProxyError::Json(format!("Failed to parse JSON: {}", e)))?;

    // if we can get a mutable json hashmap, go ahead
    if let Some(obj) = json.as_object_mut() {
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

        let mut limit_exceed: bool = false;
        let mut no_limit: bool = false;
        let mut tokens_req: u64 = 0;
        // Check for max_tokens
        if let Some(max_tokens) = obj.get("max_tokens") {
            if let Some(max_tokens_val) = max_tokens.as_u64() {
                if max_tokens_val > vllm_overrides.max_tokens {
                    limit_exceed = true;
                }
                tokens_req = max_tokens_val;
            }
        // Check for max_completion_tokens
        } else if let Some(max_tokens) = obj.get("max_completion_tokens") {
            if let Some(max_tokens_val) = max_tokens.as_u64() {
                if max_tokens_val > vllm_overrides.max_tokens {
                    limit_exceed = true;
                }
                tokens_req = max_tokens_val;
            }
        } else {
            // No field requesting generation limit, so, we need to limit this
            no_limit = true;
        }
        if tokens_req > 0 {
            debug!(
                "[MODIFY] 🔍 Found requested {} tokens of a max of {}. Is exceding limit? : {}",
                tokens_req, vllm_overrides.max_tokens, limit_exceed
            );
        }

        if (vllm_overrides.crop_max_tokens && limit_exceed) || no_limit {
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
                return Err(ProxyError::Validation(format!(
                    "Unable to get total input tokens."
                )));
            }
            // Add a max completitions field to respect the max tokens override we set
            if (vllm_overrides.max_position_embeddings - input_tokens) > 0 {
                // Get available tokens from model length
                let available_tokens = vllm_overrides.max_position_embeddings - input_tokens;
                // Check if the context length is limmiting the generation
                let max_completion = if available_tokens < vllm_overrides.max_tokens {
                    available_tokens
                } else {
                    vllm_overrides.max_tokens
                };
                obj.insert(
                    "max_completion_tokens".to_string(),
                    Value::from(max_completion),
                );
            } else {
                return Err(ProxyError::Validation(
                    "Number of input tokens exceed the model's maximum allowed tokens.".to_string(),
                ));
            }
        } else if limit_exceed {
            return Err(ProxyError::Validation(
                    format!("This model's maximum context length is {} tokens. However, you requested {} tokens (plus input tokens).", vllm_overrides.max_tokens, tokens_req)
                ));
        }

        debug!("[MODIFY] 🔧 Modified JSON payload");
    }

    let modified_json = serde_json::to_vec(&json)
        .map_err(|e| ProxyError::Json(format!("Failed to serialize JSON: {}", e)))?;

    Ok(Bytes::from(modified_json))
}

/// Stream response handler, will send the proxied request response as it arrives
async fn handle_streaming_response(
    upstream_response: reqwest::Response,
    new_model_name: String,
) -> Result<AxumResponse, ProxyError> {
    let status = upstream_response.status();
    let headers = upstream_response.headers().clone();

    debug!("[STREAM] 🌊 Handling streaming response with status: {}", status);

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
                        debug!("[STREAM] ✏️ Modified JSON chunk: {} bytes", modified_chunk.len());
                        Ok(modified_chunk)
                    }
                    Err(e) => {
                        // If parsing fails, pass through original chunk
                        // This handles cases where chunk might not be complete JSON
                        debug!("[STREAM] ⚠️ Failed to parse chunk as JSON (passing through): {}", e);
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

    debug!("[RESPONSE] 📄 Handling regular response with status: {}", status);

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
pub async fn health_handler(State(state): State<Arc<ProxyState>>) -> Result<impl IntoResponse, ProxyError> {
    let target_url = format!("{}:{}/health", state.backend_url, state.backend_port);

    let health_timeout = Duration::from_secs(3);

    let request_future = state.client.get(&target_url).send();

    match tokio::time::timeout(health_timeout, request_future).await {
        Ok(Ok(response)) => {
            if response.status().is_success() {
                Ok((StatusCode::OK, axum::Json(serde_json::json!({"status": "healthy"}))))
            } else {
                error!("[HEALTH] ❌ Backend health check returned non-2xx status: {}", response.status());
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
            error!("[HEALTH] ❌ Backend health check timed out after {} seconds", health_timeout.as_secs());
            Err(ProxyError::Upstream(format!(
                "Backend health check timed out after {} seconds",
                health_timeout.as_secs()
            )))
        }
    }
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
    let body_bytes = match axum::body::to_bytes(body, state.max_payload_size_mb * 1024 * 1024).await {
        Ok(bytes) => bytes,
        Err(e) => {
            error!("[EMBED] ❌ Failed to read embeddings body: {}", e);
            return Err(ProxyError::Internal(format!(
                "Failed to read request body: {}",
                e
            )));
        }
    };

    let mut json: Value = serde_json::from_slice(&body_bytes)
        .map_err(|e| ProxyError::Json(format!("Failed to parse JSON: {}", e)))?;

    let model_name = state.vllm_overrides.model_name.clone();

    if let Some(obj) = json.as_object_mut() {
        obj.insert("model".to_string(), Value::String(model_name));

        // Count texts / input
        let count = if let Some(texts) = obj.get("texts").and_then(|v| v.as_array()) {
            texts.len()
        } else if let Some(input) = obj.get("input") {
            if let Some(arr) = input.as_array() {
                arr.len()
            } else {
                1
            }
        } else {
            0
        };

        if let Some(max) = state.vllm_overrides.max_batched_texts_embedding {
            if count > max {
                return Err(ProxyError::Validation(format!(
                    "Too many texts for embedding: {} > maximum allowed {}",
                    count, max
                )));
            }
        }
    }

    let modified_json = serde_json::to_vec(&json)
        .map_err(|e| ProxyError::Json(format!("Failed to serialize JSON: {}", e)))?;

    forward_json_request(&state, "v1/embeddings", headers, Bytes::from(modified_json)).await
}

/// Rerank handler: validates document count and proxies to backend.
pub async fn rerank_handler(
    State(state): State<Arc<ProxyState>>,
    headers: HeaderMap,
    body: Body,
) -> Result<AxumResponse, ProxyError> {
    debug!("[RERANK] 📥 Reading rerank request body...");
    let body_bytes = match axum::body::to_bytes(body, state.max_payload_size_mb * 1024 * 1024).await {
        Ok(bytes) => bytes,
        Err(e) => {
            error!("[RERANK] ❌ Failed to read rerank body: {}", e);
            return Err(ProxyError::Internal(format!(
                "Failed to read request body: {}",
                e
            )));
        }
    };

    let mut json: Value = serde_json::from_slice(&body_bytes)
        .map_err(|e| ProxyError::Json(format!("Failed to parse JSON: {}", e)))?;

    let model_name = state.vllm_overrides.model_name.clone();

    if let Some(obj) = json.as_object_mut() {
        obj.insert("model".to_string(), Value::String(model_name));

        if let Some(docs) = obj.get("documents").and_then(|v| v.as_array()) {
            let count = docs.len();
            if let Some(max) = state.vllm_overrides.max_rerank_documents {
                if count > max {
                    return Err(ProxyError::Validation(format!(
                        "Too many documents for rerank: {} > maximum allowed {}",
                        count, max
                    )));
                }
            }
        }
    }

    let modified_json = serde_json::to_vec(&json)
        .map_err(|e| ProxyError::Json(format!("Failed to serialize JSON: {}", e)))?;

    forward_json_request(&state, "v1/rerank", headers, Bytes::from(modified_json)).await
}

/// Fallback handler for unmatched routes, returns a JSON 404 error
pub async fn not_found_handler() -> impl IntoResponse {
    let error_response = ErrorResponse {
        code: 404,
        message: "Not found".to_string(),
    };
    (StatusCode::NOT_FOUND, axum::Json(error_response))
}
