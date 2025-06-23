use axum::{
    body::{Body, Bytes},
    extract::{Path, Query, State},
    http::{HeaderMap, HeaderName, HeaderValue, Method, StatusCode},
    response::{IntoResponse, Response},
};
use log::{debug, error, warn};
use reqwest::Client;
use serde_json::Value;
use std::{collections::HashMap, str::FromStr, sync::Arc, time::Duration};
use tokio_stream::StreamExt;

#[derive(Clone)]
pub struct VllmOverrides {
    pub model_name: String,
    pub allow_logprobs: bool,
    pub max_tokens: u64,
    pub overriden_name: String,
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

impl IntoResponse for ProxyError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            ProxyError::Internal(msg) => {
                error!("❌ Internal error: {}", msg);
                (StatusCode::INTERNAL_SERVER_ERROR, msg)
            }
            ProxyError::Upstream(msg) => {
                error!("❌ Upstream error: {}", msg);
                (StatusCode::BAD_GATEWAY, msg)
            }
            ProxyError::Json(msg) => {
                error!("❌ JSON error: {}", msg);
                (StatusCode::BAD_REQUEST, msg)
            }
            ProxyError::Validation(msg) => {
                error!("❌ Validation error: {}", msg);
                (StatusCode::BAD_REQUEST, msg)
            }
        };

        (status, message).into_response()
    }
}

pub async fn proxy_handler(
    State(state): State<Arc<ProxyState>>,
    Path(path): Path<String>,
    Query(params): Query<HashMap<String, String>>,
    method: Method,
    headers: HeaderMap,
    body: Body,
) -> Result<Response, ProxyError> {
    // Set the backend target endpoint
    let target_base = format!("{}:{}", state.backend_url, state.backend_port);
    // Append the target path
    let target_url = format!("{}/{}", target_base, path);

    debug!("🎯 Proxying {} {} -> {}", method, path, target_url);

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
    debug!("📥 Reading request body...");
    let body_bytes = match axum::body::to_bytes(body, state.max_payload_size_mb * 1024 * 1024).await
    {
        Ok(bytes) => {
            debug!("✅ Body read successfully: {} bytes", bytes.len());
            bytes
        }
        Err(e) => {
            error!("❌ Failed to read body: {}", e);
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
        debug!("🔧 Modifying request body...");
        match modify_json_payload(body_bytes, state.vllm_overrides.clone()) {
            Ok(modified) => {
                debug!("✅ Body modified successfully");
                modified
            }
            Err(e) => {
                warn!("❌ Failed to modify body: {:?}", e);
                return Err(e);
            }
        }
    } else {
        body_bytes
    };

    // Prepare headers
    debug!("📋 Preparing headers...");
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
    debug!("📋 Headers to send:");
    for (name, value) in upstream_headers.iter() {
        debug!("   {}: {}", name, value.to_str().unwrap_or("[unprintable]"));
    }
    debug!("🚀 Making upstream request...");
    debug!("   URL: {}", full_url);
    debug!("   Method: {}", method);
    debug!("   Body size: {} bytes", processed_body.len());

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

    debug!("⏳ Sending request to vLLM...");
    debug!("🔍 About to call request_builder.send()...");

    // Add a timeout wrapper to catch hanging requests
    let request_future = request_builder.send();
    let timeout_duration = Duration::from_secs(state.timeout);

    debug!(
        "⏰ Starting request with {} second timeout...",
        timeout_duration.as_secs()
    );

    // Await the request future
    let upstream_response = match tokio::time::timeout(timeout_duration, request_future).await {
        Ok(Ok(response)) => {
            debug!(
                "✅ Got response from backend: {} - Headers: {:?}",
                response.status(),
                response.headers()
            );
            response
        }
        Ok(Err(e)) => {
            error!("❌ Request failed after send(): {:?}", e);
            error!("❌ Is timeout: {}", e.is_timeout());
            error!("❌ Is connect: {}", e.is_connect());
            error!("❌ Is request: {}", e.is_request());
            error!("❌ Is decode: {}", e.is_decode());

            if e.is_timeout() {
                return Err(ProxyError::Upstream(format!(
                    "Request timeout to {}: {}",
                    full_url, e
                )));
            } else if e.is_connect() {
                return Err(ProxyError::Upstream(format!(
                    "Connection failed to {}: {} - Check if backend vLLM server is running",
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
                "❌ Request timed out after {} seconds",
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

    println!(
        "📦 Response type: {}",
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
        handle_regular_response(upstream_response).await
    }
}

/// Checks whether a request body should be modified or not
fn should_modify_request(method: &Method, path: &str, headers: &HeaderMap) -> bool {
    // If the method is POST
    method == Method::POST
        // if the path is either a completions or chat completions
        && (path.contains("completions") || path.contains("chat"))
        // and if the content type is application/json
        && headers
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .map(|v| v.contains("application/json"))
            .unwrap_or(false)
}

/// Modifies the json payload of a vllm request, overridingthe model name and
/// raising an error if logprobs are requested but not allowed
fn modify_json_payload(body: Bytes, vllm_overrides: VllmOverrides) -> Result<Bytes, ProxyError> {
    // just in case empty body check
    if body.is_empty() {
        return Ok(body);
    }

    // Read all the body as a json, it should be a json
    let mut json: Value = serde_json::from_slice(&body)
        .map_err(|e| ProxyError::Json(format!("Failed to parse JSON: {}", e)))?;

    // if we can get a mutable jsojn hashmap, go ahead
    if let Some(obj) = json.as_object_mut() {
        // Model override
        if obj.contains_key("model") {
            obj.insert(
                "model".to_string(),
                Value::String(vllm_overrides.model_name),
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

        // Check for max_tokens
        if let Some(max_tokens) = obj.get("max_tokens") {
            if let Some(max_tokens_val) = max_tokens.as_u64() {
                if max_tokens_val > vllm_overrides.max_tokens {
                    // TODO : Account for context
                    obj.insert(
                        "max_tokens".to_string(),
                        Value::from(vllm_overrides.max_tokens),
                    );
                }
            }
        }

        debug!("🔧 Modified JSON payload");
    }

    let modified_json = serde_json::to_vec(&json)
        .map_err(|e| ProxyError::Json(format!("Failed to serialize JSON: {}", e)))?;

    Ok(Bytes::from(modified_json))
}

/// Stream response handler, will send the proxied request response as it arrives
async fn handle_streaming_response(
    upstream_response: reqwest::Response,
    new_model_name: String,
) -> Result<Response, ProxyError> {
    let status = upstream_response.status();
    let headers = upstream_response.headers().clone();

    debug!("🌊 Handling streaming response with status: {}", status);

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
                debug!("📦 Streaming chunk: {} bytes", chunk.len());

                // Try to parse and modify the JSON chunk
                match modify_json_chunk(&chunk, &new_model_name) {
                    Ok(modified_chunk) => {
                        debug!("✏️ Modified JSON chunk: {} bytes", modified_chunk.len());
                        Ok(modified_chunk)
                    }
                    Err(e) => {
                        // If parsing fails, pass through original chunk
                        // This handles cases where chunk might not be complete JSON
                        debug!("⚠️ Failed to parse chunk as JSON (passing through): {}", e);
                        Ok(chunk)
                    }
                }
            }
            Err(e) => {
                error!("❌ Stream error: {}", e);
                Err(std::io::Error::new(std::io::ErrorKind::Other, e))
            }
        }
    });

    let body = Body::from_stream(body_stream);

    let mut response = Response::builder().status(status.as_u16());

    for (name, value) in response_headers.iter() {
        response = response.header(name, value);
    }

    debug!("✅ Streaming response built successfully");

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
        // Not SSE format, probably not vLLM, do not modify
        return Ok(chunk.clone());
    }
}

/// Regular response handler, will receive the response and then send it back to
/// the proxied source
async fn handle_regular_response(
    upstream_response: reqwest::Response,
) -> Result<Response, ProxyError> {
    let status = upstream_response.status();
    let headers = upstream_response.headers().clone();

    debug!("📄 Handling regular response with status: {}", status);

    let body_bytes = upstream_response
        .bytes()
        .await
        .map_err(|e| ProxyError::Upstream(format!("Failed to read response body: {}", e)))?;

    debug!("📥 Response body: {} bytes", body_bytes.len());

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

    let mut response = Response::builder().status(status.as_u16());

    for (name, value) in response_headers.iter() {
        response = response.header(name, value);
    }

    debug!("✅ Regular response built successfully");

    response
        .body(Body::from(body_bytes))
        .map_err(|e| ProxyError::Internal(format!("Failed to build regular response: {}", e)))
}
