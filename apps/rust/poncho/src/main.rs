mod config;

use axum::{
    body::{Body, Bytes},
    extract::{Path, Query, State},
    http::{HeaderMap, HeaderName, HeaderValue, Method, StatusCode},
    response::{IntoResponse, Response},
    routing::any,
    Router,
};
use log::{debug, error, info, trace, warn};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{collections::HashMap, str::FromStr, sync::Arc, time::Duration};
use tokio_stream::StreamExt;
use tower::ServiceBuilder;
use tower_http::{cors::CorsLayer, limit::RequestBodyLimitLayer};

use config::Config;

#[tokio::main(flavor = "multi_thread", worker_threads = 8)]
async fn main() {
    // Load config
    let config = match Config::load() {
        Ok(config) => config,
        Err(e) => {
            eprintln!("Failed to load config: {}", e);
            std::process::exit(1);
        }
    };

    // Set log level from config
    std::env::set_var("RUST_LOG", &config.log_level);
    // Start the logger
    env_logger::init();

    // Start the server
    run_server(
        config.server.host,
        config.server.port.to_string(),
        config.vllm_backend.host,
        config.vllm_backend.port.to_string(),
        config.routing.timeout_seconds.into(),
        config.routing.max_payload_size_mb.into(),
        VllmOverrides {
            model_name: config.vllm_backend.model_name_override,
            allow_logprobs: config.vllm_backend.allow_logprobs,
        },
    )
    .await;
}

struct VllmOverrides {
    model_name: String,
    allow_logprobs: bool,
}

async fn run_server(
    server_addr: String,
    server_port: String,
    backend_addr: String,
    backend_port: String,
    tcp_timeout: u64,
    max_payload_size_mb: usize,
    vllm_overrides: VllmOverrides,
) {
    // Simple HTTP1 client
    let client = Client::builder()
        // Maximum time to wait for a complete request/response
        .timeout(Duration::from_secs(tcp_timeout))
        // Maximum time to establish a TCP connection
        .connect_timeout(Duration::from_secs(5))
        // VLLM needs no more
        .http1_only()
        // Disables Nagle's algorithm for lower latency
        .tcp_nodelay(true)
        //Bypasses system proxy settings
        .no_proxy()
        // Build it
        .build()
        // Panics if the server fails to start
        .expect("Failed to create HTTP client");

    // Wrap the HTTP client in an Arc (atomic reference counter) so it can be safely shared across multiple async tasks
    let state = Arc::new(ProxyState {
        client,
        backend_url: backend_addr,
        backend_port: backend_port,
        max_payload_size_mb: max_payload_size_mb,
    });

    // Sets up the web application routing:
    let app = Router::new()
        // Catches all paths and HTTP methods, routing them to proxy_handler
        .route("/*path", any(proxy_handler))
        // Allows cross-origin requests from any domain
        .layer(CorsLayer::permissive())
        // Limits request body size to 10 megabytes
        .layer(RequestBodyLimitLayer::new(
            max_payload_size_mb * 1024 * 1024,
        ))
        // Makes the shared client available to handlers
        .with_state(state);

    // Bind the server to the provided address
    let listener = tokio::net::TcpListener::bind(format!("{}:{}", server_addr, server_port))
        .await
        // Panics if the server fails to start
        .expect("Failed to bind to port {}");

    // Log if success
    if let Ok(socket) = listener.local_addr() {
        info!("High-performance proxy server running on {}", socket);
        info!("Worker threads: 8 (configured)");
        info!("Ready to handle high traffic loads...");
    }

    // Start the server
    axum::serve(
        listener,
        // Converts the app into a service that can access client connection info
        app.into_make_service_with_connect_info::<std::net::SocketAddr>(),
    )
    // Sets up signal handling for clean shutdown (Ctrl+C, etc.)
    .with_graceful_shutdown(shutdown_signal())
    // Runs the server until shutdown
    .await
    // Panics if the server fails to start
    .expect("Server failed");
}

// In web servers, each incoming request is handled by a separate async task.
// The ProxyState struct allows you to share the HTTP Client across all these
//  concurrent request handlers without creating a new client for each request.
#[derive(Clone)]
struct ProxyState {
    client: Client,
    backend_url: String,
    backend_port: String,
    max_payload_size_mb: usize,
}

async fn proxy_handler(
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
        match modify_json_payload(body_bytes) {
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

    // Prepare headers - only essential ones like curl
    debug!("📋 Preparing headers...");
    let mut upstream_headers = reqwest::header::HeaderMap::new();

    // Add only essential headers that curl sends
    upstream_headers.insert(
        reqwest::header::HOST,
        reqwest::header::HeaderValue::from_str(&format!(
            "{}:{}",
            target_base
                .strip_prefix("http://")
                .unwrap_or(&target_base)
                .split(':')
                .next()
                .unwrap_or("localhost"),
            target_base.split(':').last().unwrap_or("9087")
        ))
        .unwrap_or_else(|_| reqwest::header::HeaderValue::from_static("localhost:9087")),
    );

    upstream_headers.insert(
        reqwest::header::USER_AGENT,
        reqwest::header::HeaderValue::from_static("axum-proxy/1.0"),
    );

    upstream_headers.insert(
        reqwest::header::ACCEPT,
        reqwest::header::HeaderValue::from_static("*/*"),
    );

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

    // Build request exactly like curl does
    let request_builder = state
        .client
        .request(method.clone(), &full_url)
        .headers(upstream_headers)
        .body(processed_body); // Always set body, even if empty

    debug!("⏳ Sending request to vLLM...");
    debug!("🔍 About to call request_builder.send()...");

    // Add a timeout wrapper to catch hanging requests
    let request_future = request_builder.send();
    let timeout_duration = Duration::from_secs(10); // Even shorter timeout

    debug!(
        "⏰ Starting request with {} second timeout...",
        timeout_duration.as_secs()
    );

    let upstream_response = match tokio::time::timeout(timeout_duration, request_future).await {
        Ok(Ok(response)) => {
            debug!(
                "✅ Got response from vLLM: {} - Headers: {:?}",
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
                    "Connection failed to {}: {} - Check if vLLM server is running",
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
                "❌ Request timed out after {} seconds - this suggests the request is hanging",
                timeout_duration.as_secs()
            );
            return Err(ProxyError::Upstream(format!(
                "Request hung/timed out after {} seconds to {}",
                timeout_duration.as_secs(),
                full_url
            )));
        }
    };

    // Check if streaming
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
        handle_streaming_response(upstream_response).await
    } else {
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
/// returning a flag if the body requested any banned feature (like "logprobs")
fn modify_json_payload(body: Bytes) -> Result<Bytes, ProxyError> {
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
        if let Ok(model_override) = std::env::var("MODEL_OVERRIDE") {
            if obj.contains_key("model") {
                obj.insert("model".to_string(), Value::String(model_override));
            }
        }

        // Check for logprobs
        if obj.contains_key("logprobs") || obj.contains_key("prompt_logprobs") {}

        // Max tokens limits
        if let Some(max_tokens) = obj.get("max_tokens") {
            if let Some(tokens_val) = max_tokens.as_u64() {
                let max_allowed = std::env::var("MAX_TOKENS_LIMIT")
                    .ok()
                    .and_then(|s| s.parse::<u64>().ok())
                    .unwrap_or(4096);
                let clamped_tokens = tokens_val.min(max_allowed);
                obj.insert("max_tokens".to_string(), Value::from(clamped_tokens));
            }
        }

        debug!("🔧 Modified JSON payload");
    }

    let modified_json = serde_json::to_vec(&json)
        .map_err(|e| ProxyError::Json(format!("Failed to serialize JSON: {}", e)))?;

    Ok(Bytes::from(modified_json))
}

async fn handle_streaming_response(
    upstream_response: reqwest::Response,
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
    let body_stream = stream.map(|chunk_result| match chunk_result {
        Ok(chunk) => {
            debug!("📦 Streaming chunk: {} bytes", chunk.len());
            Ok(chunk)
        }
        Err(e) => {
            error!("❌ Stream error: {}", e);
            Err(std::io::Error::new(std::io::ErrorKind::Other, e))
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

#[inline]
fn should_forward_header(header_name: &str) -> bool {
    !matches!(
        header_name.to_ascii_lowercase().as_str(),
        "host"
            | "connection"
            | "upgrade"
            | "proxy-connection"
            | "proxy-authenticate"
            | "proxy-authorization"
            | "te"
            | "trailers"
            | "transfer-encoding"
    )
}

async fn shutdown_signal() {
    use tokio::signal;

    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }

    info!("Gracefully shutting down proxy server...");
}

#[derive(Debug)]
enum ProxyError {
    Internal(String),
    Upstream(String),
    Json(String),
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
        };

        (status, message).into_response()
    }
}
