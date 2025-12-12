use axum::{
    body::Body,
    extract::{Path, Query, State},
    http::{HeaderMap, Method, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use log::{debug, error};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc};

#[derive(Clone)]
pub struct PoktState {
    pub model_data: PoktModelData,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct PoktModelData {
    pub model_public_name: String,
    pub max_position_embeddings: String,
    pub max_tokens: String,
}

pub enum PoktError {
    Validation(String),
}

impl IntoResponse for PoktError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            PoktError::Validation(msg) => {
                error!("❌ Validation error: {}", msg);
                (StatusCode::BAD_REQUEST, msg)
            }
        };

        (status, message).into_response()
    }
}

pub async fn pokt_handler(
    State(state): State<Arc<PoktState>>,
    Path(path): Path<String>,
    Query(_params): Query<HashMap<String, String>>,
    method: Method,
    _headers: HeaderMap,
    _body: Body,
) -> Result<Response, PoktError> {
    debug!("🎯 POKT");

    if method == Method::GET {
        if path == "config" {
            Ok(Json(state.model_data.clone()).into_response())
        } else {
            return Err(PoktError::Validation(format!(
                "path \"{}\" not implemented",
                path
            )));
        }
    } else {
        return Err(PoktError::Validation(format!("invalid request method")));
    }
}
