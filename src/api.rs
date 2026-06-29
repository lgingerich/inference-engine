// OpenAI-compatible HTTP API.
//
// Exposes `POST /v1/chat/completions` with SSE streaming,
// plus `GET /health` and `GET /v1/models` for compatibility.

use std::convert::Infallible;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::{
    extract::State,
    http::StatusCode,
    response::sse::{Event, Sse},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_stream::{wrappers::UnboundedReceiverStream, StreamExt};
use tower_http::cors::CorsLayer;
use uuid::Uuid;

use crate::MODEL_ID;
use crate::{GenConfig, LoadedModel, generate};

// ── request types ────────────────────────────────────────────────────────────

#[allow(dead_code)]
#[derive(Deserialize)]
pub(crate) struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f64,
    #[serde(default = "default_seed")]
    pub seed: u64,
}

fn default_max_tokens() -> usize { 256 }
fn default_temperature() -> f64 { 0.8 }
fn default_seed() -> u64 { 42 }

#[derive(Deserialize)]
pub(crate) struct Message {
    pub role: String,
    pub content: String,
}

// ── response types ───────────────────────────────────────────────────────────

#[derive(Serialize)]
struct ChatCompletionChunk {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<ChoiceDelta>,
}

#[derive(Serialize)]
struct ChoiceDelta {
    index: u32,
    delta: Delta,
    #[serde(skip_serializing_if = "Option::is_none")]
    finish_reason: Option<String>,
}

#[derive(Serialize)]
struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
}

#[derive(Serialize)]
struct ModelEntry {
    id: String,
    object: String,
    created: u64,
    owned_by: String,
}

#[derive(Serialize)]
struct ModelList {
    object: String,
    data: Vec<ModelEntry>,
}

// ── chat template ───────────────────────────────────────────────────────────

/// Build a Llama 3.2 chat-formatted prompt from a list of messages.
///
/// Format: `<|begin_of_text|><|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>...<|start_header_id|>assistant<|end_header_id|>\n\n`
fn build_prompt(messages: &[Message]) -> String {
    let has_system = messages
        .iter()
        .any(|m| m.role == "system" && !m.content.trim().is_empty());

    let mut prompt = String::with_capacity(4096);
    prompt.push_str("<|begin_of_text|>");

    // Default system prompt to encourage substantive output for benchmarks
    if !has_system {
        prompt.push_str("<|start_header_id|>system<|end_header_id|>\n\n");
        prompt.push_str(
            "You are a helpful, verbose AI assistant. Always provide detailed, \
             thorough, multi-paragraph responses. Never respond with a single word or sentence.",
        );
        prompt.push_str("<|eot_id|>");
    }

    for msg in messages {
        // Skip empty system messages (llmperf compatibility)
        if msg.role == "system" && msg.content.trim().is_empty() {
            continue;
        }
        prompt.push_str("<|start_header_id|>");
        prompt.push_str(&msg.role);
        prompt.push_str("<|end_header_id|>\n\n");
        prompt.push_str(&msg.content);
        prompt.push_str("<|eot_id|>");
    }
    prompt.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
    prompt
}

// ── server state ─────────────────────────────────────────────────────────────

pub(crate) struct AppState {
    pub model: Mutex<LoadedModel>,
}

// ── handlers ─────────────────────────────────────────────────────────────────

async fn health() -> StatusCode {
    StatusCode::OK
}

async fn list_models() -> Json<ModelList> {
    Json(ModelList {
        object: "list".into(),
        data: vec![ModelEntry {
            id: MODEL_ID.into(),
            object: "model".into(),
            created: 0,
            owned_by: "local".into(),
        }],
    })
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Sse<impl futures_core::Stream<Item = Result<Event, Infallible>>> {
    let prompt = build_prompt(&req.messages);
    let gen_cfg = GenConfig {
        max_new_tokens: req.max_tokens,
        seed: req.seed,
        temperature: if req.temperature <= 0.0 { 1e-6 } else { req.temperature },
    };

    let completion_id = format!("chatcmpl-{}", Uuid::new_v4());
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let model_name = req.model.clone();

    // Channel: generation thread sends token text, API streams as SSE
    let (tx, rx) = mpsc::unbounded_channel::<String>();

    let state_arc = state.clone();
    let id_for_thread = completion_id.clone();
    let model_name_for_thread = model_name.clone();

    // Run generation on a blocking thread so it doesn't stall the async runtime
    tokio::task::spawn_blocking(move || {
        let mut model = state_arc.model.lock().unwrap();
        // Fresh KV cache per request — avoids contamination from previous runs
        let mut cache = match model.new_cache() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("cache creation failed: {e}");
                return;
            }
        };
        let _ = generate(
            &mut model,
            &mut cache,
            &prompt,
            &gen_cfg,
            &mut |text| {
                let _ = tx.send(text.to_string());
            },
        );
        // tx dropped here → channel closes → SSE stream ends
    });

    // Step 1: role chunk (sent immediately, before any content)
    let id0 = id_for_thread.clone();
    let role_chunk = ChatCompletionChunk {
        id: id0,
        object: "chat.completion.chunk".into(),
        created,
        model: model_name_for_thread.clone(),
        choices: vec![ChoiceDelta {
            index: 0,
            delta: Delta {
                role: Some("assistant".into()),
                content: None,
            },
            finish_reason: None,
        }],
    };

    let role_event = Event::default()
        .data(serde_json::to_string(&role_chunk).unwrap());

    // Step 2: token stream from generation
    let id1 = completion_id.clone();
    let model1 = model_name.clone();
    let token_stream = UnboundedReceiverStream::new(rx).map(move |text| {
        let chunk = ChatCompletionChunk {
            id: id1.clone(),
            object: "chat.completion.chunk".into(),
            created,
            model: model1.clone(),
            choices: vec![ChoiceDelta {
                index: 0,
                delta: Delta {
                    role: None,
                    content: Some(text),
                },
                finish_reason: None,
            }],
        };
        Ok(Event::default().data(serde_json::to_string(&chunk).unwrap()))
    });

    // Step 3: stop chunk + [DONE]
    let id2 = completion_id.clone();
    let model2 = model_name.clone();
    let stop_chunk = ChatCompletionChunk {
        id: id2,
        object: "chat.completion.chunk".into(),
        created,
        model: model2,
        choices: vec![ChoiceDelta {
            index: 0,
            delta: Delta {
                role: None,
                content: None,
            },
            finish_reason: Some("stop".into()),
        }],
    };

    let stop_event = Event::default()
        .data(serde_json::to_string(&stop_chunk).unwrap());

    let done_event = Event::default().data("[DONE]");

    let stream = tokio_stream::iter(vec![Ok(role_event)])
        .chain(token_stream)
        .chain(tokio_stream::iter(vec![Ok(stop_event), Ok(done_event)]));

    Sse::new(stream)
}

// ── server entry point ───────────────────────────────────────────────────────

pub(crate) async fn serve(model: LoadedModel, port: u16) -> anyhow::Result<()> {
    let state = Arc::new(AppState {
        model: Mutex::new(model),
    });

    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let addr = format!("0.0.0.0:{port}");
    println!("listening on http://{addr}");
    println!("  POST /v1/chat/completions  (OpenAI-compatible, SSE streaming)");
    println!("  GET  /v1/models");
    println!("  GET  /health");

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
