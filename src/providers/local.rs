//! An opt-in OpenAI-compatible Chat Completions backend.
use super::opencode::{chat, client::OpenCodeClient, model::EndpointKind};
use crate::{
    anthropic::{error::json_error, schema::MessagesRequest},
    provider::{
        CliHandlers, Generation, GenerationBody, Provider, ProviderError, ProviderErrorKind,
        RequestContext,
    },
};
use async_trait::async_trait;
use axum::{
    Json,
    http::StatusCode,
    response::{IntoResponse, Response},
};

pub struct LocalProvider;
pub struct InferxProvider;
pub fn advertised_models() -> Vec<String> {
    std::env::var("CCP_LOCAL_MODELS")
        .unwrap_or_default()
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| format!("local/{s}"))
        .collect()
}
pub fn inferx_models() -> Vec<String> {
    std::env::var("CCP_INFERX_MODELS")
        .unwrap_or_default()
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .collect()
}
fn failure(error: impl std::fmt::Display) -> ProviderError {
    ProviderError::new(
        StatusCode::BAD_GATEWAY,
        ProviderErrorKind::Api,
        error.to_string(),
    )
}
fn error_response(error: ProviderError) -> Response {
    json_error(error.status, error.error_type(), error.message)
}
#[async_trait]
impl Provider for LocalProvider {
    fn name(&self) -> &'static str {
        "local"
    }
    fn supported_models(&self) -> Vec<String> {
        advertised_models()
    }
    fn cli(&self) -> &'static dyn CliHandlers {
        &LOCAL_CLI
    }
    async fn handle_count_tokens(&self, body: MessagesRequest, _: RequestContext) -> Response {
        Json(serde_json::json!({"input_tokens": super::kimi::count_tokens::count_tokens(&body)}))
            .into_response()
    }
    async fn handle_messages(&self, body: MessagesRequest, ctx: RequestContext) -> Response {
        if body.stream {
            return match self.generate_anthropic_stream(body, ctx).await {
                Ok(generation) => {
                    let GenerationBody::LiveSse(body) = generation.body else {
                        unreachable!()
                    };
                    (
                        [
                            (http::header::CONTENT_TYPE, "text/event-stream"),
                            (http::header::CACHE_CONTROL, "no-cache"),
                        ],
                        body,
                    )
                        .into_response()
                }
                Err(error) => error_response(error),
            };
        }
        let requested = body.model.clone().unwrap_or_default();
        match upstream(&body, &ctx, false).await {
            Ok(response) => match response.into_bytes().await {
                Ok(bytes) => match chat::accumulate_response(
                    &bytes,
                    &format!("msg_{}", uuid::Uuid::new_v4().simple()),
                    &requested,
                ) {
                    Ok(value) => Json(value).into_response(),
                    Err(error) => error_response(failure(error)),
                },
                Err(error) => error_response(failure(error.message)),
            },
            Err(error) => error_response(error),
        }
    }
    async fn generate_anthropic_stream(
        &self,
        body: MessagesRequest,
        ctx: RequestContext,
    ) -> Result<Generation, ProviderError> {
        let requested = body.model.clone().unwrap_or_default();
        let response = upstream(&body, &ctx, false).await?;
        Ok(Generation {
            body: GenerationBody::LiveSse(chat::stream_body(
                response,
                format!("msg_{}", uuid::Uuid::new_v4().simple()),
                requested.clone(),
                ctx.monitor,
                ctx.req_id,
                ctx.traffic,
            )),
            resolved_model: requested,
        })
    }
}
#[async_trait]
impl Provider for InferxProvider {
    fn name(&self) -> &'static str {
        "inferx"
    }
    fn supported_models(&self) -> Vec<String> {
        inferx_models()
    }
    fn cli(&self) -> &'static dyn CliHandlers {
        &LOCAL_CLI
    }
    async fn handle_count_tokens(&self, body: MessagesRequest, _: RequestContext) -> Response {
        Json(serde_json::json!({"input_tokens": super::kimi::count_tokens::count_tokens(&body)}))
            .into_response()
    }
    async fn handle_messages(&self, body: MessagesRequest, ctx: RequestContext) -> Response {
        if body.stream {
            return match self.generate_anthropic_stream(body, ctx).await {
                Ok(generation) => {
                    let GenerationBody::LiveSse(body) = generation.body else {
                        unreachable!()
                    };
                    (
                        [
                            (http::header::CONTENT_TYPE, "text/event-stream"),
                            (http::header::CACHE_CONTROL, "no-cache"),
                        ],
                        body,
                    )
                        .into_response()
                }
                Err(error) => error_response(error),
            };
        }
        let requested = body.model.clone().unwrap_or_default();
        match upstream(&body, &ctx, true).await {
            Ok(response) => match response.into_bytes().await {
                Ok(bytes) => match chat::accumulate_response(
                    &bytes,
                    &format!("msg_{}", uuid::Uuid::new_v4().simple()),
                    &requested,
                ) {
                    Ok(value) => Json(value).into_response(),
                    Err(error) => error_response(failure(error)),
                },
                Err(error) => error_response(failure(error.message)),
            },
            Err(error) => error_response(error),
        }
    }
    async fn generate_anthropic_stream(
        &self,
        body: MessagesRequest,
        ctx: RequestContext,
    ) -> Result<Generation, ProviderError> {
        let requested = body.model.clone().unwrap_or_default();
        let response = upstream(&body, &ctx, true).await?;
        Ok(Generation {
            body: GenerationBody::LiveSse(chat::stream_body(
                response,
                format!("msg_{}", uuid::Uuid::new_v4().simple()),
                requested.clone(),
                ctx.monitor,
                ctx.req_id,
                ctx.traffic,
            )),
            resolved_model: requested,
        })
    }
}
async fn upstream(
    body: &MessagesRequest,
    ctx: &RequestContext,
    inferx: bool,
) -> Result<super::opencode::client::OpenCodeResponse, ProviderError> {
    let requested = body.model.as_deref().unwrap_or_default();
    let model = if inferx {
        Some(requested).filter(|_| inferx_models().iter().any(|m| m == requested))
    } else {
        requested
            .strip_prefix("local/")
            .filter(|_| advertised_models().iter().any(|m| m == requested))
    }
    .ok_or_else(|| {
        ProviderError::new(
            StatusCode::BAD_REQUEST,
            ProviderErrorKind::InvalidRequest,
            "Unknown local model; configure CCP_LOCAL_MODELS",
        )
    })?;
    let mut translated = chat::prepare_request(body, model).map_err(|error| {
        ProviderError::new(
            StatusCode::BAD_REQUEST,
            ProviderErrorKind::InvalidRequest,
            error.to_string(),
        )
    })?;
    translated.reasoning_effort = None;
    let client = OpenCodeClient::new(
        if inferx {
            std::env::var("CCP_INFERX_BASE_URL")
                .unwrap_or_else(|_| "https://model.inferx.net/endpoints/v1".into())
        } else {
            std::env::var("CCP_LOCAL_BASE_URL")
                .unwrap_or_else(|_| "http://127.0.0.1:1234/v1".into())
        },
        Some(if inferx {
            std::env::var("CCP_INFERX_API_KEY").unwrap_or_default()
        } else {
            std::env::var("CCP_LOCAL_API_KEY").unwrap_or_else(|_| "local".into())
        }),
    )
    .map_err(failure)?;
    if let Some(monitor) = ctx.monitor.as_ref() {
        monitor.model_resolved(&ctx.req_id, model);
        monitor.upstream_started(&ctx.req_id);
    }
    client
        .post(
            EndpointKind::ChatCompletions,
            &translated,
            true,
            ctx.traffic.clone(),
            ctx.session_id.as_deref(),
        )
        .await
        .map_err(|error| {
            let mut mapped =
                ProviderError::new(error.status, ProviderErrorKind::Api, error.message);
            mapped.retry_after = error.retry_after;
            mapped
        })
}
struct LocalCli;
static LOCAL_CLI: LocalCli = LocalCli;
impl CliHandlers for LocalCli {
    fn login(&self) -> anyhow::Result<()> {
        anyhow::bail!("Configure CCP_LOCAL_API_KEY if your server requires authentication")
    }
    fn device(&self) -> anyhow::Result<()> {
        self.login()
    }
    fn status(&self) -> anyhow::Result<()> {
        Ok(())
    }
    fn logout(&self) -> anyhow::Result<()> {
        Ok(())
    }
}
