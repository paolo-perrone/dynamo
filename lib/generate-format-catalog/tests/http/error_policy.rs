// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use dynamo_kv_router::overload_status::overload_status_code;
use dynamo_kv_router::scheduling::{QueueLimitKind, QueueRejection};
use dynamo_llm::generate_format::ErrorPhase;
use dynamo_runtime::error::{ErrorReason, ErrorType as Backend};

/// Fails with a chain of these kinds, outermost first; with none, with the
/// router's typed queue-limit refusal, which only dispatch returns.
struct ErrorEngine(Vec<Backend>, bool);

fn limit() -> QueueRejection {
    QueueRejection {
        policy_class: "latency".to_string(),
        limit_kind: QueueLimitKind::Requests,
        current: 4,
        limit: 4,
    }
}

impl ErrorEngine {
    fn error(&self) -> DynamoError {
        let mut cause = None;
        for kind in self.0.iter().rev() {
            let mut error = DynamoError::builder()
                .error_type(*kind)
                .message("invalid params");
            if *kind == Backend::DeadlineExceeded {
                error = error.reason(ErrorReason::new("router.queue_deadline_exceeded").unwrap());
            }
            if let Some(cause) = cause {
                error = error.cause(cause);
            }
            cause = Some(error.build());
        }
        cause.unwrap()
    }
}

#[async_trait::async_trait]
impl AsyncEngine<Input, Output, Error> for ErrorEngine {
    async fn generate(&self, request: Input) -> Result<Output, Error> {
        if self.0.is_empty() {
            return Err(limit().into());
        }
        let error = self.error();
        if !self.1 {
            return Err(error.into());
        }
        Ok(ResponseStream::new(
            Box::pin(futures::stream::iter([Annotated::from_err(error)])),
            request.context(),
        ))
    }
}

// Literal outcomes from the two handlers at 5235cfc93d, with router class queue
// drops as the CoDel handlers answered them: before dispatch, the configured
// overload status as `Overload` without a rejection; in an SGLang response
// stream, a 503 `Overload`. Each row gives dispatch, SGLang response-stream, and
// vLLM response-stream outcomes. `DYN_HTTP_OVERLOAD_STATUS_CODE` is read once,
// so check other statuses in a fresh process.
#[tokio::test]
async fn native_error_phase_and_nested_chain_compatibility() {
    use ErrorType::{Cancelled, Internal, Overload, Unavailable, Validation};
    let overload = overload_status_code();
    let limit_message = limit().to_string();
    for (kinds, expected) in [
        (
            vec![Backend::DeadlineExceeded],
            [
                (overload, Overload, "request deadline exceeded"),
                (503, Overload, "request deadline exceeded"),
                (500, Internal, "internal server error"),
            ],
        ),
        (
            vec![Backend::WorkerOverloaded],
            [
                (503, Unavailable, "engine rejected the request"),
                (503, Overload, "Service temporarily overloaded"),
                (500, Internal, "internal server error"),
            ],
        ),
        (
            vec![Backend::WorkerUnavailable],
            [
                (503, Unavailable, "Service temporarily unavailable"),
                (503, Unavailable, "Service temporarily unavailable"),
                (503, Unavailable, "Service temporarily unavailable"),
            ],
        ),
        (
            vec![Backend::Cancelled],
            [
                (499, Cancelled, "request was cancelled"),
                (499, Cancelled, "Request cancelled"),
                (499, Cancelled, "request was cancelled"),
            ],
        ),
        (
            vec![Backend::WorkerUnavailable, Backend::Cancelled],
            [
                (499, Cancelled, "request was cancelled"),
                (503, Unavailable, "Service temporarily unavailable"),
                (499, Cancelled, "request was cancelled"),
            ],
        ),
        (
            vec![Backend::InvalidArgument, Backend::Cancelled],
            [
                (499, Cancelled, "request was cancelled"),
                (400, Validation, "invalid params"),
                (499, Cancelled, "request was cancelled"),
            ],
        ),
        (
            vec![Backend::WorkerOverloaded, Backend::DeadlineExceeded],
            [
                (overload, Overload, "request deadline exceeded"),
                (503, Overload, "request deadline exceeded"),
                (500, Internal, "internal server error"),
            ],
        ),
        (
            vec![Backend::DeadlineExceeded, Backend::Cancelled],
            [
                (overload, Overload, "request deadline exceeded"),
                (503, Overload, "request deadline exceeded"),
                (499, Cancelled, "request was cancelled"),
            ],
        ),
        (
            vec![],
            [
                (overload, Overload, limit_message.as_str()),
                (503, Overload, limit_message.as_str()),
                (500, Internal, "internal server error"),
            ],
        ),
    ] {
        for is_sglang in [false, true] {
            for after_dispatch in [false, true] {
                for streaming in [false, true].into_iter().filter(|s| is_sglang || !s) {
                    let (status, metric, message) = expected[if !after_dispatch {
                        0
                    } else if is_sglang {
                        1
                    } else {
                        2
                    }]
                    .clone();
                    let state = state();
                    let format: SharedFormat = if is_sglang {
                        sglang()
                    } else {
                        Arc::new(dynamo_vllm_generate::VllmFormat)
                    };
                    if after_dispatch && kinds.is_empty() {
                        // No response stream carries the typed refusal; classify it directly.
                        let error =
                            format.worker_error(&limit(), ErrorPhase::Stream { cancelled: false });
                        assert_eq!(
                            (error.status.as_u16(), error.kind, error.message.as_str()),
                            (status, metric, message)
                        );
                        continue;
                    }
                    let body = if is_sglang {
                        serde_json::json!({"input_ids":[1],"stream":streaming})
                    } else {
                        serde_json::json!({"token_ids":[1],"sampling_params":{}})
                    };
                    let response = respond(
                        state.clone(),
                        format,
                        Arc::new(ErrorEngine(kinds.clone(), after_dispatch)),
                        body,
                        HeaderMap::new(),
                    )
                    .await;
                    assert_eq!(
                        response.status().as_u16(),
                        if streaming && after_dispatch {
                            200
                        } else {
                            status
                        },
                        "{kinds:?}"
                    );
                    assert!(!response.headers().contains_key("retry-after"));
                    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                        .await
                        .unwrap();
                    let body = String::from_utf8(bytes.to_vec()).unwrap();
                    let body = if streaming && after_dispatch {
                        body.lines()
                            .find_map(|line| line.strip_prefix("data: "))
                            .unwrap()
                    } else {
                        &body
                    };
                    let body: Value = serde_json::from_str(body).unwrap();
                    let expected_body = if is_sglang {
                        serde_json::json!({"error":{"message":message}})
                    } else {
                        let kind = match status {
                            // A queue drop names its cause at any configured status.
                            _ if !after_dispatch && metric == Overload => {
                                if kinds.is_empty() {
                                    "queue_rejected"
                                } else {
                                    "deadline_exceeded"
                                }
                            }
                            400 => "invalid_request_error",
                            429 => "deadline_exceeded",
                            499 => "request_cancelled",
                            503 => "service_unavailable",
                            _ => "internal_error",
                        };
                        serde_json::json!({"error":{"message":message,"type":kind,"code":status}})
                    };
                    assert_eq!(body, expected_body, "{kinds:?}");
                    assert_eq!(
                        state.metrics_clone().get_request_counter(
                            state.manager().metric_model_for("test-model"),
                            &Endpoint::Generate,
                            &if streaming {
                                RequestType::Stream
                            } else {
                                RequestType::Unary
                            },
                            &Status::Error,
                            &metric,
                        ),
                        1,
                        "{kinds:?}"
                    );
                    assert_eq!(
                        state
                            .metrics_clone()
                            .get_rejection_count("test-model", Endpoint::Generate),
                        u64::from(!after_dispatch && kinds == [Backend::WorkerOverloaded])
                    );
                }
            }
        }
    }
}
