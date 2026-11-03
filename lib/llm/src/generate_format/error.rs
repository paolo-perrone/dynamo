// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use axum::http::StatusCode;

use crate::http::service::{
    error::{SanitizedError, overload_status_code},
    metrics::*,
};

/// Whether dispatch failed or a returned response stream failed.
#[derive(Clone, Copy)]
pub enum ErrorPhase {
    Dispatch,
    Stream { cancelled: bool },
}

/// Why a router class queue dropped a request before dispatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueueDrop {
    /// Its due time passed while it waited.
    DeadlineExceeded,
    /// A class limit refused it.
    LimitReached,
}

/// A format's public worker error and its request-metric category.
pub struct WorkerError {
    pub kind: ErrorType,
    pub status: StatusCode,
    pub message: String,
    /// A router class queue drop: an `Overload` that the frontend counts as neither a
    /// rejection nor a cancellation, whatever overload status is configured.
    pub queue_drop: Option<QueueDrop>,
}

impl WorkerError {
    pub fn new(kind: ErrorType, status: StatusCode, message: impl ToString) -> Self {
        Self {
            kind,
            status,
            message: message.to_string(),
            queue_drop: None,
        }
    }

    /// The configured overload status (`DYN_HTTP_OVERLOAD_STATUS_CODE`, 529 by default).
    fn dropped(drop: QueueDrop, message: impl ToString) -> Self {
        Self {
            queue_drop: Some(drop),
            ..Self::new(ErrorType::Overload, overload_status_code(), message)
        }
    }

    /// Public validation detail from the canonical error chain, with private
    /// diagnostics redacted. Used by formats following the OpenAI policy.
    pub fn canonical_invalid_request_message(
        error: &(dyn std::error::Error + 'static),
    ) -> Option<String> {
        crate::http::service::error::find_canonical_error_in_chain(error)
            .filter(|error| error.class() == dynamo_runtime::error::ErrorClass::InvalidRequest)
            .map(|error| {
                error
                    .public_message()
                    .unwrap_or("Invalid request")
                    .to_string()
            })
    }

    /// Common pre-stream policy of the native Generate endpoints. The format
    /// supplies its validation message; transport failures take precedence.
    /// A router class queue drop comes first, the deadline before the limit, as
    /// on the OpenAI surface. Dispatch counts any other `Overload` as an
    /// unavailable request plus a rejection.
    pub fn dispatch(error: &(dyn std::error::Error + 'static), invalid: Option<String>) -> Self {
        if request_deadline_exceeded(error) {
            Self::dropped(
                QueueDrop::DeadlineExceeded,
                REQUEST_DEADLINE_EXCEEDED_MESSAGE,
            )
        } else if let Some(rejection) = queue_rejection(error) {
            Self::dropped(QueueDrop::LimitReached, rejection)
        } else if request_was_cancelled(error) {
            Self::new(
                ErrorType::Cancelled,
                StatusCode::from_u16(499).unwrap(),
                "request was cancelled",
            )
        } else if request_was_rejected(error) {
            Self::new(
                ErrorType::Overload,
                StatusCode::SERVICE_UNAVAILABLE,
                "engine rejected the request",
            )
        } else if request_was_unavailable(error) {
            Self::new(
                ErrorType::Unavailable,
                StatusCode::SERVICE_UNAVAILABLE,
                SanitizedError::Unavailable,
            )
        } else if let Some(message) = invalid {
            Self::new(ErrorType::Validation, StatusCode::BAD_REQUEST, message)
        } else {
            Self::new(
                ErrorType::Internal,
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal server error",
            )
        }
    }
}
