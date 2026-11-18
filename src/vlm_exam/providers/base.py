# Copyright 2026 Roboflow, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import logging
import os
import random
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, TypeVar

import httpx
from PIL import Image

try:
    import httpx2
except ImportError:
    httpx2 = None

_logger = logging.getLogger(__name__)

_T = TypeVar("_T")

REQUEST_TIMEOUT_SECONDS = 600.0
"""Per-request wall-clock timeout applied to every provider SDK call."""

MAX_RETRIES = 3
"""Retries attempted after the first try on a transient (retryable) error."""

EMPTY_RESPONSE_TEXT = "[model returned no content]"
"""Sentinel recorded when a provider returns a blocked or empty response."""

RETRY_INITIAL_DELAY_SECONDS = 2.0
"""First backoff delay applied when retrying a transient error."""

RETRY_BACKOFF_BASE = 2.0
"""Multiplier applied to the delay after each retry."""

RETRY_JITTER_SECONDS = 0.5
"""Maximum random jitter added to each backoff delay."""
_RETRYABLE_STATUS_CODES = frozenset({408, 425, 500, 502, 503, 504, 529})
_RETRYABLE_ERROR_NAMES = frozenset(
    {
        "APITimeoutError",
        "APIConnectionError",
        "InternalServerError",
        "OverloadedError",
        "ServerError",
    }
)


def is_retryable_error(error: Exception) -> bool:
    """Report whether an error is a transient failure worth retrying.

    Covers request timeouts, connection/transport failures, and server-side
    5xx responses across provider SDKs using ``httpx`` or ``httpx2``.
    Rate-limit errors are deliberately excluded; those are handled by
    :class:`~vlm_exam.providers.fallback.FallbackProvider` route failover.

    Args:
        error: Exception raised by a provider call.

    Returns:
        ``True`` when the call should be retried.
    """
    if isinstance(error, (TimeoutError, httpx.TimeoutException, httpx.TransportError)):
        return True
    if httpx2 is not None and isinstance(error, httpx2.TransportError):
        return True
    # openai/anthropic expose the HTTP status as `status_code`; google-genai
    # exposes it as `code`.
    for attribute in ("status_code", "code"):
        if getattr(error, attribute, None) in _RETRYABLE_STATUS_CODES:
            return True
    return type(error).__name__ in _RETRYABLE_ERROR_NAMES


@dataclass(frozen=True)
class RetryStats:
    """Telemetry describing a (possibly retried) provider call.

    Attributes:
        attempts: Total attempts made, including the successful one (``1``
            when the call succeeded on the first try).
        inference_seconds: Wall-clock duration of the successful attempt
            only, excluding failed attempts and backoff sleeps.
        transient_error_types: Class names of the retryable errors caught
            before the call finally succeeded, in the order they occurred.
    """

    attempts: int
    inference_seconds: float
    transient_error_types: tuple[str, ...] = ()


def error_diagnostics(error: Exception) -> dict[str, Any]:
    """Return bounded provider diagnostics with credentials and payloads redacted."""
    message = str(error)
    for name, value in os.environ.items():
        if value and any(
            word in name.upper() for word in ("KEY", "TOKEN", "SECRET", "PASSWORD")
        ):
            message = message.replace(value, "[redacted]")
    message = re.sub(r"(?i)bearer\s+[^\s'\"]+", "Bearer [redacted]", message)
    message = re.sub(
        r"(?i)(api[_-]?key|token|authorization)([\"'\s:=]+)[^\s,}]+",
        r"\1\2[redacted]",
        message,
    )
    message = re.sub(
        r"data:image/[^;]+;base64,[A-Za-z0-9+/=]+", "[image payload]", message
    )
    result: dict[str, Any] = {
        "error_type": type(error).__name__,
        "message": message[:2000],
    }
    status = getattr(error, "status_code", getattr(error, "code", None))
    if isinstance(status, int):
        result["http_status"] = status
    request_id = getattr(error, "request_id", None)
    if isinstance(request_id, str):
        result["request_id"] = request_id[:200]
    return result


def _retry_after_seconds(error: Exception) -> float:
    headers = getattr(getattr(error, "response", None), "headers", {}) or {}
    value = headers.get("retry-after")
    if value is None:
        return 0.0
    try:
        seconds = float(value)
    except (ValueError, TypeError):
        try:
            seconds = (
                parsedate_to_datetime(value) - datetime.now(timezone.utc)
            ).total_seconds()
        except (ValueError, TypeError, OverflowError):
            return 0.0
    return min(60.0, max(0.0, seconds))


def call_with_retries(
    operation: Callable[[], _T], *, retry_rate_limits: bool = False
) -> tuple[_T, RetryStats]:
    """Run a provider call, retrying transient failures with backoff.

    Each attempt is bounded by the provider SDK's per-request timeout, so a
    stalled connection can no longer hang a run indefinitely. Retryable
    errors (see :func:`is_retryable_error`) are retried up to
    :data:`MAX_RETRIES` times with exponential backoff and jitter; all other
    errors propagate immediately.

    Args:
        operation: Zero-argument callable that performs the provider request
            and returns its result.
        retry_rate_limits: Retry HTTP 429 on the same route, honoring bounded
            Retry-After delays. Disabled by default for legacy route failover.

    Returns:
        A tuple of the operation result and a :class:`RetryStats` describing
        how many attempts it took and how long the successful attempt ran.
    """
    delay = RETRY_INITIAL_DELAY_SECONDS
    transient_error_types: list[str] = []
    for attempt in range(1, MAX_RETRIES + 2):
        try:
            start_time = time.perf_counter()
            result = operation()
            inference_seconds = time.perf_counter() - start_time
            return result, RetryStats(
                attempts=attempt,
                inference_seconds=inference_seconds,
                transient_error_types=tuple(transient_error_types),
            )
        except Exception as error:
            rate_limited = any(
                getattr(error, attribute, None) == 429
                for attribute in ("status_code", "code")
            )
            if attempt > MAX_RETRIES or not (
                is_retryable_error(error) or (retry_rate_limits and rate_limited)
            ):
                raise
            transient_error_types.append(type(error).__name__)
            sleep_seconds = max(
                delay, _retry_after_seconds(error) if retry_rate_limits else 0.0
            ) + random.uniform(0, RETRY_JITTER_SECONDS)
            _logger.warning(
                "Retryable error on attempt %d/%d (%s: %s); retrying in %.1fs.",
                attempt,
                MAX_RETRIES + 1,
                type(error).__name__,
                error_diagnostics(error)["message"],
                sleep_seconds,
            )
            time.sleep(sleep_seconds)
            delay *= RETRY_BACKOFF_BASE
    raise AssertionError("unreachable: retry loop exited without return or raise")


@dataclass(frozen=True)
class Usage:
    """Token usage reported by a model after a single prediction."""

    input_tokens: int
    output_tokens: int
    metadata: dict[str, Any] = field(default_factory=dict)


class Provider(ABC):
    """Abstract base for VLM inference providers."""

    @property
    @abstractmethod
    def model(self) -> str:
        """Model identifier this provider instance serves."""
        ...

    @abstractmethod
    def predict(
        self,
        image: Image.Image,
        prompt: str,
        effort: str,
    ) -> tuple[str, Usage, RetryStats]:
        """Run inference on an image with a text prompt.

        Args:
            image: Input image in RGB mode.
            prompt: Text prompt to send alongside the image.
            effort: Effort level (e.g. ``"low"``, ``"high"``).

        Returns:
            A tuple of ``(answer_text, usage, retry_stats)``.
        """
        ...

    def uploaded_image_size(self, image: Image.Image) -> tuple[int, int] | None:
        """Return the pixel dimensions the provider uploads for an image.

        Providers that resize an image before sending it override this so
        callers can map returned pixel coordinates back onto the original.

        Args:
            image: Input image in RGB mode.

        Returns:
            The ``(width, height)`` actually sent to the provider, or
            ``None`` when the provider sends the image unchanged.
        """
        return None
