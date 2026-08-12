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

import base64
import io
import os
from typing import Any

import openai
from PIL import Image

from vlm_exam.providers.base import (
    REQUEST_TIMEOUT_SECONDS,
    Provider,
    RetryStats,
    Usage,
    call_with_retries,
)

_BASE_URL = "https://api.x.ai/v1"


def _png_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    base64_data = base64.standard_b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{base64_data}"


class XAIProvider(Provider):
    """xAI Grok provider via the OpenAI-compatible Responses API."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        provider_model_id: str | None = None,
    ) -> None:
        """Initialize the provider.

        Args:
            model: vlm-exam model key, reported as ``model`` and used in
                result filenames and config lookups.
            api_key: Optional xAI API key. Falls back to the
                ``XAI_API_KEY`` environment variable.
            provider_model_id: Upstream xAI model id to call (e.g.
                ``"grok-4.5"``). Defaults to ``model``.
        """
        self._model = model
        self._provider_model_id = provider_model_id or model
        self._client = openai.OpenAI(
            base_url=_BASE_URL,
            api_key=api_key or os.environ.get("XAI_API_KEY"),
            timeout=REQUEST_TIMEOUT_SECONDS,
            max_retries=0,
        )

    @property
    def model(self) -> str:
        return self._model

    def predict(
        self,
        image: Image.Image,
        prompt: str,
        effort: str,
    ) -> tuple[str, Usage, RetryStats]:
        data_url = _png_data_url(image)
        request: dict[str, Any] = {
            "model": self._provider_model_id,
            "reasoning": {"effort": effort},
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": data_url,
                            "detail": "high",
                        },
                        {"type": "input_text", "text": prompt},
                    ],
                }
            ],
        }

        try:
            response, retry_stats = call_with_retries(
                lambda: self._client.responses.create(**request)
            )
        except Exception as error:
            if not _is_unsupported_reasoning_error(error):
                raise
            request.pop("reasoning", None)
            response, retry_stats = call_with_retries(
                lambda: self._client.responses.create(**request)
            )

        answer = (response.output_text or "").strip()
        usage = response.usage
        return (
            answer,
            Usage(
                input_tokens=usage.input_tokens if usage else 0,
                output_tokens=usage.output_tokens if usage else 0,
            ),
            retry_stats,
        )


def _is_unsupported_reasoning_error(error: Exception) -> bool:
    message = str(error).lower()
    return "reasoning" in message and (
        "unsupported" in message
        or "not supported" in message
        or "unknown" in message
        or "invalid" in message
    )
