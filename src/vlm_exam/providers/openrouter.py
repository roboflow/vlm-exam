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

import os
from typing import Any

import openai
from PIL import Image

from vlm_exam.providers.base import (
    EMPTY_RESPONSE_TEXT,
    REQUEST_TIMEOUT_SECONDS,
    Provider,
    RetryStats,
    Usage,
    call_with_retries,
)
from vlm_exam.providers.image_upload import (
    OPENROUTER_JPEG_QUALITY,
    OPENROUTER_MAX_BASE64_BYTES,
    jpeg_data_url_under_max_base64_bytes,
)

_BASE_URL = "https://openrouter.ai/api/v1"
_MAX_OUTPUT_TOKENS = 8192


def _reasoning_config(effort: str, provider_model_id: str) -> dict[str, Any]:
    # Qwen and GLM default to extended reasoning, which at "low" effort
    # bloats latency and truncates the answer inside the reasoning trace;
    # disabling it keeps low-effort runs fast and well-formed.
    # Gemini and Muse Spark on OpenRouter require reasoning and reject
    # enabled=False.
    if provider_model_id.startswith(("google/", "meta/")):
        return {"effort": effort}
    if effort == "low":
        return {"enabled": False}
    return {"effort": effort}


class OpenRouterProvider(Provider):
    """OpenRouter provider for OpenAI-compatible vision models.

    Serves any OpenRouter model that accepts image input through the
    chat completions API. The vlm-exam model key is decoupled from the
    OpenRouter slug via ``provider_model_id`` so result files and
    leaderboard lookups keep using the short key.
    """

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
            api_key: Optional OpenRouter API key. Falls back to the
                ``OPENROUTER_API_KEY`` environment variable.
            provider_model_id: OpenRouter model slug to call (e.g.
                ``"qwen/qwen3.7-plus"``). Defaults to ``model``.
        """
        self._model = model
        self._provider_model_id = provider_model_id or model
        self._encoded_cache: tuple[Image.Image, str, tuple[int, int]] | None = None
        self._client = openai.OpenAI(
            base_url=_BASE_URL,
            api_key=api_key or os.environ.get("OPENROUTER_API_KEY"),
            timeout=REQUEST_TIMEOUT_SECONDS,
            max_retries=0,
        )

    @property
    def model(self) -> str:
        return self._model

    def _encode_image(self, image: Image.Image) -> tuple[str, tuple[int, int]]:
        # The runner probes uploaded_image_size() and then predict() with
        # the same image object; a one-slot cache avoids encoding twice.
        cached = self._encoded_cache
        if cached is not None and cached[0] is image:
            return cached[1], cached[2]
        data_url, uploaded_size = jpeg_data_url_under_max_base64_bytes(
            image,
            OPENROUTER_MAX_BASE64_BYTES,
            quality=OPENROUTER_JPEG_QUALITY,
        )
        self._encoded_cache = (image, data_url, uploaded_size)
        return data_url, uploaded_size

    def uploaded_image_size(self, image: Image.Image) -> tuple[int, int] | None:
        _, uploaded_size = self._encode_image(image)
        if uploaded_size == image.size:
            return None
        return uploaded_size

    def predict(
        self,
        image: Image.Image,
        prompt: str,
        effort: str,
    ) -> tuple[str, Usage, RetryStats]:
        data_url, _ = self._encode_image(image)

        response, retry_stats = call_with_retries(
            lambda: self._client.chat.completions.create(
                model=self._provider_model_id,
                max_tokens=_MAX_OUTPUT_TOKENS,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": data_url}},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                extra_body={
                    "reasoning": _reasoning_config(effort, self._provider_model_id)
                },
            )
        )

        if not response.choices:
            answer = EMPTY_RESPONSE_TEXT
        else:
            message = response.choices[0].message
            answer = (message.content or EMPTY_RESPONSE_TEXT).strip()

        usage = response.usage
        return (
            answer,
            Usage(
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
            ),
            retry_stats,
        )
