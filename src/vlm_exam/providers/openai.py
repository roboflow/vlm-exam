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

import openai
from PIL import Image

from vlm_exam.providers.base import (
    REQUEST_TIMEOUT_SECONDS,
    Provider,
    RetryStats,
    Usage,
    call_with_retries,
)


def _image_to_base64_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    base64_data = base64.standard_b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{base64_data}"


class OpenAIProvider(Provider):
    """OpenAI GPT provider."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        provider_model_id: str | None = None,
    ) -> None:
        self._model = model
        self._wire_model_id = provider_model_id or model
        self._client = openai.OpenAI(
            api_key=api_key,
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
        data_url = _image_to_base64_url(image)

        response, retry_stats = call_with_retries(
            lambda: self._client.responses.create(
                model=self._wire_model_id,
                reasoning={"effort": effort},
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_image", "image_url": data_url},
                            {"type": "input_text", "text": prompt},
                        ],
                    }
                ],
            )
        )

        answer = response.output_text.strip()

        return (
            answer,
            Usage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            ),
            retry_stats,
        )
