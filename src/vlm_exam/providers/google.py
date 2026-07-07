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

import io

from google import genai
from google.genai import types
from PIL import Image

from vlm_exam.providers.base import Provider, Usage

_LEGACY_THINKING_BUDGETS = {"low": 128, "medium": 2048, "high": 8192}


def _image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _thinking_config(model: str, effort: str) -> types.ThinkingConfig:
    # Gemini 2.x models reject thinking_level and require the legacy
    # thinking_budget parameter; mixing the two returns a 400 error.
    if model.startswith("gemini-2."):
        budget = _LEGACY_THINKING_BUDGETS.get(effort, -1)
        return types.ThinkingConfig(thinking_budget=budget)
    return types.ThinkingConfig(thinking_level=effort)


class GoogleProvider(Provider):
    """Google Gemini provider."""

    def __init__(self, model: str, api_key: str | None = None) -> None:
        self._model = model
        self._client = genai.Client(api_key=api_key)

    @property
    def model(self) -> str:
        return self._model

    def predict(
        self,
        image: Image.Image,
        prompt: str,
        effort: str,
    ) -> tuple[str, Usage]:
        png_bytes = _image_to_png_bytes(image)

        response = self._client.models.generate_content(
            model=self._model,
            contents=[
                types.Part.from_bytes(data=png_bytes, mime_type="image/png"),
                prompt,
            ],
            config=types.GenerateContentConfig(
                thinking_config=_thinking_config(self._model, effort),
            ),
        )

        answer = response.text.strip()
        output_tokens = response.usage_metadata.candidates_token_count or 0
        thoughts_tokens = response.usage_metadata.thoughts_token_count or 0

        return answer, Usage(
            input_tokens=response.usage_metadata.prompt_token_count,
            output_tokens=output_tokens + thoughts_tokens,
        )
