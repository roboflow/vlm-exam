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

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from PIL import Image

from vlm_exam.providers import create_provider
from vlm_exam.providers.xai import XAIProvider


def _usage(input_tokens: int = 3, output_tokens: int = 5) -> SimpleNamespace:
    return SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)


class TestXAIProvider:
    def test_create_provider_registers_xai(self) -> None:
        with patch("vlm_exam.providers.xai.openai.OpenAI") as client_class:
            provider = create_provider("xai", model="grok-4.5")
        assert isinstance(provider, XAIProvider)
        assert provider.model == "grok-4.5"
        client_class.assert_called_once()
        assert client_class.call_args.kwargs["base_url"] == "https://api.x.ai/v1"

    def test_predict_uses_responses_api_with_reasoning(self) -> None:
        image = Image.new("RGB", (32, 32), color=(10, 20, 30))
        response = SimpleNamespace(output_text=" a cat ", usage=_usage())
        client = MagicMock()
        client.responses.create.return_value = response

        with patch("vlm_exam.providers.xai.openai.OpenAI", return_value=client):
            provider = XAIProvider(
                model="grok-4.5",
                api_key="test-key",
                provider_model_id="grok-4.5",
            )
            answer, usage, retry_stats = provider.predict(image, "what?", "low")

        assert answer == "a cat"
        assert usage.input_tokens == 3
        assert usage.output_tokens == 5
        assert retry_stats.attempts == 1
        kwargs = client.responses.create.call_args.kwargs
        assert kwargs["model"] == "grok-4.5"
        assert kwargs["store"] is False
        assert kwargs["reasoning"] == {"effort": "low"}
        content = kwargs["input"][0]["content"]
        assert content[0]["type"] == "input_image"
        assert content[0]["image_url"].startswith("data:image/png;base64,")
        assert content[0]["detail"] == "high"
        assert content[1] == {"type": "input_text", "text": "what?"}

    def test_predict_retries_without_reasoning_when_unsupported(self) -> None:
        image = Image.new("RGB", (16, 16))
        response = SimpleNamespace(output_text="ok", usage=_usage(1, 1))
        client = MagicMock()
        client.responses.create.side_effect = [
            ValueError("reasoning effort is not supported for this model"),
            response,
        ]

        with patch("vlm_exam.providers.xai.openai.OpenAI", return_value=client):
            provider = XAIProvider(model="grok-4.5", api_key="test-key")
            answer, _, _ = provider.predict(image, "prompt", "high")

        assert answer == "ok"
        assert client.responses.create.call_count == 2
        first_kwargs = client.responses.create.call_args_list[0].kwargs
        second_kwargs = client.responses.create.call_args_list[1].kwargs
        assert "reasoning" in first_kwargs
        assert "reasoning" not in second_kwargs
