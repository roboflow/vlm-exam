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

from unittest.mock import MagicMock, patch

import anthropic
import openai
import pytest
from PIL import Image

from vlm_exam.config import ModelConfig, PricingConfig, RouteConfig
from vlm_exam.providers import build_model_provider
from vlm_exam.providers.base import Provider, Usage
from vlm_exam.providers.fallback import FallbackProvider, is_rate_limit_error


class _StubProvider(Provider):
    def __init__(
        self,
        model: str,
        *,
        responses: list[tuple[str, Usage] | Exception],
    ) -> None:
        self._model = model
        self._responses = list(responses)
        self.call_count = 0

    @property
    def model(self) -> str:
        return self._model

    def predict(
        self,
        image: Image.Image,
        prompt: str,
        effort: str,
    ) -> tuple[str, Usage]:
        self.call_count += 1
        outcome = self._responses.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _rate_limit_error() -> anthropic.RateLimitError:
    return anthropic.RateLimitError("limited", response=MagicMock(), body=None)


class TestIsRateLimitError:
    def test_anthropic_rate_limit(self) -> None:
        assert is_rate_limit_error(_rate_limit_error())

    def test_openai_rate_limit(self) -> None:
        error = openai.RateLimitError("limited", response=MagicMock(), body=None)
        assert is_rate_limit_error(error)

    def test_google_quota_message(self) -> None:
        error = Exception(
            "429 RESOURCE_EXHAUSTED. quota exceeded for "
            "generate_requests_per_model_per_day"
        )
        assert is_rate_limit_error(error)

    def test_non_rate_limit_propagates(self) -> None:
        assert not is_rate_limit_error(ValueError("bad request"))


class TestFallbackProvider:
    def test_single_success(self) -> None:
        image = Image.new("RGB", (10, 10))
        primary = _StubProvider(
            "test-model",
            responses=[("answer", Usage(1, 2))],
        )
        provider = FallbackProvider(model="test-model", providers=[primary])
        answer, usage = provider.predict(image, "prompt", "low")
        assert answer == "answer"
        assert usage.output_tokens == 2
        assert primary.call_count == 1

    def test_fails_over_on_rate_limit_and_sticks(self) -> None:
        image = Image.new("RGB", (10, 10))
        primary = _StubProvider(
            "test-model",
            responses=[
                _rate_limit_error(),
                _rate_limit_error(),
            ],
        )
        fallback = _StubProvider(
            "test-model",
            responses=[
                ("first", Usage(1, 1)),
                ("second", Usage(2, 2)),
            ],
        )
        provider = FallbackProvider(model="test-model", providers=[primary, fallback])
        answer_one, _ = provider.predict(image, "prompt", "low")
        answer_two, _ = provider.predict(image, "prompt", "low")
        assert answer_one == "first"
        assert answer_two == "second"
        assert primary.call_count == 1
        assert fallback.call_count == 2
        assert provider.active_route_index == 1

    def test_non_rate_limit_raises(self) -> None:
        image = Image.new("RGB", (10, 10))
        primary = _StubProvider("test-model", responses=[ValueError("broken")])
        provider = FallbackProvider(model="test-model", providers=[primary])
        with pytest.raises(ValueError, match="broken"):
            provider.predict(image, "prompt", "low")

    def test_all_routes_exhausted_raises_last_error(self) -> None:
        image = Image.new("RGB", (10, 10))
        primary = _StubProvider(
            "test-model",
            responses=[_rate_limit_error()],
        )
        fallback = _StubProvider(
            "test-model",
            responses=[_rate_limit_error()],
        )
        provider = FallbackProvider(model="test-model", providers=[primary, fallback])
        with pytest.raises(anthropic.RateLimitError):
            provider.predict(image, "prompt", "low")


class TestBuildModelProvider:
    def test_single_route_returns_concrete_provider(self) -> None:
        model_config = ModelConfig(
            name="GPT",
            lab="openai",
            routes=(RouteConfig("openai"),),
            pricing=PricingConfig(1.0, 2.0),
        )
        stub = _StubProvider("gpt-5.5", responses=[("ok", Usage(1, 1))])
        with patch(
            "vlm_exam.providers.create_provider",
            return_value=stub,
        ) as create_provider:
            provider = build_model_provider("gpt-5.5", model_config)
        create_provider.assert_called_once_with(
            "openai",
            model="gpt-5.5",
            api_key=None,
            provider_model_id=None,
        )
        assert provider is stub

    def test_multiple_routes_returns_fallback(self) -> None:
        model_config = ModelConfig(
            name="Gemini",
            lab="google",
            routes=(
                RouteConfig("google"),
                RouteConfig("openrouter", "google/gemini-3.1-pro-preview"),
            ),
            pricing=PricingConfig(2.0, 12.0),
        )
        primary = _StubProvider("gemini-3.1-pro-preview", responses=[])
        fallback = _StubProvider("gemini-3.1-pro-preview", responses=[])
        with patch(
            "vlm_exam.providers.create_provider",
            side_effect=[primary, fallback],
        ):
            provider = build_model_provider("gemini-3.1-pro-preview", model_config)
        assert isinstance(provider, FallbackProvider)
