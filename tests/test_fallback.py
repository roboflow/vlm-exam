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

from typing import Any
from unittest.mock import MagicMock, patch

import anthropic
import openai
import pytest
from PIL import Image

from vlm_exam.config import ModelConfig, PricingConfig, RouteConfig, load_config
from vlm_exam.providers import build_model_provider, create_provider
from vlm_exam.providers.base import Provider, RetryStats, Usage
from vlm_exam.providers.fallback import FallbackProvider, is_rate_limit_error
from vlm_exam.tasks.detection import DetectionCoordinateFormat


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
    ) -> tuple[str, Usage, RetryStats]:
        self.call_count += 1
        outcome = self._responses.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        answer, usage = outcome
        return answer, usage, RetryStats(attempts=1, inference_seconds=0.0)


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
        answer, usage, _ = provider.predict(image, "prompt", "low")
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
        answer_one, _, _ = provider.predict(image, "prompt", "low")
        answer_two, _, _ = provider.predict(image, "prompt", "low")
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

    def test_last_route_rate_limit_retries_with_backoff(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("vlm_exam.providers.fallback.time.sleep", lambda _: None)
        image = Image.new("RGB", (10, 10))
        primary = _StubProvider(
            "test-model",
            responses=[
                _rate_limit_error(),
                _rate_limit_error(),
                ("recovered", Usage(1, 1)),
            ],
        )
        provider = FallbackProvider(model="test-model", providers=[primary])
        answer, _, _ = provider.predict(image, "prompt", "low")
        assert answer == "recovered"
        assert primary.call_count == 3

    def test_all_routes_exhausted_raises_last_error(self) -> None:
        image = Image.new("RGB", (10, 10))
        primary = _StubProvider(
            "test-model",
            responses=[_rate_limit_error()],
        )
        fallback = _StubProvider(
            "test-model",
            responses=[
                _rate_limit_error(),
                _rate_limit_error(),
                _rate_limit_error(),
                _rate_limit_error(),
            ],
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
            detection_coordinate_format=(
                DetectionCoordinateFormat.XYXY_ABSOLUTE_ORIGINAL_IMAGE
            ),
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
            resolution_tier="high",
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
            detection_coordinate_format=(
                DetectionCoordinateFormat.YXYX_NORMALIZED_0_TO_1000
            ),
        )
        primary = _StubProvider("gemini-3.1-pro-preview", responses=[])
        fallback = _StubProvider("gemini-3.1-pro-preview", responses=[])
        with patch(
            "vlm_exam.providers.create_provider",
            side_effect=[primary, fallback],
        ):
            provider = build_model_provider("gemini-3.1-pro-preview", model_config)
        assert isinstance(provider, FallbackProvider)

    def test_mixed_resize_routes_raise(self) -> None:
        model_config = ModelConfig(
            name="Claude Mixed",
            lab="anthropic",
            routes=(
                RouteConfig("anthropic"),
                RouteConfig("openrouter", "anthropic/claude"),
            ),
            pricing=PricingConfig(2.0, 12.0),
            detection_coordinate_format=(
                DetectionCoordinateFormat.YXYX_NORMALIZED_0_TO_1000
            ),
        )
        with pytest.raises(ValueError, match="pre-resize"):
            build_model_provider("claude-mixed", model_config)


class TestUploadedImageSize:
    def test_default_provider_returns_none(self) -> None:
        stub = _StubProvider("gpt-5.5", responses=[])
        assert stub.uploaded_image_size(Image.new("RGB", (640, 480))) is None


@pytest.mark.parametrize("effort,expected", [("low", "low"), ("high", "xhigh")])
def test_google_preview_request(effort: str, expected: str) -> None:
    from pathlib import Path

    configuration = load_config(
        Path(__file__).parents[1] / "src/vlm_exam/configs/previews/hydrogen.yaml"
    )
    with patch("vlm_exam.providers.google.genai.Client") as client:
        response = client.return_value.models.generate_content.return_value
        response.text = "  detected  "
        response.usage_metadata.prompt_token_count = 10
        response.usage_metadata.candidates_token_count = 3
        response.usage_metadata.thoughts_token_count = 7
        provider = build_model_provider(
            "gemini-4-hydrogen",
            configuration.models["gemini-4-hydrogen"],
            api_key="test-key",
        )
        text, usage, _ = provider.predict(Image.new("RGB", (10, 10)), "find", effort)
        assert client.call_args.kwargs["http_options"].api_version == "v1beta"
        request = client.return_value.models.generate_content.call_args.kwargs
        assert request["model"] == "gemini-4-hydrogen"
        assert request["config"].max_output_tokens == 131072
        assert request["config"].thinking_config.thinking_level.lower() == expected
        assert text == "detected"
        assert usage.input_tokens == 10
        assert usage.output_tokens == 10


@pytest.mark.parametrize(
    "model,effort,expected",
    [("gemini-4-argon-eap", "high", "high"), ("gemini-2.5-pro", "high", 8192)],
)
def test_google_existing_thinking_is_unchanged(
    model: str, effort: str, expected: str | int
) -> None:
    with patch("vlm_exam.providers.google.genai.Client") as client:
        response = client.return_value.models.generate_content.return_value
        response.text = "ok"
        response.usage_metadata.prompt_token_count = 1
        response.usage_metadata.candidates_token_count = 1
        response.usage_metadata.thoughts_token_count = 0
        provider = create_provider("google", model, api_key="test-key")
        provider.predict(Image.new("RGB", (10, 10)), "find", effort)
        thinking = client.return_value.models.generate_content.call_args.kwargs[
            "config"
        ].thinking_config
        if isinstance(expected, int):
            assert thinking.thinking_budget == expected
            assert thinking.thinking_level is None
        else:
            assert thinking.thinking_level.lower() == expected
        assert client.call_args.kwargs["http_options"].api_version is None


@pytest.mark.parametrize(
    "options",
    [
        {"api_version": "invalid"},
        {"thinking_levels": {"high": "typo"}},
        {"thinking_levels": []},
        {"thinking_levels": {"high": ["xhigh"]}},
        {"api_version": ["v1beta"]},
        {"thinking_level": "xhigh"},
    ],
)
def test_google_rejects_invalid_options(options: dict[str, Any]) -> None:
    with patch("vlm_exam.providers.google.genai.Client") as client:
        with pytest.raises(ValueError):
            create_provider("google", "gemini-4-hydrogen", options=options)
        client.assert_not_called()


def test_google_output_limit_and_finish_metadata() -> None:
    from google.genai import types

    with patch("vlm_exam.providers.google.genai.Client") as client:
        client.return_value.models.generate_content.return_value = (
            types.GenerateContentResponse(
                candidates=[types.Candidate(finish_reason="MAX_TOKENS")],
                usage_metadata=types.GenerateContentResponseUsageMetadata(
                    prompt_token_count=12,
                    candidates_token_count=0,
                    thoughts_token_count=131072,
                ),
            )
        )
        provider = create_provider(
            "google",
            "gemini-4-hydrogen",
            api_key="test-key",
            options={"max_output_tokens": 131072, "thinking_levels": {"high": "xhigh"}},
        )
        _, usage, _ = provider.predict(Image.new("RGB", (10, 10)), "find", "high")
        request = client.return_value.models.generate_content.call_args.kwargs
        assert request["config"].max_output_tokens == 131072
        assert usage.output_tokens == 131072
        assert usage.metadata["thinking_tokens"] == 131072
        assert usage.metadata["candidate_tokens"] == 0
        assert usage.metadata["finish_reasons"] == ["FinishReason.MAX_TOKENS"]


@pytest.mark.parametrize("limit", [0, -1, True, "131072", 1.5])
def test_google_rejects_invalid_output_limit(limit: Any) -> None:
    with pytest.raises(ValueError, match="max_output_tokens"):
        create_provider(
            "google", "gemini-4-hydrogen", options={"max_output_tokens": limit}
        )


def test_google_unfinished_continuation_is_not_scored_as_empty() -> None:
    from types import SimpleNamespace

    with patch("vlm_exam.providers.google.genai.Client") as client:
        response = client.return_value.models.generate_content.return_value
        response.candidates = [SimpleNamespace(finish_reason="CONTINUATION")]
        provider = create_provider("google", "gemini-4-hydrogen", api_key="test-key")
        with pytest.raises(RuntimeError, match="requires continuation"):
            provider.predict(Image.new("RGB", (10, 10)), "find", "high")
