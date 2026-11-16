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

from unittest.mock import MagicMock

from PIL import Image

from vlm_exam.providers.openrouter import (
    _MAX_OUTPUT_TOKENS,
    OpenRouterProvider,
    _reasoning_config,
)


def test_gemini_keeps_reasoning_at_low_effort() -> None:
    assert _reasoning_config("low", "google/gemini-3.1-pro-preview") == {
        "effort": "low"
    }


def test_qwen_disables_reasoning_at_low_effort() -> None:
    assert _reasoning_config("low", "qwen/qwen3-vl-235b-a22b-instruct") == {
        "enabled": False
    }


def test_muse_spark_keeps_reasoning_at_low_effort() -> None:
    assert _reasoning_config("low", "meta/muse-spark-1.1") == {"effort": "low"}


def test_glm_5_3_flash_keeps_reasoning_at_low_effort() -> None:
    assert _reasoning_config("low", "z-ai/glm-5.3-flash") == {"effort": "low"}


def _requested_max_tokens(provider_model_id: str) -> int:
    provider = OpenRouterProvider(
        "key", api_key="test", provider_model_id=provider_model_id
    )
    create = MagicMock()
    create.return_value.choices = []
    create.return_value.usage = None
    provider._client.chat.completions.create = create
    provider.predict(Image.new("RGB", (8, 8)), "prompt", "high")
    return create.call_args.kwargs["max_tokens"]


def test_mistral_large_4_gets_a_larger_output_budget() -> None:
    assert _requested_max_tokens("mistralai/mistral-large-4-0") == 65536


def test_other_models_keep_the_default_output_budget() -> None:
    assert _requested_max_tokens("qwen/qwen3.8-max") == _MAX_OUTPUT_TOKENS


def test_explicit_route_keeps_low_reasoning_and_pins_privacy() -> None:
    options = {
        "max_tokens": 32768,
        "provider": {
            "only": ["minimax"],
            "allow_fallbacks": False,
            "data_collection": "deny",
            "require_parameters": True,
        },
        "reasoning": {"low": {"effort": "low"}},
    }
    provider = OpenRouterProvider("test", api_key="test", options=options)
    options["provider"]["only"].append("unexpected")
    body = provider.request_options("low")
    assert body["reasoning"] == {"effort": "low"}
    assert body["provider"]["only"] == ["minimax"]
    assert body["provider"]["data_collection"] == "deny"
    body["provider"]["only"].append("unexpected")
    assert provider.request_options("low")["provider"]["only"] == ["minimax"]


def test_route_options_capture_reported_usage_and_finish_reason() -> None:
    from openai.types.chat import ChatCompletion

    provider = OpenRouterProvider("test", api_key="test", options={"max_tokens": 32768})
    response = ChatCompletion.model_validate(
        {
            "id": "generation-id",
            "model": "upstream-model",
            "provider": "Together",
            "created": 0,
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "answer"},
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 30,
                "total_tokens": 40,
                "cost": 0.001,
                "completion_tokens_details": {"reasoning_tokens": 20},
            },
        }
    )
    create = MagicMock(return_value=response)
    provider._client.chat.completions.create = create
    answer, usage, _ = provider.predict(Image.new("RGB", (8, 8)), "prompt", "high")
    assert answer == "answer"
    assert create.call_args.kwargs["max_tokens"] == 32768
    assert usage.metadata["provider"] == "Together"
    assert usage.metadata["reported"]["cost"] == 0.001
    assert (
        usage.metadata["reported"]["completion_tokens_details"]["reasoning_tokens"]
        == 20
    )


def test_qwen4_profile_sends_native_efforts_and_validated_route() -> None:
    from pathlib import Path

    from openai.types.chat import ChatCompletion

    from vlm_exam.config import load_config
    from vlm_exam.providers import build_model_provider

    path = Path(__file__).parents[1] / "src/vlm_exam/configs/qwen4-max.yaml"
    model = load_config(path).models["qwen-4-max"]
    provider = build_model_provider("qwen-4-max", model, api_key="test")
    response = ChatCompletion.model_validate(
        {
            "id": "qwen4-generation",
            "model": "qwen/qwen4-max",
            "provider": "Alibaba",
            "created": 0,
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "[]"},
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }
    )
    create = MagicMock(return_value=response)
    provider._client.chat.completions.create = create
    for effort in ("low", "high"):
        answer, usage, _ = provider.predict(Image.new("RGB", (8, 8)), "prompt", effort)
        request = create.call_args.kwargs
        assert request["model"] == "qwen/qwen4-max"
        assert request["extra_body"] == {
            "reasoning": {"effort": effort},
            "provider": {
                "only": ["alibaba"],
                "allow_fallbacks": False,
                "require_parameters": True,
            },
        }
        assert request["max_tokens"] == 131072
        assert answer == "[]"
        assert usage.metadata["provider"] == "Alibaba"
        assert usage.metadata["finish_reason"] == "stop"
    assert model.detection_coordinate_format == "xyxy_normalized_0_to_1000"
