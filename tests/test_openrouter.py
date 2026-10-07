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


def test_inkling_maps_low_effort_to_its_low_preset() -> None:
    assert _reasoning_config("low", "thinkingmachines/inkling") == {"effort": "low"}
    assert _reasoning_config("high", "thinkingmachines/inkling") == {"effort": "high"}


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
