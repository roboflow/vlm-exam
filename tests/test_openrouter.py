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

from vlm_exam.providers.openrouter import _reasoning_config


def test_gemini_keeps_reasoning_at_low_effort() -> None:
    assert _reasoning_config("low", "google/gemini-3.1-pro-preview") == {
        "effort": "low"
    }


def test_qwen_disables_reasoning_at_low_effort() -> None:
    assert _reasoning_config("low", "qwen/qwen3-vl-235b-a22b-instruct") == {
        "enabled": False
    }
