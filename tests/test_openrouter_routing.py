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

import pytest

from vlm_exam.providers.openrouter_routing import validate_openrouter_routing


def profile(provider: str = "alibaba") -> dict:
    return {
        "provider": "openrouter",
        "wire": "qwen/qwen4-max",
        "routing": {
            "only": [provider],
            "allow_fallbacks": False,
            "require_parameters": True,
        },
    }


def test_native_route() -> None:
    validate_openrouter_routing(profile())


@pytest.mark.parametrize(
    "routing",
    [
        None,
        {},
        {"only": []},
        {"only": ["alibaba", "together"]},
        {"only": ["alibaba"]},
        {"only": ["alibaba"], "allow_fallbacks": True, "require_parameters": True},
        {"only": ["alibaba"], "allow_fallbacks": False, "require_parameters": False},
    ],
)
def test_reject_unpinned_or_permissive_routes(routing: object) -> None:
    p = profile()
    p["routing"] = routing
    with pytest.raises(ValueError):
        validate_openrouter_routing(p)


def test_non_native_requires_evidence() -> None:
    p = profile("together")
    with pytest.raises(ValueError):
        validate_openrouter_routing(p)
    p["routing_exception"] = {
        "reason": "Native endpoint unavailable",
        "evidence": "Endpoint lookup, 2026-11-25",
    }
    validate_openrouter_routing(p)


def test_unknown_family_requires_registration() -> None:
    p = profile()
    p["wire"] = "unknown/model"
    with pytest.raises(ValueError):
        validate_openrouter_routing(p)


def test_native_endpoint_variant() -> None:
    p = profile("xiaomi/fp8")
    p["wire"] = "xiaomi/mimo-v2.6-pro"
    validate_openrouter_routing(p)


def test_other_provider_unchanged() -> None:
    validate_openrouter_routing({"provider": "google"})


def test_inference_rejects_unpinned_before_loading_image() -> None:
    from pathlib import Path

    from vlm_exam.detection_v2_inference import infer

    with pytest.raises(ValueError, match="routing.only"):
        infer({}, "qwen", "minimum", {"provider": "openrouter"}, Path("missing"))
