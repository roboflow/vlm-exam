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

_NATIVE_PROVIDERS = {
    "qwen": "alibaba",
    "mistralai": "mistral",
    "meta": "meta",
    "moonshotai": "moonshotai",
    "xiaomi": "xiaomi",
    "z-ai": "z-ai",
    "google": "google",
    "openai": "openai",
    "anthropic": "anthropic",
    "x-ai": "xai",
}


def validate_openrouter_routing(profile: dict[str, Any]) -> None:
    """Reject new OpenRouter calls without explicit, native-first routing.

    Non-native routes require a documented availability exception. This checks
    the recorded policy, not live endpoint availability or historical results.
    """
    if profile.get("provider") != "openrouter":
        return
    routing = profile.get("routing")
    if not isinstance(routing, dict):
        raise ValueError("OpenRouter requires explicit routing.only provider ID")
    only = routing.get("only")
    if (
        not isinstance(only, list)
        or len(only) != 1
        or not isinstance(only[0], str)
        or not only[0].strip()
        or only[0] != only[0].strip()
    ):
        raise ValueError("OpenRouter routing.only must contain exactly one provider ID")
    if routing.get("allow_fallbacks") is not False:
        raise ValueError("OpenRouter requires allow_fallbacks=false")
    if routing.get("require_parameters") is not True:
        raise ValueError("OpenRouter requires require_parameters=true")
    native = _NATIVE_PROVIDERS.get(str(profile.get("wire", "")).split("/")[0])
    if native is None:
        raise ValueError("Register the model's native provider before running it")
    if only[0].split("/")[0] != native:
        exception = profile.get("routing_exception")
        if not isinstance(exception, dict) or any(
            not isinstance(exception.get(key), str) or not exception[key].strip()
            for key in ("reason", "evidence")
        ):
            raise ValueError(
                f"Use native provider {native}; a non-native route requires "
                "routing_exception.reason and evidence of native unavailability"
            )
