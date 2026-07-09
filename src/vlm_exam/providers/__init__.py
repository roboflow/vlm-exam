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

import importlib

from vlm_exam.config import ModelConfig
from vlm_exam.providers.base import Provider, Usage
from vlm_exam.providers.fallback import FallbackProvider

_PROVIDER_REGISTRY: dict[str, str] = {
    "anthropic": "vlm_exam.providers.anthropic.AnthropicProvider",
    "google": "vlm_exam.providers.google.GoogleProvider",
    "openai": "vlm_exam.providers.openai.OpenAIProvider",
    "openrouter": "vlm_exam.providers.openrouter.OpenRouterProvider",
}

__all__ = [
    "FallbackProvider",
    "Provider",
    "Usage",
    "build_model_provider",
    "create_provider",
]


def create_provider(
    provider_name: str,
    model: str,
    api_key: str | None = None,
    provider_model_id: str | None = None,
) -> Provider:
    """Create a provider instance by name.

    Args:
        provider_name: Key in the provider registry
            (e.g. ``"anthropic"``, ``"google"``, ``"openai"``,
            ``"openrouter"``).
        model: vlm-exam model key used in result filenames and config
            lookups.
        api_key: Optional API key. When ``None``, the provider falls
            back to its default environment variable.
        provider_model_id: Optional upstream model identifier sent to the
            provider API. Defaults to ``model`` when omitted.

    Returns:
        A ready-to-use provider instance.

    Raises:
        KeyError: If ``provider_name`` is not registered.
    """
    if provider_name not in _PROVIDER_REGISTRY:
        available = ", ".join(sorted(_PROVIDER_REGISTRY))
        raise KeyError(
            f"Unknown provider {provider_name!r}. Available providers: {available}"
        )

    qualified_name = _PROVIDER_REGISTRY[provider_name]
    module_path, class_name = qualified_name.rsplit(".", 1)
    module = importlib.import_module(module_path)
    provider_class = getattr(module, class_name)
    return provider_class(
        model=model,
        api_key=api_key,
        provider_model_id=provider_model_id,
    )


def build_model_provider(
    model_id: str,
    model_config: ModelConfig,
    api_key: str | None = None,
) -> Provider:
    """Build a provider for a configured model, with route failover when set.

    Args:
        model_id: vlm-exam model key.
        model_config: Parsed model configuration.
        api_key: Optional API key override for all routes.

    Returns:
        A single-route provider, or a :class:`FallbackProvider` when
        multiple routes are configured.
    """
    route_providers = [
        create_provider(
            route.provider,
            model=model_id,
            api_key=api_key,
            provider_model_id=route.provider_model_id,
        )
        for route in model_config.routes
    ]
    if len(route_providers) == 1:
        return route_providers[0]
    return FallbackProvider(model=model_id, providers=route_providers)
