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

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_PACKAGE_DIRECTORY = Path(__file__).resolve().parent
_DEFAULT_CONFIG_PATH = _PACKAGE_DIRECTORY / "configs" / "models.yaml"
_DEFAULT_LEADERBOARD_GROUPS_PATH = (
    _PACKAGE_DIRECTORY / "configs" / "leaderboard_groups.yaml"
)

_DEFAULT_DETECTION_FORMATS: dict[str, str] = {
    "anthropic": "pixel",
    "openrouter": "normalized_1000_xyxy",
}


@dataclass(frozen=True)
class PricingConfig:
    """Per-model token pricing in USD per million tokens."""

    input_per_million_tokens: float
    output_per_million_tokens: float


@dataclass(frozen=True)
class LabConfig:
    """AI lab branding information for visualization."""

    name: str
    color: str
    logo_url: str


@dataclass(frozen=True)
class RouteConfig:
    """Single inference route for a model."""

    provider: str
    provider_model_id: str | None = None


@dataclass(frozen=True)
class ModelConfig:
    """Single model definition with lab affiliation, routes, and pricing."""

    name: str
    lab: str
    routes: tuple[RouteConfig, ...]
    pricing: PricingConfig
    detection_coordinate_format: str | None = None

    @property
    def provider(self) -> str:
        """Primary route's provider."""
        return self.routes[0].provider

    @property
    def provider_model_id(self) -> str | None:
        """Primary route's upstream model identifier."""
        return self.routes[0].provider_model_id


@dataclass(frozen=True)
class BenchmarkConfig:
    """Top-level benchmark configuration with all labs and models."""

    labs: dict[str, LabConfig]
    models: dict[str, ModelConfig]


def detection_coordinate_format(model_config: ModelConfig) -> str:
    """Resolve the detection coordinate format for a model.

    Uses the model's declared ``detection_coordinate_format`` when set.
    Otherwise falls back to the primary route's provider default.

    Args:
        model_config: Parsed model configuration.

    Returns:
        A ``coordinate_format`` value accepted by :class:`DetectionTask`.
    """
    if model_config.detection_coordinate_format is not None:
        return model_config.detection_coordinate_format
    return _DEFAULT_DETECTION_FORMATS.get(model_config.provider, "normalized_1000")


def _parse_lab(raw: dict[str, Any]) -> LabConfig:
    return LabConfig(
        name=raw["name"],
        color=raw["color"],
        logo_url=raw["logo_url"],
    )


def _parse_routes(raw: dict[str, Any]) -> tuple[RouteConfig, ...]:
    if "routes" in raw:
        return tuple(
            RouteConfig(
                provider=route["provider"],
                provider_model_id=route.get("provider_model_id"),
            )
            for route in raw["routes"]
        )
    return (
        RouteConfig(
            provider=raw["provider"],
            provider_model_id=raw.get("provider_model_id"),
        ),
    )


def _parse_model(raw: dict[str, Any]) -> ModelConfig:
    pricing_raw = raw["pricing"]
    return ModelConfig(
        name=raw["name"],
        lab=raw["lab"],
        routes=_parse_routes(raw),
        pricing=PricingConfig(
            input_per_million_tokens=pricing_raw["input_per_million_tokens"],
            output_per_million_tokens=pricing_raw["output_per_million_tokens"],
        ),
        detection_coordinate_format=raw.get("detection_coordinate_format"),
    )


def load_config(config_path: Path | None = None) -> BenchmarkConfig:
    """Load benchmark configuration from a YAML file.

    Args:
        config_path: Path to the YAML config file. When ``None``, the
            default config bundled with the package is used.

    Returns:
        Parsed benchmark configuration.
    """
    path = config_path or _DEFAULT_CONFIG_PATH
    with open(path) as file:
        raw = yaml.safe_load(file)

    labs = {key: _parse_lab(value) for key, value in raw["labs"].items()}
    models = {key: _parse_model(value) for key, value in raw["models"].items()}

    return BenchmarkConfig(labs=labs, models=models)


def load_leaderboard_groups(
    groups_path: Path | None = None,
) -> dict[str, tuple[str, ...]]:
    """Load named leaderboard model groups from a YAML file.

    Args:
        groups_path: Path to the YAML groups file. When ``None``, the
            default config bundled with the package is used.

    Returns:
        Mapping from group name to an ordered tuple of model keys.
    """
    path = groups_path or _DEFAULT_LEADERBOARD_GROUPS_PATH
    with open(path) as file:
        raw = yaml.safe_load(file)

    return {key: tuple(value) for key, value in raw.items()}
