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

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from vlm_exam.tasks.detection import DetectionCoordinateFormat

_PACKAGE_DIRECTORY = Path(__file__).resolve().parent
_DEFAULT_CONFIG_PATH = _PACKAGE_DIRECTORY / "configs" / "models.yaml"
_DEFAULT_LEADERBOARD_GROUPS_PATH = (
    _PACKAGE_DIRECTORY / "configs" / "leaderboard_groups.yaml"
)


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
    detection_coordinate_format: DetectionCoordinateFormat
    resolution_tier: str = "high"

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
    from vlm_exam.tasks.detection import DetectionCoordinateFormat

    pricing_raw = raw["pricing"]
    routes = _parse_routes(raw)
    coordinate_format = DetectionCoordinateFormat(raw["detection_coordinate_format"])
    resolution_tier = raw.get("resolution_tier", "high")
    _validate_resolution_tier(raw["name"], resolution_tier)
    _validate_provider_upload_routes(raw["name"], coordinate_format, routes)
    return ModelConfig(
        name=raw["name"],
        lab=raw["lab"],
        routes=routes,
        pricing=PricingConfig(
            input_per_million_tokens=pricing_raw["input_per_million_tokens"],
            output_per_million_tokens=pricing_raw["output_per_million_tokens"],
        ),
        detection_coordinate_format=coordinate_format,
        resolution_tier=resolution_tier,
    )


def _validate_resolution_tier(model_name: str, resolution_tier: str) -> None:
    from vlm_exam.providers.anthropic import RESOLUTION_TIERS

    if resolution_tier not in RESOLUTION_TIERS:
        valid = ", ".join(sorted(RESOLUTION_TIERS))
        raise ValueError(
            f"Model {model_name!r} has unknown resolution_tier "
            f"{resolution_tier!r}. Valid tiers: {valid}."
        )


def _validate_provider_upload_routes(
    model_name: str,
    coordinate_format: DetectionCoordinateFormat,
    routes: tuple[RouteConfig, ...],
) -> None:
    from vlm_exam.providers import PRE_RESIZING_PROVIDERS
    from vlm_exam.tasks.detection import DetectionCoordinateFormat

    if coordinate_format != DetectionCoordinateFormat.XYXY_ABSOLUTE_RESIZED_IMAGE:
        return
    offending = sorted(
        {
            route.provider
            for route in routes
            if route.provider not in PRE_RESIZING_PROVIDERS
        }
    )
    if offending:
        providers = ", ".join(offending)
        supported = ", ".join(sorted(PRE_RESIZING_PROVIDERS))
        raise ValueError(
            f"Model {model_name!r} uses coordinate format "
            f"{coordinate_format.value!r}, whose pixel scaling only holds for "
            f"providers that pre-resize uploads ({supported}); "
            f"incompatible routes: {providers}."
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
