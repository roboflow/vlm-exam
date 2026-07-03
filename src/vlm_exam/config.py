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
class ModelConfig:
    """Single model definition with lab affiliation and pricing."""

    name: str
    lab: str
    provider: str
    pricing: PricingConfig


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


def _parse_model(raw: dict[str, Any]) -> ModelConfig:
    pricing_raw = raw["pricing"]
    return ModelConfig(
        name=raw["name"],
        lab=raw["lab"],
        provider=raw["provider"],
        pricing=PricingConfig(
            input_per_million_tokens=pricing_raw["input_per_million_tokens"],
            output_per_million_tokens=pricing_raw["output_per_million_tokens"],
        ),
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
