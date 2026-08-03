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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from vlm_exam.config import LabConfig
from vlm_exam.tasks.detection import DetectionCoordinateFormat

_DEFAULT_REFERENCE_CONFIG_PATH = (
    Path(__file__).resolve().parent / "configs" / "reference_models.yaml"
)


@dataclass(frozen=True)
class ReferenceInferenceConfig:
    """Default inference settings for a reference model."""

    conf: float
    iou: float | None
    imgsz: int | None
    max_det: int | None
    agnostic_nms: bool | None


@dataclass(frozen=True)
class ReferenceModelConfig:
    """Single reference model definition."""

    key: str
    name: str
    lab: str | None
    family: str
    adapter: str
    checkpoint: str
    coordinate_format: DetectionCoordinateFormat
    supported_devices: tuple[str, ...]
    inference: ReferenceInferenceConfig
    card_confidence_threshold: float | None = None
    checkpoint_revision: str | None = None
    checkpoint_sha256: str | None = None


@dataclass(frozen=True)
class ReferenceConfig:
    """Top-level reference model registry."""

    models: dict[str, ReferenceModelConfig]
    labs: dict[str, LabConfig] = field(default_factory=dict)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _parse_inference(raw: dict[str, Any]) -> ReferenceInferenceConfig:
    agnostic_nms = raw.get("agnostic_nms")
    if agnostic_nms is not None and not isinstance(agnostic_nms, bool):
        raise ValueError("agnostic_nms must be a boolean or null.")
    return ReferenceInferenceConfig(
        conf=float(raw["conf"]),
        iou=_optional_float(raw.get("iou")),
        imgsz=_optional_int(raw.get("imgsz")),
        max_det=_optional_int(raw.get("max_det")),
        agnostic_nms=agnostic_nms,
    )


def _parse_model(key: str, raw: dict[str, Any]) -> ReferenceModelConfig:
    lab = raw.get("lab")
    if lab is not None:
        lab = str(lab)
    return ReferenceModelConfig(
        key=key,
        name=raw["name"],
        lab=lab,
        family=raw["family"],
        adapter=raw["adapter"],
        checkpoint=raw["checkpoint"],
        coordinate_format=DetectionCoordinateFormat(raw["coordinate_format"]),
        supported_devices=tuple(raw["supported_devices"]),
        inference=_parse_inference(raw["inference"]),
        card_confidence_threshold=_optional_float(raw.get("card_confidence_threshold")),
        checkpoint_revision=raw.get("checkpoint_revision"),
        checkpoint_sha256=raw.get("checkpoint_sha256"),
    )


def _parse_lab(raw: dict[str, Any]) -> LabConfig:
    return LabConfig(
        name=str(raw["name"]),
        color=str(raw["color"]),
        logo_url=str(raw["logo_url"]),
    )


def load_reference_config(
    config_path: Path | None = None,
) -> ReferenceConfig:
    """Load reference model definitions from YAML.

    Args:
        config_path: Path to the YAML config file. When ``None``, the
            bundled default is used.

    Returns:
        Parsed reference configuration.
    """
    path = config_path or _DEFAULT_REFERENCE_CONFIG_PATH
    with open(path) as file:
        raw = yaml.safe_load(file)

    labs = {key: _parse_lab(value) for key, value in raw["labs"].items()}
    models = {key: _parse_model(key, value) for key, value in raw["models"].items()}
    return ReferenceConfig(labs=labs, models=models)


def assert_no_vlm_model_overlap(
    reference_config: ReferenceConfig,
    vlm_model_keys: set[str],
) -> None:
    """Ensure reference keys do not collide with VLM model keys.

    Args:
        reference_config: Loaded reference model registry.
        vlm_model_keys: Model keys from ``models.yaml``.

    Raises:
        ValueError: When a key appears in both registries.
    """
    overlap = sorted(set(reference_config.models) & vlm_model_keys)
    if overlap:
        raise ValueError(
            f"Reference model keys must not appear in models.yaml: {', '.join(overlap)}"
        )
