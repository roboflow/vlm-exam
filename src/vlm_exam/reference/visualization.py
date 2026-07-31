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

from typing import Any

import supervision as sv

from vlm_exam.config import BenchmarkConfig, ModelConfig, PricingConfig, RouteConfig
from vlm_exam.reference.config import ReferenceConfig
from vlm_exam.tasks.detection import detection_labels


def build_reference_card_config(
    reference_key: str,
    reference_config: ReferenceConfig,
    vlm_config: BenchmarkConfig,
) -> BenchmarkConfig:
    """Build a minimal benchmark config for reference detection hero cards.

    Args:
        reference_key: Reference model key from ``reference_models.yaml``.
        reference_config: Loaded reference model registry.
        vlm_config: VLM benchmark config supplying shared renderer settings.

    Returns:
        Synthetic config with one model entry for card rendering.

    Raises:
        KeyError: When the reference key is unknown.
        ValueError: When the reference model has no configured lab branding.
    """
    reference_model = reference_config.models[reference_key]
    if reference_model.lab is None:
        raise ValueError(
            f"Reference model {reference_key!r} has no lab; "
            "set lab in reference_models.yaml for card rendering."
        )
    labs = {**vlm_config.labs, **reference_config.labs}
    if reference_model.lab not in labs:
        raise ValueError(
            f"Lab {reference_model.lab!r} for reference model {reference_key!r} "
            "is not defined in reference_models.yaml."
        )

    model_config = ModelConfig(
        name=reference_model.name,
        lab=reference_model.lab,
        routes=(RouteConfig(provider="reference"),),
        pricing=PricingConfig(
            input_per_million_tokens=0.0,
            output_per_million_tokens=0.0,
        ),
        detection_coordinate_format=reference_model.coordinate_format,
    )
    return BenchmarkConfig(
        labs=labs,
        models={reference_key: model_config},
    )


def prompt_label_map_from_metadata(metadata: dict[str, Any]) -> dict[str, str] | None:
    """Build a canonical-class to prompt-text map from one sample's metadata."""
    prompt_class_names = metadata.get("prompt_class_names")
    prompt_texts = metadata.get("prompt_texts")
    if not isinstance(prompt_class_names, list) or not isinstance(prompt_texts, list):
        return None
    if len(prompt_class_names) != len(prompt_texts):
        return None
    return {
        str(class_name): str(prompt_text)
        for class_name, prompt_text in zip(
            prompt_class_names,
            prompt_texts,
            strict=True,
        )
    }


def detection_labels_for_card(
    detections: sv.Detections,
    classes: list[str],
    *,
    label_classes: str,
    prompt_label_map: dict[str, str] | None,
) -> list[str]:
    """Resolve card box labels, optionally swapping in augmented prompt text."""
    labels = detection_labels(detections, classes)
    if label_classes != "augmented" or prompt_label_map is None:
        return labels
    return [prompt_label_map.get(label, label) for label in labels]


def resolve_card_confidence_threshold(
    reference_key: str,
    reference_config: ReferenceConfig,
    override: float | None = None,
) -> float | None:
    """Resolve the post-hoc confidence cutoff used when drawing detection cards.

    Args:
        reference_key: Reference model key from ``reference_models.yaml``.
        reference_config: Loaded reference model registry.
        override: Explicit CLI threshold; when set, replaces the config default.

    Returns:
        Minimum confidence for boxes drawn on cards, or ``None`` when no filter
        applies.
    """
    if override is not None:
        return override
    return reference_config.models[reference_key].card_confidence_threshold
