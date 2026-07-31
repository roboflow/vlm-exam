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
from importlib import import_module
from typing import Any, Protocol, cast

import numpy as np
from PIL import Image

from vlm_exam.reference.config import ReferenceModelConfig


@dataclass(frozen=True)
class ReferencePrediction:
    """Structured prediction from a reference model adapter."""

    boxes_xyxy: np.ndarray
    labels: tuple[str, ...]
    confidences: np.ndarray | None = None
    raw: dict[str, Any] = field(default_factory=dict)


class ReferenceAdapter(Protocol):
    """Local open-vocabulary detector adapter."""

    @property
    def device(self) -> str:
        """Device backend used for inference."""
        ...

    def set_vocabulary(self, class_names: tuple[str, ...]) -> None:
        """Configure the open-vocabulary class list for subsequent predictions.

        Args:
            class_names: Dataset class names to detect.
        """
        ...

    def predict(self, image: Image.Image) -> ReferencePrediction:
        """Run inference on a single image.

        Args:
            image: Input image in RGB mode.

        Returns:
            Structured detections in original-image pixel space.
        """
        ...


def create_reference_adapter(
    model_config: ReferenceModelConfig,
    device: str,
) -> ReferenceAdapter:
    """Instantiate a reference adapter by family name.

    Args:
        model_config: Reference model configuration entry.
        device: Device backend (``mps``, ``cpu``, or ``cuda``).

    Returns:
        Loaded adapter ready for inference.

    Raises:
        ValueError: When the adapter family is unknown.
    """
    adapter_modules = {
        "sam3": ("vlm_exam_sam3", "Sam3Adapter"),
        "yoloe": ("vlm_exam_yoloe", "YoloeAdapter"),
    }
    adapter_spec = adapter_modules.get(model_config.adapter)
    if adapter_spec is None:
        raise ValueError(f"Unknown reference adapter {model_config.adapter!r}.")

    module_name, class_name = adapter_spec
    try:
        adapter_class = getattr(import_module(module_name), class_name)
    except ImportError as error:
        raise RuntimeError(
            f"Reference adapter {model_config.adapter!r} is not installed. "
            f"Run this command from reference/{model_config.family} with "
            "`uv run vlm-exam ...`."
        ) from error
    return cast(
        ReferenceAdapter,
        adapter_class(model_config, device=device),
    )
