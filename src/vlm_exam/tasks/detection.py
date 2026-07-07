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

import json
import logging
import os
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import supervision as sv
from supervision.metrics import MeanAveragePrecision

from vlm_exam.tasks.base import EvaluationResult, Sample, Task

if TYPE_CHECKING:
    from vlm_exam.judge import Judge

_logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE = (
    "Detect all objects in this image. "
    "Output a JSON list where each entry contains the 2D bounding box "
    'in the key "box_2d" and the text label in the key "label". '
    "Only use these labels: {class_list}"
)

PROMPT_CLASS_MODES = ("image", "all")
"""Valid values for the detection prompt class listing mode."""


@dataclass(frozen=True)
class DetectionSample(Sample):
    """A detection benchmark sample with ground-truth bounding boxes."""

    image_width: int
    image_height: int
    ground_truth: sv.Detections
    classes: tuple[str, ...] = field(default_factory=tuple)


class DetectionTask(Task):
    """Object detection benchmark task using COCO-format annotations."""

    def __init__(self, prompt_classes: str = "image") -> None:
        """Initialize the detection task.

        Args:
            prompt_classes: Which classes to list in the prompt. ``"image"``
                lists only the classes present in the image's ground truth;
                ``"all"`` lists every dataset class.
        """
        if prompt_classes not in PROMPT_CLASS_MODES:
            modes = ", ".join(PROMPT_CLASS_MODES)
            raise ValueError(
                f"Unknown prompt_classes mode {prompt_classes!r}. Valid modes: {modes}"
            )
        self._prompt_classes = prompt_classes
        self._classes: list[str] = []

    @property
    def classes(self) -> list[str]:
        """Category names loaded from the dataset."""
        return self._classes

    def load_samples(self, data_directory: str) -> list[Sample]:
        """Load detection samples from a COCO annotations file.

        Expects the directory to contain ``_annotations.coco.json``
        alongside the image files. Roboflow placeholder categories
        (``supercategory == "none"``) are skipped.

        Args:
            data_directory: Path to the dataset directory.

        Returns:
            List of detection samples.
        """
        annotations_path = os.path.join(data_directory, "_annotations.coco.json")

        with open(annotations_path) as f:
            coco = json.load(f)

        category_id_to_index: dict[int, int] = {}
        self._classes = []
        for category in coco["categories"]:
            if category.get("supercategory") == "none":
                continue
            category_id_to_index[category["id"]] = len(self._classes)
            self._classes.append(category["name"])

        image_id_to_info: dict[int, dict[str, Any]] = {}
        for image_info in coco["images"]:
            image_id_to_info[image_info["id"]] = image_info

        annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for annotation in coco["annotations"]:
            if annotation["category_id"] not in category_id_to_index:
                continue
            annotations_by_image[annotation["image_id"]].append(annotation)

        samples: list[Sample] = []
        classes_tuple = tuple(self._classes)

        for image_id, image_info in image_id_to_info.items():
            image_path = os.path.join(data_directory, image_info["file_name"])
            width = image_info["width"]
            height = image_info["height"]

            image_annotations = annotations_by_image.get(image_id, [])

            if not image_annotations:
                xyxy = np.empty((0, 4), dtype=np.float32)
                class_ids = np.empty((0,), dtype=int)
            else:
                xyxy_list = []
                class_id_list = []
                for ann in image_annotations:
                    x, y, w, h = ann["bbox"]
                    xyxy_list.append([x, y, x + w, y + h])
                    class_id_list.append(category_id_to_index[ann["category_id"]])
                xyxy = np.array(xyxy_list, dtype=np.float32)
                class_ids = np.array(class_id_list, dtype=int)

            ground_truth = sv.Detections(
                xyxy=xyxy,
                class_id=class_ids,
            )

            samples.append(
                DetectionSample(
                    image_path=image_path,
                    image_width=width,
                    image_height=height,
                    ground_truth=ground_truth,
                    classes=classes_tuple,
                )
            )

        return samples

    def build_prompt(self, sample: Sample) -> str:
        """Build a detection prompt with a class list.

        In ``"image"`` mode only the classes present in the sample's
        ground truth are listed; in ``"all"`` mode every dataset class
        is listed.

        Args:
            sample: A ``DetectionSample`` instance.

        Returns:
            Formatted prompt string.
        """
        assert isinstance(sample, DetectionSample)
        if (
            self._prompt_classes == "image"
            and sample.ground_truth.class_id is not None
            and len(sample.ground_truth) > 0
        ):
            present_ids = set(sample.ground_truth.class_id)
            image_classes = [sample.classes[cid] for cid in sorted(present_ids)]
        else:
            image_classes = list(sample.classes)
        class_list = ", ".join(image_classes)
        return _PROMPT_TEMPLATE.format(class_list=class_list)

    def evaluate(
        self,
        sample: Sample,
        prediction: str,
        *,
        match_mode: str = "strict",
        judge: Judge | None = None,
    ) -> EvaluationResult:
        """Evaluate a detection prediction against the ground truth.

        Parses the model JSON response into ``sv.Detections`` using
        ``sv.Detections.from_vlm``, then computes per-image mAP@50.

        Args:
            sample: A ``DetectionSample`` with ground-truth boxes.
            prediction: Raw JSON output from the model.
            match_mode: Unused for detection (kept for interface compat).
            judge: Unused for detection.

        Returns:
            Evaluation result with mAP details.
        """
        assert isinstance(sample, DetectionSample)

        resolution_wh = (sample.image_width, sample.image_height)
        predicted_detections = parse_prediction(
            prediction, resolution_wh, list(sample.classes)
        )

        details: dict[str, Any] = {
            "num_predictions": len(predicted_detections),
            "num_ground_truth": len(sample.ground_truth),
            "prompt_classes": self._prompt_classes,
        }

        if len(sample.ground_truth) == 0 and len(predicted_detections) == 0:
            details["map50"] = 1.0
            return EvaluationResult(correct=True, match_method="map", details=details)

        if len(sample.ground_truth) == 0 or len(predicted_detections) == 0:
            details["map50"] = 0.0
            return EvaluationResult(correct=False, match_method="map", details=details)

        map_metric = MeanAveragePrecision()
        map_metric.update([predicted_detections], [sample.ground_truth])
        result = map_metric.compute()

        map50 = float(result.map50)
        details["map50"] = map50

        correct = map50 >= 0.5
        return EvaluationResult(correct=correct, match_method="map", details=details)


def parse_prediction(
    prediction: str,
    resolution_wh: tuple[int, int],
    classes: list[str],
) -> sv.Detections:
    """Parse model JSON output into supervision Detections.

    Args:
        prediction: Raw JSON string from the model.
        resolution_wh: Image (width, height) for coordinate scaling.
        classes: List of class names for class_id assignment.

    Returns:
        Parsed detections, or empty detections on failure.
    """
    try:
        return sv.Detections.from_vlm(
            vlm=sv.VLM.GOOGLE_GEMINI_2_5,
            result=prediction,
            resolution_wh=resolution_wh,
            classes=classes,
        )
    except Exception:
        _logger.warning("Failed to parse detection response; returning empty.")
        return sv.Detections.empty()


def build_sample_index(samples: Iterable[Sample]) -> dict[str, DetectionSample]:
    """Index detection samples by image file basename.

    Args:
        samples: Samples produced by ``DetectionTask.load_samples``.

    Returns:
        Mapping of image basename to detection sample.
    """
    index: dict[str, DetectionSample] = {}
    for sample in samples:
        if isinstance(sample, DetectionSample):
            index[os.path.basename(sample.image_path)] = sample
    return index
