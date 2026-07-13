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
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
import supervision as sv
from supervision.metrics import MeanAveragePrecision

from vlm_exam.providers.anthropic import compute_resize_dimensions
from vlm_exam.results import RunResult
from vlm_exam.tasks.base import EvaluationResult, Sample, Task

if TYPE_CHECKING:
    from vlm_exam.judge import Judge

_logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE = (
    "Detect all objects in this image. "
    "Output a JSON list where each entry contains the 2D bounding box "
    'in the key "box_2d" and the text label in the key "label". '
    'The "box_2d" value must be [y_min, x_min, y_max, x_max]: integers '
    "between 0 and 1000, normalized to the image height and width. "
    "Return only the JSON list, with no extra text. "
    "Only use these labels: {class_list}"
)

_PIXEL_PROMPT_TEMPLATE = (
    "Detect all objects in this image. "
    "Output a JSON list where each entry contains the 2D bounding box "
    'in the key "box_2d" and the text label in the key "label". '
    'The "box_2d" value must be [x_min, y_min, x_max, y_max]: the '
    "top-left and bottom-right corners in absolute pixel coordinates "
    "of the {width}x{height} pixel image. "
    "Return only the JSON list, with no extra text. "
    "Only use these labels: {class_list}"
)

_PIXEL_YXYX_PROMPT_TEMPLATE = (
    "Detect all objects in this image. "
    "Output a JSON list where each entry contains the 2D bounding box "
    'in the key "box_2d" and the text label in the key "label". '
    'The "box_2d" value must be [y_min, x_min, y_max, x_max]: the '
    "top-left and bottom-right corners in absolute pixel coordinates "
    "of the {width}x{height} pixel image. "
    "Return only the JSON list, with no extra text. "
    "Only use these labels: {class_list}"
)

_NORMALIZED_XYXY_PROMPT_TEMPLATE = (
    "Detect all objects in this image. "
    "Output a JSON list where each entry contains the 2D bounding box "
    'in the key "box_2d" and the text label in the key "label". '
    'The "box_2d" value must be [x_min, y_min, x_max, y_max]: the '
    "top-left and bottom-right corners as integers between 0 and 1000, "
    "normalized to the image width (x) and height (y). "
    "Return only the JSON list, with no extra text. "
    "Only use these labels: {class_list}"
)

PROMPT_CLASS_MODES = ("image", "all")
"""Valid values for the detection prompt class listing mode."""


class DetectionCoordinateFormat(str, Enum):
    """Detection bounding-box coordinate conventions."""

    YXYX_NORMALIZED_0_TO_1000 = "yxyx_normalized_0_to_1000"
    XYXY_NORMALIZED_0_TO_1000 = "xyxy_normalized_0_to_1000"
    XYXY_ABSOLUTE_PROVIDER_UPLOAD = "xyxy_absolute_provider_upload"
    XYXY_ABSOLUTE_ORIGINAL_IMAGE = "xyxy_absolute_original_image"
    YXYX_ABSOLUTE_ORIGINAL_IMAGE = "yxyx_absolute_original_image"


MAP_PASS_THRESHOLD = 0.8
"""Minimum per-image mAP@50 for a sample to count as correct."""


@dataclass(frozen=True)
class DetectionSample(Sample):
    """A detection benchmark sample with ground-truth bounding boxes."""

    image_width: int
    image_height: int
    ground_truth: sv.Detections
    classes: tuple[str, ...] = field(default_factory=tuple)


class DetectionTask(Task):
    """Object detection benchmark task using COCO-format annotations."""

    def __init__(
        self,
        prompt_classes: str = "image",
        coordinate_format: str | DetectionCoordinateFormat = (
            DetectionCoordinateFormat.YXYX_NORMALIZED_0_TO_1000
        ),
    ) -> None:
        """Initialize the detection task.

        Args:
            prompt_classes: Which classes to list in the prompt. ``"image"``
                lists only the classes present in the image's ground truth;
                ``"all"`` lists every dataset class.
            coordinate_format: Coordinate convention requested from the
                model. See :class:`DetectionCoordinateFormat` for valid
                values and box-order semantics.
        """
        if prompt_classes not in PROMPT_CLASS_MODES:
            modes = ", ".join(PROMPT_CLASS_MODES)
            raise ValueError(
                f"Unknown prompt_classes mode {prompt_classes!r}. Valid modes: {modes}"
            )
        self._prompt_classes = prompt_classes
        self._coordinate_format = DetectionCoordinateFormat(coordinate_format)
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
        is listed. In ``"pixel"`` coordinate format the prompt states
        the dimensions of the image as uploaded to Claude (after the
        provider's pre-resize), per Anthropic's coordinates guidance.

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

        match self._coordinate_format:
            case DetectionCoordinateFormat.XYXY_ABSOLUTE_PROVIDER_UPLOAD:
                uploaded_width, uploaded_height = compute_resize_dimensions(
                    sample.image_width, sample.image_height
                )
                return _PIXEL_PROMPT_TEMPLATE.format(
                    width=uploaded_width,
                    height=uploaded_height,
                    class_list=class_list,
                )
            case DetectionCoordinateFormat.XYXY_ABSOLUTE_ORIGINAL_IMAGE:
                return _PIXEL_PROMPT_TEMPLATE.format(
                    width=sample.image_width,
                    height=sample.image_height,
                    class_list=class_list,
                )
            case DetectionCoordinateFormat.YXYX_ABSOLUTE_ORIGINAL_IMAGE:
                return _PIXEL_YXYX_PROMPT_TEMPLATE.format(
                    width=sample.image_width,
                    height=sample.image_height,
                    class_list=class_list,
                )
            case DetectionCoordinateFormat.XYXY_NORMALIZED_0_TO_1000:
                return _NORMALIZED_XYXY_PROMPT_TEMPLATE.format(class_list=class_list)
            case _:
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
            prediction,
            resolution_wh,
            list(sample.classes),
            coordinate_format=self._coordinate_format,
        )

        details: dict[str, Any] = {
            "num_predictions": len(predicted_detections),
            "num_ground_truth": len(sample.ground_truth),
            "prompt_classes": self._prompt_classes,
            "coordinate_format": self._coordinate_format.value,
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

        correct = map50 >= MAP_PASS_THRESHOLD
        return EvaluationResult(correct=correct, match_method="map", details=details)


def parse_prediction(
    prediction: str,
    resolution_wh: tuple[int, int],
    classes: list[str],
    coordinate_format: str | DetectionCoordinateFormat = (
        DetectionCoordinateFormat.YXYX_NORMALIZED_0_TO_1000
    ),
) -> sv.Detections:
    """Parse model JSON output into supervision Detections.

    Falls back to the outermost JSON array substring when the model
    wraps the JSON in prose or unexpected formatting.

    Args:
        prediction: Raw JSON string from the model.
        resolution_wh: Original image (width, height) for coordinate
            scaling.
        classes: List of class names for class_id assignment.
        coordinate_format: Coordinate convention of the prediction.

    Returns:
        Parsed detections, or empty detections on failure.
    """
    format_enum = DetectionCoordinateFormat(coordinate_format)
    match format_enum:
        case DetectionCoordinateFormat.XYXY_ABSOLUTE_PROVIDER_UPLOAD:
            parser = _parse_pixel_json
        case DetectionCoordinateFormat.XYXY_ABSOLUTE_ORIGINAL_IMAGE:
            parser = _parse_pixel_native_json
        case DetectionCoordinateFormat.YXYX_ABSOLUTE_ORIGINAL_IMAGE:
            parser = _parse_pixel_yxyx_native_json
        case DetectionCoordinateFormat.XYXY_NORMALIZED_0_TO_1000:
            parser = _parse_normalized_xyxy_json
        case _:
            parser = _parse_with_supervision

    detections = parser(prediction, resolution_wh, classes)
    if len(detections) > 0:
        return detections

    start = prediction.find("[")
    stop = prediction.rfind("]")
    if start == -1 or stop <= start:
        return detections

    return parser(prediction[start : stop + 1], resolution_wh, classes)


def _parse_with_supervision(
    prediction: str,
    resolution_wh: tuple[int, int],
    classes: list[str],
) -> sv.Detections:
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


def _parse_pixel_json(
    prediction: str,
    resolution_wh: tuple[int, int],
    classes: list[str],
) -> sv.Detections:
    try:
        entries = json.loads(prediction)
    except json.JSONDecodeError:
        return sv.Detections.empty()
    if not isinstance(entries, list):
        return sv.Detections.empty()

    original_width, original_height = resolution_wh
    uploaded_width, uploaded_height = compute_resize_dimensions(
        original_width, original_height
    )
    scale_x = original_width / uploaded_width
    scale_y = original_height / uploaded_height
    class_index = {name: index for index, name in enumerate(classes)}

    xyxy_list: list[list[float]] = []
    class_ids: list[int] = []
    class_names: list[str] = []

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        box = entry.get("box_2d")
        label = entry.get("label")
        if (
            not isinstance(box, list)
            or len(box) != 4
            or not all(isinstance(value, (int, float)) for value in box)
            or label not in class_index
        ):
            continue
        x_min, y_min, x_max, y_max = (float(value) for value in box)
        xyxy_list.append(
            [x_min * scale_x, y_min * scale_y, x_max * scale_x, y_max * scale_y]
        )
        class_ids.append(class_index[label])
        class_names.append(label)

    if not xyxy_list:
        return sv.Detections.empty()

    detections = sv.Detections(
        xyxy=np.array(xyxy_list, dtype=np.float32),
        class_id=np.array(class_ids, dtype=int),
    )
    detections.data["class_name"] = np.array(class_names)
    return detections


def _parse_pixel_native_json(
    prediction: str,
    resolution_wh: tuple[int, int],
    classes: list[str],
) -> sv.Detections:
    return _parse_absolute_pixel_json(
        prediction,
        resolution_wh,
        classes,
        box_order="xyxy",
        scale_x=1.0,
        scale_y=1.0,
    )


def _parse_pixel_yxyx_native_json(
    prediction: str,
    resolution_wh: tuple[int, int],
    classes: list[str],
) -> sv.Detections:
    return _parse_absolute_pixel_json(
        prediction,
        resolution_wh,
        classes,
        box_order="yxyx",
        scale_x=1.0,
        scale_y=1.0,
    )


def _parse_absolute_pixel_json(
    prediction: str,
    resolution_wh: tuple[int, int],
    classes: list[str],
    *,
    box_order: str,
    scale_x: float,
    scale_y: float,
) -> sv.Detections:
    try:
        entries = json.loads(prediction)
    except json.JSONDecodeError:
        return sv.Detections.empty()
    if not isinstance(entries, list):
        return sv.Detections.empty()

    class_index = {name: index for index, name in enumerate(classes)}
    xyxy_list: list[list[float]] = []
    class_ids: list[int] = []
    class_names: list[str] = []

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        box = entry.get("box_2d")
        label = entry.get("label")
        if (
            not isinstance(box, list)
            or len(box) != 4
            or not all(isinstance(value, (int, float)) for value in box)
            or label not in class_index
        ):
            continue
        first, second, third, fourth = (float(value) for value in box)
        if box_order == "yxyx":
            y_min, x_min, y_max, x_max = first, second, third, fourth
        else:
            x_min, y_min, x_max, y_max = first, second, third, fourth
        xyxy_list.append(
            [
                x_min * scale_x,
                y_min * scale_y,
                x_max * scale_x,
                y_max * scale_y,
            ]
        )
        class_ids.append(class_index[label])
        class_names.append(label)

    if not xyxy_list:
        return sv.Detections.empty()

    detections = sv.Detections(
        xyxy=np.array(xyxy_list, dtype=np.float32),
        class_id=np.array(class_ids, dtype=int),
    )
    detections.data["class_name"] = np.array(class_names)
    return detections


def _parse_normalized_xyxy_json(
    prediction: str,
    resolution_wh: tuple[int, int],
    classes: list[str],
) -> sv.Detections:
    try:
        entries = json.loads(prediction)
    except json.JSONDecodeError:
        return sv.Detections.empty()
    if not isinstance(entries, list):
        return sv.Detections.empty()

    width, height = resolution_wh
    scale_x = width / 1000.0
    scale_y = height / 1000.0
    class_index = {name: index for index, name in enumerate(classes)}

    xyxy_list: list[list[float]] = []
    class_ids: list[int] = []
    class_names: list[str] = []

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        box = entry.get("box_2d", entry.get("bbox_2d"))
        label = entry.get("label")
        if (
            not isinstance(box, list)
            or len(box) != 4
            or not all(isinstance(value, (int, float)) for value in box)
            or label not in class_index
        ):
            continue
        x_min, y_min, x_max, y_max = (float(value) for value in box)
        xyxy_list.append(
            [x_min * scale_x, y_min * scale_y, x_max * scale_x, y_max * scale_y]
        )
        class_ids.append(class_index[label])
        class_names.append(label)

    if not xyxy_list:
        return sv.Detections.empty()

    detections = sv.Detections(
        xyxy=np.array(xyxy_list, dtype=np.float32),
        class_id=np.array(class_ids, dtype=int),
    )
    detections.data["class_name"] = np.array(class_names)
    return detections


@dataclass(frozen=True)
class DatasetMapResult:
    """Dataset-level mean average precision over a benchmark run."""

    map50: float
    map75: float
    map50_95: float
    image_count: int


def compute_dataset_map(
    run_result: RunResult,
    sample_index: dict[str, DetectionSample],
) -> DatasetMapResult | None:
    """Compute dataset-level mAP for a detection run.

    Re-parses the stored raw predictions against the dataset ground
    truth and aggregates them into a single mAP computation. Each
    sample's ``coordinate_format`` metadata determines how its stored
    prediction is parsed, defaulting to ``yxyx_normalized_0_to_1000`` for
    runs saved before the format was recorded.

    Args:
        run_result: A detection benchmark run loaded from disk.
        sample_index: Mapping of image basename to detection sample,
            as produced by :func:`build_sample_index`.

    Returns:
        Dataset-level mAP result, or ``None`` when no run sample could
        be matched to the dataset.
    """
    all_predictions: list[sv.Detections] = []
    all_targets: list[sv.Detections] = []

    for sample_result in run_result.samples:
        sample = sample_index.get(sample_result.image)
        if sample is None:
            continue

        resolution_wh = (sample.image_width, sample.image_height)
        predicted = parse_prediction(
            sample_result.predicted,
            resolution_wh,
            list(sample.classes),
            coordinate_format=sample_result.metadata.get(
                "coordinate_format",
                DetectionCoordinateFormat.YXYX_NORMALIZED_0_TO_1000.value,
            ),
        )
        all_predictions.append(predicted)
        all_targets.append(sample.ground_truth)

    if not all_predictions:
        return None

    map_metric = MeanAveragePrecision()
    map_metric.update(all_predictions, all_targets)
    result = map_metric.compute()

    return DatasetMapResult(
        map50=float(result.map50),
        map75=float(result.map75),
        map50_95=float(result.map50_95),
        image_count=len(all_predictions),
    )


def detection_labels(detections: sv.Detections, classes: list[str]) -> list[str]:
    """Resolve display labels for detections.

    Prefers class names embedded by the parser (which survive classes
    absent from the dataset taxonomy) and falls back to indexing the
    class list with ``class_id``.

    Args:
        detections: Detections to label.
        classes: Dataset class names indexed by ``class_id``.

    Returns:
        One label per detection; empty when detections carry no class
        information.
    """
    if "class_name" in detections.data:
        return list(detections.data["class_name"])
    if detections.class_id is not None:
        return [classes[class_id] for class_id in detections.class_id]
    return []


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
