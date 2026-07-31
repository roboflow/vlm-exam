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
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import supervision as sv
from supervision.metrics import MeanAveragePrecision

from vlm_exam.results import RunResult
from vlm_exam.tasks.detection import (
    DetectionCoordinateFormat,
    DetectionSample,
    filter_detections_by_confidence,
    filter_prediction_json,
    parse_prediction,
)

IOU_MATCH_THRESHOLD = 0.5
CONFIDENCE_REPORT_THRESHOLD = 0.25
LOW_SUPPORT_GT_COUNT = 5
GOOD_AP50_THRESHOLD = 0.3


class FailureMode(str, Enum):
    """Per-class failure classification for reference detection analysis."""

    NEVER_PREDICTED = "never-predicted"
    PREDICTED_ELSEWHERE = "predicted-elsewhere"
    MISLABELED = "mislabeled"
    PARTIAL_RECALL = "partial-recall"
    POOR_LOCALIZATION = "poor-localization"
    GOOD = "good"


class ObjectCountBucket(str, Enum):
    """Image groups by ground-truth object count."""

    ONE_TO_TWO = "1-2"
    THREE_TO_FIVE = "3-5"
    SIX_TO_FIFTEEN = "6-15"
    SIXTEEN_TO_FORTY = "16-40"
    FORTY_ONE_PLUS = "41+"


class SizeBucket(str, Enum):
    """COCO-style ground-truth box area buckets."""

    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


@dataclass(frozen=True)
class ClassAnalysisRow:
    """Per-class detection metrics for one reference run."""

    class_name: str
    class_id: int
    ap50: float
    ap50_95: float
    ground_truth_count: int
    prediction_count: int
    prediction_count_conf025: int
    recall: float
    mean_matched_iou: float
    false_positive_count: int
    failure_mode: FailureMode
    low_support: bool
    name_difficulty: str


@dataclass(frozen=True)
class BucketAnalysisRow:
    """Pooled metrics for a group of images."""

    bucket: str
    image_count: int
    map50: float
    map75: float
    map50_95: float
    recall_class_aware: float
    recall_class_agnostic: float
    mean_matched_iou: float
    false_negative_count: int
    false_positive_count: int
    predictions_per_image: float
    duplicate_detection_count: int


@dataclass(frozen=True)
class ConfusionPair:
    """Ground-truth class matched by a prediction of another class."""

    ground_truth_class: str
    predicted_class: str
    count: int


@dataclass(frozen=True)
class ReferenceAnalysisReport:
    """Full analysis output for one reference detection run."""

    model: str
    map50: float
    map75: float
    map50_95: float
    recall_class_aware: float
    recall_class_agnostic: float
    per_class: tuple[ClassAnalysisRow, ...]
    object_count_buckets: tuple[BucketAnalysisRow, ...]
    size_buckets: tuple[BucketAnalysisRow, ...]
    confusion_pairs: tuple[ConfusionPair, ...]
    confidence_summary: dict[str, float]


def object_count_bucket(object_count: int) -> ObjectCountBucket:
    """Assign an image to an object-count bucket."""
    if object_count <= 2:
        return ObjectCountBucket.ONE_TO_TWO
    if object_count <= 5:
        return ObjectCountBucket.THREE_TO_FIVE
    if object_count <= 15:
        return ObjectCountBucket.SIX_TO_FIFTEEN
    if object_count <= 40:
        return ObjectCountBucket.SIXTEEN_TO_FORTY
    return ObjectCountBucket.FORTY_ONE_PLUS


def size_bucket(area: float) -> SizeBucket:
    """Assign a ground-truth box to a COCO-style size bucket."""
    if area < 32 * 32:
        return SizeBucket.SMALL
    if area < 96 * 96:
        return SizeBucket.MEDIUM
    return SizeBucket.LARGE


def _coordinate_format(
    sample_result_metadata: dict[str, Any],
) -> DetectionCoordinateFormat:
    return DetectionCoordinateFormat(
        sample_result_metadata.get(
            "coordinate_format",
            DetectionCoordinateFormat.XYXY_ABSOLUTE_ORIGINAL_IMAGE.value,
        )
    )


def _parse_run_predictions(
    run: RunResult,
    sample_index: dict[str, DetectionSample],
    *,
    min_confidence: float | None = None,
) -> tuple[list[sv.Detections], list[sv.Detections], list[str]]:
    predictions: list[sv.Detections] = []
    targets: list[sv.Detections] = []
    images: list[str] = []
    for sample_result in run.samples:
        sample = sample_index.get(sample_result.image)
        if sample is None:
            continue
        prediction_text = sample_result.predicted
        if min_confidence is not None:
            prediction_text = filter_prediction_json(prediction_text, min_confidence)
        predicted = parse_prediction(
            prediction_text,
            (sample.image_width, sample.image_height),
            list(sample.classes),
            coordinate_format=_coordinate_format(sample_result.metadata),
        )
        if min_confidence is not None:
            predicted = filter_detections_by_confidence(predicted, min_confidence)
        predictions.append(predicted)
        targets.append(sample.ground_truth)
        images.append(sample_result.image)
    return predictions, targets, images


def _prediction_order(predictions: sv.Detections) -> np.ndarray:
    if predictions.confidence is not None and len(predictions) > 0:
        return np.argsort(-predictions.confidence)
    return np.arange(len(predictions))


def _match_predictions(
    targets: sv.Detections,
    predictions: sv.Detections,
    *,
    class_aware: bool,
) -> tuple[set[int], set[int], list[float], list[tuple[int, int]]]:
    matched_ground_truth: set[int] = set()
    matched_predictions: set[int] = set()
    matched_ious: list[float] = []
    cross_class_matches: list[tuple[int, int]] = []

    if len(targets) == 0 or len(predictions) == 0:
        return (
            matched_ground_truth,
            matched_predictions,
            matched_ious,
            cross_class_matches,
        )

    ious = sv.box_iou_batch(targets.xyxy, predictions.xyxy)
    for prediction_index in _prediction_order(predictions):
        best_ground_truth_index = -1
        best_iou = IOU_MATCH_THRESHOLD
        for ground_truth_index in range(len(targets)):
            if ground_truth_index in matched_ground_truth:
                continue
            if (
                class_aware
                and targets.class_id[ground_truth_index]
                != predictions.class_id[prediction_index]
            ):
                continue
            overlap = float(ious[ground_truth_index, prediction_index])
            if overlap >= best_iou:
                best_iou = overlap
                best_ground_truth_index = ground_truth_index
        if best_ground_truth_index < 0:
            continue
        matched_ground_truth.add(best_ground_truth_index)
        matched_predictions.add(prediction_index)
        matched_ious.append(best_iou)
        if (
            not class_aware
            and targets.class_id[best_ground_truth_index]
            != predictions.class_id[prediction_index]
        ):
            cross_class_matches.append((best_ground_truth_index, prediction_index))

    return matched_ground_truth, matched_predictions, matched_ious, cross_class_matches


def _recall_metrics(
    predictions: list[sv.Detections],
    targets: list[sv.Detections],
    *,
    class_aware: bool,
) -> tuple[float, float, int, int, int]:
    true_positive_count = 0
    false_negative_count = 0
    false_positive_count = 0
    matched_ious: list[float] = []
    duplicate_detection_count = 0

    for prediction, target in zip(predictions, targets, strict=True):
        matched_ground_truth, matched_predictions, image_ious, _ = _match_predictions(
            target,
            prediction,
            class_aware=class_aware,
        )
        true_positive_count += len(matched_ground_truth)
        false_negative_count += len(target) - len(matched_ground_truth)
        false_positive_count += len(prediction) - len(matched_predictions)
        matched_ious.extend(image_ious)

        if class_aware and len(prediction) > 0 and len(target) > 0:
            ious = sv.box_iou_batch(target.xyxy, prediction.xyxy)
            for prediction_index in range(len(prediction)):
                if prediction_index in matched_predictions:
                    continue
                overlap = float(np.max(ious[:, prediction_index]))
                if overlap >= IOU_MATCH_THRESHOLD:
                    duplicate_detection_count += 1

    total_ground_truth = true_positive_count + false_negative_count
    recall = true_positive_count / total_ground_truth if total_ground_truth else 0.0
    mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0
    return (
        recall,
        mean_iou,
        false_negative_count,
        false_positive_count,
        duplicate_detection_count,
    )


def _compute_map(
    predictions: list[sv.Detections],
    targets: list[sv.Detections],
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    metric = MeanAveragePrecision()
    metric.update(predictions, targets)
    result = metric.compute()
    return (
        float(result.map50),
        float(result.map75),
        float(result.map50_95),
        result.ap_per_class,
        result.matched_classes,
    )


def _prediction_counts(
    run: RunResult,
    classes: tuple[str, ...],
    *,
    min_confidence: float | None = None,
) -> tuple[dict[str, int], dict[str, int]]:
    total_counts: dict[str, int] = defaultdict(int)
    conf_counts: dict[str, int] = defaultdict(int)
    for sample_result in run.samples:
        try:
            entries = json.loads(sample_result.predicted)
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            label = entry.get("label")
            if not isinstance(label, str) or label not in classes:
                continue
            confidence = entry.get("confidence")
            try:
                score = float(confidence)
            except (TypeError, ValueError):
                score = 0.0
            if min_confidence is not None and score < min_confidence:
                continue
            total_counts[label] += 1
            if score >= CONFIDENCE_REPORT_THRESHOLD:
                conf_counts[label] += 1
    return total_counts, conf_counts


def _ground_truth_counts(
    targets: list[sv.Detections], classes: tuple[str, ...]
) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for target in targets:
        if target.class_id is None:
            continue
        for class_id in target.class_id.tolist():
            counts[classes[class_id]] += 1
    return counts


def _class_failure_mode(
    *,
    class_name: str,
    ap50: float,
    ground_truth_count: int,
    prediction_count: int,
    recall: float,
    mean_matched_iou: float,
    mislabeled_count: int,
) -> FailureMode:
    if prediction_count == 0:
        return FailureMode.NEVER_PREDICTED
    if recall <= 0.0 and mislabeled_count > 0:
        return FailureMode.MISLABELED
    if recall <= 0.0:
        return FailureMode.PREDICTED_ELSEWHERE
    if 0.0 < recall < 1.0 and mean_matched_iou >= IOU_MATCH_THRESHOLD:
        return FailureMode.PARTIAL_RECALL
    if recall > 0.0 and mean_matched_iou < IOU_MATCH_THRESHOLD:
        return FailureMode.POOR_LOCALIZATION
    if ap50 >= GOOD_AP50_THRESHOLD:
        return FailureMode.GOOD
    if recall > 0.0:
        return FailureMode.PARTIAL_RECALL
    return FailureMode.PREDICTED_ELSEWHERE


def _per_class_recall_and_confusion(
    predictions: list[sv.Detections],
    targets: list[sv.Detections],
    classes: tuple[str, ...],
) -> tuple[
    dict[str, float], dict[str, float], dict[str, int], Counter[tuple[str, str]]
]:
    per_class_true_positive: dict[str, int] = defaultdict(int)
    per_class_ground_truth: dict[str, int] = defaultdict(int)
    per_class_false_positive: dict[str, int] = defaultdict(int)
    per_class_matched_iou: dict[str, list[float]] = defaultdict(list)
    per_class_mislabeled: dict[str, int] = defaultdict(int)
    confusion_counter: Counter[tuple[str, str]] = Counter()

    for prediction, target in zip(predictions, targets, strict=True):
        if len(target) == 0:
            continue
        if target.class_id is not None:
            for class_id in target.class_id.tolist():
                per_class_ground_truth[classes[class_id]] += 1

        matched_ground_truth, matched_predictions, _, cross_class_matches = (
            _match_predictions(
                target,
                prediction,
                class_aware=True,
            )
        )
        for ground_truth_index in matched_ground_truth:
            class_name = classes[int(target.class_id[ground_truth_index])]
            per_class_true_positive[class_name] += 1

        _, _, agnostic_ious, cross_matches = _match_predictions(
            target,
            prediction,
            class_aware=False,
        )
        for ground_truth_index, prediction_index in cross_matches:
            ground_truth_class = classes[int(target.class_id[ground_truth_index])]
            predicted_class = classes[int(prediction.class_id[prediction_index])]
            per_class_mislabeled[ground_truth_class] += 1
            confusion_counter[(ground_truth_class, predicted_class)] += 1

        aware_ground_truth, _, aware_ious, _ = _match_predictions(
            target,
            prediction,
            class_aware=True,
        )
        for ground_truth_index, iou in zip(
            sorted(aware_ground_truth), aware_ious, strict=False
        ):
            class_name = classes[int(target.class_id[ground_truth_index])]
            per_class_matched_iou[class_name].append(iou)

        for prediction_index in range(len(prediction)):
            if prediction_index in matched_predictions:
                continue
            predicted_class = classes[int(prediction.class_id[prediction_index])]
            per_class_false_positive[predicted_class] += 1

    recalls: dict[str, float] = {}
    mean_ious: dict[str, float] = {}
    for class_name, ground_truth_count in per_class_ground_truth.items():
        recalls[class_name] = (
            per_class_true_positive[class_name] / ground_truth_count
            if ground_truth_count
            else 0.0
        )
        ious = per_class_matched_iou.get(class_name, [])
        mean_ious[class_name] = float(np.mean(ious)) if ious else 0.0

    return recalls, mean_ious, dict(per_class_false_positive), confusion_counter


def _confidence_summary(
    predictions: list[sv.Detections],
    targets: list[sv.Detections],
) -> dict[str, float]:
    true_positive_scores: list[float] = []
    false_positive_scores: list[float] = []

    for prediction, target in zip(predictions, targets, strict=True):
        if len(prediction) == 0:
            continue
        scores = (
            prediction.confidence.tolist()
            if prediction.confidence is not None
            else [0.0] * len(prediction)
        )
        matched_ground_truth, matched_predictions, _, _ = _match_predictions(
            target,
            prediction,
            class_aware=True,
        )
        for prediction_index in matched_predictions:
            true_positive_scores.append(float(scores[prediction_index]))
        for prediction_index in range(len(prediction)):
            if prediction_index not in matched_predictions:
                false_positive_scores.append(float(scores[prediction_index]))

    def _percentile(values: list[float], percentile: float) -> float:
        if not values:
            return 0.0
        return float(np.percentile(np.array(values), percentile))

    return {
        "true_positive_median": _percentile(true_positive_scores, 50),
        "false_positive_median": _percentile(false_positive_scores, 50),
        "true_positive_count": float(len(true_positive_scores)),
        "false_positive_count": float(len(false_positive_scores)),
    }


def _bucket_rows(
    predictions: list[sv.Detections],
    targets: list[sv.Detections],
    bucket_key: str,
    bucket_assignments: list[str],
) -> tuple[BucketAnalysisRow, ...]:
    grouped_predictions: dict[str, list[sv.Detections]] = defaultdict(list)
    grouped_targets: dict[str, list[sv.Detections]] = defaultdict(list)
    for prediction, target, bucket in zip(
        predictions, targets, bucket_assignments, strict=True
    ):
        grouped_predictions[bucket].append(prediction)
        grouped_targets[bucket].append(target)

    rows: list[BucketAnalysisRow] = []
    for bucket in sorted(grouped_predictions):
        bucket_predictions = grouped_predictions[bucket]
        bucket_targets = grouped_targets[bucket]
        map50, map75, map50_95, _, _ = _compute_map(bucket_predictions, bucket_targets)
        (
            recall_aware,
            mean_iou,
            false_negative_count,
            false_positive_count,
            duplicate_count,
        ) = _recall_metrics(bucket_predictions, bucket_targets, class_aware=True)
        recall_agnostic, _, _, _, _ = _recall_metrics(
            bucket_predictions,
            bucket_targets,
            class_aware=False,
        )
        prediction_total = sum(len(prediction) for prediction in bucket_predictions)
        rows.append(
            BucketAnalysisRow(
                bucket=bucket,
                image_count=len(bucket_predictions),
                map50=map50,
                map75=map75,
                map50_95=map50_95,
                recall_class_aware=recall_aware,
                recall_class_agnostic=recall_agnostic,
                mean_matched_iou=mean_iou,
                false_negative_count=false_negative_count,
                false_positive_count=false_positive_count,
                predictions_per_image=(
                    prediction_total / len(bucket_predictions)
                    if bucket_predictions
                    else 0.0
                ),
                duplicate_detection_count=duplicate_count,
            )
        )
    return tuple(rows)


def build_reference_analysis_report(
    run: RunResult,
    sample_index: dict[str, DetectionSample],
    *,
    name_difficulty: dict[str, str] | None = None,
    min_confidence: float | None = None,
) -> ReferenceAnalysisReport:
    """Build a full analysis report for one reference detection run.

    Args:
        run: Reference detection run loaded from disk.
        sample_index: Mapping of image basename to detection sample.
        name_difficulty: Optional mapping of class name to difficulty tag.
        min_confidence: When set, drop predictions below this confidence
            before computing metrics.

    Returns:
        Structured analysis report with per-class and bucketed metrics.
    """
    predictions, targets, _ = _parse_run_predictions(
        run,
        sample_index,
        min_confidence=min_confidence,
    )
    if not predictions:
        raise ValueError("No predictions could be parsed for this run.")

    classes = next(iter(sample_index.values())).classes
    map50, map75, map50_95, ap_per_class, matched_classes = _compute_map(
        predictions, targets
    )
    recall_aware, _, _, _, _ = _recall_metrics(predictions, targets, class_aware=True)
    recall_agnostic, _, _, _, _ = _recall_metrics(
        predictions, targets, class_aware=False
    )

    ap50_by_class_id = {
        int(class_id): float(ap50)
        for class_id, ap50 in zip(
            matched_classes.tolist(), ap_per_class[:, 0].tolist(), strict=True
        )
    }
    ap50_95_by_class_id = {
        int(class_id): float(ap)
        for class_id, ap in zip(
            matched_classes.tolist(),
            ap_per_class[:, 1:].mean(axis=1).tolist(),
            strict=True,
        )
    }

    ground_truth_counts = _ground_truth_counts(targets, classes)
    prediction_counts, prediction_counts_conf025 = _prediction_counts(
        run,
        classes,
        min_confidence=min_confidence,
    )
    recalls, mean_ious, false_positive_counts, confusion_counter = (
        _per_class_recall_and_confusion(
            predictions,
            targets,
            classes,
        )
    )
    mislabeled_counts: dict[str, int] = defaultdict(int)
    for (ground_truth_class, _), count in confusion_counter.items():
        mislabeled_counts[ground_truth_class] += count

    difficulty = name_difficulty or {}
    per_class_rows: list[ClassAnalysisRow] = []
    for class_id, class_name in enumerate(classes):
        ground_truth_count = ground_truth_counts.get(class_name, 0)
        if ground_truth_count == 0:
            continue
        prediction_count = prediction_counts.get(class_name, 0)
        recall = recalls.get(class_name, 0.0)
        mean_iou = mean_ious.get(class_name, 0.0)
        ap50 = ap50_by_class_id.get(class_id, 0.0)
        ap50_95 = ap50_95_by_class_id.get(class_id, 0.0)
        per_class_rows.append(
            ClassAnalysisRow(
                class_name=class_name,
                class_id=class_id,
                ap50=ap50,
                ap50_95=ap50_95,
                ground_truth_count=ground_truth_count,
                prediction_count=prediction_count,
                prediction_count_conf025=prediction_counts_conf025.get(class_name, 0),
                recall=recall,
                mean_matched_iou=mean_iou,
                false_positive_count=false_positive_counts.get(class_name, 0),
                failure_mode=_class_failure_mode(
                    class_name=class_name,
                    ap50=ap50,
                    ground_truth_count=ground_truth_count,
                    prediction_count=prediction_count,
                    recall=recall,
                    mean_matched_iou=mean_iou,
                    mislabeled_count=mislabeled_counts.get(class_name, 0),
                ),
                low_support=ground_truth_count < LOW_SUPPORT_GT_COUNT,
                name_difficulty=difficulty.get(class_name, "unknown"),
            )
        )

    per_class_rows.sort(key=lambda row: (-row.ap50, row.class_name))

    object_count_assignments = [
        object_count_bucket(len(target)).value for target in targets
    ]
    size_assignments: list[str] = []
    for target in targets:
        if len(target) == 0:
            size_assignments.append(SizeBucket.MEDIUM.value)
            continue
        areas = (target.xyxy[:, 2] - target.xyxy[:, 0]) * (
            target.xyxy[:, 3] - target.xyxy[:, 1]
        )
        dominant_bucket = Counter(
            size_bucket(float(area)).value for area in areas
        ).most_common(1)[0][0]
        size_assignments.append(dominant_bucket)

    confusion_pairs = tuple(
        ConfusionPair(ground_truth_class=gt, predicted_class=pred, count=count)
        for (gt, pred), count in confusion_counter.most_common(30)
    )

    return ReferenceAnalysisReport(
        model=run.model,
        map50=map50,
        map75=map75,
        map50_95=map50_95,
        recall_class_aware=recall_aware,
        recall_class_agnostic=recall_agnostic,
        per_class=tuple(per_class_rows),
        object_count_buckets=_bucket_rows(
            predictions,
            targets,
            "object_count",
            object_count_assignments,
        ),
        size_buckets=_bucket_rows(predictions, targets, "size", size_assignments),
        confusion_pairs=confusion_pairs,
        confidence_summary=_confidence_summary(predictions, targets),
    )
