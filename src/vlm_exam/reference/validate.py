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

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from vlm_exam.reference.manifest import (
    RunManifest,
    load_manifest,
    manifest_path_for_results,
)
from vlm_exam.results import SampleResult, is_failed_sample, load_results
from vlm_exam.tasks.detection import (
    DetectionCoordinateFormat,
    DetectionTask,
    parse_prediction,
)


@dataclass(frozen=True)
class ValidationIssue:
    """Single validation problem in a reference run."""

    sample_index: int | None
    image: str | None
    message: str


@dataclass(frozen=True)
class ValidationReport:
    """Outcome of validating a reference run."""

    results_path: Path
    manifest_path: Path | None
    issues: tuple[ValidationIssue, ...]

    @property
    def ok(self) -> bool:
        """Report whether the run passed validation."""
        return not self.issues


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_sample_prediction(
    sample: SampleResult,
    classes: list[str],
    image_width: int,
    image_height: int,
    coordinate_format: str,
) -> list[ValidationIssue]:
    if is_failed_sample(sample):
        return []

    issues: list[ValidationIssue] = []
    try:
        entries = json.loads(sample.predicted)
    except json.JSONDecodeError as error:
        return [
            ValidationIssue(
                sample.index,
                sample.image,
                f"Invalid JSON prediction: {error}",
            )
        ]

    if not isinstance(entries, list):
        return [
            ValidationIssue(
                sample.index,
                sample.image,
                "Prediction must be a JSON list.",
            )
        ]

    class_set = set(classes)
    for entry in entries:
        if not isinstance(entry, dict):
            issues.append(
                ValidationIssue(sample.index, sample.image, "Entry is not an object.")
            )
            continue
        label = entry.get("label")
        if label not in class_set:
            issues.append(
                ValidationIssue(
                    sample.index,
                    sample.image,
                    f"Unknown label {label!r}.",
                )
            )
        box = entry.get("box_2d")
        if not isinstance(box, list) or len(box) != 4:
            issues.append(
                ValidationIssue(
                    sample.index,
                    sample.image,
                    "box_2d must be a four-number list.",
                )
            )
            continue
        try:
            x_min, y_min, x_max, y_max = (float(value) for value in box)
        except (TypeError, ValueError):
            issues.append(
                ValidationIssue(
                    sample.index,
                    sample.image,
                    "box_2d values must be numeric.",
                )
            )
            continue
        if not all(math.isfinite(value) for value in (x_min, y_min, x_max, y_max)):
            issues.append(
                ValidationIssue(
                    sample.index,
                    sample.image,
                    "box_2d values must be finite.",
                )
            )
            continue
        if x_min >= x_max or y_min >= y_max:
            issues.append(
                ValidationIssue(
                    sample.index,
                    sample.image,
                    "box_2d must have x_min < x_max and y_min < y_max.",
                )
            )
        if x_min < 0 or y_min < 0 or x_max > image_width or y_max > image_height:
            issues.append(
                ValidationIssue(
                    sample.index,
                    sample.image,
                    "box_2d extends outside image bounds.",
                )
            )
        confidence = entry.get("confidence")
        if confidence is not None:
            try:
                score = float(confidence)
            except (TypeError, ValueError):
                issues.append(
                    ValidationIssue(
                        sample.index,
                        sample.image,
                        "confidence must be numeric.",
                    )
                )
            else:
                if not math.isfinite(score) or score < 0.0 or score > 1.0:
                    issues.append(
                        ValidationIssue(
                            sample.index,
                            sample.image,
                            "confidence must be in [0, 1].",
                        )
                    )

    detections = parse_prediction(
        sample.predicted,
        (image_width, image_height),
        classes,
        coordinate_format=DetectionCoordinateFormat(
            sample.metadata.get("coordinate_format", coordinate_format)
        ),
    )
    if len(entries) > 0 and len(detections) == 0:
        issues.append(
            ValidationIssue(
                sample.index,
                sample.image,
                "Prediction JSON did not parse into detections.",
            )
        )
    return issues


def validate_reference_run(
    results_path: Path,
    dataset_directory: str,
) -> ValidationReport:
    """Validate a reference detection JSONL file.

    Args:
        results_path: Path to the reference run JSONL.
        dataset_directory: Dataset directory with ground truth.

    Returns:
        Validation report with any issues found.
    """
    run = load_results(results_path)
    manifest_path = manifest_path_for_results(results_path)
    manifest: RunManifest | None = None
    issues: list[ValidationIssue] = []
    if results_path.suffix != ".jsonl":
        issues.append(ValidationIssue(None, None, "Results file must use .jsonl."))
    expected_stem = f"detection_{run.model}_{run.effort}_{run.timestamp}"
    if results_path.stem != expected_stem:
        issues.append(
            ValidationIssue(
                None,
                None,
                f"Filename stem {results_path.stem!r} != expected {expected_stem!r}.",
            )
        )
    if run.task != "detection":
        issues.append(ValidationIssue(None, None, "Run task is not detection."))
    if run.effort != "reference":
        issues.append(ValidationIssue(None, None, "Run effort is not reference."))

    if not manifest_path.exists():
        issues.append(ValidationIssue(None, None, f"Missing manifest: {manifest_path}"))
    else:
        manifest = load_manifest(manifest_path)
        if manifest.run_type != "reference":
            issues.append(
                ValidationIssue(None, None, "Manifest run_type is not reference.")
            )
        if manifest.model != run.model:
            issues.append(
                ValidationIssue(
                    None,
                    None,
                    f"Manifest model {manifest.model!r} != run model {run.model!r}.",
                )
            )
        if manifest.timestamp != run.timestamp:
            issues.append(
                ValidationIssue(None, None, "Manifest timestamp does not match run.")
            )
        if manifest.task != run.task:
            issues.append(
                ValidationIssue(None, None, "Manifest task does not match run.")
            )
        if manifest.effort != run.effort:
            issues.append(
                ValidationIssue(None, None, "Manifest effort does not match run.")
            )

    task = DetectionTask()
    samples = task.load_samples(dataset_directory)
    sample_by_image = {Path(sample.image_path).name: sample for sample in samples}
    if not run.samples:
        issues.append(ValidationIssue(None, None, "Run contains no samples."))
        return ValidationReport(
            results_path=results_path,
            manifest_path=manifest_path if manifest_path.exists() else None,
            issues=tuple(issues),
        )
    coordinate_format = run.samples[0].metadata.get(
        "coordinate_format",
        "xyxy_absolute_original_image",
    )
    run_images = [sample.image for sample in run.samples]
    duplicate_images = sorted(
        image for image, count in Counter(run_images).items() if count > 1
    )
    if duplicate_images:
        issues.append(
            ValidationIssue(
                None,
                None,
                f"Run contains {len(duplicate_images)} duplicate image(s).",
            )
        )
    missing_images = sorted(set(sample_by_image) - set(run_images))
    if missing_images:
        issues.append(
            ValidationIssue(
                None,
                None,
                f"Run is missing {len(missing_images)} dataset image(s).",
            )
        )
    expected_indexes = list(range(len(run.samples)))
    if [sample.index for sample in run.samples] != expected_indexes:
        issues.append(
            ValidationIssue(
                None, None, "Sample indexes are not contiguous and ordered."
            )
        )

    for sample in run.samples:
        detection_sample = sample_by_image.get(sample.image)
        if detection_sample is None:
            issues.append(
                ValidationIssue(
                    sample.index,
                    sample.image,
                    "Image not found in dataset.",
                )
            )
            continue
        issues.extend(
            _validate_sample_prediction(
                sample,
                list(detection_sample.classes),
                detection_sample.image_width,
                detection_sample.image_height,
                coordinate_format,
            )
        )
        if is_failed_sample(sample):
            issues.append(
                ValidationIssue(sample.index, sample.image, "Sample inference failed.")
            )

    if manifest is not None:
        annotations_path = Path(dataset_directory) / "_annotations.coco.json"
        if manifest.dataset_annotations_sha256 != _file_sha256(annotations_path):
            issues.append(
                ValidationIssue(
                    None,
                    None,
                    "Manifest dataset annotations hash does not match.",
                )
            )
        expected_counts = {
            "dataset_image_count": len(samples),
            "completed_sample_count": len(run.samples),
            "failed_sample_count": sum(
                1 for sample in run.samples if is_failed_sample(sample)
            ),
        }
        for field_name, expected_count in expected_counts.items():
            if getattr(manifest, field_name) != expected_count:
                issues.append(
                    ValidationIssue(
                        None,
                        None,
                        f"Manifest {field_name} does not match run.",
                    )
                )

    return ValidationReport(
        results_path=results_path,
        manifest_path=manifest_path if manifest_path.exists() else None,
        issues=tuple(issues),
    )
