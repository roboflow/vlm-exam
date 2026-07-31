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
from typing import Literal

import supervision as sv

from vlm_exam.reference.constants import REFERENCE_EFFORT
from vlm_exam.results import RunResult, SampleResult
from vlm_exam.tasks.detection import (
    DetectionSample,
    compute_image_map50,
    parse_prediction,
    recorded_coordinate_format,
    recorded_uploaded_wh,
)

SelectedPrompt = Literal["baseline", "image-conditioned"]


@dataclass(frozen=True)
class BestPromptSelection:
    """Per-image winner between baseline and image-conditioned prompts."""

    image: str
    baseline_map50: float
    image_conditioned_map50: float
    selected: SelectedPrompt


@dataclass(frozen=True)
class BestPromptMergeResult:
    """Merged reference run built from per-image prompt winners."""

    merged_run: RunResult
    selections: tuple[BestPromptSelection, ...]
    baseline_wins: int
    image_conditioned_wins: int
    ties: int


def _parse_sample_predictions(
    sample_result: SampleResult,
    sample: DetectionSample,
) -> sv.Detections:
    resolution_wh = (sample.image_width, sample.image_height)
    return parse_prediction(
        sample_result.predicted,
        resolution_wh,
        list(sample.classes),
        coordinate_format=recorded_coordinate_format(sample_result.metadata),
        uploaded_wh=recorded_uploaded_wh(sample_result.metadata),
    )


def merge_best_prompt_run(
    baseline_run: RunResult,
    image_conditioned_run: RunResult,
    sample_index: dict[str, DetectionSample],
) -> BestPromptMergeResult:
    """Build a synthetic run using the better per-image prompt for each image.

    Compares native per-image mAP@50 between baseline and image-conditioned
    predictions. When scores tie, baseline predictions are kept.

    Args:
        baseline_run: Reference run with original class-name prompts.
        image_conditioned_run: Reference run with image-conditioned prompts.
        sample_index: Mapping of image basename to detection sample.

    Returns:
        Merged run and per-image selection metadata.

    Raises:
        ValueError: When runs differ in model or image sets.
    """
    if baseline_run.model != image_conditioned_run.model:
        raise ValueError(
            f"Runs must share a model key; got {baseline_run.model!r} and "
            f"{image_conditioned_run.model!r}."
        )

    baseline_by_image = {sample.image: sample for sample in baseline_run.samples}
    image_conditioned_by_image = {
        sample.image: sample for sample in image_conditioned_run.samples
    }
    baseline_images = set(baseline_by_image)
    image_conditioned_images = set(image_conditioned_by_image)
    if baseline_images != image_conditioned_images:
        missing_from_baseline = sorted(image_conditioned_images - baseline_images)
        missing_from_image_conditioned = sorted(
            baseline_images - image_conditioned_images
        )
        raise ValueError(
            "Baseline and image-conditioned runs must cover the same images. "
            f"Missing from baseline: {missing_from_baseline[:5]}; "
            f"missing from image-conditioned: {missing_from_image_conditioned[:5]}."
        )

    selections: list[BestPromptSelection] = []
    merged_samples: list[SampleResult] = []
    baseline_wins = 0
    image_conditioned_wins = 0
    ties = 0

    for index, image in enumerate(sorted(baseline_images)):
        baseline_sample = baseline_by_image[image]
        image_conditioned_sample = image_conditioned_by_image[image]
        dataset_sample = sample_index.get(image)
        if dataset_sample is None:
            continue

        baseline_predictions = _parse_sample_predictions(
            baseline_sample,
            dataset_sample,
        )
        image_conditioned_predictions = _parse_sample_predictions(
            image_conditioned_sample,
            dataset_sample,
        )
        baseline_map50 = compute_image_map50(
            baseline_predictions,
            dataset_sample.ground_truth,
        )
        image_conditioned_map50 = compute_image_map50(
            image_conditioned_predictions,
            dataset_sample.ground_truth,
        )

        if image_conditioned_map50 > baseline_map50:
            selected: SelectedPrompt = "image-conditioned"
            winner = image_conditioned_sample
            image_conditioned_wins += 1
        else:
            selected = "baseline"
            winner = baseline_sample
            if image_conditioned_map50 < baseline_map50:
                baseline_wins += 1
            else:
                ties += 1

        selections.append(
            BestPromptSelection(
                image=image,
                baseline_map50=baseline_map50,
                image_conditioned_map50=image_conditioned_map50,
                selected=selected,
            )
        )
        merged_samples.append(
            SampleResult(
                index=index,
                image=image,
                expected=baseline_sample.expected,
                predicted=winner.predicted,
                correct=True,
                input_tokens=0,
                output_tokens=0,
                metadata={
                    **winner.metadata,
                    "best_prompt_selected": selected,
                    "best_prompt_baseline_map50": baseline_map50,
                    "best_prompt_image_conditioned_map50": image_conditioned_map50,
                },
            )
        )

    merged_run = RunResult(
        model=baseline_run.model,
        effort=REFERENCE_EFFORT,
        task=baseline_run.task,
        timestamp=baseline_run.timestamp,
        samples=merged_samples,
    )
    return BestPromptMergeResult(
        merged_run=merged_run,
        selections=tuple(selections),
        baseline_wins=baseline_wins,
        image_conditioned_wins=image_conditioned_wins,
        ties=ties,
    )
