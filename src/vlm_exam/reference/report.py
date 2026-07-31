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

import supervision as sv

from vlm_exam.config import BenchmarkConfig
from vlm_exam.metrics import build_latest_runs_index
from vlm_exam.reference.constants import REFERENCE_EFFORT
from vlm_exam.reference.manifest import load_manifest, manifest_path_for_results
from vlm_exam.results import RunResult, load_results, load_results_directory
from vlm_exam.tasks.detection import (
    DetectionCoordinateFormat,
    DetectionSample,
    DetectionTask,
    build_sample_index,
    compute_dataset_map,
    parse_prediction,
    recorded_coordinate_format,
)


@dataclass(frozen=True)
class ReferenceReportRow:
    """One row in a reference comparison table."""

    model: str
    run_type: str
    map50: float
    map75: float
    map50_95: float
    image_count: int
    map50_no_confidence: float | None = None


def _strip_confidence_map(
    run: RunResult,
    sample_index: dict[str, DetectionSample],
) -> float | None:
    all_predictions: list[sv.Detections] = []
    all_targets: list[sv.Detections] = []

    for sample_result in run.samples:
        sample = sample_index.get(sample_result.image)
        if sample is None:
            continue
        resolution_wh = (sample.image_width, sample.image_height)
        predicted = parse_prediction(
            sample_result.predicted,
            resolution_wh,
            list(sample.classes),
            coordinate_format=recorded_coordinate_format(
                sample_result.metadata,
                default=DetectionCoordinateFormat.XYXY_ABSOLUTE_ORIGINAL_IMAGE,
            ),
        )
        if predicted.confidence is not None:
            predicted = sv.Detections(
                xyxy=predicted.xyxy,
                class_id=predicted.class_id,
                data=predicted.data,
            )
        all_predictions.append(predicted)
        all_targets.append(sample.ground_truth)

    if not all_predictions:
        return None

    from supervision.metrics import MeanAveragePrecision

    map_metric = MeanAveragePrecision()
    map_metric.update(all_predictions, all_targets)
    result = map_metric.compute()
    return float(result.map50)


def build_reference_report_rows(
    vlm_results_directory: Path,
    reference_results_directory: Path,
    dataset_directory: str,
    config: BenchmarkConfig,
) -> list[ReferenceReportRow]:
    """Build comparison rows for VLM and reference detection runs.

    Args:
        vlm_results_directory: Directory containing VLM JSONL runs.
        reference_results_directory: Directory containing reference JSONL runs.
        dataset_directory: Detection dataset directory.
        config: VLM benchmark configuration.

    Returns:
        Rows sorted by mAP@50 descending.
    """
    task = DetectionTask()
    samples = task.load_samples(dataset_directory)
    sample_index = build_sample_index(samples)

    rows: list[ReferenceReportRow] = []

    vlm_runs = load_results_directory(
        vlm_results_directory,
        pattern="detection_*.jsonl",
    )
    latest_vlm = build_latest_runs_index(vlm_runs, config)
    for (_, effort, model), run in latest_vlm.items():
        if effort != "low" or run.task != "detection":
            continue
        map_result = compute_dataset_map(run, sample_index)
        if map_result is None:
            continue
        rows.append(
            ReferenceReportRow(
                model=model,
                run_type="vlm",
                map50=map_result.map50,
                map75=map_result.map75,
                map50_95=map_result.map50_95,
                image_count=map_result.image_count,
            )
        )

    reference_runs = load_results_directory(
        reference_results_directory,
        pattern="detection_*.jsonl",
    )
    latest_reference: dict[str, RunResult] = {}
    for run in reference_runs:
        if run.effort != REFERENCE_EFFORT:
            continue
        existing = latest_reference.get(run.model)
        if existing is None or run.timestamp > existing.timestamp:
            latest_reference[run.model] = run

    for model, run in latest_reference.items():
        map_result = compute_dataset_map(run, sample_index)
        if map_result is None:
            continue
        rows.append(
            ReferenceReportRow(
                model=f"{model} (reference)",
                run_type="reference",
                map50=map_result.map50,
                map75=map_result.map75,
                map50_95=map_result.map50_95,
                image_count=map_result.image_count,
                map50_no_confidence=_strip_confidence_map(run, sample_index),
            )
        )

    return sorted(rows, key=lambda row: row.map50, reverse=True)


def format_reference_report(rows: list[ReferenceReportRow]) -> str:
    """Format comparison rows as a plain-text table.

    Args:
        rows: Report rows from :func:`build_reference_report_rows`.

    Returns:
        Multi-line table string.
    """
    lines = [
        f"{'Model':<34} {'Type':<10} {'mAP@50':>8} {'mAP@75':>8} "
        f"{'mAP@50:95':>10} {'Images':>7}"
    ]
    lines.append("-" * 82)
    for row in rows:
        lines.append(
            f"{row.model:<34} {row.run_type:<10} "
            f"{row.map50:>8.4f} {row.map75:>8.4f} {row.map50_95:>10.4f} "
            f"{row.image_count:>7}"
        )
        if row.map50_no_confidence is not None:
            lines.append(f"{'':34} {'no-conf':<10} {row.map50_no_confidence:>8.4f}")
    return "\n".join(lines)


@dataclass(frozen=True)
class PromptExperimentRow:
    """One row comparing a prompt-assisted reference run to baseline."""

    model: str
    prompt_asset_type: str
    prompt_set_version: str | None
    map50: float
    map50_delta: float
    recall_class_aware: float
    recall_class_agnostic: float
    timestamp: str


@dataclass(frozen=True)
class _IndexedReferenceRun:
    run: RunResult
    prompt_asset_type: str
    prompt_set_version: str | None


def _latest_runs_by_key(
    results_directory: Path,
) -> dict[tuple[str, str], _IndexedReferenceRun]:
    latest: dict[tuple[str, str], _IndexedReferenceRun] = {}
    for path in sorted(results_directory.glob("detection_*.jsonl")):
        run = load_results(path)
        if run.effort != REFERENCE_EFFORT:
            continue
        manifest_path = manifest_path_for_results(path)
        prompt_type = "none"
        prompt_version: str | None = None
        if manifest_path.exists():
            manifest = load_manifest(manifest_path)
            prompt_type = manifest.prompt_asset_type
            prompt_version = manifest.prompt_set_version
        key = (run.model, prompt_type)
        indexed = _IndexedReferenceRun(
            run=run,
            prompt_asset_type=prompt_type,
            prompt_set_version=prompt_version,
        )
        existing = latest.get(key)
        if existing is None or run.timestamp > existing.run.timestamp:
            latest[key] = indexed
    return latest


def build_prompt_experiment_rows(
    baseline_directory: Path,
    experiment_directory: Path,
    dataset_directory: str,
) -> list[PromptExperimentRow]:
    """Build rows comparing prompt experiments against baseline reference runs.

    Args:
        baseline_directory: Directory with baseline reference runs.
        experiment_directory: Directory with prompt-variant runs.
        dataset_directory: Detection dataset directory.

    Returns:
        Rows sorted by model then mAP@50 descending.
    """
    from vlm_exam.reference.analysis import build_reference_analysis_report

    task = DetectionTask()
    samples = task.load_samples(dataset_directory)
    sample_index = build_sample_index(samples)

    baseline_runs = _latest_runs_by_key(baseline_directory)
    baseline_map: dict[str, float] = {}
    for (model, prompt_type), indexed in baseline_runs.items():
        if prompt_type != "none":
            continue
        map_result = compute_dataset_map(indexed.run, sample_index)
        if map_result is not None:
            baseline_map[model] = map_result.map50

    experiment_runs = _latest_runs_by_key(experiment_directory)
    rows: list[PromptExperimentRow] = []
    for (model, prompt_type), indexed in experiment_runs.items():
        if prompt_type == "none":
            continue
        run = indexed.run
        map_result = compute_dataset_map(run, sample_index)
        if map_result is None:
            continue
        analysis = build_reference_analysis_report(run, sample_index)
        baseline = baseline_map.get(model, 0.0)
        rows.append(
            PromptExperimentRow(
                model=model,
                prompt_asset_type=prompt_type,
                prompt_set_version=indexed.prompt_set_version,
                map50=map_result.map50,
                map50_delta=map_result.map50 - baseline,
                recall_class_aware=analysis.recall_class_aware,
                recall_class_agnostic=analysis.recall_class_agnostic,
                timestamp=run.timestamp,
            )
        )
    return sorted(rows, key=lambda row: (row.model, -row.map50))


def format_prompt_experiment_report(rows: list[PromptExperimentRow]) -> str:
    """Format prompt experiment comparison rows."""
    lines = [
        f"{'Model':<18} {'Prompt':<18} {'Ver':<6} "
        f"{'mAP@50':>8} {'Delta':>8} {'Recall':>8} {'Rec-Agn':>8}"
    ]
    lines.append("-" * 82)
    for row in rows:
        lines.append(
            f"{row.model:<18} {row.prompt_asset_type:<18} "
            f"{(row.prompt_set_version or '-'):<6} "
            f"{row.map50:>8.4f} {row.map50_delta:>8.4f} "
            f"{row.recall_class_aware:>8.4f} {row.recall_class_agnostic:>8.4f}"
        )
    return "\n".join(lines)
