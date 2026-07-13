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
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vlm_exam.config import BenchmarkConfig, ModelConfig
from vlm_exam.metrics import (
    BENCHMARK_TASK_NAMES,
    build_latest_runs_index,
    run_accuracy,
    run_mean_similarity,
    sample_cost,
)
from vlm_exam.results import (
    RunResult,
    SampleResult,
    is_failed_sample,
    load_results_directory,
)

if TYPE_CHECKING:
    from vlm_exam.tasks.detection import DetectionSample

_EFFORT_ORDER = {"low": 0, "medium": 1, "high": 2}


@dataclass(frozen=True)
class MetricDefinition:
    """Metadata describing one reported metric for a task."""

    key: str
    label: str
    unit: str = "percent"


@dataclass(frozen=True)
class _TaskDefinition:
    name: str
    primary_metric: str
    metrics: tuple[MetricDefinition, ...]


_TASK_DEFINITIONS: dict[str, _TaskDefinition] = {
    "ocr": _TaskDefinition(
        name="OCR",
        primary_metric="similarity",
        metrics=(MetricDefinition("similarity", "Mean Similarity"),),
    ),
    "extraction": _TaskDefinition(
        name="Data Extraction",
        primary_metric="accuracy",
        metrics=(MetricDefinition("accuracy", "Accuracy"),),
    ),
    "counting": _TaskDefinition(
        name="Counting",
        primary_metric="accuracy",
        metrics=(MetricDefinition("accuracy", "Accuracy"),),
    ),
    "identification": _TaskDefinition(
        name="Identification",
        primary_metric="accuracy",
        metrics=(MetricDefinition("accuracy", "Accuracy"),),
    ),
    "reasoning": _TaskDefinition(
        name="Reasoning",
        primary_metric="accuracy",
        metrics=(MetricDefinition("accuracy", "Accuracy"),),
    ),
    "detection": _TaskDefinition(
        name="Detection",
        primary_metric="map50",
        metrics=(
            MetricDefinition("map50", "mAP@50"),
            MetricDefinition("map75", "mAP@75"),
            MetricDefinition("map50_95", "mAP@50:95"),
        ),
    ),
}

_UNREGISTERED_TASKS = set(BENCHMARK_TASK_NAMES) - set(_TASK_DEFINITIONS)
if _UNREGISTERED_TASKS:
    raise RuntimeError(
        f"Tasks missing from _TASK_DEFINITIONS: {sorted(_UNREGISTERED_TASKS)}"
    )


@dataclass(frozen=True)
class TokenSummary:
    """Aggregated token usage for a set of samples."""

    input: int
    output: int
    total: int
    average_per_sample: float


@dataclass(frozen=True)
class CostSummary:
    """Aggregated estimated USD cost for a set of samples."""

    total_usd: float
    average_per_sample_usd: float


@dataclass(frozen=True)
class SpeedSummary:
    """Aggregated inference wall-clock time for a set of samples."""

    total_seconds: float
    average_seconds_per_sample: float


@dataclass(frozen=True)
class MetricValue:
    """A named metric value in percent (0-100)."""

    name: str
    value: float


@dataclass(frozen=True)
class ModelTaskResult:
    """One model's result on one task, with quality and efficiency."""

    primary_metric: MetricValue | None
    metrics: dict[str, float]
    sample_count: int
    failed_sample_count: int
    tokens: TokenSummary
    cost: CostSummary
    speed: SpeedSummary
    timestamp: str
    evaluated_sample_count: int | None = None


@dataclass(frozen=True)
class ModelOverall:
    """A model's efficiency pooled across all its benchmarked tasks."""

    task_count: int
    sample_count: int
    tokens: TokenSummary
    cost: CostSummary
    speed: SpeedSummary


@dataclass(frozen=True)
class ModelSummary:
    """A single model's complete summary at one effort level."""

    id: str
    key: str
    name: str
    lab: str
    effort: str
    tasks: dict[str, ModelTaskResult]
    overall: ModelOverall


@dataclass(frozen=True)
class TaskSummary:
    """Metadata describing one benchmark task in the summary."""

    key: str
    name: str
    primary_metric: str
    metrics: tuple[MetricDefinition, ...]


@dataclass(frozen=True)
class BenchmarkSummary:
    """Frontend-facing rollup of all benchmark results."""

    generated_at: str | None
    efforts: tuple[str, ...]
    tasks: list[TaskSummary]
    models: list[ModelSummary]


def _iso_timestamp(raw: str) -> str:
    return datetime.strptime(raw, "%Y%m%d_%H%M%S").strftime("%Y-%m-%dT%H:%M:%SZ")


def _effort_sort_key(effort: str) -> tuple[int, str]:
    return (_EFFORT_ORDER.get(effort, len(_EFFORT_ORDER)), effort)


def _token_summary(samples: list[SampleResult]) -> TokenSummary:
    total_input = sum(sample.input_tokens for sample in samples)
    total_output = sum(sample.output_tokens for sample in samples)
    total = total_input + total_output
    count = len(samples)
    return TokenSummary(
        input=total_input,
        output=total_output,
        total=total,
        average_per_sample=total / count if count else 0.0,
    )


def _cost_summary(samples: list[SampleResult], pricing: ModelConfig) -> CostSummary:
    total = sum(sample_cost(sample, pricing) for sample in samples)
    count = len(samples)
    return CostSummary(
        total_usd=total,
        average_per_sample_usd=total / count if count else 0.0,
    )


def _speed_summary(samples: list[SampleResult]) -> SpeedSummary:
    timed = [
        sample.elapsed_seconds
        for sample in samples
        if sample.elapsed_seconds is not None
    ]
    total = sum(timed)
    return SpeedSummary(
        total_seconds=total,
        average_seconds_per_sample=total / len(timed) if timed else 0.0,
    )


def _detection_quality(
    run: RunResult,
    detection_index: dict[str, DetectionSample] | None,
) -> tuple[dict[str, float], int | None]:
    if detection_index is None:
        return {}, None

    from vlm_exam.tasks.detection import compute_dataset_map

    map_result = compute_dataset_map(run, detection_index)
    if map_result is None:
        print(
            f"Warning: no detection predictions matched the dataset for "
            f"{run.model} ({run.effort}); mAP omitted."
        )
        return {}, None
    if map_result.image_count != len(run.samples):
        print(
            f"Warning: detection mAP for {run.model} ({run.effort}) covers "
            f"{map_result.image_count} of {len(run.samples)} samples; "
            f"check that the dataset directory matches the benchmarked data."
        )
    metrics = {
        "map50": map_result.map50 * 100,
        "map75": map_result.map75 * 100,
        "map50_95": map_result.map50_95 * 100,
    }
    return metrics, map_result.image_count


def _quality_metrics(
    run: RunResult,
    detection_index: dict[str, DetectionSample] | None,
) -> tuple[dict[str, float], int | None]:
    if not run.samples:
        return {}, None
    if run.task == "detection":
        return _detection_quality(run, detection_index)
    if run.task == "ocr":
        return {"similarity": run_mean_similarity(run)}, None
    return {"accuracy": run_accuracy(run)}, None


def _model_task_result(
    run: RunResult,
    pricing: ModelConfig,
    detection_index: dict[str, DetectionSample] | None,
) -> ModelTaskResult:
    metrics, evaluated_sample_count = _quality_metrics(run, detection_index)
    primary_name = _TASK_DEFINITIONS[run.task].primary_metric
    primary = (
        MetricValue(name=primary_name, value=metrics[primary_name])
        if primary_name in metrics
        else None
    )
    return ModelTaskResult(
        primary_metric=primary,
        metrics=metrics,
        sample_count=len(run.samples),
        failed_sample_count=sum(
            1 for sample in run.samples if is_failed_sample(sample)
        ),
        tokens=_token_summary(run.samples),
        cost=_cost_summary(run.samples, pricing),
        speed=_speed_summary(run.samples),
        timestamp=run.timestamp,
        evaluated_sample_count=evaluated_sample_count,
    )


def _load_detection_index(
    dataset_directory: Path,
) -> dict[str, DetectionSample]:
    from vlm_exam.tasks.detection import DetectionTask, build_sample_index

    task = DetectionTask()
    samples = task.load_samples(str(dataset_directory))
    return build_sample_index(samples)


def build_summary(
    results_directory: Path,
    config: BenchmarkConfig,
    effort: str | None = None,
    *,
    models: set[str] | None = None,
    detection_dataset_directory: Path | None = None,
) -> BenchmarkSummary:
    """Compile all result files into a single frontend-facing summary.

    Only the newest run per (task, effort, model) is included. Runs for
    tasks outside the registered benchmark tasks are skipped with a
    warning.

    Args:
        results_directory: Directory containing result JSONL files.
        config: Benchmark config supplying model names and pricing.
        effort: Effort level to include. When ``None``, every effort is
            compiled and the same model appears once per effort.
        models: Optional set of model keys to include.
        detection_dataset_directory: Detection dataset directory used to
            compute mAP. When ``None``, detection quality metrics are
            omitted while token, cost, and speed metrics are kept.

    Returns:
        The assembled benchmark summary.
    """
    runs = load_results_directory(results_directory)
    latest = build_latest_runs_index(runs, config, models=models)

    detection_index: dict[str, DetectionSample] | None = None
    if detection_dataset_directory is not None and any(
        task == "detection" and (effort is None or run_effort == effort)
        for task, run_effort, _ in latest
    ):
        detection_index = _load_detection_index(detection_dataset_directory)

    runs_by_model_effort: dict[tuple[str, str], dict[str, RunResult]] = {}
    skipped_tasks: set[str] = set()
    for (task, run_effort, model), run in latest.items():
        if task not in BENCHMARK_TASK_NAMES:
            skipped_tasks.add(task)
            continue
        if effort is not None and run_effort != effort:
            continue
        runs_by_model_effort.setdefault((model, run_effort), {})[task] = run

    if skipped_tasks:
        print(
            f"Warning: skipping runs for unregistered task(s): "
            f"{', '.join(sorted(skipped_tasks))}"
        )

    efforts_by_model: dict[str, list[str]] = {}
    for model, run_effort in runs_by_model_effort:
        efforts_by_model.setdefault(model, []).append(run_effort)

    included_tasks: set[str] = set()
    model_summaries: list[ModelSummary] = []
    latest_run_timestamp = ""

    for model_key, model_config in config.models.items():
        for run_effort in sorted(
            efforts_by_model.get(model_key, []), key=_effort_sort_key
        ):
            task_runs = runs_by_model_effort[(model_key, run_effort)]

            ordered_tasks: dict[str, ModelTaskResult] = {}
            pooled_samples: list[SampleResult] = []
            for task in BENCHMARK_TASK_NAMES:
                run = task_runs.get(task)
                if run is None:
                    continue
                ordered_tasks[task] = _model_task_result(
                    run, model_config, detection_index
                )
                pooled_samples.extend(run.samples)
                included_tasks.add(task)
                latest_run_timestamp = max(latest_run_timestamp, run.timestamp)

            if not ordered_tasks:
                continue

            overall = ModelOverall(
                task_count=len(ordered_tasks),
                sample_count=len(pooled_samples),
                tokens=_token_summary(pooled_samples),
                cost=_cost_summary(pooled_samples, model_config),
                speed=_speed_summary(pooled_samples),
            )
            model_summaries.append(
                ModelSummary(
                    id=f"{model_key}:{run_effort}",
                    key=model_key,
                    name=model_config.name,
                    lab=model_config.lab,
                    effort=run_effort,
                    tasks=ordered_tasks,
                    overall=overall,
                )
            )

    task_summaries: list[TaskSummary] = []
    for task in BENCHMARK_TASK_NAMES:
        if task not in included_tasks:
            continue
        definition = _TASK_DEFINITIONS[task]
        task_summaries.append(
            TaskSummary(
                key=task,
                name=definition.name,
                primary_metric=definition.primary_metric,
                metrics=definition.metrics,
            )
        )

    efforts = tuple(
        sorted(
            {model.effort for model in model_summaries},
            key=_effort_sort_key,
        )
    )

    return BenchmarkSummary(
        generated_at=(
            _iso_timestamp(latest_run_timestamp) if latest_run_timestamp else None
        ),
        efforts=efforts,
        tasks=task_summaries,
        models=model_summaries,
    )


def _round_percent(value: float) -> float:
    return round(value, 2)


def _token_dict(tokens: TokenSummary) -> dict[str, Any]:
    return {
        "input": tokens.input,
        "output": tokens.output,
        "total": tokens.total,
        "average_per_sample": round(tokens.average_per_sample, 1),
    }


def _cost_dict(cost: CostSummary) -> dict[str, Any]:
    return {
        "total_usd": round(cost.total_usd, 6),
        "average_per_sample_usd": round(cost.average_per_sample_usd, 6),
    }


def _speed_dict(speed: SpeedSummary) -> dict[str, Any]:
    return {
        "total_seconds": round(speed.total_seconds, 3),
        "average_seconds_per_sample": round(speed.average_seconds_per_sample, 3),
    }


def _task_result_dict(result: ModelTaskResult) -> dict[str, Any]:
    primary = (
        {
            "name": result.primary_metric.name,
            "value": _round_percent(result.primary_metric.value),
        }
        if result.primary_metric is not None
        else None
    )
    return {
        "primary_metric": primary,
        "metrics": {
            name: _round_percent(value) for name, value in result.metrics.items()
        },
        "sample_count": result.sample_count,
        "evaluated_sample_count": result.evaluated_sample_count,
        "failed_sample_count": result.failed_sample_count,
        "tokens": _token_dict(result.tokens),
        "cost": _cost_dict(result.cost),
        "speed": _speed_dict(result.speed),
        "timestamp": _iso_timestamp(result.timestamp),
    }


def summary_to_dict(summary: BenchmarkSummary) -> dict[str, Any]:
    """Serialize a benchmark summary into a JSON-ready dictionary.

    Args:
        summary: The benchmark summary to serialize.

    Returns:
        A dictionary suitable for :func:`json.dump`.
    """
    return {
        "generated_at": summary.generated_at,
        "efforts": list(summary.efforts),
        "tasks": [
            {
                "key": task.key,
                "name": task.name,
                "primary_metric": task.primary_metric,
                "metrics": [
                    {
                        "key": metric.key,
                        "label": metric.label,
                        "unit": metric.unit,
                    }
                    for metric in task.metrics
                ],
            }
            for task in summary.tasks
        ],
        "models": [
            {
                "id": model.id,
                "key": model.key,
                "name": model.name,
                "lab": model.lab,
                "effort": model.effort,
                "tasks": {
                    task: _task_result_dict(result)
                    for task, result in model.tasks.items()
                },
                "overall": {
                    "task_count": model.overall.task_count,
                    "sample_count": model.overall.sample_count,
                    "tokens": _token_dict(model.overall.tokens),
                    "cost": _cost_dict(model.overall.cost),
                    "speed": _speed_dict(model.overall.speed),
                },
            }
            for model in summary.models
        ],
    }
