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

import difflib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vlm_exam.config import BenchmarkConfig, ModelConfig
from vlm_exam.judge import DEFAULT_JUDGE_MODEL
from vlm_exam.metrics import (
    BENCHMARK_TASK_NAMES,
    RepeatedMetric,
    RunGroups,
    group_runs,
    run_judge_accuracy,
    run_mean_similarity,
    run_strict_accuracy,
    sample_cost,
)
from vlm_exam.protocol import PROTOCOL
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


_JUDGE_ACCURACY_METRIC = "accuracy_judge"
_STRICT_ACCURACY_METRIC = "accuracy_strict"

_QA_ACCURACY_METRICS = (
    MetricDefinition(_JUDGE_ACCURACY_METRIC, "Accuracy (LLM judge)"),
    MetricDefinition(_STRICT_ACCURACY_METRIC, "Accuracy (strict match)"),
)

_TASK_DEFINITIONS: dict[str, _TaskDefinition] = {
    "ocr": _TaskDefinition(
        name="OCR",
        primary_metric="similarity",
        metrics=(MetricDefinition("similarity", "Mean Similarity"),),
    ),
    "extraction": _TaskDefinition(
        name="Data Extraction",
        primary_metric=_JUDGE_ACCURACY_METRIC,
        metrics=_QA_ACCURACY_METRICS,
    ),
    "counting": _TaskDefinition(
        name="Counting",
        primary_metric=_JUDGE_ACCURACY_METRIC,
        metrics=_QA_ACCURACY_METRICS,
    ),
    "identification": _TaskDefinition(
        name="Identification",
        primary_metric=_JUDGE_ACCURACY_METRIC,
        metrics=_QA_ACCURACY_METRICS,
    ),
    "reasoning": _TaskDefinition(
        name="Reasoning",
        primary_metric=_JUDGE_ACCURACY_METRIC,
        metrics=_QA_ACCURACY_METRICS,
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
    """One model's result on one task, averaged over its repeated runs.

    ``metrics`` holds the mean of each quality metric across runs and
    ``metric_runs`` the per-run values behind it. Token, cost, and speed
    figures describe one complete run (the mean over repeats); the failed
    sample count is summed over every run.
    """

    primary_metric: MetricValue | None
    metrics: dict[str, float]
    metric_runs: dict[str, tuple[float, ...]]
    run_count: int
    sample_count: int
    failed_sample_count: int
    tokens: TokenSummary
    cost: CostSummary
    speed: SpeedSummary
    timestamp: str
    timestamps: tuple[str, ...]
    evaluated_sample_count: int | None = None


@dataclass(frozen=True)
class ModelOverall:
    """A model's per-run efficiency summed across all its benchmarked tasks."""

    task_count: int
    sample_count: int
    tokens: TokenSummary
    cost: CostSummary
    speed: SpeedSummary


PROTOCOL_COMPLETE = "complete"
"""Every required configuration has exactly the required repeats."""

PROTOCOL_INCOMPLETE = "incomplete"
"""A full-protocol model still missing runs; CI fails until fixed."""

PROTOCOL_LEGACY = "legacy"
"""A pre-protocol model with gaps that are reported but not enforced."""


@dataclass(frozen=True)
class ModelProtocolSummary:
    """How a model stands against the benchmark protocol, across all efforts."""

    name: str
    status: str
    runs_present: int
    runs_required: int


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
    protocol: ModelProtocolSummary


@dataclass(frozen=True)
class TaskSummary:
    """Metadata describing one benchmark task in the summary."""

    key: str
    name: str
    primary_metric: str
    metrics: tuple[MetricDefinition, ...]


@dataclass(frozen=True)
class ScoringSummary:
    """How the QA tasks' two accuracy metrics are produced."""

    judge_model: str
    judge_metric: str
    strict_metric: str


@dataclass(frozen=True)
class ProtocolSummary:
    """What every fully benchmarked model is expected to have."""

    repeats: int
    efforts: tuple[str, ...]
    tasks: tuple[str, ...]

    @property
    def runs_per_model(self) -> int:
        """Total result files a complete model has."""
        return self.repeats * len(self.efforts) * len(self.tasks)


@dataclass(frozen=True)
class BenchmarkSummary:
    """Frontend-facing rollup of all benchmark results."""

    generated_at: str | None
    efforts: tuple[str, ...]
    tasks: list[TaskSummary]
    models: list[ModelSummary]
    scoring: ScoringSummary = ScoringSummary(
        judge_model=DEFAULT_JUDGE_MODEL,
        judge_metric=_JUDGE_ACCURACY_METRIC,
        strict_metric=_STRICT_ACCURACY_METRIC,
    )
    protocol: ProtocolSummary = ProtocolSummary(
        repeats=PROTOCOL.repeats,
        efforts=PROTOCOL.efforts,
        tasks=PROTOCOL.tasks,
    )


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
    return {
        _JUDGE_ACCURACY_METRIC: run_judge_accuracy(run),
        _STRICT_ACCURACY_METRIC: run_strict_accuracy(run),
    }, None


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _mean_tokens(runs: list[RunResult]) -> TokenSummary:
    summaries = [_token_summary(run.samples) for run in runs]
    return TokenSummary(
        input=round(_mean([summary.input for summary in summaries])),
        output=round(_mean([summary.output for summary in summaries])),
        total=round(_mean([summary.total for summary in summaries])),
        average_per_sample=_mean([summary.average_per_sample for summary in summaries]),
    )


def _mean_cost(runs: list[RunResult], pricing: ModelConfig) -> CostSummary:
    summaries = [_cost_summary(run.samples, pricing) for run in runs]
    return CostSummary(
        total_usd=_mean([summary.total_usd for summary in summaries]),
        average_per_sample_usd=_mean(
            [summary.average_per_sample_usd for summary in summaries]
        ),
    )


def _mean_speed(runs: list[RunResult]) -> SpeedSummary:
    summaries = [_speed_summary(run.samples) for run in runs]
    return SpeedSummary(
        total_seconds=_mean([summary.total_seconds for summary in summaries]),
        average_seconds_per_sample=_mean(
            [summary.average_seconds_per_sample for summary in summaries]
        ),
    )


def _repeated_quality(
    runs: list[RunResult],
    detection_index: dict[str, DetectionSample] | None,
) -> tuple[dict[str, RepeatedMetric], int | None]:
    values_by_metric: dict[str, list[float]] = {}
    evaluated_counts: list[int] = []
    for run in runs:
        metrics, evaluated = _quality_metrics(run, detection_index)
        for name, value in metrics.items():
            values_by_metric.setdefault(name, []).append(value)
        if evaluated is not None:
            evaluated_counts.append(evaluated)
    repeated = {
        name: RepeatedMetric(values=tuple(values))
        for name, values in values_by_metric.items()
    }
    return repeated, min(evaluated_counts) if evaluated_counts else None


def _model_task_result(
    runs: list[RunResult],
    pricing: ModelConfig,
    detection_index: dict[str, DetectionSample] | None,
) -> ModelTaskResult:
    task = runs[0].task
    repeated, evaluated_sample_count = _repeated_quality(runs, detection_index)
    primary_name = _TASK_DEFINITIONS[task].primary_metric
    primary = (
        MetricValue(name=primary_name, value=repeated[primary_name].mean)
        if primary_name in repeated
        else None
    )
    return ModelTaskResult(
        primary_metric=primary,
        metrics={name: metric.mean for name, metric in repeated.items()},
        metric_runs={name: metric.values for name, metric in repeated.items()},
        run_count=len(runs),
        sample_count=max(len(run.samples) for run in runs),
        failed_sample_count=sum(
            1 for run in runs for sample in run.samples if is_failed_sample(sample)
        ),
        tokens=_mean_tokens(runs),
        cost=_mean_cost(runs, pricing),
        speed=_mean_speed(runs),
        timestamp=runs[-1].timestamp,
        timestamps=tuple(run.timestamp for run in runs),
        evaluated_sample_count=evaluated_sample_count,
    )


def _overall(task_results: dict[str, ModelTaskResult]) -> ModelOverall:
    results = list(task_results.values())
    sample_count = sum(result.sample_count for result in results)
    total_input = sum(result.tokens.input for result in results)
    total_output = sum(result.tokens.output for result in results)
    total_cost = sum(result.cost.total_usd for result in results)
    total_seconds = sum(result.speed.total_seconds for result in results)
    return ModelOverall(
        task_count=len(results),
        sample_count=sample_count,
        tokens=TokenSummary(
            input=total_input,
            output=total_output,
            total=total_input + total_output,
            average_per_sample=(
                (total_input + total_output) / sample_count if sample_count else 0.0
            ),
        ),
        cost=CostSummary(
            total_usd=total_cost,
            average_per_sample_usd=total_cost / sample_count if sample_count else 0.0,
        ),
        speed=SpeedSummary(
            total_seconds=total_seconds,
            average_seconds_per_sample=(
                total_seconds / sample_count if sample_count else 0.0
            ),
        ),
    )


def _model_protocol(
    model_key: str,
    model_config: ModelConfig,
    all_groups: RunGroups,
) -> ModelProtocolSummary:
    counts = [
        len(all_groups.get((task, effort, model_key), ()))
        for task, effort in PROTOCOL.configurations
    ]
    complete = all(count == PROTOCOL.repeats for count in counts)
    if complete:
        status = PROTOCOL_COMPLETE
    elif model_config.is_legacy:
        status = PROTOCOL_LEGACY
    else:
        status = PROTOCOL_INCOMPLETE
    return ModelProtocolSummary(
        name=model_config.benchmark_protocol,
        status=status,
        runs_present=sum(counts),
        runs_required=PROTOCOL.required_runs,
    )


def _warn_on_failed_samples(runs: list[RunResult]) -> None:
    for run in runs:
        failed = sum(1 for sample in run.samples if is_failed_sample(sample))
        if failed:
            print(
                f"Warning: {run.task} run for {run.model} ({run.effort}, "
                f"{run.timestamp}) has {failed} failed samples; resume it with "
                "`vlm-exam run --resume-file` so the average is not dragged down."
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

    Every run per (task, effort, model) is treated as one repeat and the
    reported metrics are means across repeats. Runs for tasks outside the
    registered benchmark tasks are skipped with a warning.

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
    all_groups = group_runs(runs, config, models=models)
    groups = (
        all_groups
        if effort is None
        else {key: group for key, group in all_groups.items() if key[1] == effort}
    )

    detection_index: dict[str, DetectionSample] | None = None
    if detection_dataset_directory is not None and any(
        task == "detection" for task, _, _ in groups
    ):
        detection_index = _load_detection_index(detection_dataset_directory)

    runs_by_model_effort: dict[tuple[str, str], dict[str, list[RunResult]]] = {}
    skipped_tasks: set[str] = set()
    for (task, run_effort, model), group in groups.items():
        if task not in BENCHMARK_TASK_NAMES:
            skipped_tasks.add(task)
            continue
        _warn_on_failed_samples(group)
        runs_by_model_effort.setdefault((model, run_effort), {})[task] = group

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
            for task in BENCHMARK_TASK_NAMES:
                group = task_runs.get(task)
                if group is None:
                    continue
                ordered_tasks[task] = _model_task_result(
                    group, model_config, detection_index
                )
                included_tasks.add(task)
                latest_run_timestamp = max(
                    latest_run_timestamp, ordered_tasks[task].timestamp
                )

            if not ordered_tasks:
                continue

            model_summaries.append(
                ModelSummary(
                    id=f"{model_key}:{run_effort}",
                    key=model_key,
                    name=model_config.name,
                    lab=model_config.lab,
                    effort=run_effort,
                    tasks=ordered_tasks,
                    overall=_overall(ordered_tasks),
                    protocol=_model_protocol(model_key, model_config, all_groups),
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
        "metric_runs": {
            name: [_round_percent(value) for value in values]
            for name, values in result.metric_runs.items()
        },
        "run_count": result.run_count,
        "timestamps": [_iso_timestamp(stamp) for stamp in result.timestamps],
        "sample_count": result.sample_count,
        "evaluated_sample_count": result.evaluated_sample_count,
        "failed_sample_count": result.failed_sample_count,
        "tokens": _token_dict(result.tokens),
        "cost": _cost_dict(result.cost),
        "speed": _speed_dict(result.speed),
        "timestamp": _iso_timestamp(result.timestamp),
    }


_DATASET_DEPENDENT_KEYS = ("primary_metric", "metrics", "metric_runs")


def _without_detection_quality(payload: dict[str, Any]) -> dict[str, Any]:
    stripped = json.loads(json.dumps(payload))
    for model in stripped.get("models", []):
        detection = model.get("tasks", {}).get("detection")
        if detection is None:
            continue
        for key in _DATASET_DEPENDENT_KEYS:
            detection.pop(key, None)
        detection.pop("evaluated_sample_count", None)
    return stripped


def summary_drift(
    committed: dict[str, Any],
    fresh: dict[str, Any],
    *,
    ignore_detection_quality: bool,
) -> list[str]:
    """Diff a committed summary payload against a freshly built one.

    Args:
        committed: Payload loaded from ``web/benchmark_summary.json``.
        fresh: Payload just produced by :func:`summary_to_dict`.
        ignore_detection_quality: Drop detection mAP fields from both sides,
            for environments without the detection dataset.

    Returns:
        Unified diff lines; empty when the payloads agree.
    """
    if ignore_detection_quality:
        committed = _without_detection_quality(committed)
        fresh = _without_detection_quality(fresh)
    committed_text = json.dumps(committed, indent=2, sort_keys=True).splitlines()
    fresh_text = json.dumps(fresh, indent=2, sort_keys=True).splitlines()
    return list(
        difflib.unified_diff(
            committed_text,
            fresh_text,
            fromfile="committed",
            tofile="regenerated",
            lineterm="",
        )
    )


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
        "scoring": {
            "judge_model": summary.scoring.judge_model,
            "judge_metric": summary.scoring.judge_metric,
            "strict_metric": summary.scoring.strict_metric,
        },
        "protocol": {
            "repeats": summary.protocol.repeats,
            "efforts": list(summary.protocol.efforts),
            "tasks": list(summary.protocol.tasks),
            "runs_per_model": summary.protocol.runs_per_model,
        },
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
                "protocol": {
                    "name": model.protocol.name,
                    "status": model.protocol.status,
                    "runs_present": model.protocol.runs_present,
                    "runs_required": model.protocol.runs_required,
                },
            }
            for model in summary.models
        ],
    }
