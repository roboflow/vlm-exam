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

from vlm_exam.config import BenchmarkConfig, ModelConfig, load_leaderboard_groups
from vlm_exam.results import RunResult, SampleResult, load_results_directory
from vlm_exam.tasks import QA_TASK_NAMES

BENCHMARK_TASK_NAMES: tuple[str, ...] = (*QA_TASK_NAMES, "detection")
"""Registered benchmark tasks included in cross-task efficiency rollups."""


@dataclass(frozen=True)
class ModelEfficiency:
    """Pooled efficiency metrics for one model across benchmark tasks."""

    model: str
    task_count: int
    sample_count: int
    average_tokens: float
    average_cost: float
    average_time_seconds: float
    total_cost: float
    total_time_seconds: float


def _sample_cost(sample: SampleResult, pricing: ModelConfig) -> float:
    return (
        sample.input_tokens / 1_000_000
    ) * pricing.pricing.input_per_million_tokens + (
        sample.output_tokens / 1_000_000
    ) * pricing.pricing.output_per_million_tokens


def parse_model_filter(models: str, config: BenchmarkConfig) -> list[str]:
    """Parse and validate a comma-separated model filter string.

    Args:
        models: Comma-separated model identifiers.
        config: Benchmark config used to validate model keys.

    Returns:
        Ordered list of validated model identifiers.

    Raises:
        ValueError: If the string is empty or contains unknown model keys.
    """
    model_ids = [model_id.strip() for model_id in models.split(",") if model_id.strip()]
    if not model_ids:
        raise ValueError("--models must list at least one model.")
    unknown = [model_id for model_id in model_ids if model_id not in config.models]
    if unknown:
        raise ValueError(f"Unknown model(s): {', '.join(unknown)}")
    return model_ids


def resolve_leaderboard_model_list(
    config: BenchmarkConfig,
    *,
    models: str | None = None,
    group: str | None = None,
) -> list[str] | None:
    """Resolve an ordered model list from ``--models`` or ``--group``.

    When both are given, ``group`` takes precedence.

    Args:
        config: Benchmark config used to validate model keys.
        models: Optional comma-separated model identifiers.
        group: Optional named leaderboard group from ``leaderboard_groups.yaml``.

    Returns:
        Ordered model identifiers, or ``None`` when no filter was requested.

    Raises:
        ValueError: If the group or model list is invalid.
    """
    if group is not None:
        groups = load_leaderboard_groups()
        if group not in groups:
            known = ", ".join(sorted(groups))
            raise ValueError(
                f"Unknown leaderboard group {group!r}. Known groups: {known}"
            )
        model_ids = list(groups[group])
        unknown = [model_id for model_id in model_ids if model_id not in config.models]
        if unknown:
            raise ValueError(
                f"Group {group!r} references unknown model(s): {', '.join(unknown)}"
            )
        return model_ids
    if models is not None:
        return parse_model_filter(models, config)
    return None


def build_latest_runs_index(
    runs: list[RunResult],
    config: BenchmarkConfig,
    *,
    models: set[str] | None = None,
) -> dict[tuple[str, str, str], RunResult]:
    """Keep the newest run per task, effort, and model.

    Args:
        runs: All loaded run results.
        config: Benchmark config used to filter unknown models.
        models: Optional set of model keys to include.

    Returns:
        Mapping from ``(task, effort, model)`` to the latest matching run.
    """
    latest: dict[tuple[str, str, str], RunResult] = {}
    for run_result in runs:
        if run_result.model not in config.models:
            continue
        if models is not None and run_result.model not in models:
            continue
        key = (run_result.task, run_result.effort, run_result.model)
        existing = latest.get(key)
        if existing is None or run_result.timestamp > existing.timestamp:
            latest[key] = run_result
    return latest


def latest_runs_by_task_model(
    runs: list[RunResult],
    config: BenchmarkConfig,
    effort: str,
    *,
    models: set[str] | None = None,
) -> dict[tuple[str, str], RunResult]:
    """Keep the newest run per registered task and model at a given effort.

    Args:
        runs: All loaded run results.
        config: Benchmark config used to filter unknown models.
        effort: Effort level to include (e.g. ``"low"``).
        models: Optional set of model keys to include.

    Returns:
        Mapping from ``(task, model)`` to the latest matching run.
    """
    latest: dict[tuple[str, str], RunResult] = {}
    for run in runs:
        if run.effort != effort:
            continue
        if run.task not in BENCHMARK_TASK_NAMES:
            continue
        if run.model not in config.models:
            continue
        if models is not None and run.model not in models:
            continue
        key = (run.task, run.model)
        existing = latest.get(key)
        if existing is None or run.timestamp > existing.timestamp:
            latest[key] = run
    return latest


def aggregate_efficiency_by_model(
    results_directory: Path,
    config: BenchmarkConfig,
    effort: str = "low",
    *,
    models: set[str] | None = None,
) -> list[ModelEfficiency]:
    """Pool per-sample metrics across latest benchmark runs for each model.

    Args:
        results_directory: Directory containing result JSONL files.
        config: Benchmark config with per-model pricing.
        effort: Effort level to aggregate (default ``"low"``).
        models: Optional set of model keys to include.

    Returns:
        Efficiency rows sorted by model identifier.
    """
    runs = load_results_directory(results_directory)
    latest = latest_runs_by_task_model(runs, config, effort, models=models)

    samples_by_model: dict[str, list[SampleResult]] = {}
    tasks_by_model: dict[str, set[str]] = {}
    for (task, model), run in latest.items():
        samples_by_model.setdefault(model, []).extend(run.samples)
        tasks_by_model.setdefault(model, set()).add(task)

    rows: list[ModelEfficiency] = []
    for model in sorted(samples_by_model):
        samples = samples_by_model[model]
        pricing = config.models[model]
        sample_count = len(samples)
        if sample_count == 0:
            continue

        total_input = sum(sample.input_tokens for sample in samples)
        total_output = sum(sample.output_tokens for sample in samples)
        total_cost = sum(_sample_cost(sample, pricing) for sample in samples)
        timed = [
            sample.elapsed_seconds
            for sample in samples
            if sample.elapsed_seconds is not None
        ]
        total_time = sum(timed)

        rows.append(
            ModelEfficiency(
                model=model,
                task_count=len(tasks_by_model[model]),
                sample_count=sample_count,
                average_tokens=(total_input + total_output) / sample_count,
                average_cost=total_cost / sample_count,
                average_time_seconds=total_time / len(timed) if timed else 0.0,
                total_cost=total_cost,
                total_time_seconds=total_time,
            )
        )

    return rows
