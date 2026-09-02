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

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from vlm_exam.config import BenchmarkConfig, ModelConfig, load_leaderboard_groups
from vlm_exam.protocol import PROTOCOL
from vlm_exam.results import RunResult, SampleResult, load_results_directory

BENCHMARK_TASK_NAMES: tuple[str, ...] = PROTOCOL.tasks
"""Registered benchmark tasks included in cross-task efficiency rollups."""

REPEATS_PER_CONFIGURATION = PROTOCOL.repeats
"""Number of full runs every committed (task, model, effort) should have."""

RunGroups = dict[tuple[str, str, str], list[RunResult]]
"""Runs keyed by ``(task, effort, model)``, oldest first."""


@dataclass(frozen=True)
class RepeatedMetric:
    """One metric measured on several repeats of the same configuration."""

    values: tuple[float, ...]

    @property
    def mean(self) -> float:
        """Arithmetic mean of the per-run values."""
        return sum(self.values) / len(self.values)

    @property
    def run_count(self) -> int:
        """Number of runs the metric was measured on."""
        return len(self.values)

    @property
    def minimum(self) -> float:
        """Lowest per-run value."""
        return min(self.values)

    @property
    def maximum(self) -> float:
        """Highest per-run value."""
        return max(self.values)

    @property
    def spread(self) -> float:
        """Difference between the highest and lowest per-run value."""
        return self.maximum - self.minimum


@dataclass(frozen=True)
class ModelEfficiency:
    """Efficiency metrics for one model, averaged over repeated runs.

    Per-sample averages pool every sample of every run. Totals are the
    mean cost and time of one complete pass over each benchmarked task,
    so a model with three repeats is not reported as three times more
    expensive than a model with one.
    """

    model: str
    task_count: int
    sample_count: int
    average_tokens: float
    average_cost: float
    average_time_seconds: float
    total_cost: float
    total_time_seconds: float


def run_accuracy(run: RunResult) -> float:
    """Compute the headline accuracy of a run from its ``correct`` column.

    For judge-scored tasks this equals :func:`run_judge_accuracy`.

    Args:
        run: A benchmark run loaded from disk.

    Returns:
        Accuracy in percent (0-100), or 0.0 for an empty run.
    """
    if not run.samples:
        return 0.0
    return sum(sample.correct for sample in run.samples) / len(run.samples) * 100


def _verdict_accuracy(run: RunResult, key: str) -> float:
    if not run.samples:
        return 0.0
    verdicts = [sample.metadata.get(key) for sample in run.samples]
    missing = sum(1 for verdict in verdicts if not isinstance(verdict, bool))
    if missing:
        raise ValueError(
            f"{run.task} run for {run.model} ({run.effort}) lacks {key!r} on "
            f"{missing} of {len(verdicts)} samples; backfill it with "
            "`vlm-exam rescore` before reporting."
        )
    return sum(1 for verdict in verdicts if verdict) / len(verdicts) * 100


def run_strict_accuracy(run: RunResult) -> float:
    """Compute the deterministic strict-match accuracy of a run.

    Args:
        run: A run whose samples carry ``strict_correct`` in metadata.

    Returns:
        Strict accuracy in percent (0-100), or 0.0 for an empty run.

    Raises:
        ValueError: If any sample lacks a strict verdict.
    """
    return _verdict_accuracy(run, "strict_correct")


def run_judge_accuracy(run: RunResult) -> float:
    """Compute the LLM judge accuracy of a run.

    Args:
        run: A run whose samples carry ``judge_correct`` in metadata.

    Returns:
        Judge accuracy in percent (0-100), or 0.0 for an empty run.

    Raises:
        ValueError: If any sample lacks a judge verdict.
    """
    return _verdict_accuracy(run, "judge_correct")


def run_mean_similarity(run: RunResult) -> float:
    """Compute the mean OCR similarity score across a run's samples.

    Args:
        run: An OCR benchmark run whose samples carry a ``score`` in
            metadata.

    Returns:
        Mean similarity in percent (0-100), or 0.0 for an empty run.
    """
    if not run.samples:
        return 0.0
    return (
        sum(sample.metadata.get("score", 0.0) for sample in run.samples)
        / len(run.samples)
        * 100
    )


def sample_cost(sample: SampleResult, pricing: ModelConfig) -> float:
    """Estimate the USD cost of a single sample from token usage.

    Args:
        sample: A sample result carrying input and output token counts.
        pricing: Model config supplying per-million-token pricing.

    Returns:
        Estimated cost in USD for the sample.
    """
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


def group_runs(
    runs: list[RunResult],
    config: BenchmarkConfig,
    *,
    models: set[str] | None = None,
    effort: str | None = None,
    tasks: tuple[str, ...] | None = None,
) -> RunGroups:
    """Group every run by configuration so repeats can be averaged.

    Every file in the results directory is one repeat of its
    ``(task, effort, model)`` configuration; nothing is deduplicated.

    Args:
        runs: All loaded run results.
        config: Benchmark config used to filter unknown models.
        models: Optional set of model keys to include.
        effort: Optional effort level to include.
        tasks: Optional task names to include.

    Returns:
        Mapping from ``(task, effort, model)`` to its runs, oldest first.
    """
    groups: RunGroups = {}
    for run in runs:
        if run.model not in config.models:
            continue
        if models is not None and run.model not in models:
            continue
        if effort is not None and run.effort != effort:
            continue
        if tasks is not None and run.task not in tasks:
            continue
        groups.setdefault((run.task, run.effort, run.model), []).append(run)
    for group in groups.values():
        group.sort(key=lambda run: run.timestamp)
    return groups


def aggregate_metric(
    runs: list[RunResult],
    metric: Callable[[RunResult], float | None],
) -> RepeatedMetric | None:
    """Measure a per-run metric on each repeat and collect the values.

    Args:
        runs: Repeats of one configuration.
        metric: Per-run scoring function; ``None`` skips that run.

    Returns:
        The collected values, or ``None`` when no run produced a value.
    """
    values = tuple(
        value for value in (metric(run) for run in runs) if value is not None
    )
    if not values:
        return None
    return RepeatedMetric(values=values)


def aggregate_efficiency_by_model(
    results_directory: Path,
    config: BenchmarkConfig,
    effort: str = "low",
    *,
    models: set[str] | None = None,
) -> list[ModelEfficiency]:
    """Average per-sample and per-run efficiency over every repeat.

    Args:
        results_directory: Directory containing result JSONL files.
        config: Benchmark config with per-model pricing.
        effort: Effort level to aggregate (default ``"low"``).
        models: Optional set of model keys to include.

    Returns:
        Efficiency rows sorted by model identifier.
    """
    runs = load_results_directory(results_directory)
    groups = group_runs(
        runs, config, models=models, effort=effort, tasks=BENCHMARK_TASK_NAMES
    )

    samples_by_model: dict[str, list[SampleResult]] = {}
    tasks_by_model: dict[str, set[str]] = {}
    run_sample_count_by_model: dict[str, float] = {}
    run_cost_by_model: dict[str, float] = {}
    run_time_by_model: dict[str, float] = {}
    for (task, _, model), group in groups.items():
        pricing = config.models[model]
        tasks_by_model.setdefault(model, set()).add(task)
        for run in group:
            samples_by_model.setdefault(model, []).extend(run.samples)
        run_sample_count_by_model[model] = run_sample_count_by_model.get(
            model, 0.0
        ) + _mean(len(run.samples) for run in group)
        run_cost_by_model[model] = run_cost_by_model.get(model, 0.0) + _mean(
            sum(sample_cost(sample, pricing) for sample in run.samples) for run in group
        )
        run_time_by_model[model] = run_time_by_model.get(model, 0.0) + _mean(
            _elapsed_total(run.samples) for run in group
        )

    rows: list[ModelEfficiency] = []
    for model in sorted(samples_by_model):
        samples = samples_by_model[model]
        pricing = config.models[model]
        if not samples:
            continue

        total_tokens = sum(
            sample.input_tokens + sample.output_tokens for sample in samples
        )
        pooled_cost = sum(sample_cost(sample, pricing) for sample in samples)
        timed = [
            sample.elapsed_seconds
            for sample in samples
            if sample.elapsed_seconds is not None
        ]

        rows.append(
            ModelEfficiency(
                model=model,
                task_count=len(tasks_by_model[model]),
                sample_count=round(run_sample_count_by_model[model]),
                average_tokens=total_tokens / len(samples),
                average_cost=pooled_cost / len(samples),
                average_time_seconds=sum(timed) / len(timed) if timed else 0.0,
                total_cost=run_cost_by_model[model],
                total_time_seconds=run_time_by_model[model],
            )
        )

    return rows


def _elapsed_total(samples: list[SampleResult]) -> float:
    return sum(
        sample.elapsed_seconds
        for sample in samples
        if sample.elapsed_seconds is not None
    )


def _mean(values: Iterable[float]) -> float:
    collected = list(values)
    return sum(collected) / len(collected) if collected else 0.0
