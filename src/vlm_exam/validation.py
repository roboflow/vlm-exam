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

import shlex
from dataclasses import dataclass
from pathlib import Path

from vlm_exam.config import BenchmarkConfig
from vlm_exam.protocol import PROTOCOL, BenchmarkProtocol, run_command
from vlm_exam.rescore import JUDGE_TASK_NAMES
from vlm_exam.results import RunResult, is_failed_sample, load_results

ERROR = "error"
"""Severity of a problem that fails validation."""

WARNING = "warning"
"""Severity of a problem that is reported but does not fail validation."""

STATUS_COMPLETE = "complete"
"""Every required configuration has exactly the required number of runs."""

STATUS_INCOMPLETE = "incomplete"
"""A full-protocol model with at least one missing or surplus run."""

STATUS_LEGACY = "legacy"
"""A pre-protocol model whose gaps are reported but not enforced."""

KIND_RUNS = "runs"
"""Fewer or more runs than the protocol requires."""

KIND_PARTIAL = "partial"
"""A run covering fewer samples than the full dataset."""

KIND_FAILED_SAMPLES = "failed_samples"
"""A run with provider errors recorded as wrong answers."""

KIND_VERDICTS = "verdicts"
"""A QA run lacking strict or judge verdicts."""

KIND_ORPHAN = "orphan"
"""A file for a model, task, or effort outside the config or protocol."""

_GAP_KINDS = frozenset({KIND_RUNS, KIND_PARTIAL})


@dataclass(frozen=True)
class Problem:
    """One validation finding for a model."""

    model: str
    severity: str
    kind: str
    message: str
    task: str | None = None
    effort: str | None = None
    fix: str | None = None

    @property
    def scope(self) -> str:
        """Human-readable ``task/effort`` scope, or ``-`` for model-level."""
        if self.task is None:
            return "-"
        if self.effort is None:
            return self.task
        return f"{self.task}/{self.effort}"


@dataclass(frozen=True)
class ModelCoverage:
    """How far one model is from the benchmark protocol."""

    model: str
    protocol: str
    runs_present: int
    runs_required: int
    problems: tuple[Problem, ...]
    legacy: bool

    @property
    def gaps(self) -> tuple[Problem, ...]:
        """Missing, surplus, or partial runs, regardless of severity."""
        return tuple(p for p in self.problems if p.kind in _GAP_KINDS)

    @property
    def status(self) -> str:
        """``complete``, ``incomplete``, or ``legacy``."""
        if not self.gaps and self.runs_present == self.runs_required:
            return STATUS_COMPLETE
        return STATUS_LEGACY if self.legacy else STATUS_INCOMPLETE

    @property
    def errors(self) -> tuple[Problem, ...]:
        """Problems that fail validation."""
        return tuple(p for p in self.problems if p.severity == ERROR)

    @property
    def warnings(self) -> tuple[Problem, ...]:
        """Problems that are only reported."""
        return tuple(p for p in self.problems if p.severity == WARNING)


@dataclass(frozen=True)
class ValidationReport:
    """Coverage of every configured model plus directory-level findings."""

    coverage: tuple[ModelCoverage, ...]
    orphans: tuple[Problem, ...]
    protocol: BenchmarkProtocol

    @property
    def errors(self) -> tuple[Problem, ...]:
        """Every failing problem across models and orphans."""
        model_errors = tuple(
            problem for entry in self.coverage for problem in entry.errors
        )
        return model_errors + tuple(p for p in self.orphans if p.severity == ERROR)

    @property
    def warnings(self) -> tuple[Problem, ...]:
        """Every reported-only problem across models and orphans."""
        model_warnings = tuple(
            problem for entry in self.coverage for problem in entry.warnings
        )
        return model_warnings + tuple(p for p in self.orphans if p.severity == WARNING)

    @property
    def ok(self) -> bool:
        """Whether validation passed."""
        return not self.errors


@dataclass(frozen=True)
class _LoadedRun:
    path: Path
    run: RunResult


def _load_runs(results_directory: Path) -> list[_LoadedRun]:
    loaded: list[_LoadedRun] = []
    for path in sorted(results_directory.glob("*.jsonl")):
        try:
            loaded.append(_LoadedRun(path=path, run=load_results(path)))
        except ValueError:
            continue
    return loaded


def _full_sample_counts(runs: list[_LoadedRun]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for loaded in runs:
        counts[loaded.run.task] = max(
            counts.get(loaded.run.task, 0), len(loaded.run.samples)
        )
    return counts


def _missing_verdicts(run: RunResult) -> int:
    return sum(
        1
        for sample in run.samples
        if not isinstance(sample.metadata.get("strict_correct"), bool)
        or not isinstance(sample.metadata.get("judge_correct"), bool)
    )


def _shell(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _configuration_problems(
    model: str,
    task: str,
    effort: str,
    runs: list[_LoadedRun],
    *,
    protocol: BenchmarkProtocol,
    severity: str,
    full_sample_counts: dict[str, int],
) -> list[Problem]:
    problems: list[Problem] = []
    count = len(runs)
    if count < protocol.repeats:
        missing = protocol.repeats - count
        problems.append(
            Problem(
                kind=KIND_RUNS,
                model=model,
                severity=severity,
                task=task,
                effort=effort,
                message=f"{count} of {protocol.repeats} runs",
                fix=_shell(run_command(task, model, effort, repeats=missing)),
            )
        )
    elif count > protocol.repeats:
        problems.append(
            Problem(
                kind=KIND_RUNS,
                model=model,
                severity=severity,
                task=task,
                effort=effort,
                message=f"{count} of {protocol.repeats} runs (surplus)",
                fix=(
                    "remove the surplus file(s); every file in results/ is "
                    "averaged, so keep exactly the protocol's repeats"
                ),
            )
        )

    expected_samples = full_sample_counts.get(task, 0)
    for loaded in runs:
        sample_count = len(loaded.run.samples)
        if sample_count < expected_samples:
            problems.append(
                Problem(
                    kind=KIND_PARTIAL,
                    model=model,
                    severity=severity,
                    task=task,
                    effort=effort,
                    message=(
                        f"{loaded.path.name} has {sample_count} of "
                        f"{expected_samples} samples (partial run)"
                    ),
                    fix=(
                        f"delete {loaded.path} and re-run without --max-samples: "
                        + _shell(run_command(task, model, effort))
                    ),
                )
            )
        failed = sum(1 for sample in loaded.run.samples if is_failed_sample(sample))
        if failed:
            resume = run_command(task, model, effort) + [
                "--resume-file",
                str(loaded.path),
            ]
            problems.append(
                Problem(
                    kind=KIND_FAILED_SAMPLES,
                    model=model,
                    severity=WARNING,
                    task=task,
                    effort=effort,
                    message=(
                        f"{loaded.path.name} has {failed} failed sample(s) "
                        "(provider errors scored as wrong)"
                    ),
                    fix=_shell(resume),
                )
            )
        if task in JUDGE_TASK_NAMES:
            missing_verdicts = _missing_verdicts(loaded.run)
            if missing_verdicts:
                problems.append(
                    Problem(
                        kind=KIND_VERDICTS,
                        model=model,
                        severity=ERROR,
                        task=task,
                        effort=effort,
                        message=(
                            f"{loaded.path.name} lacks strict/judge verdicts on "
                            f"{missing_verdicts} sample(s)"
                        ),
                        fix=f"vlm-exam rescore {loaded.path}",
                    )
                )
    return problems


def validate_results(
    results_directory: Path,
    config: BenchmarkConfig,
    *,
    protocol: BenchmarkProtocol = PROTOCOL,
    strict: bool = False,
) -> ValidationReport:
    """Check ``results/`` against the benchmark protocol.

    Every configured model is reported. Gaps on full-protocol models are
    errors; gaps on legacy models are warnings unless ``strict``. Failed
    samples are always warnings. Files for models missing from the config,
    unregistered tasks, and efforts outside the protocol are reported at
    directory level.

    Args:
        results_directory: Directory containing result JSONL files.
        config: Benchmark config naming every model and its protocol.
        protocol: Requirements to validate against.
        strict: Treat legacy models' gaps as errors too.

    Returns:
        The validation report.
    """
    loaded_runs = _load_runs(results_directory)
    full_sample_counts = _full_sample_counts(loaded_runs)

    by_configuration: dict[tuple[str, str, str], list[_LoadedRun]] = {}
    orphans: list[Problem] = []
    for loaded in loaded_runs:
        run = loaded.run
        if run.model not in config.models:
            orphans.append(
                Problem(
                    kind=KIND_ORPHAN,
                    model=run.model,
                    severity=ERROR,
                    task=run.task,
                    effort=run.effort,
                    message=(
                        f"{loaded.path.name} belongs to a model missing from "
                        "models.yaml"
                    ),
                    fix="add the model to models.yaml or move the file out of results/",
                )
            )
            continue
        if run.task not in protocol.tasks:
            orphans.append(
                Problem(
                    kind=KIND_ORPHAN,
                    model=run.model,
                    severity=WARNING,
                    task=run.task,
                    effort=run.effort,
                    message=(
                        f"{loaded.path.name} is for task {run.task!r}, which is "
                        "not part of the protocol and is ignored"
                    ),
                )
            )
            continue
        if run.effort not in protocol.efforts:
            orphans.append(
                Problem(
                    kind=KIND_ORPHAN,
                    model=run.model,
                    severity=WARNING,
                    task=run.task,
                    effort=run.effort,
                    message=(
                        f"{loaded.path.name} is at effort {run.effort!r}, which "
                        "is not part of the protocol and is ignored"
                    ),
                )
            )
            continue
        by_configuration.setdefault((run.model, run.task, run.effort), []).append(
            loaded
        )

    coverage: list[ModelCoverage] = []
    for model, model_config in config.models.items():
        severity = WARNING if model_config.is_legacy and not strict else ERROR
        problems: list[Problem] = []
        present = 0
        for task, effort in protocol.configurations:
            runs = by_configuration.get((model, task, effort), [])
            present += len(runs)
            problems.extend(
                _configuration_problems(
                    model,
                    task,
                    effort,
                    runs,
                    protocol=protocol,
                    severity=severity,
                    full_sample_counts=full_sample_counts,
                )
            )
        coverage.append(
            ModelCoverage(
                model=model,
                protocol=model_config.benchmark_protocol,
                runs_present=present,
                runs_required=protocol.required_runs,
                problems=tuple(problems),
                legacy=model_config.is_legacy,
            )
        )

    return ValidationReport(
        coverage=tuple(coverage),
        orphans=tuple(orphans),
        protocol=protocol,
    )


def _status_label(entry: ModelCoverage) -> str:
    if entry.errors:
        return "FAIL"
    if entry.status == STATUS_COMPLETE:
        return "OK"
    if entry.legacy:
        return "LEGACY"
    return "WARN"


def _missing_run_count(entry: ModelCoverage) -> int:
    return max(entry.runs_required - entry.runs_present, 0)


def _legacy_summary(entry: ModelCoverage) -> str:
    configurations = len({(p.task, p.effort) for p in entry.gaps})
    return (
        f"{_missing_run_count(entry)} missing run(s) across {configurations} "
        f"configuration(s); not enforced until backfilled. "
        f"Backfill: vlm-exam benchmark --models {entry.model}"
    )


def _problem_lines(problem: Problem) -> list[str]:
    lines = [f"  [{problem.severity}] {problem.scope:<20} {problem.message}"]
    if problem.fix:
        lines.append(f"      -> {problem.fix}")
    return lines


def format_report(report: ValidationReport, *, verbose: bool = False) -> str:
    """Render a validation report for a terminal.

    Legacy models' missing runs are collapsed into one line each unless
    ``verbose``; everything else is listed with its fix command.

    Args:
        report: The report to render.
        verbose: List every missing configuration of legacy models too.

    Returns:
        Multi-line text: a coverage table, per-model findings, then a
        verdict line.
    """
    protocol = report.protocol
    lines = [
        "Benchmark protocol: "
        f"{protocol.repeats} runs x {len(protocol.tasks)} tasks x "
        f"{len(protocol.efforts)} efforts ({', '.join(protocol.efforts)}) = "
        f"{protocol.required_runs} runs per model",
        "",
        f"{'Status':<7} {'Model':<32} {'Protocol':<9} {'Runs':>7}  Problems",
        "-" * 78,
    ]
    for entry in report.coverage:
        counts = []
        if entry.errors:
            counts.append(f"{len(entry.errors)} error(s)")
        if entry.warnings:
            counts.append(f"{len(entry.warnings)} warning(s)")
        lines.append(
            f"{_status_label(entry):<7} {entry.model:<32} {entry.protocol:<9} "
            f"{entry.runs_present:>3}/{entry.runs_required:<3}  "
            f"{', '.join(counts) or '-'}"
        )

    for entry in report.coverage:
        if not entry.problems:
            continue
        collapse = entry.legacy and not entry.errors and not verbose
        shown = (
            [p for p in entry.problems if p.kind not in _GAP_KINDS]
            if collapse
            else list(entry.problems)
        )
        if collapse and not shown and not entry.gaps:
            continue
        lines.append("")
        lines.append(
            f"{_status_label(entry)} {entry.model} ({entry.protocol} protocol, "
            f"{entry.runs_present}/{entry.runs_required} runs)"
        )
        if collapse and entry.gaps:
            lines.append(f"  [warning] {_legacy_summary(entry)}")
        for problem in shown:
            lines.extend(_problem_lines(problem))

    if report.orphans:
        lines.append("")
        lines.append("Directory-level findings")
        for problem in report.orphans:
            lines.extend(_problem_lines(problem))

    lines.append("")
    error_count = len(report.errors)
    warning_count = len(report.warnings)
    if report.ok:
        lines.append(
            f"PASS: no protocol violations ({warning_count} warning(s) reported)."
        )
    else:
        lines.append(
            f"FAIL: {error_count} protocol violation(s), "
            f"{warning_count} warning(s). Fix the [error] items above."
        )
    return "\n".join(lines)


def _annotation(severity: str, title: str, message: str) -> str:
    return f"::{severity} title={title}::{message}"


def format_github_annotations(report: ValidationReport) -> str:
    """Render findings as GitHub Actions workflow commands.

    Errors are emitted one per problem. Warnings are collapsed to one per
    model so the output stays within GitHub's per-step annotation limit.

    Args:
        report: The report to render.

    Returns:
        ``::error`` and ``::warning`` lines, newline separated.
    """
    lines: list[str] = []
    for problem in report.errors:
        message = problem.message
        if problem.fix:
            message = f"{message}. Fix: {problem.fix}"
        lines.append(_annotation(ERROR, f"{problem.model} {problem.scope}", message))
    for entry in report.coverage:
        if not entry.warnings:
            continue
        parts: list[str] = []
        if entry.legacy and entry.gaps:
            parts.append(_legacy_summary(entry))
        other = [p for p in entry.warnings if p.kind not in _GAP_KINDS]
        if other:
            parts.append("; ".join(f"{p.scope}: {p.message}" for p in other))
        lines.append(_annotation(WARNING, entry.model, " | ".join(parts)))
    for problem in report.orphans:
        if problem.severity == WARNING:
            lines.append(_annotation(WARNING, problem.model, problem.message))
    return "\n".join(lines)


def format_github_summary(report: ValidationReport) -> str:
    """Render a Markdown job summary for GitHub Actions.

    Args:
        report: The report to render.

    Returns:
        Markdown with a coverage table and the findings.
    """
    protocol = report.protocol
    verdict = "PASS" if report.ok else "FAIL"
    lines = [
        f"## Benchmark protocol validation: {verdict}",
        "",
        f"Protocol: {protocol.repeats} runs x {len(protocol.tasks)} tasks x "
        f"{len(protocol.efforts)} efforts = {protocol.required_runs} runs per "
        f"model. {len(report.errors)} error(s), {len(report.warnings)} warning(s).",
        "",
        "| Status | Model | Protocol | Runs | Errors | Warnings |",
        "|---|---|---|---|---|---|",
    ]
    for entry in report.coverage:
        lines.append(
            f"| {_status_label(entry)} | `{entry.model}` | {entry.protocol} | "
            f"{entry.runs_present}/{entry.runs_required} | "
            f"{len(entry.errors)} | {len(entry.warnings)} |"
        )
    findings: list[str] = []
    for problem in report.errors:
        fix = f" Fix: `{problem.fix}`" if problem.fix else ""
        findings.append(
            f"- **error** `{problem.model}` {problem.scope}: {problem.message}.{fix}"
        )
    for entry in report.coverage:
        if entry.legacy and entry.gaps and not entry.errors:
            findings.append(f"- warning `{entry.model}`: {_legacy_summary(entry)}")
        for problem in entry.warnings:
            if problem.kind in _GAP_KINDS and entry.legacy:
                continue
            fix = f" Fix: `{problem.fix}`" if problem.fix else ""
            findings.append(
                f"- warning `{problem.model}` {problem.scope}: {problem.message}.{fix}"
            )
    for problem in report.orphans:
        if problem.severity == WARNING:
            findings.append(f"- warning `{problem.model}`: {problem.message}")
    if findings:
        lines.extend(["", "### Findings", "", *findings])
    return "\n".join(lines)
