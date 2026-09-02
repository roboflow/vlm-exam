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
import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click
from dotenv import load_dotenv

from vlm_exam.config import BenchmarkConfig, load_config
from vlm_exam.judge import DEFAULT_JUDGE_MODEL, Judge
from vlm_exam.providers import build_model_provider
from vlm_exam.reference.cli import register_reference_commands
from vlm_exam.results import (
    RunResult,
    is_failed_sample,
    load_results,
    load_results_directory,
    merge_resumed_runs,
    save_results,
)
from vlm_exam.runner import run_benchmark
from vlm_exam.tasks import QA_TASK_NAMES, create_task

if TYPE_CHECKING:
    import matplotlib.pyplot as plt

load_dotenv()

_QA_DATASET_PROJECTS = {
    "ocr": "vlm-exam-ocr",
    "extraction": "vlm-exam-data-extraction",
    "counting": "vlm-exam-counting",
    "identification": "vlm-exam-identification",
    "reasoning": "vlm-exam-reasoning",
}

_QA_DATASET_VERSIONS = {
    "ocr": 1,
    "extraction": 1,
    "counting": 1,
    "identification": 1,
    "reasoning": 2,
}


def _build_judge(judge_model: str) -> Judge:
    if not os.environ.get("GOOGLE_API_KEY"):
        raise click.UsageError(
            "GOOGLE_API_KEY is not set. Counting, extraction, identification, "
            "and reasoning are scored by an LLM judge in addition to the strict "
            "rule, so the judge credentials are required."
        )
    return Judge(model=judge_model)


def _resolve_model_filter(
    config: BenchmarkConfig,
    models: str | None,
    group: str | None,
) -> set[str] | None:
    from vlm_exam.metrics import resolve_leaderboard_model_list

    try:
        model_list = resolve_leaderboard_model_list(
            config,
            models=models,
            group=group,
        )
    except ValueError as error:
        raise click.UsageError(str(error)) from error
    if model_list is None:
        return None
    return set(model_list)


@click.group()
def main() -> None:
    """vlm-exam: Benchmark suite for Vision Language Models."""


def _save_card(
    figure: plt.Figure, output_path: Path, index: int, image_name: str
) -> None:
    import matplotlib.pyplot as plt

    output_file = (output_path / f"{index:03d}_{image_name}").with_suffix(".png")
    figure.savefig(str(output_file), dpi=150)
    plt.close(figure)


@main.command()
@click.option(
    "--data-directory",
    default="data",
    type=click.Path(),
    help="Root directory to download datasets into.",
)
@click.option(
    "--workspace",
    default="roboflow-jvuqo",
    help="Roboflow workspace containing the benchmark projects.",
)
@click.option(
    "--dataset-version",
    default=None,
    type=int,
    help="Dataset version to download for every project; defaults per task.",
)
@click.option(
    "--tasks",
    "task_names",
    default=",".join(QA_TASK_NAMES),
    help="Comma-separated QA task names to download.",
)
def download(
    data_directory: str,
    workspace: str,
    dataset_version: int | None,
    task_names: str,
) -> None:
    """Download the QA benchmark datasets from Roboflow."""
    from roboflow import Roboflow

    roboflow_client = Roboflow(api_key=os.environ.get("ROBOFLOW_API_KEY"))
    workspace_client = roboflow_client.workspace(workspace)

    for task_name in [name.strip() for name in task_names.split(",")]:
        if task_name not in _QA_DATASET_PROJECTS:
            available = ", ".join(sorted(_QA_DATASET_PROJECTS))
            raise click.UsageError(
                f"Unknown task {task_name!r}. Available tasks: {available}"
            )
        project_slug = _QA_DATASET_PROJECTS[task_name]
        version_number = dataset_version or _QA_DATASET_VERSIONS[task_name]
        target = Path(data_directory) / task_name
        click.echo(f"Downloading {workspace}/{project_slug} v{version_number} ...")
        project = workspace_client.project(project_slug)
        version = project.version(version_number)
        version.download("jsonl", location=str(target), overwrite=True)
        click.echo(f"  saved to {target}")


@main.command()
@click.option(
    "--task",
    "task_name",
    required=True,
    help="Task to run (e.g. ocr, counting, detection).",
)
@click.option(
    "--models",
    required=True,
    help="Comma-separated model identifiers.",
)
@click.option(
    "--effort",
    required=True,
    help="Effort level (e.g. low, high).",
)
@click.option(
    "--dataset-directory",
    required=True,
    type=click.Path(exists=True),
    help="Path to the dataset directory.",
)
@click.option(
    "--output-directory",
    default="results",
    type=click.Path(),
    help="Directory to save result files.",
)
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(exists=True),
    help="Path to custom models.yaml config.",
)
@click.option(
    "--judge-model",
    "judge_model",
    default=DEFAULT_JUDGE_MODEL,
    show_default=True,
    help="LLM judge scoring every counting, extraction, identification, "
    "and reasoning sample alongside the strict rule.",
)
@click.option(
    "--max-samples",
    "max_samples",
    default=None,
    type=int,
    help="Limit the number of samples to evaluate (default: all).",
)
@click.option(
    "--prompt-classes",
    "prompt_classes",
    default="image",
    type=click.Choice(["image", "all"]),
    help=(
        "Detection only: list classes present in the image ground truth "
        "or all dataset classes in the prompt."
    ),
)
@click.option(
    "--resume-file",
    "resume_file",
    default=None,
    type=click.Path(exists=True),
    help=(
        "Prior result JSONL to resume: only its failed samples are "
        "re-run and merged into a new complete result file. The prior "
        "file is deleted once the merged file is saved so the directory "
        "never holds two copies of the same run."
    ),
)
@click.option(
    "--concurrency",
    default=1,
    show_default=True,
    type=click.IntRange(min=1),
    help="Number of samples evaluated in parallel per model.",
)
@click.option(
    "--repeats",
    default=1,
    show_default=True,
    type=click.IntRange(min=1),
    help=(
        "Run the whole configuration this many times, writing one result "
        "file per repeat. Committed models use three repeats."
    ),
)
def run(
    task_name: str,
    models: str,
    effort: str,
    dataset_directory: str,
    output_directory: str,
    config_path: str | None,
    judge_model: str,
    max_samples: int | None,
    prompt_classes: str,
    resume_file: str | None,
    concurrency: int,
    repeats: int,
) -> None:
    """Run a benchmark for one or more models."""
    if resume_file is not None and repeats != 1:
        raise click.UsageError("--resume-file cannot be combined with --repeats.")
    config = load_config(Path(config_path) if config_path else None)
    task_args: dict[str, str] = {}
    if task_name == "detection":
        task_args["prompt_classes"] = prompt_classes
    task = create_task(task_name, **task_args)
    samples = task.load_samples(dataset_directory)
    if max_samples is not None:
        samples = samples[:max_samples]
    model_ids = [model_id.strip() for model_id in models.split(",")]
    output_path = Path(output_directory)

    previous_run: RunResult | None = None
    if resume_file is not None:
        if len(model_ids) != 1:
            raise click.UsageError("--resume-file requires exactly one model.")
        previous_run = load_results(Path(resume_file))
        if previous_run.model != model_ids[0]:
            raise click.UsageError(
                f"--resume-file holds results for {previous_run.model!r}, "
                f"but --models is {model_ids[0]!r}."
            )
        if previous_run.task != task_name or previous_run.effort != effort:
            raise click.UsageError(
                f"--resume-file is a {previous_run.task!r} run at effort "
                f"{previous_run.effort!r}; pass matching --task and --effort."
            )
        failed_images = {
            sample.image for sample in previous_run.samples if is_failed_sample(sample)
        }
        samples = [
            sample
            for sample in samples
            if Path(sample.image_path).name in failed_images
        ]
        kept_count = len(previous_run.samples) - len(failed_images)
        click.echo(
            f"Resuming {previous_run.model}: keeping {kept_count} samples, "
            f"re-running {len(samples)} failed samples."
        )

    judge = _build_judge(judge_model) if task.requires_judge else None

    click.echo(f"Loaded {len(samples)} samples from {dataset_directory}")
    if judge is not None:
        click.echo(f"Scoring: strict rule and LLM judge ({judge_model})")

    for model_id in model_ids:
        if model_id not in config.models:
            click.echo(f"Warning: model {model_id!r} not found in config, skipping.")
            continue

        model_config = config.models[model_id]
        provider = build_model_provider(model_id, model_config)

        model_task = task
        if task_name == "detection":
            model_task = create_task(
                task_name,
                coordinate_format=model_config.detection_coordinate_format,
                **task_args,
            )

        for repeat in range(1, repeats + 1):
            if repeats > 1:
                click.echo(f"Repeat {repeat}/{repeats} for {model_id}")
            result = run_benchmark(
                task=model_task,
                provider=provider,
                samples=samples,
                effort=effort,
                task_name=task_name,
                judge=judge,
                concurrency=concurrency,
            )

            if previous_run is not None:
                result = merge_resumed_runs(previous_run, result)

            result_path = _unique_result_path(
                output_path, task_name, model_id, effort, result.timestamp
            )
            save_results(result, result_path)
            click.echo(f"Results saved to {result_path}")
            if resume_file is not None:
                source = Path(resume_file)
                if source.resolve() != result_path.resolve():
                    source.unlink()
                    click.echo(f"Removed resumed file {source}")


def _unique_result_path(
    output_path: Path,
    task_name: str,
    model_id: str,
    effort: str,
    timestamp: str,
) -> Path:
    stem = f"{task_name}_{model_id}_{effort}_{timestamp}"
    candidate = output_path / f"{stem}.jsonl"
    suffix = 2
    while candidate.exists():
        candidate = output_path / f"{stem}_{suffix}.jsonl"
        suffix += 1
    return candidate


@main.command()
@click.option(
    "--models",
    required=True,
    help="Comma-separated model identifiers to benchmark end to end.",
)
@click.option(
    "--tasks",
    default=None,
    help="Comma-separated tasks (default: every protocol task).",
)
@click.option(
    "--efforts",
    default=None,
    help="Comma-separated efforts (default: every protocol effort).",
)
@click.option(
    "--repeats",
    default=None,
    type=click.IntRange(min=1),
    help="Runs per configuration (default: the protocol's repeats).",
)
@click.option(
    "--first-repeat",
    default=1,
    show_default=True,
    type=click.IntRange(min=1),
    help="Repeat number of the first run, for log names when topping up.",
)
@click.option(
    "--dataset-root",
    default="data",
    show_default=True,
    type=click.Path(exists=True, file_okay=False),
    help="Directory holding one <task>/train dataset per task.",
)
@click.option(
    "--output-directory",
    default="results",
    show_default=True,
    type=click.Path(file_okay=False),
    help="Directory result files are written to.",
)
@click.option(
    "--log-directory",
    default="logs",
    show_default=True,
    type=click.Path(file_okay=False),
    help="Directory receiving one log per run.",
)
@click.option(
    "--max-parallel",
    default=18,
    show_default=True,
    type=click.IntRange(min=1),
    help=(
        "Maximum runs alive at once; the default fits one effort level of one "
        "model, so high starts as low runs finish."
    ),
)
@click.option(
    "--max-samples",
    default=None,
    type=int,
    help="Smoke tests only: cap samples per run. Never commit such runs.",
)
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(exists=True),
    help="Path to custom models.yaml config.",
)
def benchmark(
    models: str,
    tasks: str | None,
    efforts: str | None,
    repeats: int | None,
    first_repeat: int,
    dataset_root: str,
    output_directory: str,
    log_directory: str,
    max_parallel: int,
    max_samples: int | None,
    config_path: str | None,
) -> None:
    """Run the full benchmark protocol for one or more models in parallel."""
    from vlm_exam.metrics import parse_model_filter
    from vlm_exam.orchestrate import format_outcomes, plan_jobs, run_jobs

    config = load_config(Path(config_path) if config_path else None)
    model_ids = parse_model_filter(models, config)
    jobs = plan_jobs(
        model_ids,
        tasks=_split_option(tasks),
        efforts=_split_option(efforts),
        repeats=repeats,
        first_repeat=first_repeat,
        dataset_root=Path(dataset_root),
        output_directory=Path(output_directory),
        log_directory=Path(log_directory),
        max_samples=max_samples,
    )
    outcomes = run_jobs(jobs, max_parallel=max_parallel, echo=click.echo)
    click.echo("")
    click.echo(format_outcomes(outcomes))
    click.echo("")
    click.echo(
        "Next: vlm-exam validate, then vlm-exam summary --dataset-directory "
        f"{Path(dataset_root) / 'detection' / 'train'} and vlm-exam leaderboard."
    )
    if any(not outcome.ok for outcome in outcomes):
        raise SystemExit(1)


def _split_option(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    return tuple(part.strip() for part in value.split(",") if part.strip())


@main.command()
@click.option(
    "--results-directory",
    default="results",
    type=click.Path(exists=True),
    help="Directory containing result JSONL files.",
)
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(exists=True),
    help="Path to custom models.yaml config.",
)
@click.option(
    "--strict",
    is_flag=True,
    help="Treat legacy models' missing runs as errors too.",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="List every missing configuration of legacy models instead of a summary.",
)
@click.option(
    "--format",
    "output_format",
    default="text",
    show_default=True,
    type=click.Choice(["text", "github"]),
    help=(
        "github prints the text report, emits ::error/::warning annotations, "
        "and appends a Markdown table to $GITHUB_STEP_SUMMARY when set."
    ),
)
def validate(
    results_directory: str,
    config_path: str | None,
    strict: bool,
    verbose: bool,
    output_format: str,
) -> None:
    """Check results/ against the benchmark protocol; exit 1 on violations."""
    from vlm_exam.validation import (
        format_github_annotations,
        format_github_summary,
        format_report,
        validate_results,
    )

    config = load_config(Path(config_path) if config_path else None)
    report = validate_results(Path(results_directory), config, strict=strict)

    click.echo(format_report(report, verbose=verbose))
    if output_format == "github":
        annotations = format_github_annotations(report)
        if annotations:
            click.echo(annotations)
        summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary_path:
            with open(summary_path, "a", encoding="utf-8") as file:
                file.write(format_github_summary(report) + "\n")

    if not report.ok:
        raise SystemExit(1)


def _format_accuracy(count: int | None, total: int) -> str:
    if count is None or total == 0:
        return "-"
    return f"{count / total * 100:.1f}%"


@main.command()
@click.argument("paths", nargs=-1, required=True, type=click.Path(exists=True))
@click.option(
    "--judge-model",
    "judge_model",
    default=DEFAULT_JUDGE_MODEL,
    show_default=True,
    help="LLM judge model.",
)
@click.option(
    "--concurrency",
    "concurrency",
    default=8,
    type=click.IntRange(min=1),
    help="In-flight judge calls.",
)
@click.option(
    "--force",
    "force",
    is_flag=True,
    help="Re-score samples that already carry both verdicts.",
)
@click.option(
    "--dry-run",
    "dry_run",
    is_flag=True,
    help="Report how many samples would be scored without calling the judge.",
)
def rescore(
    paths: tuple[str, ...],
    judge_model: str,
    concurrency: int,
    force: bool,
    dry_run: bool,
) -> None:
    """Backfill strict and judge verdicts on stored QA runs in place.

    Accepts result files or directories. Only counting, extraction,
    identification, and reasoning runs are touched. The strict verdict is
    recomputed from the stored prediction; the judge scores every real
    prediction. Samples already carrying both verdicts are skipped unless
    --force is given, so the command is idempotent.
    """
    from vlm_exam.rescore import JUDGE_TASK_NAMES, has_both_verdicts, rescore_run

    files: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            files.extend(sorted(path.glob("*.jsonl")))
        else:
            files.append(path)

    def pending_count(run_result: RunResult) -> int:
        return sum(
            1 for sample in run_result.samples if force or not has_both_verdicts(sample)
        )

    runs: list[tuple[Path, RunResult]] = []
    for path in files:
        try:
            run_result = load_results(path)
        except ValueError:
            click.echo(f"Skipping empty file: {path}")
            continue
        if run_result.task not in JUDGE_TASK_NAMES:
            continue
        if pending_count(run_result) == 0:
            continue
        runs.append((path, run_result))

    if not runs:
        click.echo("Nothing to rescore.")
        return

    pending_total = sum(pending_count(run_result) for _, run_result in runs)
    click.echo(f"{len(runs)} runs, {pending_total} samples to score ({judge_model})")
    if dry_run:
        for path, run_result in runs:
            click.echo(f"  {path.name}: {pending_count(run_result)} to score")
        return

    judge = _build_judge(judge_model)
    click.echo(
        f"{'file':<62}{'scored':>7}{'strict before':>14}{'strict after':>13}"
        f"{'judge before':>13}{'judge after':>12}"
    )
    for path, run_result in runs:
        rescored, summary = rescore_run(
            run_result, judge, concurrency=concurrency, force=force
        )
        save_results(rescored, path)
        total = summary.total
        click.echo(
            f"{path.name:<62}{summary.scored:>7}"
            f"{_format_accuracy(summary.strict_before, total):>14}"
            f"{_format_accuracy(summary.strict_after, total):>13}"
            f"{_format_accuracy(summary.judge_before, total):>13}"
            f"{_format_accuracy(summary.judge_after, total):>12}"
        )


@main.command()
@click.option(
    "--results-directory",
    default="results",
    type=click.Path(exists=True),
    help="Directory containing result JSONL files.",
)
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(exists=True),
    help="Path to custom models.yaml config.",
)
def report(
    results_directory: str,
    config_path: str | None,
) -> None:
    """Generate summary tables from saved results."""
    config = load_config(Path(config_path) if config_path else None)
    runs = load_results_directory(Path(results_directory))

    if not runs:
        click.echo(f"No usable .jsonl files found in {results_directory}")
        return

    click.echo(
        f"\n{'Task':<15} {'Model':<25} {'Effort':>6} {'Runs':>4} "
        f"{'Total':>6} {'Judge':>16} {'Strict':>16}"
    )
    click.echo("-" * 93)

    from vlm_exam.metrics import (
        RepeatedMetric,
        aggregate_metric,
        run_accuracy,
        run_judge_accuracy,
        run_mean_similarity,
        run_strict_accuracy,
    )
    from vlm_exam.rescore import JUDGE_TASK_NAMES

    def format_repeated(metric: RepeatedMetric | None, suffix: str = "%") -> str:
        if metric is None:
            return "-"
        if metric.run_count == 1:
            return f"{metric.mean:.1f}{suffix}"
        return f"{metric.mean:.1f}{suffix} \u00b1{metric.spread / 2:.1f}"

    unknown_models = sorted({run.model for run in runs} - set(config.models))
    if unknown_models:
        click.echo(
            f"Warning: skipping runs for models missing from config: "
            f"{', '.join(unknown_models)}"
        )

    from vlm_exam.metrics import group_runs

    groups = group_runs(runs, config)
    for (task_name, effort, model), group in sorted(groups.items()):
        total = max(len(run.samples) for run in group)
        strict = ""
        if task_name == "ocr":
            metric = format_repeated(
                aggregate_metric(group, run_mean_similarity), "% sim"
            )
        elif task_name in JUDGE_TASK_NAMES:
            metric = format_repeated(aggregate_metric(group, run_judge_accuracy))
            strict = format_repeated(aggregate_metric(group, run_strict_accuracy))
        else:
            metric = format_repeated(aggregate_metric(group, run_accuracy))

        click.echo(
            f"{task_name:<15} {model:<25} {effort:>6} {len(group):>4} "
            f"{total:>6} {metric:>16} {strict:>16}"
        )

    click.echo()

    click.echo(
        f"{'Model':<25} {'Effort':>6} {'Runs':>4} "
        f"{'Input Tok':>10} {'Output Tok':>11} {'Cost/run':>9}"
    )
    click.echo("-" * 70)

    from vlm_exam.metrics import sample_cost

    grand_cost = 0.0
    for (model, effort), model_groups in sorted(
        _groups_by_model_effort(groups).items()
    ):
        pricing = config.models[model]
        run_count = max(len(group) for group in model_groups)
        total_input = 0.0
        total_output = 0.0
        cost = 0.0
        for group in model_groups:
            total_input += sum(
                sample.input_tokens for run in group for sample in run.samples
            ) / len(group)
            total_output += sum(
                sample.output_tokens for run in group for sample in run.samples
            ) / len(group)
            cost += sum(
                sample_cost(sample, pricing) for run in group for sample in run.samples
            ) / len(group)
        grand_cost += cost

        click.echo(
            f"{model:<25} {effort:>6} {run_count:>4} "
            f"{round(total_input):>10,} {round(total_output):>11,} "
            f"${cost:>8.4f}"
        )

    click.echo(f"\nTotal benchmark cost per run: ${grand_cost:.4f}")


def _groups_by_model_effort(
    groups: dict[tuple[str, str, str], list[RunResult]],
) -> dict[tuple[str, str], list[list[RunResult]]]:
    by_model_effort: dict[tuple[str, str], list[list[RunResult]]] = {}
    for (_, effort, model), group in groups.items():
        by_model_effort.setdefault((model, effort), []).append(group)
    return by_model_effort


@main.command()
@click.option(
    "--results-directory",
    default="results",
    type=click.Path(exists=True),
    help="Directory containing result JSONL files.",
)
@click.option(
    "--dataset-directory",
    "dataset_directory",
    default=None,
    type=click.Path(exists=True),
    help="Detection dataset directory (required to include detection mAP).",
)
@click.option(
    "--effort",
    default=None,
    help="Effort level to include (default: all efforts).",
)
@click.option(
    "--output-file",
    default="web/benchmark_summary.json",
    type=click.Path(),
    help="Path to write the compiled summary JSON.",
)
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(exists=True),
    help="Path to custom models.yaml config.",
)
@click.option(
    "--models",
    default=None,
    help="Comma-separated model identifiers to include.",
)
@click.option(
    "--group",
    default=None,
    help="Named leaderboard model group (e.g. alternative). Overrides --models.",
)
@click.option(
    "--check",
    is_flag=True,
    help=(
        "Do not write; exit 1 if --output-file differs from a fresh build. "
        "Detection mAP fields are ignored when no --dataset-directory is given."
    ),
)
def summary(
    results_directory: str,
    dataset_directory: str | None,
    effort: str | None,
    output_file: str,
    config_path: str | None,
    models: str | None,
    group: str | None,
    check: bool,
) -> None:
    """Compile all result files into a single frontend-facing JSON."""
    from vlm_exam.summary import build_summary, summary_drift, summary_to_dict

    config = load_config(Path(config_path) if config_path else None)
    model_filter = _resolve_model_filter(config, models, group)

    results_path = Path(results_directory)
    detection_dataset = Path(dataset_directory) if dataset_directory else None
    has_detection_runs = any(results_path.glob("detection_*.jsonl"))
    if detection_dataset is None and has_detection_runs:
        click.echo(
            "No --dataset-directory given; detection quality metrics "
            "(mAP) will be omitted."
        )

    benchmark_summary = build_summary(
        results_path,
        config,
        effort,
        models=model_filter,
        detection_dataset_directory=detection_dataset,
    )

    output_path = Path(output_file)
    payload = summary_to_dict(benchmark_summary)
    if check:
        if not output_path.exists():
            raise click.ClickException(f"{output_path} does not exist.")
        with open(output_path) as file:
            committed = json.load(file)
        drift = summary_drift(
            committed, payload, ignore_detection_quality=detection_dataset is None
        )
        if drift:
            click.echo("\n".join(drift))
            raise click.ClickException(
                f"{output_path} is out of date. Regenerate it with "
                "`vlm-exam summary --dataset-directory data/detection/train` "
                "and commit the result."
            )
        click.echo(f"{output_path} matches results/ and models.yaml.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")

    if not benchmark_summary.models:
        click.echo(f"No benchmark runs found; wrote empty summary to {output_path}")
        return

    click.echo(
        f"Compiled {len(benchmark_summary.models)} model runs across "
        f"{len(benchmark_summary.tasks)} tasks to {output_path}"
    )


@main.command("efficiency-report")
@click.option(
    "--results-directory",
    default="results",
    type=click.Path(exists=True),
    help="Directory containing result JSONL files.",
)
@click.option(
    "--effort",
    default="low",
    help="Effort level to aggregate (default: low).",
)
@click.option(
    "--output-directory",
    default="visualizations/leaderboards",
    type=click.Path(),
    help="Directory to save efficiency charts.",
)
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(exists=True),
    help="Path to custom models.yaml config.",
)
@click.option(
    "--models",
    default=None,
    help="Comma-separated model identifiers to include.",
)
@click.option(
    "--group",
    default=None,
    help="Named leaderboard model group (e.g. alternative). Overrides --models.",
)
def efficiency_report(
    results_directory: str,
    effort: str,
    output_directory: str,
    config_path: str | None,
    models: str | None,
    group: str | None,
) -> None:
    """Print pooled per-model efficiency metrics and save chart PNGs."""
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    from vlm_exam.metrics import aggregate_efficiency_by_model
    from vlm_exam.visualization import plot_combined_metrics_chart, plot_metric_chart

    config = load_config(Path(config_path) if config_path else None)
    model_filter = _resolve_model_filter(config, models, group)
    rows = aggregate_efficiency_by_model(
        Path(results_directory),
        config,
        effort,
        models=model_filter,
    )

    if not rows:
        click.echo("No benchmark runs found for efficiency aggregation.")
        return

    click.echo(
        f"\n{'Model':<28} {'Tasks':>5} {'Samples':>7} "
        f"{'AvgTok':>8} {'AvgCost':>10} {'AvgTime':>8} {'TotCost':>9}"
    )
    click.echo("-" * 80)

    grand_cost = 0.0
    for row in sorted(rows, key=lambda entry: entry.total_cost):
        grand_cost += row.total_cost
        click.echo(
            f"{row.model:<28} {row.task_count:>5} {row.sample_count:>7} "
            f"{row.average_tokens:>8.0f} "
            f"{row.average_cost:>10.5f} "
            f"{row.average_time_seconds:>7.1f}s "
            f"{row.total_cost:>9.2f}"
        )

    click.echo(f"\nTotal estimated benchmark cost: ${grand_cost:.2f}")

    average_tokens = {row.model: row.average_tokens for row in rows}
    average_cost = {row.model: row.average_cost for row in rows}
    average_time = {row.model: row.average_time_seconds for row in rows}
    zero = {model: 0.0 for model in average_tokens}

    def format_cost(value: float) -> str:
        if value >= 0.001:
            return f"${value:.4f}"
        return f"${value:.5f}"

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    def save_figure(figure: plt.Figure, filename: str) -> None:
        file_path = output_path / filename
        figure.savefig(str(file_path), dpi=150)
        plt.close(figure)
        saved.append(file_path)

    save_figure(
        plot_metric_chart(
            average_tokens,
            config,
            "Benchmark Efficiency \u2014 Avg Tokens per Image",
            format_value=lambda value: f"{value:,.0f}",
            sort_ascending=True,
        ),
        f"efficiency_tokens_{effort}.png",
    )
    save_figure(
        plot_metric_chart(
            average_cost,
            config,
            "Benchmark Efficiency \u2014 Avg Cost per Image",
            format_value=format_cost,
            sort_ascending=True,
        ),
        f"efficiency_cost_{effort}.png",
    )
    save_figure(
        plot_metric_chart(
            average_time,
            config,
            "Benchmark Efficiency \u2014 Avg Time per Image",
            format_value=lambda value: f"{value:.1f}s",
            sort_ascending=True,
        ),
        f"efficiency_time_{effort}.png",
    )
    save_figure(
        plot_combined_metrics_chart(
            tokens_high=zero,
            tokens_low=average_tokens,
            cost_high=zero,
            cost_low=average_cost,
            time_high=zero,
            time_low=average_time,
            config=config,
            effort="low",
            column_order=("cost", "time", "tokens"),
            sort_by="cost",
            sort_ascending=True,
        ),
        f"efficiency_combined_{effort}.png",
    )

    click.echo(f"\nSaved {len(saved)} efficiency charts to {output_path}:")
    for file_path in saved:
        click.echo(f"  {file_path.name}")


@main.command("detection-report")
@click.option(
    "--results-directory",
    default="results",
    type=click.Path(exists=True),
    help="Directory containing detection result JSONL files.",
)
@click.option(
    "--dataset-directory",
    required=True,
    type=click.Path(exists=True),
    help="Path to the detection dataset directory (for ground truth).",
)
def detection_report(
    results_directory: str,
    dataset_directory: str,
) -> None:
    """Compute dataset-level mAP for detection runs."""
    from vlm_exam.tasks.detection import (
        DetectionTask,
        build_sample_index,
        compute_dataset_map,
    )

    task = DetectionTask()
    samples = task.load_samples(dataset_directory)
    sample_by_image = build_sample_index(samples)

    runs = load_results_directory(Path(results_directory), pattern="detection_*.jsonl")

    if not runs:
        click.echo(f"No detection result files found in {results_directory}")
        return

    groups: dict[tuple[str, str], list[RunResult]] = {}
    for run_result in runs:
        groups.setdefault((run_result.model, run_result.effort), []).append(run_result)

    for (model, effort), group in sorted(groups.items()):
        click.echo(f"\n{'=' * 60}")
        click.echo(f"  {model}  effort={effort}  runs={len(group)}")
        click.echo(f"{'=' * 60}")

        map_results = []
        for run_result in sorted(group, key=lambda run: run.timestamp):
            map_result = compute_dataset_map(run_result, sample_by_image)
            if map_result is None:
                click.echo(f"  {run_result.timestamp}: no valid predictions found.")
                continue
            map_results.append(map_result)
            click.echo(
                f"  {run_result.timestamp}: mAP@50={map_result.map50:.4f} "
                f"mAP@75={map_result.map75:.4f} "
                f"mAP@50:95={map_result.map50_95:.4f} "
                f"images={map_result.image_count}"
            )
        if not map_results:
            continue

        count = len(map_results)
        click.echo(
            f"\n  mean mAP@50:    {sum(r.map50 for r in map_results) / count:.4f}"
        )
        click.echo(f"  mean mAP@75:    {sum(r.map75 for r in map_results) / count:.4f}")
        click.echo(
            f"  mean mAP@50:95: {sum(r.map50_95 for r in map_results) / count:.4f}"
        )
        click.echo()


@main.command()
@click.option(
    "--results-directory",
    default="results",
    type=click.Path(exists=True),
    help="Directory containing result JSONL files.",
)
@click.option(
    "--dataset-directory",
    "dataset_directory",
    default=None,
    type=click.Path(exists=True),
    help="Detection dataset directory (required for detection leaderboards).",
)
@click.option(
    "--output-directory",
    default="visualizations/leaderboards",
    type=click.Path(),
    help="Directory to save leaderboard charts.",
)
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(exists=True),
    help="Path to custom models.yaml config.",
)
@click.option(
    "--models",
    default=None,
    help="Comma-separated model identifiers to include.",
)
@click.option(
    "--group",
    default=None,
    help="Named leaderboard model group (e.g. alternative). Overrides --models.",
)
def leaderboard(
    results_directory: str,
    dataset_directory: str | None,
    output_directory: str,
    config_path: str | None,
    models: str | None,
    group: str | None,
) -> None:
    """Generate leaderboard charts for all locally saved runs."""
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    from vlm_exam.metrics import RepeatedMetric, aggregate_metric, group_runs
    from vlm_exam.visualization import plot_accuracy_chart, plot_metric_chart

    config = load_config(Path(config_path) if config_path else None)
    model_filter = _resolve_model_filter(config, models, group)
    results_path = Path(results_directory)
    runs = load_results_directory(results_path)

    if not runs:
        click.echo(f"No usable .jsonl files found in {results_directory}")
        return

    groups = group_runs(runs, config, models=model_filter)

    if not groups:
        click.echo("No usable runs found.")
        return

    groups_by_task_effort: dict[tuple[str, str], dict[str, list[RunResult]]] = {}
    for (task_name, effort, model), model_runs in groups.items():
        groups_by_task_effort.setdefault((task_name, effort), {})[model] = model_runs

    def measure(
        model_runs: dict[str, list[RunResult]],
        metric: Callable[[RunResult], float | None],
    ) -> dict[str, RepeatedMetric]:
        measured = {
            model: aggregate_metric(group, metric)
            for model, group in model_runs.items()
        }
        return {model: value for model, value in measured.items() if value is not None}

    def chart_inputs(
        measured: dict[str, RepeatedMetric],
    ) -> dict[str, Any]:
        return {
            "spread": {
                model: (metric.minimum, metric.maximum)
                for model, metric in measured.items()
                if metric.run_count > 1
            },
            "run_counts": {
                model: metric.run_count for model, metric in measured.items()
            },
        }

    def means(measured: dict[str, RepeatedMetric]) -> dict[str, float]:
        return {model: metric.mean for model, metric in measured.items()}

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    def save_figure(figure: plt.Figure, filename: str) -> None:
        file_path = output_path / filename
        figure.savefig(str(file_path), dpi=150)
        plt.close(figure)
        saved.append(file_path)

    detection_index = None
    if any(task_name == "detection" for task_name, _ in groups_by_task_effort):
        if dataset_directory is None:
            click.echo(
                "Detection runs found but --dataset-directory not given; "
                "skipping detection leaderboards."
            )
        else:
            from vlm_exam.tasks.detection import DetectionTask, build_sample_index

            detection_task = DetectionTask()
            detection_samples = detection_task.load_samples(dataset_directory)
            detection_index = build_sample_index(detection_samples)

    from vlm_exam.metrics import (
        run_accuracy,
        run_judge_accuracy,
        run_mean_similarity,
        run_strict_accuracy,
    )

    efforts_by_task: dict[str, set[str]] = {}
    for task_name, effort in groups_by_task_effort:
        efforts_by_task.setdefault(task_name, set()).add(effort)

    for (task_name, effort), model_runs in sorted(groups_by_task_effort.items()):
        effort_suffix = (
            f" \u2014 {effort.title()} Effort"
            if len(efforts_by_task[task_name]) > 1
            else ""
        )
        if task_name == "ocr":
            accuracy = measure(model_runs, run_accuracy)
            figure = plot_accuracy_chart(
                means(accuracy),
                config,
                f"OCR Benchmark \u2014 Accuracy{effort_suffix}",
                **chart_inputs(accuracy),
            )
            save_figure(figure, f"ocr_accuracy_{effort}.png")
            similarity = measure(model_runs, run_mean_similarity)
            figure = plot_accuracy_chart(
                means(similarity),
                config,
                f"OCR Benchmark \u2014 Mean Similarity{effort_suffix}",
                **chart_inputs(similarity),
            )
            save_figure(figure, f"ocr_similarity_{effort}.png")

        elif task_name in QA_TASK_NAMES:
            judge_accuracy = measure(model_runs, run_judge_accuracy)
            figure = plot_accuracy_chart(
                means(judge_accuracy),
                config,
                f"{task_name.title()} Benchmark \u2014 LLM Judge{effort_suffix}",
                **chart_inputs(judge_accuracy),
            )
            save_figure(figure, f"{task_name}_accuracy_{effort}.png")
            strict_accuracy = measure(model_runs, run_strict_accuracy)
            figure = plot_accuracy_chart(
                means(strict_accuracy),
                config,
                f"{task_name.title()} Benchmark \u2014 Strict Match{effort_suffix}",
                **chart_inputs(strict_accuracy),
            )
            save_figure(figure, f"{task_name}_accuracy_strict_{effort}.png")

        elif task_name == "detection":
            if detection_index is None:
                continue

            from vlm_exam.tasks.detection import compute_dataset_map

            def dataset_map(run: RunResult, attribute: str) -> float | None:
                map_result = compute_dataset_map(run, detection_index)
                if map_result is None:
                    click.echo(
                        f"No valid predictions for {run.model} ({run.effort}, "
                        f"{run.timestamp}); skipping that run."
                    )
                    return None
                return getattr(map_result, attribute)

            metric_titles = {
                "map50": "mAP@50",
                "map75": "mAP@75",
                "map50_95": "mAP@50:95",
            }
            for metric_key, metric_title in metric_titles.items():
                measured = measure(
                    model_runs,
                    lambda run, attribute=metric_key: dataset_map(run, attribute),
                )
                if not measured:
                    continue
                figure = plot_metric_chart(
                    means(measured),
                    config,
                    f"Object Detection \u2014 {metric_title}{effort_suffix}",
                    format_value=lambda value: f"{value * 100:.1f}%",
                    sort_ascending=False,
                    full_scale=1.0,
                    **chart_inputs(measured),
                )
                save_figure(figure, f"detection_{metric_key}_{effort}.png")

        else:
            click.echo(f"No leaderboard renderer for task {task_name!r}; skipping.")

    if not saved:
        click.echo("No leaderboard charts generated.")
        return

    click.echo(f"Saved {len(saved)} leaderboard charts to {output_path}:")
    for file_path in saved:
        click.echo(f"  {file_path.name}")


@main.command()
@click.option(
    "--results-file",
    required=True,
    type=click.Path(exists=True),
    help="Path to a QA result JSONL file.",
)
@click.option(
    "--dataset-directory",
    required=True,
    type=click.Path(exists=True),
    help="Path to the dataset directory containing the images.",
)
@click.option(
    "--output-directory",
    default="visualizations",
    type=click.Path(),
    help="Directory to save case cards.",
)
@click.option(
    "--max-images",
    default=20,
    type=int,
    help="Maximum number of cards to render.",
)
@click.option(
    "--only",
    "only_filter",
    default="all",
    type=click.Choice(["all", "correct", "incorrect"]),
    help="Render all cases, only correct ones, or only incorrect ones.",
)
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(exists=True),
    help="Path to custom models.yaml config.",
)
def visualize(
    results_file: str,
    dataset_directory: str,
    output_directory: str,
    max_images: int,
    only_filter: str,
    config_path: str | None,
) -> None:
    """Render case cards for a QA benchmark run."""
    import matplotlib

    matplotlib.use("Agg")

    from PIL import Image

    from vlm_exam.visualization import render_case_card

    run_result = load_results(Path(results_file))
    if run_result.task not in QA_TASK_NAMES:
        raise click.UsageError(
            f"--results-file holds a {run_result.task!r} run; "
            f"expected one of: {', '.join(QA_TASK_NAMES)}."
        )

    config = load_config(Path(config_path) if config_path else None)
    if run_result.model not in config.models:
        click.echo(f"Model {run_result.model!r} not found in config.")
        return

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    count = 0
    for sample_result in run_result.samples:
        if count >= max_images:
            break
        if only_filter == "correct" and not sample_result.correct:
            continue
        if only_filter == "incorrect" and sample_result.correct:
            continue
        if is_failed_sample(sample_result):
            continue

        image_path = Path(dataset_directory) / sample_result.image
        if not image_path.exists():
            click.echo(f"Skipping missing image: {image_path}")
            continue

        image = Image.open(image_path).convert("RGB")
        figure = render_case_card(run_result, sample_result, image, config)
        _save_card(figure, output_path, count, sample_result.image)
        count += 1

    click.echo(f"Saved {count} case cards to {output_path}")


@main.command("detection-visualize")
@click.option(
    "--results-file",
    required=True,
    type=click.Path(exists=True),
    help="Path to a detection result JSONL file.",
)
@click.option(
    "--dataset-directory",
    required=True,
    type=click.Path(exists=True),
    help="Path to the detection dataset directory.",
)
@click.option(
    "--output-directory",
    default="visualizations",
    type=click.Path(),
    help="Directory to save annotated images.",
)
@click.option(
    "--max-images",
    default=20,
    type=int,
    help="Maximum number of images to visualize.",
)
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(exists=True),
    help="Path to custom models.yaml config.",
)
@click.option(
    "--label-mode",
    "label_mode",
    default="auto",
    type=click.Choice(["auto", "labels", "boxes"]),
    help=(
        "Draw class labels on boxes, boxes with an in-image class color "
        "legend, or pick automatically based on label density."
    ),
)
@click.option(
    "--format",
    "output_format",
    default="card",
    type=click.Choice(["card", "plain"]),
    help="Save hero cards or plain annotated PNGs.",
)
@click.option(
    "--image",
    default=None,
    help="Only visualize this image basename from the results file.",
)
@click.option(
    "--index",
    "sample_index",
    default=None,
    type=int,
    help="Only visualize this sample index from the results file.",
)
def detection_visualize(
    results_file: str,
    dataset_directory: str,
    output_directory: str,
    max_images: int,
    config_path: str | None,
    label_mode: str,
    output_format: str,
    image: str | None,
    sample_index: int | None,
) -> None:
    """Visualize detection predictions vs ground truth."""
    import cv2

    from vlm_exam.tasks.detection import (
        DetectionCoordinateFormat,
        DetectionTask,
        build_sample_index,
        detection_labels,
        parse_prediction,
        recorded_uploaded_wh,
    )
    from vlm_exam.visualization.detection import (
        plot_detection_card,
        save_annotated_detection,
    )

    task = DetectionTask()
    samples = task.load_samples(dataset_directory)
    sample_by_image = build_sample_index(samples)

    run_result = load_results(Path(results_file))
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    config = load_config(Path(config_path) if config_path else None)
    if run_result.model not in config.models:
        click.echo(f"Model {run_result.model!r} not found in config.")
        return

    use_card = output_format == "card"
    if use_card:
        import matplotlib

        matplotlib.use("Agg")

    count = 0
    for sample_result in run_result.samples:
        if count >= max_images:
            break
        if sample_index is not None and sample_result.index != sample_index:
            continue
        if image is not None and sample_result.image != image:
            continue

        sample = sample_by_image.get(sample_result.image)
        if sample is None:
            continue

        image_bgr = cv2.imread(sample.image_path)
        if image_bgr is None:
            continue

        resolution_wh = (sample.image_width, sample.image_height)
        predicted = parse_prediction(
            sample_result.predicted,
            resolution_wh,
            list(sample.classes),
            coordinate_format=DetectionCoordinateFormat(
                sample_result.metadata.get(
                    "coordinate_format",
                    DetectionCoordinateFormat.YXYX_NORMALIZED_0_TO_1000.value,
                )
            ),
            uploaded_wh=recorded_uploaded_wh(sample_result.metadata),
        )

        pred_labels = detection_labels(predicted, list(sample.classes))
        stem = f"{sample_result.index:03d}_{sample_result.image}"
        output_file = (output_path / stem).with_suffix(".png")

        if use_card:
            gt_labels = detection_labels(sample.ground_truth, list(sample.classes))
            map_score = sample_result.metadata.get("map50")
            figure = plot_detection_card(
                image=image_bgr,
                ground_truth=sample.ground_truth,
                predictions=predicted,
                gt_labels=gt_labels,
                pred_labels=pred_labels,
                model_id=run_result.model,
                config=config,
                map_score=map_score,
                label_mode=label_mode,
            )
            _save_card(figure, output_path, sample_result.index, sample_result.image)
        else:
            save_annotated_detection(
                image_bgr,
                predicted,
                pred_labels,
                output_file,
                label_mode=label_mode,
            )
        count += 1

    click.echo(f"Saved {count} visualizations to {output_path}")


register_reference_commands(main)


if __name__ == "__main__":
    main()
