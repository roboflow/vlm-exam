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

import os
from pathlib import Path
from typing import TYPE_CHECKING

import click
from dotenv import load_dotenv

from vlm_exam.config import load_config
from vlm_exam.judge import Judge
from vlm_exam.providers import create_provider
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
    default=1,
    type=int,
    help="Dataset version to download for every project.",
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
    dataset_version: int,
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
        target = Path(data_directory) / task_name
        click.echo(f"Downloading {workspace}/{project_slug} v{dataset_version} ...")
        project = workspace_client.project(project_slug)
        version = project.version(dataset_version)
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
    "--match-mode",
    "match_mode",
    default="strict",
    type=click.Choice(["strict", "judge"]),
    help="Answer matching mode: strict (exact) or judge (LLM fallback).",
)
@click.option(
    "--judge-model",
    "judge_model",
    default="gemini-3.5-flash",
    help="Model to use as LLM judge (only used with --match-mode=judge).",
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
        "re-run and merged into a new complete result file."
    ),
)
def run(
    task_name: str,
    models: str,
    effort: str,
    dataset_directory: str,
    output_directory: str,
    config_path: str | None,
    match_mode: str,
    judge_model: str,
    max_samples: int | None,
    prompt_classes: str,
    resume_file: str | None,
) -> None:
    """Run a benchmark for one or more models."""
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

    judge: Judge | None = None
    if match_mode == "judge":
        judge = Judge(model=judge_model)

    click.echo(f"Loaded {len(samples)} samples from {dataset_directory}")
    if judge:
        click.echo(f"Match mode: {match_mode} (judge: {judge_model})")
    else:
        click.echo(f"Match mode: {match_mode}")

    for model_id in model_ids:
        if model_id not in config.models:
            click.echo(f"Warning: model {model_id!r} not found in config, skipping.")
            continue

        model_config = config.models[model_id]
        provider = create_provider(model_config.provider, model=model_id)

        model_task = task
        if task_name == "detection" and model_config.provider == "anthropic":
            model_task = create_task(task_name, coordinate_format="pixel", **task_args)

        result = run_benchmark(
            task=model_task,
            provider=provider,
            samples=samples,
            effort=effort,
            task_name=task_name,
            match_mode=match_mode,
            judge=judge,
        )

        if previous_run is not None:
            result = merge_resumed_runs(previous_run, result)

        filename = f"{task_name}_{model_id}_{effort}_{result.timestamp}.jsonl"
        result_path = output_path / filename
        save_results(result, result_path)
        click.echo(f"Results saved to {result_path}")


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
        f"\n{'Task':<15} {'Model':<25} {'Effort':>6} "
        f"{'Correct':>8} {'Total':>6} {'Metric':>10}"
    )
    click.echo("-" * 76)

    for run_result in sorted(runs, key=lambda run: (run.task, run.model)):
        correct = sum(sample.correct for sample in run_result.samples)
        total = len(run_result.samples)

        if run_result.task == "ocr":
            mean_similarity = (
                sum(sample.metadata.get("score", 0.0) for sample in run_result.samples)
                / total
                * 100
                if total > 0
                else 0.0
            )
            metric = f"{mean_similarity:.1f}% sim"
        else:
            accuracy = correct / total * 100 if total > 0 else 0.0
            metric = f"{accuracy:.1f}%"

        click.echo(
            f"{run_result.task:<15} {run_result.model:<25} {run_result.effort:>6} "
            f"{correct:>8} {total:>6} {metric:>10}"
        )

    click.echo()

    click.echo(
        f"{'Model':<25} {'Effort':>6} {'Input Tok':>10} {'Output Tok':>11} {'Cost':>9}"
    )
    click.echo("-" * 67)

    grand_cost = 0.0
    for run_result in runs:
        total_input = sum(sample.input_tokens for sample in run_result.samples)
        total_output = sum(sample.output_tokens for sample in run_result.samples)

        pricing = config.models.get(run_result.model)
        if pricing:
            cost = (
                total_input / 1_000_000
            ) * pricing.pricing.input_per_million_tokens + (
                total_output / 1_000_000
            ) * pricing.pricing.output_per_million_tokens
        else:
            cost = 0.0
        grand_cost += cost

        click.echo(
            f"{run_result.model:<25} {run_result.effort:>6} "
            f"{total_input:>10,} {total_output:>11,} "
            f"${cost:>8.4f}"
        )

    click.echo(f"\nTotal benchmark cost: ${grand_cost:.4f}")


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

    for run_result in runs:
        click.echo(f"\n{'=' * 60}")
        click.echo(f"  {run_result.model}  effort={run_result.effort}")
        click.echo(f"{'=' * 60}")

        map_result = compute_dataset_map(run_result, sample_by_image)
        if map_result is None:
            click.echo("  No valid predictions found.")
            continue

        click.echo(f"\n  mAP@50:    {map_result.map50:.4f}")
        click.echo(f"  mAP@75:    {map_result.map75:.4f}")
        click.echo(f"  mAP@50:95: {map_result.map50_95:.4f}")
        click.echo(f"  Images:    {map_result.image_count}")
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
def leaderboard(
    results_directory: str,
    dataset_directory: str | None,
    output_directory: str,
    config_path: str | None,
) -> None:
    """Generate leaderboard charts for all locally saved runs."""
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    from vlm_exam.visualization import plot_accuracy_chart, plot_metric_chart

    config = load_config(Path(config_path) if config_path else None)
    results_path = Path(results_directory)
    runs = load_results_directory(results_path)

    if not runs:
        click.echo(f"No usable .jsonl files found in {results_directory}")
        return

    latest_runs: dict[tuple[str, str, str], RunResult] = {}
    for run_result in runs:
        if run_result.model not in config.models:
            click.echo(f"Skipping run with unknown model {run_result.model!r}")
            continue
        key = (run_result.task, run_result.effort, run_result.model)
        existing = latest_runs.get(key)
        if existing is None or run_result.timestamp > existing.timestamp:
            latest_runs[key] = run_result

    if not latest_runs:
        click.echo("No usable runs found.")
        return

    runs_by_task_effort: dict[tuple[str, str], list[RunResult]] = {}
    for (task_name, effort, _), run_result in latest_runs.items():
        runs_by_task_effort.setdefault((task_name, effort), []).append(run_result)

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    def save_figure(figure: plt.Figure, filename: str) -> None:
        file_path = output_path / filename
        figure.savefig(str(file_path), dpi=150)
        plt.close(figure)
        saved.append(file_path)

    detection_index = None
    if any(task_name == "detection" for task_name, _ in runs_by_task_effort):
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

    for (task_name, effort), runs in sorted(runs_by_task_effort.items()):
        if task_name in QA_TASK_NAMES:
            accuracy = {
                run.model: sum(s.correct for s in run.samples) / len(run.samples) * 100
                for run in runs
            }
            figure = plot_accuracy_chart(
                accuracy,
                config,
                f"{task_name.title()} Benchmark",
            )
            save_figure(figure, f"{task_name}_accuracy_{effort}.png")

        elif task_name == "detection":
            if detection_index is None:
                continue

            from vlm_exam.tasks.detection import compute_dataset_map

            metrics: dict[str, dict[str, float]] = {
                "map50": {},
                "map75": {},
                "map50_95": {},
            }
            for run in runs:
                map_result = compute_dataset_map(run, detection_index)
                if map_result is None:
                    click.echo(
                        f"No valid predictions for {run.model} ({effort}); skipping."
                    )
                    continue
                metrics["map50"][run.model] = map_result.map50
                metrics["map75"][run.model] = map_result.map75
                metrics["map50_95"][run.model] = map_result.map50_95

            metric_titles = {
                "map50": "mAP@50",
                "map75": "mAP@75",
                "map50_95": "mAP@50:95",
            }
            for metric_key, values in metrics.items():
                if not values:
                    continue
                figure = plot_metric_chart(
                    values,
                    config,
                    f"Object Detection \u2014 {metric_titles[metric_key]}",
                    format_value=lambda value: f"{value * 100:.1f}%",
                    sort_ascending=False,
                    full_scale=1.0,
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
    help="Draw class labels on boxes, boxes only, or pick automatically.",
)
def detection_visualize(
    results_file: str,
    dataset_directory: str,
    output_directory: str,
    max_images: int,
    config_path: str | None,
    label_mode: str,
) -> None:
    """Visualize detection predictions vs ground truth."""
    import cv2
    import matplotlib

    matplotlib.use("Agg")

    from vlm_exam.tasks.detection import (
        DetectionTask,
        build_sample_index,
        detection_labels,
        parse_prediction,
    )
    from vlm_exam.visualization.detection import plot_detection_card

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

    count = 0
    for sample_result in run_result.samples:
        if count >= max_images:
            break

        sample = sample_by_image.get(sample_result.image)
        if sample is None:
            continue

        image = cv2.imread(sample.image_path)
        if image is None:
            continue

        resolution_wh = (sample.image_width, sample.image_height)
        predicted = parse_prediction(
            sample_result.predicted,
            resolution_wh,
            list(sample.classes),
            coordinate_format=sample_result.metadata.get(
                "coordinate_format", "normalized_1000"
            ),
        )

        gt_labels = detection_labels(sample.ground_truth, list(sample.classes))
        pred_labels = detection_labels(predicted, list(sample.classes))
        map_score = sample_result.metadata.get("map50")

        figure = plot_detection_card(
            image=image,
            ground_truth=sample.ground_truth,
            predictions=predicted,
            gt_labels=gt_labels,
            pred_labels=pred_labels,
            model_id=run_result.model,
            config=config,
            map_score=map_score,
            label_mode=label_mode,
        )
        _save_card(figure, output_path, count, sample_result.image)
        count += 1

    click.echo(f"Saved {count} visualizations to {output_path}")
