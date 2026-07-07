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

from pathlib import Path

import click
from dotenv import load_dotenv

from vlm_exam.config import load_config
from vlm_exam.judge import Judge
from vlm_exam.providers import create_provider
from vlm_exam.results import RunResult, load_results, save_results
from vlm_exam.runner import run_benchmark
from vlm_exam.tasks import create_task

load_dotenv()


@click.group()
def main() -> None:
    """vlm-exam: Benchmark suite for Vision Language Models."""


@main.command()
@click.option(
    "--task",
    "task_name",
    required=True,
    help="Task to run (e.g. vqa).",
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

        result = run_benchmark(
            task=task,
            provider=provider,
            samples=samples,
            effort=effort,
            task_name=task_name,
            match_mode=match_mode,
            judge=judge,
        )

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
    results_path = Path(results_directory)
    result_files = sorted(results_path.glob("*.jsonl"))

    if not result_files:
        click.echo(f"No .jsonl files found in {results_directory}")
        return

    runs: list[RunResult] = []
    for file_path in result_files:
        try:
            runs.append(load_results(file_path))
        except ValueError:
            click.echo(f"Skipping empty file: {file_path}")

    click.echo(
        f"\n{'Model':<25} {'Effort':>6} {'Correct':>8} {'Total':>6} {'Accuracy':>9}"
    )
    click.echo("-" * 60)

    for run_result in runs:
        correct = sum(sample.correct for sample in run_result.samples)
        total = len(run_result.samples)
        accuracy = correct / total * 100 if total > 0 else 0.0

        click.echo(
            f"{run_result.model:<25} {run_result.effort:>6} "
            f"{correct:>8} {total:>6} {accuracy:>8.1f}%"
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

    results_path = Path(results_directory)
    result_files = sorted(results_path.glob("detection_*.jsonl"))

    if not result_files:
        click.echo(f"No detection result files found in {results_directory}")
        return

    for file_path in result_files:
        try:
            run_result = load_results(file_path)
        except ValueError:
            click.echo(f"Skipping empty file: {file_path}")
            continue

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
    result_files = sorted(results_path.glob("*.jsonl"))

    if not result_files:
        click.echo(f"No .jsonl files found in {results_directory}")
        return

    latest_runs: dict[tuple[str, str, str], RunResult] = {}
    for file_path in result_files:
        try:
            run_result = load_results(file_path)
        except ValueError:
            click.echo(f"Skipping empty file: {file_path}")
            continue
        if run_result.model not in config.models:
            click.echo(
                f"Skipping run with unknown model {run_result.model!r}: {file_path}"
            )
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
        if task_name == "vqa":
            accuracy = {
                run.model: sum(s.correct for s in run.samples) / len(run.samples) * 100
                for run in runs
            }
            figure = plot_accuracy_chart(
                accuracy,
                config,
                f"VQA / OCR Benchmark \u2014 {effort.title()} Effort",
            )
            save_figure(figure, f"vqa_accuracy_{effort}.png")

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
                    f"Object Detection \u2014 {metric_titles[metric_key]} "
                    f"({effort.title()} Effort)",
                    format_value=lambda value: f"{value:.3f}",
                    sort_ascending=False,
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
    type=click.Choice(["auto", "labels", "legend"]),
    help="Draw class labels on boxes, use a color legend, or pick automatically.",
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

    import matplotlib.pyplot as plt

    from vlm_exam.tasks.detection import (
        DetectionTask,
        build_sample_index,
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
            sample_result.predicted, resolution_wh, list(sample.classes)
        )

        gt_labels = (
            [sample.classes[cid] for cid in sample.ground_truth.class_id]
            if sample.ground_truth.class_id is not None
            else []
        )

        pred_labels = []
        if "class_name" in predicted.data:
            pred_labels = list(predicted.data["class_name"])
        elif predicted.class_id is not None:
            pred_labels = [sample.classes[cid] for cid in predicted.class_id]

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

        output_file = output_path / f"{count:03d}_{sample_result.image}"
        output_file = output_file.with_suffix(".png")
        figure.savefig(str(output_file), dpi=150)
        plt.close(figure)
        count += 1

    click.echo(f"Saved {count} visualizations to {output_path}")
