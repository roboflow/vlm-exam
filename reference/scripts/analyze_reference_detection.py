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

import csv
from pathlib import Path

import click
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from vlm_exam.reference.analysis import (
    ReferenceAnalysisReport,
    build_reference_analysis_report,
)
from vlm_exam.reference.best_prompt import BestPromptMergeResult, merge_best_prompt_run
from vlm_exam.reference.constants import CANONICAL_BEST_PAIRS, REFERENCE_EFFORT
from vlm_exam.reference.manifest import load_manifest, manifest_path_for_results
from vlm_exam.results import load_results
from vlm_exam.tasks.detection import DetectionTask, build_sample_index


def _load_name_difficulty(config_path: Path | None) -> dict[str, str]:
    if config_path is None or not config_path.exists():
        return {}
    with open(config_path) as file:
        raw = yaml.safe_load(file) or {}
    if not isinstance(raw, dict):
        return {}
    return {str(key): str(value) for key, value in raw.items()}


def _write_per_class_csv(report: ReferenceAnalysisReport, output_path: Path) -> None:
    fieldnames = [
        "class_name",
        "class_id",
        "ap50",
        "ap50_95",
        "ground_truth_count",
        "prediction_count",
        "prediction_count_conf025",
        "recall",
        "mean_matched_iou",
        "false_positive_count",
        "failure_mode",
        "low_support",
        "name_difficulty",
    ]
    with open(output_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in report.per_class:
            writer.writerow(
                {
                    "class_name": row.class_name,
                    "class_id": row.class_id,
                    "ap50": f"{row.ap50:.4f}",
                    "ap50_95": f"{row.ap50_95:.4f}",
                    "ground_truth_count": row.ground_truth_count,
                    "prediction_count": row.prediction_count,
                    "prediction_count_conf025": row.prediction_count_conf025,
                    "recall": f"{row.recall:.4f}",
                    "mean_matched_iou": f"{row.mean_matched_iou:.4f}",
                    "false_positive_count": row.false_positive_count,
                    "failure_mode": row.failure_mode.value,
                    "low_support": row.low_support,
                    "name_difficulty": row.name_difficulty,
                }
            )


def _write_summary_markdown(report: ReferenceAnalysisReport, output_path: Path) -> None:
    zero_ap = sum(1 for row in report.per_class if row.ap50 <= 0.001)
    never_predicted = sum(
        1 for row in report.per_class if row.failure_mode.value == "never-predicted"
    )
    lines = [
        f"# Reference analysis: {report.model}",
        "",
        "## Aggregate metrics",
        "",
        f"- mAP@50: {report.map50:.4f}",
        f"- mAP@75: {report.map75:.4f}",
        f"- mAP@50:95: {report.map50_95:.4f}",
        f"- Recall@IoU0.5 (class-aware): {report.recall_class_aware:.4f}",
        f"- Recall@IoU0.5 (class-agnostic): {report.recall_class_agnostic:.4f}",
        f"- Classes with zero AP@50: {zero_ap} / {len(report.per_class)}",
        f"- Classes never predicted: {never_predicted}",
        "",
        "## Confidence summary",
        "",
        f"- True-positive median confidence: "
        f"{report.confidence_summary['true_positive_median']:.4f}",
        f"- False-positive median confidence: "
        f"{report.confidence_summary['false_positive_median']:.4f}",
        "",
        "## Object-count buckets",
        "",
        "| Bucket | Images | mAP@50 | Recall (aware) | Recall (agnostic) | FN | FP |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report.object_count_buckets:
        lines.append(
            f"| {row.bucket} | {row.image_count} | {row.map50:.3f} | "
            f"{row.recall_class_aware:.3f} | {row.recall_class_agnostic:.3f} | "
            f"{row.false_negative_count} | {row.false_positive_count} |"
        )

    lines.extend(
        [
            "",
            "## Top confusion pairs",
            "",
            "| Ground truth | Predicted as | Count |",
            "| --- | --- | ---: |",
        ]
    )
    for pair in report.confusion_pairs[:15]:
        lines.append(
            f"| {pair.ground_truth_class} | {pair.predicted_class} | {pair.count} |"
        )

    lines.extend(["", "## Best classes (AP@50)", ""])
    for row in report.per_class[:10]:
        lines.append(
            f"- {row.class_name}: AP@50={row.ap50:.3f}, recall={row.recall:.3f}, "
            f"gt={row.ground_truth_count}"
        )

    lines.extend(["", "## Worst classes (AP@50, gt >= 5)", ""])
    stable_worst = [
        row
        for row in reversed(report.per_class)
        if not row.low_support and row.ap50 <= 0.1
    ][:10]
    for row in stable_worst:
        lines.append(
            f"- {row.class_name}: AP@50={row.ap50:.3f}, mode={row.failure_mode.value}, "
            f"gt={row.ground_truth_count}"
        )

    output_path.write_text("\n".join(lines) + "\n")


def _write_selection_markdown(
    merge_result: BestPromptMergeResult,
    output_path: Path,
) -> None:
    image_count = len(merge_result.selections)
    lines = [
        "# Best-prompt selection summary",
        "",
        "Per-image oracle: keep baseline or image-conditioned predictions "
        "whichever scores higher native mAP@50. Ties keep baseline.",
        "",
        f"- Images: {image_count}",
        f"- Baseline wins: {merge_result.baseline_wins}",
        f"- Image-conditioned wins: {merge_result.image_conditioned_wins}",
        f"- Ties (baseline kept): {merge_result.ties}",
        "",
        "| Image | Baseline mAP@50 | Image-conditioned mAP@50 | Selected |",
        "| --- | ---: | ---: | --- |",
    ]
    for selection in merge_result.selections:
        lines.append(
            f"| {selection.image} | {selection.baseline_map50:.4f} | "
            f"{selection.image_conditioned_map50:.4f} | {selection.selected} |"
        )
    output_path.write_text("\n".join(lines) + "\n")


def _write_charts(report: ReferenceAnalysisReport, output_directory: Path) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)

    ap_values = [row.ap50 for row in report.per_class]
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.hist(ap_values, bins=20, edgecolor="black")
    axis.set_xlabel("AP@50")
    axis.set_ylabel("Class count")
    axis.set_title(f"{report.model}: AP@50 distribution")
    figure.tight_layout()
    figure.savefig(output_directory / "ap50_histogram.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.scatter(
        [row.ground_truth_count for row in report.per_class],
        [row.ap50 for row in report.per_class],
        alpha=0.7,
    )
    axis.set_xlabel("Ground-truth count")
    axis.set_ylabel("AP@50")
    axis.set_title(f"{report.model}: AP@50 vs support")
    figure.tight_layout()
    figure.savefig(output_directory / "ap50_vs_support.png", dpi=150)
    plt.close(figure)

    difficulty_groups: dict[str, list[float]] = {}
    for row in report.per_class:
        difficulty_groups.setdefault(row.name_difficulty, []).append(row.ap50)
    if difficulty_groups:
        labels = sorted(difficulty_groups)
        means = [
            sum(values) / len(values)
            for values in (difficulty_groups[label] for label in labels)
        ]
        figure, axis = plt.subplots(figsize=(8, 4))
        axis.bar(labels, means)
        axis.set_ylabel("Mean AP@50")
        axis.set_title(f"{report.model}: mean AP@50 by name difficulty")
        figure.tight_layout()
        figure.savefig(output_directory / "ap50_by_name_difficulty.png", dpi=150)
        plt.close(figure)

    bucket_labels = [row.bucket for row in report.object_count_buckets]
    bucket_map50 = [row.map50 for row in report.object_count_buckets]
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.bar(bucket_labels, bucket_map50)
    axis.set_ylabel("mAP@50")
    axis.set_title(f"{report.model}: mAP@50 by object-count bucket")
    figure.tight_layout()
    figure.savefig(output_directory / "map50_by_object_count.png", dpi=150)
    plt.close(figure)


def _latest_reference_runs(results_directory: Path) -> dict[str, Path]:
    latest: dict[str, Path] = {}
    for path in sorted(results_directory.glob("detection_*.jsonl")):
        run = load_results(path)
        if run.effort != REFERENCE_EFFORT:
            continue
        manifest_path = manifest_path_for_results(path)
        if (
            manifest_path.exists()
            and load_manifest(manifest_path).prompt_asset_type != "none"
        ):
            continue
        existing = latest.get(run.model)
        if existing is None or run.timestamp > load_results(existing).timestamp:
            latest[run.model] = path
    return latest


def _run_output_subdirectory(results_file: Path) -> str:
    manifest_path = manifest_path_for_results(results_file)
    if manifest_path.exists() and (
        load_manifest(manifest_path).prompt_asset_type == "image_conditioned"
    ):
        return "image-conditioned"
    return "baseline"


def _analyze_run(
    results_file: Path,
    sample_index: dict,
    name_difficulty: dict[str, str],
    output_directory: Path,
    *,
    min_confidence: float | None = None,
) -> ReferenceAnalysisReport:
    run = load_results(results_file)
    report = build_reference_analysis_report(
        run,
        sample_index,
        name_difficulty=name_difficulty,
        min_confidence=min_confidence,
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    _write_per_class_csv(report, output_directory / "per_class.csv")
    _write_summary_markdown(report, output_directory / "summary.md")
    _write_charts(report, output_directory / "charts")
    return report


def _analyze_best_of(
    baseline_file: Path,
    image_conditioned_file: Path,
    sample_index: dict,
    name_difficulty: dict[str, str],
    output_directory: Path,
) -> tuple[ReferenceAnalysisReport, BestPromptMergeResult]:
    baseline_run = load_results(baseline_file)
    image_conditioned_run = load_results(image_conditioned_file)
    merge_result = merge_best_prompt_run(
        baseline_run,
        image_conditioned_run,
        sample_index,
    )
    report = build_reference_analysis_report(
        merge_result.merged_run,
        sample_index,
        name_difficulty=name_difficulty,
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    _write_per_class_csv(report, output_directory / "per_class.csv")
    _write_summary_markdown(report, output_directory / "summary.md")
    _write_selection_markdown(merge_result, output_directory / "selection.md")
    _write_charts(report, output_directory / "charts")
    return report, merge_result


@click.command()
@click.option(
    "--results-directory",
    default="reference/results",
    type=click.Path(exists=True),
    help="Directory containing reference detection JSONL runs.",
)
@click.option(
    "--dataset-directory",
    default="data/detection/train",
    type=click.Path(exists=True),
    help="Detection dataset directory.",
)
@click.option(
    "--output-directory",
    default="visualizations/reference-analysis",
    type=click.Path(),
    help="Base directory for analysis outputs.",
)
@click.option(
    "--results-file",
    default=None,
    type=click.Path(exists=True),
    help="Analyze one specific results JSONL instead of latest per model.",
)
@click.option(
    "--name-difficulty-config",
    default="src/vlm_exam/reference/configs/class_name_difficulty.yaml",
    type=click.Path(),
    help="YAML mapping class names to difficulty tags.",
)
@click.option(
    "--confidence-threshold",
    default=None,
    type=float,
    help="Drop predictions below this confidence before computing metrics.",
)
@click.option(
    "--best-of",
    is_flag=True,
    default=False,
    help="Merge baseline and image-conditioned runs with per-image oracle selection.",
)
@click.option(
    "--best-all-models",
    is_flag=True,
    default=False,
    help="Run oracle best-prompt analysis for all canonical model pairs.",
)
@click.option(
    "--baseline-file",
    default=None,
    type=click.Path(exists=True),
    help="Baseline results JSONL for --best-of.",
)
@click.option(
    "--image-conditioned-file",
    default=None,
    type=click.Path(exists=True),
    help="Image-conditioned results JSONL for --best-of.",
)
def main(
    results_directory: str,
    dataset_directory: str,
    output_directory: str,
    results_file: str | None,
    name_difficulty_config: str,
    confidence_threshold: float | None,
    best_of: bool,
    best_all_models: bool,
    baseline_file: str | None,
    image_conditioned_file: str | None,
) -> None:
    """Analyze reference detection runs with per-class and bucketed metrics."""
    task = DetectionTask()
    samples = task.load_samples(dataset_directory)
    sample_index = build_sample_index(samples)
    difficulty = _load_name_difficulty(Path(name_difficulty_config))
    output_root = Path(output_directory)

    if best_all_models or best_of:
        if confidence_threshold is not None:
            raise click.ClickException(
                "--confidence-threshold is not supported with --best-of."
            )
        if results_file is not None:
            raise click.ClickException(
                "--results-file cannot be combined with --best-of."
            )

        pairs: list[tuple[str, Path, Path, Path]] = []
        if best_all_models:
            for model, baseline_path, image_conditioned_path in CANONICAL_BEST_PAIRS:
                pairs.append(
                    (
                        model,
                        Path(baseline_path),
                        Path(image_conditioned_path),
                        output_root / model / "best",
                    )
                )
        else:
            if baseline_file is None or image_conditioned_file is None:
                raise click.ClickException(
                    "--best-of requires --baseline-file and --image-conditioned-file."
                )
            baseline_path = Path(baseline_file)
            image_conditioned_path = Path(image_conditioned_file)
            model = load_results(baseline_path).model
            pairs.append(
                (
                    model,
                    baseline_path,
                    image_conditioned_path,
                    output_root / model / "best",
                )
            )

        for model, baseline_path, image_conditioned_path, run_output in pairs:
            report, merge_result = _analyze_best_of(
                baseline_path,
                image_conditioned_path,
                sample_index,
                difficulty,
                run_output,
            )
            click.echo(
                f"{model} best: mAP@50={report.map50:.4f}, "
                f"mAP@50:95={report.map50_95:.4f}, "
                f"recall aware={report.recall_class_aware:.4f}, "
                f"agnostic={report.recall_class_agnostic:.4f} "
                f"(baseline wins={merge_result.baseline_wins}, "
                f"image-conditioned wins={merge_result.image_conditioned_wins}, "
                f"ties={merge_result.ties})"
            )
            click.echo(f"Wrote analysis to {run_output}")
        return

    if results_file is not None:
        path = Path(results_file)
        model = load_results(path).model
        setting = _run_output_subdirectory(path)
        run_output = output_root / model / setting
        if confidence_threshold is not None:
            threshold_label = f"conf{int(round(confidence_threshold * 100)):03d}"
            run_output = output_root / model / threshold_label / setting
        report = _analyze_run(
            path,
            sample_index,
            difficulty,
            run_output,
            min_confidence=confidence_threshold,
        )
        threshold_note = (
            f" (conf>={confidence_threshold})"
            if confidence_threshold is not None
            else ""
        )
        click.echo(
            f"{model}{threshold_note}: mAP@50={report.map50:.4f}, "
            f"recall aware={report.recall_class_aware:.4f}, "
            f"agnostic={report.recall_class_agnostic:.4f}"
        )
        click.echo(f"Wrote analysis to {run_output}")
        return

    latest_runs = _latest_reference_runs(Path(results_directory))
    if not latest_runs:
        raise click.ClickException(f"No reference runs found in {results_directory}")

    for model, path in sorted(latest_runs.items()):
        run_output = output_root / model
        report = _analyze_run(
            path,
            sample_index,
            difficulty,
            run_output,
            min_confidence=confidence_threshold,
        )
        threshold_note = (
            f" (conf>={confidence_threshold})"
            if confidence_threshold is not None
            else ""
        )
        click.echo(
            f"{model}{threshold_note}: mAP@50={report.map50:.4f}, "
            f"recall aware={report.recall_class_aware:.4f}, "
            f"agnostic={report.recall_class_agnostic:.4f}"
        )
    click.echo(f"Wrote analysis to {output_root}")


if __name__ == "__main__":
    main()
