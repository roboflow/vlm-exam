# Copyright 2026 Roboflow, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
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

from vlm_exam.config import load_config
from vlm_exam.results import load_results

if TYPE_CHECKING:
    import matplotlib.pyplot as plt

_REFERENCE_RESULTS_DIRECTORY = (
    Path(__file__).resolve().parents[3] / "reference" / "results"
)


def _save_card(
    figure: plt.Figure, output_path: Path, index: int, image_name: str
) -> None:
    import matplotlib.pyplot as plt

    output_file = (output_path / f"{index:03d}_{image_name}").with_suffix(".png")
    figure.savefig(str(output_file), dpi=150)
    plt.close(figure)


def _resolve_reference_dataset_directory(
    results_path: Path,
    dataset_directory: str | None,
) -> Path:
    if dataset_directory is not None:
        return Path(dataset_directory)
    from vlm_exam.reference.manifest import resolve_dataset_directory_from_manifest

    try:
        return resolve_dataset_directory_from_manifest(results_path)
    except FileNotFoundError as error:
        raise click.UsageError(str(error)) from error
    except ValueError as error:
        raise click.UsageError(str(error)) from error


@click.command("reference-detection-visualize")
@click.option(
    "--results-file",
    default=None,
    type=click.Path(exists=True),
    help="Path to a reference detection result JSONL file.",
)
@click.option(
    "--baseline-file",
    default=None,
    type=click.Path(exists=True),
    help=(
        "Baseline run JSONL for best-of visualization (with --image-conditioned-file)."
    ),
)
@click.option(
    "--image-conditioned-file",
    default=None,
    type=click.Path(exists=True),
    help=(
        "Image-conditioned run JSONL for best-of visualization (with --baseline-file)."
    ),
)
@click.option(
    "--dataset-directory",
    default=None,
    type=click.Path(exists=True),
    help="Detection dataset directory; defaults to the run manifest path.",
)
@click.option(
    "--output-directory",
    required=True,
    type=click.Path(),
    help="Directory to save annotated images.",
)
@click.option(
    "--max-images",
    default=250,
    type=int,
    help="Maximum number of images to visualize.",
)
@click.option(
    "--reference-config",
    "reference_config_path",
    default=None,
    type=click.Path(exists=True),
    help="Path to custom reference_models.yaml config.",
)
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(exists=True),
    help="Path to custom models.yaml for lab branding.",
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
@click.option(
    "--confidence-threshold",
    default=None,
    type=float,
    help="Override post-hoc box filter for cards (mAP on cards stays native).",
)
@click.option(
    "--label-classes",
    default="canonical",
    type=click.Choice(["canonical", "augmented"]),
    help=(
        "Draw canonical dataset class names or augmented prompt text from run metadata."
    ),
)
def reference_detection_visualize(
    results_file: str | None,
    baseline_file: str | None,
    image_conditioned_file: str | None,
    dataset_directory: str | None,
    output_directory: str,
    max_images: int,
    reference_config_path: str | None,
    config_path: str | None,
    label_mode: str,
    output_format: str,
    image: str | None,
    sample_index: int | None,
    confidence_threshold: float | None,
    label_classes: str,
) -> None:
    """Visualize reference detection predictions vs ground truth."""
    import cv2

    from vlm_exam.reference.config import load_reference_config
    from vlm_exam.reference.visualization import (
        build_reference_card_config,
        detection_labels_for_card,
        prompt_label_map_from_metadata,
        resolve_card_confidence_threshold,
    )
    from vlm_exam.tasks.detection import (
        DetectionCoordinateFormat,
        DetectionTask,
        build_sample_index,
        filter_prediction_json,
        parse_prediction,
        recorded_uploaded_wh,
    )
    from vlm_exam.visualization.detection import (
        plot_detection_card,
        save_annotated_detection,
    )

    best_of = baseline_file is not None or image_conditioned_file is not None
    if best_of and (baseline_file is None or image_conditioned_file is None):
        raise click.UsageError(
            "--baseline-file and --image-conditioned-file must be given together."
        )
    if best_of == (results_file is not None):
        raise click.UsageError(
            "Provide either --results-file or the "
            "--baseline-file/--image-conditioned-file pair."
        )

    anchor_path = Path(results_file if results_file is not None else baseline_file)
    dataset_path = _resolve_reference_dataset_directory(
        anchor_path,
        dataset_directory,
    )

    task = DetectionTask()
    samples = task.load_samples(str(dataset_path))
    sample_by_image = build_sample_index(samples)

    if best_of:
        from vlm_exam.reference.best_prompt import merge_best_prompt_run

        try:
            merge_result = merge_best_prompt_run(
                load_results(Path(baseline_file)),
                load_results(Path(image_conditioned_file)),
                sample_by_image,
            )
        except ValueError as error:
            raise click.UsageError(str(error)) from error
        run_result = merge_result.merged_run
    else:
        run_result = load_results(anchor_path)
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    reference_config = load_reference_config(
        Path(reference_config_path) if reference_config_path else None
    )
    if run_result.model not in reference_config.models:
        click.echo(f"Reference model {run_result.model!r} not found in config.")
        return

    vlm_config = load_config(Path(config_path) if config_path else None)
    try:
        config = build_reference_card_config(
            run_result.model,
            reference_config,
            vlm_config,
        )
    except ValueError as error:
        click.echo(str(error))
        return

    use_card = output_format == "card"
    if use_card:
        import matplotlib

        matplotlib.use("Agg")

    box_confidence_threshold = resolve_card_confidence_threshold(
        run_result.model,
        reference_config,
        confidence_threshold,
    )
    if box_confidence_threshold is not None:
        click.echo(
            f"Drawing boxes with confidence >= {box_confidence_threshold}; "
            "card mAP uses native per-image scores."
        )
    else:
        click.echo("Drawing all stored boxes; card mAP uses native per-image scores.")

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
        prediction_text = sample_result.predicted
        if box_confidence_threshold is not None:
            prediction_text = filter_prediction_json(
                prediction_text,
                box_confidence_threshold,
            )
        predicted = parse_prediction(
            prediction_text,
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

        prompt_label_map = prompt_label_map_from_metadata(sample_result.metadata)
        if label_classes == "augmented" and prompt_label_map is None:
            click.echo(
                f"Skipping {sample_result.image}: no prompt_texts in run metadata."
            )
            continue

        pred_labels = detection_labels_for_card(
            predicted,
            list(sample.classes),
            label_classes=label_classes,
            prompt_label_map=prompt_label_map,
        )
        stem = f"{sample_result.index:03d}_{sample_result.image}"
        output_file = (output_path / stem).with_suffix(".png")

        if use_card:
            gt_labels = detection_labels_for_card(
                sample.ground_truth,
                list(sample.classes),
                label_classes=label_classes,
                prompt_label_map=prompt_label_map,
            )
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


@click.command("reference-run")
@click.option(
    "--model",
    "model_key",
    required=True,
    help="Reference model key from reference_models.yaml.",
)
@click.option(
    "--dataset-directory",
    required=True,
    type=click.Path(exists=True),
    help="Path to the detection dataset directory.",
)
@click.option(
    "--output-directory",
    default="reference/results",
    type=click.Path(),
    help="Directory to save reference result files.",
)
@click.option(
    "--reference-config",
    "reference_config_path",
    default=None,
    type=click.Path(exists=True),
    help="Path to custom reference_models.yaml config.",
)
@click.option(
    "--prompt-classes",
    "prompt_classes",
    default="image",
    type=click.Choice(["image", "all"]),
    help="List per-image ground-truth classes or all dataset classes.",
)
@click.option(
    "--device",
    default="auto",
    type=click.Choice(["auto", "mps", "cpu", "cuda"]),
    help="Inference device backend.",
)
@click.option(
    "--max-samples",
    "max_samples",
    default=None,
    type=int,
    help="Limit the number of samples to evaluate (default: all).",
)
@click.option(
    "--image",
    "image_filter",
    default=None,
    help="Only run inference for this image basename.",
)
@click.option(
    "--resume-file",
    "resume_file",
    default=None,
    type=click.Path(exists=True),
    help="Partial reference JSONL to resume.",
)
@click.option(
    "--prompt-set",
    "prompt_set_path",
    default=None,
    type=click.Path(exists=True),
    help="JSONL image-conditioned prompt asset.",
)
def reference_run(
    model_key: str,
    dataset_directory: str,
    output_directory: str,
    reference_config_path: str | None,
    prompt_classes: str,
    device: str,
    max_samples: int | None,
    image_filter: str | None,
    resume_file: str | None,
    prompt_set_path: str | None,
) -> None:
    """Run a local reference detection model on the benchmark dataset."""
    from vlm_exam.reference.config import load_reference_config
    from vlm_exam.reference.manifest import new_run_timestamp
    from vlm_exam.reference.prompts import load_prompt_set, validate_prompt_set_coverage
    from vlm_exam.reference.runner import run_reference_benchmark
    from vlm_exam.tasks.detection import DetectionTask

    reference_config = load_reference_config(
        Path(reference_config_path) if reference_config_path else None
    )
    if model_key not in reference_config.models:
        available = ", ".join(sorted(reference_config.models))
        raise click.UsageError(
            f"Unknown reference model {model_key!r}. Available models: {available}"
        )

    model_config = reference_config.models[model_key]
    output_path = Path(output_directory)
    if (
        max_samples is not None or image_filter is not None
    ) and output_path.resolve() == _REFERENCE_RESULTS_DIRECTORY.resolve():
        raise click.UsageError(
            "Partial runs cannot use reference/results. "
            "Pass --output-directory results-reference-smoke."
        )
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = new_run_timestamp()
    if resume_file is not None:
        timestamp = load_results(Path(resume_file)).timestamp
    filename = f"detection_{model_key}_reference_{timestamp}.jsonl"
    result_path = output_path / filename

    deviations = ["Mask outputs are ignored; evaluation uses boxes only."]
    if model_config.adapter == "yoloe":
        deviations.append(
            f"Confidence floor set to {model_config.inference.conf} "
            "for COCO-style evaluation."
        )
    elif model_config.adapter == "sam3":
        deviations.append(
            f"Instance segmentation threshold set to {model_config.inference.conf}."
        )
    if model_config.adapter == "yoloe" and model_config.inference.agnostic_nms is None:
        deviations.append("Checkpoint uses NMS-free end-to-end decoding.")

    prompt_set = None
    if prompt_set_path is not None:
        prompt_set = load_prompt_set(Path(prompt_set_path))
        task = DetectionTask(prompt_classes=prompt_classes)
        samples = task.load_samples(dataset_directory)
        required_pairs = [
            (os.path.basename(sample.image_path), class_name)
            for sample in samples
            for class_name in _prompt_classes_for_validation(sample, prompt_classes)
        ]
        coverage_errors = validate_prompt_set_coverage(
            prompt_set,
            all_classes=tuple(task.classes),
            required_pairs=required_pairs,
        )
        if coverage_errors:
            raise click.ClickException(" ".join(coverage_errors))

    run, manifest = run_reference_benchmark(
        model_config=model_config,
        dataset_directory=dataset_directory,
        output_path=result_path,
        timestamp=timestamp,
        prompt_classes=prompt_classes,
        device=device,
        max_samples=max_samples,
        image_filter=image_filter,
        resume_file=Path(resume_file) if resume_file else None,
        deviations=deviations,
        prompt_set=prompt_set,
    )
    click.echo(f"Reference results saved to {result_path}")
    click.echo(
        f"Completed {manifest.completed_sample_count} samples "
        f"({manifest.failed_sample_count} failed)."
    )
    click.echo(f"Manifest saved to {result_path.with_suffix('.manifest.json')}")


def _prompt_classes_for_validation(
    sample: object,
    prompt_classes: str,
) -> tuple[str, ...]:
    from vlm_exam.tasks.detection import DetectionSample

    assert isinstance(sample, DetectionSample)
    if (
        prompt_classes == "image"
        and sample.ground_truth.class_id is not None
        and len(sample.ground_truth) > 0
    ):
        present_ids = set(sample.ground_truth.class_id)
        return tuple(sample.classes[class_id] for class_id in sorted(present_ids))
    return sample.classes


@click.command("reference-validate")
@click.option(
    "--results-file",
    required=True,
    type=click.Path(exists=True),
    help="Path to a reference detection JSONL file.",
)
@click.option(
    "--dataset-directory",
    required=True,
    type=click.Path(exists=True),
    help="Path to the detection dataset directory.",
)
def reference_validate(results_file: str, dataset_directory: str) -> None:
    """Validate a reference detection run file."""
    from vlm_exam.reference.validate import validate_reference_run

    report = validate_reference_run(Path(results_file), dataset_directory)
    if report.ok:
        click.echo(f"Validation passed: {results_file}")
        return

    click.echo(f"Validation failed: {results_file}")
    for issue in report.issues:
        location = issue.image or "run"
        click.echo(f"  [{location}] {issue.message}")
    raise click.ClickException(
        f"Reference validation failed with {len(report.issues)} issue(s)."
    )


@click.command("reference-report")
@click.option(
    "--results-directory",
    default="results",
    type=click.Path(exists=True),
    help="Directory containing VLM detection JSONL files.",
)
@click.option(
    "--reference-results-directory",
    default="reference/results",
    type=click.Path(exists=True),
    help="Directory containing reference detection JSONL files.",
)
@click.option(
    "--dataset-directory",
    required=True,
    type=click.Path(exists=True),
    help="Path to the detection dataset directory.",
)
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(exists=True),
    help="Path to custom models.yaml config.",
)
def reference_report(
    results_directory: str,
    reference_results_directory: str,
    dataset_directory: str,
    config_path: str | None,
) -> None:
    """Compare VLM and reference detection mAP on the same dataset."""
    from vlm_exam.reference.report import (
        build_reference_report_rows,
        format_reference_report,
    )

    config = load_config(Path(config_path) if config_path else None)
    rows = build_reference_report_rows(
        Path(results_directory),
        Path(reference_results_directory),
        dataset_directory,
        config,
    )
    if not rows:
        click.echo("No detection runs found for comparison.")
        return
    click.echo(format_reference_report(rows))


@click.command("reference-experiment-report")
@click.option(
    "--baseline-directory",
    default="reference/results",
    type=click.Path(exists=True),
    help="Directory containing baseline reference runs.",
)
@click.option(
    "--experiment-directory",
    default="reference/results",
    type=click.Path(exists=True),
    help="Directory containing prompt-variant reference runs.",
)
@click.option(
    "--dataset-directory",
    required=True,
    type=click.Path(exists=True),
    help="Path to the detection dataset directory.",
)
def reference_experiment_report(
    baseline_directory: str,
    experiment_directory: str,
    dataset_directory: str,
) -> None:
    """Compare prompt-variant reference runs against baseline mAP."""
    from vlm_exam.reference.report import (
        build_prompt_experiment_rows,
        format_prompt_experiment_report,
    )

    rows = build_prompt_experiment_rows(
        Path(baseline_directory),
        Path(experiment_directory),
        dataset_directory,
    )
    if not rows:
        click.echo("No prompt experiment runs found.")
        return
    click.echo(format_prompt_experiment_report(rows))


@click.command("reference-detection-leaderboard")
@click.option(
    "--results-directory",
    default="results",
    type=click.Path(exists=True),
    help="Directory containing VLM detection JSONL runs.",
)
@click.option(
    "--dataset-directory",
    default="data/detection/train",
    type=click.Path(exists=True),
    help="Detection dataset directory.",
)
@click.option(
    "--output-directory",
    default="reference/leaderboards",
    type=click.Path(),
    help="Directory to save mixed leaderboard charts and tables.",
)
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(exists=True),
    help="Path to custom models.yaml config.",
)
@click.option(
    "--reference-config",
    "reference_config_path",
    default=None,
    type=click.Path(exists=True),
    help="Path to custom reference_models.yaml config.",
)
@click.option(
    "--repo-root",
    default=".",
    type=click.Path(exists=True),
    help="Repository root for resolving canonical reference result paths.",
)
def reference_detection_leaderboard(
    results_directory: str,
    dataset_directory: str,
    output_directory: str,
    config_path: str | None,
    reference_config_path: str | None,
    repo_root: str,
) -> None:
    """Generate a mixed VLM and reference detection leaderboard."""
    import json

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from vlm_exam.reference.config import load_reference_config
    from vlm_exam.reference.leaderboard import (
        LEADERBOARD_FAMILIES,
        YOLOE_GEMINI_FOCUS_CHART_TITLE,
        YOLOE_GEMINI_FOCUS_VLM,
        build_mixed_detection_leaderboard,
        build_mixed_leaderboard_config,
        chart_config_with_row_labels,
        format_mixed_detection_leaderboard_markdown,
        format_yoloe_gemini_focus_markdown,
        leaderboard_rows_for_family,
        leaderboard_rows_for_yoloe_gemini_focus,
        mixed_detection_leaderboard_payload,
    )
    from vlm_exam.tasks.detection import DetectionTask, build_sample_index
    from vlm_exam.visualization import plot_metric_chart

    vlm_config = load_config(Path(config_path) if config_path else None)
    reference_config = load_reference_config(
        Path(reference_config_path) if reference_config_path else None
    )
    task = DetectionTask()
    sample_index = build_sample_index(task.load_samples(dataset_directory))
    leaderboard = build_mixed_detection_leaderboard(
        Path(results_directory),
        sample_index,
        vlm_config,
        reference_config,
        repo_root=Path(repo_root),
    )
    chart_config = build_mixed_leaderboard_config(vlm_config, reference_config)

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    metric_specs = (
        ("map50", "mAP@50"),
        ("map75", "mAP@75"),
        ("map50_95", "mAP@50:95"),
    )
    family_titles = {
        "sam3": "Mixed detection leaderboard — SAM 3",
        "yoloe": "Mixed detection leaderboard — YOLO-E",
    }
    saved: list[Path] = []
    for family in LEADERBOARD_FAMILIES:
        family_leaderboard = leaderboard_rows_for_family(leaderboard, family)
        family_output = output_path / family
        family_output.mkdir(parents=True, exist_ok=True)
        for metric_key, metric_title in metric_specs:
            values = getattr(family_leaderboard, metric_key)
            if not values:
                continue
            figure = plot_metric_chart(
                values,
                chart_config,
                f"Object Detection — {metric_title}",
                format_value=lambda value: f"{value * 100:.1f}%",
                sort_ascending=False,
                full_scale=1.0,
            )
            file_path = family_output / f"detection_{metric_key}.png"
            figure.savefig(str(file_path), dpi=150)
            plt.close(figure)
            saved.append(file_path)

        markdown_path = family_output / "leaderboard.md"
        markdown_path.write_text(
            format_mixed_detection_leaderboard_markdown(
                family_leaderboard,
                title=family_titles[family],
            )
        )
        click.echo(f"{family}: {len(family_leaderboard.rows)} models")
        if family_leaderboard.rows:
            top = family_leaderboard.rows[0]
            click.echo(
                f"  Top mAP@50: {top.display_name} ({top.source}) = {top.map50:.4f}"
            )

    gemini_focus_output = output_path / "yoloe" / "gemini-focus"
    gemini_focus_output.mkdir(parents=True, exist_ok=True)
    class_names_leaderboard = leaderboard_rows_for_yoloe_gemini_focus(
        leaderboard,
        "class_names",
    )
    augmented_prompt_leaderboard = leaderboard_rows_for_yoloe_gemini_focus(
        leaderboard,
        "augmented_prompt",
    )
    for prompt, focus_leaderboard in (
        ("class_names", class_names_leaderboard),
        ("augmented_prompt", augmented_prompt_leaderboard),
    ):
        focus_chart_config = chart_config_with_row_labels(
            chart_config, focus_leaderboard
        )
        values = focus_leaderboard.map50
        if not values:
            continue
        figure = plot_metric_chart(
            values,
            focus_chart_config,
            YOLOE_GEMINI_FOCUS_CHART_TITLE[prompt],
            format_value=lambda value: f"{value * 100:.1f}%",
            sort_ascending=False,
            full_scale=1.0,
        )
        file_path = gemini_focus_output / f"detection_map50_{prompt}.png"
        figure.savefig(str(file_path), dpi=150)
        plt.close(figure)
        saved.append(file_path)
        click.echo(f"yoloe/gemini-focus/{prompt}: {len(focus_leaderboard.rows)} models")
    gemini_focus_markdown = gemini_focus_output / "leaderboard.md"
    gemini_focus_markdown.write_text(
        format_yoloe_gemini_focus_markdown(
            class_names_leaderboard,
            augmented_prompt_leaderboard,
            vlm_name=vlm_config.models[YOLOE_GEMINI_FOCUS_VLM].name,
        )
    )

    json_path = output_path / "leaderboard.json"
    json_path.write_text(
        json.dumps(mixed_detection_leaderboard_payload(leaderboard), indent=2) + "\n"
    )

    for file_path in saved:
        click.echo(f"  {file_path.relative_to(output_path)}")
    click.echo("  leaderboard.json")
    for family in LEADERBOARD_FAMILIES:
        click.echo(f"  {family}/leaderboard.md")
    click.echo("  yoloe/gemini-focus/leaderboard.md")
    click.echo(f"Wrote mixed leaderboard to {output_path}")


def register_reference_commands(group: click.Group) -> None:
    """Register reference-model commands on the main CLI group.

    Args:
        group: Click group receiving the reference commands.
    """
    for command in (
        reference_detection_visualize,
        reference_run,
        reference_validate,
        reference_report,
        reference_experiment_report,
        reference_detection_leaderboard,
    ):
        group.add_command(command)
