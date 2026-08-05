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

import textwrap
from pathlib import Path
from typing import Any

import click
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import supervision as sv
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from PIL import Image, ImageDraw

from vlm_exam.box_prompting import (
    ReferenceCase,
    crop_reference,
    draw_reference_box,
    load_arm_records,
    load_case_image,
    record_to_detections,
    run_analysis,
    select_cases,
    target_detections,
)
from vlm_exam.providers.image_upload import resize_image_to_max_edge
from vlm_exam.tasks.detection import (
    DetectionSample,
    DetectionTask,
    build_sample_index,
    compute_image_map50,
)

_PAGE_SIZE = (11.69, 8.27)
_REFERENCE_COLOR = (255, 0, 0)
_TARGET_COLOR = (0, 200, 83)
_PREDICTION_COLOR = (41, 98, 255)


def _line_width(image: Image.Image) -> int:
    return max(3, round(max(image.size) / 300))


def _input_panel(
    arm: str,
    image: Image.Image,
    case: ReferenceCase,
) -> Image.Image:
    if arm == "drawn_box":
        return draw_reference_box(image, case.reference_xyxy)
    if arm == "crop":
        return crop_reference(image, case.reference_xyxy)
    return image


def _overlay_panel(
    image: Image.Image,
    case: ReferenceCase,
    detections: sv.Detections,
) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    width = _line_width(image)
    for box in case.target_xyxy:
        draw.rectangle(box, outline=_TARGET_COLOR, width=width)
    for box in detections.xyxy:
        predicted = tuple(float(value) for value in box)
        draw.rectangle(predicted, outline=_PREDICTION_COLOR, width=width)
    draw.rectangle(case.reference_xyxy, outline=_REFERENCE_COLOR, width=width)
    return annotated


def _show_image(axis: plt.Axes, image: Image.Image, max_edge: int) -> None:
    axis.imshow(np.asarray(resize_image_to_max_edge(image, max_edge)))
    axis.axis("off")


def _show_message(axis: plt.Axes, message: str) -> None:
    axis.text(
        0.5,
        0.5,
        textwrap.fill(message, width=60),
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=8,
        color="firebrick",
    )
    axis.axis("off")


def _render_summary_page(
    pdf: PdfPages,
    results: dict[str, dict[str, Any]],
    arms: list[str],
) -> None:
    figure = plt.figure(figsize=_PAGE_SIZE)
    figure.suptitle(
        "Qwen3.8-Max box-prompting experiment", fontsize=14, fontweight="bold"
    )
    axis = figure.add_subplot(111)
    axis.axis("off")
    columns = [
        "Arm",
        "Mean img mAP@50",
        "Parse fail",
        "Errors",
        "Ref re-detected",
        "Avg pred boxes",
        "Avg target boxes",
    ]
    rows = []
    for arm in arms:
        if arm not in results:
            continue
        metrics = results[arm]["metrics"]
        rows.append(
            [
                arm,
                f"{metrics['mean_image_map50'] * 100:.1f}%",
                str(metrics["parse_failures"]),
                str(metrics["errors"]),
                str(metrics["reference_redetections"]),
                f"{metrics['mean_predicted_boxes']:.1f}",
                f"{metrics['mean_target_boxes']:.1f}",
            ]
        )
    table = axis.table(
        cellText=rows,
        colLabels=columns,
        loc="upper center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.6)
    legend = (
        "Each following page shows one image: per arm, the input as sent to "
        "the model (left), the prediction overlay (right), and the full "
        "prompt text.\n\n"
        "Box colors: red = reference box, green = ground-truth targets "
        "(other instances of the reference class), blue = model predictions."
    )
    axis.text(
        0.5,
        0.45,
        legend,
        horizontalalignment="center",
        verticalalignment="top",
        fontsize=10,
        wrap=True,
    )
    pdf.savefig(figure)
    plt.close(figure)


def _render_case_page(
    pdf: PdfPages,
    *,
    case: ReferenceCase,
    sample: DetectionSample,
    records_by_arm: dict[str, dict[str, Any] | None],
    max_edge: int,
) -> None:
    image = load_case_image(sample)
    targets = target_detections(case)
    arm_count = len(records_by_arm)

    figure = plt.figure(figsize=_PAGE_SIZE)
    figure.suptitle(
        f"{case.image_name}\n"
        f"reference class: {case.class_name} | targets: {len(targets)}",
        fontsize=9,
        fontweight="bold",
    )
    grid = GridSpec(
        2 * arm_count,
        2,
        figure=figure,
        height_ratios=[1.0, 0.34] * arm_count,
        hspace=0.25,
        wspace=0.05,
        top=0.90,
        bottom=0.03,
        left=0.03,
        right=0.97,
    )

    for row, (arm, record) in enumerate(records_by_arm.items()):
        input_axis = figure.add_subplot(grid[2 * row, 0])
        result_axis = figure.add_subplot(grid[2 * row, 1])
        prompt_axis = figure.add_subplot(grid[2 * row + 1, :])
        prompt_axis.axis("off")

        _show_image(input_axis, _input_panel(arm, image, case), max_edge)
        input_axis.set_title(f"{arm} — input sent", fontsize=8)

        if record is None:
            _show_message(result_axis, "no record collected")
            continue

        prompt_axis.text(
            0.0,
            1.0,
            textwrap.fill(record.get("prompt", ""), width=170),
            horizontalalignment="left",
            verticalalignment="top",
            fontsize=5.8,
            family="monospace",
        )

        if record.get("error") is not None:
            _show_message(result_axis, f"error: {record['error']}")
            continue

        detections, failed = record_to_detections(record, sample)
        map50 = compute_image_map50(detections, targets)
        _show_image(result_axis, _overlay_panel(image, case, detections), max_edge)
        status = " | parse failed" if failed else ""
        result_axis.set_title(
            f"prediction — mAP@50 {map50 * 100:.0f}% | "
            f"predicted {len(detections)} / targets {len(targets)}{status}",
            fontsize=8,
        )

    pdf.savefig(figure)
    plt.close(figure)


@click.command()
@click.option(
    "--dataset-directory",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("data/detection/train"),
    show_default=True,
)
@click.option(
    "--output-directory",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("results-box-prompting-qwen38-max"),
    show_default=True,
)
@click.option("--arms", default="text_box,drawn_box", show_default=True)
@click.option(
    "--image-count", type=click.IntRange(min=1), default=25, show_default=True
)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option(
    "--max-edge", type=click.IntRange(min=200), default=1400, show_default=True
)
def main(
    dataset_directory: Path,
    output_directory: Path,
    arms: str,
    image_count: int,
    seed: int,
    max_edge: int,
) -> None:
    """Render the box-prompting examination PDF from collected raw data."""
    arm_list = [arm.strip() for arm in arms.split(",") if arm.strip()]
    sample_index = build_sample_index(
        DetectionTask().load_samples(str(dataset_directory))
    )
    cases = select_cases(sample_index, count=image_count, seed=seed)
    cases_by_image = {case.image_name: case for case in cases}
    raw_directory = output_directory / "raw"

    records: dict[str, dict[str, dict[str, Any]]] = {
        arm: {
            record["image"]: record for record in load_arm_records(raw_directory, arm)
        }
        for arm in arm_list
    }
    results = run_analysis(
        raw_directory=raw_directory,
        cases_by_image=cases_by_image,
        sample_index=sample_index,
    )

    pdf_path = output_directory / "analysis" / "box_prompting_report.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(pdf_path) as pdf:
        _render_summary_page(pdf, results, arm_list)
        for case in sorted(cases, key=lambda item: item.image_name):
            _render_case_page(
                pdf,
                case=case,
                sample=sample_index[case.image_name],
                records_by_arm={
                    arm: records[arm].get(case.image_name) for arm in arm_list
                },
                max_edge=max_edge,
            )
    click.echo(f"PDF written to {pdf_path}")


if __name__ == "__main__":
    main()
