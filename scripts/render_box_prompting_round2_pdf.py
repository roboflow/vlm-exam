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
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from PIL import Image

from vlm_exam.box_prompting import (
    load_arm_records,
    load_case_image,
    record_to_detections,
)
from vlm_exam.box_prompting_round2 import (
    DISPLAY_NEGATIVE_HEX,
    DISPLAY_POSITIVE_HEX,
    DISPLAY_PREDICTION_HEX,
    ROUND2_ARMS,
    ExampleCase,
    overlay_panel,
    prompt_panel,
    run_round2_analysis,
    select_example_cases,
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
_PAGE_ARMS = (
    ("text_box_single", "text_box_multi"),
    ("drawn_box_single", "drawn_box_multi"),
)


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
) -> None:
    figure = plt.figure(figsize=_PAGE_SIZE)
    figure.suptitle("Qwen3.8-Max box-prompting round 2", fontsize=14, fontweight="bold")
    axis = figure.add_subplot(111)
    axis.axis("off")
    columns = [
        "Arm",
        "Mean img mAP@50",
        "Multiclass",
        "Single-class",
        "Errors",
        "Pos re-detected",
        "Neg hits",
        "Avg pred boxes",
        "Avg target boxes",
    ]

    def percent(value: float | None) -> str:
        return "n/a" if value is None else f"{value * 100:.1f}%"

    rows = []
    for arm in ROUND2_ARMS:
        if arm not in results:
            continue
        metrics = results[arm]["metrics"]
        rows.append(
            [
                arm,
                percent(metrics["mean_image_map50"]),
                percent(metrics["mean_image_map50_multiclass"]),
                percent(metrics["mean_image_map50_single_class"]),
                str(metrics["errors"]),
                str(metrics["positive_redetections"]),
                str(metrics["negative_hits"]),
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
    table.set_fontsize(8)
    table.scale(1.0, 1.6)
    legend = (
        "Each image gets two pages: text_box arms (single reference vs "
        "multi-example) and drawn_box arms. Per arm: prompt visualization "
        "(left), output (right), full prompt text below.\n\n"
        "Box colors: green = positive prompt, red = negative prompt, "
        "blue = prediction. Ground truth is not rendered. Note: the images "
        "actually sent to the model drew positives red and negatives blue, "
        "as named in the prompt text."
    )
    axis.text(
        0.5,
        0.5,
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
    arms: tuple[str, str],
    case: ExampleCase,
    sample: DetectionSample,
    records_by_arm: dict[str, dict[str, Any] | None],
    max_edge: int,
) -> None:
    image = load_case_image(sample)
    targets = target_detections(case)

    figure = plt.figure(figsize=_PAGE_SIZE)
    figure.suptitle(
        f"{case.image_name}\n"
        f"positive class: {case.class_name} | targets: {len(targets)} | "
        f"positives: {len(case.positive_xyxy)} | "
        f"negatives: {len(case.negative_xyxy)}",
        fontsize=9,
        fontweight="bold",
    )
    entries = [("positive prompt", DISPLAY_POSITIVE_HEX)]
    if case.negative_xyxy:
        entries.append(("negative prompt", DISPLAY_NEGATIVE_HEX))
    entries.append(("prediction", DISPLAY_PREDICTION_HEX))
    figure.legend(
        handles=[Patch(facecolor=color, label=label) for label, color in entries],
        loc="lower center",
        ncol=len(entries),
        fontsize=7,
        frameon=False,
    )
    grid = GridSpec(
        4,
        2,
        figure=figure,
        height_ratios=[1.0, 0.34, 1.0, 0.34],
        hspace=0.25,
        wspace=0.05,
        top=0.88,
        bottom=0.06,
        left=0.03,
        right=0.97,
    )

    for row, arm in enumerate(arms):
        record = records_by_arm.get(arm)
        input_axis = figure.add_subplot(grid[2 * row, 0])
        result_axis = figure.add_subplot(grid[2 * row, 1])
        prompt_axis = figure.add_subplot(grid[2 * row + 1, :])
        prompt_axis.axis("off")

        _show_image(input_axis, prompt_panel(arm, image, case), max_edge)
        input_axis.set_title(f"{arm} — prompt", fontsize=8)

        if record is None:
            _show_message(result_axis, "no record collected")
            continue

        prompt_axis.text(
            0.0,
            1.0,
            textwrap.fill(record.get("prompt", ""), width=170),
            horizontalalignment="left",
            verticalalignment="top",
            fontsize=5.4,
            family="monospace",
        )

        if record.get("error") is not None:
            _show_message(result_axis, f"error: {record['error']}")
            continue

        detections, failed = record_to_detections(record, sample)
        map50 = compute_image_map50(detections, targets)
        _show_image(result_axis, overlay_panel(arm, image, case, detections), max_edge)
        status = " | parse failed" if failed else ""
        result_axis.set_title(
            f"output — mAP@50 {map50 * 100:.0f}% | "
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
    default=Path("results-box-prompting-qwen38-max-round2"),
    show_default=True,
)
@click.option(
    "--image-count", type=click.IntRange(min=1), default=50, show_default=True
)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option(
    "--max-edge", type=click.IntRange(min=200), default=1400, show_default=True
)
def main(
    dataset_directory: Path,
    output_directory: Path,
    image_count: int,
    seed: int,
    max_edge: int,
) -> None:
    """Render the round-2 examination PDF from collected raw data."""
    sample_index = build_sample_index(
        DetectionTask().load_samples(str(dataset_directory))
    )
    cases = select_example_cases(sample_index, count=image_count, seed=seed)
    cases_by_image = {case.image_name: case for case in cases}
    raw_directory = output_directory / "raw"

    records: dict[str, dict[str, dict[str, Any]]] = {
        arm: {
            record["image"]: record for record in load_arm_records(raw_directory, arm)
        }
        for arm in ROUND2_ARMS
    }
    results = run_round2_analysis(
        raw_directory=raw_directory,
        cases_by_image=cases_by_image,
        sample_index=sample_index,
    )

    pdf_path = output_directory / "analysis" / "box_prompting_round2_report.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(pdf_path) as pdf:
        _render_summary_page(pdf, results)
        for case in sorted(cases, key=lambda item: item.image_name):
            for arms in _PAGE_ARMS:
                _render_case_page(
                    pdf,
                    arms=arms,
                    case=case,
                    sample=sample_index[case.image_name],
                    records_by_arm={
                        arm: records[arm].get(case.image_name) for arm in arms
                    },
                    max_edge=max_edge,
                )
    click.echo(f"PDF written to {pdf_path}")


if __name__ == "__main__":
    main()
