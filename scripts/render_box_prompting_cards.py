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
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import supervision as sv
from PIL import Image, ImageColor

from vlm_exam.box_prompting import (
    load_arm_records,
    load_case_image,
    record_to_detections,
)
from vlm_exam.box_prompting_common import SUPPORTED_MODELS
from vlm_exam.box_prompting_cross import (
    build_cross_case,
    load_cross_records,
    record_target_detections,
)
from vlm_exam.box_prompting_groups import build_image_groups
from vlm_exam.box_prompting_round2 import (
    DISPLAY_NEGATIVE_HEX,
    DISPLAY_POSITIVE_HEX,
    DISPLAY_PREDICTION_HEX,
    ROUND2_ARMS,
    annotate_box_groups,
    build_example_case,
    target_detections,
)
from vlm_exam.config import BenchmarkConfig, load_config
from vlm_exam.tasks.detection import (
    DetectionSample,
    DetectionTask,
    build_sample_index,
    compute_image_map50,
)
from vlm_exam.visualization.theme import (
    CARD_FIGURE_SIZE,
    DIVIDER_COLOR,
    PANEL_LABEL_COLOR,
    add_top_accent,
    draw_brand_footer,
    draw_identity_row,
    draw_image_stage,
    load_fonts,
    score_color,
)

_DEFAULT_ARMS = ("drawn_box_single", "drawn_box_multi")
_CROSS_ARM = "pairwise"
_PROMPT_RECT = (0.03, 0.27, 0.462, 0.52)
_TARGET_RECT = (0.508, 0.27, 0.462, 0.52)
_CAPTION_Y = 0.845
_SCORE_DIVIDER_Y = 0.235
_SCORE_LABEL_Y = 0.198
_SCORE_VALUE_Y = 0.138

_Box = tuple[float, float, float, float]


def _fit_rect(
    rect: tuple[float, float, float, float],
    image_size: tuple[int, int],
) -> tuple[float, float, float, float]:
    left, bottom, width, height = rect
    rect_width_inches = width * CARD_FIGURE_SIZE[0]
    rect_height_inches = height * CARD_FIGURE_SIZE[1]
    image_width, image_height = image_size
    scale = min(rect_width_inches / image_width, rect_height_inches / image_height)
    drawn_width = image_width * scale / CARD_FIGURE_SIZE[0]
    drawn_height = image_height * scale / CARD_FIGURE_SIZE[1]
    return (
        left + (width - drawn_width) / 2,
        bottom + (height - drawn_height) / 2,
        drawn_width,
        drawn_height,
    )


def _panel_center_x(rect: tuple[float, float, float, float]) -> float:
    left, _, width, _ = rect
    return (left + width / 2 - 0.03) / 0.94


def _annotated_boxes(
    image: Image.Image,
    groups: list[tuple[tuple[_Box, ...], str]],
) -> Image.Image:
    converted = [(boxes, ImageColor.getrgb(hex_color)) for boxes, hex_color in groups]
    return annotate_box_groups(image.convert("RGB"), converted)


def _draw_panel(
    figure: plt.Figure,
    rect: tuple[float, float, float, float],
    image: Image.Image,
) -> None:
    fitted = _fit_rect(rect, image.size)
    axes = figure.add_axes(fitted)
    axes.set_axis_off()
    axes.imshow(np.asarray(image))
    draw_image_stage(figure, axes)


def plot_box_prompting_card(
    prompt_image: Image.Image,
    positives: tuple[_Box, ...],
    negatives: tuple[_Box, ...],
    target_image: Image.Image,
    predictions: sv.Detections,
    model_id: str,
    config: BenchmarkConfig,
    map_score: float | None = None,
    positives_on_target: bool = True,
    footer_prefix: str | None = None,
) -> plt.Figure:
    """Render a 16:9 box-prompting hero card (prompt left, target right).

    Args:
        prompt_image: Original prompt image.
        positives: Positive example boxes drawn in green on the prompt panel
            and, when ``positives_on_target`` is set, on the target panel.
        negatives: Negative example boxes drawn in red on the prompt panel.
        target_image: Original target image.
        predictions: Predicted detections drawn in blue, under the positives.
        model_id: Config key of the model that produced the predictions.
        config: Benchmark config for display info.
        map_score: Per-image mAP@50 in the 0-1 range, if available.
        positives_on_target: Whether the positives also overlay the target;
            disable when the target is a different image.
        footer_prefix: Optional text placed before the prompt statistics in
            the footer.

    Returns:
        Matplotlib figure with the box-prompting card.
    """
    fonts = load_fonts()
    model_info = config.models[model_id]
    lab_info = config.labs[model_info.lab]

    prompt_panel_image = _annotated_boxes(
        prompt_image,
        [(negatives, DISPLAY_NEGATIVE_HEX), (positives, DISPLAY_POSITIVE_HEX)],
    )
    predicted_boxes = tuple(
        tuple(float(value) for value in box) for box in predictions.xyxy
    )
    target_groups = [(predicted_boxes, DISPLAY_PREDICTION_HEX)]
    if positives_on_target:
        target_groups.append((positives, DISPLAY_POSITIVE_HEX))
    target_panel_image = _annotated_boxes(target_image, target_groups)

    figure = plt.figure(figsize=CARD_FIGURE_SIZE, facecolor="#FAFAFA")
    add_top_accent(figure)

    chrome = figure.add_axes([0.03, 0.045, 0.94, 0.91])
    chrome.set_axis_off()
    chrome.set_xlim(0, 1)
    chrome.set_ylim(0, 1)

    draw_identity_row(
        chrome,
        model_info.name,
        lab_info.name,
        lab_info.logo_url,
        "BOX PROMPTING",
    )

    if map_score is not None:
        chrome.plot(
            [0, 1],
            [_SCORE_DIVIDER_Y, _SCORE_DIVIDER_Y],
            color=DIVIDER_COLOR,
            lw=1,
            clip_on=False,
        )
        chrome.text(
            0.5,
            _SCORE_LABEL_Y,
            "mAP@50",
            fontsize=9.5,
            ha="center",
            va="center",
            color=PANEL_LABEL_COLOR,
            font=fonts.bold,
        )
        chrome.text(
            0.5,
            _SCORE_VALUE_Y,
            f"{map_score * 100:.1f}%",
            fontsize=26,
            ha="center",
            va="center",
            color=score_color(map_score),
            font=fonts.display,
        )

    for rect, caption in ((_PROMPT_RECT, "PROMPT"), (_TARGET_RECT, "TARGET")):
        chrome.text(
            _panel_center_x(rect),
            _CAPTION_Y,
            caption,
            fontsize=12,
            ha="center",
            va="center",
            color=PANEL_LABEL_COLOR,
            font=fonts.medium,
        )

    _draw_panel(figure, _PROMPT_RECT, prompt_panel_image)
    _draw_panel(figure, _TARGET_RECT, target_panel_image)

    prediction_count = len(predictions)
    prediction_word = "prediction" if prediction_count == 1 else "predictions"
    footer = (
        f"{len(positives)} positive \u00b7 {len(negatives)} negative prompts "
        f"\u00b7 {prediction_count} {prediction_word}"
    )
    if footer_prefix:
        footer = f"{footer_prefix} \u00b7 {footer}"
    draw_brand_footer(chrome, footer)
    return figure


def _load_example(
    record: dict[str, Any],
    arm: str,
    sample: DetectionSample,
) -> dict[str, Any]:
    case = build_example_case(sample, max_objects=None)
    if case is None:
        raise click.UsageError(f"no example case for image {record['image']}")
    detections, _ = record_to_detections(record, sample)
    positives = tuple(tuple(box) for box in record["positive_xyxy"])
    negatives = tuple(tuple(box) for box in record["negative_xyxy"])
    if arm.endswith("_single"):
        positives = positives[:1]
        negatives = ()
    image = load_case_image(sample)
    return {
        "prompt_image": image,
        "positives": positives,
        "negatives": negatives,
        "target_image": image,
        "predictions": detections,
        "map_score": compute_image_map50(detections, target_detections(case)),
    }


def _load_cross_example(
    record: dict[str, Any],
    prompt_sample: DetectionSample,
    target_sample: DetectionSample,
    target_xyxy: tuple[_Box, ...],
) -> dict[str, Any]:
    detections, _ = record_target_detections(record, 0, target_sample)
    ground_truth = (
        sv.Detections(
            xyxy=np.array(target_xyxy, dtype=np.float32),
            class_id=np.zeros(len(target_xyxy), dtype=int),
        )
        if target_xyxy
        else sv.Detections.empty()
    )
    return {
        "prompt_image": load_case_image(prompt_sample),
        "positives": tuple(tuple(box) for box in record["positive_xyxy"]),
        "negatives": tuple(tuple(box) for box in record["negative_xyxy"]),
        "target_image": load_case_image(target_sample),
        "predictions": detections,
        "map_score": compute_image_map50(detections, ground_truth),
    }


@dataclass(frozen=True)
class CardJob:
    """One card to render, resolved in the worker process.

    Attributes:
        record: Raw collection record.
        arm: Arm name used in the card file name.
        prompt_sample: Sample of the image carrying the example boxes.
        target_sample: Sample of the image the predictions belong to.
        target_xyxy: Cross-image ground truth for the target, or ``None``
            for round-2 records where the case is rebuilt from the sample.
        card_path: Destination PNG path.
        footer_prefix: Optional footer text.
    """

    record: dict[str, Any]
    arm: str
    prompt_sample: DetectionSample
    target_sample: DetectionSample
    target_xyxy: tuple[_Box, ...] | None
    card_path: Path
    footer_prefix: str | None = None


def _render_card(job: CardJob, model: str, config: BenchmarkConfig) -> Path:
    if job.target_xyxy is None:
        example = _load_example(job.record, job.arm, job.prompt_sample)
    else:
        example = _load_cross_example(
            job.record, job.prompt_sample, job.target_sample, job.target_xyxy
        )
    figure = plot_box_prompting_card(
        prompt_image=example["prompt_image"],
        positives=example["positives"],
        negatives=example["negatives"],
        target_image=example["target_image"],
        predictions=example["predictions"],
        model_id=model,
        config=config,
        map_score=example["map_score"],
        positives_on_target=job.target_xyxy is None,
        footer_prefix=job.footer_prefix,
    )
    figure.savefig(str(job.card_path), dpi=150)
    plt.close(figure)
    return job.card_path


def _round2_jobs(
    *,
    raw_directory: Path,
    model: str,
    arms: tuple[str, ...],
    image_name: str | None,
    sample_index: dict[str, DetectionSample],
    output_directory: Path,
) -> list[CardJob]:
    jobs: list[CardJob] = []
    for arm in arms:
        records = load_arm_records(raw_directory, model, arm)
        if image_name is not None:
            records = [record for record in records if record["image"] == image_name]
        for record in sorted(records, key=lambda item: item["image"]):
            sample = sample_index.get(record["image"])
            if record.get("error") is not None or sample is None:
                continue
            jobs.append(
                CardJob(
                    record=record,
                    arm=arm,
                    prompt_sample=sample,
                    target_sample=sample,
                    target_xyxy=None,
                    card_path=output_directory
                    / f"{arm}__{Path(record['image']).stem}.png",
                )
            )
    return jobs


def _cross_jobs(
    *,
    raw_directory: Path,
    model: str,
    image_name: str | None,
    sample_index: dict[str, DetectionSample],
    output_directory: Path,
) -> list[CardJob]:
    cases_by_group = {
        group.group_id: build_cross_case(group, sample_index)
        for group in build_image_groups(sample_index)
    }
    jobs: list[CardJob] = []
    records = load_cross_records(raw_directory, model, _CROSS_ARM)
    for record in sorted(records, key=lambda item: item["request_key"]):
        target_name = record["target_images"][0]
        if image_name is not None and target_name != image_name:
            continue
        case = cases_by_group.get(record["group_id"])
        prompt_sample = sample_index.get(record["prompt_image"])
        target_sample = sample_index.get(target_name)
        if (
            record.get("error") is not None
            or case is None
            or prompt_sample is None
            or target_sample is None
        ):
            continue
        class_label = case.class_name or "all objects"
        jobs.append(
            CardJob(
                record=record,
                arm=_CROSS_ARM,
                prompt_sample=prompt_sample,
                target_sample=target_sample,
                target_xyxy=case.target_xyxy[target_name],
                card_path=output_directory
                / f"{_CROSS_ARM}__{case.group_id}__{Path(target_name).stem}.png",
                footer_prefix=f"{case.group_id} \u00b7 {class_label}",
            )
        )
    return jobs


@click.command()
@click.option(
    "--model",
    type=click.Choice(SUPPORTED_MODELS),
    default="gpt-6-astra",
    show_default=True,
)
@click.option(
    "--effort", type=click.Choice(["low", "high"]), default="low", show_default=True
)
@click.option(
    "--experiment",
    type=click.Choice(["round2", "cross"]),
    default="round2",
    show_default=True,
    help="round2: same-image arms; cross: pairwise cross-image records.",
)
@click.option(
    "--arms",
    default=",".join(_DEFAULT_ARMS),
    show_default=True,
    help="Comma-separated round-2 arms to render (ignored for cross).",
)
@click.option(
    "--image",
    "image_name",
    default=None,
    help="Render one image only (the target image for cross).",
)
@click.option(
    "--dataset-directory",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("data/detection/train"),
    show_default=True,
)
@click.option(
    "--results-directory",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Defaults to results-box-prompting-<model>-<round2|cross>-<effort>.",
)
@click.option(
    "--output-directory",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Defaults to visualizations/box-prompting-cards/<model>[-cross]-<effort>.",
)
@click.option(
    "--max-workers",
    type=click.IntRange(min=1),
    default=max(1, (os.cpu_count() or 2) - 1),
    show_default=True,
    help="Render processes.",
)
def main(
    model: str,
    effort: str,
    experiment: str,
    arms: str,
    image_name: str | None,
    dataset_directory: Path,
    results_directory: Path | None,
    output_directory: Path | None,
    max_workers: int,
) -> None:
    """Render polished box-prompting cards for round-2 or cross-image results."""
    selected_arms = tuple(arm.strip() for arm in arms.split(",") if arm.strip())
    unknown = [arm for arm in selected_arms if arm not in ROUND2_ARMS]
    if unknown:
        raise click.UsageError(f"unknown arms: {', '.join(unknown)}")
    if results_directory is None:
        results_directory = Path(f"results-box-prompting-{model}-{experiment}-{effort}")
    if output_directory is None:
        suffix = (
            f"{model}-cross-{effort}" if experiment == "cross" else f"{model}-{effort}"
        )
        output_directory = Path("visualizations/box-prompting-cards") / suffix
    raw_directory = results_directory / "raw"
    if not raw_directory.exists():
        raise click.UsageError(f"missing raw directory {raw_directory}")

    sample_index = build_sample_index(
        DetectionTask().load_samples(str(dataset_directory))
    )
    config = load_config(None)
    output_directory.mkdir(parents=True, exist_ok=True)

    if experiment == "cross":
        jobs = _cross_jobs(
            raw_directory=raw_directory,
            model=model,
            image_name=image_name,
            sample_index=sample_index,
            output_directory=output_directory,
        )
    else:
        jobs = _round2_jobs(
            raw_directory=raw_directory,
            model=model,
            arms=selected_arms,
            image_name=image_name,
            sample_index=sample_index,
            output_directory=output_directory,
        )

    rendered: dict[str, int] = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_render_card, job, model, config): job.arm for job in jobs
        }
        for future in as_completed(futures):
            future.result()
            rendered[futures[future]] = rendered.get(futures[future], 0) + 1
    for arm, count in sorted(rendered.items()):
        click.echo(f"{arm}: {count} cards written to {output_directory}")


if __name__ == "__main__":
    main()
