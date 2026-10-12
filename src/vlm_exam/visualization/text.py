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

import math
import textwrap
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import supervision as sv
from PIL import Image

from vlm_exam.config import VisualizationConfig
from vlm_exam.results import SampleResult
from vlm_exam.tasks.text import TextSample, TextTask
from vlm_exam.tasks.text_scoring import match_regions, parse_json
from vlm_exam.visualization import theme
from vlm_exam.visualization.detection import region_diff_image
from vlm_exam.visualization.single_value import draw_single_value_panel
from vlm_exam.visualization.structured import draw_structured_panel
from vlm_exam.visualization.transcription import draw_transcription_panel


def _frame(
    image: Image.Image, model: str, config: VisualizationConfig, label: str
) -> tuple[plt.Figure, plt.Axes]:
    figure, axes, rail = theme.create_hero_card()
    axes.imshow(image)
    theme.draw_image_stage(figure, axes)
    identity = config.models[model]
    lab = config.labs[identity.lab]
    theme.draw_identity_row(rail, identity.name, lab.name, lab.logo_url, label)
    return figure, rail


def _json_cards(
    image: Image.Image,
    sample: TextSample,
    result: SampleResult,
    model: str,
    config: VisualizationConfig,
) -> list[plt.Figure]:
    figure, rail = _frame(image, model, config, "DATA EXTRACTION")
    draw_structured_panel(rail, sample.answer, result.predicted)
    fonts = theme.load_fonts()
    rail.plot([0, 1], [0.262, 0.262], color=theme.DIVIDER_COLOR, lw=1, clip_on=False)
    rail.text(
        0.5,
        0.212,
        "FIELD F1",
        fontsize=9.5,
        ha="center",
        va="center",
        color=theme.PANEL_LABEL_COLOR,
        font=fonts.bold,
    )
    score = result.metadata.get("score")
    scored = type(score) in (int, float) and math.isfinite(score) and 0 <= score <= 1
    rail.text(
        0.5,
        0.142,
        f"{score * 100:.1f}%" if scored else "—",
        fontsize=26,
        ha="center",
        va="center",
        font=fonts.display,
        color=theme.score_color(score) if scored else theme.TEXT_SECONDARY,
    )
    theme.draw_brand_footer(rail, "evaluated via exact field values")
    return [figure]


def _region_rows(
    reference: list[dict[str, Any]], predictions: list[dict[str, Any] | None]
) -> list[dict[str, Any]]:
    paired = match_regions(reference, predictions)
    used_reference = {a for a, _ in paired}
    used_predictions = {b for _, b in paired}
    remaining_reference = [i for i in range(len(reference)) if i not in used_reference]
    remaining_predictions = [
        i for i in range(len(predictions)) if i not in used_predictions
    ]
    spatial = match_regions(
        [reference[i] for i in remaining_reference],
        [predictions[i] for i in remaining_predictions],
        require_text=False,
    )
    paired += [(remaining_reference[a], remaining_predictions[b]) for a, b in spatial]
    rows = []
    for a, b in paired:
        predicted = predictions[b]
        rows.append(
            {
                "bbox": predicted["bbox"],
                "expected": reference[a]["text"],
                "predicted": predicted["text"],
                "correct": (a in used_reference),
            }
        )
    used_reference = {a for a, _ in paired}
    used_predictions = {b for _, b in paired}
    rows.extend(
        {
            "bbox": region["bbox"],
            "expected": region["text"],
            "predicted": "[missing region]",
            "correct": False,
        }
        for index, region in enumerate(reference)
        if index not in used_reference
    )
    rows.extend(
        {
            "bbox": region["bbox"],
            "expected": "[extra region]",
            "predicted": region["text"],
            "correct": False,
        }
        for index, region in enumerate(predictions)
        if index not in used_predictions and region is not None
    )
    return sorted(
        rows,
        key=lambda row: (
            row["correct"],
            (row["bbox"][2] - row["bbox"][0]) < 1.2 * (row["bbox"][3] - row["bbox"][1]),
            row["bbox"][1],
            row["bbox"][0],
        ),
    )


def _localization_card(
    image: Image.Image,
    sample: TextSample,
    result: SampleResult,
    task: TextTask,
    model: str,
    config: VisualizationConfig,
    uploaded_size: tuple[int, int] | None,
) -> plt.Figure:
    try:
        parsed, _ = parse_json(result.predicted)
        predictions = task.parse_regions(parsed, sample, uploaded_size)
    except (ValueError, TypeError, KeyError, RecursionError):
        predictions = [None]
    reference = sample.answer
    rows = _region_rows(reference, predictions)

    display = image.copy()
    display.thumbnail((2200, 2200))
    scale = np.array([display.width / image.width, display.height / image.height] * 2)

    def boxes(regions: list[dict[str, Any] | None]) -> sv.Detections:
        return sv.Detections(
            xyxy=np.asarray(
                [region["bbox"] for region in regions if region is not None],
                dtype=float,
            ).reshape(-1, 4)
            * scale
        )

    overlay = region_diff_image(
        np.asarray(display), boxes(reference), boxes(predictions)
    )
    figure, rail = _frame(
        Image.fromarray(overlay), model, config, "TEXT LOCATION + RECOGNITION"
    )
    fonts = theme.load_fonts()
    rail_width, rail_height = theme.axes_size_inches(rail)
    longest = max(
        (max(len(str(row["expected"])), len(str(row["predicted"]))) for row in rows),
        default=0,
    )
    columns = 3 if longest <= 20 else 2 if longest <= 32 else 1
    selected = rows[: 12 if columns == 3 else 8 if columns == 2 else 3]
    gutter = 0.045
    tile_width = (1 - gutter * (columns - 1)) / columns
    font_size = 10 if columns == 3 else 11
    line_height = font_size * 1.25 / 72 / rail_height
    crops = []
    for row in selected:
        left, top, right, bottom = row["bbox"]
        crops.append(
            image.crop(
                (
                    max(0, int(left) - 3),
                    max(0, int(top) - 3),
                    min(image.width, math.ceil(right) + 3),
                    min(image.height, math.ceil(bottom) + 3),
                )
            )
        )
    if crops:
        scale = min(
            3.0 if columns == 3 else 2.0,
            min(tile_width * rail_width * 150 / crop.width for crop in crops),
            (90 if columns == 3 else 115) / max(crop.height for crop in crops),
        )
        tiles = []
        for row, crop in zip(selected, crops):
            values = (
                [("", row["predicted"], theme.SUCCESS_TEXT_COLOR)]
                if row["correct"]
                else [
                    ("− ", row["expected"], theme.SUCCESS_TEXT_COLOR),
                    ("+ ", row["predicted"], theme.FAILURE_TEXT_COLOR),
                ]
            )
            blocks = [
                (
                    textwrap.fill(
                        prefix + ("null" if value is None else str(value)),
                        width=55 if columns == 1 else 29 if columns == 2 else 21,
                    ),
                    color,
                )
                for prefix, value, color in values
            ]
            text_height = sum(
                (text.count("\n") + 1) * line_height for text, _ in blocks
            ) + 0.004 * (len(blocks) - 1)
            tiles.append(
                (
                    crop,
                    crop.width * scale / (rail_width * 150),
                    crop.height * scale / (rail_height * 150),
                    blocks,
                    text_height,
                )
            )
        groups = [tiles[i : i + columns] for i in range(0, len(tiles), columns)]
        heights = [max(t[2] + 0.013 + t[4] for t in group) for group in groups]
        gap = 0.040 if columns == 3 else 0.05
        lower = 0.19 if len(rows) > len(selected) else 0.13
        available = 0.85 - lower
        while groups and sum(heights) + gap * (len(groups) - 1) > available:
            groups.pop()
            heights.pop()
        top = (0.85 + lower + sum(heights) + gap * max(0, len(groups) - 1)) / 2
        shown = sum(len(group) for group in groups)
        for group, group_height in zip(groups, heights):
            for column, (crop, width, height, blocks, text_height) in enumerate(group):
                center = (
                    (1 - (len(group) * tile_width + (len(group) - 1) * gutter)) / 2
                    + column * (tile_width + gutter)
                    + tile_width / 2
                )
                crop_top = top - (group_height - height - 0.013 - text_height) / 2
                axes = rail.inset_axes(
                    [center - width / 2, crop_top - height, width, height]
                )
                axes.imshow(crop)
                axes.set_xticks([])
                axes.set_yticks([])
                for spine in axes.spines.values():
                    spine.set_color(theme.ROBOFLOW_PURPLE)
                    spine.set_linewidth(1.6)
                y = crop_top - height - 0.013
                for text, color in blocks:
                    rail.text(
                        center,
                        y,
                        text,
                        font=fonts.mono,
                        fontsize=font_size,
                        color=color,
                        va="top",
                        ha="center",
                        multialignment="center",
                        parse_math=False,
                        linespacing=1.25,
                    )
                    y -= (text.count("\n") + 1) * line_height + 0.004
            top -= group_height + gap
        if len(rows) > shown:
            rail.text(
                0.5,
                0.14,
                f"+ {len(rows) - shown} more regions",
                font=fonts.medium,
                fontsize=11,
                color=theme.TEXT_SECONDARY,
                va="top",
                ha="center",
            )
    else:
        rail.text(
            0.5,
            0.5,
            "No valid text regions",
            font=fonts.medium,
            fontsize=14,
            color=theme.TEXT_SECONDARY,
            ha="center",
        )
    invalid = sum(region is None for region in predictions)
    theme.draw_brand_footer(
        rail,
        "evaluated via box overlap and exact text"
        + (f" · {invalid} invalid regions" if invalid else ""),
    )
    return figure


def plot_text_cards(
    image: Image.Image,
    sample: TextSample,
    result: SampleResult,
    task: TextTask,
    model: str,
    config: VisualizationConfig,
    uploaded_size: tuple[int, int] | None = None,
) -> list[plt.Figure]:
    """Render one native hero card for each text sample.

    Args:
        image: Original source image.
        sample: Dataset question and reference.
        result: Saved prediction and deterministic score.
        task: Recorded model coordinate profile.
        model: Model key for display branding.
        config: Benchmark display configuration.
        uploaded_size: Recorded provider image dimensions for pixel conversion.

    Returns:
        A single native hero-card figure in a list.
    """
    if sample.category == "structured":
        return _json_cards(image, sample, result, model, config)
    if sample.category == "localization_recognition":
        return [
            _localization_card(
                image, sample, result, task, model, config, uploaded_size
            )
        ]
    if sample.category in ("transcription", "single_string"):
        single_value = sample.category == "single_string"
        figure, rail = _frame(
            image, model, config, "DATA EXTRACTION" if single_value else "TRANSCRIPTION"
        )
        if single_value:
            draw_single_value_panel(
                rail, sample.question, sample.expected, result.predicted
            )
        else:
            draw_transcription_panel(rail, sample.expected, result.predicted)
        fonts = theme.load_fonts()
        exact = single_value or sample.scoring_profile == "exact"
        rail.plot(
            [0, 1], [0.262, 0.262], color=theme.DIVIDER_COLOR, lw=1, clip_on=False
        )
        rail.text(
            0.5,
            0.212,
            "EXACT MATCH" if exact else "CHARACTER SIMILARITY",
            fontsize=9.5,
            ha="center",
            va="center",
            color=theme.PANEL_LABEL_COLOR,
            font=fonts.bold,
        )
        score = result.metadata.get("score")
        scored = (
            type(score) in (int, float) and math.isfinite(score) and 0 <= score <= 1
        )
        rail.text(
            0.5,
            0.142,
            f"{score * 100:.1f}%" if scored else "—",
            fontsize=26,
            ha="center",
            va="center",
            font=fonts.display,
            color=theme.score_color(score) if scored else theme.TEXT_SECONDARY,
        )
        theme.draw_brand_footer(
            rail,
            "evaluated via exact match"
            if exact
            else "evaluated via character similarity",
        )
        return [figure]

    raise ValueError(f"Unsupported text category: {sample.category}")
