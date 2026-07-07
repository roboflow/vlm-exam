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

import copy

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import supervision as sv
from matplotlib import gridspec
from matplotlib.colors import to_rgb
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

from vlm_exam.config import BenchmarkConfig
from vlm_exam.visualization.theme import (
    BACKGROUND_COLOR,
    DIVIDER_COLOR,
    FAILURE_COLOR,
    SUCCESS_COLOR,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    add_top_accent,
    fetch_logo,
    load_fonts,
)

_COLOR_PALETTE = sv.ColorPalette(
    colors=[
        sv.Color.from_hex("#7DD3FC"),
        sv.Color.from_hex("#5EEAD4"),
        sv.Color.from_hex("#C4B5FD"),
        sv.Color.from_hex("#FDA4AF"),
        sv.Color.from_hex("#F0ABFC"),
        sv.Color.from_hex("#FDE047"),
        sv.Color.from_hex("#67E8F9"),
        sv.Color.from_hex("#6EE7B7"),
        sv.Color.from_hex("#D8B4FE"),
        sv.Color.from_hex("#F9A8D4"),
        sv.Color.from_hex("#A5B4FC"),
        sv.Color.from_hex("#86EFAC"),
    ]
)

_TEXT_COLOR_PALETTE = sv.ColorPalette(
    colors=[
        sv.Color.from_hex("#172554"),
        sv.Color.from_hex("#134E4A"),
        sv.Color.from_hex("#2E1065"),
        sv.Color.from_hex("#4C0519"),
        sv.Color.from_hex("#4A044E"),
        sv.Color.from_hex("#422006"),
        sv.Color.from_hex("#083344"),
        sv.Color.from_hex("#042F2E"),
        sv.Color.from_hex("#3B0764"),
        sv.Color.from_hex("#500724"),
        sv.Color.from_hex("#1E1B4B"),
        sv.Color.from_hex("#052E16"),
    ]
)

_TARGET_LONG_EDGE = 1024
_FIGURE_WIDTH = 14.0
_MARGIN_LEFT = 0.04
_MARGIN_RIGHT = 0.96
_MARGIN_TOP = 0.97
_MARGIN_BOTTOM = 0.03
_PANEL_GAP = 0.04
_HEADER_HEIGHT = 0.9
_DIVIDER_HEIGHT = 0.04
_COLUMN_HEADER_HEIGHT = 0.35
_FOOTER_HEIGHT = 0.85
_MIN_IMAGE_HEIGHT = 3.0
_MAX_IMAGE_HEIGHT = 9.0


def _annotate_image(
    image: np.ndarray,
    detections: sv.Detections,
    labels: list[str],
) -> np.ndarray:
    """Annotate with consistent visual size regardless of source resolution.

    Resizes the image to a fixed long-edge before annotation so that font
    size and line thickness appear identical across different resolutions.
    """
    height, width = image.shape[:2]
    long_edge = max(width, height)
    if long_edge > _TARGET_LONG_EDGE:
        scale = _TARGET_LONG_EDGE / long_edge
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = cv2.resize(image, (new_width, new_height))
        if len(detections) > 0:
            detections = _scale_detections(detections, scale)

    fonts = load_fonts()
    box_annotator = sv.BoxAnnotator(color=_COLOR_PALETTE, thickness=2)
    label_annotator = sv.RichLabelAnnotator(
        color=_COLOR_PALETTE,
        text_color=_TEXT_COLOR_PALETTE,
        font_path=fonts.medium.get_file(),
        font_size=16,
        text_padding=8,
        text_position=sv.Position.TOP_LEFT,
    )
    scene = image.copy()
    scene = box_annotator.annotate(scene=scene, detections=detections)
    scene = label_annotator.annotate(scene=scene, detections=detections, labels=labels)
    return scene


def _scale_detections(detections: sv.Detections, scale: float) -> sv.Detections:
    scaled = copy.deepcopy(detections)
    scaled.xyxy = (scaled.xyxy * scale).astype(np.float32)
    return scaled


def _pluralize(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _draw_header(
    axes: plt.Axes,
    model_name: str,
    lab_name: str,
    lab_logo_url: str,
) -> None:
    fonts = load_fonts()
    axes.set_axis_off()
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)

    axes.text(
        0.0,
        0.50,
        "Object Detection",
        fontsize=12,
        va="center",
        ha="left",
        color=TEXT_PRIMARY,
        font=fonts.medium,
    )
    axes.text(
        1.0,
        0.58,
        model_name,
        fontsize=11.5,
        va="bottom",
        ha="right",
        color=TEXT_PRIMARY,
        font=fonts.bold,
    )
    axes.text(
        1.0,
        0.38,
        lab_name,
        fontsize=9.5,
        va="top",
        ha="right",
        color=TEXT_SECONDARY,
        font=fonts.medium,
    )

    try:
        logo_image = fetch_logo(lab_logo_url, size=28)
        image_box = OffsetImage(logo_image, zoom=0.30)
        annotation = AnnotationBbox(
            image_box,
            (0.85, 0.48),
            frameon=False,
            xycoords="axes fraction",
            box_alignment=(0.5, 0.5),
            zorder=5,
        )
        axes.add_artist(annotation)
    except Exception:
        pass


def _draw_divider_line(axes: plt.Axes) -> None:
    axes.set_axis_off()
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    axes.axhline(
        y=0.5,
        xmin=0.0,
        xmax=1.0,
        color=DIVIDER_COLOR,
        linewidth=1,
    )


def _figure_height(image_row_height: float) -> float:
    content = (
        _HEADER_HEIGHT
        + _DIVIDER_HEIGHT
        + _COLUMN_HEADER_HEIGHT
        + image_row_height
        + _DIVIDER_HEIGHT
        + _FOOTER_HEIGHT
    )
    return content / (_MARGIN_TOP - _MARGIN_BOTTOM)


def plot_detection_card(
    image: np.ndarray,
    ground_truth: sv.Detections,
    predictions: sv.Detections,
    gt_labels: list[str],
    pred_labels: list[str],
    model_id: str,
    config: BenchmarkConfig,
    map_score: float | None = None,
) -> plt.Figure:
    """Render a detection comparison card with ground truth vs predictions.

    Args:
        image: Original BGR image as numpy array.
        ground_truth: Ground truth detections.
        predictions: Model predicted detections.
        gt_labels: Class labels for ground truth boxes.
        pred_labels: Class labels for prediction boxes.
        model_id: Identifier of the model that produced the predictions.
        config: Benchmark config for display info.
        map_score: Per-image mAP@50 score if available.

    Returns:
        Matplotlib figure with the detection card.
    """
    fonts = load_fonts()
    model_info = config.models[model_id]
    lab_info = config.labs[model_info.lab]

    gt_annotated = _annotate_image(image, ground_truth, gt_labels)[:, :, ::-1]
    pred_annotated = _annotate_image(image, predictions, pred_labels)[:, :, ::-1]

    image_height, image_width = gt_annotated.shape[:2]
    usable_width = _FIGURE_WIDTH * (_MARGIN_RIGHT - _MARGIN_LEFT)
    panel_width = usable_width / (2 + _PANEL_GAP)
    image_row_height = panel_width * image_height / image_width
    image_row_height = min(max(image_row_height, _MIN_IMAGE_HEIGHT), _MAX_IMAGE_HEIGHT)

    figure_height = _figure_height(image_row_height)
    figure = plt.figure(
        figsize=(_FIGURE_WIDTH, figure_height), facecolor=BACKGROUND_COLOR
    )
    add_top_accent(figure)

    grid = gridspec.GridSpec(
        6,
        2,
        height_ratios=[
            _HEADER_HEIGHT,
            _DIVIDER_HEIGHT,
            _COLUMN_HEADER_HEIGHT,
            image_row_height,
            _DIVIDER_HEIGHT,
            _FOOTER_HEIGHT,
        ],
        hspace=0.03,
        wspace=_PANEL_GAP,
        left=_MARGIN_LEFT,
        right=_MARGIN_RIGHT,
        top=_MARGIN_TOP,
        bottom=_MARGIN_BOTTOM,
    )

    header_axes = figure.add_subplot(grid[0, :])
    _draw_header(header_axes, model_info.name, lab_info.name, lab_info.logo_url)

    divider_top = figure.add_subplot(grid[1, :])
    _draw_divider_line(divider_top)

    col_header_axes = figure.add_subplot(grid[2, :])
    col_header_axes.set_axis_off()
    col_header_axes.set_xlim(0, 1)
    col_header_axes.set_ylim(0, 1)

    col_header_axes.text(
        0.25,
        0.5,
        f"Ground Truth  \u00b7  {_pluralize(len(ground_truth), 'object')}",
        fontsize=9.5,
        va="center",
        ha="center",
        color=TEXT_SECONDARY,
        font=fonts.medium,
    )
    col_header_axes.text(
        0.75,
        0.5,
        f"Prediction  \u00b7  {_pluralize(len(predictions), 'object')}",
        fontsize=9.5,
        va="center",
        ha="center",
        color=TEXT_SECONDARY,
        font=fonts.medium,
    )

    gt_axes = figure.add_subplot(grid[3, 0])
    gt_axes.imshow(gt_annotated)
    gt_axes.set_axis_off()

    pred_axes = figure.add_subplot(grid[3, 1])
    pred_axes.imshow(pred_annotated)
    pred_axes.set_axis_off()

    divider_bottom = figure.add_subplot(grid[4, :])
    _draw_divider_line(divider_bottom)

    footer_axes = figure.add_subplot(grid[5, :])
    footer_axes.set_axis_off()
    footer_axes.set_xlim(0, 1)
    footer_axes.set_ylim(0, 1)

    if map_score is not None:
        score_color = SUCCESS_COLOR if map_score >= 0.5 else FAILURE_COLOR
        red, green, blue = to_rgb(score_color)

        position = footer_axes.get_position()
        figure.patches.append(
            mpatches.FancyBboxPatch(
                (position.x0, position.y0),
                position.width,
                position.height,
                transform=figure.transFigure,
                facecolor=(red, green, blue, 0.07),
                edgecolor="none",
                boxstyle="round,pad=0.003",
                zorder=0,
                clip_on=False,
            )
        )
        figure.patches.append(
            plt.Rectangle(
                (position.x0, position.y0),
                0.004,
                position.height,
                transform=figure.transFigure,
                facecolor=score_color,
                edgecolor="none",
                zorder=1,
                clip_on=False,
            )
        )

        footer_axes.text(
            0.50,
            0.78,
            "mAP@50",
            fontsize=10,
            ha="center",
            va="center",
            color=TEXT_SECONDARY,
            font=fonts.bold,
        )
        footer_axes.text(
            0.50,
            0.32,
            f"{map_score:.3f}",
            fontsize=26,
            ha="center",
            va="center",
            color=score_color,
            font=fonts.display,
        )

    return figure
