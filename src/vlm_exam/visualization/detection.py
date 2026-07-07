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

from vlm_exam.config import BenchmarkConfig
from vlm_exam.tasks.detection import MAP_PASS_THRESHOLD
from vlm_exam.visualization.theme import (
    BACKGROUND_COLOR,
    DIVIDER_COLOR,
    FAILURE_COLOR,
    SUCCESS_COLOR,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    add_top_accent,
    draw_card_header,
    load_fonts,
)

LABEL_MODES = ("auto", "labels", "legend")
"""Valid values for the detection card label rendering mode."""

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
_LABEL_FONT_SIZE = 16
_LABEL_PADDING = 8
_LABEL_CHAR_WIDTH_FACTOR = 0.62
_COLLISION_FRACTION = 0.5
_MIN_COLLIDING_LABELS = 8
_MAX_LABELED_BOXES = 30

_FIGURE_WIDTH = 14.0
_MARGIN_LEFT = 0.04
_MARGIN_RIGHT = 0.96
_MARGIN_TOP = 0.97
_MARGIN_BOTTOM = 0.03
_PANEL_GAP = 0.04
_HEADER_HEIGHT = 0.9
_DIVIDER_HEIGHT = 0.04
_COLUMN_HEADER_HEIGHT = 0.5
_FOOTER_HEIGHT = 0.85
_MIN_IMAGE_HEIGHT = 3.0
_MAX_IMAGE_HEIGHT = 9.0

_LEGEND_ROW_HEIGHT = 0.32
_LEGEND_SWATCH_WIDTH = 0.28
_LEGEND_SWATCH_HEIGHT = 0.16
_LEGEND_TEXT_GAP = 0.08
_LEGEND_ITEM_GAP = 0.45
_LEGEND_CHAR_WIDTH = 0.075


def _display_scale(width: int, height: int) -> float:
    long_edge = max(width, height)
    if long_edge <= _TARGET_LONG_EDGE:
        return 1.0
    return _TARGET_LONG_EDGE / long_edge


def _scale_detections(detections: sv.Detections, scale: float) -> sv.Detections:
    scaled = copy.deepcopy(detections)
    scaled.xyxy = (scaled.xyxy * scale).astype(np.float32)
    return scaled


def _label_rectangles(
    detections: sv.Detections,
    labels: list[str],
) -> list[tuple[float, float, float, float]]:
    rectangles = []
    for (x1, y1, _, _), label in zip(detections.xyxy, labels):
        width = (
            len(label) * _LABEL_CHAR_WIDTH_FACTOR * _LABEL_FONT_SIZE
            + 2 * _LABEL_PADDING
        )
        height = _LABEL_FONT_SIZE + 2 * _LABEL_PADDING
        rectangles.append((x1, y1 - height, x1 + width, y1))
    return rectangles


def _rectangles_intersect(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> bool:
    return (
        first[0] < second[2]
        and second[0] < first[2]
        and first[1] < second[3]
        and second[1] < first[3]
    )


def _labels_collide(
    detections: sv.Detections,
    labels: list[str],
    image_wh: tuple[int, int],
) -> bool:
    """Estimate whether drawn label pills would overlap each other.

    Legend mode is reserved for genuinely dense images: many boxes, or
    a large number of label pills that mostly intersect each other.

    Args:
        detections: Detections in original image coordinates.
        labels: Class labels, one per detection.
        image_wh: Original image (width, height).

    Returns:
        ``True`` when labels are too crowded to remain readable.
    """
    if len(detections) > _MAX_LABELED_BOXES:
        return True
    if len(detections) < 2:
        return False

    scale = _display_scale(*image_wh)
    scaled = _scale_detections(detections, scale)
    rectangles = _label_rectangles(scaled, labels)

    colliding: set[int] = set()
    for i in range(len(rectangles)):
        for j in range(i + 1, len(rectangles)):
            if _rectangles_intersect(rectangles[i], rectangles[j]):
                colliding.add(i)
                colliding.add(j)

    return (
        len(colliding) >= _MIN_COLLIDING_LABELS
        and len(colliding) / len(rectangles) > _COLLISION_FRACTION
    )


def _collect_legend_entries(
    ground_truth: sv.Detections,
    gt_labels: list[str],
    predictions: sv.Detections,
    pred_labels: list[str],
) -> list[tuple[int, str]]:
    """Collect unique (class_id, name) pairs across both panels."""
    entries: dict[int, str] = {}
    for detections, labels in ((ground_truth, gt_labels), (predictions, pred_labels)):
        if detections.class_id is None:
            continue
        for class_id, label in zip(detections.class_id, labels):
            entries.setdefault(int(class_id), label)
    return sorted(entries.items())


def _layout_legend_rows(
    entries: list[tuple[int, str]],
    usable_width: float,
) -> list[list[tuple[int, str]]]:
    rows: list[list[tuple[int, str]]] = [[]]
    x = 0.0
    for class_id, name in entries:
        item_width = (
            _LEGEND_SWATCH_WIDTH
            + _LEGEND_TEXT_GAP
            + len(name) * _LEGEND_CHAR_WIDTH
            + _LEGEND_ITEM_GAP
        )
        if rows[-1] and x + item_width > usable_width:
            rows.append([])
            x = 0.0
        rows[-1].append((class_id, name))
        x += item_width
    return rows


def _draw_legend(
    axes: plt.Axes,
    rows: list[list[tuple[int, str]]],
    usable_width: float,
) -> None:
    # Axes data units are set to physical inches (the axes spans
    # usable_width inches horizontally), so layout math stays in inches.
    fonts = load_fonts()
    axes.set_axis_off()
    height = len(rows) * _LEGEND_ROW_HEIGHT
    axes.set_xlim(0, usable_width)
    axes.set_ylim(0, height)

    for row_index, row in enumerate(rows):
        y_center = height - (row_index + 0.5) * _LEGEND_ROW_HEIGHT
        x = 0.0
        for class_id, name in row:
            color = _COLOR_PALETTE.by_idx(class_id).as_hex()
            axes.add_patch(
                mpatches.FancyBboxPatch(
                    (x, y_center - _LEGEND_SWATCH_HEIGHT / 2),
                    _LEGEND_SWATCH_WIDTH,
                    _LEGEND_SWATCH_HEIGHT,
                    boxstyle=mpatches.BoxStyle.Round(
                        pad=0, rounding_size=_LEGEND_SWATCH_HEIGHT / 2
                    ),
                    facecolor=color,
                    edgecolor="none",
                )
            )
            axes.text(
                x + _LEGEND_SWATCH_WIDTH + _LEGEND_TEXT_GAP,
                y_center,
                name,
                fontsize=9.5,
                va="center",
                ha="left",
                color=TEXT_PRIMARY,
                font=fonts.medium,
            )
            x += (
                _LEGEND_SWATCH_WIDTH
                + _LEGEND_TEXT_GAP
                + len(name) * _LEGEND_CHAR_WIDTH
                + _LEGEND_ITEM_GAP
            )


def _annotate_image(
    image: np.ndarray,
    detections: sv.Detections,
    labels: list[str],
    draw_labels: bool,
) -> np.ndarray:
    """Annotate with consistent visual size regardless of source resolution.

    Resizes the image to a fixed long-edge before annotation so that font
    size and line thickness appear identical across different resolutions.
    """
    height, width = image.shape[:2]
    scale = _display_scale(width, height)
    if scale < 1.0:
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = cv2.resize(image, (new_width, new_height))
        if len(detections) > 0:
            detections = _scale_detections(detections, scale)

    box_annotator = sv.BoxAnnotator(color=_COLOR_PALETTE, thickness=2)
    scene = image.copy()
    scene = box_annotator.annotate(scene=scene, detections=detections)

    if draw_labels:
        fonts = load_fonts()
        label_annotator = sv.RichLabelAnnotator(
            color=_COLOR_PALETTE,
            text_color=_TEXT_COLOR_PALETTE,
            font_path=fonts.medium.get_file(),
            font_size=_LABEL_FONT_SIZE,
            text_padding=_LABEL_PADDING,
            text_position=sv.Position.TOP_LEFT,
        )
        scene = label_annotator.annotate(
            scene=scene, detections=detections, labels=labels
        )
    return scene


def _pluralize(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


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


def plot_detection_card(
    image: np.ndarray,
    ground_truth: sv.Detections,
    predictions: sv.Detections,
    gt_labels: list[str],
    pred_labels: list[str],
    model_id: str,
    config: BenchmarkConfig,
    map_score: float | None = None,
    label_mode: str = "auto",
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
        label_mode: ``"labels"`` draws class labels on boxes, ``"legend"``
            draws boxes only with a color legend below the images, and
            ``"auto"`` picks based on estimated label overlap.

    Returns:
        Matplotlib figure with the detection card.
    """
    if label_mode not in LABEL_MODES:
        modes = ", ".join(LABEL_MODES)
        raise ValueError(f"Unknown label_mode {label_mode!r}. Valid modes: {modes}")

    fonts = load_fonts()
    model_info = config.models[model_id]
    lab_info = config.labs[model_info.lab]

    height, width = image.shape[:2]
    if label_mode == "auto":
        crowded = _labels_collide(
            ground_truth, gt_labels, (width, height)
        ) or _labels_collide(predictions, pred_labels, (width, height))
        label_mode = "legend" if crowded else "labels"

    draw_labels = label_mode == "labels"
    gt_annotated = _annotate_image(image, ground_truth, gt_labels, draw_labels)
    pred_annotated = _annotate_image(image, predictions, pred_labels, draw_labels)
    gt_annotated = gt_annotated[:, :, ::-1]
    pred_annotated = pred_annotated[:, :, ::-1]

    usable_width = _FIGURE_WIDTH * (_MARGIN_RIGHT - _MARGIN_LEFT)

    legend_rows: list[list[tuple[int, str]]] = []
    legend_height = 0.0
    if label_mode == "legend":
        entries = _collect_legend_entries(
            ground_truth, gt_labels, predictions, pred_labels
        )
        if entries:
            legend_rows = _layout_legend_rows(entries, usable_width)
            legend_height = len(legend_rows) * _LEGEND_ROW_HEIGHT + 0.1

    image_height, image_width = gt_annotated.shape[:2]
    panel_width = usable_width / (2 + _PANEL_GAP)
    image_row_height = panel_width * image_height / image_width
    image_row_height = min(max(image_row_height, _MIN_IMAGE_HEIGHT), _MAX_IMAGE_HEIGHT)

    height_ratios = [
        _HEADER_HEIGHT,
        _DIVIDER_HEIGHT,
        _COLUMN_HEADER_HEIGHT,
        image_row_height,
    ]
    if legend_rows:
        height_ratios.append(legend_height)
    height_ratios.extend([_DIVIDER_HEIGHT, _FOOTER_HEIGHT])

    content_height = sum(height_ratios)
    figure_height = content_height / (_MARGIN_TOP - _MARGIN_BOTTOM)
    figure = plt.figure(
        figsize=(_FIGURE_WIDTH, figure_height), facecolor=BACKGROUND_COLOR
    )
    add_top_accent(figure)

    grid = gridspec.GridSpec(
        len(height_ratios),
        2,
        height_ratios=height_ratios,
        hspace=0.03,
        wspace=_PANEL_GAP,
        left=_MARGIN_LEFT,
        right=_MARGIN_RIGHT,
        top=_MARGIN_TOP,
        bottom=_MARGIN_BOTTOM,
    )

    header_axes = figure.add_subplot(grid[0, :])
    draw_card_header(
        header_axes,
        "Object Detection",
        model_info.name,
        lab_info.name,
        lab_info.logo_url,
    )

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
        fontsize=13,
        va="center",
        ha="center",
        color=TEXT_PRIMARY,
        font=fonts.bold,
    )
    col_header_axes.text(
        0.75,
        0.5,
        f"Prediction  \u00b7  {_pluralize(len(predictions), 'object')}",
        fontsize=13,
        va="center",
        ha="center",
        color=TEXT_PRIMARY,
        font=fonts.bold,
    )

    gt_axes = figure.add_subplot(grid[3, 0])
    gt_axes.imshow(gt_annotated)
    gt_axes.set_axis_off()

    pred_axes = figure.add_subplot(grid[3, 1])
    pred_axes.imshow(pred_annotated)
    pred_axes.set_axis_off()

    next_row = 4
    if legend_rows:
        legend_axes = figure.add_subplot(grid[next_row, :])
        _draw_legend(legend_axes, legend_rows, usable_width)
        next_row += 1

    divider_bottom = figure.add_subplot(grid[next_row, :])
    _draw_divider_line(divider_bottom)

    footer_axes = figure.add_subplot(grid[next_row + 1, :])
    footer_axes.set_axis_off()
    footer_axes.set_xlim(0, 1)
    footer_axes.set_ylim(0, 1)

    if map_score is not None:
        score_color = (
            SUCCESS_COLOR if map_score >= MAP_PASS_THRESHOLD else FAILURE_COLOR
        )
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
