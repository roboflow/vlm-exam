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
import math
from pathlib import Path

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import supervision as sv
from matplotlib.colors import to_rgb

from vlm_exam.config import BenchmarkConfig
from vlm_exam.visualization.theme import (
    CARD_FIGURE_SIZE,
    DIVIDER_COLOR,
    FAILURE_COLOR,
    FAILURE_TEXT_COLOR,
    HERO_IMAGE_RECT,
    HERO_RAIL_RECT,
    PANEL_LABEL_COLOR,
    ROBOFLOW_PURPLE,
    SUCCESS_COLOR,
    SUCCESS_TEXT_COLOR,
    TEXT_PRIMARY,
    create_hero_card,
    draw_brand_footer,
    draw_identity_row,
    draw_image_stage,
    draw_legend_chip,
    load_fonts,
    score_color,
)

LABEL_MODES = ("auto", "labels", "boxes")
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

# Independent knobs that happen to share a value: display fit is downscale-only
# for card layout, annotation normalization scales up or down so box thickness
# and label size stay consistent across source resolutions.
_TARGET_LONG_EDGE = 1024
_ANNOTATION_LONG_EDGE = 1024
_LABEL_FONT_SIZE = 18
_LABEL_PADDING = 9
_LABEL_CHAR_WIDTH_FACTOR = 0.62
_COLLISION_FRACTION = 0.5
_MIN_COLLIDING_LABELS = 8
_MAX_LABELED_BOXES = 30

_BOX_THICKNESS = 3

_LEGEND_MAX_ROWS_PER_COLUMN = 8
_LEGEND_ROW_HEIGHT_INCHES = 0.28
_LEGEND_SWATCH_INCHES = 0.16
_LEGEND_PADDING_INCHES = 0.12
_LEGEND_SWATCH_GAP_INCHES = 0.09
_LEGEND_COLUMN_GAP_INCHES = 0.28
_LEGEND_FONT_SIZE = 11.5
_LEGEND_CHAR_WIDTH_INCHES = _LEGEND_FONT_SIZE * 0.55 / 72
_LEGEND_MARGIN_INCHES = 0.14
_LEGEND_PANEL_ALPHA = 0.88
_LEGEND_PANEL_CORNER_INCHES = 0.06
_LEGEND_SWATCH_CORNER_INCHES = 0.03
_LEGEND_SWATCH_EDGE_COLOR = (0.0, 0.0, 0.0, 0.28)
_LEGEND_SWATCH_EDGE_WIDTH = 0.6

_DIFF_ONLY_EXPECTED_COLOR = SUCCESS_COLOR
_DIFF_ONLY_MODEL_COLOR = FAILURE_COLOR
_DIFF_BOTH_COLOR = ROBOFLOW_PURPLE
_DIFF_BASE_FADE = 0.78
_DIFF_TINT_ALPHA = 0.55
_BOTH_TEXT_COLOR = "#6B21A8"


def _display_scale(width: int, height: int) -> float:
    long_edge = max(width, height)
    if long_edge <= _TARGET_LONG_EDGE:
        return 1.0
    return _TARGET_LONG_EDGE / long_edge


def _annotation_scale(width: int, height: int) -> float:
    return _ANNOTATION_LONG_EDGE / max(width, height)


def _scale_detections(detections: sv.Detections, scale: float) -> sv.Detections:
    scaled = copy.deepcopy(detections)
    scaled.xyxy = (scaled.xyxy * scale).astype(np.float32)
    return scaled


def _normalize_for_annotation(
    image: np.ndarray,
    detections_list: list[sv.Detections],
) -> tuple[np.ndarray, list[sv.Detections]]:
    height, width = image.shape[:2]
    scale = _annotation_scale(width, height)
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(
        image,
        (round(width * scale), round(height * scale)),
        interpolation=interpolation,
    )
    scaled = [
        _scale_detections(detections, scale) if len(detections) > 0 else detections
        for detections in detections_list
    ]
    return resized, scaled


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

    Boxes-only mode is reserved for genuinely dense images: many boxes,
    or a large number of label pills that mostly intersect each other.

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

    scale = _annotation_scale(*image_wh)
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
    image, (detections,) = _normalize_for_annotation(image, [detections])

    box_annotator = sv.BoxAnnotator(color=_COLOR_PALETTE, thickness=_BOX_THICKNESS)
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


def _legend_entries(
    class_ids: np.ndarray | None,
    labels: list[str],
) -> list[tuple[str, sv.Color]]:
    if class_ids is None:
        return []
    entries: list[tuple[str, sv.Color]] = []
    seen: set[int] = set()
    for class_id, label in zip(class_ids, labels):
        class_id = int(class_id)
        if class_id in seen:
            continue
        seen.add(class_id)
        entries.append((label, _COLOR_PALETTE.by_idx(class_id)))
    return entries


def _draw_class_legend_overlay(
    figure: plt.Figure,
    image_rect: tuple[float, float, float, float],
    entries: list[tuple[str, sv.Color]],
) -> None:
    if not entries:
        return
    fonts = load_fonts()
    columns = 1 if len(entries) <= _LEGEND_MAX_ROWS_PER_COLUMN else 2
    rows = math.ceil(len(entries) / columns)
    longest = max(len(name) for name, _ in entries)
    column_width = (
        _LEGEND_SWATCH_INCHES
        + _LEGEND_SWATCH_GAP_INCHES
        + longest * _LEGEND_CHAR_WIDTH_INCHES
    )
    width_inches = (
        2 * _LEGEND_PADDING_INCHES
        + columns * column_width
        + (columns - 1) * _LEGEND_COLUMN_GAP_INCHES
    )
    height_inches = 2 * _LEGEND_PADDING_INCHES + rows * _LEGEND_ROW_HEIGHT_INCHES

    fig_width, fig_height = CARD_FIGURE_SIZE
    left, bottom, _, image_height = image_rect
    panel_width = width_inches / fig_width
    panel_height = height_inches / fig_height
    panel_left = left + _LEGEND_MARGIN_INCHES / fig_width
    panel_bottom = (
        bottom + image_height - _LEGEND_MARGIN_INCHES / fig_height - panel_height
    )

    axes = figure.add_axes((panel_left, panel_bottom, panel_width, panel_height))
    axes.set_axis_off()
    axes.set_xlim(0, width_inches)
    axes.set_ylim(0, height_inches)
    axes.add_patch(
        mpatches.FancyBboxPatch(
            (0.0, 0.0),
            width_inches,
            height_inches,
            boxstyle=f"round,pad=0,rounding_size={_LEGEND_PANEL_CORNER_INCHES}",
            facecolor=(1.0, 1.0, 1.0, _LEGEND_PANEL_ALPHA),
            edgecolor=DIVIDER_COLOR,
            linewidth=1.0,
            clip_on=False,
            zorder=5,
        )
    )

    column_stride = column_width + _LEGEND_COLUMN_GAP_INCHES

    for index, (name, color) in enumerate(entries):
        column = index // rows
        row = index % rows
        x = _LEGEND_PADDING_INCHES + column * column_stride
        y = (
            height_inches
            - _LEGEND_PADDING_INCHES
            - (row + 0.5) * _LEGEND_ROW_HEIGHT_INCHES
        )
        axes.add_patch(
            mpatches.FancyBboxPatch(
                (x, y - _LEGEND_SWATCH_INCHES / 2),
                _LEGEND_SWATCH_INCHES,
                _LEGEND_SWATCH_INCHES,
                boxstyle=f"round,pad=0,rounding_size={_LEGEND_SWATCH_CORNER_INCHES}",
                facecolor=color.as_hex(),
                edgecolor=_LEGEND_SWATCH_EDGE_COLOR,
                linewidth=_LEGEND_SWATCH_EDGE_WIDTH,
                clip_on=False,
                zorder=6,
            )
        )
        axes.text(
            x + _LEGEND_SWATCH_INCHES + _LEGEND_SWATCH_GAP_INCHES,
            y,
            name,
            fontsize=_LEGEND_FONT_SIZE,
            ha="left",
            va="center",
            color=TEXT_PRIMARY,
            font=fonts.medium,
            zorder=6,
            clip_on=False,
        )


def _draw_rounded_rectangle(
    image: np.ndarray,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    radius: int,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    x1, y1 = top_left
    x2, y2 = bottom_right
    radius = max(0, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
    corners = (
        (x1 + radius, y1 + radius, 180),
        (x2 - radius, y1 + radius, 270),
        (x2 - radius, y2 - radius, 0),
        (x1 + radius, y2 - radius, 90),
    )
    if thickness < 0:
        cv2.rectangle(image, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(image, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        for center_x, center_y, _ in corners:
            cv2.circle(image, (center_x, center_y), radius, color, -1)
        return
    cv2.line(image, (x1 + radius, y1), (x2 - radius, y1), color, thickness, cv2.LINE_AA)
    cv2.line(image, (x1 + radius, y2), (x2 - radius, y2), color, thickness, cv2.LINE_AA)
    cv2.line(image, (x1, y1 + radius), (x1, y2 - radius), color, thickness, cv2.LINE_AA)
    cv2.line(image, (x2, y1 + radius), (x2, y2 - radius), color, thickness, cv2.LINE_AA)
    for center_x, center_y, start_angle in corners:
        cv2.ellipse(
            image,
            (center_x, center_y),
            (radius, radius),
            start_angle,
            0,
            90,
            color,
            thickness,
            cv2.LINE_AA,
        )


def _draw_class_legend_cv2(
    image: np.ndarray,
    entries: list[tuple[str, sv.Color]],
) -> np.ndarray:
    if not entries:
        return image

    height, width = image.shape[:2]
    unit = max(width, height) / _ANNOTATION_LONG_EDGE
    padding = round(13 * unit)
    swatch = round(18 * unit)
    row_height = round(28 * unit)
    swatch_gap = round(9 * unit)
    column_gap = round(22 * unit)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.52 * unit
    text_thickness = max(1, round(unit))

    columns = 1 if len(entries) <= _LEGEND_MAX_ROWS_PER_COLUMN else 2
    rows = math.ceil(len(entries) / columns)
    text_widths = [
        cv2.getTextSize(name, font, font_scale, text_thickness)[0][0]
        for name, _ in entries
    ]
    column_text_widths = []
    for column in range(columns):
        widths = text_widths[column * rows : (column + 1) * rows]
        column_text_widths.append(max(widths) if widths else 0)

    panel_width = (
        2 * padding
        + sum(column_text_widths)
        + columns * (swatch + swatch_gap)
        + (columns - 1) * column_gap
    )
    panel_height = 2 * padding + rows * row_height

    origin_x = round(10 * unit)
    origin_y = origin_x
    panel_radius = round(8 * unit)
    swatch_radius = round(4 * unit)
    border_thickness = max(1, round(unit))

    overlay = image.copy()
    _draw_rounded_rectangle(
        overlay,
        (origin_x, origin_y),
        (origin_x + panel_width, origin_y + panel_height),
        panel_radius,
        (255, 255, 255),
        -1,
    )
    cv2.addWeighted(
        overlay, _LEGEND_PANEL_ALPHA, image, 1 - _LEGEND_PANEL_ALPHA, 0, image
    )
    _draw_rounded_rectangle(
        image,
        (origin_x, origin_y),
        (origin_x + panel_width, origin_y + panel_height),
        panel_radius,
        (237, 232, 232),
        border_thickness,
    )

    column_x = origin_x + padding
    for column in range(columns):
        for row in range(rows):
            index = column * rows + row
            if index >= len(entries):
                break
            name, color = entries[index]
            row_top = origin_y + padding + row * row_height
            swatch_y = row_top + (row_height - swatch) // 2
            _draw_rounded_rectangle(
                image,
                (column_x, swatch_y),
                (column_x + swatch, swatch_y + swatch),
                swatch_radius,
                color.as_bgr(),
                -1,
            )
            _draw_rounded_rectangle(
                image,
                (column_x, swatch_y),
                (column_x + swatch, swatch_y + swatch),
                swatch_radius,
                (120, 120, 130),
                border_thickness,
            )
            text_x = column_x + swatch + swatch_gap
            text_y = row_top + row_height // 2 + round(6 * unit)
            cv2.putText(
                image,
                name,
                (text_x, text_y),
                font,
                font_scale,
                (46, 26, 26),
                text_thickness,
                cv2.LINE_AA,
            )
        column_x += swatch + swatch_gap + column_text_widths[column] + column_gap
    return image


def save_annotated_detection(
    image: np.ndarray,
    detections: sv.Detections,
    labels: list[str],
    output_path: Path,
    label_mode: str = "auto",
) -> Path:
    """Save a plain PNG with detection boxes and optional class labels.

    Args:
        image: Original BGR image as numpy array.
        detections: Detections to draw.
        labels: Class labels, one per detection.
        output_path: Destination PNG path.
        label_mode: ``"labels"`` draws class labels on boxes, ``"boxes"``
            draws boxes with an in-image class color legend, and ``"auto"``
            picks based on estimated label overlap.

    Returns:
        The path written.
    """
    if label_mode not in LABEL_MODES:
        modes = ", ".join(LABEL_MODES)
        raise ValueError(f"Unknown label_mode {label_mode!r}. Valid modes: {modes}")

    height, width = image.shape[:2]
    if label_mode == "auto":
        crowded = _labels_collide(detections, labels, (width, height))
        label_mode = "boxes" if crowded else "labels"

    annotated = _annotate_image(
        image,
        detections,
        labels,
        label_mode == "labels",
    )
    if label_mode == "boxes":
        annotated = _draw_class_legend_cv2(
            annotated, _legend_entries(detections.class_id, labels)
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), annotated)
    return output_path


def _pluralize(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _fit_figure_rect(
    rect: tuple[float, float, float, float],
    image_shape: tuple[int, ...],
) -> tuple[float, float, float, float]:
    """Center an aspect-true image rect inside a figure-fraction rect."""
    left, bottom, width, height = rect
    rect_width_inches = width * CARD_FIGURE_SIZE[0]
    rect_height_inches = height * CARD_FIGURE_SIZE[1]
    image_height, image_width = image_shape[:2]
    scale = min(rect_width_inches / image_width, rect_height_inches / image_height)
    drawn_width = image_width * scale / CARD_FIGURE_SIZE[0]
    drawn_height = image_height * scale / CARD_FIGURE_SIZE[1]
    return (
        left + (width - drawn_width) / 2,
        bottom + (height - drawn_height) / 2,
        drawn_width,
        drawn_height,
    )


def _scale_for_display(
    image: np.ndarray,
    detections_list: list[sv.Detections],
) -> tuple[np.ndarray, list[sv.Detections]]:
    height, width = image.shape[:2]
    scale = _display_scale(width, height)
    if scale >= 1.0:
        return image, detections_list
    resized = cv2.resize(image, (int(width * scale), int(height * scale)))
    scaled = [
        _scale_detections(detections, scale) if len(detections) > 0 else detections
        for detections in detections_list
    ]
    return resized, scaled


def _boxes_mask(detections: sv.Detections, shape: tuple[int, ...]) -> np.ndarray:
    mask = np.zeros(shape[:2], dtype=bool)
    height, width = shape[:2]
    for x1, y1, x2, y2 in detections.xyxy.astype(int):
        left = max(x1, 0)
        top = max(y1, 0)
        right = min(x2, width)
        bottom = min(y2, height)
        if right > left and bottom > top:
            mask[top:bottom, left:right] = True
    return mask


def _region_diff_image(
    image_rgb: np.ndarray,
    ground_truth: sv.Detections,
    predictions: sv.Detections,
) -> np.ndarray:
    base = image_rgb.astype(np.float32)
    base = base * (1 - _DIFF_BASE_FADE) + 255.0 * _DIFF_BASE_FADE
    gt_mask = _boxes_mask(ground_truth, image_rgb.shape)
    pred_mask = _boxes_mask(predictions, image_rgb.shape)
    regions = (
        (gt_mask & ~pred_mask, _DIFF_ONLY_EXPECTED_COLOR),
        (~gt_mask & pred_mask, _DIFF_ONLY_MODEL_COLOR),
        (gt_mask & pred_mask, _DIFF_BOTH_COLOR),
    )
    for mask, color in regions:
        tint = np.array(to_rgb(color), dtype=np.float32) * 255.0
        base[mask] = base[mask] * (1 - _DIFF_TINT_ALPHA) + tint * _DIFF_TINT_ALPHA
    return base.astype(np.uint8)


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
    """Render a 16:9 hero card for an object detection result.

    Mirrors the QA hero cards: the prediction-annotated image fills the
    left half; the right rail stacks the model identity row with an
    OBJECT DETECTION tag, a spatial ground-truth-versus-model region
    diff (green where only ground-truth boxes cover, red where only
    predicted boxes cover, purple where both agree), a prominent mAP@50
    score, and a footer with expected-versus-predicted object counts
    next to the brand wordmark.

    Args:
        image: Original BGR image as numpy array.
        ground_truth: Ground truth detections.
        predictions: Model predicted detections.
        gt_labels: Class labels for ground truth boxes.
        pred_labels: Class labels for prediction boxes.
        model_id: Identifier of the model that produced the predictions.
        config: Benchmark config for display info.
        map_score: Per-image mAP@50 score if available.
        label_mode: ``"labels"`` draws class labels on boxes, ``"boxes"``
            draws boxes with an in-image class color legend, and ``"auto"``
            picks based on estimated label overlap.

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
        crowded = _labels_collide(predictions, pred_labels, (width, height))
        label_mode = "boxes" if crowded else "labels"

    pred_annotated = _annotate_image(
        image, predictions, pred_labels, label_mode == "labels"
    )
    pred_annotated = pred_annotated[:, :, ::-1]

    figure, image_axes, rail = create_hero_card()
    image_axes.imshow(pred_annotated)
    draw_image_stage(figure, image_axes)

    if label_mode == "boxes":
        _draw_class_legend_overlay(
            figure,
            _fit_figure_rect(HERO_IMAGE_RECT, pred_annotated.shape),
            _legend_entries(predictions.class_id, pred_labels),
        )

    draw_identity_row(
        rail,
        model_info.name,
        lab_info.name,
        lab_info.logo_url,
        "OBJECT DETECTION",
    )

    header_y = 0.856
    rail.text(
        0.0,
        header_y,
        "PREDICTION VS GROUND TRUTH",
        fontsize=10,
        ha="left",
        va="center",
        color=PANEL_LABEL_COLOR,
        font=fonts.bold,
    )
    cursor = 1.0
    for label, color in (
        ("both", _BOTH_TEXT_COLOR),
        ("only in model", FAILURE_TEXT_COLOR),
        ("only in expected", SUCCESS_TEXT_COLOR),
    ):
        cursor = draw_legend_chip(rail, cursor, header_y, label, color)

    scaled_image, (gt_scaled, pred_scaled) = _scale_for_display(
        image, [ground_truth, predictions]
    )
    diff = _region_diff_image(scaled_image[:, :, ::-1], gt_scaled, pred_scaled)

    rail_left, rail_bottom, rail_width, rail_height = HERO_RAIL_RECT
    diff_rect = _fit_figure_rect(
        (
            rail_left,
            rail_bottom + rail_height * 0.285,
            rail_width,
            rail_height * 0.535,
        ),
        diff.shape,
    )
    diff_axes = figure.add_axes(diff_rect)
    diff_axes.imshow(diff)
    diff_axes.set_axis_off()
    draw_image_stage(figure, diff_axes)

    rail.plot([0, 1], [0.262, 0.262], color=DIVIDER_COLOR, lw=1, clip_on=False)

    if map_score is not None:
        rail.text(
            0.5,
            0.212,
            "mAP@50",
            fontsize=9.5,
            ha="center",
            va="center",
            color=PANEL_LABEL_COLOR,
            font=fonts.bold,
        )
        rail.text(
            0.5,
            0.142,
            f"{map_score * 100:.1f}%",
            fontsize=26,
            ha="center",
            va="center",
            color=score_color(map_score),
            font=fonts.display,
        )

    draw_brand_footer(
        rail,
        f"{_pluralize(len(ground_truth), 'expected object')} \u00b7 "
        f"{_pluralize(len(predictions), 'predicted object')}",
    )
    return figure
