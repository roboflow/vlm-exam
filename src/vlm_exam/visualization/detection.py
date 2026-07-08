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
    HERO_RAIL_RECT,
    PANEL_LABEL_COLOR,
    ROBOFLOW_PURPLE,
    SUCCESS_COLOR,
    SUCCESS_TEXT_COLOR,
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

_TARGET_LONG_EDGE = 1024
_LABEL_FONT_SIZE = 16
_LABEL_PADDING = 8
_LABEL_CHAR_WIDTH_FACTOR = 0.62
_COLLISION_FRACTION = 0.5
_MIN_COLLIDING_LABELS = 8
_MAX_LABELED_BOXES = 30

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
    image, (detections,) = _scale_for_display(image, [detections])

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
            draws boxes only, and ``"auto"`` picks based on estimated
            label overlap.

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
