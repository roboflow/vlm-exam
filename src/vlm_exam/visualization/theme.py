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

import io
import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image
from pyfonts import load_google_font

ROBOFLOW_PURPLE = "#A351FB"
BACKGROUND_COLOR = "#FAFAFA"
BAR_TRACK_COLOR = "#EDEDF0"
DIVIDER_COLOR = "#E8E8ED"
TEXT_PRIMARY = "#1A1A2E"
TEXT_SECONDARY = "#8B8FA3"
SUCCESS_COLOR = "#16A34A"
WARNING_COLOR = "#D97706"
FAILURE_COLOR = "#DC2626"
SUCCESS_TEXT_COLOR = "#15803D"
FAILURE_TEXT_COLOR = "#B91C1C"
LABEL_AREA_WIDTH = 46


def score_color(score: float) -> str:
    """Map a [0, 1] score onto the shared three-tier color scale.

    Args:
        score: Score fraction, e.g. OCR character similarity or mAP@50.

    Returns:
        Green for scores of at least 0.95, amber for at least 0.80,
        red otherwise.
    """
    if score >= 0.95:
        return SUCCESS_COLOR
    if score >= 0.80:
        return WARNING_COLOR
    return FAILURE_COLOR


@dataclass(frozen=True)
class FontSet:
    """Collection of fonts used across all charts."""

    regular: Any
    medium: Any
    bold: Any
    display: Any
    mono: Any


@lru_cache(maxsize=1)
def load_fonts() -> FontSet:
    """Load and cache the standard font set from Google Fonts.

    Returns:
        A ``FontSet`` with Inter (regular, medium, bold), Space Grotesk
        (display), and JetBrains Mono (mono).
    """
    return FontSet(
        regular=load_google_font("Inter"),
        medium=load_google_font("Inter", weight=500),
        bold=load_google_font("Inter", weight=700),
        display=load_google_font("Space Grotesk", weight=700),
        mono=load_google_font("JetBrains Mono"),
    )


def _ensure_cairo_on_path() -> None:
    # cairocffi cannot find Homebrew's libcairo on macOS because dyld does
    # not search Homebrew lib dirs by default; ctypes.util.find_library
    # reads DYLD_FALLBACK_LIBRARY_PATH at call time, so extending it here
    # (before cairosvg is first imported) makes the lookup succeed.
    if sys.platform != "darwin":
        return
    homebrew_libs = [
        path for path in ("/opt/homebrew/lib", "/usr/local/lib") if os.path.isdir(path)
    ]
    current = os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "")
    entries = [entry for entry in current.split(":") if entry]
    for path in homebrew_libs:
        if path not in entries:
            entries.append(path)
    os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = ":".join(entries)


@lru_cache(maxsize=32)
def fetch_logo(url: str, size: int = 64) -> Image.Image:
    """Download an SVG logo and convert it to a PIL image.

    Args:
        url: URL of the SVG logo.
        size: Target pixel size (rendered at 2x for sharpness).

    Returns:
        RGBA PIL image.
    """
    _ensure_cairo_on_path()
    import cairosvg

    png_bytes = cairosvg.svg2png(url=url, output_width=size * 2, output_height=size * 2)
    return Image.open(io.BytesIO(png_bytes)).convert("RGBA")


def _perceived_lightness(hex_color: str) -> float:
    stripped = hex_color.lstrip("#")
    red = int(stripped[0:2], 16)
    green = int(stripped[2:4], 16)
    blue = int(stripped[4:6], 16)
    return (0.299 * red + 0.587 * green + 0.114 * blue) / 255


def text_color_for_brand(hex_color: str) -> str:
    """Choose a readable text color for a given brand color.

    Returns a darkened version of light brand colors, or the
    original color if it is already dark enough.

    Args:
        hex_color: Brand color as a hex string (e.g. ``"#D97757"``).

    Returns:
        Hex color string suitable for text on a light background.
    """
    if _perceived_lightness(hex_color) > 0.65:
        stripped = hex_color.lstrip("#")
        red = int(int(stripped[0:2], 16) * 0.55)
        green = int(int(stripped[2:4], 16) * 0.55)
        blue = int(int(stripped[4:6], 16) * 0.55)
        return f"#{red:02x}{green:02x}{blue:02x}"
    return hex_color


def lighten_color(hex_color: str, factor: float = 0.45) -> str:
    """Lighten a hex color by blending toward white.

    Args:
        hex_color: Input color as a hex string.
        factor: Blend factor between 0 (no change) and 1 (white).

    Returns:
        Lightened hex color string.
    """
    stripped = hex_color.lstrip("#")
    red = int(int(stripped[0:2], 16) + (255 - int(stripped[0:2], 16)) * factor)
    green = int(int(stripped[2:4], 16) + (255 - int(stripped[2:4], 16)) * factor)
    blue = int(int(stripped[4:6], 16) + (255 - int(stripped[4:6], 16)) * factor)
    return f"#{red:02x}{green:02x}{blue:02x}"


def draw_rounded_bar(
    axes: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    radius: float,
    **kwargs: Any,
) -> mpatches.FancyBboxPatch:
    """Draw a horizontally rounded bar on the given axes.

    Args:
        axes: Matplotlib axes to draw on.
        x: Left edge x-coordinate.
        y: Center y-coordinate.
        width: Bar width.
        height: Bar height.
        radius: Corner rounding radius.
        **kwargs: Additional arguments passed to ``FancyBboxPatch``.

    Returns:
        The patch object that was added.
    """
    if width < 2 * radius:
        radius = width / 2
    bar = mpatches.FancyBboxPatch(
        (x, y - height / 2),
        width,
        height,
        boxstyle=mpatches.BoxStyle.Round(pad=0, rounding_size=radius),
        **kwargs,
    )
    axes.add_patch(bar)
    return bar


BRAND_LABEL = "Roboflow Playground"
PANEL_LABEL_COLOR = "#4A4E60"
IMAGE_STAGE_COLOR = "#F0F0F3"

CARD_FIGURE_SIZE = (13.33, 7.5)
"""Physical size of the 16:9 hero cards in inches."""

HERO_IMAGE_RECT = (0.025, 0.045, 0.46, 0.91)
"""Figure-fraction rect of the hero-card image half."""

HERO_RAIL_RECT = (0.535, 0.045, 0.44, 0.91)
"""Figure-fraction rect of the hero-card info rail."""

_STATUS_LABEL_OFFSET_INCHES = 1.115
_STATUS_BAR_LEFT_INCHES = 2.698
_STATUS_BAR_TRACK_COLOR = "#E4E4EA"
_CHIP_FONT_SIZE = 8.0
_CHIP_HIGHLIGHT_ALPHA = 0.14
_CHIP_GAP_INCHES = 0.20
# JetBrains Mono has an advance width of exactly 0.6 em, which lets us
# size chips deterministically without a renderer round-trip.
MONO_ADVANCE_EM = 0.6
"""Advance width of the mono font as a fraction of the font size."""


def axes_size_inches(axes: plt.Axes) -> tuple[float, float]:
    """Return the physical (width, height) of an axes in inches.

    Args:
        axes: Matplotlib axes.

    Returns:
        Tuple of width and height in inches.
    """
    figure = axes.get_figure()
    position = axes.get_position()
    return (
        position.width * figure.get_size_inches()[0],
        position.height * figure.get_size_inches()[1],
    )


def create_hero_card() -> tuple[plt.Figure, plt.Axes, plt.Axes]:
    """Create the shared 16:9 hero-card frame.

    Builds the figure with the purple top stripe, an image axes filling
    the left half, and a rail axes on the right configured with unit
    limits for layout in axes fractions.

    Returns:
        Tuple of ``(figure, image_axes, rail)``.
    """
    figure = plt.figure(figsize=CARD_FIGURE_SIZE, facecolor=BACKGROUND_COLOR)
    figure.patches.append(
        plt.Rectangle(
            (0, 0.994),
            1,
            0.006,
            transform=figure.transFigure,
            facecolor=ROBOFLOW_PURPLE,
            edgecolor="none",
            zorder=10,
            clip_on=False,
        )
    )
    image_axes = figure.add_axes(HERO_IMAGE_RECT)
    image_axes.set_axis_off()
    rail = figure.add_axes(HERO_RAIL_RECT)
    rail.set_axis_off()
    rail.set_xlim(0, 1)
    rail.set_ylim(0, 1)
    return figure, image_axes, rail


def draw_legend_chip(
    axes: plt.Axes,
    x_right: float,
    y: float,
    label: str,
    color: str,
) -> float:
    """Draw a right-aligned mono legend chip with a soft highlight.

    Matches the styling of the diff runs so readers map chip colors to
    meaning without a separate visual idiom.

    Args:
        axes: Axes to draw into.
        x_right: Right edge of the chip in axes fractions.
        y: Vertical center of the chip in axes fractions.
        label: Chip text.
        color: Text color; the highlight is the same color at low alpha.

    Returns:
        The x cursor for the next chip to the left.
    """
    fonts = load_fonts()
    width_inches, height_inches = axes_size_inches(axes)
    character_width = (MONO_ADVANCE_EM * _CHIP_FONT_SIZE / 72) / width_inches
    line_height = (_CHIP_FONT_SIZE / 72) / height_inches
    left = x_right - len(label) * character_width
    red, green, blue = to_rgb(color)
    axes.add_patch(
        plt.Rectangle(
            (left, y - line_height * 0.62),
            len(label) * character_width,
            line_height * 1.24,
            facecolor=(red, green, blue, _CHIP_HIGHLIGHT_ALPHA),
            edgecolor="none",
            zorder=1,
            transform=axes.transAxes,
            clip_on=False,
        )
    )
    axes.text(
        left,
        y,
        label,
        fontsize=_CHIP_FONT_SIZE,
        ha="left",
        va="center",
        color=color,
        font=fonts.mono,
        zorder=2,
        transform=axes.transAxes,
    )
    return left - _CHIP_GAP_INCHES / width_inches


def draw_identity_row(
    axes: plt.Axes,
    model_name: str,
    lab_name: str,
    lab_logo_url: str,
    task_label: str,
) -> None:
    """Draw the hero-card identity row with a divider below it.

    Renders the lab logo, model name, and lab name on the left and a
    purple task tag on the right, at the top of the given axes.

    Args:
        axes: Axes to draw into (spanning the card content height).
        model_name: Display name of the model.
        lab_name: Display name of the AI lab.
        lab_logo_url: URL of the lab's SVG logo.
        task_label: Task tag shown on the right (e.g. "OCR").
    """
    fonts = load_fonts()
    y = 0.955
    text_x = 0.0
    try:
        logo_image = fetch_logo(lab_logo_url, size=30)
        image_box = OffsetImage(logo_image, zoom=0.34)
        annotation = AnnotationBbox(
            image_box,
            (0.0, y),
            frameon=False,
            xycoords="axes fraction",
            box_alignment=(0.0, 0.5),
        )
        axes.add_artist(annotation)
        width_inches, _ = axes_size_inches(axes)
        text_x = (30 * 2 * 0.34 / 72 + 0.11) / width_inches
    except Exception:
        pass
    axes.text(
        text_x,
        y + 0.018,
        model_name,
        fontsize=15,
        ha="left",
        va="center",
        color=TEXT_PRIMARY,
        font=fonts.bold,
    )
    axes.text(
        text_x,
        y - 0.022,
        lab_name,
        fontsize=10.5,
        ha="left",
        va="center",
        color=TEXT_SECONDARY,
        font=fonts.medium,
    )
    axes.text(
        1.0,
        y + 0.018,
        task_label,
        fontsize=11.5,
        ha="right",
        va="center",
        color=ROBOFLOW_PURPLE,
        font=fonts.bold,
    )
    axes.plot([0, 1], [0.896, 0.896], color=DIVIDER_COLOR, lw=1, clip_on=False)


def draw_status_line(
    axes: plt.Axes,
    value_text: str,
    label: str,
    fraction: float,
    color: str,
) -> None:
    """Draw the compact hero-card status line with a slim progress bar.

    Args:
        axes: Axes to draw into (spanning the card content height).
        value_text: Headline value (e.g. "97.5%").
        label: Small uppercase label shown next to the value.
        fraction: Bar fill fraction in [0, 1].
        color: Color for the value and bar fill.
    """
    fonts = load_fonts()
    width_inches, _ = axes_size_inches(axes)
    y = 0.856
    axes.text(
        0.0,
        y,
        value_text,
        fontsize=17,
        ha="left",
        va="center",
        color=color,
        font=fonts.display,
        clip_on=False,
    )
    axes.text(
        _STATUS_LABEL_OFFSET_INCHES / width_inches,
        y,
        label,
        fontsize=8,
        ha="left",
        va="center",
        color=TEXT_SECONDARY,
        font=fonts.bold,
    )
    bar_left = _STATUS_BAR_LEFT_INCHES / width_inches
    bar_length = 1.0 - bar_left
    for width, face in (
        (bar_length, _STATUS_BAR_TRACK_COLOR),
        (max(fraction, 0.02) * bar_length, color),
    ):
        axes.add_patch(
            mpatches.FancyBboxPatch(
                (bar_left, y - 0.008),
                width,
                0.016,
                transform=axes.transAxes,
                facecolor=face,
                edgecolor="none",
                boxstyle="round,pad=0.001,rounding_size=0.008",
                clip_on=False,
            )
        )


def draw_image_stage(figure: plt.Figure, axes: plt.Axes) -> None:
    """Draw a rounded neutral stage behind an image axes.

    Args:
        figure: Figure owning the axes.
        axes: Image axes whose bounding box the stage should cover.
    """
    position = axes.get_position()
    # Negative zorder keeps the stage below the axes when the figure
    # composites figure-level patches and axes together.
    figure.patches.append(
        mpatches.FancyBboxPatch(
            (position.x0, position.y0),
            position.width,
            position.height,
            transform=figure.transFigure,
            facecolor=IMAGE_STAGE_COLOR,
            edgecolor="none",
            boxstyle="round,pad=0.004",
            zorder=-1,
            clip_on=False,
        )
    )


def draw_brand_footer(axes: plt.Axes, left_text: str | None = None) -> None:
    """Draw the hero-card footer: divider, optional left text, brand.

    Args:
        axes: Axes to draw into (spanning the card content height).
        left_text: Optional small text on the left side of the footer.
    """
    fonts = load_fonts()
    axes.plot([0, 1], [0.085, 0.085], color=DIVIDER_COLOR, lw=1, clip_on=False)
    if left_text:
        axes.text(
            0.0,
            0.042,
            left_text,
            fontsize=9.5,
            ha="left",
            va="center",
            color=PANEL_LABEL_COLOR,
            font=fonts.medium,
        )
    axes.text(
        1.0,
        0.042,
        BRAND_LABEL,
        fontsize=9.5,
        ha="right",
        va="center",
        color=ROBOFLOW_PURPLE,
        font=fonts.bold,
    )


def add_top_accent(figure: plt.Figure) -> None:
    """Add the Roboflow purple accent line at the top of a figure.

    Args:
        figure: Matplotlib figure to decorate.
    """
    figure.patches.append(
        plt.Rectangle(
            (0, 1),
            1,
            0.005,
            transform=figure.transFigure,
            facecolor=ROBOFLOW_PURPLE,
            edgecolor="none",
            zorder=10,
            clip_on=False,
        )
    )
