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
from PIL import Image
from pyfonts import load_google_font

ROBOFLOW_PURPLE = "#A351FB"
BACKGROUND_COLOR = "#FAFAFA"
BAR_TRACK_COLOR = "#EDEDF0"
DIVIDER_COLOR = "#E8E8ED"
TEXT_PRIMARY = "#1A1A2E"
TEXT_SECONDARY = "#8B8FA3"
SUCCESS_COLOR = "#16A34A"
FAILURE_COLOR = "#DC2626"
LABEL_AREA_WIDTH = 46


@dataclass(frozen=True)
class FontSet:
    """Collection of fonts used across all charts."""

    regular: Any
    medium: Any
    bold: Any
    display: Any


@lru_cache(maxsize=1)
def load_fonts() -> FontSet:
    """Load and cache the standard font set from Google Fonts.

    Returns:
        A ``FontSet`` with Inter (regular, medium, bold) and
        Space Grotesk (display).
    """
    return FontSet(
        regular=load_google_font("Inter"),
        medium=load_google_font("Inter", weight=500),
        bold=load_google_font("Inter", weight=700),
        display=load_google_font("Space Grotesk", weight=700),
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
