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

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from itertools import groupby
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ft2font import FT2Font
from PIL import Image, ImageDraw, ImageFont

from vlm_exam.tasks.text_scoring import equal, parse_json
from vlm_exam.visualization import theme

_WIDTH = 879
_HEIGHT = 604
_MISSING = object()
_MAP_MATCH = "#E4E3E9"
_MAP_CHANGED = "#8253BC"
_CARD_MATCH = "#F3F2F5"
_CARD_CHANGED = "#F0EAF8"
_VALUE_CHANGED = "#743AC5"


@dataclass(frozen=True)
class _Field:
    path: tuple[str | int, ...]
    expected: Any
    predicted: Any

    @property
    def matches(self) -> bool:
        return (
            self.expected is not _MISSING
            and self.predicted is not _MISSING
            and equal(self.expected, self.predicted)
        )


@dataclass
class _Tile:
    field: _Field
    x: int
    y: int
    width: int
    height: int
    labels: list[str]
    values: list[list[str]]
    size: int


@lru_cache(maxsize=256)
def _coverage(path: str) -> frozenset[int]:
    try:
        return frozenset(FT2Font(path).get_charmap())
    except (OSError, RuntimeError):
        return frozenset()


@lru_cache(maxsize=1)
def _fallbacks() -> tuple[str, ...]:
    fonts = sorted(
        (
            font
            for font in font_manager.fontManager.ttflist
            if "lastresort" not in font.name.lower().replace(" ", "")
        ),
        key=lambda font: (
            not any(
                name in font.name.lower() for name in ("unicode", "noto", "hiragino")
            ),
            font.style != "normal",
            abs(
                (
                    font_manager.weight_dict.get(font.weight, 400)
                    if isinstance(font.weight, str)
                    else font.weight
                )
                - 400
            ),
            font.name,
            font.fname,
        ),
    )
    return tuple(dict.fromkeys(font.fname for font in fonts))


@lru_cache(maxsize=256)
def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def _text_font(value: str, size: int) -> ImageFont.FreeTypeFont:
    primary = font_manager.findfont(theme.load_fonts().regular)
    points = {ord(character) for character in value}
    if points <= _coverage(primary):
        return _font(primary, size)
    best = primary
    supported = len(points & _coverage(primary))
    for path in _fallbacks():
        coverage = _coverage(path)
        if points <= coverage:
            return _font(path, size)
        count = len(points & coverage)
        if count > supported:
            best, supported = path, count
    return _font(best, size)


def _path(path: tuple[str | int, ...]) -> str:
    label = ""
    for part in path:
        if isinstance(part, int):
            label += f"[{part}]"
        elif part.isidentifier():
            label += ("." if label else "") + part
        else:
            label += "[" + json.dumps(part, ensure_ascii=False) + "]"
    return label or "$"


def _value(value: Any) -> str:
    if value is _MISSING:
        return "[missing]"
    if not isinstance(value, str):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    try:
        json.loads(value)
        ambiguous = True
    except RecursionError:
        ambiguous = True
    except ValueError:
        ambiguous = False
    if not value or ambiguous or value == "[missing]" or value != value.strip():
        return json.dumps(value, ensure_ascii=False)
    return value.replace("\n", " ↵ ").replace("\r", " ↵ ").replace("\t", " ⇥ ")


def _leaves(value: Any) -> dict[tuple[str | int, ...], Any]:
    leaves = {}
    pending: list[tuple[tuple[str | int, ...], Any]] = [((), value)]
    while pending:
        path, current = pending.pop()
        if isinstance(current, dict) and current:
            pending.extend(
                ((*path, key), child) for key, child in reversed(current.items())
            )
        elif isinstance(current, list) and current:
            pending.extend(
                ((*path, index), current[index])
                for index in range(len(current) - 1, -1, -1)
            )
        else:
            leaves[path] = current
    return leaves


def _fields(expected: Any, predicted: Any) -> list[_Field]:
    reference, model = _leaves(expected), _leaves(predicted)
    paths = list(reference) + [path for path in model if path not in reference]
    return [
        _Field(path, reference.get(path, _MISSING), model.get(path, _MISSING))
        for path in paths
    ]


def _wrap(value: str, width: int, size: int, limit: int = 30) -> list[str]:
    font = _text_font(value, size)
    lines = []
    while value and len(lines) < limit:
        low, high = 1, min(len(value), 2048)
        while low < high:
            middle = (low + high + 1) // 2
            if font.getlength(value[:middle]) <= width:
                low = middle
            else:
                high = middle - 1
        lines.append(value[:low])
        value = value[low:]
    if value:
        while lines[-1] and font.getlength(lines[-1] + "…") > width:
            lines[-1] = lines[-1][:-1]
        lines[-1] += "…"
    return lines or [""]


def _layout(
    fields: list[_Field], columns: int, size: int, top: int = 5, limit: int = 30
) -> tuple[list[_Tile], int]:
    width = (_WIDTH - (columns - 1) * 18) // columns
    bottoms = [top] * columns
    tiles = []
    for field in fields:
        labels = _wrap(_path(field.path), width - 28, 14, 2)
        values = (
            []
            if field.matches
            else [_wrap(_value(field.expected), width - 28, size, limit)]
        )
        values.append(_wrap(_value(field.predicted), width - 28, size, limit))
        height = (
            len(labels) * 18 + 38 + sum(len(lines) * (size + 5) for lines in values)
        )
        column = min(range(columns), key=bottoms.__getitem__)
        tiles.append(
            _Tile(
                field,
                column * (width + 18),
                bottoms[column],
                _WIDTH - column * (width + 18) if column == columns - 1 else width,
                height,
                labels,
                values,
                size,
            )
        )
        bottoms[column] += height + 9
    return tiles, max(bottoms)


def _map_layout(fields: list[_Field], height: int) -> list[tuple[int, int, int, bool]]:
    groups = [
        list(group) for _, group in groupby(fields, key=lambda field: field.path[:-1])
    ]
    for size in range(22, 0, -1):
        x, y = 0, 4
        cells = []
        for group in groups:
            width = len(group) * (size + 2) - 2
            if x and x + width > _WIDTH:
                x, y = 0, y + size + 9
            for field in group:
                if x + size > _WIDTH:
                    x, y = 0, y + size + 9
                cells.append((x, y, size, field.matches))
                x += size + 2
            x += 8
        if y + size + 4 <= height:
            return cells
    # Above pixel capacity, a bin is changed if any of its fields differs.
    capacity = (_WIDTH // 3) * ((height - 8) // 3)
    step = math.ceil(len(fields) / capacity)
    return [
        (
            (index % (_WIDTH // 3)) * 3,
            4 + (index // (_WIDTH // 3)) * 3,
            2,
            all(field.matches for field in fields[offset : offset + step]),
        )
        for index, offset in enumerate(range(0, len(fields), step))
    ]


def _draw_tiles(draw: ImageDraw.ImageDraw, tiles: list[_Tile]) -> None:
    for tile in tiles:
        x, y = tile.x, tile.y
        draw.rounded_rectangle(
            (x, y, x + tile.width - 1, y + tile.height - 1),
            radius=7,
            fill=_CARD_MATCH if tile.field.matches else _CARD_CHANGED,
        )
        x, y = x + 14, y + 11
        label_font = _text_font(_path(tile.field.path), 14)
        for line in tile.labels:
            draw.text((x, y), line, fill=theme.TEXT_SECONDARY, font=label_font)
            y += 18
        y += 7
        for index, lines in enumerate(tile.values):
            strike = not tile.field.matches and index == 0
            value = tile.field.expected if strike else tile.field.predicted
            font = _text_font(_value(value), tile.size)
            color = (
                theme.TEXT_SECONDARY
                if strike
                else theme.TEXT_PRIMARY
                if tile.field.matches
                else _VALUE_CHANGED
            )
            for line in lines:
                draw.text((x, y), line, fill=color, font=font)
                if strike:
                    bounds = draw.textbbox((x, y), line, font=font)
                    middle = (bounds[1] + bounds[3]) / 2
                    draw.line(
                        (x, middle, x + font.getlength(line), middle),
                        fill=color,
                        width=1,
                    )
                y += tile.size + 5


def _header(
    draw: ImageDraw.ImageDraw,
    title: str,
    y: int,
    total: int | None = None,
    different: int = 0,
) -> None:
    font = _font(font_manager.findfont(theme.load_fonts().bold), 17)
    draw.text((0, y + 5), title, font=font, fill=theme.PANEL_LABEL_COLOR)
    if total is not None:
        draw.text(
            (font.getlength(title) + 14, y + 6),
            f"{total} fields",
            font=_text_font("fields", 17),
            fill=theme.TEXT_SECONDARY,
        )
        label = f"{different} different"
        font = _text_font(label, 17)
        draw.text(
            (_WIDTH - 1 - font.getlength(label), y + 6),
            label,
            font=font,
            fill=_VALUE_CHANGED,
        )
        return
    font = _text_font("Reference Model", 16)
    model_x = _WIDTH - 1 - font.getlength("Model")
    reference_x = model_x - 20 - font.getlength("Reference")
    draw.text((reference_x, y + 6), "Reference", font=font, fill=theme.TEXT_SECONDARY)
    draw.line(
        (reference_x, y + 16, reference_x + font.getlength("Reference"), y + 16),
        fill=theme.TEXT_SECONDARY,
        width=1,
    )
    draw.text((model_x, y + 6), "Model", font=font, fill=_VALUE_CHANGED)


def _panel(expected: Any, response: str) -> Image.Image:
    panel = Image.new("RGB", (_WIDTH, _HEIGHT), theme.BACKGROUND_COLOR)
    draw = ImageDraw.Draw(panel)
    try:
        predicted, _ = parse_json(response)
    except (ValueError, TypeError, RecursionError):
        message = "Could not parse JSON" if response.strip() else "No response"
        font = _text_font(message, 22)
        left, top, right, bottom = draw.textbbox((0, 0), message, font=font)
        x = (_WIDTH - (right - left)) / 2 - left
        y = (_HEIGHT - (bottom - top)) / 2 - top
        draw.rounded_rectangle(
            (x - 16, y + top - 12, x + right + 16, y + bottom + 12),
            radius=5,
            fill=_CARD_CHANGED,
        )
        draw.text((x, y), message, font=font, fill=_VALUE_CHANGED)
        return panel
    fields = _fields(expected, predicted)
    for columns, size in ((2, 21), (2, 20), (2, 19), (3, 19), (3, 18), (3, 17)):
        tiles, bottom = _layout(fields, columns, size, top=36)
        if bottom <= _HEIGHT:
            _header(draw, "FIELD VALUES", 0)
            _draw_tiles(draw, tiles)
            return panel
    if len(fields) <= 2:
        tiles, _ = _layout(fields, 1, 20, top=36, limit=8 if len(fields) == 1 else 3)
        _header(draw, "FIELD VALUES", 0)
        _draw_tiles(draw, tiles)
        return panel
    map_height = min(180, max(65, math.ceil(len(fields) / 40) * 14))
    values_y = 36 + map_height + 16
    ranked = sorted(fields, key=lambda field: field.matches)
    for count in range(min(12, len(fields)), 0, -1):
        tiles, bottom = _layout(ranked[:count], 2, 20, top=values_y + 36, limit=2)
        if bottom <= _HEIGHT:
            break
    _header(
        draw, "JSON MAP", 0, len(fields), sum(not field.matches for field in fields)
    )
    cells = _map_layout(fields, map_height)
    scale = _WIDTH / max(x + size for x, _, size, _ in cells)
    for x, y, size, matches in cells:
        draw.rectangle(
            (
                round(x * scale),
                y + 36,
                round((x + size) * scale) - 1,
                y + 36 + size - 1,
            ),
            fill=_MAP_MATCH if matches else _MAP_CHANGED,
        )
    draw.line((0, values_y - 10, _WIDTH - 1, values_y - 10), fill="#E2E2E9", width=1)
    _header(draw, "SELECTED VALUES", values_y)
    _draw_tiles(draw, tiles)
    return panel


def draw_structured_panel(rail: plt.Axes, expected: Any, response: str) -> None:
    """Draw a bounded field comparison with a grouped map for large JSON.

    Args:
        rail: Native hero-card right-hand axes.
        expected: Parsed reference JSON.
        response: Model response, including potentially invalid JSON.
    """
    rail.imshow(
        _panel(expected, response),
        extent=(0, 1, 0.279, 0.869),
        aspect="auto",
        interpolation="nearest",
        zorder=0,
    )
    rail.set_xlim(0, 1)
    rail.set_ylim(0, 1)
