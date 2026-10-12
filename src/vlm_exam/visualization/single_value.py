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

import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image, ImageDraw

from vlm_exam.tasks.text_scoring import normalize
from vlm_exam.visualization import structured, theme

_WIDTH = 879
_HEIGHT = 604


def _question_lines(value: str) -> list[str]:
    font = structured._text_font(value, 22)
    lines = []
    current = ""
    for word in value.split():
        candidate = f"{current} {word}".strip()
        if current and font.getlength(candidate) > _WIDTH - 8:
            lines.append(current)
            current = ""
        pieces = (
            structured._wrap(word, _WIDTH - 8, 22, 100)
            if font.getlength(word) > _WIDTH - 8
            else [word]
        )
        for piece in pieces[:-1]:
            lines.append(piece)
        current = f"{current} {pieces[-1]}".strip()
    if current:
        lines.append(current)
    if len(lines) > 7:
        lines = lines[:7]
        while lines[-1] and font.getlength(lines[-1] + "…") > _WIDTH - 8:
            lines[-1] = lines[-1][:-1]
        lines[-1] += "…"
    return lines


def _panel(question: str, expected: str, predicted: str) -> Image.Image:
    panel = Image.new("RGB", (_WIDTH, _HEIGHT), theme.BACKGROUND_COLOR)
    draw = ImageDraw.Draw(panel)
    try:
        parsed = json.loads(question)
        if isinstance(parsed, dict):
            question = str(parsed.get("question", question))
    except (ValueError, TypeError):
        pass
    bold = structured._font(font_manager.findfont(theme.load_fonts().bold), 17)
    draw.text((0, 5), "QUESTION", font=bold, fill=theme.PANEL_LABEL_COLOR)
    font = structured._text_font(question, 22)
    lines = _question_lines(question)
    for index, line in enumerate(lines):
        draw.text((0, 43 + index * 29), line, font=font, fill=theme.TEXT_PRIMARY)
    rule = 43 + len(lines) * 29 + 25
    draw.line((0, rule, _WIDTH - 1, rule), fill="#E2E2E9", width=1)
    structured._header(draw, "VALUE", rule + 18)
    expected, predicted = normalize(expected), normalize(predicted)
    same = expected == predicted
    items = (
        [(predicted or "No response", theme.TEXT_PRIMARY, False)]
        if same
        else [
            (expected or "Empty reference", theme.TEXT_SECONDARY, True),
            (predicted or "No response", structured._VALUE_CHANGED, False),
        ]
    )
    items = [
        (value.replace("\n", " ↵ ").replace("\t", " ⇥ "), color, strike)
        for value, color, strike in items
    ]
    available_top = rule + 70
    gap = 0 if same else 18
    limit = max(1, min(4, (_HEIGHT - available_top - gap) // (31 * len(items))))
    laid = [
        (
            structured._wrap(value, _WIDTH - 40, 24, limit),
            structured._text_font(value, 24),
            color,
            strike,
        )
        for value, color, strike in items
    ]
    height = sum(len(lines) * 31 for lines, _, _, _ in laid) + gap
    y = available_top + max(0, (_HEIGHT - available_top - height) / 2)
    for lines, font, color, strike in laid:
        for line in lines:
            x = (_WIDTH - font.getlength(line)) / 2
            draw.text((x, y), line, font=font, fill=color)
            if strike:
                bounds = draw.textbbox((x, y), line, font=font)
                middle = (bounds[1] + bounds[3]) / 2
                draw.line(
                    (x, middle, x + font.getlength(line), middle), fill=color, width=1
                )
            y += 31
        y += 18
    return panel


def draw_single_value_panel(
    rail: plt.Axes, question: str, expected: str, predicted: str
) -> None:
    """Draw the question and centered single-value comparison.

    Args:
        rail: Native card results axes.
        question: Dataset extraction question.
        expected: Reference value.
        predicted: Saved model response.
    """
    rail.imshow(
        _panel(question, expected, predicted),
        extent=(0, 1, 0.279, 0.869),
        aspect="auto",
        interpolation="nearest",
        zorder=0,
    )
    rail.set_xlim(0, 1)
    rail.set_ylim(0, 1)
