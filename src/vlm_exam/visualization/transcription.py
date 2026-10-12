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

import difflib
import re

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from vlm_exam.tasks.text_scoring import normalize
from vlm_exam.visualization import structured, theme

_WIDTH = 879
_HEIGHT = 604


def _flow(draw: ImageDraw.ImageDraw, runs: list[tuple[str, str]]) -> bool:
    x = 0
    y = 36
    size = 18
    line = size + 5
    for value, kind in runs:
        for token in re.findall(r"\n|[^\S\n]+|\S+", value):
            if token == "\n":
                x = 0
                y += line
                continue
            token = token.expandtabs(4)
            parts = (
                list(token)
                if structured._text_font(token, size).getlength(token) > _WIDTH
                else [token]
            )
            for part in parts:
                f = structured._text_font(part, size)
                width = f.getlength(part)
                if x + width > _WIDTH:
                    x = 0
                    y += line
                    if part.isspace():
                        continue
                if y > _HEIGHT - size - 30:
                    return True
                color = (
                    structured._VALUE_CHANGED
                    if kind == "insert"
                    else theme.TEXT_SECONDARY
                    if kind == "delete"
                    else theme.TEXT_PRIMARY
                )
                draw.text((x, y), part, font=f, fill=color)
                if kind == "delete" and not part.isspace():
                    box = draw.textbbox((x, y), part, font=f)
                    draw.line(
                        (x, (box[1] + box[3]) / 2, x + width, (box[1] + box[3]) / 2),
                        fill=theme.TEXT_SECONDARY,
                    )
                x += width
    return False


def _panel(expected: str, predicted: str) -> Image.Image:
    a, b = normalize(expected), normalize(predicted)
    if not b:
        return structured._panel({}, "")
    panel = Image.new("RGB", (_WIDTH, _HEIGHT), theme.BACKGROUND_COLOR)
    draw = ImageDraw.Draw(panel)
    tokens_a = re.findall(r"\n|[^\S\n]+|\S+", a)
    tokens_b = re.findall(r"\n|[^\S\n]+|\S+", b)
    runs = []
    for kind, i, j, k, end in difflib.SequenceMatcher(
        None, tokens_a, tokens_b, autojunk=False
    ).get_opcodes():
        if kind == "equal":
            runs.append(("".join(tokens_b[k:end]), "equal"))
        else:
            if i < j:
                runs.append(("".join(tokens_a[i:j]), "delete"))
            if (
                i < j
                and k < end
                and not tokens_a[j - 1].isspace()
                and not tokens_b[k].isspace()
            ):
                runs.append((" ", "equal"))
            if k < end:
                runs.append(("".join(tokens_b[k:end]), "insert"))
    structured._header(draw, "TRANSCRIPTION", 0)
    if _flow(draw, runs):
        draw.text(
            (0, _HEIGHT - 26),
            "…",
            font=structured._text_font("…", 20),
            fill=theme.TEXT_SECONDARY,
        )
    return panel


def draw_transcription_panel(rail: plt.Axes, expected: str, predicted: str) -> None:
    """Draw a bounded inline comparison preserving original line breaks.

    Args:
        rail: Native card results axes.
        expected: Reference transcription.
        predicted: Saved model response.
    """
    rail.imshow(
        _panel(expected, predicted),
        extent=(0, 1, 0.279, 0.869),
        aspect="auto",
        interpolation="nearest",
        zorder=0,
    )
    rail.set_xlim(0, 1)
    rail.set_ylim(0, 1)
