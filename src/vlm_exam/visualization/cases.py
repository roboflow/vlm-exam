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

import textwrap

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import to_rgb
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image

from vlm_exam.config import BenchmarkConfig
from vlm_exam.visualization.theme import (
    BACKGROUND_COLOR,
    DIVIDER_COLOR,
    FAILURE_COLOR,
    ROBOFLOW_PURPLE,
    SUCCESS_COLOR,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    fetch_logo,
    load_fonts,
)


def _draw_question_header(
    axes: plt.Axes,
    question: str,
    model_name: str,
    lab_name: str,
    lab_logo_url: str,
) -> None:
    fonts = load_fonts()
    axes.set_axis_off()
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)

    wrapped_question = textwrap.fill(question, width=65)
    axes.text(
        0.04,
        0.50,
        wrapped_question,
        fontsize=12,
        va="center",
        ha="left",
        color=TEXT_PRIMARY,
        font=fonts.medium,
        linespacing=1.6,
    )
    axes.text(
        0.96,
        0.58,
        model_name,
        fontsize=11.5,
        va="bottom",
        ha="right",
        color=TEXT_PRIMARY,
        font=fonts.bold,
    )
    axes.text(
        0.96,
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
            (0.79, 0.48),
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
        xmin=0.03,
        xmax=0.97,
        color=DIVIDER_COLOR,
        linewidth=1,
    )


def plot_success_card(
    image: Image.Image,
    question: str,
    answer: str,
    model_id: str,
    config: BenchmarkConfig,
) -> plt.Figure:
    """Render a success case card with image, question, and answer.

    Args:
        image: The input image shown to the model.
        question: Question text.
        answer: Model's correct answer.
        model_id: Identifier of the model that produced the answer.
        config: Benchmark config for display info.

    Returns:
        Matplotlib figure.
    """
    fonts = load_fonts()
    model_info = config.models[model_id]
    lab_info = config.labs[model_info.lab]

    figure = plt.figure(figsize=(10, 10), facecolor=BACKGROUND_COLOR)
    figure.patches.append(
        plt.Rectangle(
            (0, 1),
            1,
            0.003,
            transform=figure.transFigure,
            facecolor=ROBOFLOW_PURPLE,
            edgecolor="none",
            zorder=10,
            clip_on=False,
        )
    )

    grid = gridspec.GridSpec(
        5,
        1,
        height_ratios=[0.6, 0.015, 4.5, 0.015, 0.75],
        hspace=0.03,
        left=0.04,
        right=0.96,
        top=0.97,
        bottom=0.03,
    )

    question_axes = figure.add_subplot(grid[0])
    _draw_question_header(
        question_axes,
        question,
        model_info.name,
        lab_info.name,
        lab_info.logo_url,
    )

    divider_top = figure.add_subplot(grid[1])
    _draw_divider_line(divider_top)

    image_axes = figure.add_subplot(grid[2])
    image_axes.imshow(image)
    image_axes.set_axis_off()

    divider_bottom = figure.add_subplot(grid[3])
    _draw_divider_line(divider_bottom)

    answer_axes = figure.add_subplot(grid[4])
    answer_axes.set_axis_off()
    answer_axes.set_xlim(0, 1)
    answer_axes.set_ylim(0, 1)

    position = answer_axes.get_position()
    red, green, blue = to_rgb(SUCCESS_COLOR)
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
            facecolor=SUCCESS_COLOR,
            edgecolor="none",
            zorder=1,
            clip_on=False,
        )
    )

    answer_axes.text(
        0.50,
        0.78,
        "ANSWER",
        fontsize=10,
        ha="center",
        va="center",
        color=TEXT_SECONDARY,
        font=fonts.bold,
    )
    answer_axes.text(
        0.50,
        0.32,
        answer,
        fontsize=26,
        ha="center",
        va="center",
        color=SUCCESS_COLOR,
        font=fonts.display,
    )

    return figure


def plot_failure_card(
    image: Image.Image,
    question: str,
    expected: str,
    predicted: str,
    model_id: str,
    config: BenchmarkConfig,
) -> plt.Figure:
    """Render a failure case card comparing expected vs. predicted.

    Args:
        image: The input image shown to the model.
        question: Question text.
        expected: Ground-truth answer.
        predicted: Model's incorrect answer.
        model_id: Identifier of the model that produced the answer.
        config: Benchmark config for display info.

    Returns:
        Matplotlib figure.
    """
    fonts = load_fonts()
    model_info = config.models[model_id]
    lab_info = config.labs[model_info.lab]

    figure = plt.figure(figsize=(10, 10.5), facecolor=BACKGROUND_COLOR)
    figure.patches.append(
        plt.Rectangle(
            (0, 1),
            1,
            0.003,
            transform=figure.transFigure,
            facecolor=ROBOFLOW_PURPLE,
            edgecolor="none",
            zorder=10,
            clip_on=False,
        )
    )

    grid = gridspec.GridSpec(
        5,
        1,
        height_ratios=[0.6, 0.015, 4.5, 0.015, 0.9],
        hspace=0.03,
        left=0.04,
        right=0.96,
        top=0.97,
        bottom=0.03,
    )

    question_axes = figure.add_subplot(grid[0])
    _draw_question_header(
        question_axes,
        question,
        model_info.name,
        lab_info.name,
        lab_info.logo_url,
    )

    divider_top = figure.add_subplot(grid[1])
    _draw_divider_line(divider_top)

    image_axes = figure.add_subplot(grid[2])
    image_axes.imshow(image)
    image_axes.set_axis_off()

    divider_bottom = figure.add_subplot(grid[3])
    _draw_divider_line(divider_bottom)

    answer_axes = figure.add_subplot(grid[4])
    answer_axes.set_axis_off()
    answer_axes.set_xlim(0, 1)
    answer_axes.set_ylim(0, 1)

    position = answer_axes.get_position()
    red, green, blue = to_rgb(FAILURE_COLOR)
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
            facecolor=FAILURE_COLOR,
            edgecolor="none",
            zorder=1,
            clip_on=False,
        )
    )

    answer_axes.text(
        0.25,
        0.78,
        "EXPECTED",
        fontsize=10,
        ha="center",
        va="center",
        color=TEXT_SECONDARY,
        font=fonts.bold,
    )
    answer_axes.text(
        0.25,
        0.35,
        expected,
        fontsize=24,
        ha="center",
        va="center",
        color=SUCCESS_COLOR,
        font=fonts.display,
    )
    answer_axes.plot(
        [0.50, 0.50],
        [0.10, 0.92],
        color=DIVIDER_COLOR,
        linewidth=1.0,
        transform=answer_axes.transAxes,
        clip_on=False,
    )
    answer_axes.text(
        0.75,
        0.78,
        "MODEL",
        fontsize=10,
        ha="center",
        va="center",
        color=TEXT_SECONDARY,
        font=fonts.bold,
    )
    answer_axes.text(
        0.75,
        0.35,
        predicted,
        fontsize=24,
        ha="center",
        va="center",
        color=FAILURE_COLOR,
        font=fonts.display,
    )

    return figure
