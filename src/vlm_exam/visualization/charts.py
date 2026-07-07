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

from collections.abc import Callable

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

from vlm_exam.config import BenchmarkConfig
from vlm_exam.visualization.theme import (
    BACKGROUND_COLOR,
    BAR_TRACK_COLOR,
    DIVIDER_COLOR,
    LABEL_AREA_WIDTH,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    add_top_accent,
    draw_rounded_bar,
    fetch_logo,
    lighten_color,
    load_fonts,
    text_color_for_brand,
)


def _add_model_label(
    axes: plt.Axes,
    model_id: str,
    config: BenchmarkConfig,
    y: float,
) -> None:
    fonts = load_fonts()
    model_info = config.models[model_id]
    lab_info = config.labs[model_info.lab]
    logo_x = -LABEL_AREA_WIDTH

    try:
        logo_image = fetch_logo(lab_info.logo_url, size=32)
        image_box = OffsetImage(logo_image, zoom=0.35)
        annotation = AnnotationBbox(
            image_box,
            (logo_x + 1.0, y + 0.04),
            frameon=False,
            xycoords=("data", "data"),
            box_alignment=(0.5, 0.5),
        )
        axes.add_artist(annotation)
    except Exception:
        pass

    text_x = logo_x + 7.0
    axes.text(
        text_x,
        y + 0.10,
        model_info.name,
        va="bottom",
        ha="left",
        fontsize=17,
        color=TEXT_PRIMARY,
        font=fonts.bold,
    )
    axes.text(
        text_x,
        y - 0.03,
        lab_info.name,
        va="top",
        ha="left",
        fontsize=12,
        color=TEXT_SECONDARY,
        font=fonts.medium,
    )


def _add_row_divider(
    axes: plt.Axes,
    y: float,
    right_edge: float,
) -> None:
    axes.plot(
        [-1, right_edge],
        [y, y],
        color=DIVIDER_COLOR,
        linewidth=0.8,
        zorder=1,
        clip_on=False,
    )


def _configure_clean_axes(
    axes: plt.Axes,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> None:
    axes.set_xlim(x_min, x_max)
    axes.set_ylim(y_min, y_max)
    axes.set_yticks([])
    axes.xaxis.set_visible(False)
    for spine in axes.spines.values():
        spine.set_visible(False)


def plot_accuracy_chart(
    accuracy: dict[str, float],
    config: BenchmarkConfig,
    title: str,
) -> plt.Figure:
    """Horizontal bar chart showing accuracy per model.

    Args:
        accuracy: Mapping of model identifier to accuracy percentage
            (0-100 scale).
        config: Benchmark config for model/lab display info.
        title: Chart title.

    Returns:
        Matplotlib figure.
    """
    fonts = load_fonts()
    sorted_models = sorted(
        accuracy.keys(), key=lambda model: accuracy[model], reverse=True
    )

    count = len(sorted_models)
    row_spacing = 1.6
    bar_height = 0.50
    corner_radius = bar_height / 2
    bar_max = 100

    figure_height = max(4.0, count * row_spacing + 2.8)
    figure, axes = plt.subplots(figsize=(13, figure_height))
    figure.patch.set_facecolor(BACKGROUND_COLOR)
    axes.set_facecolor(BACKGROUND_COLOR)
    add_top_accent(figure)

    y_positions = [i * row_spacing for i in range(count - 1, -1, -1)]
    total_y_range = (count - 1) * row_spacing

    for index, model_id in enumerate(sorted_models):
        value = accuracy[model_id]
        lab_info = config.labs[config.models[model_id].lab]
        color = lab_info.color
        y = y_positions[index]

        if index > 0:
            _add_row_divider(axes, y + row_spacing / 2, bar_max + 16)

        draw_rounded_bar(
            axes,
            0,
            y,
            bar_max,
            bar_height,
            corner_radius,
            facecolor=BAR_TRACK_COLOR,
            edgecolor="none",
            zorder=2,
        )
        draw_rounded_bar(
            axes,
            0,
            y,
            value,
            bar_height,
            corner_radius,
            facecolor=color,
            edgecolor="none",
            zorder=3,
        )

        axes.text(
            bar_max + 2.0,
            y,
            f"{value:.1f}%",
            va="center",
            ha="left",
            fontsize=19,
            color=text_color_for_brand(color),
            font=fonts.display,
        )

        _add_model_label(axes, model_id, config, y)

    _configure_clean_axes(axes, -LABEL_AREA_WIDTH - 2, 118, -1.0, total_y_range + 2.2)

    axes.text(
        -LABEL_AREA_WIDTH - 2,
        total_y_range + 1.8,
        title,
        fontsize=28,
        color=TEXT_PRIMARY,
        font=fonts.display,
        va="bottom",
        ha="left",
    )

    plt.tight_layout(rect=[0.01, 0.03, 0.99, 0.97])
    return figure


def plot_metric_chart(
    metric: dict[str, float],
    config: BenchmarkConfig,
    title: str,
    format_value: Callable[[float], str],
    sort_ascending: bool = True,
    full_scale: float | None = None,
) -> plt.Figure:
    """Single-bar horizontal chart for one metric across models.

    Args:
        metric: Mapping of model identifier to metric value.
        config: Benchmark config for model/lab display info.
        title: Chart title.
        format_value: Callable to format the metric value as a string.
        sort_ascending: Whether to sort models from lowest to highest.
        full_scale: Value that corresponds to a full-length bar. When
            ``None``, bars are scaled relative to the highest value.

    Returns:
        Matplotlib figure.
    """
    fonts = load_fonts()
    sorted_models = sorted(
        metric.keys(),
        key=lambda model: metric[model],
        reverse=not sort_ascending,
    )

    count = len(sorted_models)
    row_spacing = 1.6
    bar_height = 0.50
    corner_radius = bar_height / 2

    reference = full_scale if full_scale is not None else max(metric.values())
    scale = 100.0 / reference if reference > 0 else 1.0
    bar_max = 100

    figure_height = max(4.0, count * row_spacing + 2.8)
    figure, axes = plt.subplots(figsize=(13, figure_height))
    figure.patch.set_facecolor(BACKGROUND_COLOR)
    axes.set_facecolor(BACKGROUND_COLOR)
    add_top_accent(figure)

    y_positions = [i * row_spacing for i in range(count - 1, -1, -1)]
    total_y_range = (count - 1) * row_spacing

    for index, model_id in enumerate(sorted_models):
        value = metric[model_id]
        bar_width = value * scale
        lab_info = config.labs[config.models[model_id].lab]
        color = lab_info.color
        y = y_positions[index]

        if index > 0:
            _add_row_divider(axes, y + row_spacing / 2, bar_max + 16)

        draw_rounded_bar(
            axes,
            0,
            y,
            bar_max,
            bar_height,
            corner_radius,
            facecolor=BAR_TRACK_COLOR,
            edgecolor="none",
            zorder=2,
        )
        draw_rounded_bar(
            axes,
            0,
            y,
            bar_width,
            bar_height,
            corner_radius,
            facecolor=color,
            edgecolor="none",
            zorder=3,
        )

        axes.text(
            bar_max + 2.0,
            y,
            format_value(value),
            va="center",
            ha="left",
            fontsize=19,
            color=text_color_for_brand(color),
            font=fonts.display,
        )

        _add_model_label(axes, model_id, config, y)

    _configure_clean_axes(
        axes,
        -LABEL_AREA_WIDTH - 2,
        118,
        -1.0,
        total_y_range + 2.2,
    )

    axes.text(
        -LABEL_AREA_WIDTH - 2,
        total_y_range + 1.8,
        title,
        fontsize=28,
        color=TEXT_PRIMARY,
        font=fonts.display,
        va="bottom",
        ha="left",
    )

    plt.tight_layout(rect=[0.01, 0.03, 0.99, 0.97])
    return figure


def plot_dual_effort_chart(
    metric_high: dict[str, float],
    metric_low: dict[str, float],
    config: BenchmarkConfig,
    title: str,
    format_value: Callable[[float], str],
    sort_ascending: bool = True,
) -> plt.Figure:
    """Horizontal bar chart with paired high/low effort bars per model.

    Args:
        metric_high: Model-to-value mapping for high effort.
        metric_low: Model-to-value mapping for low effort.
        config: Benchmark config for model/lab display info.
        title: Chart title.
        format_value: Callable to format metric values.
        sort_ascending: Whether to sort from lowest to highest.

    Returns:
        Matplotlib figure.
    """
    fonts = load_fonts()
    sorted_models = sorted(
        metric_high.keys(),
        key=lambda model: metric_high[model],
        reverse=not sort_ascending,
    )

    count = len(sorted_models)
    row_spacing = 1.8
    bar_height = 0.24
    bar_gap = 0.04
    corner_radius = bar_height / 2

    max_value = max(max(metric_high.values()), max(metric_low.values()))
    scale = 100.0 / max_value if max_value > 0 else 1.0
    bar_max = 100

    figure_height = max(4.5, count * row_spacing + 3.2)
    figure, axes = plt.subplots(figsize=(13, figure_height))
    figure.patch.set_facecolor(BACKGROUND_COLOR)
    axes.set_facecolor(BACKGROUND_COLOR)
    add_top_accent(figure)

    y_positions = [i * row_spacing for i in range(count - 1, -1, -1)]
    total_y_range = (count - 1) * row_spacing

    for index, model_id in enumerate(sorted_models):
        value_high = metric_high[model_id]
        value_low = metric_low[model_id]
        bar_high = value_high * scale
        bar_low = value_low * scale

        lab_info = config.labs[config.models[model_id].lab]
        color = lab_info.color
        color_light = lighten_color(color)
        y = y_positions[index]

        if index > 0:
            _add_row_divider(axes, y + row_spacing / 2, bar_max + 16)

        y_high = y + bar_gap / 2 + bar_height / 2
        y_low = y - bar_gap / 2 - bar_height / 2

        draw_rounded_bar(
            axes,
            0,
            y_high,
            bar_max,
            bar_height,
            corner_radius,
            facecolor=BAR_TRACK_COLOR,
            edgecolor="none",
            zorder=2,
        )
        draw_rounded_bar(
            axes,
            0,
            y_high,
            bar_high,
            bar_height,
            corner_radius,
            facecolor=color,
            edgecolor="none",
            zorder=3,
        )
        draw_rounded_bar(
            axes,
            0,
            y_low,
            bar_max,
            bar_height,
            corner_radius,
            facecolor=BAR_TRACK_COLOR,
            edgecolor="none",
            zorder=2,
        )
        draw_rounded_bar(
            axes,
            0,
            y_low,
            bar_low,
            bar_height,
            corner_radius,
            facecolor=color_light,
            edgecolor="none",
            zorder=3,
        )

        axes.text(
            bar_max + 2.0,
            y_high,
            format_value(value_high),
            va="center",
            ha="left",
            fontsize=15,
            color=text_color_for_brand(color),
            font=fonts.display,
        )
        axes.text(
            bar_max + 2.0,
            y_low,
            format_value(value_low),
            va="center",
            ha="left",
            fontsize=15,
            color=TEXT_SECONDARY,
            font=fonts.display,
        )

        _add_model_label(axes, model_id, config, y)

    pill_y = total_y_range + 2.05
    swatch_width, swatch_height = 6, 0.18
    draw_rounded_bar(
        axes,
        64,
        pill_y,
        swatch_width,
        swatch_height,
        swatch_height / 2,
        facecolor=TEXT_PRIMARY,
        edgecolor="none",
        zorder=5,
    )
    axes.text(
        64 + swatch_width + 1.2,
        pill_y,
        "High",
        va="center",
        ha="left",
        fontsize=11,
        color=TEXT_PRIMARY,
        font=fonts.medium,
    )
    draw_rounded_bar(
        axes,
        84,
        pill_y,
        swatch_width,
        swatch_height,
        swatch_height / 2,
        facecolor=DIVIDER_COLOR,
        edgecolor="none",
        zorder=5,
    )
    axes.text(
        84 + swatch_width + 1.2,
        pill_y,
        "Low",
        va="center",
        ha="left",
        fontsize=11,
        color=TEXT_SECONDARY,
        font=fonts.medium,
    )

    _configure_clean_axes(
        axes,
        -LABEL_AREA_WIDTH - 2,
        118,
        -1.2,
        total_y_range + 2.4,
    )

    axes.text(
        -LABEL_AREA_WIDTH - 2,
        total_y_range + 1.8,
        title,
        fontsize=28,
        color=TEXT_PRIMARY,
        font=fonts.display,
        va="bottom",
        ha="left",
    )

    plt.tight_layout(rect=[0.01, 0.03, 0.99, 0.97])
    return figure


def plot_cost_bar_chart(
    cost_by_model: dict[str, float],
    config: BenchmarkConfig,
    title: str = "Average Cost per Image",
) -> plt.Figure:
    """Vertical bar chart comparing per-image cost across models.

    Args:
        cost_by_model: Mapping of model identifier to average cost.
        config: Benchmark config for model/lab display info.
        title: Chart title.

    Returns:
        Matplotlib figure.
    """
    import numpy as np

    fonts = load_fonts()
    model_ids = list(cost_by_model.keys())
    x_positions = np.arange(len(model_ids))
    bar_width = 0.5

    figure, axes = plt.subplots(figsize=(13, 6))
    figure.patch.set_facecolor(BACKGROUND_COLOR)
    axes.set_facecolor(BACKGROUND_COLOR)
    add_top_accent(figure)

    for index, model_id in enumerate(model_ids):
        lab_info = config.labs[config.models[model_id].lab]
        color = lab_info.color
        cost = cost_by_model[model_id]

        axes.bar(
            x_positions[index],
            cost,
            bar_width,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        axes.text(
            x_positions[index],
            cost + axes.get_ylim()[1] * 0.01,
            f"${cost:.4f}",
            ha="center",
            va="bottom",
            fontsize=11,
            color=text_color_for_brand(color),
            font=fonts.bold,
        )

    labels = [config.models[model_id].name for model_id in model_ids]
    axes.set_xticks(x_positions)
    axes.set_xticklabels(
        labels,
        fontsize=13,
        font=fonts.bold,
        color=TEXT_PRIMARY,
    )

    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    axes.spines["left"].set_color(DIVIDER_COLOR)
    axes.spines["bottom"].set_color(DIVIDER_COLOR)
    axes.tick_params(axis="y", colors=TEXT_SECONDARY, labelsize=10)
    axes.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"${value:.4f}"))
    axes.set_ylabel("")
    axes.grid(axis="y", color=DIVIDER_COLOR, linewidth=0.6, zorder=0)

    axes.set_title(
        title,
        fontsize=28,
        color=TEXT_PRIMARY,
        font=fonts.display,
        loc="left",
        pad=20,
    )

    plt.tight_layout(rect=[0.01, 0.03, 0.99, 0.97])
    return figure


def plot_combined_metrics_chart(
    tokens_high: dict[str, float],
    tokens_low: dict[str, float],
    cost_high: dict[str, float],
    cost_low: dict[str, float],
    time_high: dict[str, float],
    time_low: dict[str, float],
    config: BenchmarkConfig,
    sort_by: str = "tokens",
) -> plt.Figure:
    """Three-column chart showing tokens, cost, and time per model.

    Args:
        tokens_high: Average tokens per image at high effort.
        tokens_low: Average tokens per image at low effort.
        cost_high: Average cost per image at high effort.
        cost_low: Average cost per image at low effort.
        time_high: Average inference time at high effort.
        time_low: Average inference time at low effort.
        config: Benchmark config for model/lab display info.
        sort_by: Column to sort by (``"tokens"``, ``"cost"``,
            or ``"time"``).

    Returns:
        Matplotlib figure.
    """
    fonts = load_fonts()
    sort_map = {"tokens": tokens_high, "cost": cost_high, "time": time_high}
    sorted_models = sorted(
        tokens_high.keys(),
        key=lambda model: sort_map.get(sort_by, tokens_high)[model],
    )

    count = len(sorted_models)
    row_spacing = 1.8
    bar_height = 0.20
    bar_gap = 0.04
    corner_radius = bar_height / 2

    column_width = 22
    column_gap = 14
    column_starts = [
        0,
        column_width + column_gap,
        2 * (column_width + column_gap),
    ]

    columns = [
        {
            "high": tokens_high,
            "low": tokens_low,
            "format": lambda value: f"{value:,.0f}",
            "header": "Tokens",
        },
        {
            "high": cost_high,
            "low": cost_low,
            "format": lambda value: (
                f"${value:.4f}" if value >= 0.001 else f"${value:.5f}"
            ),
            "header": "Cost",
        },
        {
            "high": time_high,
            "low": time_low,
            "format": lambda value: f"{value:.1f}s",
            "header": "Time",
        },
    ]

    scales = []
    for column in columns:
        max_value = max(max(column["high"].values()), max(column["low"].values()))
        scales.append(column_width / max_value if max_value > 0 else 1.0)

    figure_height = max(5.0, count * row_spacing + 3.5)
    figure, axes = plt.subplots(figsize=(16, figure_height))
    figure.patch.set_facecolor(BACKGROUND_COLOR)
    axes.set_facecolor(BACKGROUND_COLOR)
    add_top_accent(figure)

    y_positions = [i * row_spacing for i in range(count - 1, -1, -1)]
    total_y_range = (count - 1) * row_spacing
    chart_right = column_starts[2] + column_width

    for column_index, (column, column_scale, x_start) in enumerate(
        zip(columns, scales, column_starts)
    ):
        header_x = x_start + column_width / 2
        header_y = total_y_range + 1.35
        axes.text(
            header_x,
            header_y,
            column["header"],
            va="bottom",
            ha="center",
            fontsize=14,
            color=TEXT_PRIMARY,
            font=fonts.bold,
        )
        axes.plot(
            [x_start, x_start + column_width],
            [header_y - 0.15, header_y - 0.15],
            color=DIVIDER_COLOR,
            linewidth=1.2,
            zorder=1,
            clip_on=False,
        )

        if column_index > 0:
            separator_x = x_start - column_gap / 2
            axes.plot(
                [separator_x, separator_x],
                [-0.8, total_y_range + 1.1],
                color=DIVIDER_COLOR,
                linewidth=0.6,
                zorder=1,
                clip_on=False,
                linestyle=(0, (8, 6)),
            )

    for index, model_id in enumerate(sorted_models):
        model_info = config.models[model_id]
        lab_info = config.labs[model_info.lab]
        color = lab_info.color
        color_light = lighten_color(color)
        y = y_positions[index]

        if index > 0:
            _add_row_divider(axes, y + row_spacing / 2, chart_right + 12)

        logo_x = -LABEL_AREA_WIDTH
        try:
            logo_image = fetch_logo(lab_info.logo_url, size=32)
            image_box = OffsetImage(logo_image, zoom=0.35)
            annotation = AnnotationBbox(
                image_box,
                (logo_x + 1.0, y + 0.04),
                frameon=False,
                xycoords=("data", "data"),
                box_alignment=(0.5, 0.5),
            )
            axes.add_artist(annotation)
        except Exception:
            pass

        text_x = logo_x + 7.0
        axes.text(
            text_x,
            y + 0.10,
            model_info.name,
            va="bottom",
            ha="left",
            fontsize=15,
            color=TEXT_PRIMARY,
            font=fonts.bold,
        )
        axes.text(
            text_x,
            y - 0.03,
            lab_info.name,
            va="top",
            ha="left",
            fontsize=11,
            color=TEXT_SECONDARY,
            font=fonts.medium,
        )

        for column_index, (column, column_scale, x_start) in enumerate(
            zip(columns, scales, column_starts)
        ):
            value_high = column["high"][model_id]
            value_low = column["low"][model_id]
            bar_high = value_high * column_scale
            bar_low = value_low * column_scale

            y_high = y + bar_gap / 2 + bar_height / 2
            y_low = y - bar_gap / 2 - bar_height / 2

            draw_rounded_bar(
                axes,
                x_start,
                y_high,
                column_width,
                bar_height,
                corner_radius,
                facecolor=BAR_TRACK_COLOR,
                edgecolor="none",
                zorder=2,
            )
            draw_rounded_bar(
                axes,
                x_start,
                y_high,
                bar_high,
                bar_height,
                corner_radius,
                facecolor=color,
                edgecolor="none",
                zorder=3,
            )
            draw_rounded_bar(
                axes,
                x_start,
                y_low,
                column_width,
                bar_height,
                corner_radius,
                facecolor=BAR_TRACK_COLOR,
                edgecolor="none",
                zorder=2,
            )
            draw_rounded_bar(
                axes,
                x_start,
                y_low,
                bar_low,
                bar_height,
                corner_radius,
                facecolor=color_light,
                edgecolor="none",
                zorder=3,
            )

            axes.text(
                x_start + column_width + 0.8,
                y_high,
                column["format"](value_high),
                va="center",
                ha="left",
                fontsize=11,
                color=text_color_for_brand(color),
                font=fonts.display,
            )
            axes.text(
                x_start + column_width + 0.8,
                y_low,
                column["format"](value_low),
                va="center",
                ha="left",
                fontsize=11,
                color=TEXT_SECONDARY,
                font=fonts.display,
            )

    pill_y = total_y_range + 2.05
    swatch_width, swatch_height = 5, 0.16
    pill_x_high = chart_right - 28
    draw_rounded_bar(
        axes,
        pill_x_high,
        pill_y,
        swatch_width,
        swatch_height,
        swatch_height / 2,
        facecolor=TEXT_PRIMARY,
        edgecolor="none",
        zorder=5,
    )
    axes.text(
        pill_x_high + swatch_width + 1.0,
        pill_y,
        "High",
        va="center",
        ha="left",
        fontsize=10,
        color=TEXT_PRIMARY,
        font=fonts.medium,
    )
    pill_x_low = chart_right - 12
    draw_rounded_bar(
        axes,
        pill_x_low,
        pill_y,
        swatch_width,
        swatch_height,
        swatch_height / 2,
        facecolor=DIVIDER_COLOR,
        edgecolor="none",
        zorder=5,
    )
    axes.text(
        pill_x_low + swatch_width + 1.0,
        pill_y,
        "Low",
        va="center",
        ha="left",
        fontsize=10,
        color=TEXT_SECONDARY,
        font=fonts.medium,
    )

    _configure_clean_axes(
        axes,
        -LABEL_AREA_WIDTH - 2,
        chart_right + 14,
        -1.2,
        total_y_range + 2.6,
    )

    axes.text(
        -LABEL_AREA_WIDTH - 2,
        total_y_range + 1.8,
        "Model Efficiency Overview",
        fontsize=26,
        color=TEXT_PRIMARY,
        font=fonts.display,
        va="bottom",
        ha="left",
    )

    plt.tight_layout(rect=[0.01, 0.03, 0.99, 0.97])
    return figure
