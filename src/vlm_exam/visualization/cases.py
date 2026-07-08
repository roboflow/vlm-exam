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
import textwrap

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from PIL import Image

from vlm_exam.config import BenchmarkConfig
from vlm_exam.results import RunResult, SampleResult
from vlm_exam.tasks.qa import normalize_transcription
from vlm_exam.visualization.theme import (
    DIVIDER_COLOR,
    FAILURE_COLOR,
    FAILURE_TEXT_COLOR,
    MONO_ADVANCE_EM,
    PANEL_LABEL_COLOR,
    SUCCESS_COLOR,
    SUCCESS_TEXT_COLOR,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    axes_size_inches,
    create_hero_card,
    draw_brand_footer,
    draw_identity_row,
    draw_image_stage,
    draw_legend_chip,
    draw_status_line,
    load_fonts,
    score_color,
)

_LINE_HEIGHT_EM = 1.55
_DIFF_FONT_LADDER = (14.0, 13.0, 12.0, 11.0, 10.0, 9.5, 9.0, 8.5, 8.0)
_FRAGMENT_FONT_SIZE = 9.0
_FRAGMENT_MERGE_GAP = 30
_SHORT_ANSWER_LIMIT = 40
_ANSWER_FONT_LADDER = (34.0, 30.0, 26.0, 22.0, 19.0, 16.0, 14.0, 12.0, 11.0, 10.0, 9.0)
_QUESTION_FONT_SIZE = 11.0
_QUESTION_MAX_LINES = 5

_MATCH_METHOD_LABELS = {
    "strict": "exact match",
    "count": "count parse",
    "judge": "LLM judge",
    "similarity": "character similarity",
}
# JetBrains Mono lacks the single-glyph temperature signs, which render
# as tofu boxes; substitute the two-character equivalents for display.
_GLYPH_SUBSTITUTIONS = {"\u2103": "\u00b0C", "\u2109": "\u00b0F"}


def _substitute_missing_glyphs(text: str) -> str:
    for missing, replacement in _GLYPH_SUBSTITUTIONS.items():
        text = text.replace(missing, replacement)
    return text


_RAIL_CONTENT_TOP = 0.742
_RAIL_CONTENT_BOTTOM = 0.13
_RAIL_SECTION_HEADER_Y = 0.786

_DiffRun = tuple[str, str | None]

_KIND_TEXT_COLORS = {
    "expected": SUCCESS_TEXT_COLOR,
    "predicted": FAILURE_TEXT_COLOR,
}


def _wrap_question(question: str, width: int, max_lines: int) -> str:
    flattened = " ".join(question.split())
    lines = textwrap.wrap(flattened, width=width)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1][: width - 4] + " ..."
    return "\n".join(lines)


def _characters_per_line(rail: plt.Axes, font_size: float) -> int:
    width_inches, _ = axes_size_inches(rail)
    return int(width_inches * 0.97 / (MONO_ADVANCE_EM * font_size / 72))


def _line_height(rail: plt.Axes, font_size: float) -> float:
    _, height_inches = axes_size_inches(rail)
    return (_LINE_HEIGHT_EM * font_size / 72) / height_inches


def _unified_diff_runs(expected: str, predicted: str) -> list[_DiffRun]:
    matcher = difflib.SequenceMatcher(None, expected, predicted, autojunk=False)
    runs: list[_DiffRun] = []
    for operation, e_start, e_end, p_start, p_end in matcher.get_opcodes():
        if operation == "equal":
            runs.append((expected[e_start:e_end], None))
            continue
        if operation in ("delete", "replace"):
            runs.append((expected[e_start:e_end], "expected"))
        if operation in ("insert", "replace"):
            runs.append((predicted[p_start:p_end], "predicted"))
    return runs


def _chunk_runs(
    runs: list[_DiffRun],
    width: int,
    break_on_newline: bool = True,
) -> list[list[_DiffRun]]:
    lines: list[list[tuple[str, str | None]]] = []
    line: list[tuple[str, str | None]] = []

    def flush() -> None:
        nonlocal line
        lines.append(line)
        line = []

    def emit(character: str, kind: str | None) -> None:
        nonlocal line
        if len(line) >= width:
            split = next(
                (
                    index
                    for index in range(len(line) - 1, width // 2, -1)
                    if line[index][0] == " "
                ),
                None,
            )
            if split is not None and character != " ":
                carried = line[split + 1 :]
                line = line[: split + 1]
                flush()
                line = carried
            else:
                flush()
            if character == " " and not line:
                return
        line.append((character, kind))

    for text, kind in runs:
        for character in text:
            if character == "\n":
                emit("\u00b6", kind)
                if break_on_newline:
                    flush()
            else:
                emit(character, kind)
    if line:
        flush()

    merged_lines: list[list[_DiffRun]] = []
    for characters in lines:
        if not characters:
            continue
        merged: list[list[str | None]] = []
        for character, kind in characters:
            if merged and merged[-1][1] == kind:
                merged[-1][0] += character
            else:
                merged.append([character, kind])
        merged_lines.append([(text, kind) for text, kind in merged])
    return merged_lines


def _draw_run_line(
    rail: plt.Axes,
    x: float,
    y: float,
    runs: list[_DiffRun],
    font_size: float,
    base_color: str = TEXT_PRIMARY,
) -> None:
    fonts = load_fonts()
    width_inches, height_inches = axes_size_inches(rail)
    character_width = (MONO_ADVANCE_EM * font_size / 72) / width_inches
    line_height = (font_size / 72) / height_inches

    column = 0
    for text, kind in runs:
        if not text:
            continue
        color = _KIND_TEXT_COLORS.get(kind, base_color)
        if kind is not None:
            red, green, blue = to_rgb(color)
            rail.add_patch(
                plt.Rectangle(
                    (x + column * character_width, y - line_height * 0.62),
                    len(text) * character_width,
                    line_height * 1.24,
                    facecolor=(red, green, blue, 0.14),
                    edgecolor="none",
                    zorder=1,
                    transform=rail.transAxes,
                    clip_on=False,
                )
            )
        rail.text(
            x + column * character_width,
            y,
            text,
            fontsize=font_size,
            ha="left",
            va="center",
            color=color,
            font=fonts.mono,
            zorder=2,
            parse_math=False,
            transform=rail.transAxes,
        )
        column += len(text)


def _is_paragraph_mark(line: list[_DiffRun]) -> bool:
    return len(line) == 1 and line[0][0] == "\u00b6"


def _draw_wrapped_lines(
    rail: plt.Axes,
    lines: list[list[_DiffRun]],
    top: float,
    font_size: float,
    base_color: str = TEXT_PRIMARY,
) -> float:
    line_height = _line_height(rail, font_size)
    y = top - line_height / 2
    for line in lines:
        # Lone paragraph marks read better as vertical whitespace.
        if _is_paragraph_mark(line):
            y -= 0.55 * line_height
            continue
        _draw_run_line(rail, 0.0, y, line, font_size, base_color)
        y -= line_height
    return y


def _count_drawn_lines(lines: list[list[_DiffRun]]) -> float:
    return sum(0.55 if _is_paragraph_mark(line) else 1.0 for line in lines)


def _draw_run_line_centered(
    rail: plt.Axes,
    y: float,
    runs: list[_DiffRun],
    font_size: float,
    base_color: str = TEXT_PRIMARY,
) -> None:
    width_inches, _ = axes_size_inches(rail)
    character_width = (MONO_ADVANCE_EM * font_size / 72) / width_inches
    total = sum(len(text) for text, _ in runs)
    x = max(0.0, 0.5 - total * character_width / 2)
    _draw_run_line(rail, x, y, runs, font_size, base_color)


def _truncate_lines(
    lines: list[list[_DiffRun]],
    max_lines: int,
) -> list[list[_DiffRun]]:
    if len(lines) <= max_lines:
        return lines
    kept = lines[:max_lines]
    kept[-1] = kept[-1] + [("\u2026", None)]
    return kept


def _compress_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"( ?\n ?)+", "\n", text)


def _extract_fragments(
    expected: str,
    predicted: str,
    context: int,
) -> list[tuple[int, list[_DiffRun], list[_DiffRun]]]:
    matcher = difflib.SequenceMatcher(None, expected, predicted, autojunk=False)
    opcodes = [op for op in matcher.get_opcodes() if op[0] != "equal"]

    merged: list[list[int]] = []
    for _, e_start, e_end, p_start, p_end in opcodes:
        if merged and e_start - merged[-1][1] < _FRAGMENT_MERGE_GAP:
            merged[-1][1] = e_end
            merged[-1][3] = p_end
        else:
            merged.append([e_start, e_end, p_start, p_end])

    fragments: list[tuple[int, list[_DiffRun], list[_DiffRun]]] = []
    for e_start, e_end, p_start, p_end in merged:
        expected_runs: list[_DiffRun] = [
            (_compress_whitespace(expected[max(0, e_start - context) : e_start]), None),
            (_compress_whitespace(expected[e_start:e_end]), "expected"),
            (_compress_whitespace(expected[e_end : e_end + context]), None),
        ]
        predicted_runs: list[_DiffRun] = [
            (
                _compress_whitespace(predicted[max(0, p_start - context) : p_start]),
                None,
            ),
            (_compress_whitespace(predicted[p_start:p_end]), "predicted"),
            (_compress_whitespace(predicted[p_end : p_end + context]), None),
        ]
        fragments.append((e_start, expected_runs, predicted_runs))
    return fragments


def _draw_section_header(
    rail: plt.Axes,
    title: str,
    legend: bool = False,
) -> None:
    fonts = load_fonts()
    y = _RAIL_SECTION_HEADER_Y
    rail.text(
        0.0,
        y,
        title,
        fontsize=10,
        ha="left",
        va="center",
        color=PANEL_LABEL_COLOR,
        font=fonts.bold,
    )
    if legend:
        cursor = 1.0
        for label, kind in (
            ("only in model", "predicted"),
            ("only in expected", "expected"),
        ):
            cursor = draw_legend_chip(rail, cursor, y, label, _KIND_TEXT_COLORS[kind])


def _draw_short_comparison(
    rail: plt.Axes,
    expected: str,
    predicted: str,
) -> None:
    fonts = load_fonts()
    width_inches, _ = axes_size_inches(rail)
    longest = max(len(expected), len(predicted), 1)
    fitted = width_inches * 0.9 / (longest * MONO_ADVANCE_EM / 72)
    font_size = max(min(fitted, 26.0), 11.0)

    if expected == predicted:
        _draw_section_header(rail, "MODEL ANSWER")
        _draw_run_line_centered(
            rail,
            0.50,
            [(predicted, None)],
            font_size,
            base_color=SUCCESS_TEXT_COLOR,
        )
        rail.text(
            0.5,
            0.38,
            "exact match",
            fontsize=10.5,
            ha="center",
            va="center",
            color=PANEL_LABEL_COLOR,
            font=fonts.medium,
        )
        return

    _draw_section_header(rail, "EXPECTED VS MODEL")
    matcher = difflib.SequenceMatcher(None, expected, predicted, autojunk=False)
    expected_runs: list[_DiffRun] = []
    predicted_runs: list[_DiffRun] = []
    for operation, e_start, e_end, p_start, p_end in matcher.get_opcodes():
        expected_runs.append(
            (expected[e_start:e_end], None if operation == "equal" else "expected")
        )
        predicted_runs.append(
            (predicted[p_start:p_end], None if operation == "equal" else "predicted")
        )

    rail.text(
        0.5,
        0.635,
        "EXPECTED",
        fontsize=9.5,
        ha="center",
        va="center",
        color=PANEL_LABEL_COLOR,
        font=fonts.bold,
    )
    _draw_run_line_centered(rail, 0.55, expected_runs, font_size)
    rail.text(
        0.5,
        0.425,
        "MODEL",
        fontsize=9.5,
        ha="center",
        va="center",
        color=PANEL_LABEL_COLOR,
        font=fonts.bold,
    )
    _draw_run_line_centered(rail, 0.34, predicted_runs, font_size)


def _try_draw_full_diff(
    rail: plt.Axes,
    expected: str,
    predicted: str,
) -> bool:
    runs = _unified_diff_runs(expected, predicted)
    has_diff_spans = any(kind is not None for _, kind in runs)

    for font_size in _DIFF_FONT_LADDER:
        line_height = _line_height(rail, font_size)
        max_lines = int((_RAIL_CONTENT_TOP - _RAIL_CONTENT_BOTTOM) / line_height)
        lines = _chunk_runs(runs, _characters_per_line(rail, font_size))
        drawn = _count_drawn_lines(lines)
        if drawn > max_lines:
            continue
        _draw_section_header(rail, "FULL TEXT DIFF", legend=True)
        slack = (max_lines - drawn) * line_height
        top = _RAIL_CONTENT_TOP - min(slack * 0.5, 0.16)
        _draw_wrapped_lines(rail, lines, top, font_size)
        return True

    if has_diff_spans:
        return False

    # Equal transcriptions have no fragments to fall back to, so render
    # the text truncated at the smallest font instead.
    font_size = _DIFF_FONT_LADDER[-1]
    line_height = _line_height(rail, font_size)
    max_lines = int((_RAIL_CONTENT_TOP - _RAIL_CONTENT_BOTTOM) / line_height)
    lines = _truncate_lines(
        _chunk_runs(runs, _characters_per_line(rail, font_size)), max_lines
    )
    _draw_section_header(rail, "FULL TEXT DIFF", legend=True)
    _draw_wrapped_lines(rail, lines, _RAIL_CONTENT_TOP, font_size)
    return True


def _draw_fragment_diff(
    rail: plt.Axes,
    expected: str,
    predicted: str,
) -> None:
    fonts = load_fonts()
    font_size = _FRAGMENT_FONT_SIZE
    line_height = _line_height(rail, font_size)
    label_width = 4
    width = _characters_per_line(rail, font_size) - label_width
    x_text = (
        label_width * (MONO_ADVANCE_EM * font_size / 72) / axes_size_inches(rail)[0]
    )

    fragments = _extract_fragments(expected, predicted, context=width)
    _draw_section_header(rail, "DIFFS", legend=True)

    y = _RAIL_CONTENT_TOP
    shown = 0
    for position, expected_runs, predicted_runs in fragments:
        expected_lines = _truncate_lines(
            _chunk_runs(expected_runs, width, break_on_newline=False), 2
        )
        predicted_lines = _truncate_lines(
            _chunk_runs(predicted_runs, width, break_on_newline=False), 2
        )
        needed = (0.95 + len(expected_lines) + len(predicted_lines) + 0.7) * line_height
        if y - needed < _RAIL_CONTENT_BOTTOM:
            break

        rail.text(
            x_text,
            y,
            f"at character {position:,}",
            fontsize=8,
            ha="left",
            va="center",
            color=PANEL_LABEL_COLOR,
            font=fonts.medium,
        )
        y -= 0.9 * line_height

        rail.text(
            0.0,
            y - line_height * 0.5,
            "GT",
            fontsize=8,
            ha="left",
            va="center",
            color=SUCCESS_TEXT_COLOR,
            font=fonts.bold,
        )
        for line in expected_lines:
            _draw_run_line(rail, x_text, y - line_height * 0.5, line, font_size)
            y -= line_height
        y -= 0.25 * line_height
        rail.text(
            0.0,
            y - line_height * 0.5,
            "AI",
            fontsize=8,
            ha="left",
            va="center",
            color=FAILURE_TEXT_COLOR,
            font=fonts.bold,
        )
        for line in predicted_lines:
            _draw_run_line(rail, x_text, y - line_height * 0.5, line, font_size)
            y -= line_height
        y -= 0.8 * line_height
        shown += 1

    if shown < len(fragments):
        hidden = len(fragments) - shown
        noun = "difference" if hidden == 1 else "differences"
        rail.text(
            0.0,
            max(y, _RAIL_CONTENT_BOTTOM),
            f"+ {hidden} more {noun}",
            fontsize=9,
            ha="left",
            va="center",
            color=TEXT_SECONDARY,
            font=fonts.medium,
        )


def _draw_rail_footer(rail: plt.Axes, expected: str, predicted: str) -> None:
    draw_brand_footer(
        rail,
        f"{len(expected):,} expected chars \u00b7 {len(predicted):,} generated chars",
    )


def plot_transcription_card(
    image: Image.Image,
    expected: str,
    predicted: str,
    score: float,
    model_id: str,
    config: BenchmarkConfig,
) -> plt.Figure:
    """Render a social-friendly hero card for an OCR result.

    The benchmark image fills the left half; the right rail carries the
    model identity, a compact similarity score, and a diff whose format
    adapts to the content: short answers get a large expected-versus-
    model comparison, texts whose full diff fits get a wrapped inline
    diff, and longer texts get multiline diff fragments.

    Args:
        image: The input image shown to the model.
        expected: Ground-truth transcription.
        predicted: Model-produced transcription.
        score: Character similarity in [0, 1].
        model_id: Identifier of the model that produced the answer.
        config: Benchmark config for display info.

    Returns:
        Matplotlib figure.
    """
    model_info = config.models[model_id]
    lab_info = config.labs[model_info.lab]
    expected = normalize_transcription(expected)
    predicted = normalize_transcription(predicted)

    figure, image_axes, rail = create_hero_card()
    image_axes.imshow(image)
    draw_image_stage(figure, image_axes)

    draw_identity_row(rail, model_info.name, lab_info.name, lab_info.logo_url, "OCR")
    draw_status_line(
        rail,
        f"{score * 100:.1f}%",
        "CHARACTER SIMILARITY",
        score,
        score_color(score),
    )

    is_short = (
        len(expected) <= _SHORT_ANSWER_LIMIT
        and len(predicted) <= _SHORT_ANSWER_LIMIT
        and "\n" not in expected
        and "\n" not in predicted
    )
    if is_short:
        _draw_short_comparison(rail, expected, predicted)
    elif not _try_draw_full_diff(rail, expected, predicted):
        _draw_fragment_diff(rail, expected, predicted)

    _draw_rail_footer(rail, expected, predicted)
    return figure


def _draw_rail_verdict(rail: plt.Axes, correct: bool) -> None:
    fonts = load_fonts()
    y = 0.856
    color = SUCCESS_COLOR if correct else FAILURE_COLOR
    rail.text(
        0.0,
        y,
        "CORRECT" if correct else "INCORRECT",
        fontsize=17,
        ha="left",
        va="center",
        color=color,
        font=fonts.display,
        clip_on=False,
    )


def _draw_rail_question(rail: plt.Axes, question: str) -> float:
    fonts = load_fonts()
    _draw_section_header(rail, "QUESTION")
    wrapped = _wrap_question(question, width=64, max_lines=_QUESTION_MAX_LINES)
    text = rail.text(
        0.0,
        _RAIL_CONTENT_TOP,
        wrapped,
        fontsize=_QUESTION_FONT_SIZE,
        ha="left",
        va="top",
        color=TEXT_PRIMARY,
        font=fonts.medium,
        linespacing=1.45,
        parse_math=False,
    )
    # Measure the rendered extent so the divider below keeps a constant
    # gap regardless of line count and descenders.
    figure = rail.get_figure()
    renderer = figure.canvas.get_renderer()
    extent = text.get_window_extent(renderer=renderer)
    bottom = rail.transAxes.inverted().transform((0, extent.y0))[1]
    return bottom - 0.038


def _side_diff_runs(
    expected: str,
    predicted: str,
) -> tuple[list[_DiffRun], list[_DiffRun]]:
    matcher = difflib.SequenceMatcher(None, expected, predicted, autojunk=False)
    expected_runs: list[_DiffRun] = []
    predicted_runs: list[_DiffRun] = []
    for operation, e_start, e_end, p_start, p_end in matcher.get_opcodes():
        expected_runs.append(
            (expected[e_start:e_end], None if operation == "equal" else "expected")
        )
        predicted_runs.append(
            (predicted[p_start:p_end], None if operation == "equal" else "predicted")
        )
    return expected_runs, predicted_runs


def _draw_rail_answer(
    rail: plt.Axes,
    expected: str,
    predicted: str,
    correct: bool,
    top: float,
) -> None:
    fonts = load_fonts()
    rail.plot([0, 1], [top, top], color=DIVIDER_COLOR, lw=1, clip_on=False)
    area_top = top - 0.042
    area_bottom = _RAIL_CONTENT_BOTTOM

    if correct:
        runs_list: list[tuple[str, list[_DiffRun]]] = [
            ("MODEL ANSWER", [(predicted, None)])
        ]
    else:
        expected_runs, predicted_runs = _side_diff_runs(expected, predicted)
        runs_list = [("EXPECTED", expected_runs), ("MODEL", predicted_runs)]

    label_height = 0.034
    block_gap = 0.045

    chosen_font = _ANSWER_FONT_LADDER[-1]
    chosen_lines: list[list[list[_DiffRun]]] = []
    for font_size in _ANSWER_FONT_LADDER:
        line_height = _line_height(rail, font_size)
        blocks = [
            _chunk_runs(runs, _characters_per_line(rail, font_size))
            for _, runs in runs_list
        ]
        # Hero-sized fonts are reserved for answers that stay on one line;
        # wrapped paragraphs at that scale overwhelm the rail.
        if font_size > 26.0 and any(len(block) > 1 for block in blocks):
            continue
        total = (
            sum(_count_drawn_lines(block) for block in blocks) * line_height
            + len(blocks) * label_height
            + (len(blocks) - 1) * block_gap
        )
        if total <= area_top - area_bottom:
            chosen_font = font_size
            chosen_lines = blocks
            break
    else:
        line_height = _line_height(rail, chosen_font)
        budget = area_top - area_bottom - len(runs_list) * label_height
        budget -= (len(runs_list) - 1) * block_gap
        max_lines = max(int(budget / line_height / len(runs_list)), 1)
        chosen_lines = [
            _truncate_lines(
                _chunk_runs(runs, _characters_per_line(rail, chosen_font)),
                max_lines,
            )
            for _, runs in runs_list
        ]

    line_height = _line_height(rail, chosen_font)
    total = (
        sum(_count_drawn_lines(block) for block in chosen_lines) * line_height
        + len(chosen_lines) * label_height
        + (len(chosen_lines) - 1) * block_gap
    )
    # Bias the answer group toward the question rather than dead-center,
    # so short answers do not leave a large gap under the question.
    slack = max(0.0, area_top - area_bottom - total)
    y = area_top - min(slack * 0.5, 0.06)

    for (label, _), block in zip(runs_list, chosen_lines, strict=True):
        rail.text(
            0.0,
            y - label_height / 2,
            label,
            fontsize=9.5,
            ha="left",
            va="center",
            color=PANEL_LABEL_COLOR,
            font=fonts.bold,
        )
        y -= label_height + 0.012
        base_color = SUCCESS_TEXT_COLOR if correct else TEXT_PRIMARY
        y = _draw_wrapped_lines(rail, block, y, chosen_font, base_color)
        y -= block_gap


def _draw_rail_footer_qa(rail: plt.Axes, match_method: str | None) -> None:
    left_text = None
    if match_method:
        method_label = _MATCH_METHOD_LABELS.get(match_method, match_method)
        left_text = f"evaluated via {method_label}"
    draw_brand_footer(rail, left_text)


def plot_qa_card(
    image: Image.Image,
    question: str,
    expected: str,
    predicted: str,
    correct: bool,
    model_id: str,
    config: BenchmarkConfig,
    task_label: str,
    match_method: str | None = None,
) -> plt.Figure:
    """Render a social-friendly hero card for a QA benchmark result.

    Shares the OCR hero layout: the benchmark image fills the left half
    and the right rail stacks the model identity, a compact verdict, the
    question, and the answer. Correct answers show a single green model
    answer; incorrect ones show stacked expected-versus-model blocks
    with character-level diff highlights.

    Args:
        image: The input image shown to the model.
        question: Question text posed to the model.
        expected: Ground-truth answer.
        predicted: Model-produced answer.
        correct: Whether the prediction was evaluated as correct.
        model_id: Identifier of the model that produced the answer.
        config: Benchmark config for display info.
        task_label: Task tag shown in the identity row (e.g. "COUNTING").
        match_method: Evaluation method recorded for the sample.

    Returns:
        Matplotlib figure.
    """
    model_info = config.models[model_id]
    lab_info = config.labs[model_info.lab]
    expected = _substitute_missing_glyphs(" ".join(expected.split()))
    predicted = _substitute_missing_glyphs(" ".join(predicted.split()))

    figure, image_axes, rail = create_hero_card()
    image_axes.imshow(image)
    draw_image_stage(figure, image_axes)

    draw_identity_row(
        rail, model_info.name, lab_info.name, lab_info.logo_url, task_label
    )
    _draw_rail_verdict(rail, correct)
    question_bottom = _draw_rail_question(rail, question)
    _draw_rail_answer(rail, expected, predicted, correct, question_bottom)
    _draw_rail_footer_qa(rail, match_method)
    return figure


_TASK_CARD_LABELS = {"extraction": "DATA EXTRACTION"}


def render_case_card(
    run_result: RunResult,
    sample_result: SampleResult,
    image: Image.Image,
    config: BenchmarkConfig,
) -> plt.Figure:
    """Render the hero card matching a QA sample result's task.

    OCR runs get the transcription diff card; the other QA tasks get
    the question-and-answer card.

    Args:
        run_result: The benchmark run the sample belongs to.
        sample_result: The sample to visualize.
        image: The input image shown to the model.
        config: Benchmark config for display info.

    Returns:
        Matplotlib figure with the case card.
    """
    if run_result.task == "ocr":
        score = sample_result.metadata.get("score")
        return plot_transcription_card(
            image=image,
            expected=sample_result.expected,
            predicted=sample_result.predicted,
            score=score if score is not None else 0.0,
            model_id=run_result.model,
            config=config,
        )
    task_label = _TASK_CARD_LABELS.get(run_result.task, run_result.task.upper())
    return plot_qa_card(
        image=image,
        question=sample_result.metadata.get("question", ""),
        expected=sample_result.expected,
        predicted=sample_result.predicted,
        correct=sample_result.correct,
        model_id=run_result.model,
        config=config,
        task_label=task_label,
        match_method=sample_result.metadata.get("match_method"),
    )
