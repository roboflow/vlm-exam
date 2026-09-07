# Copyright 2026 Roboflow, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click

from vlm_exam.box_prompting import ARMS
from vlm_exam.box_prompting_round2 import ROUND2_ARMS

_SINGLE_TO_MULTI = {
    "text_box_single": "text_box_multi",
    "drawn_box_single": "drawn_box_multi",
}


@dataclass(frozen=True)
class Run:
    """One scored experiment directory.

    Attributes:
        label: Column label, for example ``Astra low``.
        directory: Experiment root containing ``analysis/summary.json``.
    """

    label: str
    directory: Path


def _load_summary(run: Run) -> dict[str, dict[str, Any]] | None:
    path = run.directory / "analysis" / "summary.json"
    if not path.exists():
        return None
    with open(path) as file:
        return json.load(file)


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _number(value: float | None, digits: int = 0) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _metric_table(
    arms: tuple[str, ...],
    summaries: dict[str, dict[str, dict[str, Any]]],
    metric: str,
    formatter: Any,
) -> list[str]:
    labels = list(summaries)
    lines = ["| Arm | " + " | ".join(labels) + " |", "|---|" + "---:|" * len(labels)]
    for arm in arms:
        cells = []
        for label in labels:
            scored = summaries[label].get(arm)
            cells.append(formatter(scored["metrics"].get(metric)) if scored else "n/a")
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    return lines


def _section(
    heading: str,
    arms: tuple[str, ...],
    summaries: dict[str, dict[str, dict[str, Any]]],
) -> list[str]:
    lines = [f"## {heading}", ""]
    tables = (
        ("Mean image mAP@50", "mean_image_map50", _percent),
        ("Parse failures", "parse_failures", lambda value: _number(value)),
        ("Errors", "errors", lambda value: _number(value)),
        ("Average output tokens", "avg_output_tokens", lambda value: _number(value)),
        ("Average seconds", "avg_seconds", lambda value: _number(value, 1)),
    )
    for title, metric, formatter in tables:
        lines.append(f"### {title}")
        lines.append("")
        lines.extend(_metric_table(arms, summaries, metric, formatter))
        lines.append("")
    return lines


def _round2_extras(summaries: dict[str, dict[str, dict[str, Any]]]) -> list[str]:
    lines = ["### Single vs multi (mAP@50 points)", ""]
    labels = list(summaries)
    lines.append("| Arm pair | " + " | ".join(labels) + " |")
    lines.append("|---|" + "---:|" * len(labels))
    for single, multi in _SINGLE_TO_MULTI.items():
        cells = []
        for label in labels:
            summary = summaries[label]
            if single in summary and multi in summary:
                delta = (
                    summary[multi]["metrics"]["mean_image_map50"]
                    - summary[single]["metrics"]["mean_image_map50"]
                )
                cells.append(f"{delta * 100:+.1f}")
            else:
                cells.append("n/a")
        lines.append(f"| {single} -> {multi} | " + " | ".join(cells) + " |")
    lines.append("")
    for title, metric in (
        ("Positive examples re-detected", "positive_redetections"),
        ("Negative example hits", "negative_hits"),
    ):
        lines.append(f"### {title}")
        lines.append("")
        lines.extend(
            _metric_table(ROUND2_ARMS, summaries, metric, lambda value: _number(value))
        )
        lines.append("")
    return lines


def build_report(round1: list[Run], round2: list[Run]) -> str:
    """Render the cross-model comparison as markdown.

    Args:
        round1: Round 1 experiment directories to compare.
        round2: Round 2 experiment directories to compare.

    Returns:
        Markdown text.
    """
    lines = ["# Box prompting: Qwen3.8-Max vs GPT-6 Astra", ""]
    summaries1 = {
        run.label: summary
        for run in round1
        if (summary := _load_summary(run)) is not None
    }
    summaries2 = {
        run.label: summary
        for run in round2
        if (summary := _load_summary(run)) is not None
    }
    if summaries1:
        lines.extend(
            _section("Round 1: single reference (25 images)", ARMS, summaries1)
        )
    if summaries2:
        lines.extend(
            _section(
                "Round 2: single vs multi, with negatives (50 images)",
                ROUND2_ARMS,
                summaries2,
            )
        )
        lines.extend(_round2_extras(summaries2))
    return "\n".join(lines).rstrip() + "\n"


@click.command()
@click.option(
    "--qwen-root",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("/Users/skalskip/Documents/vlm-exam"),
    show_default=True,
    help="Workspace holding the original Qwen3.8-Max result directories.",
)
@click.option(
    "--astra-root",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("."),
    show_default=True,
)
@click.option(
    "--output-directory",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("results-box-prompting-comparison"),
    show_default=True,
)
def main(qwen_root: Path, astra_root: Path, output_directory: Path) -> None:
    """Write a Qwen vs Astra comparison of the box-prompting experiments."""
    round1 = [
        Run("Qwen3.8-Max low", qwen_root / "results-box-prompting-qwen38-max"),
        Run("Astra low", astra_root / "results-box-prompting-gpt-6-astra-low"),
        Run("Astra high", astra_root / "results-box-prompting-gpt-6-astra-high"),
    ]
    round2 = [
        Run("Qwen3.8-Max low", qwen_root / "results-box-prompting-qwen38-max-round2"),
        Run("Astra low", astra_root / "results-box-prompting-gpt-6-astra-round2-low"),
        Run("Astra high", astra_root / "results-box-prompting-gpt-6-astra-round2-high"),
    ]
    report = build_report(round1, round2)
    output_directory.mkdir(parents=True, exist_ok=True)
    report_path = output_directory / "report.md"
    with open(report_path, "w") as file:
        file.write(report)
    click.echo(report)
    click.echo(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
