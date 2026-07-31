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

import csv
from dataclasses import dataclass
from pathlib import Path

import click


@dataclass(frozen=True)
class PromptBoostRow:
    """Per-class AP@50 gain from image-conditioned over baseline prompts."""

    class_name: str
    ground_truth_count: int
    baseline_ap50: float
    image_conditioned_ap50: float
    delta_ap50: float
    image_conditioned_recall: float
    image_conditioned_failure_mode: str


def _load_per_class_csv(path: Path) -> dict[str, dict[str, str]]:
    with open(path, newline="") as file:
        reader = csv.DictReader(file)
        return {row["class_name"]: row for row in reader}


def compute_prompt_boost_rows(
    baseline_csv: Path,
    image_conditioned_csv: Path,
) -> list[PromptBoostRow]:
    """Compute per-class AP@50 deltas between two reference analysis CSVs.

    Args:
        baseline_csv: Per-class metrics for the base-prompt run.
        image_conditioned_csv: Per-class metrics for the image-conditioned run.

    Returns:
        Rows for every class present in both CSV files.
    """
    baseline_rows = _load_per_class_csv(baseline_csv)
    image_conditioned_rows = _load_per_class_csv(image_conditioned_csv)
    shared_classes = sorted(set(baseline_rows) & set(image_conditioned_rows))
    rows: list[PromptBoostRow] = []
    for class_name in shared_classes:
        baseline = baseline_rows[class_name]
        image_conditioned = image_conditioned_rows[class_name]
        baseline_ap50 = float(baseline["ap50"])
        image_conditioned_ap50 = float(image_conditioned["ap50"])
        rows.append(
            PromptBoostRow(
                class_name=class_name,
                ground_truth_count=int(baseline["ground_truth_count"]),
                baseline_ap50=baseline_ap50,
                image_conditioned_ap50=image_conditioned_ap50,
                delta_ap50=image_conditioned_ap50 - baseline_ap50,
                image_conditioned_recall=float(image_conditioned["recall"]),
                image_conditioned_failure_mode=image_conditioned["failure_mode"],
            )
        )
    return rows


def _format_boost_table(
    rows: list[PromptBoostRow],
    *,
    title: str,
) -> str:
    lines = [
        f"## {title}",
        "",
        (
            "| Rank | Class | GT | Baseline AP@50 | IC AP@50 | "
            "Delta (pp) | IC recall | IC mode |"
        ),
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for rank, row in enumerate(rows, start=1):
        lines.append(
            f"| {rank} | {row.class_name} | {row.ground_truth_count} | "
            f"{row.baseline_ap50:.4f} | {row.image_conditioned_ap50:.4f} | "
            f"{row.delta_ap50 * 100:+.1f} | {row.image_conditioned_recall:.3f} | "
            f"{row.image_conditioned_failure_mode} |"
        )
    return "\n".join(lines) + "\n"


def format_prompt_boost_markdown(
    stable_rows: list[PromptBoostRow],
    low_support_rows: list[PromptBoostRow],
    *,
    model: str,
    min_ground_truth: int,
    top: int,
) -> str:
    """Format ranked prompt-boost tables as markdown."""
    lines = [
        f"# {model}: top {top} class prompt boosts (image-conditioned vs base)",
        "",
        "Delta AP@50 = image-conditioned minus base class-name prompt. "
        f"Stable classes require ground-truth count >= {min_ground_truth}.",
        "",
        _format_boost_table(
            stable_rows[:top],
            title=f"Top {top} stable classes (GT >= {min_ground_truth})",
        ),
    ]
    if low_support_rows:
        lines.append(
            _format_boost_table(
                low_support_rows[:top],
                title=f"Top {top} low-support classes (GT < {min_ground_truth})",
            )
        )
    return "\n".join(lines)


@click.command()
@click.option(
    "--baseline-csv",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Per-class CSV from the base-prompt reference run.",
)
@click.option(
    "--image-conditioned-csv",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Per-class CSV from the image-conditioned reference run.",
)
@click.option(
    "--output",
    required=True,
    type=click.Path(path_type=Path),
    help="Markdown file to write.",
)
@click.option(
    "--model",
    default="sam3",
    help="Model label used in the markdown title.",
)
@click.option(
    "--top",
    default=20,
    type=int,
    help="Number of classes to include in each ranked table.",
)
@click.option(
    "--min-ground-truth",
    default=5,
    type=int,
    help="Minimum ground-truth count for the stable ranking table.",
)
def main(
    baseline_csv: Path,
    image_conditioned_csv: Path,
    output: Path,
    model: str,
    top: int,
    min_ground_truth: int,
) -> None:
    """Rank classes by AP@50 gain from image-conditioned over base prompts."""
    rows = compute_prompt_boost_rows(baseline_csv, image_conditioned_csv)
    stable_rows = sorted(
        [row for row in rows if row.ground_truth_count >= min_ground_truth],
        key=lambda row: row.delta_ap50,
        reverse=True,
    )
    low_support_rows = sorted(
        [row for row in rows if row.ground_truth_count < min_ground_truth],
        key=lambda row: row.delta_ap50,
        reverse=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        format_prompt_boost_markdown(
            stable_rows,
            low_support_rows,
            model=model,
            min_ground_truth=min_ground_truth,
            top=top,
        )
    )

    click.echo(f"Wrote {output}")
    click.echo(f"Top {top} stable class boosts (delta AP@50):")
    for rank, row in enumerate(stable_rows[:top], start=1):
        click.echo(
            f"  {rank:2d}. {row.class_name}: "
            f"{row.baseline_ap50:.4f} -> {row.image_conditioned_ap50:.4f} "
            f"({row.delta_ap50 * 100:+.1f} pp, gt={row.ground_truth_count})"
        )


if __name__ == "__main__":
    main()
