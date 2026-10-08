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

import hashlib
import io
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import click
from PIL import Image

from vlm_exam.config import DisplayConfig, load_display_config
from vlm_exam.metrics import resolve_leaderboard_model_list
from vlm_exam.results import RunResult, SampleResult, is_failed_sample, load_results
from vlm_exam.tasks.text import TEXT_CATEGORIES, TEXT_PROTOCOL, TextSample, TextTask
from vlm_exam.text_benchmark import summarize_text

_CATEGORY_LABELS = {
    "single_string": "Single String Extraction",
    "transcription": "Transcription",
    "structured": "JSON Extraction",
    "localization_recognition": "Text Location & Recognition",
}


def _validated_results(
    run: RunResult, samples: dict[str, TextSample]
) -> dict[str, SampleResult]:
    if run.task != "text":
        raise ValueError("Expected a text run")
    profiles = {
        (
            result.metadata.get("inference_hash"),
            result.metadata.get("coordinate_format"),
        )
        for result in run.samples
    }
    if len(profiles) != 1 or not all(next(iter(profiles))):
        raise ValueError("Missing or mixed inference provenance")
    output = {}
    for result in run.samples:
        identity = result.metadata.get("sample_id")
        sample = samples.get(identity)
        if sample is None or identity in output:
            raise ValueError("Unknown or duplicate image/question pair")
        if (
            result.metadata.get("dataset_hash") != sample.dataset_hash
            or result.metadata.get("text_protocol") != TEXT_PROTOCOL
            or result.metadata.get("question") != sample.prefix
            or result.expected != sample.expected
            or result.image != Path(sample.image_path).name
        ):
            raise ValueError("Results do not match the dataset snapshot and protocol")
        output[identity] = result
    return output


def _panel(
    run: RunResult,
    sample: TextSample,
    result: SampleResult,
    output: Path,
    config: DisplayConfig,
) -> tuple[list[Image.Image], list[str]]:
    import matplotlib.pyplot as plt

    from vlm_exam.visualization.text import plot_text_cards

    task = TextTask(coordinate_format=result.metadata["coordinate_format"])
    width = result.metadata.get("uploaded_width")
    height = result.metadata.get("uploaded_height")
    uploaded = (width, height) if width is not None and height is not None else None
    evaluation = task.evaluate(sample, result.predicted, uploaded_size=uploaded)
    score = result.metadata.get("score")
    if (
        type(score) not in (int, float)
        or not math.isfinite(score)
        or not 0 <= score <= 1
        or abs(evaluation.score - score) > 0.000051
        or result.correct != evaluation.correct
    ):
        raise ValueError("Saved score disagrees with the current deterministic scorer")
    with Image.open(sample.image_path) as source:
        if sample.category != "localization_recognition":
            source.thumbnail((2200, 2200))
        image = source.convert("RGB")
    figures = plot_text_cards(image, sample, result, task, run.model, config, uploaded)
    images, paths = [], []
    fingerprint = hashlib.sha256(
        (result.predicted + json.dumps(result.metadata, sort_keys=True)).encode()
    ).hexdigest()[:8]
    stem = re.sub(
        r"[^A-Za-z0-9_.-]",
        "_",
        f"{run.model}-{run.effort}-{run.timestamp}-{fingerprint}",
    )
    for number, figure in enumerate(figures, 1):
        path = (
            output
            / f"{result.index:03d}-{sample.identity[:12]}-{stem}-page-{number}.png"
        )
        try:
            buffer = io.BytesIO()
            figure.savefig(
                buffer, format="png", dpi=150, facecolor=figure.get_facecolor()
            )
            buffer.seek(0)
            with Image.open(buffer) as rendered:
                panel = rendered.convert("RGB")
        finally:
            plt.close(figure)
        panel.save(path)
        images.append(panel)
        paths.append(path.name)
    return images, paths


def register_text_render_commands(main: click.Group) -> None:
    """Register rendering from saved text results, without making model calls.

    Args:
        main: Existing VLM Exam Click command group.
    """

    @main.command("text-visualize")
    @click.argument("result_file", type=click.Path(exists=True, dir_okay=False))
    @click.option("--compare-file", type=click.Path(exists=True, dir_okay=False))
    @click.option("--dataset-directory", required=True, type=click.Path(exists=True))
    @click.option("--output-directory", default="visualizations")
    @click.option(
        "--config", "config_path", type=click.Path(exists=True, path_type=Path)
    )
    @click.option(
        "--model-labels",
        type=click.Path(exists=True, path_type=Path),
        help="Display-only JSON identities for saved models.",
    )
    @click.option("--category", type=click.Choice(TEXT_CATEGORIES))
    @click.option(
        "--max-samples", default=10, show_default=True, type=click.IntRange(min=1)
    )
    def visualize(
        result_file: str,
        compare_file: str | None,
        dataset_directory: str,
        output_directory: str,
        category: str | None,
        max_samples: int,
        config_path: Path | None,
        model_labels: Path | None,
    ) -> None:
        """Render native PNGs; an optional second run is stacked below the first."""
        samples = {
            sample.identity: sample
            for sample in TextTask().load_samples(dataset_directory)
        }
        runs = [load_results(Path(result_file))]
        if compare_file:
            runs.append(load_results(Path(compare_file)))
        results = [_validated_results(run, samples) for run in runs]
        config = load_display_config(config_path, model_labels)
        if any(run.model not in config.models for run in runs):
            raise click.ClickException(
                "Missing model identity; supply --config or --model-labels"
            )
        if len(runs) == 2 and (
            runs[0].effort != runs[1].effort or results[0].keys() != results[1].keys()
        ):
            raise click.ClickException(
                "Compare the same effort and image/question selection"
            )
        output = Path(output_directory)
        output.mkdir(parents=True, exist_ok=True)
        manifest = []
        for identity, result in results[0].items():
            sample = samples[identity]
            if category and sample.category != category:
                continue
            if any(is_failed_sample(items[identity]) for items in results):
                click.echo(f"Skipping failed pair {identity[:12]}")
                continue
            panels, paths = [], []
            for run, items in zip(runs, results):
                images, names = _panel(run, sample, items[identity], output, config)
                panels.extend(images)
                paths.append(names)
            combined = None
            if compare_file:
                widths = {panel.width for panel in panels}
                if len(widths) != 1:
                    raise ValueError("Native card widths differ")
                stack = Image.new(
                    "RGB", (panels[0].width, sum(panel.height for panel in panels))
                )
                y = 0
                for panel in panels:
                    stack.paste(panel, (0, y))
                    y += panel.height
                combined = f"{result.index:03d}-{identity[:12]}-comparison.png"
                stack.save(output / combined)
            manifest.append(
                {
                    "sample_id": identity,
                    "category": sample.category,
                    "question": sample.question,
                    "expected": sample.expected,
                    "panels": paths,
                    "comparison": combined,
                    "sources": [
                        str(Path(p).resolve()) for p in [result_file, compare_file] if p
                    ],
                }
            )
            click.echo(combined or paths[0][0])
            if len(manifest) == max_samples:
                break
        (output / "text_cards_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
        )
        click.echo(f"Rendered {len(manifest)} pairs into {output}")

    @main.command("text-leaderboard")
    @click.option(
        "--results-directory", default="results", type=click.Path(exists=True)
    )
    @click.option("--output-directory", default="visualizations/leaderboards")
    @click.option(
        "--config", "config_path", type=click.Path(exists=True, path_type=Path)
    )
    @click.option(
        "--model-labels",
        type=click.Path(exists=True, path_type=Path),
        help="Display-only JSON identities for saved models.",
    )
    @click.option("--models", help="Comma-separated model identifiers.")
    @click.option("--group", help="Named leaderboard group; overrides --models.")
    @click.option("--effort", type=click.Choice(("low", "high")))
    @click.option(
        "--allow-incomplete",
        is_flag=True,
        help="Label full-dataset runs with fewer repeats as preliminary.",
    )
    def leaderboard(
        results_directory: str,
        output_directory: str,
        allow_incomplete: bool,
        model_labels: Path | None,
        config_path: Path | None,
        models: str | None,
        group: str | None,
        effort: str | None,
    ) -> None:
        """Render per-category low/high charts from compatible, scored repeats."""
        from vlm_exam.visualization.charts import (
            plot_accuracy_chart,
            save_leaderboard_chart,
        )

        summary = summarize_text(Path(results_directory))
        config = load_display_config(config_path, model_labels)
        selected = resolve_leaderboard_model_list(config, models=models, group=group)
        summary["configurations"] = [
            entry
            for entry in summary["configurations"]
            if (selected is None or entry["model"] in selected)
            and (effort is None or entry["effort"] == effort)
        ]
        groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for entry in summary["configurations"]:
            if not entry["complete"] and not allow_incomplete:
                click.echo(f"Skipping incomplete: {entry['model']} {entry['effort']}")
                continue
            if not entry["full_dataset"] or entry["mean"] is None:
                continue
            groups[
                (entry["dataset_hash"], entry["protocol"], entry["selection_hash"])
            ].append(entry)
        if not groups:
            raise click.ClickException(
                "No complete scored configurations. Use --allow-incomplete "
                "for preliminary full-dataset runs."
            )
        if len(groups) != 1:
            raise click.ClickException(
                "Multiple snapshots or selections; separate result directories "
                "and choose explicit output destinations"
            )
        output = Path(output_directory)
        output.mkdir(parents=True, exist_ok=True)
        rendered = []
        for (dataset, protocol, selection), entries in sorted(groups.items()):
            folder = output
            for effort in ("low", "high"):
                configurations = [
                    entry for entry in entries if entry["effort"] == effort
                ]
                names = [entry["model"] for entry in configurations]
                if len(set(names)) != len(names):
                    raise click.ClickException(
                        "Multiple preprocessing profiles for one model; "
                        "separate result directories"
                    )
                if set(names) - config.models.keys():
                    raise click.ClickException(
                        "Missing model identity; supply --config or --model-labels"
                    )
                for category, label in _CATEGORY_LABELS.items():
                    category_entries = [
                        entry
                        for entry in configurations
                        if category in entry["by_category"]
                    ]
                    if not category_entries:
                        continue
                    scores = {
                        entry["model"]: entry["by_category"][category]["mean"] * 100
                        for entry in category_entries
                    }
                    spread = {
                        entry["model"]: (
                            min(
                                row["mean"]
                                for row in entry["by_category"][category]["runs"]
                            )
                            * 100,
                            max(
                                row["mean"]
                                for row in entry["by_category"][category]["runs"]
                            )
                            * 100,
                        )
                        for entry in category_entries
                        if entry["run_count"] > 1
                    }
                    counts = {
                        entry["model"]: entry["run_count"] for entry in category_entries
                    }
                    preliminary = any(
                        entry["run_count"] == 1 or not entry["complete"]
                        for entry in category_entries
                    )
                    figure = plot_accuracy_chart(
                        scores,
                        config,
                        f"{label} — {effort.title()} Effort"
                        + (" — Preliminary" if preliminary else ""),
                        spread=spread,
                        run_counts=counts,
                    )
                    path = folder / f"text_{category}_{effort}.png"
                    save_leaderboard_chart(figure, path)
                    rendered.append(str(path.relative_to(output)))
                    click.echo(path)
        from vlm_exam.visualization.artifacts import chart_manifest

        manifest = chart_manifest(output, rendered, summary, config)
        (output / "text_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
