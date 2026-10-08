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
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import click

from vlm_exam.config import load_display_config
from vlm_exam.results import is_failed_sample, load_results
from vlm_exam.text_benchmark import summarize_text
from vlm_exam.validation import ERROR, Problem
from vlm_exam.visualization.artifacts import chart_manifest

_DEFAULT_POLICY = Path(__file__).parent / "configs" / "text_release.json"


def validate_text_release(directory: Path, policy: dict[str, Any]) -> list[Problem]:
    """Validate frozen release inventory, provenance, pair coverage and allowed gaps.

    Args:
        directory: Public JSONL collection.
        policy: Independently reviewed release requirements, never inferred at runtime.

    Returns:
        Problems using the same representation as the standard result validator.
    """
    problems: list[Problem] = []
    inventory: Counter[tuple[str, str]] = Counter()
    reference: dict[str, tuple[Any, ...]] | None = None
    expected = {
        (model, effort) for model in policy["models"] for effort in policy["efforts"]
    }
    for path in sorted(directory.glob("*.jsonl")):
        model, effort = path.stem, "unknown"
        try:
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            run = load_results(path)
            model, effort = run.model, run.effort
            inventory[model, effort] += 1
            if (model, effort) not in expected:
                raise ValueError("Unexpected model/effort configuration")
            profile = policy["configurations"][model + "/" + effort]
            if any(
                (row["model"], row["effort"], row["task"], row["timestamp"])
                != (model, effort, "text", run.timestamp)
                for row in rows
            ):
                raise ValueError("Mixed run identity or timestamps")
            identities = sorted(sample.metadata["sample_id"] for sample in run.samples)
            if (
                len(identities) != policy["pairs"]
                or len(set(identities)) != policy["pairs"]
            ):
                raise ValueError("Missing, duplicate or extra image/question pairs")
            selection = hashlib.sha256(
                json.dumps(
                    {"sample_ids": tuple(identities), "pairs": len(identities)}
                ).encode()
            ).hexdigest()
            if selection != policy["selection_hash"]:
                raise ValueError("Unexpected image/question selection")
            failed = []
            content = {}
            for sample in run.samples:
                metadata = sample.metadata
                for field, value in {
                    "dataset_hash": policy["dataset_hash"],
                    "dataset_pairs": policy["pairs"],
                    "text_protocol": policy["protocol"],
                    "inference_hash": profile["inference_hash"],
                    "coordinate_format": profile["coordinate_format"],
                }.items():
                    if metadata.get(field) != value:
                        raise ValueError(f"Unexpected {field}")
                content[metadata["sample_id"]] = (
                    sample.image,
                    sample.expected,
                    metadata["image_sha256"],
                    metadata["question"],
                    metadata["category"],
                    metadata["subset"],
                    metadata.get("scoring_profile"),
                )
                score = metadata.get("score")
                if is_failed_sample(sample):
                    failed.append(metadata["sample_id"])
                    if score is not None:
                        raise ValueError("Failed response must remain unscored")
                elif (
                    type(score) not in (int, float)
                    or not math.isfinite(score)
                    or not 0 <= score <= 1
                ):
                    raise ValueError("Invalid successful-response score")
            if sorted(failed) != profile["allowed_failures"]:
                raise ValueError("Failure sample IDs differ from the accepted gaps")
            if reference is None:
                reference = content
            elif content != reference:
                raise ValueError(
                    "Image, reference or scoring metadata differs between runs"
                )
        except (ValueError, KeyError, TypeError) as error:
            problems.append(
                Problem(
                    model=model,
                    effort=effort,
                    task="text",
                    severity=ERROR,
                    kind="text_release",
                    message=f"{path.name}: {error}",
                )
            )
    for model, effort in sorted(expected):
        count = inventory[model, effort]
        if count != policy["repeats"]:
            problems.append(
                Problem(
                    model=model,
                    effort=effort,
                    task="text",
                    severity=ERROR,
                    kind="runs",
                    message=f"Expected {policy['repeats']} runs, found {count}",
                )
            )
    return problems


def release_index(directory: Path, policy: dict[str, Any]) -> dict[str, Any]:
    """Build the deterministic public results index from unchanged JSONL bytes."""
    runs = []
    for path in sorted(directory.glob("*.jsonl")):
        run = load_results(path)
        runs.append(
            {
                "model": run.model,
                "effort": run.effort,
                "file": path.name,
                "pairs": len(run.samples),
                "successful": sum(not is_failed_sample(s) for s in run.samples),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return {
        **{
            key: policy[key]
            for key in ("dataset_hash", "pairs", "protocol", "repeats", "efforts")
        },
        "models": len(policy["models"]),
        "runs": runs,
    }


def register_text_release_commands(main: click.Group) -> None:
    """Register deterministic publication and validation for the frozen text release."""

    @main.command("text-publish")
    @click.option(
        "--results-directory",
        default="results-text",
        type=click.Path(exists=True, path_type=Path),
    )
    @click.option(
        "--policy",
        default=str(_DEFAULT_POLICY),
        type=click.Path(exists=True, path_type=Path),
    )
    @click.option(
        "--model-labels",
        default="results-text/model-labels.json",
        type=click.Path(exists=True, path_type=Path),
    )
    @click.option(
        "--config", "config_path", type=click.Path(exists=True, path_type=Path)
    )
    @click.option("--output-directory", default="web", type=click.Path(path_type=Path))
    @click.option(
        "--charts-directory",
        default="visualizations/leaderboards",
        type=click.Path(path_type=Path),
    )
    @click.option(
        "--check",
        is_flag=True,
        help="Check release coverage and artifact freshness without writing.",
    )
    def publish(
        results_directory: Path,
        policy: Path,
        model_labels: Path,
        config_path: Path | None,
        output_directory: Path,
        charts_directory: Path,
        check: bool,
    ) -> None:
        """Validate coverage and rebuild public metadata and charts."""
        requirements = json.loads(policy.read_text())
        problems = validate_text_release(results_directory, requirements)
        if problems:
            for problem in problems:
                click.echo(
                    f"{problem.model} {problem.scope}: {problem.message}", err=True
                )
            raise click.ClickException("Text release validation failed")
        summary = summarize_text(results_directory)
        config = load_display_config(config_path, model_labels)
        charts = sorted(
            {
                f"text_{category}_{entry['effort']}.png"
                for entry in summary["configurations"]
                if entry["full_dataset"] and entry["mean"] is not None
                for category in entry["by_category"]
            }
        )
        if check:
            try:
                manifest = chart_manifest(charts_directory, charts, summary, config)
            except (OSError, KeyError, ValueError) as error:
                raise click.ClickException(
                    f"Invalid chart artifacts: {error}"
                ) from error
        else:
            context = click.get_current_context()
            context.invoke(
                main.commands["text-leaderboard"],
                results_directory=str(results_directory),
                output_directory=str(charts_directory),
                allow_incomplete=requirements["repeats"] < 3,
                model_labels=model_labels,
                config_path=config_path,
                models=None,
                group=None,
                effort=None,
            )
            manifest = chart_manifest(charts_directory, charts, summary, config)
        artifacts = {
            output_directory / "text_summary.json": summary,
            output_directory / "text-results.json": release_index(
                results_directory, requirements
            ),
            charts_directory / "text_manifest.json": manifest,
        }
        for path, data in artifacts.items():
            content = json.dumps(data, indent=2) + "\n"
            if check:
                if not path.exists() or path.read_text() != content:
                    raise click.ClickException(
                        f"Outdated artifact: {path}; run text-publish"
                    )
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
        click.echo(
            f"Validated {len(requirements['models'])} models, both efforts; "
            f"public artifacts {'are current' if check else 'rebuilt'}."
        )
