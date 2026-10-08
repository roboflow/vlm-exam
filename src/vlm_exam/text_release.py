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

from vlm_exam.config import BenchmarkConfig
from vlm_exam.results import is_failed_sample, load_results
from vlm_exam.validation import ERROR, Problem

_DEFAULT_POLICY = Path(__file__).parent / "configs" / "text_release.json"


def text_result_paths(directory: Path) -> list[Path]:
    """Select text runs from the shared collection, retaining malformed text files.

    A text filename or the first row's task identifies a candidate. Validation
    checks every row, so malformed text runs cannot silently disappear.
    """
    paths = []
    for path in sorted(directory.glob("*.jsonl")):
        if path.name.startswith("text_"):
            paths.append(path)
            continue
        try:
            with path.open() as source:
                row = json.loads(source.readline())
            if isinstance(row, dict) and row.get("task") == "text":
                paths.append(path)
        except (ValueError, OSError):
            continue
    return paths


def requires_text_release(directory: Path, config: BenchmarkConfig) -> bool:
    """Require reviewed text coverage when configured or present in results.

    The registered release inventory keeps validation active even when every
    text file is removed. Custom configurations without that inventory can
    still summarize the original tasks alone.
    """
    policy = load_text_release_policy()
    return bool(text_result_paths(directory)) or set(policy["models"]).issubset(
        config.models
    )


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
    for path in text_result_paths(directory):
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


def load_text_release_policy() -> dict[str, Any]:
    """Return the reviewed frozen text release requirements."""
    return json.loads(_DEFAULT_POLICY.read_text())


def check_text_charts(
    results_directory: Path, charts_directory: Path, config_path: Path | None = None
) -> None:
    """Verify text PNGs and their source fingerprints during summary validation.

    Args:
        results_directory: Frozen public text results.
        charts_directory: Shared leaderboard directory.
        config_path: Optional model configuration supplying display identities.

    Raises:
        ValueError: The committed chart manifest is stale.
    """
    from vlm_exam.config import load_display_config
    from vlm_exam.text_benchmark import summarize_text
    from vlm_exam.visualization.artifacts import chart_manifest

    summary = summarize_text(results_directory)
    config = load_display_config(config_path)

    charts = sorted(
        {
            f"text_{category}_{entry['effort']}.png"
            for entry in summary["configurations"]
            if entry["full_dataset"] and entry["mean"] is not None
            for category in entry["by_category"]
        }
    )
    expected = chart_manifest(charts_directory, charts, summary, config)
    path = charts_directory / "text_manifest.json"
    if json.loads(path.read_text()) != expected:
        raise ValueError(
            f"Stale chart manifest: {path}; regenerate "
            "text-leaderboard --allow-incomplete"
        )
