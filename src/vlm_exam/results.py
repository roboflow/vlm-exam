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
import os
import tempfile
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from vlm_exam.providers.base import EMPTY_RESPONSE_TEXT

ERROR_PREDICTION_PREFIX = "ERROR:"
"""Prefix marking a sample whose provider call failed."""


def is_failed_sample(sample: SampleResult) -> bool:
    """Report whether a sample's prediction is a recorded provider error.

    Args:
        sample: A sample result loaded from a run file.

    Returns:
        True when the prediction holds an error marker instead of
        model output.
    """
    return sample.predicted.startswith(ERROR_PREDICTION_PREFIX)


def is_incomplete_sample(sample: SampleResult) -> bool:
    """Report whether a sample should be re-run on ``--resume-file``.

    Args:
        sample: A sample result loaded from a run file.

    Returns:
        True when the prediction is a provider error or the empty-content
        sentinel.
    """
    return is_failed_sample(sample) or sample.predicted == EMPTY_RESPONSE_TEXT


@dataclass
class SampleResult:
    """Result of evaluating a single sample."""

    index: int
    image: str
    expected: str
    predicted: str
    correct: bool
    input_tokens: int
    output_tokens: int
    elapsed_seconds: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunResult:
    """Complete result of a benchmark run (one model, one effort level)."""

    model: str
    effort: str
    task: str
    timestamp: str
    samples: list[SampleResult] = field(default_factory=list)


def _sample_key(sample: SampleResult, task: str) -> tuple[str, str]:
    question = "" if task == "detection" else str(sample.metadata.get("question", ""))
    return (sample.image, question)


def _index_samples(
    samples: list[SampleResult],
    task: str,
) -> dict[tuple[str, str], SampleResult]:
    indexed: dict[tuple[str, str], SampleResult] = {}
    for sample in samples:
        key = _sample_key(sample, task)
        if key in indexed:
            raise ValueError(f"Duplicate sample in resumed run: {key!r}")
        indexed[key] = sample
    return indexed


def merge_resumed_runs(previous: RunResult, resumed: RunResult) -> RunResult:
    """Merge a resumed run into a partial previous run.

    Incomplete samples from the previous run (provider errors or empty
    content) are replaced by the resumed run's sample for the same
    image and question; other samples are kept as-is. Resumed samples
    absent from the previous run are appended, so resuming an
    incomplete run keeps its new samples.
    Sample order follows the previous run and indexes are rewritten to
    be contiguous.

    Args:
        previous: The partial run containing incomplete samples.
        resumed: A run covering (at least) the previously incomplete images.

    Returns:
        A complete run result carrying the resumed run's timestamp.
    """
    if (
        previous.model,
        previous.effort,
        previous.task,
    ) != (
        resumed.model,
        resumed.effort,
        resumed.task,
    ):
        raise ValueError("Cannot merge runs with different model, effort, or task.")

    resumed_by_key = _index_samples(resumed.samples, resumed.task)
    previous_by_key = _index_samples(previous.samples, previous.task)

    merged: list[SampleResult] = []
    for sample in previous.samples:
        replacement = resumed_by_key.get(_sample_key(sample, previous.task))
        if is_incomplete_sample(sample) and replacement is not None:
            merged.append(replacement)
        else:
            merged.append(sample)
    for sample in resumed.samples:
        if _sample_key(sample, resumed.task) not in previous_by_key:
            merged.append(sample)

    merged = [replace(sample, index=position) for position, sample in enumerate(merged)]

    return RunResult(
        model=resumed.model,
        effort=resumed.effort,
        task=resumed.task,
        timestamp=resumed.timestamp,
        samples=merged,
    )


def save_results(run: RunResult, path: Path) -> None:
    """Save benchmark results to a JSONL file.

    Each line contains a single sample result with run metadata
    embedded for self-contained querying.

    Args:
        run: The complete run result to save.
        path: Output file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            for sample in run.samples:
                record = {
                    "model": run.model,
                    "effort": run.effort,
                    "task": run.task,
                    "timestamp": run.timestamp,
                    **asdict(sample),
                }
                file.write(json.dumps(record) + "\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def load_results(path: Path) -> RunResult:
    """Load benchmark results from a JSONL file.

    Args:
        path: Path to a JSONL results file previously written by
            :func:`save_results`.

    Returns:
        Reconstructed run result.

    Raises:
        ValueError: If the file is empty.
    """
    samples: list[SampleResult] = []
    model = ""
    effort = ""
    task = ""
    timestamp = ""

    with open(path) as file:
        for line in file:
            record = json.loads(line)
            model = record["model"]
            effort = record["effort"]
            task = record["task"]
            timestamp = record["timestamp"]
            samples.append(
                SampleResult(
                    index=record["index"],
                    image=record["image"],
                    expected=record["expected"],
                    predicted=record["predicted"],
                    correct=record["correct"],
                    input_tokens=record["input_tokens"],
                    output_tokens=record["output_tokens"],
                    elapsed_seconds=record.get("elapsed_seconds"),
                    metadata=record.get("metadata", {}),
                )
            )

    if not samples:
        raise ValueError(f"No results found in {path}")

    return RunResult(
        model=model,
        effort=effort,
        task=task,
        timestamp=timestamp,
        samples=samples,
    )


def load_results_directory(path: Path, pattern: str = "*.jsonl") -> list[RunResult]:
    """Load every non-empty JSONL results file in a directory.

    Empty files are skipped with a message rather than failing the
    whole load.

    Args:
        path: Directory containing result files.
        pattern: Glob pattern selecting the files to load.

    Returns:
        Run results sorted by file name.
    """
    runs: list[RunResult] = []
    for file_path in sorted(path.glob(pattern)):
        try:
            runs.append(load_results(file_path))
        except ValueError:
            print(f"Skipping empty file: {file_path}")
    return runs
