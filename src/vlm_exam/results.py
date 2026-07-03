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

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


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


def save_results(run: RunResult, path: Path) -> None:
    """Save benchmark results to a JSONL file.

    Each line contains a single sample result with run metadata
    embedded for self-contained querying.

    Args:
        run: The complete run result to save.
        path: Output file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as file:
        for sample in run.samples:
            record = {
                "model": run.model,
                "effort": run.effort,
                "task": run.task,
                "timestamp": run.timestamp,
                **asdict(sample),
            }
            file.write(json.dumps(record) + "\n")


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
