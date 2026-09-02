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

from dataclasses import dataclass
from pathlib import Path

from vlm_exam.tasks import QA_TASK_NAMES


@dataclass(frozen=True)
class BenchmarkProtocol:
    """What a fully benchmarked model must have in ``results/``.

    Every task in ``tasks`` at every effort in ``efforts``, each with
    ``repeats`` complete runs. Reported metrics are the mean over repeats.
    """

    repeats: int = 3
    efforts: tuple[str, ...] = ("low", "high")
    tasks: tuple[str, ...] = (*QA_TASK_NAMES, "detection")

    @property
    def configurations(self) -> tuple[tuple[str, str], ...]:
        """Every required ``(task, effort)`` pair, task-major."""
        return tuple((task, effort) for task in self.tasks for effort in self.efforts)

    @property
    def required_runs(self) -> int:
        """Total result files a fully benchmarked model has."""
        return len(self.configurations) * self.repeats


PROTOCOL = BenchmarkProtocol()
"""The protocol every committed model is measured against."""

TASK_CONCURRENCY: dict[str, int] = {
    "detection": 6,
    "reasoning": 4,
    "ocr": 3,
    "extraction": 2,
    "counting": 2,
    "identification": 2,
}
"""Default ``--concurrency`` per task, sized so long tasks finish with short ones."""

DEFAULT_DATASET_ROOT = Path("data")
"""Directory holding one ``<task>/train`` dataset per benchmark task."""


def dataset_directory(task: str, root: Path = DEFAULT_DATASET_ROOT) -> Path:
    """Locate the dataset directory a task is benchmarked on.

    Args:
        task: Benchmark task name.
        root: Directory holding the per-task datasets.

    Returns:
        The ``<root>/<task>/train`` directory.
    """
    return root / task / "train"


def run_command(
    task: str,
    model: str,
    effort: str,
    *,
    repeats: int = 1,
    root: Path = DEFAULT_DATASET_ROOT,
    output_directory: Path = Path("results"),
) -> list[str]:
    """Build the ``vlm-exam run`` command for one configuration.

    Args:
        task: Benchmark task name.
        model: Model key from ``models.yaml``.
        effort: Effort level.
        repeats: How many result files the command should produce.
        root: Directory holding the per-task datasets.
        output_directory: Where result files are written.

    Returns:
        The command as an argument list.
    """
    command = [
        "vlm-exam",
        "run",
        "--task",
        task,
        "--models",
        model,
        "--effort",
        effort,
        "--dataset-directory",
        str(dataset_directory(task, root)),
        "--output-directory",
        str(output_directory),
        "--concurrency",
        str(TASK_CONCURRENCY.get(task, 1)),
    ]
    if repeats != 1:
        command.extend(["--repeats", str(repeats)])
    return command
