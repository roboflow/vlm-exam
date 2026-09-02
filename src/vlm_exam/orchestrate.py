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

import os
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from vlm_exam.protocol import PROTOCOL, BenchmarkProtocol, run_command

STAGGER_SECONDS = 2.0
"""Delay between process launches so result timestamps never collide."""


@dataclass(frozen=True)
class Job:
    """One ``vlm-exam run`` process producing a single result file."""

    model: str
    task: str
    effort: str
    repeat: int
    command: tuple[str, ...]
    log_path: Path


@dataclass(frozen=True)
class JobOutcome:
    """Exit status of a finished job."""

    job: Job
    return_code: int
    elapsed_seconds: float

    @property
    def ok(self) -> bool:
        """Whether the process exited cleanly."""
        return self.return_code == 0


def plan_jobs(
    models: list[str],
    *,
    protocol: BenchmarkProtocol = PROTOCOL,
    tasks: tuple[str, ...] | None = None,
    efforts: tuple[str, ...] | None = None,
    repeats: int | None = None,
    first_repeat: int = 1,
    dataset_root: Path = Path("data"),
    output_directory: Path = Path("results"),
    log_directory: Path = Path("logs"),
    max_samples: int | None = None,
) -> list[Job]:
    """Expand models into one job per ``(model, task, effort, repeat)``.

    Args:
        models: Model keys from ``models.yaml``.
        protocol: Protocol supplying default tasks, efforts, and repeats.
        tasks: Tasks to run; defaults to the protocol's tasks.
        efforts: Efforts to run; defaults to the protocol's efforts.
        repeats: Runs per configuration; defaults to the protocol's.
        first_repeat: Repeat number of the first run, for log naming when
            topping up an existing configuration.
        dataset_root: Directory holding ``<task>/train`` datasets.
        output_directory: Where result files are written.
        log_directory: Where per-job logs are written.
        max_samples: Optional sample cap forwarded to ``vlm-exam run``
            (smoke tests only; never for committed runs).

    Returns:
        Jobs ordered by effort, then longest task first, then repeat.
    """
    tasks = tasks or protocol.tasks
    efforts = efforts or protocol.efforts
    repeats = protocol.repeats if repeats is None else repeats
    jobs: list[Job] = []
    for effort in efforts:
        for task in tasks:
            for model in models:
                for offset in range(repeats):
                    repeat = first_repeat + offset
                    command = run_command(
                        task,
                        model,
                        effort,
                        root=dataset_root,
                        output_directory=output_directory,
                    )
                    if max_samples is not None:
                        command.extend(["--max-samples", str(max_samples)])
                    jobs.append(
                        Job(
                            model=model,
                            task=task,
                            effort=effort,
                            repeat=repeat,
                            command=tuple(command),
                            log_path=log_directory
                            / f"{task}_{model}_{effort}_r{repeat}.log",
                        )
                    )
    return jobs


def _python_module_command(command: tuple[str, ...]) -> list[str]:
    # Runs subprocesses on this interpreter so they share its environment even
    # when the console script is not on PATH.
    if command and command[0] == "vlm-exam":
        return [sys.executable, "-m", "vlm_exam.cli", *command[1:]]
    return list(command)


def _launch(job: Job) -> subprocess.Popen[bytes]:
    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, PYTHONUNBUFFERED="1")
    with open(job.log_path, "wb") as log_file:
        log_file.write((" ".join(job.command) + "\n").encode())
        return subprocess.Popen(
            _python_module_command(job.command),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=environment,
        )


def run_jobs(
    jobs: list[Job],
    *,
    max_parallel: int,
    stagger_seconds: float = STAGGER_SECONDS,
    echo: Callable[[str], None] = print,
    launch: Callable[[Job], subprocess.Popen[bytes]] = _launch,
    poll_seconds: float = 5.0,
) -> list[JobOutcome]:
    """Run jobs with at most ``max_parallel`` processes alive at once.

    Every job's log path is announced before it starts so progress can be
    tailed independently; completions are announced as they happen.

    Args:
        jobs: Jobs to run, in launch order.
        max_parallel: Maximum concurrently running processes.
        stagger_seconds: Delay between launches.
        echo: Sink for progress lines.
        launch: Process starter, replaceable in tests.
        poll_seconds: How often to check for finished processes.

    Returns:
        One outcome per job, in launch order.
    """
    if max_parallel < 1:
        raise ValueError(f"max_parallel must be >= 1, got {max_parallel}")

    echo(f"Launching {len(jobs)} runs (max {max_parallel} in parallel). Logs:")
    for job in jobs:
        echo(f"  {job.log_path}")

    pending = list(jobs)
    running: dict[int, tuple[Job, subprocess.Popen[bytes], float]] = {}
    outcomes: dict[int, JobOutcome] = {}
    next_index = 0

    while pending or running:
        while pending and len(running) < max_parallel:
            job = pending.pop(0)
            process = launch(job)
            running[next_index] = (job, process, time.monotonic())
            echo(f"started  {job.task}/{job.effort} r{job.repeat} -> {job.log_path}")
            next_index += 1
            if pending and len(running) < max_parallel:
                time.sleep(stagger_seconds)

        finished = [
            index
            for index, (_, process, _) in running.items()
            if process.poll() is not None
        ]
        for index in finished:
            job, process, started = running.pop(index)
            outcome = JobOutcome(
                job=job,
                return_code=process.returncode,
                elapsed_seconds=time.monotonic() - started,
            )
            outcomes[index] = outcome
            status = "finished" if outcome.ok else f"FAILED ({outcome.return_code})"
            echo(
                f"{status} {job.task}/{job.effort} r{job.repeat} "
                f"in {outcome.elapsed_seconds / 60:.1f} min"
            )
        if running and not finished:
            time.sleep(poll_seconds)

    return [outcomes[index] for index in sorted(outcomes)]


def format_outcomes(outcomes: list[JobOutcome]) -> str:
    """Summarize finished jobs as a table.

    Args:
        outcomes: Outcomes returned by :func:`run_jobs`.

    Returns:
        Multi-line text with one row per job and a totals line.
    """
    lines = [f"{'Status':<8} {'Task':<15} {'Effort':<6} {'Run':>3} {'Minutes':>8}  Log"]
    lines.append("-" * 78)
    for outcome in outcomes:
        job = outcome.job
        status = "ok" if outcome.ok else f"exit {outcome.return_code}"
        lines.append(
            f"{status:<8} {job.task:<15} {job.effort:<6} {job.repeat:>3} "
            f"{outcome.elapsed_seconds / 60:>8.1f}  {job.log_path}"
        )
    failed = sum(1 for outcome in outcomes if not outcome.ok)
    lines.append("")
    lines.append(
        f"{len(outcomes) - failed} of {len(outcomes)} runs finished cleanly"
        + (f"; {failed} failed, see their logs." if failed else ".")
    )
    return "\n".join(lines)
