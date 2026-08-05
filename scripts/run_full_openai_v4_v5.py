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

"""Run full-dataset OpenAI open_ai@v4/@v5 detection evaluations concurrently."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import IO, Any

import click

MODELS = (
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
    "gpt-5.5",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.4-nano",
    "gpt-5.2",
    "gpt-5.1",
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-4o",
    "gpt-4o-mini",
)
# The Responses API rejects the reasoning parameter for these models, and the
# open_ai block manifests reject a reasoning_effort value for them.
NON_REASONING_MODELS = frozenset(
    ("gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o", "gpt-4o-mini")
)
VERSIONS = ("v4", "v5")
IMAGES_PER_RUN = 250
TOTAL_REQUESTS = len(MODELS) * len(VERSIONS) * IMAGES_PER_RUN


@dataclass
class BenchmarkJob:
    model: str
    version: str
    output_directory: Path
    log_path: Path
    process: subprocess.Popen[str] | None = None
    completed: int = 0
    started_at: float = 0.0
    finished_at: float = 0.0

    @property
    def key(self) -> str:
        return f"{self.version}/{self.model}"


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def build_command(
    *,
    project_directory: Path,
    job: BenchmarkJob,
    openai_api_mode: str,
    reasoning_effort: str,
    max_tokens: int,
) -> list[str]:
    job_reasoning_effort = (
        "" if job.model in NON_REASONING_MODELS else reasoning_effort
    )
    return [
        sys.executable,
        "-u",
        str(project_directory / "scripts" / "compare_workflows_detection.py"),
        "--models",
        job.model,
        "--layers",
        "L0_vlm_exam_baseline,L1_workflows_e2e",
        "--block-family",
        "openai",
        "--openai-block-version",
        job.version,
        "--openai-api-mode",
        openai_api_mode,
        "--reasoning-effort",
        job_reasoning_effort,
        "--max-tokens",
        str(max_tokens),
        "--all-images",
        "--output-directory",
        str(job.output_directory),
    ]


def write_log(
    combined_log: IO[str],
    lock: threading.Lock,
    message: str,
    *,
    job_key: str | None = None,
) -> None:
    prefix = f"[{timestamp()}]"
    if job_key is not None:
        prefix += f" [{job_key}]"
    with lock:
        combined_log.write(f"{prefix} {message}\n")
        combined_log.flush()


def stream_job_output(
    *,
    job: BenchmarkJob,
    combined_log: IO[str],
    lock: threading.Lock,
    aggregate_completed: list[int],
) -> None:
    assert job.process is not None
    assert job.process.stdout is not None
    with open(job.log_path, "w", buffering=1) as job_log:
        for raw_line in job.process.stdout:
            job_log.write(raw_line)
            line = raw_line.rstrip()
            with lock:
                combined_log.write(f"[{timestamp()}] [{job.key}] {line}\n")
                if line.startswith("Completed L1_workflows_e2e"):
                    job.completed += 1
                    aggregate_completed[0] += 1
                    combined_log.write(
                        f"[{timestamp()}] [{job.key}] PROGRESS "
                        f"run={job.completed}/{IMAGES_PER_RUN} "
                        f"aggregate={aggregate_completed[0]}/{TOTAL_REQUESTS}\n"
                    )
                combined_log.flush()
    job.finished_at = time.monotonic()


def load_job_result(job: BenchmarkJob) -> dict[str, Any]:
    summaries = sorted(
        job.output_directory.glob("summary_*.json"),
        key=lambda path: path.stat().st_mtime,
    )
    if not summaries:
        raise RuntimeError(f"No summary artifact produced for {job.key}")
    with open(summaries[-1]) as file:
        summary = json.load(file)
    layer_map50 = {}
    for result in summary["layer_results"]:
        if result["model"] == job.model:
            layer_map50[result["layer"]] = float(result["map50"])
    map50 = layer_map50.get("L1_workflows_e2e")
    if map50 is None:
        raise RuntimeError(f"No L1 result found in {summaries[-1]} for {job.key}")

    image_result_paths = sorted(
        job.output_directory.glob("image_results_*.jsonl"),
        key=lambda path: path.stat().st_mtime,
    )
    if not image_result_paths:
        raise RuntimeError(f"No image results artifact produced for {job.key}")
    image_results = []
    with open(image_result_paths[-1]) as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("layer") == "L1_workflows_e2e":
                image_results.append(record)

    numeric_fields = (
        "prompt_tokens",
        "candidate_tokens",
        "thought_tokens",
        "generated_tokens",
        "total_tokens",
        "inference_seconds",
    )
    field_summary = {}
    for field in numeric_fields:
        values = [row[field] for row in image_results if row.get(field) is not None]
        field_summary[field] = {
            "count": len(values),
            "sum": sum(values),
            "average": sum(values) / len(values) if values else None,
            "maximum": max(values) if values else None,
        }
    return {
        "map50": map50,
        "baseline_map50": layer_map50.get("L0_vlm_exam_baseline"),
        "image_count": len(image_results),
        "parse_failures": sum(bool(row.get("parse_failure")) for row in image_results),
        "max_tokens_errors": sum(
            "max_tokens" in str(row.get("execution_error", "")).lower()
            for row in image_results
        ),
        "tokens": {
            field: field_summary[field]
            for field in numeric_fields
            if field != "inference_seconds"
        },
        "inference_seconds": field_summary["inference_seconds"],
    }


def aggregate_results(jobs: list[BenchmarkJob], output_directory: Path) -> str:
    results: dict[str, dict[str, dict[str, Any]]] = {model: {} for model in MODELS}
    for job in jobs:
        results[job.model][job.version] = load_job_result(job)

    lines = [
        "| Model | vlm-exam L0 mAP@50 | v4 mAP@50 | v5 mAP@50 | v5 - v4 |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        v4 = results[model]["v4"]["map50"]
        v5 = results[model]["v5"]["map50"]
        baseline = results[model]["v5"]["baseline_map50"]
        baseline_rendered = f"{baseline * 100:.1f}%" if baseline is not None else "n/a"
        lines.append(
            f"| {model} | {baseline_rendered} | {v4 * 100:.1f}% | {v5 * 100:.1f}% | "
            f"{(v5 - v4) * 100:+.1f} pp |"
        )
    lines.extend(
        [
            "",
            "| Model | Version | Avg input | Avg visible | Avg reasoning | "
            "Avg output | Avg total | Avg seconds | Parse failures | MAX_TOKENS |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for model in MODELS:
        for version in VERSIONS:
            result = results[model][version]
            tokens = result["tokens"]
            averages = [
                tokens[field]["average"]
                for field in (
                    "prompt_tokens",
                    "candidate_tokens",
                    "thought_tokens",
                    "generated_tokens",
                    "total_tokens",
                )
            ]
            rendered_averages = [
                f"{value:.1f}" if value is not None else "n/a" for value in averages
            ]
            average_seconds = result["inference_seconds"]["average"]
            rendered_seconds = (
                f"{average_seconds:.1f}" if average_seconds is not None else "n/a"
            )
            lines.append(
                f"| {model} | {version} | {' | '.join(rendered_averages)} | "
                f"{rendered_seconds} | "
                f"{result['parse_failures']} | {result['max_tokens_errors']} |"
            )

    aggregate_path = output_directory / "aggregate_summary.json"
    with open(aggregate_path, "w") as file:
        json.dump(
            {
                "generated_at": timestamp(),
                "images_per_run": IMAGES_PER_RUN,
                "results": results,
            },
            file,
            indent=2,
        )
    return "\n".join(lines)


@click.command()
@click.option(
    "--output-directory",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("results-workflows-comparison/full-250-openai-v4-v5-new"),
    show_default=True,
)
@click.option(
    "--openai-api-mode",
    type=click.Choice(["direct", "roboflow-proxy"]),
    default="direct",
    show_default=True,
)
@click.option(
    "--reasoning-effort",
    default="low",
    show_default=True,
    help="reasoning_effort passed to every open_ai request ('' disables).",
)
@click.option(
    "--max-tokens",
    type=click.IntRange(min=16),
    default=16384,
    show_default=True,
    help="Maximum output tokens passed to every OpenAI request.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Print all commands without launching them.",
)
def main(
    output_directory: Path,
    openai_api_mode: str,
    reasoning_effort: str,
    max_tokens: int,
    dry_run: bool,
) -> None:
    """Launch all supported GPT models for both open_ai workflow block versions."""
    project_directory = Path(__file__).resolve().parents[1]
    jobs = [
        BenchmarkJob(
            model=model,
            version=version,
            output_directory=output_directory / "artifacts" / version / model,
            log_path=output_directory / "logs" / f"{version}__{model}.log",
        )
        for version in VERSIONS
        for model in MODELS
    ]

    if dry_run:
        for job in jobs:
            command = build_command(
                project_directory=project_directory,
                job=job,
                openai_api_mode=openai_api_mode,
                reasoning_effort=reasoning_effort,
                max_tokens=max_tokens,
            )
            click.echo(f"[{job.key}] {shlex.join(command)}")
        return

    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "logs").mkdir(parents=True, exist_ok=True)
    for job in jobs:
        job.output_directory.mkdir(parents=True, exist_ok=True)

    progress_path = output_directory / "progress.log"
    lock = threading.Lock()
    aggregate_completed = [0]
    threads: list[threading.Thread] = []
    environment = dict(os.environ, PYTHONUNBUFFERED="1")

    with open(progress_path, "w", buffering=1) as combined_log:
        write_log(
            combined_log,
            lock,
            f"Launching {len(jobs)} jobs; total_requests={TOTAL_REQUESTS}",
        )
        write_log(
            combined_log,
            lock,
            f"Watch with: tail -f {progress_path}",
        )

        for job in jobs:
            command = build_command(
                project_directory=project_directory,
                job=job,
                openai_api_mode=openai_api_mode,
                reasoning_effort=reasoning_effort,
                max_tokens=max_tokens,
            )
            write_log(
                combined_log,
                lock,
                f"START command={shlex.join(command)}",
                job_key=job.key,
            )
            job.started_at = time.monotonic()
            job.process = subprocess.Popen(
                command,
                cwd=project_directory,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            thread = threading.Thread(
                target=stream_job_output,
                kwargs={
                    "job": job,
                    "combined_log": combined_log,
                    "lock": lock,
                    "aggregate_completed": aggregate_completed,
                },
                name=f"log-{job.version}-{job.model}",
            )
            thread.start()
            threads.append(thread)

        exit_codes = []
        for job in jobs:
            assert job.process is not None
            exit_codes.append(job.process.wait())
        for thread in threads:
            thread.join()

        failed_jobs = []
        for job, exit_code in zip(jobs, exit_codes):
            duration = job.finished_at - job.started_at
            write_log(
                combined_log,
                lock,
                f"FINISH exit={exit_code} completed={job.completed}/"
                f"{IMAGES_PER_RUN} duration={duration / 60:.1f}m",
                job_key=job.key,
            )
            if exit_code != 0 or job.completed != IMAGES_PER_RUN:
                failed_jobs.append(job.key)

        if failed_jobs:
            write_log(
                combined_log,
                lock,
                f"FAILED jobs={','.join(failed_jobs)}",
            )
            raise click.ClickException(
                f"{len(failed_jobs)} benchmark jobs failed; see {progress_path}"
            )

        table = aggregate_results(jobs, output_directory)
        write_log(combined_log, lock, "All benchmark jobs completed successfully.")
        for line in table.splitlines():
            write_log(combined_log, lock, line)

    click.echo(table)
    click.echo(f"\nProgress log: {progress_path}")


if __name__ == "__main__":
    main()
