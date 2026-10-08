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
import statistics
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import click

from vlm_exam.config import ModelConfig, load_config
from vlm_exam.orchestrate import format_outcomes, plan_jobs, run_jobs
from vlm_exam.protocol import BenchmarkProtocol
from vlm_exam.results import RunResult, is_failed_sample, load_results_directory
from vlm_exam.tasks.base import Sample
from vlm_exam.tasks.text import TEXT_PROTOCOL, TextSample, TextTask

TEXT_BENCHMARK_PROTOCOL = BenchmarkProtocol(tasks=("text",))
"""Independent text benchmark: low/high, three complete repeats per model."""


def inference_hash(model: ModelConfig) -> str:
    """Fingerprint inference settings without display names or token prices."""
    settings = {
        "routes": [
            {
                "provider": route.provider,
                "provider_model_id": route.provider_model_id,
            }
            for route in model.routes
        ],
        "resolution_tier": model.resolution_tier,
        "coordinate_format": model.detection_coordinate_format.value,
    }
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()


def validate_resume(
    previous: RunResult, samples: list[Sample], model: ModelConfig
) -> None:
    """Reject resume when dataset, scoring or inference settings changed."""
    current = {
        sample.identity: sample for sample in samples if isinstance(sample, TextSample)
    }
    if len(current) != len(previous.samples):
        raise ValueError("Resume requires the same complete sample selection")
    expected_inference = inference_hash(model)
    for result in previous.samples:
        sample = current.get(result.metadata.get("sample_id"))
        if (
            sample is None
            or result.metadata.get("dataset_hash") != sample.dataset_hash
            or result.image != Path(sample.image_path).name
        ):
            raise ValueError("Dataset or ground truths changed; start a new run")
        if (
            result.metadata.get("text_protocol") != TEXT_PROTOCOL
            or result.metadata.get("coordinate_format")
            != model.detection_coordinate_format.value
            or result.metadata.get("inference_hash") != expected_inference
        ):
            raise ValueError("Protocol or inference settings changed; start a new run")


def _aggregate(samples: list[Any]) -> dict[str, Any]:
    scored = [
        sample
        for sample in samples
        if not is_failed_sample(sample)
        and isinstance(sample.metadata.get("score"), (int, float))
    ]
    total = sum(sample.metadata["score"] for sample in scored)
    count = len(samples)
    return {
        "pairs": count,
        "scored": len(scored),
        "failed": count - len(scored),
        "mean": total / count if count and len(scored) == count else None,
        "observed_mean": total / len(scored) if scored else None,
        "bounds": [total / count, (total + count - len(scored)) / count]
        if count
        else None,
    }


def summarize_text(directory: Path) -> dict[str, Any]:
    """Average compatible repeats and report category/subset metrics and coverage."""
    groups: dict[
        tuple[str, str, str, str, str, str, tuple[str, ...]], list[RunResult]
    ] = defaultdict(list)
    for run in load_results_directory(directory):
        if run.task != "text":
            continue
        if run.effort not in TEXT_BENCHMARK_PROTOCOL.efforts:
            raise ValueError(f"Unsupported text effort {run.effort!r}; use low or high")
        if not run.samples:
            raise ValueError("Empty text run")
        for sample in run.samples:
            score = sample.metadata.get("score")
            if not is_failed_sample(sample) and (
                type(score) not in (int, float) or not 0 <= score <= 1
            ):
                raise ValueError("Missing or invalid text score")
        signatures = {
            (
                sample.metadata.get("dataset_hash"),
                sample.metadata.get("text_protocol"),
                sample.metadata.get("coordinate_format"),
                sample.metadata.get("inference_hash"),
            )
            for sample in run.samples
        }
        if len(signatures) != 1 or any(
            not isinstance(value, str) or not value for value in next(iter(signatures))
        ):
            raise ValueError("Missing or inconsistent text provenance")
        dataset, protocol, coordinates, inference = next(iter(signatures))
        if protocol != TEXT_PROTOCOL:
            raise ValueError(f"Unsupported text protocol {protocol!r}; start a new run")
        if any(
            not isinstance(sample.metadata.get("sample_id"), str)
            or not sample.metadata["sample_id"]
            for sample in run.samples
        ):
            raise ValueError("Missing text sample identity")
        identities = tuple(
            sorted(sample.metadata["sample_id"] for sample in run.samples)
        )
        if len(set(identities)) != len(identities):
            raise ValueError("Duplicate pair in results")
        groups[
            (
                run.model,
                run.effort,
                dataset,
                protocol,
                coordinates,
                inference,
                identities,
            )
        ].append(run)
    output = []
    for (
        model,
        effort,
        dataset,
        protocol,
        coordinates,
        inference,
        identities,
    ), runs in sorted(groups.items()):
        aggregate = [_aggregate(run.samples) for run in runs]
        expected = {
            sample.metadata.get("dataset_pairs")
            for run in runs
            for sample in run.samples
        }
        complete = (
            len(expected) == 1
            and len(runs[0].samples) == next(iter(expected))
            and all(item["failed"] == 0 for item in aggregate)
        )
        entry: dict[str, Any] = {
            "model": model,
            "effort": effort,
            "dataset_hash": dataset,
            "protocol": protocol,
            "coordinate_format": coordinates,
            "inference_hash": inference,
            "selection_hash": hashlib.sha256(
                json.dumps(
                    {"sample_ids": identities, "pairs": len(identities)}
                ).encode()
            ).hexdigest(),
            "pairs": len(identities),
            "run_count": len(runs),
            "required_runs": TEXT_BENCHMARK_PROTOCOL.repeats,
            "full_dataset": complete,
            "complete": complete and len(runs) == TEXT_BENCHMARK_PROTOCOL.repeats,
            "coverage": aggregate,
            "mean": statistics.mean(item["mean"] for item in aggregate)
            if all(item["mean"] is not None for item in aggregate)
            else None,
            "input_tokens": statistics.mean(
                sum(s.input_tokens for s in run.samples) for run in runs
            ),
            "output_tokens": statistics.mean(
                sum(s.output_tokens for s in run.samples) for run in runs
            ),
        }
        for field in ("category", "subset"):
            labels = sorted(
                {sample.metadata[field] for run in runs for sample in run.samples}
            )
            entry["by_" + field] = {}
            for label in labels:
                scores = [
                    _aggregate([s for s in run.samples if s.metadata[field] == label])
                    for run in runs
                ]
                entry["by_" + field][label] = {
                    "runs": scores,
                    "mean": statistics.mean(x["mean"] for x in scores)
                    if all(x["mean"] is not None for x in scores)
                    else None,
                }
        output.append(entry)
    return {
        "protocol": TEXT_PROTOCOL,
        "efforts": list(TEXT_BENCHMARK_PROTOCOL.efforts),
        "repeats": TEXT_BENCHMARK_PROTOCOL.repeats,
        "note": (
            "Unweighted image-task means of different task metrics, "
            "not uniform accuracy. Incompatible snapshots remain separate."
        ),
        "configurations": output,
    }


def register_text_commands(main: click.Group) -> None:
    """Register additive mixed-text utilities without changing the legacy protocol."""

    @main.command("text-validate")
    @click.option("--dataset-directory", required=True, type=click.Path(exists=True))
    @click.option("--expected-pairs", type=click.IntRange(min=1))
    def validate(dataset_directory: str, expected_pairs: int | None) -> None:
        """Validate JSONL metadata, answers, image paths and all four task types."""
        samples = TextTask().load_samples(dataset_directory)
        if expected_pairs is not None and len(samples) != expected_pairs:
            raise click.ClickException(
                f"Expected {expected_pairs} pairs, found {len(samples)}"
            )
        click.echo(
            json.dumps(
                {
                    "pairs": len(samples),
                    "images": len({s.image_path for s in samples}),
                    "categories": dict(Counter(s.category for s in samples)),
                    "subsets": len({s.subset for s in samples}),
                    "dataset_hash": samples[0].dataset_hash,
                },
                indent=2,
            )
        )

    @main.command("text-import")
    @click.argument("archive", type=click.Path(exists=True))
    @click.option("--output-directory", default="data/text", show_default=True)
    @click.option(
        "--expected-pairs", default=600, show_default=True, type=click.IntRange(min=1)
    )
    def import_dataset(
        archive: str, output_directory: str, expected_pairs: int
    ) -> None:
        """Extract a downloaded ZIP into a new directory and "
        "validate its train split."""
        destination = Path(output_directory).resolve()
        if destination.exists():
            raise click.ClickException(
                "Output directory already exists; choose a new snapshot"
            )
        with zipfile.ZipFile(archive) as zipped:
            paths = [destination / name for name in zipped.namelist()]
            if any(not path.resolve().is_relative_to(destination) for path in paths):
                raise click.ClickException("Unsafe path in ZIP")
            zipped.extractall(destination)
        samples = TextTask().load_samples(str(destination / "train"))
        if len(samples) != expected_pairs:
            raise click.ClickException(
                f"Expected {expected_pairs} pairs, found {len(samples)}; "
                "do not benchmark this incomplete export"
            )
        click.echo(f"Validated {len(samples)} pairs in {destination / 'train'}")

    @main.command("text-benchmark")
    @click.option("--models", required=True)
    @click.option(
        "--config", "config_path", type=click.Path(exists=True, path_type=Path)
    )
    @click.option("--efforts", default="low,high", show_default=True)
    @click.option("--repeats", type=click.IntRange(min=1), default=3, show_default=True)
    @click.option(
        "--first-repeat", type=click.IntRange(min=1), default=1, show_default=True
    )
    @click.option("--dataset-root", default="data", type=click.Path(exists=True))
    @click.option("--output-directory", default="results-text")
    @click.option("--log-directory", default="logs/text")
    @click.option("--max-parallel", default=2, type=click.IntRange(min=1))
    @click.option("--max-samples", type=click.IntRange(min=1))
    def benchmark(
        models: str,
        config_path: Path | None,
        efforts: str,
        repeats: int,
        first_repeat: int,
        dataset_root: str,
        output_directory: str,
        log_directory: str,
        max_parallel: int,
        max_samples: int | None,
    ) -> None:
        """Run low/high three times per model using the existing provider runner."""
        names = [name.strip() for name in models.split(",")]
        config = load_config(config_path)
        selected_efforts = tuple(value.strip() for value in efforts.split(","))
        if len(set(selected_efforts)) != len(selected_efforts) or set(
            selected_efforts
        ) - set(TEXT_BENCHMARK_PROTOCOL.efforts):
            raise click.UsageError(
                "--efforts must contain low and/or high without duplicates"
            )
        if len(names) != len(set(names)):
            raise click.UsageError("--models contains duplicates")
        unknown = set(names) - config.models.keys()
        if unknown:
            raise click.ClickException(f"Unknown models: {sorted(unknown)}")
        TextTask().load_samples(str(Path(dataset_root) / "text" / "train"))
        jobs = plan_jobs(
            names,
            protocol=TEXT_BENCHMARK_PROTOCOL,
            efforts=selected_efforts,
            repeats=repeats,
            first_repeat=first_repeat,
            config_path=config_path,
            dataset_root=Path(dataset_root),
            output_directory=Path(output_directory),
            log_directory=Path(log_directory),
            max_samples=max_samples,
        )
        for job in jobs:
            click.echo(f"{job.model} {job.effort} repeat {job.repeat}: {job.log_path}")
        outcomes = run_jobs(jobs, max_parallel=max_parallel)
        click.echo(format_outcomes(outcomes))
        if not all(outcome.ok for outcome in outcomes):
            raise click.ClickException("Some runs failed; inspect the listed logs")
