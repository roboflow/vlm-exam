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

"""Probe the native detection output format of OpenAI models."""

from __future__ import annotations

import os
from pathlib import Path

import click
from dotenv import load_dotenv

from vlm_exam.format_probe import (
    PROMPT_VARIANTS,
    format_probe_report,
    load_openai_block_models,
    preflight_models,
    run_analysis,
    run_probe_collection,
    write_analysis_artifacts,
)
from vlm_exam.tasks.detection import DetectionTask, build_sample_index
from vlm_exam.workflows_comparison import image_classes, select_image_subset

DATASET_OPTION = click.option(
    "--dataset-directory",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("data/detection/train"),
    show_default=True,
    help="COCO detection dataset directory (control set).",
)
OUTPUT_OPTION = click.option(
    "--output-directory",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("results-format-probe"),
    show_default=True,
    help="Directory for raw responses and analysis artifacts.",
)
MODELS_OPTION = click.option(
    "--models",
    default="",
    help="Comma-separated model ids; empty means every open_ai block model.",
)


def resolve_models(models_argument: str) -> list[dict]:
    descriptors = load_openai_block_models()
    if not models_argument.strip():
        return descriptors
    requested = {item.strip() for item in models_argument.split(",") if item.strip()}
    by_id = {descriptor["id"]: descriptor for descriptor in descriptors}
    unknown = requested - set(by_id)
    if unknown:
        raise click.ClickException(f"Unknown model ids: {sorted(unknown)}")
    return [by_id[model_id] for model_id in sorted(requested)]


def load_probe_samples(
    dataset_directory: Path,
    *,
    image_count: int,
    per_bucket: int,
    seed: int,
):
    detection_task = DetectionTask()
    sample_index = build_sample_index(
        detection_task.load_samples(str(dataset_directory))
    )
    selected_images = select_image_subset(
        sample_index,
        {},
        per_bucket=per_bucket,
        seed=seed,
    )[:image_count]
    samples = [sample_index[image_name] for image_name in selected_images]
    return samples, sample_index


@click.group()
def main() -> None:
    """Probe which detection output format OpenAI models natively produce."""
    load_dotenv()


@main.command()
@MODELS_OPTION
def preflight(models: str) -> None:
    """Verify every model id answers a minimal request."""
    if not os.getenv("OPENAI_API_KEY"):
        raise click.ClickException("OPENAI_API_KEY is required.")
    descriptors = resolve_models(models)
    results = preflight_models(descriptors)
    failed = 0
    for model_id, error in results.items():
        status = "ok" if error is None else f"FAILED ({error})"
        click.echo(f"{model_id:<16} {status}")
        failed += error is not None
    if failed:
        raise click.ClickException(f"{failed} model ids failed pre-flight.")


@main.command()
@DATASET_OPTION
@OUTPUT_OPTION
@MODELS_OPTION
@click.option(
    "--variants",
    default=",".join(PROMPT_VARIANTS),
    show_default=True,
    help="Comma-separated prompt variants to send.",
)
@click.option(
    "--image-count",
    type=click.IntRange(min=1),
    default=20,
    show_default=True,
    help="Number of control-set images to probe.",
)
@click.option(
    "--per-bucket",
    type=click.IntRange(min=1),
    default=7,
    show_default=True,
    help="Images per resolution bucket before truncating to --image-count.",
)
@click.option(
    "--seed",
    type=int,
    default=42,
    show_default=True,
    help="Random seed for the stratified image subset.",
)
@click.option(
    "--max-workers",
    type=click.IntRange(min=1),
    default=6,
    show_default=True,
    help="Concurrent (model, variant) jobs.",
)
def collect(
    dataset_directory: Path,
    output_directory: Path,
    models: str,
    variants: str,
    image_count: int,
    per_bucket: int,
    seed: int,
    max_workers: int,
) -> None:
    """Collect raw probe responses (resumable)."""
    if not os.getenv("OPENAI_API_KEY"):
        raise click.ClickException("OPENAI_API_KEY is required.")
    variant_list = [item.strip() for item in variants.split(",") if item.strip()]
    unknown = set(variant_list) - set(PROMPT_VARIANTS)
    if unknown:
        raise click.ClickException(f"Unknown prompt variants: {sorted(unknown)}")
    descriptors = resolve_models(models)
    samples, _ = load_probe_samples(
        dataset_directory,
        image_count=image_count,
        per_bucket=per_bucket,
        seed=seed,
    )
    classes_by_image = {
        Path(sample.image_path).name: image_classes(sample) for sample in samples
    }
    click.echo(
        f"Probing {len(descriptors)} models x {len(variant_list)} variants "
        f"x {len(samples)} images "
        f"= {len(descriptors) * len(variant_list) * len(samples)} requests"
    )
    run_probe_collection(
        models=descriptors,
        variants=variant_list,
        samples=samples,
        classes_by_image=classes_by_image,
        output_directory=output_directory,
        max_workers=max_workers,
    )
    click.echo("Collection complete.")


@main.command()
@DATASET_OPTION
@OUTPUT_OPTION
def analyze(dataset_directory: Path, output_directory: Path) -> None:
    """Analyze collected raw responses and write the format report."""
    raw_directory = output_directory / "raw"
    if not raw_directory.exists():
        raise click.ClickException(f"No raw responses found in {raw_directory}")
    detection_task = DetectionTask()
    sample_index = build_sample_index(
        detection_task.load_samples(str(dataset_directory))
    )
    samples_by_image = {
        Path(sample.image_path).name: sample for sample in sample_index.values()
    }
    summaries, analyses = run_analysis(
        raw_directory=raw_directory,
        samples_by_image=samples_by_image,
    )
    report_path = write_analysis_artifacts(
        output_directory=output_directory,
        summaries=summaries,
        analyses=analyses,
    )
    click.echo(format_probe_report(summaries))
    click.echo(f"\nReport written to {report_path}")


if __name__ == "__main__":
    main()
