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

"""Run the prompt/coordinate/preprocessing detection experiment matrix."""

from __future__ import annotations

import os
from pathlib import Path

import click
from dotenv import load_dotenv

from vlm_exam.format_matrix import (
    ARMS,
    ARMS_BY_ID,
    MATRIX_MODELS,
    format_matrix_report,
    load_workflow_anchor,
    run_matrix_analysis,
    run_matrix_collection,
    select_matrix_images,
    write_matrix_artifacts,
)
from vlm_exam.tasks.detection import DetectionTask, build_sample_index
from vlm_exam.workflows_comparison import image_classes

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
    default=Path("results-format-matrix"),
    show_default=True,
    help="Directory for raw responses and analysis artifacts.",
)


def load_dataset(dataset_directory: Path):
    detection_task = DetectionTask()
    return build_sample_index(detection_task.load_samples(str(dataset_directory)))


@click.group()
def main() -> None:
    """Prompt format x coordinate system x preprocessing matrix."""
    load_dotenv()


@main.command()
@DATASET_OPTION
@OUTPUT_OPTION
@click.option(
    "--models",
    default=",".join(MATRIX_MODELS),
    show_default=True,
    help="Comma-separated model ids.",
)
@click.option(
    "--arms",
    default="",
    help="Comma-separated arm ids; empty means all arms.",
)
@click.option(
    "--image-count",
    type=click.IntRange(min=1),
    default=100,
    show_default=True,
    help="Number of control-set images (smaller counts are prefixes).",
)
@click.option(
    "--seed",
    type=int,
    default=42,
    show_default=True,
    help="Random seed for the fixed image subset.",
)
@click.option(
    "--max-workers",
    type=click.IntRange(min=1),
    default=10,
    show_default=True,
    help="Concurrent (model, arm) jobs.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Print the job matrix and selected images without calling the API.",
)
def collect(
    dataset_directory: Path,
    output_directory: Path,
    models: str,
    arms: str,
    image_count: int,
    seed: int,
    max_workers: int,
    dry_run: bool,
) -> None:
    """Collect raw matrix responses (resumable per model x arm file)."""
    model_list = [item.strip() for item in models.split(",") if item.strip()]
    if arms.strip():
        arm_ids = [item.strip() for item in arms.split(",") if item.strip()]
        unknown = set(arm_ids) - set(ARMS_BY_ID)
        if unknown:
            raise click.ClickException(f"Unknown arm ids: {sorted(unknown)}")
        arm_list = [ARMS_BY_ID[arm_id] for arm_id in arm_ids]
    else:
        arm_list = list(ARMS)

    sample_index = load_dataset(dataset_directory)
    selected_images = select_matrix_images(sample_index, count=image_count, seed=seed)
    samples = [sample_index[name] for name in selected_images]
    classes_by_image = {
        Path(sample.image_path).name: image_classes(sample) for sample in samples
    }
    total = len(model_list) * len(arm_list) * len(samples)
    click.echo(
        f"Matrix: {len(model_list)} models x {len(arm_list)} arms "
        f"x {len(samples)} images = {total} requests"
    )
    if dry_run:
        for model_id in model_list:
            for arm in arm_list:
                click.echo(f"  {model_id} x {arm.arm_id} ({arm.description})")
        click.echo("Images: " + ", ".join(selected_images[:10]) + " ...")
        return
    if not os.getenv("OPENAI_API_KEY"):
        raise click.ClickException("OPENAI_API_KEY is required.")
    run_matrix_collection(
        model_ids=model_list,
        arms=arm_list,
        samples=samples,
        classes_by_image=classes_by_image,
        output_directory=output_directory,
        max_workers=max_workers,
    )
    click.echo("Collection complete.")


@main.command()
@DATASET_OPTION
@OUTPUT_OPTION
@click.option(
    "--benchmark-artifacts",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("results-workflows-comparison/full-250-openai-v4-v5/artifacts"),
    show_default=True,
    help="Stored workflow benchmark artifacts for anchor comparison.",
)
def analyze(
    dataset_directory: Path,
    output_directory: Path,
    benchmark_artifacts: Path,
) -> None:
    """Score collected responses and write the matrix report."""
    raw_directory = output_directory / "raw"
    if not raw_directory.exists():
        raise click.ClickException(f"No raw responses found in {raw_directory}")
    sample_index = load_dataset(dataset_directory)
    samples_by_image = {
        Path(sample.image_path).name: sample for sample in sample_index.values()
    }
    results = run_matrix_analysis(
        raw_directory=raw_directory,
        samples_by_image=samples_by_image,
    )

    anchors: dict[str, dict[str, float | None]] | None = None
    subset_path = output_directory / "image_subset.json"
    if benchmark_artifacts.exists() and subset_path.exists():
        import json  # noqa: PLC0415

        with open(subset_path) as file:
            image_subset = set(json.load(file))
        anchors = {
            model_id: {
                version: load_workflow_anchor(
                    benchmark_artifacts,
                    model_id=model_id,
                    version=version,
                    image_subset=image_subset,
                )
                for version in ("v4", "v5")
            }
            for model_id in results
        }

    report_path = write_matrix_artifacts(
        output_directory=output_directory,
        results=results,
        anchors=anchors,
    )
    click.echo(format_matrix_report(results, anchors=anchors))
    click.echo(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
