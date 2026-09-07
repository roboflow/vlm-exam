# Copyright 2026 Roboflow, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import os
from pathlib import Path

import click
from dotenv import load_dotenv

from vlm_exam.box_prompting import (
    ARMS,
    format_report,
    render_arm_overlays,
    run_analysis,
    run_collection,
    select_cases,
    write_artifacts,
)
from vlm_exam.box_prompting_common import (
    SUPPORTED_MODELS,
    build_backend,
    required_api_key_name,
)
from vlm_exam.tasks.detection import DetectionTask, build_sample_index


@click.command()
@click.option(
    "--model",
    type=click.Choice(SUPPORTED_MODELS),
    default="gpt-6-astra",
    show_default=True,
)
@click.option(
    "--effort", type=click.Choice(["low", "high"]), default="low", show_default=True
)
@click.option(
    "--dataset-directory",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("data/detection/train"),
    show_default=True,
)
@click.option(
    "--output-directory",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Defaults to results-box-prompting-<model>-<effort>.",
)
@click.option(
    "--image-count", type=click.IntRange(min=1), default=25, show_default=True
)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option("--max-workers", type=click.IntRange(min=1), default=3, show_default=True)
@click.option("--render/--no-render", default=True, show_default=True)
def main(
    model: str,
    effort: str,
    dataset_directory: Path,
    output_directory: Path | None,
    image_count: int,
    seed: int,
    max_workers: int,
    render: bool,
) -> None:
    """Run round 1 of the box-prompting experiment end to end."""
    load_dotenv()
    api_key_name = required_api_key_name(model)
    if not os.getenv(api_key_name):
        raise click.ClickException(f"{api_key_name} is required.")
    if output_directory is None:
        output_directory = Path(f"results-box-prompting-{model}-{effort}")

    sample_index = build_sample_index(
        DetectionTask().load_samples(str(dataset_directory))
    )
    cases = select_cases(sample_index, count=image_count, seed=seed)
    if len(cases) < image_count:
        click.echo(f"Only {len(cases)} usable images found (requested {image_count}).")
    cases_by_image = {case.image_name: case for case in cases}

    backend = build_backend(model)
    run_collection(
        cases=cases,
        sample_index=sample_index,
        backend=backend,
        effort=effort,
        output_directory=output_directory,
        max_workers=max_workers,
    )

    raw_directory = output_directory / "raw"
    results = run_analysis(
        raw_directory=raw_directory,
        model_key=model,
        cases_by_image=cases_by_image,
        sample_index=sample_index,
    )
    title = f"{model} box-prompting experiment (effort {effort})"
    report_path = write_artifacts(
        output_directory=output_directory, results=results, title=title
    )

    if render:
        renders_directory = output_directory / "renders"
        for arm in ARMS:
            render_arm_overlays(
                arm=arm,
                raw_directory=raw_directory,
                model_key=model,
                cases_by_image=cases_by_image,
                sample_index=sample_index,
                renders_directory=renders_directory,
            )
        click.echo(f"Renders written to {renders_directory}")

    click.echo(format_report(results, title))
    click.echo(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
