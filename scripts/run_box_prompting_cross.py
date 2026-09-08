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

from vlm_exam.box_prompting_common import (
    SUPPORTED_MODELS,
    build_backend,
    required_api_key_name,
)
from vlm_exam.box_prompting_cross import (
    CROSS_ARMS,
    build_cross_case,
    format_cross_report,
    run_cross_analysis,
    run_cross_collection,
    write_cross_artifacts,
)
from vlm_exam.box_prompting_groups import build_image_groups, write_groups
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
    help="Defaults to results-box-prompting-<model>-cross-<effort>.",
)
@click.option(
    "--group-count",
    type=click.IntRange(min=1),
    default=None,
    help="Only run the first N groups (smoke tests).",
)
@click.option(
    "--arms",
    default=",".join(CROSS_ARMS),
    show_default=True,
    help="Comma-separated subset of arms to collect and score.",
)
@click.option("--max-workers", type=click.IntRange(min=1), default=4, show_default=True)
@click.option(
    "--request-timeout-seconds",
    type=click.FloatRange(min=1),
    default=900.0,
    show_default=True,
    help="Joint requests on dense groups exceed the default 300 s timeout.",
)
def main(
    model: str,
    effort: str,
    dataset_directory: Path,
    output_directory: Path | None,
    group_count: int | None,
    arms: str,
    max_workers: int,
    request_timeout_seconds: float,
) -> None:
    """Run the cross-image box-prompting experiment end to end."""
    load_dotenv()
    api_key_name = required_api_key_name(model)
    if not os.getenv(api_key_name):
        raise click.ClickException(f"{api_key_name} is required.")
    if output_directory is None:
        output_directory = Path(f"results-box-prompting-{model}-cross-{effort}")
    selected_arms = tuple(arm.strip() for arm in arms.split(",") if arm.strip())
    unknown_arms = [arm for arm in selected_arms if arm not in CROSS_ARMS]
    if unknown_arms:
        raise click.ClickException(f"Unknown arms: {', '.join(unknown_arms)}")

    sample_index = build_sample_index(
        DetectionTask().load_samples(str(dataset_directory))
    )
    groups = build_image_groups(sample_index)
    output_directory.mkdir(parents=True, exist_ok=True)
    write_groups(groups, output_directory / "groups.json")
    if group_count is not None:
        groups = groups[:group_count]
    cases = [build_cross_case(group, sample_index) for group in groups]
    cases_by_group = {case.group_id: case for case in cases}
    merged = sum(1 for case in cases if case.class_name is None)
    click.echo(
        f"Built {len(cases)} cross-image cases ({merged} without a shared class)."
    )

    backend = build_backend(model, timeout_seconds=request_timeout_seconds)
    run_cross_collection(
        cases=cases,
        sample_index=sample_index,
        backend=backend,
        effort=effort,
        output_directory=output_directory,
        max_workers=max_workers,
        arms=selected_arms,
    )

    results = run_cross_analysis(
        raw_directory=output_directory / "raw",
        model_key=model,
        cases_by_group=cases_by_group,
        sample_index=sample_index,
        arms=selected_arms,
    )
    title = f"{model} cross-image box prompting (effort {effort})"
    report_path = write_cross_artifacts(
        output_directory=output_directory,
        results=results,
        cases_by_group=cases_by_group,
        title=title,
    )
    click.echo(format_cross_report(results, cases_by_group, title))
    click.echo(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
