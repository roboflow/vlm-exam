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

from pathlib import Path

import click
from PIL import Image

from vlm_exam.box_prompting import (
    load_arm_records,
    load_case_image,
    record_to_detections,
)
from vlm_exam.box_prompting_round2 import (
    overlay_panel,
    prompt_panel,
    select_example_cases,
)
from vlm_exam.providers.image_upload import resize_image_to_max_edge
from vlm_exam.tasks.detection import DetectionTask, build_sample_index

_BACKGROUND_COLOR = (20, 20, 20)
_GAP = 12


def _compose(
    first: Image.Image,
    second: Image.Image,
    *,
    vertical: bool,
) -> Image.Image:
    if vertical:
        width = max(first.size[0], second.size[0])
        height = first.size[1] + second.size[1] + _GAP
        combined = Image.new("RGB", (width, height), _BACKGROUND_COLOR)
        combined.paste(first, ((width - first.size[0]) // 2, 0))
        combined.paste(second, ((width - second.size[0]) // 2, first.size[1] + _GAP))
        return combined
    height = max(first.size[1], second.size[1])
    width = first.size[0] + second.size[0] + _GAP
    combined = Image.new("RGB", (width, height), _BACKGROUND_COLOR)
    combined.paste(first, (0, 0))
    combined.paste(second, (first.size[0] + _GAP, 0))
    return combined


@click.command()
@click.option(
    "--dataset-directory",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("data/detection/train"),
    show_default=True,
)
@click.option(
    "--output-directory",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("results-box-prompting-qwen38-max-round2"),
    show_default=True,
)
@click.option("--arms", default="drawn_box_single,drawn_box_multi", show_default=True)
@click.option(
    "--image-count", type=click.IntRange(min=1), default=50, show_default=True
)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option(
    "--max-edge", type=click.IntRange(min=200), default=1600, show_default=True
)
def main(
    dataset_directory: Path,
    output_directory: Path,
    arms: str,
    image_count: int,
    seed: int,
    max_edge: int,
) -> None:
    """Render orientation-aware input/output stacks for round 2."""
    arm_list = [arm.strip() for arm in arms.split(",") if arm.strip()]
    sample_index = build_sample_index(
        DetectionTask().load_samples(str(dataset_directory))
    )
    cases = select_example_cases(sample_index, count=image_count, seed=seed)
    cases_by_image = {case.image_name: case for case in cases}
    raw_directory = output_directory / "raw"

    for arm in arm_list:
        arm_directory = output_directory / "renders" / "stacked" / arm
        arm_directory.mkdir(parents=True, exist_ok=True)
        written = 0
        skipped = 0
        for record in load_arm_records(raw_directory, arm):
            case = cases_by_image.get(record["image"])
            sample = sample_index.get(record["image"])
            if case is None or sample is None or record.get("error") is not None:
                skipped += 1
                continue
            image = load_case_image(sample)
            detections, _ = record_to_detections(record, sample)
            combined = _compose(
                resize_image_to_max_edge(prompt_panel(arm, image, case), max_edge),
                resize_image_to_max_edge(
                    overlay_panel(arm, image, case, detections), max_edge
                ),
                vertical=image.size[0] > image.size[1],
            )
            combined.save(arm_directory / f"{Path(record['image']).stem}.png")
            written += 1
        click.echo(
            f"{arm}: wrote {written} stacks to {arm_directory} (skipped {skipped})"
        )


if __name__ == "__main__":
    main()
