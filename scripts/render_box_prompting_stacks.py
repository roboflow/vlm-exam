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
import supervision as sv
from PIL import Image, ImageDraw, ImageFont

from vlm_exam.box_prompting import (
    ReferenceCase,
    crop_reference,
    draw_reference_box,
    load_arm_records,
    load_case_image,
    record_to_detections,
    select_cases,
)
from vlm_exam.providers.image_upload import resize_image_to_max_edge
from vlm_exam.tasks.detection import DetectionTask, build_sample_index

_REFERENCE_COLOR = (255, 0, 0)
_TARGET_COLOR = (0, 200, 83)
_PREDICTION_COLOR = (41, 98, 255)
_BACKGROUND_COLOR = (20, 20, 20)
_STRIP_HEIGHT = 44
_GAP = 12


def _input_panel(
    arm: str,
    image: Image.Image,
    case: ReferenceCase,
) -> Image.Image:
    if arm == "drawn_box":
        return draw_reference_box(image, case.reference_xyxy)
    if arm == "crop":
        return crop_reference(image, case.reference_xyxy)
    return image


def _overlay_panel(
    image: Image.Image,
    case: ReferenceCase,
    detections: sv.Detections,
) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    width = max(3, round(max(image.size) / 300))
    for box in case.target_xyxy:
        draw.rectangle(box, outline=_TARGET_COLOR, width=width)
    for box in detections.xyxy:
        predicted = tuple(float(value) for value in box)
        draw.rectangle(predicted, outline=_PREDICTION_COLOR, width=width)
    draw.rectangle(case.reference_xyxy, outline=_REFERENCE_COLOR, width=width)
    return annotated


def _label_font() -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    try:
        return ImageFont.load_default(size=round(_STRIP_HEIGHT * 0.55))
    except TypeError:
        return ImageFont.load_default()


def _stack_panels(
    top: Image.Image,
    bottom: Image.Image,
    *,
    top_label: str,
    bottom_label: str,
) -> Image.Image:
    width = max(top.size[0], bottom.size[0])
    height = top.size[1] + bottom.size[1] + 2 * _STRIP_HEIGHT + _GAP
    combined = Image.new("RGB", (width, height), _BACKGROUND_COLOR)
    draw = ImageDraw.Draw(combined)
    font = _label_font()

    top_y = _STRIP_HEIGHT
    draw.text((10, 8), top_label, fill=(255, 255, 255), font=font)
    combined.paste(top, ((width - top.size[0]) // 2, top_y))

    strip_y = top_y + top.size[1] + _GAP
    bottom_y = strip_y + _STRIP_HEIGHT
    draw.text((10, strip_y + 8), bottom_label, fill=(255, 255, 255), font=font)
    combined.paste(bottom, ((width - bottom.size[0]) // 2, bottom_y))
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
    default=Path("results-box-prompting-qwen38-max"),
    show_default=True,
)
@click.option("--arms", default="drawn_box", show_default=True)
@click.option(
    "--image-count", type=click.IntRange(min=1), default=25, show_default=True
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
    """Render vertical input/output stacks from collected raw data."""
    arm_list = [arm.strip() for arm in arms.split(",") if arm.strip()]
    sample_index = build_sample_index(
        DetectionTask().load_samples(str(dataset_directory))
    )
    cases = select_cases(sample_index, count=image_count, seed=seed)
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
            stacked = _stack_panels(
                resize_image_to_max_edge(_input_panel(arm, image, case), max_edge),
                resize_image_to_max_edge(
                    _overlay_panel(image, case, detections), max_edge
                ),
                top_label=f"{arm} input",
                bottom_label="prediction",
            )
            output_path = arm_directory / f"{Path(record['image']).stem}.png"
            stacked.save(output_path)
            written += 1
        click.echo(
            f"{arm}: wrote {written} stacks to {arm_directory} (skipped {skipped})"
        )


if __name__ == "__main__":
    main()
