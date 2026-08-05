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

import json
from pathlib import Path
from typing import Any

import click
from PIL import Image, ImageDraw, ImageFont

from vlm_exam.box_prompting import load_case_image
from vlm_exam.box_prompting_incontext import (
    SETS,
    TARGET_CLASS,
    parse_entries,
    payload_from_output,
    prompt_panel,
    resolve_image_name,
    result_panel,
)
from vlm_exam.providers.image_upload import resize_image_to_max_edge
from vlm_exam.tasks.detection import DetectionTask, build_sample_index

_ARMS = ("single", "multi")
_GAP = 12
_BACKGROUND_COLOR = (20, 20, 20)
_HEADER_COLOR = (0, 0, 0)
_HEADER_TEXT_COLOR = (255, 255, 255)
_FONT_PATH = (
    Path(__file__).resolve().parents[1] / "assets" / "fonts" / ("GeistMono-Regular.ttf")
)

_CLASS_OVERRIDES = {
    "set2_football": "referee",
    "set5_basketball": "jersey number",
}
_PROMPT_OVERRIDES = {
    "set4_technical_drawing": (
        "V000078_0_0_jpeg_jpg.rf.ea93b54e17810ab845506f60bf69b7dd.jpg"
    ),
}


def _baseline_config(
    set_name: str,
    sample_index: dict[str, Any],
) -> tuple[str, str, list[str]]:
    class_name = _CLASS_OVERRIDES.get(set_name, TARGET_CLASS[set_name])
    names = [resolve_image_name(name, sample_index) for name in SETS[set_name]]
    if set_name in _PROMPT_OVERRIDES:
        prompt_name = resolve_image_name(_PROMPT_OVERRIDES[set_name], sample_index)
        target_names = [name for name in names if name != prompt_name]
    else:
        prompt_name = names[0]
        target_names = names[1:]
    return class_name, prompt_name, target_names


def _fit(image: Image.Image, box_width: int, box_height: int) -> Image.Image:
    scale = min(box_width / image.size[0], box_height / image.size[1])
    size = (max(1, round(image.size[0] * scale)), max(1, round(image.size[1] * scale)))
    return image.resize(size, Image.LANCZOS)


def _cell(
    panel: Image.Image,
    label: str,
    cell_width: int,
    cell_height: int,
    font: ImageFont.FreeTypeFont,
    header_height: int,
) -> Image.Image:
    cell = Image.new(
        "RGB", (cell_width, header_height + cell_height), _BACKGROUND_COLOR
    )
    draw = ImageDraw.Draw(cell)
    draw.rectangle((0, 0, cell_width, header_height), fill=_HEADER_COLOR)
    _, top, _, bottom = draw.textbbox((0, 0), label, font=font)
    text_y = (header_height - (bottom - top)) // 2 - top
    draw.text(
        (round(header_height * 0.35), text_y),
        label,
        fill=_HEADER_TEXT_COLOR,
        font=font,
    )
    fitted = _fit(panel, cell_width, cell_height)
    offset_x = (cell_width - fitted.size[0]) // 2
    offset_y = header_height + (cell_height - fitted.size[1]) // 2
    cell.paste(fitted, (offset_x, offset_y))
    return cell


def _grid(cells: list[Image.Image]) -> Image.Image:
    cell_width, cell_height = cells[0].size
    width = 2 * cell_width + _GAP
    height = 2 * cell_height + _GAP
    grid = Image.new("RGB", (width, height), _BACKGROUND_COLOR)
    positions = [
        (0, 0),
        (cell_width + _GAP, 0),
        (0, cell_height + _GAP),
        (cell_width + _GAP, cell_height + _GAP),
    ]
    for cell, position in zip(cells, positions):
        grid.paste(cell, position)
    return grid


def _render_set_arm(
    *,
    set_name: str,
    arm: str,
    sample_index: dict[str, Any],
    output_directory: Path,
    max_edge: int,
) -> Path | None:
    raw_path = output_directory / set_name / "raw" / f"{arm}.json"
    if not raw_path.exists():
        return None
    record = json.loads(raw_path.read_text())
    class_name, prompt_name, target_names = _baseline_config(set_name, sample_index)

    prompt_sample = sample_index[prompt_name]
    positives = tuple(tuple(box) for box in record["positive_xyxy"])
    negatives = tuple(tuple(box) for box in record["negative_xyxy"])
    panels = [
        resize_image_to_max_edge(
            prompt_panel(load_case_image(prompt_sample), positives, negatives),
            max_edge,
        )
    ]
    labels = ["prompt"]

    payload = payload_from_output(record.get("raw_output", ""))
    for index, name in enumerate(target_names[:3]):
        sample = sample_index[name]
        detections, _ = parse_entries(payload, f"image_{index + 2}", sample)
        panels.append(
            resize_image_to_max_edge(
                result_panel(load_case_image(sample), detections), max_edge
            )
        )
        labels.append(f"target {index + 1}")

    cell_width = max(panel.size[0] for panel in panels)
    cell_height = max(panel.size[1] for panel in panels)
    header_height = round(cell_width * 0.08)
    font = ImageFont.truetype(str(_FONT_PATH), round(header_height * 0.55))

    cells = [
        _cell(panel, label, cell_width, cell_height, font, header_height)
        for panel, label in zip(panels, labels)
    ]
    grid = _grid(cells)

    renders = output_directory / set_name / "renders" / arm
    renders.mkdir(parents=True, exist_ok=True)
    grid_path = renders / "grid.png"
    grid.save(grid_path)
    print(f"[{set_name}/{arm}] wrote {grid_path}", flush=True)
    return grid_path


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
    default=Path("results-box-prompting-qwen38-max-baseline"),
    show_default=True,
)
@click.option("--set", "set_name", default="set1_bottle_cap", show_default=True)
@click.option("--arm", type=click.Choice(_ARMS), default="multi", show_default=True)
@click.option(
    "--max-edge", type=click.IntRange(min=200), default=1200, show_default=True
)
@click.option("--all", "render_all", is_flag=True, default=False)
def main(
    dataset_directory: Path,
    output_directory: Path,
    set_name: str,
    arm: str,
    max_edge: int,
    render_all: bool,
) -> None:
    """Render 2x2 baseline grids (prompt + first three target results)."""
    sample_index = build_sample_index(
        DetectionTask().load_samples(str(dataset_directory))
    )
    if render_all:
        jobs = [(name, arm_name) for name in SETS for arm_name in _ARMS]
    else:
        jobs = [(set_name, arm)]
    for name, arm_name in jobs:
        _render_set_arm(
            set_name=name,
            arm=arm_name,
            sample_index=sample_index,
            output_directory=output_directory,
            max_edge=max_edge,
        )


if __name__ == "__main__":
    main()
