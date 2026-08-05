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
import numpy as np
import supervision as sv
from dotenv import load_dotenv
from PIL import Image, ImageDraw

from vlm_exam.box_prompting import (
    PREDICTION_LABEL,
    PROVIDER_MODEL_ID,
    call_qwen,
    load_case_image,
)
from vlm_exam.box_prompting_round2 import (
    DISPLAY_NEGATIVE_HEX,
    DISPLAY_POSITIVE_HEX,
    DISPLAY_PREDICTION_HEX,
    build_round2_client,
    draw_example_boxes,
)
from vlm_exam.format_probe import extract_json_payload
from vlm_exam.providers.image_upload import resize_image_to_max_edge
from vlm_exam.tasks.detection import (
    DetectionCoordinateFormat,
    DetectionSample,
    DetectionTask,
    build_sample_index,
    compute_image_map50,
    parse_prediction,
)

_Box = tuple[float, float, float, float]

_DEFAULT_PROMPT_IMAGE = "IMG_3898_JPG.rf.7efec82c7f803d0db60f9543524df2d2.jpg"
_DEFAULT_TARGET_IMAGES = (
    "IMG_3864_JPG.rf.85dd012a4824e9d1050c085dd0a2e519.jpg",
    "IMG_3900_JPG.rf.5ee75659ac02193213ba7636038947a4.jpg",
)
_DEFAULT_CLASS_NAME = "100 poker chip"
_MAX_NEGATIVES = 3
_FILL_ALPHA = 82
_GAP = 12
_BACKGROUND_COLOR = (20, 20, 20)
_ORDINALS = ("second", "third", "fourth", "fifth", "sixth", "seventh", "eighth")

_LEAD = (
    "The first image contains example objects of a target type marked "
    "with red rectangles. The objects marked with blue rectangles in the "
    "first image are counterexamples that must NOT be detected. "
)

_INSTRUCTION = (
    "Detect all objects in the {ordinals} images that are visually "
    "similar to the objects marked in red in the first image. Do not "
    "report any detections for the first image, and ignore the "
    "rectangles. "
)

_OUTPUT_CLAUSE = (
    "Output a JSON object with the keys {keys}, where {key_meanings}. "
    "Each value must be a JSON list where each entry contains the 2D "
    'bounding box in the key "box_2d" and the text label in the key '
    '"label". The "box_2d" value must be [x_min, y_min, x_max, y_max]: '
    "the top-left and bottom-right corners as integers between 0 and "
    "1000, normalized to the width (x) and height (y) of the image the "
    f'entry belongs to. Use the label "{PREDICTION_LABEL}" for every '
    "entry. Return only the JSON object, with no extra text."
)


def _class_id(sample: DetectionSample, class_name: str) -> int:
    return sample.classes.index(class_name)


def _class_boxes(sample: DetectionSample, class_name: str) -> tuple[_Box, ...]:
    mask = sample.ground_truth.class_id == _class_id(sample, class_name)
    return tuple(
        tuple(float(value) for value in box) for box in sample.ground_truth.xyxy[mask]
    )


def _box_area(box: _Box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _select_negatives(
    sample: DetectionSample,
    positive_class_name: str,
    count: int,
) -> tuple[tuple[_Box, ...], tuple[str, ...]]:
    positive_class = _class_id(sample, positive_class_name)
    class_ids = sample.ground_truth.class_id
    unique_ids, counts = np.unique(class_ids, return_counts=True)
    other_classes = [
        int(class_id)
        for class_id, _ in sorted(
            zip(unique_ids, counts), key=lambda item: (-item[1], item[0])
        )
        if int(class_id) != positive_class
    ]
    candidates: list[tuple[int, int, int, _Box]] = []
    for class_order, class_id in enumerate(other_classes):
        boxes = sample.ground_truth.xyxy[class_ids == class_id]
        area_order = np.argsort([-_box_area(box) for box in boxes], kind="stable")
        for rank, index in enumerate(area_order):
            candidates.append(
                (
                    rank,
                    class_order,
                    class_id,
                    tuple(float(value) for value in boxes[index]),
                )
            )
    candidates.sort(key=lambda item: (item[0], item[1]))
    negatives = tuple(box for _, _, _, box in candidates[:count])
    negative_classes = tuple(
        sample.classes[class_id] for _, _, class_id, _ in candidates[:count]
    )
    return negatives, negative_classes


def _build_prompt(target_count: int) -> str:
    ordinal_names = _ORDINALS[:target_count]
    if len(ordinal_names) == 1:
        ordinals = ordinal_names[0]
    else:
        ordinals = ", ".join(ordinal_names[:-1]) + " and " + ordinal_names[-1]
    keys = ", ".join(f'"image_{index + 2}"' for index in range(target_count))
    key_meanings = ", ".join(
        f'key "image_{index + 2}" refers to the {ordinal_names[index]} image'
        for index in range(target_count)
    )
    return (
        _LEAD
        + _INSTRUCTION.format(ordinals=ordinals)
        + _OUTPUT_CLAUSE.format(keys=keys, key_meanings=key_meanings)
    )


def _display_rgb(hex_color: str) -> tuple[int, int, int]:
    color = sv.Color.from_hex(hex_color)
    return (color.r, color.g, color.b)


def _annotate_box_groups(
    image: Image.Image,
    groups: list[tuple[tuple[_Box, ...], tuple[int, int, int]]],
) -> Image.Image:
    line_width = max(3, round(max(image.size) / 300))
    fill_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    fill_draw = ImageDraw.Draw(fill_layer)
    for boxes, color in groups:
        for box in boxes:
            fill_draw.rectangle(box, fill=(*color, _FILL_ALPHA))
    annotated = Image.alpha_composite(image.convert("RGBA"), fill_layer)
    annotated = annotated.convert("RGB")
    draw = ImageDraw.Draw(annotated)
    for boxes, color in groups:
        for box in boxes:
            draw.rectangle(box, outline=color, width=line_width)
    return annotated


def _prompt_panel(
    image: Image.Image,
    positives: tuple[_Box, ...],
    negatives: tuple[_Box, ...],
) -> Image.Image:
    return _annotate_box_groups(
        image,
        [
            (negatives, _display_rgb(DISPLAY_NEGATIVE_HEX)),
            (positives, _display_rgb(DISPLAY_POSITIVE_HEX)),
        ],
    )


def _result_panel(image: Image.Image, detections: sv.Detections) -> Image.Image:
    predictions = tuple(tuple(float(value) for value in box) for box in detections.xyxy)
    return _annotate_box_groups(
        image,
        [(predictions, _display_rgb(DISPLAY_PREDICTION_HEX))],
    )


def _stack_horizontal(first: Image.Image, second: Image.Image) -> Image.Image:
    height = max(first.size[1], second.size[1])
    width = first.size[0] + second.size[0] + _GAP
    combined = Image.new("RGB", (width, height), _BACKGROUND_COLOR)
    combined.paste(first, (0, 0))
    combined.paste(second, (first.size[0] + _GAP, 0))
    return combined


def _parse_target_entries(
    payload: Any,
    key: str,
    sample: DetectionSample,
) -> tuple[sv.Detections, bool]:
    entries = payload.get(key) if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        return sv.Detections.empty(), True
    if not entries:
        return sv.Detections.empty(), False
    collapsed = []
    for entry in entries:
        if isinstance(entry, dict):
            relabeled = dict(entry)
            relabeled["label"] = PREDICTION_LABEL
            collapsed.append(relabeled)
    detections = parse_prediction(
        json.dumps(collapsed),
        (sample.image_width, sample.image_height),
        [PREDICTION_LABEL],
        coordinate_format=DetectionCoordinateFormat.XYXY_NORMALIZED_0_TO_1000,
    )
    return detections, len(detections) == 0


def _ground_truth_detections(boxes: tuple[_Box, ...]) -> sv.Detections:
    if not boxes:
        return sv.Detections.empty()
    xyxy = np.array(boxes, dtype=np.float32)
    return sv.Detections(xyxy=xyxy, class_id=np.zeros(len(xyxy), dtype=int))


def _pairwise_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    left = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    top = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    right = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    bottom = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])
    intersection = np.clip(right - left, 0, None) * np.clip(bottom - top, 0, None)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - intersection
    return np.where(union > 0, intersection / union, 0.0)


def _leak_count(
    detections: sv.Detections,
    sample: DetectionSample,
    class_name: str,
) -> int:
    if class_name not in sample.classes or len(detections) == 0:
        return 0
    boxes = sample.ground_truth.xyxy[
        sample.ground_truth.class_id == _class_id(sample, class_name)
    ]
    if len(boxes) == 0:
        return 0
    iou = _pairwise_iou(detections.xyxy.astype(float), np.asarray(boxes, dtype=float))
    return int((iou.max(axis=1) >= 0.5).sum())


@click.command()
@click.option(
    "--dataset-directory",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("data/detection/train"),
    show_default=True,
)
@click.option(
    "--output-directory",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("results-box-prompting-qwen38-max-cross"),
    show_default=True,
)
@click.option("--prompt-image", default=_DEFAULT_PROMPT_IMAGE, show_default=True)
@click.option(
    "--target-images",
    default=",".join(_DEFAULT_TARGET_IMAGES),
    show_default=True,
    help="Comma-separated target image names.",
)
@click.option("--class-name", default=_DEFAULT_CLASS_NAME, show_default=True)
@click.option(
    "--max-edge", type=click.IntRange(min=200), default=1600, show_default=True
)
@click.option(
    "--force/--no-force",
    default=False,
    show_default=True,
    help="Re-request even when a raw response already exists.",
)
def main(
    dataset_directory: Path,
    output_directory: Path,
    prompt_image: str,
    target_images: str,
    class_name: str,
    max_edge: int,
    force: bool,
) -> None:
    """Cross-image box prompting: one Qwen3.8-Max request where drawn boxes
    on a prompt image steer detection in separate target images."""
    load_dotenv()
    sample_index = build_sample_index(
        DetectionTask().load_samples(str(dataset_directory))
    )
    target_names = [name.strip() for name in target_images.split(",") if name.strip()]
    prompt_sample = sample_index[prompt_image]
    target_samples = [sample_index[name] for name in target_names]

    positives = _class_boxes(prompt_sample, class_name)
    negatives, negative_classes = _select_negatives(
        prompt_sample, class_name, _MAX_NEGATIVES
    )
    prompt = _build_prompt(len(target_samples))
    click.echo(
        f"Prompt image {prompt_image}: {len(positives)} positives "
        f"({class_name}), negatives from {', '.join(negative_classes)}."
    )

    raw_path = output_directory / "raw" / "response.json"
    if raw_path.exists() and not force:
        record = json.loads(raw_path.read_text())
        click.echo(f"Reusing existing response at {raw_path}")
    else:
        prompt_pil = load_case_image(prompt_sample)
        marked = draw_example_boxes(prompt_pil, positives, negatives)
        images = [marked] + [load_case_image(sample) for sample in target_samples]
        client = build_round2_client()
        record = {
            "model": PROVIDER_MODEL_ID,
            "prompt_image": prompt_image,
            "target_images": target_names,
            "class_name": class_name,
            "positive_xyxy": [list(box) for box in positives],
            "negative_xyxy": [list(box) for box in negatives],
            "negative_classes": list(negative_classes),
            "prompt": prompt,
        }
        click.echo("Sending single 3-image request to Qwen3.8-Max...")
        record.update(call_qwen(client, images=images, prompt=prompt))
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(json.dumps(record, indent=2))
        click.echo(f"Raw response written to {raw_path}")

    payload, _ = extract_json_payload(record.get("raw_output", ""))
    lines = [
        "# Qwen3.8-Max cross-image box prompting",
        "",
        f"- Prompt image: `{prompt_image}`",
        f"- Positive class: {class_name} ({len(positives)} positive prompts)",
        f"- Negative prompts: {', '.join(negative_classes)}",
        f"- Output tokens: {record.get('output_tokens')} | "
        f"inference: {record.get('inference_seconds', 0):.1f}s",
        "",
        "| Target image | Targets | Predicted | mAP@50 | Parse failed | "
        "Negative-class leaks |",
        "|---|---:|---:|---:|---|---:|",
    ]

    renders_directory = output_directory / "renders"
    stacked_directory = renders_directory / "stacked"
    stacked_directory.mkdir(parents=True, exist_ok=True)

    prompt_pil = load_case_image(prompt_sample)
    prompt_panel = resize_image_to_max_edge(
        _prompt_panel(prompt_pil, positives, negatives), max_edge
    )
    prompt_panel.save(renders_directory / "prompt.png")

    for index, (name, sample) in enumerate(zip(target_names, target_samples)):
        key = f"image_{index + 2}"
        detections, failed = _parse_target_entries(payload, key, sample)
        targets = _ground_truth_detections(_class_boxes(sample, class_name))
        map50 = compute_image_map50(detections, targets)
        leaks = sum(
            _leak_count(detections, sample, negative_class)
            for negative_class in set(negative_classes)
        )
        lines.append(
            f"| {name} | {len(targets)} | {len(detections)} | "
            f"{map50 * 100:.0f}% | {'yes' if failed else 'no'} | {leaks} |"
        )
        click.echo(
            f"{key} -> {name}: predicted {len(detections)} / "
            f"targets {len(targets)}, mAP@50 {map50 * 100:.0f}%, leaks {leaks}"
        )

        target_pil = load_case_image(sample)
        result_panel = resize_image_to_max_edge(
            _result_panel(target_pil, detections), max_edge
        )
        stem = Path(name).stem
        result_panel.save(renders_directory / f"{stem}.png")
        _stack_horizontal(prompt_panel, result_panel).save(
            stacked_directory / f"{stem}.png"
        )

    analysis_directory = output_directory / "analysis"
    analysis_directory.mkdir(parents=True, exist_ok=True)
    report_path = analysis_directory / "report.md"
    report_path.write_text("\n".join(lines) + "\n")
    click.echo(f"Report written to {report_path}")
    click.echo(f"Renders written to {renders_directory}")


if __name__ == "__main__":
    main()
