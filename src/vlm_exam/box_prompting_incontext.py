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
from typing import Any

import numpy as np
import openai
import supervision as sv
from PIL import Image, ImageDraw

from vlm_exam.box_prompting import PREDICTION_LABEL
from vlm_exam.box_prompting_round2 import (
    DISPLAY_NEGATIVE_HEX,
    DISPLAY_POSITIVE_HEX,
    DISPLAY_PREDICTION_HEX,
)
from vlm_exam.format_probe import extract_json_payload
from vlm_exam.providers.base import EMPTY_RESPONSE_TEXT, call_with_retries
from vlm_exam.providers.image_upload import (
    OPENROUTER_JPEG_QUALITY,
    OPENROUTER_MAX_BASE64_BYTES,
    jpeg_data_url_under_max_base64_bytes,
)
from vlm_exam.tasks.detection import (
    DetectionCoordinateFormat,
    DetectionSample,
    compute_image_map50,
    parse_prediction,
)

Box = tuple[float, float, float, float]

PROVIDER_MODEL_ID = "qwen3.8-max"
"""DashScope model id called by these experiments."""

MAX_EXAMPLES = 3
"""Maximum positive and maximum negative reference boxes for the multi arm."""

SETS: dict[str, tuple[str, ...]] = {
    "set1_bottle_cap": (
        "frame_0002_03_jpg.rf.dda0be95dbf94773efee5983feb50582.jpg",
        "frame_0008_jpg.rf.87aed24f253375f69a3bb398f6521d18.jpg",
        "frame_0086_jpg.rf.5ae0e80356e2c5bb993a052e5023c7f1.jpg",
        "frame_0089_jpg.rf.c7cd5f3b9d3ee6d5f30b8357b148d8d4.jpg",
        "frame_0102_jpg.rf.17d6c3b1d93b1bd2e8d8880687a7f794.jpg",
    ),
    "set2_football": (
        "a9f16c_8_1_png_jpg.rf.21f65c07fc1fd4b937623a9b2c4bf919.jpg",
        "2e57b9_3_10_png_jpg.rf.93746b4a960366b1a012ad339a5d68e1.jpg",
        "08fd33_0_1_png_jpg.rf.49b1ec7a3f580266e0267dce06883980.jpg",
        "42ba34_3_4_png_jpg.rf.963d6715a391a3a2fee3176cace44325.jpg",
        "42ba34_9_9_png_jpg.rf.30ad6b17178011a7360591145cf422a2.jpg",
    ),
    "set3_logic_gate": (
        "45_png.rf.5c5cd31ecf12f414c9d6c446b6599fc0.jpg",
        "54_png.rf.e797ccd3ab1732fcc95dc8f12eac1e73.jpg",
        "66_png.rf.39bf6d177f61078e6d35cc7449f636fc.jpg",
        "img268_png.rf.77932c9124f7e63f288e2be4179d983b.jpg",
        "img283_png.rf.c0609f1f38390a89165e7c735d693ab6.jpg",
    ),
    "set4_technical_drawing": (
        "V000087_0_0-Drive-Shaft_jpeg_jpg.rf.148a90e2fb96f98fb22ea082d09655d1.jpg",
        "V000078_0_0_jpeg_jpg.rf.ea93b54e17810ab845506f60bf69b7dd.jpg",
        "indir-2-_jpeg_jpg.rf.cabde9f9d89b7527f52820f6df64e1b7.jpg",
        "indir-1-_jpeg_jpg.rf.8d614705afcf35f2ac222aa7d9e608ff.jpg",
        "ind_jpeg_jpg.rf.838960e9a8a04626878832d54c63b122.jpg",
    ),
    "set5_basketball": (
        "basketball-player-detection_boston-celtics-new-york-knicks-game-1-q1-03-16-03-11-0135_png.rf.87404d3bb76bef98d9e926e5ec1e2966.jpg",
        "basketball-player-detection_boston-celtics-new-york-knicks-game-1-q1-04-28-04-20-0185_png.rf.5bf3baa6a3ce5adfe29dab142647c8bc.jpg",
        "basketball-player-detection_boston-celtics-new-york-knicks-game-1-q1-05-13-05-09-0000_png.rf.31fd4573143b8c82031c8804c4754e40.jpg",
        "basketball-player-detection_boston-celtics-new-york-knicks-game-4-q1-00-05-00-01-0115_png.rf.268163bdbdd382dbeaa9df9d19cf50ae.jpg",
        "basketball-player-detection_boston-celtics-new-york-knicks-game-4-q1-01-22-01-16-0112_png.rf.813de841eef35ee5de0aa209e2d3e51f.jpg",
    ),
    "set6_solar_hotspot": (
        "DJI_20230529095121_0346_T_JPG_jpg.rf.a2baa2ffdce5ec84e5f5d22342f6b310.jpg",
        "DJI_20230529095953_0646_T_JPG_jpg.rf.9806615efd3123cda770b6e1a3358930.jpg",
        "DJI_20230529111632_0087_T_JPG_jpg.rf.a0bcf6f9fb3a8158bd511491a1f333d9.jpg",
        "DJI_20230529111835_0160_T_JPG_jpg.rf.cb5b9a9d87fdc7c9ff407eb5530abb88.jpg",
        "DJI_20230529112019_0221_T_JPG_jpg.rf.8e72d3393da7d92494fccfb7bb75bfc8.jpg",
    ),
    "set7_banana_tree": (
        "uav20221018_ixfZ_3857_0_png.rf.68f3022f53a4bfe08c731fb3a58473cf.jpg",
        "uav20221018_ixfZ_3857_14_png.rf.64d1ce8897202b1ab5141c12ebcde0db.jpg",
        "uav20221018_ixfZ_3857_215_png.rf.8020e1329eefd7e68bf3f7c0394e8bdf.jpg",
        "uav20221018_ixfZ_3857_391_png.rf.363cfc1a99bee6043b1535d35e5cf8f5.jpg",
        "uav20221018_ixfZ_3857_431_png.rf.d4d8056faccd6284b96584515af21de0.jpg",
    ),
}
"""Seven experiment sets, five images each in the requested order."""

TARGET_CLASS: dict[str, str] = {
    "set1_bottle_cap": "soda bottle cap",
    "set2_football": "player",
    "set3_logic_gate": "not gate symbol",
    "set4_technical_drawing": "dimension annotation",
    "set5_basketball": "player",
    "set6_solar_hotspot": "solar panel hot spot",
    "set7_banana_tree": "banana tree",
}
"""Detection target class per set."""

_ORDINALS = ("second", "third", "fourth", "fifth", "sixth", "seventh", "eighth")
_SENT_POSITIVE_COLOR = (255, 0, 0)
_SENT_NEGATIVE_COLOR = (0, 80, 255)
_FILL_ALPHA = 82
_GAP = 12
_BACKGROUND_COLOR = (20, 20, 20)

_BASELINE_OUTPUT_CLAUSE = (
    "Output a JSON object with the keys {keys}, where {key_meanings}. "
    "Each value must be a JSON list where each entry contains the 2D "
    'bounding box in the key "box_2d" and the text label in the key '
    '"label". The "box_2d" value must be [x_min, y_min, x_max, y_max]: '
    "the top-left and bottom-right corners as integers between 0 and "
    "1000, normalized to the width (x) and height (y) of the image the "
    f'entry belongs to. Use the label "{PREDICTION_LABEL}" for every '
    "entry. Return only the JSON object, with no extra text."
)

_SINGLE_TARGET_OUTPUT_CLAUSE = (
    'Output a JSON object with a single key "target" whose value is a '
    "JSON list where each entry contains the 2D bounding box in the key "
    '"box_2d" and the text label in the key "label". The "box_2d" value '
    "must be [x_min, y_min, x_max, y_max]: integers between 0 and 1000, "
    "normalized to the width (x) and height (y) of the last image. Use "
    f'the label "{PREDICTION_LABEL}" for every entry. Return only the '
    "JSON object, with no extra text."
)


def resolve_image_name(name: str, sample_index: dict[str, DetectionSample]) -> str:
    """Resolve a possibly extension-less name to a dataset key.

    Args:
        name: Image name, with or without a file extension.
        sample_index: Mapping of image basename to detection sample.

    Returns:
        The matching key present in ``sample_index``.
    """
    if name in sample_index:
        return name
    for suffix in (".jpg", ".png", ".jpeg"):
        if name + suffix in sample_index:
            return name + suffix
    raise KeyError(name)


def class_id(sample: DetectionSample, class_name: str) -> int:
    """Return the integer id of a class name in a sample.

    Args:
        sample: Detection sample.
        class_name: Class name to look up.

    Returns:
        The class index.
    """
    return sample.classes.index(class_name)


def class_boxes(sample: DetectionSample, class_name: str) -> tuple[Box, ...]:
    """Return all ground-truth boxes of a class, largest area first.

    Args:
        sample: Detection sample.
        class_name: Class name to select.

    Returns:
        Absolute-pixel boxes ordered by descending area.
    """
    if class_name not in sample.classes:
        return ()
    mask = sample.ground_truth.class_id == class_id(sample, class_name)
    boxes = [
        tuple(float(value) for value in box) for box in sample.ground_truth.xyxy[mask]
    ]
    boxes.sort(key=_box_area, reverse=True)
    return tuple(boxes)


def _box_area(box: Box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def select_negatives(
    sample: DetectionSample,
    positive_class_name: str,
    count: int,
) -> tuple[tuple[Box, ...], tuple[str, ...]]:
    """Select up to ``count`` negative-example boxes from other classes.

    Mirrors the round-2 heuristic: one largest box per other class first
    (classes ordered by instance count then class id), then next-largest.

    Args:
        sample: Detection sample.
        positive_class_name: The target class to exclude.
        count: Maximum number of negatives.

    Returns:
        Tuple of (negative boxes, their class names).
    """
    if positive_class_name not in sample.classes:
        return (), ()
    positive_class = class_id(sample, positive_class_name)
    class_ids = sample.ground_truth.class_id
    unique_ids, counts = np.unique(class_ids, return_counts=True)
    other_classes = [
        int(cid)
        for cid, _ in sorted(
            zip(unique_ids, counts), key=lambda item: (-item[1], item[0])
        )
        if int(cid) != positive_class
    ]
    candidates: list[tuple[int, int, int, Box]] = []
    for class_order, cid in enumerate(other_classes):
        boxes = sample.ground_truth.xyxy[class_ids == cid]
        area_order = np.argsort([-_box_area(box) for box in boxes], kind="stable")
        for rank, index in enumerate(area_order):
            candidates.append(
                (rank, class_order, cid, tuple(float(value) for value in boxes[index]))
            )
    candidates.sort(key=lambda item: (item[0], item[1]))
    negatives = tuple(box for _, _, _, box in candidates[:count])
    negative_classes = tuple(sample.classes[cid] for _, _, cid, _ in candidates[:count])
    return negatives, negative_classes


def select_reference_boxes(
    sample: DetectionSample,
    class_name: str,
    arm: str,
) -> tuple[tuple[Box, ...], tuple[Box, ...], tuple[str, ...]]:
    """Choose the reference boxes for a baseline arm.

    Args:
        sample: Prompt-image detection sample.
        class_name: Target class.
        arm: ``"single"`` (largest positive only, no negatives) or
            ``"multi"`` (up to :data:`MAX_EXAMPLES` positives and negatives).

    Returns:
        Tuple of (positives, negatives, negative class names).
    """
    positives = class_boxes(sample, class_name)
    if arm == "single":
        return positives[:1], (), ()
    negatives, negative_classes = select_negatives(sample, class_name, MAX_EXAMPLES)
    return positives[:MAX_EXAMPLES], negatives, negative_classes


def normalize_box(box: Box, width: int, height: int) -> tuple[int, int, int, int]:
    """Scale an absolute-pixel box to integers in the 0-1000 range.

    Args:
        box: Absolute-pixel box.
        width: Image width.
        height: Image height.

    Returns:
        The normalized box.
    """
    x_min, y_min, x_max, y_max = box
    return (
        max(0, min(1000, round(x_min / width * 1000))),
        max(0, min(1000, round(y_min / height * 1000))),
        max(0, min(1000, round(x_max / width * 1000))),
        max(0, min(1000, round(y_max / height * 1000))),
    )


def draw_sent_boxes(
    image: Image.Image,
    positives: tuple[Box, ...],
    negatives: tuple[Box, ...],
) -> Image.Image:
    """Draw positives in red and negatives in blue (the sent-image colors).

    Args:
        image: Original RGB image.
        positives: Positive example boxes.
        negatives: Negative example boxes.

    Returns:
        Annotated copy sent to the model.
    """
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    line_width = max(3, round(max(image.size) / 300))
    for box in negatives:
        draw.rectangle(box, outline=_SENT_NEGATIVE_COLOR, width=line_width)
    for box in positives:
        draw.rectangle(box, outline=_SENT_POSITIVE_COLOR, width=line_width)
    return annotated


def _ordinal_phrase(target_count: int) -> str:
    names = _ORDINALS[:target_count]
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


def _keyed_output_clause(target_count: int) -> str:
    names = _ORDINALS[:target_count]
    keys = ", ".join(f'"image_{index + 2}"' for index in range(target_count))
    key_meanings = ", ".join(
        f'key "image_{index + 2}" refers to the {names[index]} image'
        for index in range(target_count)
    )
    return _BASELINE_OUTPUT_CLAUSE.format(keys=keys, key_meanings=key_meanings)


def build_baseline_prompt(
    arm: str,
    target_count: int,
    has_negatives: bool,
) -> str:
    """Build the single-request baseline prompt.

    Args:
        arm: ``"single"`` or ``"multi"``.
        target_count: Number of clean target images following the prompt.
        has_negatives: Whether the prompt image carries blue negatives.

    Returns:
        The prompt text.
    """
    ordinals = _ordinal_phrase(target_count)
    if arm == "single":
        lead = (
            "The first image contains a reference object marked with a red rectangle. "
        )
        instruction = (
            f"Detect all objects in the {ordinals} images that are visually "
            "similar to the marked reference object. Do not report any "
            "detections for the first image, and ignore the rectangle. "
        )
    else:
        lead = (
            "The first image contains example objects of a target type "
            "marked with red rectangles. "
        )
        if has_negatives:
            lead += (
                "The objects marked with blue rectangles in the first image "
                "are counterexamples that must NOT be detected. "
            )
        instruction = (
            f"Detect all objects in the {ordinals} images that are visually "
            "similar to the objects marked in red in the first image. Do not "
            "report any detections for the first image, and ignore the "
            "rectangles. "
        )
    return lead + instruction + _keyed_output_clause(target_count)


def build_count_prompt(example_count: int) -> str:
    """Build the example-count-sweep prompt.

    Args:
        example_count: Number of annotated example images preceding the
            single clean target image.

    Returns:
        The prompt text.
    """
    if example_count == 1:
        lead = (
            "The first image contains example objects of a target type "
            "marked with red rectangles. "
        )
    else:
        lead = (
            f"The first {example_count} images each contain example objects "
            "of a target type marked with red rectangles. "
        )
    instruction = (
        "Detect all objects in the last image that are visually similar to "
        "the objects marked in red. Do not report detections for the "
        "example images, and ignore the rectangles. "
    )
    return lead + instruction + _SINGLE_TARGET_OUTPUT_CLAUSE


def iterative_turn1_text() -> str:
    """Prompt text accompanying the first annotated example and target.

    Returns:
        The turn-one instruction text.
    """
    return (
        "The first image contains example objects of a target type marked "
        "with red rectangles. The second image is a new image. Detect all "
        "objects in the second image that are visually similar to the "
        "objects marked in red in the first image, ignoring the rectangles. "
        + _SINGLE_TARGET_OUTPUT_CLAUSE
    )


def iterative_next_text(correction: str | None) -> str:
    """Prompt text for a follow-up turn presenting the next target image.

    Args:
        correction: Optional correction sentence about the previous image.

    Returns:
        The follow-up instruction text.
    """
    prefix = (correction + " ") if correction else ""
    return (
        prefix + "Here is the next image. Detect all objects in it that are "
        "visually similar to the example objects marked in red in the "
        "first image. " + _SINGLE_TARGET_OUTPUT_CLAUSE
    )


def correction_text(boxes: tuple[Box, ...], width: int, height: int) -> str:
    """Describe the correct boxes for the previous image as a correction.

    Args:
        boxes: Ground-truth boxes for the previously shown image.
        width: Previous image width.
        height: Previous image height.

    Returns:
        A sentence listing the correct normalized boxes.
    """
    normalized = [normalize_box(box, width, height) for box in boxes]
    listing = ", ".join(
        f"[{item[0]}, {item[1]}, {item[2]}, {item[3]}]" for item in normalized
    )
    return (
        f"For reference, the correct objects in the previous image were at "
        f"these {len(normalized)} bounding boxes ([x_min, y_min, x_max, "
        f"y_max], integers 0-1000 normalized to that image): {listing}. Use "
        "this feedback to improve your detections."
    )


def image_content(image: Image.Image) -> tuple[dict[str, Any], list[int]]:
    """Encode a PIL image as an OpenAI image content part.

    Args:
        image: RGB image to encode.

    Returns:
        Tuple of the content dict and the uploaded (width, height).
    """
    data_url, uploaded_size = jpeg_data_url_under_max_base64_bytes(
        image, OPENROUTER_MAX_BASE64_BYTES, quality=OPENROUTER_JPEG_QUALITY
    )
    return (
        {"type": "image_url", "image_url": {"url": data_url}},
        [uploaded_size[0], uploaded_size[1]],
    )


def call_messages(
    client: openai.OpenAI,
    messages: list[dict[str, Any]],
    max_output_tokens: int = 16384,
) -> dict[str, Any]:
    """Send an arbitrary multi-turn message list to Qwen3.8-Max.

    Args:
        client: DashScope OpenAI-compatible client.
        messages: Full conversation so far.
        max_output_tokens: Maximum completion tokens.

    Returns:
        Raw output text plus token and latency telemetry.
    """
    response, retry_stats = call_with_retries(
        lambda: client.chat.completions.create(
            model=PROVIDER_MODEL_ID,
            max_tokens=max_output_tokens,
            messages=messages,
            extra_body={"enable_thinking": True},
        )
    )
    if not response.choices:
        answer = EMPTY_RESPONSE_TEXT
    else:
        message = response.choices[0].message
        answer = (message.content or EMPTY_RESPONSE_TEXT).strip()
    usage = response.usage
    return {
        "raw_output": answer,
        "input_tokens": usage.prompt_tokens if usage else None,
        "output_tokens": usage.completion_tokens if usage else None,
        "inference_seconds": retry_stats.inference_seconds,
        "attempts": retry_stats.attempts,
    }


def user_message(
    images: list[Image.Image],
    text: str,
) -> tuple[dict[str, Any], list[list[int]]]:
    """Build a user message with images followed by text.

    Args:
        images: Images to include, in order.
        text: Trailing text prompt.

    Returns:
        Tuple of the message dict and the uploaded image sizes.
    """
    content: list[dict[str, Any]] = []
    sizes: list[list[int]] = []
    for image in images:
        part, size = image_content(image)
        content.append(part)
        sizes.append(size)
    content.append({"type": "text", "text": text})
    return {"role": "user", "content": content}, sizes


def parse_entries(
    payload: Any,
    key: str | None,
    sample: DetectionSample,
) -> tuple[sv.Detections, bool]:
    """Parse one image's predictions from a JSON payload.

    Args:
        payload: Parsed JSON payload (object or list).
        key: Object key to read, or ``None`` to accept a top-level list or
            the first list value.
        sample: Target sample for coordinate scaling.

    Returns:
        Tuple of detections and a parse-failure flag.
    """
    entries: list[Any] | None = None
    if key is not None and isinstance(payload, dict):
        candidate = payload.get(key)
        entries = candidate if isinstance(candidate, list) else None
    elif isinstance(payload, list):
        entries = payload
    elif isinstance(payload, dict):
        for value in payload.values():
            if isinstance(value, list):
                entries = value
                break
    if entries is None:
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


def payload_from_output(raw_output: str) -> Any:
    """Extract the JSON payload from a raw model response.

    Args:
        raw_output: Model response text.

    Returns:
        The parsed JSON payload (may be ``None``).
    """
    payload, _ = extract_json_payload(raw_output)
    return payload


def ground_truth_detections(boxes: tuple[Box, ...]) -> sv.Detections:
    """Wrap absolute-pixel boxes as single-class detections.

    Args:
        boxes: Absolute-pixel boxes.

    Returns:
        Single-class detections.
    """
    if not boxes:
        return sv.Detections.empty()
    xyxy = np.array(boxes, dtype=np.float32)
    return sv.Detections(xyxy=xyxy, class_id=np.zeros(len(xyxy), dtype=int))


def image_map50(detections: sv.Detections, boxes: tuple[Box, ...]) -> float:
    """Compute image mAP@50 of detections against target boxes.

    Args:
        detections: Predicted detections.
        boxes: Ground-truth boxes of the target class.

    Returns:
        The mAP@50 in the 0-1 range.
    """
    return compute_image_map50(detections, ground_truth_detections(boxes))


def _display_rgb(hex_color: str) -> tuple[int, int, int]:
    color = sv.Color.from_hex(hex_color)
    return (color.r, color.g, color.b)


def annotate_box_groups(
    image: Image.Image,
    groups: list[tuple[tuple[Box, ...], tuple[int, int, int]]],
) -> Image.Image:
    """Draw filled, outlined box groups onto an image copy.

    Args:
        image: Original RGB image.
        groups: Ordered (boxes, RGB color) pairs.

    Returns:
        Annotated copy with semi-transparent fills and solid outlines.
    """
    line_width = max(3, round(max(image.size) / 300))
    fill_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    fill_draw = ImageDraw.Draw(fill_layer)
    for boxes, color in groups:
        for box in boxes:
            fill_draw.rectangle(box, fill=(*color, _FILL_ALPHA))
    annotated = Image.alpha_composite(image.convert("RGBA"), fill_layer).convert("RGB")
    draw = ImageDraw.Draw(annotated)
    for boxes, color in groups:
        for box in boxes:
            draw.rectangle(box, outline=color, width=line_width)
    return annotated


def prompt_panel(
    image: Image.Image,
    positives: tuple[Box, ...],
    negatives: tuple[Box, ...],
) -> Image.Image:
    """Render an example panel: green positives, red negatives.

    Args:
        image: Original RGB image.
        positives: Positive example boxes.
        negatives: Negative example boxes.

    Returns:
        Annotated copy.
    """
    return annotate_box_groups(
        image,
        [
            (negatives, _display_rgb(DISPLAY_NEGATIVE_HEX)),
            (positives, _display_rgb(DISPLAY_POSITIVE_HEX)),
        ],
    )


def result_panel(image: Image.Image, detections: sv.Detections) -> Image.Image:
    """Render a result panel: blue predictions.

    Args:
        image: Original RGB image.
        detections: Predicted detections.

    Returns:
        Annotated copy.
    """
    predictions = tuple(tuple(float(value) for value in box) for box in detections.xyxy)
    return annotate_box_groups(
        image, [(predictions, _display_rgb(DISPLAY_PREDICTION_HEX))]
    )


def stack_horizontal(first: Image.Image, second: Image.Image) -> Image.Image:
    """Place two panels side by side on a dark background.

    Args:
        first: Left panel.
        second: Right panel.

    Returns:
        The combined image.
    """
    height = max(first.size[1], second.size[1])
    width = first.size[0] + second.size[0] + _GAP
    combined = Image.new("RGB", (width, height), _BACKGROUND_COLOR)
    combined.paste(first, (0, 0))
    combined.paste(second, (first.size[0] + _GAP, 0))
    return combined
