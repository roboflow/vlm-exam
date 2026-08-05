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
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import openai
import supervision as sv
from PIL import Image, ImageDraw

from vlm_exam.box_prompting import (
    _OUTPUT_CLAUSE,
    PROVIDER_MODEL_ID,
    REFERENCE_IOU_THRESHOLD,
    build_arm_prompt,
    call_qwen,
    draw_reference_box,
    load_arm_records,
    load_case_image,
    record_to_detections,
)
from vlm_exam.format_matrix import select_matrix_images
from vlm_exam.format_probe import completed_images
from vlm_exam.providers.image_upload import resize_image_to_max_edge
from vlm_exam.tasks.detection import DetectionSample, compute_image_map50
from vlm_exam.visualization.detection import draw_image_legend

ROUND2_ARMS = (
    "text_box_single",
    "drawn_box_single",
    "text_box_multi",
    "drawn_box_multi",
)
"""The four arms of the second experiment round."""

MAX_EXAMPLES = 3
"""Maximum positive and maximum negative example boxes per image."""

MAX_OBJECTS_PER_IMAGE = 50
"""Images with more total ground-truth objects than this are excluded."""

REQUEST_TIMEOUT_SECONDS = 300.0
"""Per-request timeout; dense images need far more than the default 120s."""

_DASHSCOPE_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
# Colors baked into the sent images; the prompts name them ("red
# rectangles", "blue rectangles") so they must never change.
_POSITIVE_COLOR = (255, 0, 0)
_NEGATIVE_COLOR = (0, 80, 255)
_GAP = 12

DISPLAY_POSITIVE_HEX = "#00C853"
"""Display color for positive prompt boxes (vivid green)."""

DISPLAY_NEGATIVE_HEX = "#FF1744"
"""Display color for negative prompt boxes (vivid red)."""

DISPLAY_PREDICTION_HEX = "#2979FF"
"""Display color for prediction boxes (vivid blue)."""

_DISPLAY_POSITIVE_RGB = (0, 200, 83)
_DISPLAY_NEGATIVE_RGB = (255, 23, 68)
_DISPLAY_PREDICTION_RGB = (41, 121, 255)
_BOX_FILL_ALPHA = 82

_TEXT_MULTI_POSITIVE_LEAD = (
    "This image contains example objects of a target type. Their bounding "
    "boxes are [x_min, y_min, x_max, y_max] integers between 0 and 1000, "
    "normalized to the image width (x) and height (y): {positives}. "
)

_TEXT_MULTI_NEGATIVE_CLAUSE = (
    "The image also contains counterexample objects that must NOT be "
    "detected, with bounding boxes: {negatives}. "
)

_TEXT_MULTI_INSTRUCTION = (
    "Detect all other objects in the image that are visually similar to "
    "the example objects. Do not include the example objects themselves"
)

_DRAWN_MULTI_POSITIVE_LEAD = (
    "The example objects of a target type in this image are marked with "
    "red rectangles. "
)

_DRAWN_MULTI_NEGATIVE_CLAUSE = (
    "The objects marked with blue rectangles are counterexamples that "
    "must NOT be detected. "
)

_DRAWN_MULTI_INSTRUCTION = (
    "Detect all other objects in the image that are visually similar to "
    "the objects marked in red. Do not include any marked object itself, "
    "and ignore the rectangles. "
)

_Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class ExampleCase:
    """One image with positive and negative example boxes and targets.

    Attributes:
        image_name: Image file basename.
        class_name: Ground-truth class of the positive examples.
        positive_xyxy: Positive example boxes in absolute pixels, largest
            first; the first one equals the round-1 reference.
        negative_xyxy: Negative example boxes (other classes) in absolute
            pixels; empty for single-class images.
        negative_classes: Class name of each negative example box.
        target_xyxy: Remaining positive-class instances; the scoring
            ground truth.
    """

    image_name: str
    class_name: str
    positive_xyxy: tuple[_Box, ...]
    negative_xyxy: tuple[_Box, ...]
    negative_classes: tuple[str, ...]
    target_xyxy: tuple[_Box, ...]


def _box_area(box: np.ndarray) -> float:
    return float((box[2] - box[0]) * (box[3] - box[1]))


def build_example_case(sample: DetectionSample) -> ExampleCase | None:
    """Pick deterministic positive and negative examples for one sample.

    The positive class is the one with the most instances (ties broken by
    the smaller class id). Up to :data:`MAX_EXAMPLES` largest-area
    instances become positives, always leaving at least one target.
    Negatives are up to :data:`MAX_EXAMPLES` instances of other classes:
    the largest instance per other class first (classes ordered by
    instance count, then class id), then the next-largest remaining boxes.

    Args:
        sample: Detection sample with ground-truth boxes.

    Returns:
        The example case, or ``None`` when the image has no class with at
        least two instances or exceeds :data:`MAX_OBJECTS_PER_IMAGE`.
    """
    class_ids = sample.ground_truth.class_id
    if class_ids is None or len(sample.ground_truth) < 2:
        return None
    if len(sample.ground_truth) > MAX_OBJECTS_PER_IMAGE:
        return None
    unique_ids, counts = np.unique(class_ids, return_counts=True)
    best_count = counts.max()
    if best_count < 2:
        return None
    positive_class = int(unique_ids[counts == best_count].min())

    positive_boxes = sample.ground_truth.xyxy[class_ids == positive_class]
    order = np.argsort([-_box_area(box) for box in positive_boxes], kind="stable")
    positive_count = min(MAX_EXAMPLES, len(positive_boxes) - 1)
    positive_indices = list(order[:positive_count])
    positives = tuple(
        tuple(float(value) for value in positive_boxes[index])
        for index in positive_indices
    )
    targets = tuple(
        tuple(float(value) for value in box)
        for index, box in enumerate(positive_boxes)
        if index not in positive_indices
    )

    other_classes = [
        int(class_id)
        for class_id, _ in sorted(
            zip(unique_ids, counts), key=lambda item: (-item[1], item[0])
        )
        if int(class_id) != positive_class
    ]
    negative_candidates: list[tuple[int, int, int, _Box]] = []
    for class_order, class_id in enumerate(other_classes):
        boxes = sample.ground_truth.xyxy[class_ids == class_id]
        area_order = np.argsort([-_box_area(box) for box in boxes], kind="stable")
        for rank, index in enumerate(area_order):
            negative_candidates.append(
                (
                    rank,
                    class_order,
                    class_id,
                    tuple(float(value) for value in boxes[index]),
                )
            )
    # One box per other class first (in class-prominence order), then the
    # next-largest boxes per class.
    negative_candidates.sort(key=lambda item: (item[0], item[1]))
    negatives: list[_Box] = []
    negative_classes: list[str] = []
    for _, _, class_id, box in negative_candidates[:MAX_EXAMPLES]:
        negatives.append(box)
        negative_classes.append(sample.classes[class_id])

    return ExampleCase(
        image_name=Path(sample.image_path).name,
        class_name=sample.classes[positive_class],
        positive_xyxy=positives,
        negative_xyxy=tuple(negatives),
        negative_classes=tuple(negative_classes),
        target_xyxy=targets,
    )


def select_example_cases(
    sample_index: dict[str, DetectionSample],
    *,
    count: int,
    seed: int = 42,
) -> list[ExampleCase]:
    """Deterministically select usable example cases.

    Walks the same shuffled image order as round 1 and keeps the first
    ``count`` images that yield a valid example case.

    Args:
        sample_index: Mapping of image basename to detection sample.
        count: Number of cases to select.
        seed: Shuffle seed shared with round 1.

    Returns:
        Selected example cases in shuffle order.
    """
    order = select_matrix_images(sample_index, count=len(sample_index), seed=seed)
    cases: list[ExampleCase] = []
    for image_name in order:
        case = build_example_case(sample_index[image_name])
        if case is not None:
            cases.append(case)
        if len(cases) == count:
            break
    return cases


def _normalize_box(
    box: _Box,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x_min, y_min, x_max, y_max = box
    return (
        max(0, min(1000, round(x_min / width * 1000))),
        max(0, min(1000, round(y_min / height * 1000))),
        max(0, min(1000, round(x_max / width * 1000))),
        max(0, min(1000, round(y_max / height * 1000))),
    )


def _format_boxes(
    boxes: tuple[_Box, ...],
    width: int,
    height: int,
) -> str:
    normalized = [_normalize_box(box, width, height) for box in boxes]
    return ", ".join(f"[{box[0]}, {box[1]}, {box[2]}, {box[3]}]" for box in normalized)


def build_round2_prompt(
    arm: str,
    case: ExampleCase,
    sample: DetectionSample,
) -> str:
    """Build the full prompt for one round-2 arm.

    Single arms reuse the round-1 prompts verbatim with the largest
    positive as the reference.

    Args:
        arm: One of :data:`ROUND2_ARMS`.
        case: Example case for the sample.
        sample: Detection sample providing the original image size.

    Returns:
        Prompt text requesting the benchmark output format.
    """
    width = sample.image_width
    height = sample.image_height
    if arm == "text_box_single":
        return build_arm_prompt(
            "text_box", _normalize_box(case.positive_xyxy[0], width, height)
        )
    if arm == "drawn_box_single":
        return build_arm_prompt(
            "drawn_box", _normalize_box(case.positive_xyxy[0], width, height)
        )
    if arm == "text_box_multi":
        prompt = _TEXT_MULTI_POSITIVE_LEAD.format(
            positives=_format_boxes(case.positive_xyxy, width, height)
        )
        instruction_tail = ". "
        if case.negative_xyxy:
            prompt += _TEXT_MULTI_NEGATIVE_CLAUSE.format(
                negatives=_format_boxes(case.negative_xyxy, width, height)
            )
            instruction_tail = ", and do not include the counterexample objects. "
        return prompt + _TEXT_MULTI_INSTRUCTION + instruction_tail + _OUTPUT_CLAUSE
    if arm == "drawn_box_multi":
        prompt = _DRAWN_MULTI_POSITIVE_LEAD
        if case.negative_xyxy:
            prompt += _DRAWN_MULTI_NEGATIVE_CLAUSE
        return prompt + _DRAWN_MULTI_INSTRUCTION + _OUTPUT_CLAUSE
    raise ValueError(f"Unknown arm: {arm!r}")


def draw_example_boxes(
    image: Image.Image,
    positives: tuple[_Box, ...],
    negatives: tuple[_Box, ...],
) -> Image.Image:
    """Return a copy with positives drawn in red and negatives in blue.

    Args:
        image: Original image in RGB mode.
        positives: Positive example boxes in absolute pixels.
        negatives: Negative example boxes in absolute pixels.

    Returns:
        Annotated copy of the image.
    """
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    line_width = max(3, round(max(image.size) / 300))
    for box in negatives:
        draw.rectangle(box, outline=_NEGATIVE_COLOR, width=line_width)
    for box in positives:
        draw.rectangle(box, outline=_POSITIVE_COLOR, width=line_width)
    return annotated


def prepare_round2_images(
    arm: str,
    image: Image.Image,
    case: ExampleCase,
) -> list[Image.Image]:
    """Prepare the image payload for one round-2 arm.

    Args:
        arm: One of :data:`ROUND2_ARMS`.
        image: Original image in RGB mode.
        case: Example case for the sample.

    Returns:
        Ordered list of images to send with the prompt.
    """
    if arm == "text_box_single" or arm == "text_box_multi":
        return [image]
    if arm == "drawn_box_single":
        return [draw_reference_box(image, case.positive_xyxy[0])]
    if arm == "drawn_box_multi":
        return [draw_example_boxes(image, case.positive_xyxy, case.negative_xyxy)]
    raise ValueError(f"Unknown arm: {arm!r}")


def build_round2_client(api_key: str | None = None) -> openai.OpenAI:
    """Create a DashScope client with the extended round-2 timeout.

    Args:
        api_key: Optional DashScope API key; falls back to
            ``DASHSCOPE_API_KEY``.

    Returns:
        Configured client.
    """
    return openai.OpenAI(
        base_url=_DASHSCOPE_BASE_URL,
        api_key=api_key or os.environ.get("DASHSCOPE_API_KEY"),
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=0,
    )


def _collect_case(
    *,
    arm: str,
    case: ExampleCase,
    sample: DetectionSample,
    client: openai.OpenAI,
    output_path: Path,
    file_lock: threading.Lock,
    progress: dict[str, int],
    progress_lock: threading.Lock,
    total: int,
) -> None:
    prompt = build_round2_prompt(arm, case, sample)
    record: dict[str, Any] = {
        "model": PROVIDER_MODEL_ID,
        "arm": arm,
        "reasoning_effort": "low",
        "image": case.image_name,
        "original_width": sample.image_width,
        "original_height": sample.image_height,
        "class_name": case.class_name,
        "positive_xyxy": [list(box) for box in case.positive_xyxy],
        "negative_xyxy": [list(box) for box in case.negative_xyxy],
        "negative_classes": list(case.negative_classes),
        "target_count": len(case.target_xyxy),
        "prompt": prompt,
        "error": None,
    }
    try:
        image = load_case_image(sample)
        images = prepare_round2_images(arm, image, case)
        record.update(call_qwen(client, images=images, prompt=prompt))
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"
    with file_lock:
        with open(output_path, "a") as file:
            file.write(json.dumps(record) + "\n")
    with progress_lock:
        progress[arm] += 1
        done = progress[arm]
    print(f"Completed {arm} on {case.image_name} ({done}/{total})", flush=True)


def run_round2_collection(
    *,
    cases: list[ExampleCase],
    sample_index: dict[str, DetectionSample],
    output_directory: Path,
    max_workers: int = 8,
    api_key: str | None = None,
) -> None:
    """Collect all four arms with a shared request-level worker pool.

    Every (arm, image) request is an independent job; per-arm file locks
    keep the resumable JSONL files consistent under concurrency.

    Args:
        cases: Example cases to probe.
        sample_index: Mapping of image basename to detection sample.
        output_directory: Experiment root; raw files land in ``raw/``.
        max_workers: Concurrent requests across all arms.
        api_key: Optional DashScope API key.
    """
    raw_directory = output_directory / "raw"
    raw_directory.mkdir(parents=True, exist_ok=True)
    with open(output_directory / "cases.json", "w") as file:
        json.dump(
            [
                {
                    "image": case.image_name,
                    "class_name": case.class_name,
                    "positive_xyxy": [list(box) for box in case.positive_xyxy],
                    "negative_xyxy": [list(box) for box in case.negative_xyxy],
                    "negative_classes": list(case.negative_classes),
                    "target_count": len(case.target_xyxy),
                }
                for case in cases
            ],
            file,
            indent=2,
        )

    client = build_round2_client(api_key)
    file_locks = {arm: threading.Lock() for arm in ROUND2_ARMS}
    progress = {arm: 0 for arm in ROUND2_ARMS}
    progress_lock = threading.Lock()

    jobs: list[tuple[str, ExampleCase]] = []
    for arm in ROUND2_ARMS:
        output_path = raw_directory / f"{PROVIDER_MODEL_ID}__{arm}.jsonl"
        done = completed_images(output_path)
        progress[arm] = len([case for case in cases if case.image_name in done])
        jobs.extend((arm, case) for case in cases if case.image_name not in done)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _collect_case,
                arm=arm,
                case=case,
                sample=sample_index[case.image_name],
                client=client,
                output_path=raw_directory / f"{PROVIDER_MODEL_ID}__{arm}.jsonl",
                file_lock=file_locks[arm],
                progress=progress,
                progress_lock=progress_lock,
                total=len(cases),
            )
            for arm, case in jobs
        ]
        for future in futures:
            future.result()
    print("Collection finished.", flush=True)


def target_detections(case: ExampleCase) -> sv.Detections:
    """Ground-truth detections for scoring: the non-example instances.

    Args:
        case: Example case.

    Returns:
        Single-class detections of the target boxes.
    """
    if not case.target_xyxy:
        return sv.Detections.empty()
    xyxy = np.array(case.target_xyxy, dtype=np.float32)
    return sv.Detections(xyxy=xyxy, class_id=np.zeros(len(xyxy), dtype=int))


def _pairwise_iou(boxes: tuple[_Box, ...], xyxy: np.ndarray) -> np.ndarray:
    if not boxes or len(xyxy) == 0:
        return np.zeros((len(boxes), len(xyxy)))
    examples = np.array(boxes, dtype=float)
    left = np.maximum(examples[:, None, 0], xyxy[None, :, 0])
    top = np.maximum(examples[:, None, 1], xyxy[None, :, 1])
    right = np.minimum(examples[:, None, 2], xyxy[None, :, 2])
    bottom = np.minimum(examples[:, None, 3], xyxy[None, :, 3])
    intersection = np.clip(right - left, 0, None) * np.clip(bottom - top, 0, None)
    example_areas = (examples[:, 2] - examples[:, 0]) * (
        examples[:, 3] - examples[:, 1]
    )
    prediction_areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    union = example_areas[:, None] + prediction_areas[None, :] - intersection
    return np.where(union > 0, intersection / union, 0.0)


def score_round2_arm(
    records: list[dict[str, Any]],
    cases_by_image: dict[str, ExampleCase],
    sample_index: dict[str, DetectionSample],
) -> dict[str, Any]:
    """Score one arm's records against the target ground truth.

    Args:
        records: Deduplicated records of one arm.
        cases_by_image: Mapping of image basename to example case.
        sample_index: Mapping of image basename to detection sample.

    Returns:
        Dict with aggregate ``metrics`` and per-image ``images`` details.
    """
    per_image: dict[str, dict[str, Any]] = {}
    map50_values: list[float] = []
    parse_failures = 0
    errors = 0
    positive_redetections = 0
    negative_hits = 0
    predicted_counts: list[int] = []
    tokens_out: list[int] = []
    seconds: list[float] = []
    for record in sorted(records, key=lambda item: item["image"]):
        case = cases_by_image.get(record["image"])
        sample = sample_index.get(record["image"])
        if case is None or sample is None:
            continue
        targets = target_detections(case)
        detail: dict[str, Any] = {
            "class_name": case.class_name,
            "target_boxes": len(targets),
            "has_negatives": bool(case.negative_xyxy),
            "error": record.get("error"),
        }
        if record.get("error") is not None:
            errors += 1
            detail.update({"map50": 0.0, "predicted_boxes": 0})
            map50_values.append(0.0)
            per_image[record["image"]] = detail
            continue
        detections, failed = record_to_detections(record, sample)
        parse_failures += failed
        map50 = compute_image_map50(detections, targets)
        positive_hit = False
        negative_hit = False
        if len(detections) > 0:
            positive_iou = _pairwise_iou(case.positive_xyxy, detections.xyxy)
            positive_hit = bool((positive_iou >= REFERENCE_IOU_THRESHOLD).any())
            negative_iou = _pairwise_iou(case.negative_xyxy, detections.xyxy)
            negative_hit = bool((negative_iou >= REFERENCE_IOU_THRESHOLD).any())
        positive_redetections += positive_hit
        negative_hits += negative_hit
        map50_values.append(map50)
        predicted_counts.append(len(detections))
        detail.update(
            {
                "map50": map50,
                "predicted_boxes": len(detections),
                "parse_failed": failed,
                "positive_redetected": positive_hit,
                "negative_hit": negative_hit,
            }
        )
        per_image[record["image"]] = detail
        if record.get("output_tokens") is not None:
            tokens_out.append(record["output_tokens"])
        if record.get("inference_seconds") is not None:
            seconds.append(record["inference_seconds"])
    target_counts = [detail["target_boxes"] for detail in per_image.values()]
    multiclass_map50 = [
        detail["map50"] for detail in per_image.values() if detail["has_negatives"]
    ]
    single_class_map50 = [
        detail["map50"] for detail in per_image.values() if not detail["has_negatives"]
    ]
    metrics = {
        "images": len(per_image),
        "mean_image_map50": float(np.mean(map50_values)) if map50_values else 0.0,
        "mean_image_map50_multiclass": (
            float(np.mean(multiclass_map50)) if multiclass_map50 else None
        ),
        "mean_image_map50_single_class": (
            float(np.mean(single_class_map50)) if single_class_map50 else None
        ),
        "parse_failures": parse_failures,
        "errors": errors,
        "positive_redetections": positive_redetections,
        "negative_hits": negative_hits,
        "mean_predicted_boxes": (
            float(np.mean(predicted_counts)) if predicted_counts else 0.0
        ),
        "mean_target_boxes": float(np.mean(target_counts)) if target_counts else 0.0,
        "avg_output_tokens": float(np.mean(tokens_out)) if tokens_out else None,
        "avg_seconds": float(np.mean(seconds)) if seconds else None,
    }
    return {"metrics": metrics, "images": per_image}


def run_round2_analysis(
    *,
    raw_directory: Path,
    cases_by_image: dict[str, ExampleCase],
    sample_index: dict[str, DetectionSample],
) -> dict[str, dict[str, Any]]:
    """Score every collected round-2 arm.

    Args:
        raw_directory: Directory holding the raw JSONL files.
        cases_by_image: Mapping of image basename to example case.
        sample_index: Mapping of image basename to detection sample.

    Returns:
        Mapping of arm to its scored results.
    """
    results: dict[str, dict[str, Any]] = {}
    for arm in ROUND2_ARMS:
        records = load_arm_records(raw_directory, arm)
        if records:
            results[arm] = score_round2_arm(records, cases_by_image, sample_index)
    return results


def _format_percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def format_round2_report(results: dict[str, dict[str, Any]]) -> str:
    """Render the round-2 summary as markdown.

    Args:
        results: Per-arm scored results from :func:`run_round2_analysis`.

    Returns:
        Markdown report with arm summary, single-vs-multi deltas, and a
        per-image breakdown.
    """
    lines = ["# Qwen3.8-Max box-prompting round 2", ""]
    lines.append(
        "| Arm | Mean img mAP@50 | Multiclass | Single-class | Parse fail | "
        "Errors | Pos re-detected | Neg hits | Avg pred boxes | "
        "Avg target boxes | Avg out tok | Avg s |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for arm in ROUND2_ARMS:
        if arm not in results:
            continue
        metrics = results[arm]["metrics"]
        out_tokens = (
            f"{metrics['avg_output_tokens']:.0f}"
            if metrics["avg_output_tokens"] is not None
            else "n/a"
        )
        avg_seconds = (
            f"{metrics['avg_seconds']:.1f}"
            if metrics["avg_seconds"] is not None
            else "n/a"
        )
        lines.append(
            f"| {arm} | {_format_percent(metrics['mean_image_map50'])} | "
            f"{_format_percent(metrics['mean_image_map50_multiclass'])} | "
            f"{_format_percent(metrics['mean_image_map50_single_class'])} | "
            f"{metrics['parse_failures']} | {metrics['errors']} | "
            f"{metrics['positive_redetections']} | {metrics['negative_hits']} | "
            f"{metrics['mean_predicted_boxes']:.1f} | "
            f"{metrics['mean_target_boxes']:.1f} | "
            f"{out_tokens} | {avg_seconds} |"
        )
    lines.append("")

    lines.append("## Single vs multi")
    lines.append("")
    for style in ("text_box", "drawn_box"):
        single = results.get(f"{style}_single")
        multi = results.get(f"{style}_multi")
        if single is None or multi is None:
            continue
        delta = (
            multi["metrics"]["mean_image_map50"] - single["metrics"]["mean_image_map50"]
        )
        lines.append(
            f"- {style}: multi {_format_percent(multi['metrics']['mean_image_map50'])} "
            f"vs single {_format_percent(single['metrics']['mean_image_map50'])} "
            f"({delta * 100:+.1f} points)"
        )
    lines.append("")

    image_names = sorted(
        {name for scored in results.values() for name in scored["images"]}
    )
    arms_present = [arm for arm in ROUND2_ARMS if arm in results]
    lines.append("## Per-image mAP@50")
    lines.append("")
    lines.append("| Image | Class | Targets | Neg | " + " | ".join(arms_present) + " |")
    lines.append("|---|---|---:|---:|" + "---:|" * len(arms_present))
    for image_name in image_names:
        cells: list[str] = []
        class_name = ""
        target_boxes = ""
        has_negatives = ""
        for arm in arms_present:
            detail = results[arm]["images"].get(image_name)
            if detail is None:
                cells.append("n/a")
                continue
            class_name = detail["class_name"]
            target_boxes = str(detail["target_boxes"])
            has_negatives = "yes" if detail["has_negatives"] else "no"
            if detail.get("error") is not None:
                cells.append("err")
            else:
                cells.append(f"{detail['map50'] * 100:.0f}%")
        lines.append(
            f"| {image_name} | {class_name} | {target_boxes} | {has_negatives} | "
            + " | ".join(cells)
            + " |"
        )
    return "\n".join(lines)


def write_round2_artifacts(
    *,
    output_directory: Path,
    results: dict[str, dict[str, Any]],
) -> Path:
    """Write the round-2 summary JSON and markdown report.

    Args:
        output_directory: Experiment root directory.
        results: Per-arm scored results.

    Returns:
        Path of the written markdown report.
    """
    analysis_directory = output_directory / "analysis"
    analysis_directory.mkdir(parents=True, exist_ok=True)
    with open(analysis_directory / "summary.json", "w") as file:
        json.dump(results, file, indent=2)
    report_path = analysis_directory / "report.md"
    with open(report_path, "w") as file:
        file.write(format_round2_report(results) + "\n")
    return report_path


def _arm_prompt_boxes(
    arm: str,
    case: ExampleCase,
) -> tuple[tuple[_Box, ...], tuple[_Box, ...]]:
    if arm.endswith("_single"):
        return (case.positive_xyxy[0],), ()
    return case.positive_xyxy, case.negative_xyxy


def _prompt_legend_entries(
    negatives: tuple[_Box, ...],
) -> list[tuple[str, sv.Color]]:
    entries = [("positive prompt", sv.Color.from_hex(DISPLAY_POSITIVE_HEX))]
    if negatives:
        entries.append(("negative prompt", sv.Color.from_hex(DISPLAY_NEGATIVE_HEX)))
    return entries


def _with_card_legend(
    image: Image.Image,
    entries: list[tuple[str, sv.Color]],
) -> Image.Image:
    bgr = np.asarray(image)[:, :, ::-1].copy()
    annotated = draw_image_legend(bgr, entries, scale=1.8)
    return Image.fromarray(annotated[:, :, ::-1])


def _annotate_box_groups(
    image: Image.Image,
    groups: list[tuple[tuple[_Box, ...], tuple[int, int, int]]],
) -> Image.Image:
    line_width = max(3, round(max(image.size) / 300))
    fill_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    fill_draw = ImageDraw.Draw(fill_layer)
    for boxes, color in groups:
        for box in boxes:
            fill_draw.rectangle(box, fill=(*color, _BOX_FILL_ALPHA))
    annotated = Image.alpha_composite(image.convert("RGBA"), fill_layer)
    annotated = annotated.convert("RGB")
    draw = ImageDraw.Draw(annotated)
    for boxes, color in groups:
        for box in boxes:
            draw.rectangle(box, outline=color, width=line_width)
    return annotated


def prompt_panel(arm: str, image: Image.Image, case: ExampleCase) -> Image.Image:
    """Visualization of the arm's prompt: positive boxes green, negatives
    red, drawn for every arm (for text arms this visualizes the coordinates
    given as text).

    Display-only palette; the images actually sent to the model use the
    colors named in the prompts (red positives, blue negatives).

    Args:
        arm: One of :data:`ROUND2_ARMS`.
        image: Original image in RGB mode.
        case: Example case for the sample.

    Returns:
        Annotated copy with a card-style legend.
    """
    positives, negatives = _arm_prompt_boxes(arm, case)
    annotated = _annotate_box_groups(
        image,
        [
            (negatives, _DISPLAY_NEGATIVE_RGB),
            (positives, _DISPLAY_POSITIVE_RGB),
        ],
    )
    return _with_card_legend(annotated, _prompt_legend_entries(negatives))


def overlay_panel(
    arm: str,
    image: Image.Image,
    case: ExampleCase,
    detections: sv.Detections,
) -> Image.Image:
    """Output visualization: predictions blue, positive prompts green,
    negative prompts red; ground truth is not rendered.

    Args:
        arm: One of :data:`ROUND2_ARMS`.
        image: Original image in RGB mode.
        case: Example case for the sample.
        detections: Parsed predictions.

    Returns:
        Annotated copy with a card-style legend.
    """
    positives, negatives = _arm_prompt_boxes(arm, case)
    predictions = tuple(tuple(float(value) for value in box) for box in detections.xyxy)
    annotated = _annotate_box_groups(
        image,
        [
            (predictions, _DISPLAY_PREDICTION_RGB),
            (negatives, _DISPLAY_NEGATIVE_RGB),
            (positives, _DISPLAY_POSITIVE_RGB),
        ],
    )
    entries = _prompt_legend_entries(negatives)
    entries.append(("prediction", sv.Color.from_hex(DISPLAY_PREDICTION_HEX)))
    return _with_card_legend(annotated, entries)


def render_round2_stack(
    *,
    arm: str,
    case: ExampleCase,
    sample: DetectionSample,
    detections: sv.Detections,
    output_path: Path,
    max_edge: int = 1600,
) -> None:
    """Render a vertical input/prediction stack for one record.

    Args:
        arm: One of :data:`ROUND2_ARMS`.
        case: Example case for the sample.
        sample: Detection sample whose image is rendered.
        detections: Parsed predictions.
        output_path: Destination PNG path.
        max_edge: Maximum panel edge in pixels.
    """
    image = load_case_image(sample)
    top = resize_image_to_max_edge(prompt_panel(arm, image, case), max_edge)
    bottom = resize_image_to_max_edge(
        overlay_panel(arm, image, case, detections), max_edge
    )
    width = max(top.size[0], bottom.size[0])
    height = top.size[1] + bottom.size[1] + _GAP
    combined = Image.new("RGB", (width, height), (20, 20, 20))
    combined.paste(top, ((width - top.size[0]) // 2, 0))
    combined.paste(bottom, ((width - bottom.size[0]) // 2, top.size[1] + _GAP))
    combined.save(output_path)


def render_round2_arm(
    *,
    arm: str,
    raw_directory: Path,
    cases_by_image: dict[str, ExampleCase],
    sample_index: dict[str, DetectionSample],
    renders_directory: Path,
) -> None:
    """Render stacks for every scored record of one arm.

    Args:
        arm: One of :data:`ROUND2_ARMS`.
        raw_directory: Directory holding the raw JSONL files.
        cases_by_image: Mapping of image basename to example case.
        sample_index: Mapping of image basename to detection sample.
        renders_directory: Root render output directory.
    """
    arm_directory = renders_directory / arm
    arm_directory.mkdir(parents=True, exist_ok=True)
    for record in load_arm_records(raw_directory, arm):
        case = cases_by_image.get(record["image"])
        sample = sample_index.get(record["image"])
        if case is None or sample is None or record.get("error") is not None:
            continue
        detections, _ = record_to_detections(record, sample)
        render_round2_stack(
            arm=arm,
            case=case,
            sample=sample,
            detections=detections,
            output_path=arm_directory / f"{Path(record['image']).stem}.png",
        )
