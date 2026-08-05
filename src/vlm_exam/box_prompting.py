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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import openai
import supervision as sv
from PIL import Image, ImageDraw, ImageOps

from vlm_exam.format_matrix import select_matrix_images
from vlm_exam.format_probe import completed_images, extract_json_payload
from vlm_exam.providers.base import (
    EMPTY_RESPONSE_TEXT,
    REQUEST_TIMEOUT_SECONDS,
    call_with_retries,
)
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

MODEL_KEY = "qwen-3.8-max"
"""vlm-exam model key of the model under test."""

PROVIDER_MODEL_ID = "qwen3.8-max"
"""DashScope model id called by this experiment."""

ARMS = ("text_box", "drawn_box", "crop")
"""The three reference-communication arms of the experiment."""

PREDICTION_LABEL = "object"
"""Class-agnostic label requested from and assigned to every prediction."""

REFERENCE_IOU_THRESHOLD = 0.5
"""Minimum IoU with the reference box to count a prediction as re-detecting it."""

_DASHSCOPE_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
_MAX_OUTPUT_TOKENS = 16384
_CROP_PADDING_FRACTION = 0.1

_OUTPUT_CLAUSE = (
    "Output a JSON list where each entry contains the 2D bounding box "
    'in the key "box_2d" and the text label in the key "label". '
    'The "box_2d" value must be [x_min, y_min, x_max, y_max]: the '
    "top-left and bottom-right corners as integers between 0 and 1000, "
    "normalized to the image width (x) and height (y). "
    f'Use the label "{PREDICTION_LABEL}" for every entry. '
    "Return only the JSON list, with no extra text."
)

_TEXT_BOX_LEAD = (
    "This image contains a reference object whose bounding box is "
    "[x_min, y_min, x_max, y_max] = [{x_min}, {y_min}, {x_max}, {y_max}], "
    "given as integers between 0 and 1000, normalized to the image "
    "width (x) and height (y). Detect all other objects in the image that "
    "are visually similar to the reference object. Do not include the "
    "reference object itself. "
)

_DRAWN_BOX_LEAD = (
    "The reference object in this image is marked with a red rectangle. "
    "Detect all other objects in the image that are visually similar to "
    "the marked reference object. Do not include the marked reference "
    "object itself, and ignore the red rectangle. "
)

_CROP_LEAD = (
    "The second image is a cropped view of a reference object taken from "
    "the first image. Detect all other objects in the first image that "
    "are visually similar to the reference object. Do not include the "
    "reference object itself. "
)


@dataclass(frozen=True)
class ReferenceCase:
    """One image with a chosen reference instance and its scoring targets.

    Attributes:
        image_name: Image file basename.
        class_name: Ground-truth class of the reference instance.
        reference_xyxy: Reference box in absolute pixels of the original
            image.
        target_xyxy: Boxes of the remaining instances of the same class,
            in absolute pixels; these are the scoring ground truth.
    """

    image_name: str
    class_name: str
    reference_xyxy: tuple[float, float, float, float]
    target_xyxy: tuple[tuple[float, float, float, float], ...]


def build_reference_case(sample: DetectionSample) -> ReferenceCase | None:
    """Pick a deterministic reference instance for one sample.

    Chooses the class with the most ground-truth instances (ties broken
    by the smaller class id) and uses its largest-area box as the
    reference; the remaining instances of that class become the targets.

    Args:
        sample: Detection sample with ground-truth boxes.

    Returns:
        The reference case, or ``None`` when no class has at least two
        instances.
    """
    class_ids = sample.ground_truth.class_id
    if class_ids is None or len(sample.ground_truth) < 2:
        return None
    unique_ids, counts = np.unique(class_ids, return_counts=True)
    best_count = counts.max()
    if best_count < 2:
        return None
    best_class = int(unique_ids[counts == best_count].min())
    boxes = sample.ground_truth.xyxy[class_ids == best_class]
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    reference_index = int(np.argmax(areas))
    reference = tuple(float(value) for value in boxes[reference_index])
    targets = tuple(
        tuple(float(value) for value in box)
        for index, box in enumerate(boxes)
        if index != reference_index
    )
    return ReferenceCase(
        image_name=Path(sample.image_path).name,
        class_name=sample.classes[best_class],
        reference_xyxy=reference,
        target_xyxy=targets,
    )


def select_cases(
    sample_index: dict[str, DetectionSample],
    *,
    count: int,
    seed: int = 42,
) -> list[ReferenceCase]:
    """Deterministically select usable reference cases.

    Walks the same shuffled image order as the format matrix and keeps
    the first ``count`` images that yield a valid reference case.

    Args:
        sample_index: Mapping of image basename to detection sample.
        count: Number of cases to select.
        seed: Shuffle seed shared with the format matrix.

    Returns:
        Selected reference cases in shuffle order.
    """
    order = select_matrix_images(sample_index, count=len(sample_index), seed=seed)
    cases: list[ReferenceCase] = []
    for image_name in order:
        case = build_reference_case(sample_index[image_name])
        if case is not None:
            cases.append(case)
        if len(cases) == count:
            break
    return cases


def normalized_reference_box(
    case: ReferenceCase,
    sample: DetectionSample,
) -> tuple[int, int, int, int]:
    """Reference box as integers on the 0-1000 normalized grid.

    Args:
        case: Reference case for the sample.
        sample: Detection sample providing the original image size.

    Returns:
        ``(x_min, y_min, x_max, y_max)`` normalized to 0-1000.
    """
    x_min, y_min, x_max, y_max = case.reference_xyxy
    width = sample.image_width
    height = sample.image_height
    return (
        max(0, min(1000, round(x_min / width * 1000))),
        max(0, min(1000, round(y_min / height * 1000))),
        max(0, min(1000, round(x_max / width * 1000))),
        max(0, min(1000, round(y_max / height * 1000))),
    )


def build_arm_prompt(arm: str, reference_box: tuple[int, int, int, int]) -> str:
    """Build the full prompt for one arm.

    Args:
        arm: One of :data:`ARMS`.
        reference_box: Reference box on the 0-1000 normalized grid; only
            embedded for the ``text_box`` arm.

    Returns:
        Prompt text requesting the benchmark output format.
    """
    if arm == "text_box":
        x_min, y_min, x_max, y_max = reference_box
        lead = _TEXT_BOX_LEAD.format(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)
    elif arm == "drawn_box":
        lead = _DRAWN_BOX_LEAD
    elif arm == "crop":
        lead = _CROP_LEAD
    else:
        raise ValueError(f"Unknown arm: {arm!r}")
    return lead + _OUTPUT_CLAUSE


def draw_reference_box(
    image: Image.Image,
    reference_xyxy: tuple[float, float, float, float],
) -> Image.Image:
    """Return a copy of the image with the reference box drawn in red.

    Args:
        image: Original image in RGB mode.
        reference_xyxy: Reference box in absolute pixels.

    Returns:
        Annotated copy of the image.
    """
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    line_width = max(3, round(max(image.size) / 300))
    draw.rectangle(reference_xyxy, outline=(255, 0, 0), width=line_width)
    return annotated


def crop_reference(
    image: Image.Image,
    reference_xyxy: tuple[float, float, float, float],
) -> Image.Image:
    """Crop the reference object with a small padding margin.

    Args:
        image: Original image in RGB mode.
        reference_xyxy: Reference box in absolute pixels.

    Returns:
        Cropped image around the reference object.
    """
    x_min, y_min, x_max, y_max = reference_xyxy
    padding_x = (x_max - x_min) * _CROP_PADDING_FRACTION
    padding_y = (y_max - y_min) * _CROP_PADDING_FRACTION
    width, height = image.size
    left = max(0, int(round(x_min - padding_x)))
    top = max(0, int(round(y_min - padding_y)))
    right = min(width, int(round(x_max + padding_x)))
    bottom = min(height, int(round(y_max + padding_y)))
    return image.crop((left, top, right, bottom))


def prepare_arm_images(
    arm: str,
    image: Image.Image,
    reference_xyxy: tuple[float, float, float, float],
) -> list[Image.Image]:
    """Prepare the image payload for one arm.

    Args:
        arm: One of :data:`ARMS`.
        image: Original image in RGB mode.
        reference_xyxy: Reference box in absolute pixels.

    Returns:
        Ordered list of images to send with the prompt.
    """
    if arm == "text_box":
        return [image]
    if arm == "drawn_box":
        return [draw_reference_box(image, reference_xyxy)]
    if arm == "crop":
        return [image, crop_reference(image, reference_xyxy)]
    raise ValueError(f"Unknown arm: {arm!r}")


def load_case_image(sample: DetectionSample) -> Image.Image:
    """Load a sample's image EXIF-transposed in RGB mode.

    Args:
        sample: Detection sample to load.

    Returns:
        The loaded image.
    """
    return ImageOps.exif_transpose(Image.open(sample.image_path)).convert("RGB")


def build_client(api_key: str | None = None) -> openai.OpenAI:
    """Create a DashScope OpenAI-compatible client.

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


def call_qwen(
    client: openai.OpenAI,
    *,
    images: list[Image.Image],
    prompt: str,
) -> dict[str, Any]:
    """Send one multi-image request to Qwen3.8-Max on DashScope.

    Args:
        client: DashScope OpenAI-compatible client.
        images: Images sent before the prompt, in order.
        prompt: Text prompt.

    Returns:
        Raw output plus upload and telemetry fields.
    """
    content: list[dict[str, Any]] = []
    uploaded_sizes: list[list[int]] = []
    for image in images:
        data_url, uploaded_size = jpeg_data_url_under_max_base64_bytes(
            image,
            OPENROUTER_MAX_BASE64_BYTES,
            quality=OPENROUTER_JPEG_QUALITY,
        )
        content.append({"type": "image_url", "image_url": {"url": data_url}})
        uploaded_sizes.append([uploaded_size[0], uploaded_size[1]])
    content.append({"type": "text", "text": prompt})

    response, retry_stats = call_with_retries(
        lambda: client.chat.completions.create(
            model=PROVIDER_MODEL_ID,
            max_tokens=_MAX_OUTPUT_TOKENS,
            messages=[{"role": "user", "content": content}],
            # Qwen3.8-Max mandates reasoning on its primary route; keep
            # thinking on here to match the benchmark run behavior.
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
        "uploaded_sizes": uploaded_sizes,
        "input_tokens": usage.prompt_tokens if usage else None,
        "output_tokens": usage.completion_tokens if usage else None,
        "inference_seconds": retry_stats.inference_seconds,
        "attempts": retry_stats.attempts,
    }


def run_arm_collection(
    *,
    arm: str,
    cases: list[ReferenceCase],
    sample_index: dict[str, DetectionSample],
    output_path: Path,
    api_key: str | None = None,
) -> None:
    """Collect one arm over all cases; append to a resumable JSONL file.

    Args:
        arm: One of :data:`ARMS`.
        cases: Reference cases to probe.
        sample_index: Mapping of image basename to detection sample.
        output_path: JSONL file receiving one record per case.
        api_key: Optional DashScope API key.
    """
    done = completed_images(output_path)
    client = build_client(api_key)
    with open(output_path, "a", buffering=1) as file:
        for index, case in enumerate(cases, start=1):
            if case.image_name in done:
                continue
            sample = sample_index[case.image_name]
            reference_box = normalized_reference_box(case, sample)
            prompt = build_arm_prompt(arm, reference_box)
            record: dict[str, Any] = {
                "model": PROVIDER_MODEL_ID,
                "arm": arm,
                "reasoning_effort": "low",
                "image": case.image_name,
                "original_width": sample.image_width,
                "original_height": sample.image_height,
                "class_name": case.class_name,
                "reference_xyxy": list(case.reference_xyxy),
                "reference_box_normalized": list(reference_box),
                "target_count": len(case.target_xyxy),
                "prompt": prompt,
                "error": None,
            }
            try:
                image = load_case_image(sample)
                images = prepare_arm_images(arm, image, case.reference_xyxy)
                record.update(call_qwen(client, images=images, prompt=prompt))
            except Exception as error:
                record["error"] = f"{type(error).__name__}: {error}"
            file.write(json.dumps(record) + "\n")
            print(
                f"Completed {arm} on {case.image_name} ({index}/{len(cases)})",
                flush=True,
            )


def run_collection(
    *,
    cases: list[ReferenceCase],
    sample_index: dict[str, DetectionSample],
    output_directory: Path,
    max_workers: int = 3,
    api_key: str | None = None,
) -> None:
    """Collect all three arms concurrently; resumable per arm file.

    Args:
        cases: Reference cases to probe.
        sample_index: Mapping of image basename to detection sample.
        output_directory: Experiment root; raw files land in ``raw/``.
        max_workers: Concurrent arm jobs.
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
                    "reference_xyxy": list(case.reference_xyxy),
                    "target_count": len(case.target_xyxy),
                }
                for case in cases
            ],
            file,
            indent=2,
        )
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                run_arm_collection,
                arm=arm,
                cases=cases,
                sample_index=sample_index,
                output_path=raw_directory / f"{PROVIDER_MODEL_ID}__{arm}.jsonl",
                api_key=api_key,
            )
            for arm in ARMS
        ]
        for future, arm in zip(futures, ARMS):
            future.result()
            print(f"Finished arm {arm}", flush=True)


def load_arm_records(raw_directory: Path, arm: str) -> list[dict[str, Any]]:
    """Load one arm's records, keeping one record per image.

    Retried images appear multiple times; the successful record wins,
    otherwise the latest one is kept.

    Args:
        raw_directory: Directory holding the raw JSONL files.
        arm: One of :data:`ARMS`.

    Returns:
        Deduplicated records in file order.
    """
    path = raw_directory / f"{PROVIDER_MODEL_ID}__{arm}.jsonl"
    if not path.exists():
        return []
    by_image: dict[str, dict[str, Any]] = {}
    with open(path) as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            existing = by_image.get(record["image"])
            if existing is None or existing.get("error") is not None:
                by_image[record["image"]] = record
    return list(by_image.values())


def record_to_detections(
    record: dict[str, Any],
    sample: DetectionSample,
) -> tuple[sv.Detections, bool]:
    """Parse one record's raw output into class-agnostic detections.

    All labels are collapsed to :data:`PREDICTION_LABEL` before parsing
    with the benchmark's ``xyxy_normalized_0_to_1000`` parser.

    Args:
        record: One raw collection record.
        sample: Detection sample for coordinate scaling.

    Returns:
        Tuple of parsed detections and a parse-failure flag.
    """
    payload, _ = extract_json_payload(record.get("raw_output", ""))
    entries: list[Any] | None = None
    if isinstance(payload, list):
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


def target_detections(case: ReferenceCase) -> sv.Detections:
    """Ground-truth detections for scoring: the non-reference instances.

    Args:
        case: Reference case.

    Returns:
        Single-class detections of the target boxes.
    """
    if not case.target_xyxy:
        return sv.Detections.empty()
    xyxy = np.array(case.target_xyxy, dtype=np.float32)
    return sv.Detections(xyxy=xyxy, class_id=np.zeros(len(xyxy), dtype=int))


def _iou_with_reference(
    reference: tuple[float, float, float, float],
    xyxy: np.ndarray,
) -> np.ndarray:
    reference_array = np.array(reference, dtype=float)
    left = np.maximum(xyxy[:, 0], reference_array[0])
    top = np.maximum(xyxy[:, 1], reference_array[1])
    right = np.minimum(xyxy[:, 2], reference_array[2])
    bottom = np.minimum(xyxy[:, 3], reference_array[3])
    intersection = np.clip(right - left, 0, None) * np.clip(bottom - top, 0, None)
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    reference_area = (reference_array[2] - reference_array[0]) * (
        reference_array[3] - reference_array[1]
    )
    union = areas + reference_area - intersection
    return np.where(union > 0, intersection / union, 0.0)


def score_arm(
    records: list[dict[str, Any]],
    cases_by_image: dict[str, ReferenceCase],
    sample_index: dict[str, DetectionSample],
) -> dict[str, Any]:
    """Score one arm's records against the target ground truth.

    Args:
        records: Deduplicated records of one arm.
        cases_by_image: Mapping of image basename to reference case.
        sample_index: Mapping of image basename to detection sample.

    Returns:
        Dict with aggregate ``metrics`` and per-image ``images`` details.
    """
    per_image: dict[str, dict[str, Any]] = {}
    map50_values: list[float] = []
    parse_failures = 0
    errors = 0
    reference_redetections = 0
    predicted_counts: list[int] = []
    tokens_in: list[int] = []
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
        redetected = False
        if len(detections) > 0:
            ious = _iou_with_reference(case.reference_xyxy, detections.xyxy)
            redetected = bool((ious >= REFERENCE_IOU_THRESHOLD).any())
        reference_redetections += redetected
        map50_values.append(map50)
        predicted_counts.append(len(detections))
        detail.update(
            {
                "map50": map50,
                "predicted_boxes": len(detections),
                "parse_failed": failed,
                "reference_redetected": redetected,
            }
        )
        per_image[record["image"]] = detail
        if record.get("input_tokens") is not None:
            tokens_in.append(record["input_tokens"])
        if record.get("output_tokens") is not None:
            tokens_out.append(record["output_tokens"])
        if record.get("inference_seconds") is not None:
            seconds.append(record["inference_seconds"])
    target_counts = [detail["target_boxes"] for detail in per_image.values()]
    metrics = {
        "images": len(per_image),
        "mean_image_map50": float(np.mean(map50_values)) if map50_values else 0.0,
        "parse_failures": parse_failures,
        "errors": errors,
        "reference_redetections": reference_redetections,
        "mean_predicted_boxes": (
            float(np.mean(predicted_counts)) if predicted_counts else 0.0
        ),
        "mean_target_boxes": float(np.mean(target_counts)) if target_counts else 0.0,
        "avg_input_tokens": float(np.mean(tokens_in)) if tokens_in else None,
        "avg_output_tokens": float(np.mean(tokens_out)) if tokens_out else None,
        "avg_seconds": float(np.mean(seconds)) if seconds else None,
    }
    return {"metrics": metrics, "images": per_image}


def run_analysis(
    *,
    raw_directory: Path,
    cases_by_image: dict[str, ReferenceCase],
    sample_index: dict[str, DetectionSample],
) -> dict[str, dict[str, Any]]:
    """Score every collected arm.

    Args:
        raw_directory: Directory holding the raw JSONL files.
        cases_by_image: Mapping of image basename to reference case.
        sample_index: Mapping of image basename to detection sample.

    Returns:
        Mapping of arm to its scored results.
    """
    results: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        records = load_arm_records(raw_directory, arm)
        if records:
            results[arm] = score_arm(records, cases_by_image, sample_index)
    return results


def format_report(results: dict[str, dict[str, Any]]) -> str:
    """Render the experiment summary as markdown.

    Args:
        results: Per-arm scored results from :func:`run_analysis`.

    Returns:
        Markdown report with an arm summary and a per-image breakdown.
    """
    lines = ["# Qwen3.8-Max box-prompting experiment", ""]
    lines.append(
        "| Arm | Mean img mAP@50 | Parse fail | Errors | Ref re-detected | "
        "Avg pred boxes | Avg target boxes | Avg out tok | Avg s |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    ranked = sorted(
        results.items(),
        key=lambda item: item[1]["metrics"]["mean_image_map50"],
        reverse=True,
    )
    for arm, scored in ranked:
        metrics = scored["metrics"]
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
            f"| {arm} | {metrics['mean_image_map50'] * 100:.1f}% | "
            f"{metrics['parse_failures']} | {metrics['errors']} | "
            f"{metrics['reference_redetections']} | "
            f"{metrics['mean_predicted_boxes']:.1f} | "
            f"{metrics['mean_target_boxes']:.1f} | "
            f"{out_tokens} | {avg_seconds} |"
        )
    lines.append("")
    image_names = sorted(
        {name for scored in results.values() for name in scored["images"]}
    )
    arms_present = [arm for arm in ARMS if arm in results]
    header = "| Image | Class | Targets | " + " | ".join(arms_present) + " |"
    lines.append("## Per-image mAP@50")
    lines.append("")
    lines.append(header)
    lines.append("|---|---|---:|" + "---:|" * len(arms_present))
    for image_name in image_names:
        cells: list[str] = []
        class_name = ""
        target_boxes = ""
        for arm in arms_present:
            detail = results[arm]["images"].get(image_name)
            if detail is None:
                cells.append("n/a")
                continue
            class_name = detail["class_name"]
            target_boxes = str(detail["target_boxes"])
            if detail.get("error") is not None:
                cells.append("err")
            else:
                cells.append(f"{detail['map50'] * 100:.0f}%")
        lines.append(
            f"| {image_name} | {class_name} | {target_boxes} | "
            + " | ".join(cells)
            + " |"
        )
    return "\n".join(lines)


def write_artifacts(
    *,
    output_directory: Path,
    results: dict[str, dict[str, Any]],
) -> Path:
    """Write the summary JSON and markdown report.

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
        file.write(format_report(results) + "\n")
    return report_path


def render_overlay(
    *,
    sample: DetectionSample,
    case: ReferenceCase,
    detections: sv.Detections,
    output_path: Path,
) -> None:
    """Render a side-by-side ground-truth versus prediction overlay.

    The left panel shows the reference box (red) and target boxes
    (green); the right panel shows the reference box (red) and predicted
    boxes (blue).

    Args:
        sample: Detection sample whose image is rendered.
        case: Reference case for the sample.
        detections: Parsed predictions for the sample.
        output_path: Destination PNG path.
    """
    image = load_case_image(sample)
    line_width = max(2, round(max(image.size) / 400))

    ground_truth_panel = image.copy()
    draw = ImageDraw.Draw(ground_truth_panel)
    for box in case.target_xyxy:
        draw.rectangle(box, outline=(0, 200, 83), width=line_width)
    draw.rectangle(case.reference_xyxy, outline=(255, 0, 0), width=line_width)
    draw.text((10, 10), "ground truth (targets)", fill=(255, 255, 255))

    prediction_panel = image.copy()
    draw = ImageDraw.Draw(prediction_panel)
    for box in detections.xyxy:
        predicted_box = tuple(float(value) for value in box)
        draw.rectangle(predicted_box, outline=(41, 98, 255), width=line_width)
    draw.rectangle(case.reference_xyxy, outline=(255, 0, 0), width=line_width)
    draw.text((10, 10), "prediction", fill=(255, 255, 255))

    gap = 10
    combined = Image.new(
        "RGB",
        (image.size[0] * 2 + gap, image.size[1]),
        (30, 30, 30),
    )
    combined.paste(ground_truth_panel, (0, 0))
    combined.paste(prediction_panel, (image.size[0] + gap, 0))
    combined.save(output_path)


def render_arm_overlays(
    *,
    arm: str,
    raw_directory: Path,
    cases_by_image: dict[str, ReferenceCase],
    sample_index: dict[str, DetectionSample],
    renders_directory: Path,
) -> None:
    """Render overlays for every scored record of one arm.

    Args:
        arm: One of :data:`ARMS`.
        raw_directory: Directory holding the raw JSONL files.
        cases_by_image: Mapping of image basename to reference case.
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
        output_path = arm_directory / f"{Path(record['image']).stem}.png"
        render_overlay(
            sample=sample,
            case=case,
            detections=detections,
            output_path=output_path,
        )
