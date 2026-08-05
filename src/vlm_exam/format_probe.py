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

"""Probe the native detection output format of OpenAI models.

Sends deliberately format-free detection prompts to every model supported by
the ``open_ai`` workflow block, stores the raw responses, and infers which
output structure and coordinate convention each model produces when nothing
is specified.
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import openai
from PIL import Image

from vlm_exam.providers.base import (
    REQUEST_TIMEOUT_SECONDS,
    call_with_retries,
)
from vlm_exam.providers.image_upload import (
    OPENAI_MAX_EDGE_PIXELS,
    resize_image_to_max_edge,
    scale_dimensions_to_max_edge,
)
from vlm_exam.providers.openai import _png_data_url
from vlm_exam.tasks.detection import DetectionSample

PROMPT_VARIANTS: dict[str, str] = {
    "neutral": (
        "Detect all instances of the following classes in the image: "
        "{classes}. Return bounding boxes for every detection."
    ),
    "json": (
        "Detect all instances of the following classes in the image: "
        "{classes}. Return bounding boxes for every detection. "
        "Respond in JSON."
    ),
}

RATE_LIMIT_MAX_ATTEMPTS = 6
RATE_LIMIT_BASE_DELAY_SECONDS = 15.0

MIN_DECISIVE_COVERAGE = 0.2
"""Minimum GT-coverage IoU for an image to vote on the coordinate convention."""

FREE_SCALE_MARGIN = 0.05
"""How much a free-fitted scale must beat fixed scales to be reported."""


def load_openai_block_models() -> list[dict[str, Any]]:
    """Return model descriptors from the open_ai@v5 workflow block registry.

    Each entry carries the wire model id and the reasoning effort the probe
    should use (``"low"`` when supported, ``None`` for non-reasoning models).
    """
    from inference.core.workflows.core_steps.models.foundation.openai.v5 import (  # noqa: E501, PLC0415
        MODEL_REASONING_EFFORT_VALUES,
        OPENAI_MODELS,
    )

    descriptors = []
    for model in OPENAI_MODELS:
        effort_values = MODEL_REASONING_EFFORT_VALUES.get(model["id"], [])
        descriptors.append(
            {
                "id": model["id"],
                "name": model["name"],
                "reasoning_effort": "low" if "low" in effort_values else None,
            }
        )
    return descriptors


def build_probe_prompt(variant: str, classes: list[str]) -> str:
    """Render one probe prompt variant for an image's class list."""
    template = PROMPT_VARIANTS.get(variant)
    if template is None:
        raise ValueError(f"Unknown prompt variant: {variant!r}")
    return template.format(classes=", ".join(classes))


def _call_with_rate_limit_retries(operation, description: str):
    """Retry an OpenAI call on 429s, which the shared retry helper skips."""
    for attempt in range(1, RATE_LIMIT_MAX_ATTEMPTS + 1):
        try:
            return operation()
        except openai.RateLimitError:
            if attempt >= RATE_LIMIT_MAX_ATTEMPTS:
                raise
            delay = RATE_LIMIT_BASE_DELAY_SECONDS * attempt
            print(
                f"Rate limited on {description}; retrying in {delay:.0f}s "
                f"(attempt {attempt}/{RATE_LIMIT_MAX_ATTEMPTS})...",
                flush=True,
            )
            time.sleep(delay)
    raise AssertionError("unreachable")


def call_openai_detection(
    client: openai.OpenAI,
    *,
    model_id: str,
    image: Image.Image,
    prompt: str,
    reasoning_effort: str | None,
) -> dict[str, Any]:
    """Send one probe request and return raw output plus telemetry."""
    upload_image = resize_image_to_max_edge(image, OPENAI_MAX_EDGE_PIXELS)
    data_url = _png_data_url(upload_image)

    request: dict[str, Any] = {
        "model": model_id,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": data_url},
                    {"type": "input_text", "text": prompt},
                ],
            }
        ],
    }
    if reasoning_effort is not None:
        request["reasoning"] = {"effort": reasoning_effort}

    response, retry_stats = _call_with_rate_limit_retries(
        lambda: call_with_retries(lambda: client.responses.create(**request)),
        description=model_id,
    )
    usage = response.usage
    return {
        "raw_output": (response.output_text or "").strip(),
        "uploaded_width": upload_image.size[0],
        "uploaded_height": upload_image.size[1],
        "input_tokens": int(usage.input_tokens or 0) if usage else None,
        "output_tokens": int(usage.output_tokens or 0) if usage else None,
        "inference_seconds": retry_stats.inference_seconds,
        "attempts": retry_stats.attempts,
    }


def preflight_models(
    models: list[dict[str, Any]],
    api_key: str | None = None,
) -> dict[str, str | None]:
    """Send one tiny text request per model; return model id -> error or None."""
    client = openai.OpenAI(
        api_key=api_key, timeout=REQUEST_TIMEOUT_SECONDS, max_retries=0
    )
    results: dict[str, str | None] = {}
    for model in models:
        request: dict[str, Any] = {
            "model": model["id"],
            "input": "Reply with the single word OK.",
        }
        if model["reasoning_effort"] is not None:
            request["reasoning"] = {"effort": model["reasoning_effort"]}
        try:
            client.responses.create(**request)
            results[model["id"]] = None
        except Exception as error:
            results[model["id"]] = f"{type(error).__name__}: {error}"
    return results


def probe_record_key(record: dict[str, Any]) -> str:
    return str(record.get("image", ""))


def completed_images(jsonl_path: Path) -> set[str]:
    """Images already probed (successfully) in an existing raw JSONL file."""
    if not jsonl_path.exists():
        return set()
    done: set[str] = set()
    with open(jsonl_path) as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("error") is None:
                done.add(probe_record_key(record))
    return done


def run_probe_job(
    *,
    model: dict[str, Any],
    variant: str,
    samples: list[DetectionSample],
    classes_by_image: dict[str, list[str]],
    output_path: Path,
    api_key: str | None = None,
) -> int:
    """Probe one (model, variant) pair over all samples; append to JSONL.

    Returns the number of newly collected responses. Already-collected images
    are skipped so interrupted runs can resume.
    """
    job_key = f"{model['id']}/{variant}"
    done = completed_images(output_path)
    client = openai.OpenAI(
        api_key=api_key, timeout=REQUEST_TIMEOUT_SECONDS, max_retries=0
    )
    collected = 0
    with open(output_path, "a", buffering=1) as file:
        for index, sample in enumerate(samples, start=1):
            image_name = Path(sample.image_path).name
            if image_name in done:
                continue
            classes = classes_by_image[image_name]
            prompt = build_probe_prompt(variant, classes)
            record: dict[str, Any] = {
                "model": model["id"],
                "variant": variant,
                "reasoning_effort": model["reasoning_effort"],
                "image": image_name,
                "original_width": sample.image_width,
                "original_height": sample.image_height,
                "classes": classes,
                "prompt": prompt,
                "error": None,
            }
            try:
                image = Image.open(sample.image_path).convert("RGB")
                record.update(
                    call_openai_detection(
                        client,
                        model_id=model["id"],
                        image=image,
                        prompt=prompt,
                        reasoning_effort=model["reasoning_effort"],
                    )
                )
            except Exception as error:
                record["error"] = f"{type(error).__name__}: {error}"
            file.write(json.dumps(record) + "\n")
            collected += 1
            print(
                f"Completed probe {job_key} on {image_name} ({index}/{len(samples)})",
                flush=True,
            )
    return collected


def run_probe_collection(
    *,
    models: list[dict[str, Any]],
    variants: list[str],
    samples: list[DetectionSample],
    classes_by_image: dict[str, list[str]],
    output_directory: Path,
    max_workers: int = 6,
    api_key: str | None = None,
) -> None:
    """Collect raw probe responses for every (model, variant) pair."""
    raw_directory = output_directory / "raw"
    raw_directory.mkdir(parents=True, exist_ok=True)
    jobs = [(model, variant) for model in models for variant in variants]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                run_probe_job,
                model=model,
                variant=variant,
                samples=samples,
                classes_by_image=classes_by_image,
                output_path=raw_directory / f"{model['id']}__{variant}.jsonl",
                api_key=api_key,
            ): (model["id"], variant)
            for model, variant in jobs
        }
        for future, job_key in futures.items():
            future.result()
            print(f"Finished job {job_key[0]}/{job_key[1]}", flush=True)


# ---------------------------------------------------------------------------
# Analysis: JSON extraction, structural fingerprinting, coordinate inference
# ---------------------------------------------------------------------------

_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)

BOX_ARRAY_KEYS = (
    "box_2d",
    "bbox_2d",
    "bbox",
    "box",
    "bounding_box",
    "boundingbox",
    "bounding_box_2d",
    "coordinates",
    "coords",
    "rect",
    "region",
    "position",
)
FIELD_GROUPS: list[tuple[tuple[str, str, str, str], str]] = [
    (("x_min", "y_min", "x_max", "y_max"), "xyxy"),
    (("xmin", "ymin", "xmax", "ymax"), "xyxy"),
    (("x1", "y1", "x2", "y2"), "xyxy"),
    (("left", "top", "right", "bottom"), "xyxy"),
    (("x", "y", "width", "height"), "xywh"),
    (("x", "y", "w", "h"), "xywh"),
]
LABEL_KEYS = ("label", "class", "class_name", "name", "category", "object", "type")

ORDERS = ("xyxy", "yxyx", "xywh", "cxcywh")
SCALES = (
    "normalized_0_1",
    "normalized_0_100",
    "normalized_0_1000",
    "absolute_uploaded",
)


@dataclass
class ParsedDetections:
    """Boxes recovered from one response plus structural metadata."""

    boxes: list[tuple[float, float, float, float]] = field(default_factory=list)
    labels: list[str | None] = field(default_factory=list)
    order_hint: str | None = None
    fingerprint: str = "none"


def _try_json(text: str) -> Any | None:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


def extract_json_payload(raw_output: str) -> tuple[Any | None, str]:
    """Pull a JSON payload out of a raw response.

    Returns the parsed payload and a container label describing where the
    JSON lived: bare, fenced, embedded in prose, or absent.
    """
    text = raw_output.strip()
    if not text:
        return None, "empty"
    for match in _FENCE_PATTERN.finditer(text):
        parsed = _try_json(match.group(1).strip())
        if parsed is not None:
            return parsed, "fenced_json"
    parsed = _try_json(text)
    if parsed is not None:
        return parsed, "bare_json"
    for opener, closer in (("[", "]"), ("{", "}")):
        start = text.find(opener)
        while start != -1:
            depth = 0
            for index in range(start, len(text)):
                character = text[index]
                if character == opener:
                    depth += 1
                elif character == closer:
                    depth -= 1
                    if depth == 0:
                        parsed = _try_json(text[start : index + 1])
                        if parsed is not None:
                            return parsed, "embedded_json"
                        break
            start = text.find(opener, start + 1)
    # Some models emit Python-style literals, e.g. [(190, 250, 290, 345)].
    # That is meaningful "native format" signal, so classify it distinctly.
    python_like = _try_json(text.replace("(", "[").replace(")", "]"))
    if python_like is not None:
        return python_like, "python_literal"
    return None, "no_json"


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_box_array(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 4
        and all(_is_number(item) for item in value)
    )


def _label_from_item(item: dict[str, Any]) -> str | None:
    for key in LABEL_KEYS:
        value = item.get(key)
        if isinstance(value, str):
            return value
    return None


def _parse_dict_item(
    item: dict[str, Any],
) -> tuple[tuple[float, float, float, float], str | None, str, str | None] | None:
    """Extract (box, label, box_source, order_hint) from one detection dict."""
    lowered = {str(key).lower(): value for key, value in item.items()}
    for key in BOX_ARRAY_KEYS:
        value = lowered.get(key)
        if _is_box_array(value):
            box = tuple(float(item) for item in value)
            return box, _label_from_item(lowered), f"{key}:list4", None
    for fields, order in FIELD_GROUPS:
        if all(name in lowered and _is_number(lowered[name]) for name in fields):
            box = tuple(float(lowered[name]) for name in fields)
            return box, _label_from_item(lowered), "fields:" + ",".join(fields), order
    return None


def parse_detections(parsed: Any) -> ParsedDetections:
    """Recover detection boxes and a structural fingerprint from JSON."""
    if isinstance(parsed, dict):
        list_values = {
            key: value for key, value in parsed.items() if isinstance(value, list)
        }
        # Class-grouped dict: {"car": [[..4 numbers..], ...], ...}
        if list_values and all(
            all(_is_box_array(item) for item in value) and len(value) > 0
            for value in list_values.values()
        ):
            result = ParsedDetections(fingerprint="dict[class->list4]")
            for label, value in list_values.items():
                for box in value:
                    result.boxes.append(tuple(float(item) for item in box))
                    result.labels.append(str(label))
            return result
        if len(list_values) == 1:
            key, value = next(iter(list_values.items()))
            inner = parse_detections(value)
            inner.fingerprint = f"dict['{key}']->{inner.fingerprint}"
            return inner
        return ParsedDetections(fingerprint="dict:unrecognized")

    if isinstance(parsed, list):
        if all(_is_box_array(item) for item in parsed) and parsed:
            return ParsedDetections(
                boxes=[tuple(float(value) for value in item) for item in parsed],
                labels=[None] * len(parsed),
                fingerprint="list[list4]",
            )
        if all(isinstance(item, dict) for item in parsed) and parsed:
            result = ParsedDetections()
            sources: list[str] = []
            for item in parsed:
                extracted = _parse_dict_item(item)
                if extracted is None:
                    continue
                box, label, source, order_hint = extracted
                result.boxes.append(box)
                result.labels.append(label)
                sources.append(source)
                if order_hint is not None:
                    result.order_hint = order_hint
            if result.boxes:
                dominant = max(set(sources), key=sources.count)
                result.fingerprint = f"list[dict {dominant}]"
                return result
            return ParsedDetections(fingerprint="list[dict]:no_boxes")
        if not parsed:
            return ParsedDetections(fingerprint="empty_list")
        return ParsedDetections(fingerprint="list:unrecognized")

    return ParsedDetections(fingerprint="scalar:unrecognized")


def _boxes_to_xyxy(boxes: np.ndarray, order: str) -> np.ndarray:
    a, b, c, d = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    if order == "xyxy":
        return np.stack([a, b, c, d], axis=1)
    if order == "yxyx":
        return np.stack([b, a, d, c], axis=1)
    if order == "xywh":
        return np.stack([a, b, a + c, b + d], axis=1)
    if order == "cxcywh":
        return np.stack([a - c / 2, b - d / 2, a + c / 2, b + d / 2], axis=1)
    raise ValueError(f"Unknown box order: {order!r}")


def _scale_to_uploaded_pixels(
    xyxy: np.ndarray,
    scale: str,
    uploaded_wh: tuple[int, int],
) -> np.ndarray:
    width, height = uploaded_wh
    factors = {
        "normalized_0_1": (width, height),
        "normalized_0_100": (width / 100.0, height / 100.0),
        "normalized_0_1000": (width / 1000.0, height / 1000.0),
        "absolute_uploaded": (1.0, 1.0),
    }
    fx, fy = factors[scale]
    return xyxy * np.array([fx, fy, fx, fy])


def _coverage_score(pred_xyxy: np.ndarray, gt_xyxy: np.ndarray) -> float:
    """Mean over GT boxes of the best IoU any predicted box achieves."""
    if len(gt_xyxy) == 0 or len(pred_xyxy) == 0:
        return 0.0
    x1 = np.maximum(pred_xyxy[:, None, 0], gt_xyxy[None, :, 0])
    y1 = np.maximum(pred_xyxy[:, None, 1], gt_xyxy[None, :, 1])
    x2 = np.minimum(pred_xyxy[:, None, 2], gt_xyxy[None, :, 2])
    y2 = np.minimum(pred_xyxy[:, None, 3], gt_xyxy[None, :, 3])
    intersection = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    pred_area = (pred_xyxy[:, 2] - pred_xyxy[:, 0]) * (
        pred_xyxy[:, 3] - pred_xyxy[:, 1]
    )
    gt_area = (gt_xyxy[:, 2] - gt_xyxy[:, 0]) * (gt_xyxy[:, 3] - gt_xyxy[:, 1])
    union = pred_area[:, None] + gt_area[None, :] - intersection
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, intersection / union, 0.0)
    return float(iou.max(axis=0).mean())


@dataclass
class InterpretationResult:
    """Best coordinate interpretation found for one response."""

    order: str
    scale: str
    coverage: float
    runner_up_coverage: float
    free_scale: float | None = None
    free_scale_coverage: float | None = None


def infer_interpretation(
    boxes: list[tuple[float, float, float, float]],
    *,
    order_hint: str | None,
    uploaded_wh: tuple[int, int],
    original_wh: tuple[int, int],
    gt_xyxy: np.ndarray,
) -> InterpretationResult | None:
    """Score candidate (order, scale) interpretations against ground truth.

    All candidates are mapped from the uploaded image's coordinate space back
    onto the original image before scoring, mirroring how a parser would use
    the boxes. A free-fitted scalar scale is also tried to detect responses
    that are self-consistent at an unknown resolution (e.g. the provider's
    internal tiling grid).
    """
    if not boxes:
        return None
    raw = np.array(boxes, dtype=float)
    to_original = np.array(
        [
            original_wh[0] / uploaded_wh[0],
            original_wh[1] / uploaded_wh[1],
        ]
        * 2
    )
    orders = [order_hint] if order_hint else list(ORDERS)

    scored: list[tuple[float, str, str]] = []
    for order in orders:
        xyxy = _boxes_to_xyxy(raw, order)
        valid = (xyxy[:, 2] > xyxy[:, 0]) & (xyxy[:, 3] > xyxy[:, 1])
        if valid.mean() < 0.5:
            continue
        for scale in SCALES:
            pred = _scale_to_uploaded_pixels(xyxy[valid], scale, uploaded_wh)
            coverage = _coverage_score(pred * to_original, gt_xyxy)
            scored.append((coverage, order, scale))
    if not scored:
        return None
    scored.sort(reverse=True)
    best_coverage, best_order, best_scale = scored[0]
    runner_up = next(
        (
            coverage
            for coverage, order, scale in scored[1:]
            if scale != best_scale or order != best_order
        ),
        0.0,
    )
    result = InterpretationResult(
        order=best_order,
        scale=best_scale,
        coverage=best_coverage,
        runner_up_coverage=runner_up,
    )

    xyxy = _boxes_to_xyxy(raw, best_order)
    valid = (xyxy[:, 2] > xyxy[:, 0]) & (xyxy[:, 3] > xyxy[:, 1])
    best_free_scale, best_free_coverage = None, 0.0
    for factor in np.geomspace(0.05, 50.0, 121):
        pred = xyxy[valid] * factor
        coverage = _coverage_score(pred * to_original, gt_xyxy)
        if coverage > best_free_coverage:
            best_free_scale, best_free_coverage = float(factor), coverage
    if (
        best_free_scale is not None
        and best_free_coverage > best_coverage + FREE_SCALE_MARGIN
    ):
        result.free_scale = best_free_scale
        result.free_scale_coverage = best_free_coverage
    return result


@dataclass
class ImageAnalysis:
    """Per-response analysis outcome."""

    model: str
    variant: str
    image: str
    container: str
    fingerprint: str
    num_boxes: int
    order: str | None = None
    scale: str | None = None
    coverage: float | None = None
    margin: float | None = None
    free_scale: float | None = None
    free_scale_coverage: float | None = None
    error: str | None = None


def analyze_record(
    record: dict[str, Any],
    sample: DetectionSample,
) -> ImageAnalysis:
    """Analyze one raw probe record against its sample's ground truth."""
    analysis = ImageAnalysis(
        model=record["model"],
        variant=record["variant"],
        image=record["image"],
        container="error",
        fingerprint="error",
        num_boxes=0,
        error=record.get("error"),
    )
    if record.get("error") is not None:
        return analysis

    payload, container = extract_json_payload(record.get("raw_output", ""))
    analysis.container = container
    if payload is None:
        analysis.fingerprint = container
        return analysis

    parsed = parse_detections(payload)
    analysis.fingerprint = parsed.fingerprint
    analysis.num_boxes = len(parsed.boxes)
    if not parsed.boxes:
        return analysis

    uploaded_wh = (
        record.get("uploaded_width")
        or scale_dimensions_to_max_edge(
            sample.image_width, sample.image_height, OPENAI_MAX_EDGE_PIXELS
        )[0],
        record.get("uploaded_height")
        or scale_dimensions_to_max_edge(
            sample.image_width, sample.image_height, OPENAI_MAX_EDGE_PIXELS
        )[1],
    )
    interpretation = infer_interpretation(
        parsed.boxes,
        order_hint=parsed.order_hint,
        uploaded_wh=uploaded_wh,
        original_wh=(sample.image_width, sample.image_height),
        gt_xyxy=np.asarray(sample.ground_truth.xyxy, dtype=float),
    )
    if interpretation is None:
        return analysis
    analysis.order = interpretation.order
    analysis.scale = interpretation.scale
    analysis.coverage = interpretation.coverage
    analysis.margin = interpretation.coverage - interpretation.runner_up_coverage
    analysis.free_scale = interpretation.free_scale
    analysis.free_scale_coverage = interpretation.free_scale_coverage
    return analysis


def summarize_job(analyses: list[ImageAnalysis]) -> dict[str, Any]:
    """Aggregate per-image analyses for one (model, variant) pair."""
    total = len(analyses)
    errors = sum(1 for item in analyses if item.error is not None)
    with_boxes = [item for item in analyses if item.num_boxes > 0]
    containers = [item.container for item in analyses if item.error is None]
    fingerprints = [item.fingerprint for item in with_boxes]

    def _dominant(values: list[str]) -> tuple[str | None, float]:
        if not values:
            return None, 0.0
        winner = max(set(values), key=values.count)
        return winner, values.count(winner) / len(values)

    dominant_container, container_share = _dominant(containers)
    dominant_fingerprint, fingerprint_share = _dominant(fingerprints)

    decisive = [
        item
        for item in with_boxes
        if item.coverage is not None and item.coverage >= MIN_DECISIVE_COVERAGE
    ]
    conventions = [f"{item.order}/{item.scale}" for item in decisive]
    dominant_convention, convention_share = _dominant(conventions)

    coverages = [item.coverage for item in with_boxes if item.coverage is not None]
    free_scales = [item.free_scale for item in decisive if item.free_scale is not None]
    return {
        "responses": total,
        "errors": errors,
        "json_rate": (
            sum(1 for value in containers if value != "no_json" and value != "empty")
            / len(containers)
            if containers
            else 0.0
        ),
        "box_rate": len(with_boxes) / total if total else 0.0,
        "dominant_container": dominant_container,
        "container_share": container_share,
        "dominant_fingerprint": dominant_fingerprint,
        "fingerprint_share": fingerprint_share,
        "decisive_images": len(decisive),
        "dominant_convention": dominant_convention,
        "convention_share": convention_share,
        "mean_coverage": float(np.mean(coverages)) if coverages else None,
        "free_scale_images": len(free_scales),
        "median_free_scale": float(np.median(free_scales)) if free_scales else None,
    }


def run_analysis(
    *,
    raw_directory: Path,
    samples_by_image: dict[str, DetectionSample],
) -> tuple[dict[str, dict[str, Any]], list[ImageAnalysis]]:
    """Analyze every raw JSONL file; return per-job summaries and details."""
    summaries: dict[str, dict[str, Any]] = {}
    all_analyses: list[ImageAnalysis] = []
    for jsonl_path in sorted(raw_directory.glob("*.jsonl")):
        analyses: list[ImageAnalysis] = []
        with open(jsonl_path) as file:
            for line in file:
                if not line.strip():
                    continue
                record = json.loads(line)
                sample = samples_by_image.get(record["image"])
                if sample is None:
                    continue
                analyses.append(analyze_record(record, sample))
        if not analyses:
            continue
        job_key = jsonl_path.stem
        summaries[job_key] = summarize_job(analyses)
        all_analyses.extend(analyses)
    return summaries, all_analyses


def format_probe_report(summaries: dict[str, dict[str, Any]]) -> str:
    """Render the per-model format summary as a markdown table."""
    lines = [
        "| Model | Variant | JSON rate | Box rate | Dominant structure | "
        "Structure share | Convention (order/scale) | Convention share | "
        "Decisive n | Mean coverage | Free-scale n |",
        "|---|---|---:|---:|---|---:|---|---:|---:|---:|---:|",
    ]
    for job_key in sorted(summaries):
        summary = summaries[job_key]
        model, _, variant = job_key.rpartition("__")
        convention = summary["dominant_convention"] or "n/a"
        if summary["median_free_scale"] is not None:
            convention += f" (free~{summary['median_free_scale']:.2f})"
        mean_coverage = (
            f"{summary['mean_coverage']:.3f}"
            if summary["mean_coverage"] is not None
            else "n/a"
        )
        lines.append(
            f"| {model} | {variant} | {summary['json_rate'] * 100:.0f}% | "
            f"{summary['box_rate'] * 100:.0f}% | "
            f"{summary['dominant_fingerprint'] or 'n/a'} | "
            f"{summary['fingerprint_share'] * 100:.0f}% | {convention} | "
            f"{summary['convention_share'] * 100:.0f}% | "
            f"{summary['decisive_images']} | {mean_coverage} | "
            f"{summary['free_scale_images']} |"
        )
    return "\n".join(lines)


def write_analysis_artifacts(
    *,
    output_directory: Path,
    summaries: dict[str, dict[str, Any]],
    analyses: list[ImageAnalysis],
) -> Path:
    """Write summary JSON, per-image JSONL, and markdown report."""
    analysis_directory = output_directory / "analysis"
    analysis_directory.mkdir(parents=True, exist_ok=True)
    with open(analysis_directory / "format_summary.json", "w") as file:
        json.dump(summaries, file, indent=2)
    with open(analysis_directory / "image_analyses.jsonl", "w") as file:
        for item in analyses:
            file.write(json.dumps(asdict(item)) + "\n")
    report_path = analysis_directory / "report.md"
    with open(report_path, "w") as file:
        file.write(format_probe_report(summaries) + "\n")
    return report_path
