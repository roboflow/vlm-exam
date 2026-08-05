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

"""Prompt format x coordinate system x preprocessing experiment matrix.

Runs a fixed set of detection "arms" (each a complete prompt + coordinate
contract + upload preprocessing recipe) against OpenAI models over a fixed
image subset, then scores every arm against ground truth to determine the
optimal contract per model.
"""

from __future__ import annotations

import base64
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import openai
import supervision as sv
from PIL import Image

from vlm_exam.format_probe import (
    completed_images,
    extract_json_payload,
    parse_detections,
)
from vlm_exam.providers.base import REQUEST_TIMEOUT_SECONDS, call_with_retries
from vlm_exam.providers.image_upload import resize_image_to_max_edge
from vlm_exam.tasks.detection import DetectionSample
from vlm_exam.workflows_comparison import (
    compute_dataset_map_from_pairs,
    compute_map50,
)

DETECTION_MAX_EDGE_PIXELS = 2048
JPEG_QUALITY = 95

MATRIX_MODELS = (
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
    "gpt-5.5",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.4-nano",
    "gpt-5.2",
    "gpt-5.1",
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-4o",
    "gpt-4o-mini",
)

# Models with empty reasoning_effort_values in the open_ai@v5 block registry;
# the Responses API rejects the reasoning parameter for them.
NON_REASONING_MODELS = frozenset(
    ("gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o", "gpt-4o-mini")
)

RATE_LIMIT_MAX_ATTEMPTS = 6
RATE_LIMIT_BASE_DELAY_SECONDS = 15.0


@dataclass(frozen=True)
class ArmSpec:
    """One complete detection contract to be tested.

    Attributes:
        arm_id: Short unique identifier used in file names and reports.
        description: One-line human summary.
        preprocessing: Upload recipe - ``png2048`` (max-edge 2048 lossless
            PNG), ``png_noresize`` (original size PNG), or ``jpeg_noresize``
            (original size JPEG, quality 95).
        detail: Value for the image ``detail`` field, or ``None`` to omit.
        coordinate_space: How returned numbers map to pixels -
            ``absolute_sent`` (pixels of the uploaded image), ``norm999``
            (integers on a 0..999 grid), ``norm01`` (floats 0-1), or
            ``norm1000`` (0-1000 range).
        box_order: ``xyxy`` or ``yxyx``.
        structured: Whether to enforce the contract with structured outputs.
        dims_in_prompt: Whether the prompt states the uploaded image size.
        prompt_style: ``v5`` (single user message) or ``v4``
            (instructions + class list user message).
        box_key: JSON key requested for the box array (v5-style arms).
    """

    arm_id: str
    description: str
    preprocessing: str
    detail: str | None
    coordinate_space: str
    box_order: str
    structured: bool
    dims_in_prompt: bool
    prompt_style: str
    box_key: str


ARMS: tuple[ArmSpec, ...] = (
    ArmSpec(
        arm_id="A1-abs-box2d-dims-2048",
        description="v5 replica: absolute xyxy, box_2d, dims stated, 2048 PNG",
        preprocessing="png2048",
        detail=None,
        coordinate_space="absolute_sent",
        box_order="xyxy",
        structured=False,
        dims_in_prompt=True,
        prompt_style="v5",
        box_key="box_2d",
    ),
    ArmSpec(
        arm_id="A2-abs-bbox-dims-2048",
        description="A1 with the native 'bbox' key",
        preprocessing="png2048",
        detail=None,
        coordinate_space="absolute_sent",
        box_order="xyxy",
        structured=False,
        dims_in_prompt=True,
        prompt_style="v5",
        box_key="bbox",
    ),
    ArmSpec(
        arm_id="A3-abs-box2d-nodims-2048",
        description="A1 without stating image dimensions",
        preprocessing="png2048",
        detail=None,
        coordinate_space="absolute_sent",
        box_order="xyxy",
        structured=False,
        dims_in_prompt=False,
        prompt_style="v5",
        box_key="box_2d",
    ),
    ArmSpec(
        arm_id="A4-norm999-bbox-2048",
        description="Cookbook contract: xyxy integers on a 0..999 grid",
        preprocessing="png2048",
        detail=None,
        coordinate_space="norm999",
        box_order="xyxy",
        structured=False,
        dims_in_prompt=False,
        prompt_style="v5",
        box_key="bbox",
    ),
    ArmSpec(
        arm_id="A5-norm01-dict-2048",
        description="v4 prompt (normalized 0-1 dict) with v5 preprocessing",
        preprocessing="png2048",
        detail=None,
        coordinate_space="norm01",
        box_order="xyxy",
        structured=False,
        dims_in_prompt=False,
        prompt_style="v4",
        box_key="fields",
    ),
    ArmSpec(
        arm_id="A6-norm01-dict-jpeg",
        description="v4 replica: normalized 0-1 dict, unresized JPEG, detail auto",
        preprocessing="jpeg_noresize",
        detail="auto",
        coordinate_space="norm01",
        box_order="xyxy",
        structured=False,
        dims_in_prompt=False,
        prompt_style="v4",
        box_key="fields",
    ),
    ArmSpec(
        arm_id="A7-abs-box2d-dims-original",
        description="No client resize, detail=original, absolute xyxy",
        preprocessing="png_noresize",
        detail="original",
        coordinate_space="absolute_sent",
        box_order="xyxy",
        structured=False,
        dims_in_prompt=True,
        prompt_style="v5",
        box_key="box_2d",
    ),
    ArmSpec(
        arm_id="A8-abs-structured-2048",
        description="A1 contract enforced via structured outputs",
        preprocessing="png2048",
        detail=None,
        coordinate_space="absolute_sent",
        box_order="xyxy",
        structured=True,
        dims_in_prompt=True,
        prompt_style="v5",
        box_key="box_2d",
    ),
    ArmSpec(
        arm_id="A9-norm999-structured-2048",
        description="Cookbook 0..999 contract enforced via structured outputs",
        preprocessing="png2048",
        detail=None,
        coordinate_space="norm999",
        box_order="xyxy",
        structured=True,
        dims_in_prompt=False,
        prompt_style="v5",
        box_key="bbox",
    ),
    ArmSpec(
        arm_id="A10-yxyx1000-2048",
        description="Gemini-style yxyx normalized 0-1000",
        preprocessing="png2048",
        detail=None,
        coordinate_space="norm1000",
        box_order="yxyx",
        structured=False,
        dims_in_prompt=False,
        prompt_style="v5",
        box_key="box_2d",
    ),
)

ARMS_BY_ID = {arm.arm_id: arm for arm in ARMS}
CONTROL_ARM_ID = "A1-abs-box2d-dims-2048"


def _coordinate_clause(arm: ArmSpec, width: int, height: int) -> str:
    box_names = (
        "[y_min, x_min, y_max, x_max]"
        if arm.box_order == "yxyx"
        else "[x_min, y_min, x_max, y_max]"
    )
    if arm.coordinate_space == "absolute_sent":
        if arm.dims_in_prompt:
            return (
                f'The "{arm.box_key}" value must be {box_names}: the '
                "top-left and bottom-right corners in absolute pixel "
                f"coordinates of the {width}x{height} pixel image. "
            )
        return (
            f'The "{arm.box_key}" value must be {box_names}: the '
            "top-left and bottom-right corners in absolute pixel "
            "coordinates of the image. "
        )
    if arm.coordinate_space == "norm999":
        return (
            f'The "{arm.box_key}" value must be {box_names}: integers on a '
            "fixed 0..999 grid with the origin at the top-left corner of "
            "the image. "
        )
    if arm.coordinate_space == "norm1000":
        return f'The "{arm.box_key}" value must be {box_names} normalized to 0-1000. '
    raise ValueError(f"No v5-style clause for {arm.coordinate_space!r}")


V4_INSTRUCTIONS = (
    "You act as object-detection model. You must provide reasonable predictions. "
    "You are only allowed to produce JSON document. "
    'Expected structure of json: {"detections": [{"x_min": 0.1, "y_min": 0.2, '
    '"x_max": 0.3, "y_max": 0.4, "class_name": "my-class-X", "confidence": 0.7}]}. '
    "`my-class-X` must be one of the class names defined by user. All coordinates "
    "must be in range 0.0-1.0, representing percentage of image dimensions. "
    "`confidence` is a value in range 0.0-1.0 representing your confidence in "
    "prediction. You should detect all instances of classes provided by user."
)


def build_arm_prompt(
    arm: ArmSpec,
    *,
    classes: list[str],
    sent_width: int,
    sent_height: int,
) -> tuple[str | None, str]:
    """Return (instructions, user_text) for one arm and image."""
    class_list = ", ".join(classes)
    if arm.prompt_style == "v4":
        return (
            V4_INSTRUCTIONS,
            f"List of all classes to be recognised by model: {class_list}",
        )
    clause = _coordinate_clause(arm, sent_width, sent_height)
    if arm.structured:
        text = (
            "Detect all objects in this image. "
            'Output JSON with the key "detections" holding a list where each '
            f'entry contains the 2D bounding box in the key "{arm.box_key}" '
            'and the text label in the key "label". '
            f"{clause}"
            f"Only use these labels: {class_list}"
        )
        return None, text
    text = (
        "Detect all objects in this image. "
        "Output a JSON list where each entry contains the 2D bounding box "
        f'in the key "{arm.box_key}" and the text label in the key "label". '
        f"{clause}"
        "Return only the JSON list, with no extra text. "
        f"Only use these labels: {class_list}"
    )
    return None, text


def build_structured_output_format(arm: ArmSpec) -> dict[str, Any]:
    """JSON schema (Responses API text.format) enforcing an arm's contract."""
    box_items: dict[str, Any] = {"type": "integer"}
    if arm.coordinate_space == "norm999":
        box_items = {"type": "integer", "minimum": 0, "maximum": 999}
    schema = {
        "type": "object",
        "properties": {
            "detections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string"},
                        arm.box_key: {
                            "type": "array",
                            "items": box_items,
                            "minItems": 4,
                            "maxItems": 4,
                        },
                    },
                    "required": ["label", arm.box_key],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["detections"],
        "additionalProperties": False,
    }
    return {
        "format": {
            "type": "json_schema",
            "name": "detections",
            "schema": schema,
            "strict": True,
        }
    }


def _image_data_url(image: Image.Image, *, encoding: str) -> str:
    buffer = io.BytesIO()
    if encoding == "png":
        image.save(buffer, format="PNG")
        mime = "image/png"
    else:
        image.save(buffer, format="JPEG", quality=JPEG_QUALITY)
        mime = "image/jpeg"
    payload = base64.standard_b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:{mime};base64,{payload}"


def prepare_upload(
    arm: ArmSpec,
    image: Image.Image,
) -> tuple[str, int, int]:
    """Apply an arm's preprocessing; return (data_url, sent_w, sent_h)."""
    if arm.preprocessing == "png2048":
        sent = resize_image_to_max_edge(image, DETECTION_MAX_EDGE_PIXELS)
        return _image_data_url(sent, encoding="png"), sent.size[0], sent.size[1]
    if arm.preprocessing == "png_noresize":
        return _image_data_url(image, encoding="png"), image.size[0], image.size[1]
    if arm.preprocessing == "jpeg_noresize":
        return _image_data_url(image, encoding="jpeg"), image.size[0], image.size[1]
    raise ValueError(f"Unknown preprocessing: {arm.preprocessing!r}")


def _call_with_rate_limit_retries(operation, description: str):
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


def run_arm_request(
    client: openai.OpenAI,
    *,
    arm: ArmSpec,
    model_id: str,
    image: Image.Image,
    classes: list[str],
) -> dict[str, Any]:
    """Send one (arm, model, image) request; return raw output + telemetry."""
    data_url, sent_width, sent_height = prepare_upload(arm, image)
    instructions, user_text = build_arm_prompt(
        arm,
        classes=classes,
        sent_width=sent_width,
        sent_height=sent_height,
    )
    image_content: dict[str, Any] = {"type": "input_image", "image_url": data_url}
    if arm.detail is not None:
        image_content["detail"] = arm.detail
    request: dict[str, Any] = {
        "model": model_id,
        "input": [
            {
                "role": "user",
                "content": [
                    image_content,
                    {"type": "input_text", "text": user_text},
                ],
            }
        ],
    }
    if model_id not in NON_REASONING_MODELS:
        request["reasoning"] = {"effort": "low"}
    if instructions is not None:
        request["instructions"] = instructions
    if arm.structured:
        request["text"] = build_structured_output_format(arm)

    response, retry_stats = _call_with_rate_limit_retries(
        lambda: call_with_retries(lambda: client.responses.create(**request)),
        description=f"{model_id}/{arm.arm_id}",
    )
    usage = response.usage
    return {
        "raw_output": (response.output_text or "").strip(),
        "sent_width": sent_width,
        "sent_height": sent_height,
        "prompt": user_text,
        "instructions": instructions,
        "input_tokens": int(usage.input_tokens or 0) if usage else None,
        "output_tokens": int(usage.output_tokens or 0) if usage else None,
        "inference_seconds": retry_stats.inference_seconds,
        "attempts": retry_stats.attempts,
    }


def select_matrix_images(
    sample_index: dict[str, DetectionSample],
    *,
    count: int,
    seed: int = 42,
) -> list[str]:
    """Fixed random image subset; smaller counts are prefixes of larger ones."""
    names = sorted(sample_index)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(names))
    return [names[index] for index in order[:count]]


def run_matrix_job(
    *,
    model_id: str,
    arm: ArmSpec,
    samples: list[DetectionSample],
    classes_by_image: dict[str, list[str]],
    output_path: Path,
    api_key: str | None = None,
) -> None:
    """Collect one (model, arm) pair over all samples; append to JSONL."""
    done = completed_images(output_path)
    client = openai.OpenAI(
        api_key=api_key, timeout=REQUEST_TIMEOUT_SECONDS, max_retries=0
    )
    job_key = f"{model_id}/{arm.arm_id}"
    with open(output_path, "a", buffering=1) as file:
        for index, sample in enumerate(samples, start=1):
            image_name = Path(sample.image_path).name
            if image_name in done:
                continue
            record: dict[str, Any] = {
                "model": model_id,
                "arm": arm.arm_id,
                "image": image_name,
                "original_width": sample.image_width,
                "original_height": sample.image_height,
                "classes": classes_by_image[image_name],
                "error": None,
            }
            try:
                image = Image.open(sample.image_path).convert("RGB")
                record.update(
                    run_arm_request(
                        client,
                        arm=arm,
                        model_id=model_id,
                        image=image,
                        classes=classes_by_image[image_name],
                    )
                )
            except Exception as error:
                record["error"] = f"{type(error).__name__}: {error}"
            file.write(json.dumps(record) + "\n")
            print(
                f"Completed matrix {job_key} on {image_name} ({index}/{len(samples)})",
                flush=True,
            )


def run_matrix_collection(
    *,
    model_ids: list[str],
    arms: list[ArmSpec],
    samples: list[DetectionSample],
    classes_by_image: dict[str, list[str]],
    output_directory: Path,
    max_workers: int = 10,
    api_key: str | None = None,
) -> None:
    """Collect all (model, arm) jobs concurrently; resumable per job file."""
    raw_directory = output_directory / "raw"
    raw_directory.mkdir(parents=True, exist_ok=True)
    with open(output_directory / "image_subset.json", "w") as file:
        json.dump(
            [Path(sample.image_path).name for sample in samples],
            file,
            indent=2,
        )
    jobs = [(model_id, arm) for model_id in model_ids for arm in arms]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                run_matrix_job,
                model_id=model_id,
                arm=arm,
                samples=samples,
                classes_by_image=classes_by_image,
                output_path=raw_directory / f"{model_id}__{arm.arm_id}.jsonl",
                api_key=api_key,
            )
            for model_id, arm in jobs
        ]
        for future, (model_id, arm) in zip(futures, jobs):
            future.result()
            print(f"Finished matrix job {model_id}/{arm.arm_id}", flush=True)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def decode_boxes_to_original_xyxy(
    arm: ArmSpec,
    boxes: list[tuple[float, float, float, float]],
    *,
    sent_wh: tuple[int, int],
    original_wh: tuple[int, int],
) -> np.ndarray:
    """Map raw box numbers to xyxy pixels of the original image."""
    raw = np.array(boxes, dtype=float).reshape(-1, 4)
    if arm.box_order == "yxyx":
        raw = raw[:, [1, 0, 3, 2]]
    original_width, original_height = original_wh
    if arm.coordinate_space == "absolute_sent":
        sent_width, sent_height = sent_wh
        factors = np.array(
            [
                original_width / sent_width,
                original_height / sent_height,
            ]
            * 2
        )
        xyxy = raw * factors
    elif arm.coordinate_space == "norm999":
        factors = np.array(
            [
                (original_width - 1) / 999.0,
                (original_height - 1) / 999.0,
            ]
            * 2
        )
        xyxy = raw * factors
    elif arm.coordinate_space == "norm01":
        xyxy = raw * np.array([original_width, original_height] * 2)
    elif arm.coordinate_space == "norm1000":
        xyxy = raw / 1000.0 * np.array([original_width, original_height] * 2)
    else:
        raise ValueError(f"Unknown coordinate space: {arm.coordinate_space!r}")
    xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, original_width)
    xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, original_height)
    return xyxy


def record_to_detections(
    record: dict[str, Any],
    sample: DetectionSample,
) -> tuple[sv.Detections, bool]:
    """Parse one raw record into detections; returns (detections, failed)."""
    arm = ARMS_BY_ID[record["arm"]]
    payload, _ = extract_json_payload(record.get("raw_output", ""))
    if payload is None:
        return sv.Detections.empty(), True
    parsed = parse_detections(payload)
    if not parsed.boxes:
        # An explicitly empty detections list is a valid "nothing found".
        empty_ok = parsed.fingerprint in ("empty_list",) or (
            isinstance(payload, dict) and payload.get("detections") == []
        )
        return sv.Detections.empty(), not empty_ok
    xyxy = decode_boxes_to_original_xyxy(
        arm,
        parsed.boxes,
        sent_wh=(
            record.get("sent_width") or sample.image_width,
            record.get("sent_height") or sample.image_height,
        ),
        original_wh=(sample.image_width, sample.image_height),
    )
    valid = (xyxy[:, 2] > xyxy[:, 0]) & (xyxy[:, 3] > xyxy[:, 1])
    xyxy = xyxy[valid]
    labels = [label for label, keep in zip(parsed.labels, valid) if keep]
    if len(xyxy) == 0:
        return sv.Detections.empty(), True
    taxonomy = {name.lower(): idx for idx, name in enumerate(sample.classes)}
    class_id = np.array(
        [taxonomy.get(str(label).lower(), -1) for label in labels],
        dtype=int,
    )
    class_name = np.array([str(label) for label in labels])
    return (
        sv.Detections(
            xyxy=xyxy.round(0),
            confidence=np.ones(len(xyxy)),
            class_id=class_id,
            data={"class_name": class_name},
        ),
        False,
    )


def score_matrix_job(
    records: list[dict[str, Any]],
    samples_by_image: dict[str, DetectionSample],
) -> dict[str, Any]:
    """Score all records of one (model, arm) job against ground truth."""
    pairs: list[tuple[sv.Detections, sv.Detections]] = []
    per_image_map50: list[float] = []
    parse_failures = 0
    errors = 0
    tokens_in: list[int] = []
    tokens_out: list[int] = []
    seconds: list[float] = []
    for record in records:
        sample = samples_by_image.get(record["image"])
        if sample is None:
            continue
        if record.get("error") is not None:
            errors += 1
            pairs.append((sv.Detections.empty(), sample.ground_truth))
            per_image_map50.append(0.0)
            continue
        detections, failed = record_to_detections(record, sample)
        parse_failures += failed
        pairs.append((detections, sample.ground_truth))
        per_image_map50.append(compute_map50(detections, sample.ground_truth))
        if record.get("input_tokens") is not None:
            tokens_in.append(record["input_tokens"])
        if record.get("output_tokens") is not None:
            tokens_out.append(record["output_tokens"])
        if record.get("inference_seconds") is not None:
            seconds.append(record["inference_seconds"])
    map50, map75, map50_95 = compute_dataset_map_from_pairs(
        [pair[0] for pair in pairs],
        [pair[1] for pair in pairs],
    )
    return {
        "images": len(pairs),
        "map50": map50,
        "map75": map75,
        "map50_95": map50_95,
        "mean_image_map50": float(np.mean(per_image_map50)) if per_image_map50 else 0.0,
        "parse_failures": parse_failures,
        "errors": errors,
        "avg_input_tokens": float(np.mean(tokens_in)) if tokens_in else None,
        "avg_output_tokens": float(np.mean(tokens_out)) if tokens_out else None,
        "avg_seconds": float(np.mean(seconds)) if seconds else None,
    }


def run_matrix_analysis(
    *,
    raw_directory: Path,
    samples_by_image: dict[str, DetectionSample],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Score every job file; returns results[model][arm_id] = metrics."""
    results: dict[str, dict[str, dict[str, Any]]] = {}
    for jsonl_path in sorted(raw_directory.glob("*.jsonl")):
        model_id, _, arm_id = jsonl_path.stem.partition("__")
        if arm_id not in ARMS_BY_ID:
            continue
        with open(jsonl_path) as file:
            raw_records = [json.loads(line) for line in file if line.strip()]
        # Retried images appear twice (error record then success); prefer the
        # successful record, otherwise keep the latest one.
        by_image: dict[str, dict[str, Any]] = {}
        for record in raw_records:
            existing = by_image.get(record["image"])
            if existing is None or existing.get("error") is not None:
                by_image[record["image"]] = record
        records = list(by_image.values())
        if not records:
            continue
        results.setdefault(model_id, {})[arm_id] = score_matrix_job(
            records, samples_by_image
        )
    return results


def load_workflow_anchor(
    artifacts_directory: Path,
    *,
    model_id: str,
    version: str,
    image_subset: set[str],
) -> float | None:
    """Mean per-image mAP@50 of a stored workflow run on the same subset."""
    run_directory = artifacts_directory / version / model_id
    paths = sorted(run_directory.glob("image_results_*.jsonl"))
    if not paths:
        return None
    values: list[float] = []
    with open(paths[-1]) as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            if (
                record.get("layer") == "L1_workflows_e2e"
                and record.get("image") in image_subset
            ):
                values.append(float(record.get("map50", 0.0)))
    return float(np.mean(values)) if values else None


def format_matrix_report(
    results: dict[str, dict[str, dict[str, Any]]],
    *,
    anchors: dict[str, dict[str, float | None]] | None = None,
) -> str:
    """Render per-model arm rankings as markdown."""
    lines: list[str] = []
    for model_id in sorted(results):
        lines.append(f"## {model_id}")
        control = results[model_id].get(CONTROL_ARM_ID, {})
        control_map50 = control.get("map50")
        lines.append(
            "| Arm | mAP@50 | vs A1 | mAP@50:95 | mean img mAP@50 | "
            "Parse fail | Errors | Avg out tok | Avg s |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        ranked = sorted(
            results[model_id].items(),
            key=lambda item: item[1]["map50"],
            reverse=True,
        )
        for arm_id, metrics in ranked:
            delta = (
                f"{(metrics['map50'] - control_map50) * 100:+.1f}"
                if control_map50 is not None
                else "n/a"
            )
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
                f"| {arm_id} | {metrics['map50'] * 100:.1f}% | {delta} | "
                f"{metrics['map50_95'] * 100:.1f}% | "
                f"{metrics['mean_image_map50'] * 100:.1f}% | "
                f"{metrics['parse_failures']} | {metrics['errors']} | "
                f"{out_tokens} | {avg_seconds} |"
            )
        if anchors and model_id in anchors:
            anchor_parts = [
                f"workflow {version}: {value * 100:.1f}%"
                for version, value in anchors[model_id].items()
                if value is not None
            ]
            if anchor_parts:
                lines.append("")
                lines.append(
                    "Workflow anchors (mean per-image mAP@50, same subset): "
                    + ", ".join(anchor_parts)
                )
        lines.append("")
    return "\n".join(lines)


def write_matrix_artifacts(
    *,
    output_directory: Path,
    results: dict[str, dict[str, dict[str, Any]]],
    anchors: dict[str, dict[str, float | None]] | None = None,
) -> Path:
    """Write matrix summary JSON and markdown report."""
    analysis_directory = output_directory / "analysis"
    analysis_directory.mkdir(parents=True, exist_ok=True)
    with open(analysis_directory / "matrix_summary.json", "w") as file:
        json.dump({"results": results, "anchors": anchors}, file, indent=2)
    report_path = analysis_directory / "report.md"
    with open(report_path, "w") as file:
        file.write(format_matrix_report(results, anchors=anchors) + "\n")
    return report_path
