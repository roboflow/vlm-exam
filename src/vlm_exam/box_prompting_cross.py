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
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import supervision as sv

from vlm_exam.box_prompting import load_case_image, raw_file_name
from vlm_exam.box_prompting_common import (
    PREDICTION_LABEL,
    BoxPromptingBackend,
    box_convention_clause,
    format_prompt_boxes,
    parse_class_agnostic,
)
from vlm_exam.box_prompting_groups import ImageGroup
from vlm_exam.box_prompting_round2 import MAX_EXAMPLES
from vlm_exam.tasks.detection import (
    DetectionCoordinateFormat,
    DetectionSample,
    compute_image_map50,
)

CROSS_ARMS = ("joint", "pairwise")
"""Cross-image arms: one request per group, or one request per target."""

TARGET_KEY = "target"
"""JSON key holding the detections in the pairwise arm."""

_ORDINALS = ("first", "second", "third", "fourth", "fifth", "sixth")
_Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class CrossCase:
    """One group's cross-image prompt: examples on one image, targets on others.

    Attributes:
        group_id: Source group identifier.
        class_name: Prompted class, or ``None`` when the group shares no
            class and every annotated object counts as a target.
        prompt_image: Basename of the image carrying the example boxes.
        positive_xyxy: Example boxes in prompt-image pixels, largest first.
        negative_xyxy: Counterexample boxes (other classes) in prompt-image
            pixels.
        negative_classes: Class name of each counterexample box.
        target_images: Basenames of the four images to detect on.
        target_xyxy: Ground-truth boxes per target image.
    """

    group_id: str
    class_name: str | None
    prompt_image: str
    positive_xyxy: tuple[_Box, ...]
    negative_xyxy: tuple[_Box, ...]
    negative_classes: tuple[str, ...]
    target_images: tuple[str, ...]
    target_xyxy: dict[str, tuple[_Box, ...]]


def _box_area(box: np.ndarray) -> float:
    return float((box[2] - box[0]) * (box[3] - box[1]))


def _largest_first(boxes: np.ndarray) -> tuple[_Box, ...]:
    order = np.argsort([-_box_area(box) for box in boxes], kind="stable")
    return tuple(tuple(float(value) for value in boxes[index]) for index in order)


def _class_boxes(sample: DetectionSample, class_name: str | None) -> np.ndarray:
    if class_name is None:
        return sample.ground_truth.xyxy
    class_ids = sample.ground_truth.class_id
    if class_ids is None:
        return np.zeros((0, 4), dtype=np.float32)
    class_id = sample.classes.index(class_name)
    return sample.ground_truth.xyxy[class_ids == class_id]


def _pick_class(
    group: ImageGroup,
    sample_index: dict[str, DetectionSample],
) -> str | None:
    image_counts: Counter[str] = Counter()
    instance_counts: Counter[str] = Counter()
    for name in group.images:
        for class_name in group.classes_by_image[name]:
            image_counts[class_name] += 1
            instance_counts[class_name] += len(
                _class_boxes(sample_index[name], class_name)
            )
    if not image_counts or max(image_counts.values()) < 2:
        return None
    return sorted(
        image_counts,
        key=lambda name: (-image_counts[name], -instance_counts[name], name),
    )[0]


def _negatives(
    sample: DetectionSample,
    class_name: str,
) -> tuple[tuple[_Box, ...], tuple[str, ...]]:
    class_ids = sample.ground_truth.class_id
    if class_ids is None:
        return (), ()
    unique_ids, counts = np.unique(class_ids, return_counts=True)
    positive_id = sample.classes.index(class_name)
    other_classes = [
        int(class_id)
        for class_id, _ in sorted(
            zip(unique_ids, counts), key=lambda item: (-item[1], item[0])
        )
        if int(class_id) != positive_id
    ]
    candidates: list[tuple[int, int, int, _Box]] = []
    for class_order, class_id in enumerate(other_classes):
        for rank, box in enumerate(
            _largest_first(sample.ground_truth.xyxy[class_ids == class_id])
        ):
            candidates.append((rank, class_order, class_id, box))
    candidates.sort(key=lambda item: (item[0], item[1]))
    chosen = candidates[:MAX_EXAMPLES]
    return (
        tuple(box for _, _, _, box in chosen),
        tuple(sample.classes[class_id] for _, _, class_id, _ in chosen),
    )


def build_cross_case(
    group: ImageGroup,
    sample_index: dict[str, DetectionSample],
) -> CrossCase:
    """Choose the prompted class, prompt image, examples, and targets.

    The class present in the most group images (ties: most instances,
    then name) is prompted; the image with the most instances of it (ties:
    name) carries up to :data:`MAX_EXAMPLES` largest instances as
    examples plus up to :data:`MAX_EXAMPLES` counterexamples from other
    classes. Groups sharing no class prompt every annotated object of the
    densest image and score against all boxes in the targets.

    Args:
        group: Image group.
        sample_index: Mapping of image basename to detection sample.

    Returns:
        The cross-image case.
    """
    class_name = _pick_class(group, sample_index)
    prompt_image = sorted(
        group.images,
        key=lambda name: (-len(_class_boxes(sample_index[name], class_name)), name),
    )[0]
    prompt_sample = sample_index[prompt_image]
    positives = _largest_first(_class_boxes(prompt_sample, class_name))[:MAX_EXAMPLES]
    if class_name is None:
        negatives: tuple[_Box, ...] = ()
        negative_classes: tuple[str, ...] = ()
    else:
        negatives, negative_classes = _negatives(prompt_sample, class_name)
    targets = tuple(name for name in group.images if name != prompt_image)
    return CrossCase(
        group_id=group.group_id,
        class_name=class_name,
        prompt_image=prompt_image,
        positive_xyxy=positives,
        negative_xyxy=negatives,
        negative_classes=negative_classes,
        target_images=targets,
        target_xyxy={
            name: tuple(
                tuple(float(value) for value in box)
                for box in _class_boxes(sample_index[name], class_name)
            )
            for name in targets
        },
    )


def _image_convention(
    coordinate_format: DetectionCoordinateFormat,
) -> str:
    if coordinate_format == DetectionCoordinateFormat.XYXY_ABSOLUTE_RESIZED_IMAGE:
        return (
            "the top-left and bottom-right corners in absolute pixel "
            "coordinates of the image the entry belongs to"
        )
    if coordinate_format == DetectionCoordinateFormat.XYXY_NORMALIZED_0_TO_1000:
        return (
            "the top-left and bottom-right corners as integers between 0 and "
            "1000, normalized to the width (x) and height (y) of the image "
            "the entry belongs to"
        )
    raise ValueError(f"Unsupported coordinate format: {coordinate_format!r}")


def _size_note(
    coordinate_format: DetectionCoordinateFormat,
    uploaded_wh: tuple[int, int],
) -> str:
    if coordinate_format == DetectionCoordinateFormat.XYXY_ABSOLUTE_RESIZED_IMAGE:
        return f" ({uploaded_wh[0]}x{uploaded_wh[1]} pixels)"
    return ""


def _example_lead(
    case: CrossCase,
    prompt_sample: DetectionSample,
    coordinate_format: DetectionCoordinateFormat,
    uploaded_wh: tuple[int, int],
) -> tuple[str, str]:
    original_wh = (prompt_sample.image_width, prompt_sample.image_height)
    positives = format_prompt_boxes(
        case.positive_xyxy, original_wh, coordinate_format, uploaded_wh
    )
    lead = (
        "The first image contains example objects of a target type. Their "
        "bounding boxes are [x_min, y_min, x_max, y_max] given as "
        f"{box_convention_clause(coordinate_format, uploaded_wh)}: {positives}. "
    )
    tail = ". "
    if case.negative_xyxy:
        negatives = format_prompt_boxes(
            case.negative_xyxy, original_wh, coordinate_format, uploaded_wh
        )
        lead += (
            "The first image also contains counterexample objects that must "
            f"NOT be detected, with bounding boxes: {negatives}. "
        )
        tail = ", and do not detect objects like the counterexamples. "
    return lead, tail


def build_joint_prompt(
    case: CrossCase,
    sample_index: dict[str, DetectionSample],
    coordinate_format: DetectionCoordinateFormat,
    uploaded_sizes: list[tuple[int, int]],
) -> str:
    """Prompt for one request carrying the prompt image and all targets.

    Args:
        case: Cross-image case.
        sample_index: Mapping of image basename to detection sample.
        coordinate_format: Coordinate convention of the prompt and answer.
        uploaded_sizes: Uploaded ``(width, height)`` of the prompt image
            followed by each target image.

    Returns:
        Prompt text requesting a JSON object keyed by target image.
    """
    lead, tail = _example_lead(
        case, sample_index[case.prompt_image], coordinate_format, uploaded_sizes[0]
    )
    target_ordinals = [_ORDINALS[index + 1] for index in range(len(case.target_images))]
    ordinal_phrase = ", ".join(target_ordinals[:-1]) + f", and {target_ordinals[-1]}"
    keys = [f"image_{index + 2}" for index in range(len(case.target_images))]
    key_meanings = ", ".join(
        f'"{key}" is the {ordinal} image'
        f"{_size_note(coordinate_format, uploaded_sizes[index + 1])}"
        for index, (key, ordinal) in enumerate(zip(keys, target_ordinals))
    )
    key_list = ", ".join(f'"{key}"' for key in keys)
    return (
        lead + f"Detect all objects in the {ordinal_phrase} images that are visually "
        "similar to the example objects. Do not report any detections for the "
        f"first image{tail}"
        f"Output a JSON object with the keys {key_list}, where {key_meanings}. "
        "Each value must be a JSON list where each entry contains the 2D "
        'bounding box in the key "box_2d" and the text label in the key '
        '"label". The "box_2d" value must be [x_min, y_min, x_max, y_max]: '
        f"{_image_convention(coordinate_format)}. "
        f'Use the label "{PREDICTION_LABEL}" for every entry. Return only the '
        "JSON object, with no extra text."
    )


def build_pairwise_prompt(
    case: CrossCase,
    sample_index: dict[str, DetectionSample],
    coordinate_format: DetectionCoordinateFormat,
    uploaded_sizes: list[tuple[int, int]],
) -> str:
    """Prompt for one request carrying the prompt image and one target.

    Args:
        case: Cross-image case.
        sample_index: Mapping of image basename to detection sample.
        coordinate_format: Coordinate convention of the prompt and answer.
        uploaded_sizes: Uploaded ``(width, height)`` of the prompt image
            and the target image.

    Returns:
        Prompt text requesting a JSON object with a single target key.
    """
    lead, tail = _example_lead(
        case, sample_index[case.prompt_image], coordinate_format, uploaded_sizes[0]
    )
    convention = _image_convention(coordinate_format)
    return (
        lead + "Detect all objects in the second image"
        f"{_size_note(coordinate_format, uploaded_sizes[1])} that are visually "
        "similar to the example objects. Do not report any detections for the "
        f"first image{tail}"
        f'Output a JSON object with a single key "{TARGET_KEY}" whose value is a '
        "JSON list where each entry contains the 2D bounding box in the key "
        '"box_2d" and the text label in the key "label". The "box_2d" value '
        f"must be [x_min, y_min, x_max, y_max]: {convention}. "
        f'Use the label "{PREDICTION_LABEL}" for every entry. Return only the '
        "JSON object, with no extra text."
    )


def request_key(arm: str, case: CrossCase, target_image: str | None) -> str:
    """Resumable identifier of one request.

    Args:
        arm: One of :data:`CROSS_ARMS`.
        case: Cross-image case.
        target_image: Target basename for the pairwise arm.

    Returns:
        Group id for joint requests, ``group/target`` for pairwise ones.
    """
    if arm == "joint":
        return case.group_id
    return f"{case.group_id}/{target_image}"


def _completed_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with open(path) as file:
        for line in file:
            if line.strip():
                record = json.loads(line)
                if record.get("error") is None:
                    done.add(record["request_key"])
    return done


def load_cross_records(
    raw_directory: Path,
    model_key: str,
    arm: str,
) -> list[dict[str, Any]]:
    """Load one arm's records, keeping one record per request key.

    Args:
        raw_directory: Directory holding the raw JSONL files.
        model_key: vlm-exam model key.
        arm: One of :data:`CROSS_ARMS`.

    Returns:
        Deduplicated records; a successful record wins over a failed one.
    """
    path = raw_directory / raw_file_name(model_key, arm)
    if not path.exists():
        return []
    by_key: dict[str, dict[str, Any]] = {}
    with open(path) as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            existing = by_key.get(record["request_key"])
            if existing is None or existing.get("error") is not None:
                by_key[record["request_key"]] = record
    return list(by_key.values())


def _collect_request(
    *,
    arm: str,
    case: CrossCase,
    target_images: tuple[str, ...],
    sample_index: dict[str, DetectionSample],
    backend: BoxPromptingBackend,
    effort: str,
    output_path: Path,
    file_lock: threading.Lock,
    progress: dict[str, int],
    progress_lock: threading.Lock,
    total: int,
) -> None:
    key = request_key(arm, case, target_images[0] if arm == "pairwise" else None)
    record: dict[str, Any] = {
        "model": backend.model_key,
        "provider_model_id": backend.provider_model_id,
        "coordinate_format": backend.coordinate_format.value,
        "arm": arm,
        "reasoning_effort": effort,
        "request_key": key,
        "group_id": case.group_id,
        "class_name": case.class_name,
        "prompt_image": case.prompt_image,
        "target_images": list(target_images),
        "positive_xyxy": [list(box) for box in case.positive_xyxy],
        "negative_xyxy": [list(box) for box in case.negative_xyxy],
        "negative_classes": list(case.negative_classes),
        "target_counts": {name: len(case.target_xyxy[name]) for name in target_images},
        "prompt": None,
        "error": None,
    }
    try:
        images = [load_case_image(sample_index[case.prompt_image])] + [
            load_case_image(sample_index[name]) for name in target_images
        ]
        uploaded_sizes = [backend.uploaded_size(image) for image in images]
        if arm == "joint":
            prompt = build_joint_prompt(
                case, sample_index, backend.coordinate_format, uploaded_sizes
            )
        else:
            prompt = build_pairwise_prompt(
                case, sample_index, backend.coordinate_format, uploaded_sizes
            )
        record["prompt"] = prompt
        record.update(backend.call(images=images, prompt=prompt, effort=effort))
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"
    with file_lock:
        with open(output_path, "a") as file:
            file.write(json.dumps(record) + "\n")
    with progress_lock:
        progress[arm] += 1
        done = progress[arm]
    print(f"Completed {arm} {key} ({done}/{total})", flush=True)


def run_cross_collection(
    *,
    cases: list[CrossCase],
    sample_index: dict[str, DetectionSample],
    backend: BoxPromptingBackend,
    effort: str,
    output_directory: Path,
    max_workers: int = 4,
    arms: tuple[str, ...] = CROSS_ARMS,
) -> None:
    """Collect the requested arms with a shared request-level worker pool.

    Joint requests send one group per call; pairwise requests send one
    target per call. Jobs run largest target count first and append to
    resumable per-arm JSONL files.

    Args:
        cases: Cross-image cases.
        sample_index: Mapping of image basename to detection sample.
        backend: Model backend.
        effort: Reasoning effort forwarded to the backend.
        output_directory: Experiment root; raw files land in ``raw/``.
        max_workers: Concurrent requests across all arms.
        arms: Subset of :data:`CROSS_ARMS` to collect.
    """
    raw_directory = output_directory / "raw"
    raw_directory.mkdir(parents=True, exist_ok=True)
    with open(output_directory / "cases.json", "w") as file:
        json.dump(
            [
                {
                    "group_id": case.group_id,
                    "class_name": case.class_name,
                    "prompt_image": case.prompt_image,
                    "positive_xyxy": [list(box) for box in case.positive_xyxy],
                    "negative_xyxy": [list(box) for box in case.negative_xyxy],
                    "negative_classes": list(case.negative_classes),
                    "target_images": list(case.target_images),
                    "target_counts": {
                        name: len(boxes) for name, boxes in case.target_xyxy.items()
                    },
                }
                for case in cases
            ],
            file,
            indent=2,
        )

    file_locks = {arm: threading.Lock() for arm in arms}
    progress = {arm: 0 for arm in arms}
    totals = {arm: 0 for arm in arms}
    progress_lock = threading.Lock()
    jobs: list[tuple[str, CrossCase, tuple[str, ...]]] = []
    for arm in arms:
        done = _completed_keys(raw_directory / raw_file_name(backend.model_key, arm))
        for case in cases:
            if arm == "joint":
                requests = [case.target_images]
            else:
                requests = [(name,) for name in case.target_images]
            for targets in requests:
                totals[arm] += 1
                key = request_key(arm, case, targets[0] if arm == "pairwise" else None)
                if key in done:
                    progress[arm] += 1
                else:
                    jobs.append((arm, case, targets))
    jobs.sort(key=lambda job: -sum(len(job[1].target_xyxy[name]) for name in job[2]))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _collect_request,
                arm=arm,
                case=case,
                target_images=targets,
                sample_index=sample_index,
                backend=backend,
                effort=effort,
                output_path=raw_directory / raw_file_name(backend.model_key, arm),
                file_lock=file_locks[arm],
                progress=progress,
                progress_lock=progress_lock,
                total=totals[arm],
            )
            for arm, case, targets in jobs
        ]
        for future in futures:
            future.result()
    print("Collection finished.", flush=True)


def _ground_truth(boxes: tuple[_Box, ...]) -> sv.Detections:
    if not boxes:
        return sv.Detections.empty()
    xyxy = np.array(boxes, dtype=np.float32)
    return sv.Detections(xyxy=xyxy, class_id=np.zeros(len(xyxy), dtype=int))


def record_target_detections(
    record: dict[str, Any],
    target_index: int,
    sample: DetectionSample,
) -> tuple[sv.Detections, bool]:
    """Parse one target image's predictions out of a raw record.

    Args:
        record: Raw cross-image record.
        target_index: Position of the target within ``target_images``.
        sample: Target sample for coordinate scaling.

    Returns:
        Parsed detections and a parse-failure flag.
    """
    coordinate_format = DetectionCoordinateFormat(record["coordinate_format"])
    sizes = record.get("uploaded_sizes") or []
    uploaded_wh = None
    if len(sizes) > target_index + 1:
        uploaded_wh = (int(sizes[target_index + 1][0]), int(sizes[target_index + 1][1]))
    key = TARGET_KEY if record["arm"] == "pairwise" else f"image_{target_index + 2}"
    return parse_class_agnostic(
        record.get("raw_output", ""), sample, coordinate_format, uploaded_wh, key=key
    )


def score_cross_arm(
    records: list[dict[str, Any]],
    cases_by_group: dict[str, CrossCase],
    sample_index: dict[str, DetectionSample],
) -> dict[str, Any]:
    """Score one arm's records per target image and per group.

    Args:
        records: Deduplicated records of one arm.
        cases_by_group: Mapping of group id to cross-image case.
        sample_index: Mapping of image basename to detection sample.

    Returns:
        Dict with aggregate ``metrics``, per-group ``groups``, and
        per-target ``targets`` details.
    """
    targets: dict[str, dict[str, Any]] = {}
    parse_failures = 0
    errors = 0
    tokens_out: list[int] = []
    seconds: list[float] = []
    for record in records:
        case = cases_by_group.get(record["group_id"])
        if case is None:
            continue
        if record.get("error") is not None:
            errors += 1
        else:
            if record.get("output_tokens") is not None:
                tokens_out.append(record["output_tokens"])
            if record.get("inference_seconds") is not None:
                seconds.append(record["inference_seconds"])
        for index, name in enumerate(record["target_images"]):
            sample = sample_index.get(name)
            if sample is None:
                continue
            ground_truth = _ground_truth(case.target_xyxy[name])
            detail: dict[str, Any] = {
                "group_id": case.group_id,
                "class_name": case.class_name,
                "target_boxes": len(ground_truth),
                "error": record.get("error"),
            }
            if record.get("error") is not None:
                detail.update(
                    {"map50": 0.0, "predicted_boxes": 0, "parse_failed": False}
                )
            else:
                detections, failed = record_target_detections(record, index, sample)
                parse_failures += failed
                detail.update(
                    {
                        "map50": compute_image_map50(detections, ground_truth),
                        "predicted_boxes": len(detections),
                        "parse_failed": failed,
                    }
                )
            targets[f"{case.group_id}/{name}"] = detail

    groups: dict[str, dict[str, Any]] = {}
    for group_id, case in cases_by_group.items():
        scores = [
            targets[f"{group_id}/{name}"]["map50"]
            for name in case.target_images
            if f"{group_id}/{name}" in targets
        ]
        if scores:
            groups[group_id] = {
                "class_name": case.class_name,
                "prompt_image": case.prompt_image,
                "targets_scored": len(scores),
                "mean_map50": float(np.mean(scores)),
            }
    all_scores = [detail["map50"] for detail in targets.values()]
    shared = [
        detail["map50"]
        for detail in targets.values()
        if detail["class_name"] is not None
    ]
    merged = [
        detail["map50"] for detail in targets.values() if detail["class_name"] is None
    ]
    metrics = {
        "requests": len(records),
        "targets": len(targets),
        "mean_target_map50": float(np.mean(all_scores)) if all_scores else 0.0,
        "mean_target_map50_shared_class": float(np.mean(shared)) if shared else None,
        "mean_target_map50_merged": float(np.mean(merged)) if merged else None,
        "mean_group_map50": (
            float(np.mean([group["mean_map50"] for group in groups.values()]))
            if groups
            else 0.0
        ),
        "parse_failures": parse_failures,
        "errors": errors,
        "avg_output_tokens": float(np.mean(tokens_out)) if tokens_out else None,
        "avg_seconds": float(np.mean(seconds)) if seconds else None,
        "total_seconds": float(np.sum(seconds)) if seconds else None,
    }
    return {"metrics": metrics, "groups": groups, "targets": targets}


def run_cross_analysis(
    *,
    raw_directory: Path,
    model_key: str,
    cases_by_group: dict[str, CrossCase],
    sample_index: dict[str, DetectionSample],
    arms: tuple[str, ...] = CROSS_ARMS,
) -> dict[str, dict[str, Any]]:
    """Score every collected cross-image arm.

    Args:
        raw_directory: Directory holding the raw JSONL files.
        model_key: vlm-exam model key whose raw files are scored.
        cases_by_group: Mapping of group id to cross-image case.
        sample_index: Mapping of image basename to detection sample.
        arms: Subset of :data:`CROSS_ARMS` to score.

    Returns:
        Mapping of arm to its scored results.
    """
    results: dict[str, dict[str, Any]] = {}
    for arm in arms:
        records = load_cross_records(raw_directory, model_key, arm)
        if records:
            results[arm] = score_cross_arm(records, cases_by_group, sample_index)
    return results


def _format_percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def _format_number(value: float | None, digits: int) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def format_cross_report(
    results: dict[str, dict[str, Any]],
    cases_by_group: dict[str, CrossCase],
    title: str,
) -> str:
    """Render the cross-image summary as markdown.

    Args:
        results: Per-arm scored results from :func:`run_cross_analysis`.
        cases_by_group: Mapping of group id to cross-image case.
        title: Report heading.

    Returns:
        Markdown report with an arm summary, the joint-vs-pairwise delta,
        and a per-group breakdown.
    """
    lines = [f"# {title}", ""]
    lines.append(
        "| Arm | Requests | Targets | Mean target mAP@50 | Shared-class groups | "
        "Merged groups | Mean group mAP@50 | Parse fail | Errors | Avg out tok | "
        "Avg s/request | Total s |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for arm in CROSS_ARMS:
        if arm not in results:
            continue
        metrics = results[arm]["metrics"]
        lines.append(
            f"| {arm} | {metrics['requests']} | {metrics['targets']} | "
            f"{_format_percent(metrics['mean_target_map50'])} | "
            f"{_format_percent(metrics['mean_target_map50_shared_class'])} | "
            f"{_format_percent(metrics['mean_target_map50_merged'])} | "
            f"{_format_percent(metrics['mean_group_map50'])} | "
            f"{metrics['parse_failures']} | {metrics['errors']} | "
            f"{_format_number(metrics['avg_output_tokens'], 0)} | "
            f"{_format_number(metrics['avg_seconds'], 1)} | "
            f"{_format_number(metrics['total_seconds'], 0)} |"
        )
    lines.append("")

    joint = results.get("joint")
    pairwise = results.get("pairwise")
    if joint is not None and pairwise is not None:
        delta = (
            joint["metrics"]["mean_target_map50"]
            - pairwise["metrics"]["mean_target_map50"]
        )
        lines.append("## Joint vs pairwise")
        lines.append("")
        lines.append(
            f"- joint {_format_percent(joint['metrics']['mean_target_map50'])} vs "
            f"pairwise {_format_percent(pairwise['metrics']['mean_target_map50'])} "
            f"({delta * 100:+.1f} points on mean target mAP@50)"
        )
        lines.append("")

    arms_present = [arm for arm in CROSS_ARMS if arm in results]
    lines.append("## Per-group mean target mAP@50")
    lines.append("")
    lines.append(
        "| Group | Class | Prompt image | Examples | Neg | Target boxes | "
        + " | ".join(arms_present)
        + " |"
    )
    lines.append("|---|---|---|---:|---:|---:|" + "---:|" * len(arms_present))
    for group_id in sorted(cases_by_group):
        case = cases_by_group[group_id]
        cells = []
        for arm in arms_present:
            group = results[arm]["groups"].get(group_id)
            cells.append(
                "n/a" if group is None else f"{group['mean_map50'] * 100:.0f}%"
            )
        total_targets = sum(len(boxes) for boxes in case.target_xyxy.values())
        lines.append(
            f"| {group_id} | {case.class_name or 'all objects'} | "
            f"{case.prompt_image} | {len(case.positive_xyxy)} | "
            f"{len(case.negative_xyxy)} | {total_targets} | " + " | ".join(cells) + " |"
        )
    return "\n".join(lines)


def write_cross_artifacts(
    *,
    output_directory: Path,
    results: dict[str, dict[str, Any]],
    cases_by_group: dict[str, CrossCase],
    title: str,
) -> Path:
    """Write the cross-image summary JSON and markdown report.

    Args:
        output_directory: Experiment root directory.
        results: Per-arm scored results.
        cases_by_group: Mapping of group id to cross-image case.
        title: Report heading.

    Returns:
        Path of the written markdown report.
    """
    analysis_directory = output_directory / "analysis"
    analysis_directory.mkdir(parents=True, exist_ok=True)
    with open(analysis_directory / "summary.json", "w") as file:
        json.dump(results, file, indent=2)
    report_path = analysis_directory / "report.md"
    with open(report_path, "w") as file:
        file.write(format_cross_report(results, cases_by_group, title) + "\n")
    return report_path
