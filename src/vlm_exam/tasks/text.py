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

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from vlm_exam.judge import Judge
from vlm_exam.tasks.base import EvaluationResult, Sample, Task
from vlm_exam.tasks.detection import DetectionCoordinateFormat
from vlm_exam.tasks.text_scoring import (
    character_score,
    equal,
    field_score,
    parse_json,
    region_score,
    valid_box,
)

TEXT_PROTOCOL = "mixed-text-v2"
"""Version of prompt assembly, parsing and scoring for the new text benchmark."""
TEXT_CATEGORIES = (
    "single_string",
    "transcription",
    "structured",
    "localization_recognition",
)
"""Accepted category values inside a JSON-encoded prefix."""


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True)
class TextSample(Sample):
    """An image/question pair with typed ground truth and immutable provenance."""

    category: str
    question: str
    specification: str
    expected: str
    answer: Any
    image_width: int
    image_height: int
    image_hash: str
    identity: str
    dataset_hash: str
    dataset_pairs: int
    prefix: str
    subset: str
    scoring_profile: str


class TextTask(Task):
    """Evaluate mixed Roboflow JSONL rows without sending metadata to the model."""

    def __init__(
        self,
        coordinate_format: DetectionCoordinateFormat = (
            DetectionCoordinateFormat.YXYX_NORMALIZED_0_TO_1000
        ),
        inference_hash: str = "",
    ) -> None:
        self.coordinate_format = DetectionCoordinateFormat(coordinate_format)
        self.inference_hash = inference_hash

    def load_samples(self, data_directory: str) -> list[Sample]:
        """Validate all rows and preserve multiple questions per image."""
        directory = Path(data_directory).resolve()
        pending: list[dict[str, Any]] = []
        identities: set[str] = set()
        images: dict[Path, tuple[int, int, str]] = {}
        for number, line in enumerate(
            (directory / "annotations.jsonl").read_text().splitlines(), 1
        ):
            if not line.strip():
                continue
            try:
                row, fenced = parse_json(line)
                if (
                    fenced
                    or not isinstance(row, dict)
                    or set(row) != {"image", "prefix", "suffix"}
                    or not all(isinstance(value, str) for value in row.values())
                ):
                    raise ValueError(
                        "Each row must contain exactly image, prefix, suffix strings"
                    )
                prefix, fenced = parse_json(row["prefix"])
                if (
                    fenced
                    or not isinstance(prefix, dict)
                    or not {"task", "question"} <= prefix.keys()
                    or set(prefix)
                    - {"task", "question", "specification", "subset", "scoring_profile"}
                ):
                    raise ValueError(
                        "Prefix requires task/question and accepts specification, "
                        "subset and scoring_profile"
                    )
                category = prefix["task"]
                if (
                    category not in TEXT_CATEGORIES
                    or not isinstance(prefix["question"], str)
                    or not prefix["question"].strip()
                ):
                    raise ValueError("Unknown task or empty question")
                specification = prefix.get("specification", "")
                if not isinstance(specification, str) or (
                    category != "localization_recognition" and "specification" in prefix
                ):
                    raise ValueError(
                        "Specification is a string allowed only for localization"
                    )
                relative = Path(row["image"])
                image_path = (directory / relative).resolve()
                if (
                    relative.is_absolute()
                    or not image_path.is_relative_to(directory)
                    or not image_path.is_file()
                ):
                    raise ValueError(
                        "Image must be an existing relative path inside dataset"
                    )
                if image_path not in images:
                    with Image.open(image_path) as image:
                        if image.getexif().get(274, 1) != 1:
                            raise ValueError(
                                "Export upright images before benchmarking (EXIF "
                                "orientation)"
                            )
                        width, height = image.size
                    images[image_path] = width, height, _digest(image_path.read_bytes())
                width, height, image_hash = images[image_path]
                answer: Any = row["suffix"]
                if category in ("structured", "localization_recognition"):
                    answer, fenced = parse_json(answer)
                    if fenced or not isinstance(answer, (dict, list)):
                        raise ValueError(
                            "Ground truth must be an object or array without fences"
                        )
                if category == "localization_recognition":
                    if not isinstance(answer, list):
                        raise ValueError("Localization answer must be an array")
                    for region in answer:
                        if (
                            not isinstance(region, dict)
                            or set(region) != {"bbox", "text"}
                            or not valid_box(region["bbox"])
                            or not (
                                region["text"] is None
                                or isinstance(region["text"], str)
                            )
                        ):
                            raise ValueError("Invalid canonical bbox/text ground truth")
                        left, top, right, bottom = region["bbox"]
                        if (
                            not 0 <= left <= right <= width
                            or not 0 <= top <= bottom <= height
                        ):
                            raise ValueError("Ground-truth box outside original image")
                canonical = json.dumps(prefix, ensure_ascii=False, sort_keys=True)
                content = json.dumps(
                    {
                        "task": category,
                        "question": prefix["question"],
                        "specification": specification,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                identity = _digest((image_hash + content).encode())
                if identity in identities:
                    raise ValueError("Duplicate image/question pair")
                identities.add(identity)
                subset = prefix.get("subset", _digest(canonical.encode())[:16])
                profile = prefix.get("scoring_profile", category)
                if not isinstance(subset, str) or not subset.strip():
                    raise ValueError("Subset must be a nonempty string")
                allowed = {category}
                if category == "transcription":
                    allowed.update({"exact", "italian_soft_wraps"})
                if not isinstance(profile, str) or profile not in allowed:
                    raise ValueError("Unsupported scoring profile for this category")
                pending.append(
                    dict(
                        image_path=str(image_path),
                        category=category,
                        question=prefix["question"],
                        specification=specification,
                        expected=row["suffix"],
                        answer=answer,
                        image_width=width,
                        image_height=height,
                        image_hash=image_hash,
                        identity=identity,
                        prefix=canonical,
                        subset=subset,
                        scoring_profile=profile,
                    )
                )
            except (ValueError, TypeError, KeyError, OSError) as error:
                raise ValueError(f"annotations.jsonl line {number}: {error}") from error
        if not pending:
            raise ValueError("Dataset has no samples")
        dataset_hash = _digest(
            json.dumps(
                sorted(
                    (
                        item["identity"],
                        item["prefix"],
                        item["expected"],
                        item["scoring_profile"],
                    )
                    for item in pending
                ),
                ensure_ascii=False,
            ).encode()
        )
        return [
            TextSample(**item, dataset_hash=dataset_hash, dataset_pairs=len(pending))
            for item in pending
        ]

    def expected_text(self, sample: Sample) -> str:
        """Return the untouched suffix for result provenance."""
        assert isinstance(sample, TextSample)
        return sample.expected

    def sample_metadata(self, sample: Sample) -> dict[str, Any]:
        """Record pair identity, category and frozen scoring/prompt provenance."""
        assert isinstance(sample, TextSample)
        return {
            "question": sample.prefix,
            "sample_id": sample.identity,
            "category": sample.category,
            "subset": sample.subset,
            "dataset_hash": sample.dataset_hash,
            "dataset_pairs": sample.dataset_pairs,
            "image_sha256": sample.image_hash,
            "text_protocol": TEXT_PROTOCOL,
            "scoring_profile": sample.scoring_profile,
            "coordinate_format": self.coordinate_format.value,
            "inference_hash": self.inference_hash,
        }

    def _geometry(
        self, sample: TextSample, uploaded_size: tuple[int, int] | None
    ) -> tuple[str, bool, float, float]:
        value = self.coordinate_format.value
        key = "bbox" if value.endswith("_bbox") else "box_2d"
        swapped = value.startswith("yxyx")
        if "normalized" in value:
            scale = 100.0 if value == "xyxy_normalized_0_to_100" else 1000.0
            return key, swapped, scale, scale
        if "resized_image" in value:
            if uploaded_size is None:
                raise ValueError(
                    "This coordinate profile requires provider uploaded dimensions"
                )
            return key, swapped, float(uploaded_size[0]), float(uploaded_size[1])
        return key, swapped, float(sample.image_width), float(sample.image_height)

    def build_prompt(
        self, sample: Sample, *, uploaded_size: tuple[int, int] | None = None
    ) -> str:
        """Render only question/specification plus the model's coordinate contract."""
        assert isinstance(sample, TextSample)
        if sample.category != "localization_recognition":
            return sample.question
        key, swapped, width, height = self._geometry(sample, uploaded_size)
        order = (
            "y_min, x_min, y_max, x_max" if swapped else "x_min, y_min, x_max, y_max"
        )
        example = f'{{"{key}": [{order}], "text": "..."}}'
        if "meta_flat" in self.coordinate_format.value:
            example = (
                '{"x_min": <number>, "y_min": <number>, "x_max": '
                '<number>, "y_max": <number>, "text": "..."}'
            )
        elif "meta_bbox" in self.coordinate_format.value:
            example = (
                '{"bbox": [{"x_min": <number>, "y_min": <number>, '
                '"x_max": <number>, "y_max": <number>}], "text": "..."}'
            )
        if "normalized" in self.coordinate_format.value:
            instructions = (
                f"Use coordinates normalized from 0 to {width:g} "
                "relative to the image dimensions."
            )
        else:
            frame = (
                "uploaded"
                if "resized_image" in self.coordinate_format.value
                else "original"
            )
            instructions = (
                f"Use absolute pixel coordinates in the {frame} image "
                f"({width:g} pixels wide, {height:g} pixels high)."
            )
        return (
            f"{sample.question}\n\n{sample.specification}\n\n"
            f"Return only a valid JSON array with this structure:\n[{example}]\n\n"
            f"{instructions}\n\nUse exactly the keys shown. "
            "Use text, never label, for transcription. "
            "Do not include explanations or Markdown."
        )

    def parse_regions(
        self, value: Any, sample: TextSample, uploaded_size: tuple[int, int] | None
    ) -> list[dict[str, Any] | None]:
        """Convert declared model coordinates to original pixels for scoring/display."""
        if not isinstance(value, list):
            raise ValueError("Expected a JSON array")
        key, swapped, width, height = self._geometry(sample, uploaded_size)
        result: list[dict[str, Any] | None] = []
        for item in value:
            try:
                if (
                    not isinstance(item, dict)
                    or "text" not in item
                    or not (item["text"] is None or isinstance(item["text"], str))
                ):
                    raise ValueError("Expected text string or null")
                if "meta_flat" in self.coordinate_format.value:
                    if set(item) != {"x_min", "y_min", "x_max", "y_max", "text"}:
                        raise ValueError("Unexpected keys")
                    box = [item[name] for name in ("x_min", "y_min", "x_max", "y_max")]
                elif "meta_bbox" in self.coordinate_format.value:
                    if (
                        set(item) != {"bbox", "text"}
                        or not isinstance(item["bbox"], list)
                        or len(item["bbox"]) != 1
                        or set(item["bbox"][0]) != {"x_min", "y_min", "x_max", "y_max"}
                    ):
                        raise ValueError("Invalid nested bbox")
                    box = [
                        item["bbox"][0][name]
                        for name in ("x_min", "y_min", "x_max", "y_max")
                    ]
                else:
                    if set(item) != {key, "text"}:
                        raise ValueError("Unexpected keys")
                    box = item[key]
                if not valid_box(box):
                    raise ValueError("Invalid box")
                if swapped:
                    box = [box[1], box[0], box[3], box[2]]
                if (
                    not 0 <= box[0] <= box[2] <= width
                    or not 0 <= box[1] <= box[3] <= height
                ):
                    raise ValueError("Box outside declared coordinate frame")
                result.append(
                    {
                        "bbox": [
                            box[0] * sample.image_width / width,
                            box[1] * sample.image_height / height,
                            box[2] * sample.image_width / width,
                            box[3] * sample.image_height / height,
                        ],
                        "text": item["text"],
                    }
                )
            except (ValueError, TypeError, KeyError):
                result.append(None)
        return result

    def evaluate(
        self,
        sample: Sample,
        prediction: str,
        *,
        judge: Judge | None = None,
        uploaded_size: tuple[int, int] | None = None,
    ) -> EvaluationResult:
        """Dispatch deterministic metrics by category; API failures remain unscored."""
        assert isinstance(sample, TextSample)
        if prediction.startswith("ERROR:"):
            return EvaluationResult(correct=False, details={"status": "provider_error"})
        details: dict[str, Any] = {"status": "ok", "format_compliant": True}
        try:
            if sample.scoring_profile == "exact" or sample.category == "single_string":
                value = float(equal(sample.expected, prediction))
            elif sample.category == "transcription":
                value = character_score(
                    sample.expected,
                    prediction,
                    sample.scoring_profile == "italian_soft_wraps",
                )
            else:
                parsed, fenced = parse_json(prediction)
                details["format_compliant"] = not fenced
                if sample.category == "structured":
                    details.update(field_score(sample.answer, parsed))
                else:
                    details.update(
                        region_score(
                            sample.answer,
                            self.parse_regions(parsed, sample, uploaded_size),
                        )
                    )
                    details["format_compliant"] &= details["invalid_regions"] == 0
                value = details.pop("score")
        except (ValueError, TypeError, KeyError, RecursionError) as error:
            value = 0.0
            details.update(
                status="invalid_response", format_compliant=False, error=str(error)
            )
        return EvaluationResult(
            correct=value == 1.0,
            score=value,
            match_method=sample.category,
            details=details,
        )
