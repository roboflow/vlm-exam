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

import json
import math
import re
import unicodedata
from typing import Any

from rapidfuzz.distance import Levenshtein


def normalize(value: str) -> str:
    """Normalize Unicode and line endings without changing content or case."""
    return unicodedata.normalize(
        "NFC", value.replace("\r\n", "\n").replace("\r", "\n")
    ).strip()


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON value: {value}")


def parse_json(value: str) -> tuple[Any, bool]:
    """Parse a whole JSON response, allowing only a complete enclosing fence."""
    value = normalize(value)
    match = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n```", value, re.S | re.I)
    fenced = match is not None
    if match:
        value = match.group(1)
    return json.loads(
        value, object_pairs_hook=_pairs, parse_constant=_invalid_constant
    ), fenced


def flatten(value: Any, path: tuple[Any, ...] = ()) -> dict[tuple[Any, ...], Any]:
    """Map nested leaf paths to values, retaining nulls and empty containers."""
    if isinstance(value, dict) and value:
        return {
            key: leaf
            for name, child in value.items()
            for key, leaf in flatten(child, (*path, name)).items()
        }
    if isinstance(value, list) and value:
        return {
            key: leaf
            for index, child in enumerate(value)
            for key, leaf in flatten(child, (*path, index)).items()
        }
    return {path: value}


def equal(expected: Any, predicted: Any) -> bool:
    """Compare typed values, normalizing strings without numerical coercion."""
    if type(expected) is not type(predicted):
        return False
    return (
        normalize(expected) == normalize(predicted)
        if isinstance(expected, str)
        else expected == predicted
    )


def field_score(expected: Any, predicted: Any) -> dict[str, Any]:
    """Compute exact typed leaf F1, penalizing missing and extra paths."""
    if type(expected) is not type(predicted) or not isinstance(predicted, (dict, list)):
        raise ValueError("Expected matching object or array root")
    reference, output = flatten(expected), flatten(predicted)
    matches = {
        key for key in reference if key in output and equal(reference[key], output[key])
    }
    populated = {key for key, value in reference.items() if value is not None}
    return {
        "score": 2 * len(matches) / (len(reference) + len(output)),
        "matched_fields": len(matches),
        "reference_fields": len(reference),
        "predicted_fields": len(output),
        "non_null_field_accuracy": len(matches & populated) / len(populated)
        if populated
        else None,
        "object_exact_match": len(matches) == len(reference) == len(output),
    }


def character_score(expected: str, predicted: str, soft_wraps: bool = False) -> float:
    """Compute character similarity, optionally joining within-paragraph wraps."""
    left, right = normalize(expected), normalize(predicted)
    if soft_wraps:
        left, right = (
            re.sub(r"(?<!\n)\n(?!\n)", " ", value) for value in (left, right)
        )
    return Levenshtein.normalized_similarity(left, right)


def valid_box(value: Any) -> bool:
    """Require four finite, ordered numeric xyxy coordinates."""
    return (
        isinstance(value, list)
        and len(value) == 4
        and all(
            type(number) in (int, float) and math.isfinite(number) for number in value
        )
        and value[0] <= value[2]
        and value[1] <= value[3]
    )


def _iou(left: list[float], right: list[float]) -> float:
    intersection = max(0, min(left[2], right[2]) - max(left[0], right[0])) * max(
        0, min(left[3], right[3]) - max(left[1], right[1])
    )
    union = (
        (left[2] - left[0]) * (left[3] - left[1])
        + (right[2] - right[0]) * (right[3] - right[1])
        - intersection
    )
    return intersection / union if union else 0.0


def _matching(edges: list[list[int]]) -> int:
    owners: dict[int, int] = {}

    def visit(reference: int, seen: set[int]) -> bool:
        for prediction in edges[reference]:
            if prediction in seen:
                continue
            seen.add(prediction)
            if prediction not in owners or visit(owners[prediction], seen):
                owners[prediction] = reference
                return True
        return False

    return sum(visit(index, set()) for index in range(len(edges)))


def region_score(
    reference: list[dict[str, Any]], predictions: list[dict[str, Any] | None]
) -> dict[str, Any]:
    """Compute maximum-cardinality joint IoU>0.5/text F1 and detection F1."""
    detection = [
        [
            index
            for index, prediction in enumerate(predictions)
            if prediction is not None and _iou(region["bbox"], prediction["bbox"]) > 0.5
        ]
        for region in reference
    ]
    joint = [
        [
            index
            for index in candidates
            if equal(region["text"], predictions[index]["text"])
        ]
        for region, candidates in zip(reference, detection)
    ]
    total = len(reference) + len(predictions)
    matches = _matching(joint)
    return {
        "score": 2 * matches / total if total else 1.0,
        "detection_f1": 2 * _matching(detection) / total if total else 1.0,
        "matched_regions": matches,
        "reference_regions": len(reference),
        "predicted_regions": len(predictions),
        "invalid_regions": sum(item is None for item in predictions),
    }
