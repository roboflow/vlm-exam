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

import json
import math
import re
from collections import Counter
from typing import Any

import numpy as np
from supervision import Detections

from vlm_exam.tasks.detection import (
    DetectionCoordinateFormat,
)

PARSER_VERSION = "v2-bounded-recovery-2"


def effective_format(coordinate_format: str, prompt: str) -> DetectionCoordinateFormat:
    """Resolve historical pixel-box keys from the actual recorded prompt."""
    value = DetectionCoordinateFormat(coordinate_format)
    # Historical string-valued profiles used box_2d despite the bbox suffix.
    if (
        value == DetectionCoordinateFormat.XYXY_ABSOLUTE_RESIZED_IMAGE_BBOX
        and 'key "box_2d"' in prompt
    ):
        return DetectionCoordinateFormat.XYXY_ABSOLUTE_RESIZED_IMAGE
    return value


def _finite(value: int | float) -> bool:
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def strict_format_error(prediction: str, coordinate_format: str) -> str | None:
    """Check JSON and requested box schema without repairing or filtering entries.

    Extra metadata is allowed. Labels are not checked against ground truth: an
    unknown class must remain a potential false positive. Finite numeric pixel
    coordinates are accepted; normalized coordinates must be integers.
    """
    try:
        entries = _json(prediction)
    except (ValueError, TypeError):
        return "invalid_json"
    if not isinstance(entries, list):
        return "not_array"
    value = DetectionCoordinateFormat(coordinate_format)
    meta = value == DetectionCoordinateFormat.XYXY_NORMALIZED_0_TO_1000_META_BBOX
    flat = value == DetectionCoordinateFormat.XYXY_NORMALIZED_0_TO_1000_META_FLAT
    key = (
        "bbox"
        if value == DetectionCoordinateFormat.XYXY_ABSOLUTE_RESIZED_IMAGE_BBOX
        else "box_2d"
    )
    for entry in entries:
        if not isinstance(entry, dict):
            return "invalid_object"
        label_key = "object_name" if meta else "label"
        if not isinstance(entry.get(label_key), str):
            return "missing_label"
        if meta:
            boxes = entry.get("bbox")
            if not isinstance(boxes, list) or not boxes:
                return "invalid_box"
        else:
            boxes = [entry] if flat else [entry.get(key)]
        for box in boxes:
            if meta or flat:
                if not isinstance(box, dict) or not all(
                    name in box for name in ("x_min", "y_min", "x_max", "y_max")
                ):
                    return "invalid_box"
                box = [box[name] for name in ("x_min", "y_min", "x_max", "y_max")]
            if not isinstance(box, list) or len(box) != 4:
                return "invalid_box"
            if any(
                type(number) not in (int, float) or not _finite(number)
                for number in box
            ):
                return "invalid_coordinate"
            if (
                "normalized" in value
                and value != DetectionCoordinateFormat.XYXY_NORMALIZED_0_TO_100
                and any(type(number) is not int for number in box)
            ):
                return "non_integer_coordinate"
    return None


def _json(text: str) -> Any:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate_key")
            result[key] = value
        return result

    return json.loads(text, object_pairs_hook=unique)


def _object_end(text: str, start: int) -> int | None:
    """Find a complete outer object without searching its nested children."""
    stack: list[str] = []
    quoted = escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char == '"':
            quoted = True
        elif char in "[{":
            stack.append(char)
        elif char in "]}":
            if not stack or stack.pop() != ("[" if char == "]" else "{"):
                return None
            if not stack:
                return index + 1
    return None


def _entries(text: str, mode: str, notes: Counter[str]) -> list[Any]:
    try:
        value = _json(text)
        if isinstance(value, list):
            return value
        if mode == "recovery" and isinstance(value, dict):
            notes["single_object"] += 1
            return [value]
        return []
    except ValueError:
        if mode == "strict":
            return []
    # Only one payload: do not select among competing arrays or code blocks.
    stripped = text.strip()
    fence = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\s*```", stripped, re.S | re.I)
    if fence:
        stripped = fence.group(1).strip()
        notes["markdown_wrapper"] += 1
    elif "```" in stripped:
        notes["ambiguous_payload"] += 1
        return []
    try:
        value = _json(stripped)
        return value if isinstance(value, list) else [value]
    except ValueError:
        pass
    # Also allow plain prose after a complete payload, never another structure.
    if stripped.startswith(("[", "{")):
        try:
            _, end = json.JSONDecoder().raw_decode(stripped)
            tail = stripped[end:]
            if (
                tail.strip()
                and any(c.isalpha() for c in tail)
                and not any(c in tail for c in "[]{}")
            ):
                value = _json(stripped[:end])
                notes["prose_wrapper"] += 1
                return value if isinstance(value, list) else [value]
        except ValueError:
            pass
    # Prose is removable only around a single complete, valid JSON payload.
    if not stripped.startswith(("[", "{")):
        start = min(
            (i for i in (stripped.find("["), stripped.find("{")) if i >= 0), default=-1
        )
        if start < 0:
            return []
        try:
            _, end = json.JSONDecoder().raw_decode(stripped, start)
            if any(c in stripped[end:] for c in "[{"):
                return []
            value = _json(stripped[start:end])
            notes["prose_wrapper"] += 1
            return value if isinstance(value, list) else [value]
        except ValueError:
            return []
    array = stripped.startswith("[")
    index = 1 if array else 0
    entries: list[Any] = []
    notes["complete_object_extraction"] += 1
    while index < len(stripped):
        while index < len(stripped) and (
            stripped[index].isspace() or stripped[index] == ","
        ):
            index += 1
        if index == len(stripped):
            break
        if stripped[index] == "]" and array:
            if stripped[index + 1 :].strip():
                notes["ambiguous_payload"] += 1
                return []
            break
        if stripped[index] != "{":
            # No arbitrary searching for a brace later in the response.
            notes["unsupported_structure"] += 1
            return []
        end = _object_end(stripped, index)
        if end is None:
            notes["incomplete_object_discarded"] += 1
            break
        try:
            entries.append(_json(stripped[index:end]))
        except ValueError:
            notes["invalid_object_discarded"] += 1
        index = end
    return entries


def _field(
    entry: dict[str, Any],
    key: str,
    aliases: tuple[str, ...],
    mode: str,
    notes: Counter[str],
) -> Any:
    found = [name for name in aliases if name in entry]
    if len(found) > 1 and any(entry[name] != entry[found[0]] for name in found[1:]):
        raise ValueError("conflicting_fields")
    if key in entry:
        return entry[key]
    if mode == "recovery" and found:
        notes["field_alias"] += 1
        return entry[found[0]]
    raise ValueError("missing_field")


def _number(value: Any, mode: str, notes: Counter[str]) -> float:
    if type(value) is str and mode == "recovery":
        if not re.fullmatch(
            r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?", value
        ):
            raise ValueError("invalid_coordinate")
        notes["numeric_string"] += 1
    elif type(value) not in (int, float):
        raise ValueError("invalid_coordinate")
    try:
        number = float(value)
    except (ValueError, OverflowError):
        raise ValueError("invalid_coordinate") from None
    if not math.isfinite(number):
        raise ValueError("invalid_coordinate")
    return number


def _confidence(entry: dict[str, Any], parent: dict[str, Any]) -> float:
    value = entry.get("confidence", parent.get("confidence", 1.0))
    if type(value) not in (int, float):
        return 1.0
    try:
        return float(value) if math.isfinite(value) and 0 <= value <= 1 else 1.0
    except OverflowError:
        return 1.0


def parse_response(
    prediction: str,
    resolution_wh: tuple[int, int],
    classes: list[str],
    coordinate_format: str,
    uploaded_wh: tuple[int, int],
    mode: str = "recovery",
    diagnostics: dict[str, Any] | None = None,
) -> Detections:
    """Decode using bounded, model-independent recovery and explicit geometry.

    No nested-object searches, schema reconstruction, coordinate guessing,
    clipping, corner reordering, deduplication or point-to-box conversion.
    Diagnostics contain counts of operations, not an assertion of correctness.
    """
    if mode not in {"strict", "recovery"}:
        raise ValueError(f"Unknown parser mode: {mode}")
    fmt = DetectionCoordinateFormat(coordinate_format)
    notes: Counter[str] = Counter()
    if diagnostics is not None:
        diagnostics.update(version=PARSER_VERSION, mode=mode, operations={})
    if mode == "strict":
        error = strict_format_error(prediction, fmt)
        if error is not None:
            if diagnostics is not None:
                diagnostics["rejected"] = error
            return Detections.empty()
    entries = _entries(prediction, mode, notes)
    boxes_out: list[list[float]] = []
    ids: list[int] = []
    confidences: list[float] = []
    class_index = {label: index for index, label in enumerate(classes)}
    meta = fmt == DetectionCoordinateFormat.XYXY_NORMALIZED_0_TO_1000_META_BBOX
    flat = fmt == DetectionCoordinateFormat.XYXY_NORMALIZED_0_TO_1000_META_FLAT
    w, h = resolution_wh
    if "normalized" in fmt:
        maximum = fmt.normalized_maximum
        sx, sy = w / maximum, h / maximum
    elif "resized" in fmt:
        sx, sy = w / uploaded_wh[0], h / uploaded_wh[1]
    else:
        sx = sy = 1.0
    failed = False
    for entry in entries:
        try:
            if not isinstance(entry, dict):
                raise ValueError("invalid_object")
            label = _field(
                entry,
                "object_name" if meta else "label",
                ("label", "object_name", "description"),
                mode,
                notes,
            )
            if not isinstance(label, str) or label not in class_index:
                raise ValueError("invalid_label")
            if flat:
                boxes = [entry]
            else:
                key = (
                    "bbox"
                    if meta
                    or fmt == DetectionCoordinateFormat.XYXY_ABSOLUTE_RESIZED_IMAGE_BBOX
                    else "box_2d"
                )
                raw = _field(entry, key, ("box_2d", "bbox_2d", "bbox"), mode, notes)
                if meta:
                    if isinstance(raw, dict) and mode == "recovery":
                        notes["singleton_box_container"] += 1
                        raw = [raw]
                    if not isinstance(raw, list) or not raw:
                        raise ValueError("invalid_box")
                    boxes = raw
                else:
                    boxes = [raw]
            for box in boxes:
                try:
                    if meta or flat:
                        if not isinstance(box, dict):
                            raise ValueError("invalid_box")
                        values = [box[k] for k in ("x_min", "y_min", "x_max", "y_max")]
                    else:
                        values = box
                    if not isinstance(values, list) or len(values) != 4:
                        raise ValueError("invalid_box")
                    first, second, third, fourth = [
                        _number(v, mode, notes) for v in values
                    ]
                    if fmt.value.startswith("yxyx"):
                        x1, y1, x2, y2 = second, first, fourth, third
                    else:
                        x1, y1, x2, y2 = first, second, third, fourth
                    scaled = [x1 * sx, y1 * sy, x2 * sx, y2 * sy]
                    if not all(math.isfinite(v) for v in scaled):
                        raise ValueError("invalid_coordinate")
                    boxes_out.append(scaled)
                    ids.append(class_index[label])
                    confidences.append(
                        _confidence(box if isinstance(box, dict) else entry, entry)
                    )
                except (ValueError, KeyError) as error:
                    notes[str(error)] += 1
                    failed = True
        except ValueError as error:
            notes[str(error)] += 1
            failed = True
    if diagnostics is not None:
        diagnostics["operations"] = dict(notes)
    if (mode == "strict" and failed) or not boxes_out:
        return Detections.empty()
    return Detections(
        xyxy=np.asarray(boxes_out, dtype=float),
        class_id=np.asarray(ids, dtype=int),
        confidence=np.asarray(confidences, dtype=float),
    )
