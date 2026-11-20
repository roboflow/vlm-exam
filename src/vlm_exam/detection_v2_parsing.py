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
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

import numpy as np
from supervision import Detections

from vlm_exam.tasks.detection import (
    DetectionCoordinateFormat,
)

PARSER_VERSION = "v2-bounded-recovery-4"


class ParserMode(StrEnum):
    """Supported policies for decoding model responses."""

    STRICT = "strict"
    RECOVERY = "recovery"


@dataclass(frozen=True)
class _FormatSpec:
    schema: Literal["array", "flat", "grouped"] = "array"
    label_key: str = "label"
    box_key: str = "box_2d"
    normalized_maximum: int | None = None
    y_first: bool = False
    resized: bool = False


_FORMATS = {
    DetectionCoordinateFormat.YXYX_NORMALIZED_0_TO_1000: _FormatSpec(
        normalized_maximum=1000, y_first=True
    ),
    DetectionCoordinateFormat.XYXY_NORMALIZED_0_TO_1000: _FormatSpec(
        normalized_maximum=1000
    ),
    DetectionCoordinateFormat.XYXY_NORMALIZED_0_TO_999: _FormatSpec(
        normalized_maximum=999
    ),
    DetectionCoordinateFormat.XYXY_NORMALIZED_0_TO_100: _FormatSpec(
        normalized_maximum=100
    ),
    DetectionCoordinateFormat.XYXY_NORMALIZED_0_TO_1000_META_FLAT: _FormatSpec(
        schema="flat", normalized_maximum=1000
    ),
    DetectionCoordinateFormat.XYXY_NORMALIZED_0_TO_1000_META_BBOX: _FormatSpec(
        schema="grouped",
        label_key="object_name",
        box_key="bbox",
        normalized_maximum=1000,
    ),
    DetectionCoordinateFormat.XYXY_ABSOLUTE_RESIZED_IMAGE: _FormatSpec(resized=True),
    DetectionCoordinateFormat.XYXY_ABSOLUTE_RESIZED_IMAGE_BBOX: _FormatSpec(
        box_key="bbox", resized=True
    ),
    DetectionCoordinateFormat.XYXY_ABSOLUTE_ORIGINAL_IMAGE: _FormatSpec(),
    DetectionCoordinateFormat.YXYX_ABSOLUTE_ORIGINAL_IMAGE: _FormatSpec(y_first=True),
}


def effective_format(
    coordinate_format: DetectionCoordinateFormat | str, prompt: str
) -> DetectionCoordinateFormat:
    """Resolve historical pixel-box keys from the actual recorded prompt."""
    value = DetectionCoordinateFormat(coordinate_format)
    # Historical string-valued profiles used box_2d despite the bbox suffix.
    if (
        value == DetectionCoordinateFormat.XYXY_ABSOLUTE_RESIZED_IMAGE_BBOX
        and 'key "box_2d"' in prompt
    ):
        return DetectionCoordinateFormat.XYXY_ABSOLUTE_RESIZED_IMAGE
    return value


def strict_format_error(
    prediction: str, coordinate_format: DetectionCoordinateFormat | str
) -> str | None:
    """Validate the requested schema without repairing entries or filtering labels.

    Extra metadata is allowed, but conflicting aliases are ambiguous. Normalized
    coordinates must be integers except for the percentage format.
    """
    spec = _FORMATS[DetectionCoordinateFormat(coordinate_format)]
    try:
        entries = _json(prediction)
    except (ValueError, TypeError):
        return "invalid_json"
    if not isinstance(entries, list):
        return "not_array"
    notes: Counter[str] = Counter()
    for entry in entries:
        try:
            _, boxes = _entry_boxes(entry, spec, ParserMode.STRICT, notes)
            for box in boxes:
                _coordinates(box, spec, ParserMode.STRICT, notes)
        except ValueError as error:
            return str(error)
    return None


def _json(text: str) -> Any:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate_key")
            result[key] = value
        return result

    def invalid_constant(value: str) -> Any:
        raise ValueError("invalid_json_constant")

    return json.loads(text, object_pairs_hook=unique, parse_constant=invalid_constant)


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


def _entries(text: str, mode: ParserMode, notes: Counter[str]) -> list[Any]:
    try:
        value = _json(text)
        if isinstance(value, list):
            return value
        if mode == ParserMode.RECOVERY and isinstance(value, dict):
            notes["single_object"] += 1
            return [value]
        return []
    except ValueError:
        if mode == ParserMode.STRICT:
            return []
    # Only one payload: do not select among competing arrays or code blocks.
    stripped = text.strip()
    fence = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\s*```", stripped, re.S | re.I)
    if fence:
        stripped = fence.group(1).strip()
        notes["markdown_wrapper"] += 1
    elif stripped.endswith("```") and stripped.count("```") == 1:
        # A closing fence alone is harmless only after one complete JSON array.
        try:
            value = _json(stripped[:-3].strip())
        except ValueError:
            value = None
        if isinstance(value, list):
            notes["orphan_closing_fence"] += 1
            return value
        notes["ambiguous_payload"] += 1
        return []
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
            if any(c in stripped[end:] for c in "[]{}"):
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
    mode: ParserMode,
    notes: Counter[str],
    missing_error: str = "missing_field",
) -> Any:
    found = [name for name in aliases if name in entry]
    if len(found) > 1 and any(entry[name] != entry[found[0]] for name in found[1:]):
        raise ValueError("conflicting_fields")
    if key in entry:
        return entry[key]
    if mode == ParserMode.RECOVERY and found:
        notes["field_alias"] += 1
        return entry[found[0]]
    raise ValueError(missing_error)


def _number(value: Any, mode: ParserMode, notes: Counter[str]) -> float:
    if type(value) is str and mode == ParserMode.RECOVERY:
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


def _entry_boxes(
    entry: Any, spec: _FormatSpec, mode: ParserMode, notes: Counter[str]
) -> tuple[str, list[Any]]:
    if not isinstance(entry, dict):
        raise ValueError("invalid_object")
    label = _field(
        entry,
        spec.label_key,
        ("label", "object_name", "description"),
        mode,
        notes,
        "missing_label",
    )
    if not isinstance(label, str):
        raise ValueError("missing_label")
    if spec.schema == "flat":
        return label, [entry]
    raw = _field(
        entry, spec.box_key, ("box_2d", "bbox_2d", "bbox"), mode, notes, "invalid_box"
    )
    if spec.schema == "array":
        return label, [raw]
    if isinstance(raw, dict) and mode == ParserMode.RECOVERY:
        notes["singleton_box_container"] += 1
        raw = [raw]
    if not isinstance(raw, list) or not raw:
        raise ValueError("invalid_box")
    return label, raw


def _coordinates(
    box: Any, spec: _FormatSpec, mode: ParserMode, notes: Counter[str]
) -> list[float]:
    if spec.schema != "array":
        keys = ("x_min", "y_min", "x_max", "y_max")
        if not isinstance(box, dict) or not all(key in box for key in keys):
            raise ValueError("invalid_box")
        box = [box[key] for key in keys]
    if not isinstance(box, list) or len(box) != 4:
        raise ValueError("invalid_box")
    values = [_number(value, mode, notes) for value in box]
    if spec.normalized_maximum not in (None, 100):
        if mode == ParserMode.STRICT and any(type(value) is not int for value in box):
            raise ValueError("non_integer_coordinate")
        if mode == ParserMode.RECOVERY:
            count = sum(type(value) is float for value in box)
            if count:
                notes["non_integer_coordinate"] += count
    return values


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
    coordinate_format: DetectionCoordinateFormat | str,
    uploaded_wh: tuple[int, int],
    mode: ParserMode | str = ParserMode.RECOVERY,
    diagnostics: dict[str, Any] | None = None,
) -> Detections:
    """Decode using bounded, model-independent recovery and explicit geometry.

    No nested-object searches, schema reconstruction, coordinate guessing,
    clipping, corner reordering, deduplication or point-to-box conversion.
    ``classes`` seeds IDs and must contain unique labels; it does not filter
    predictions. Unknown labels receive subsequent IDs. Decoded labels are
    returned in ``data["class_name"]``.
    Diagnostics contain counts of operations, not an assertion of correctness.
    """
    mode = ParserMode(mode)
    if len(set(classes)) != len(classes):
        raise ValueError("classes must contain unique labels")
    fmt = DetectionCoordinateFormat(coordinate_format)
    spec = _FORMATS[fmt]
    notes: Counter[str] = Counter()
    if diagnostics is not None:
        diagnostics.pop("rejected", None)
        diagnostics.update(version=PARSER_VERSION, mode=mode.value, operations={})
    if mode == ParserMode.STRICT:
        error = strict_format_error(prediction, fmt)
        if error is not None:
            if diagnostics is not None:
                diagnostics["rejected"] = error
            return Detections.empty()
    entries = _entries(prediction, mode, notes)
    boxes_out: list[list[float]] = []
    ids: list[int] = []
    confidences: list[float] = []
    labels_out: list[str] = []
    class_index = {label: index for index, label in enumerate(classes)}
    w, h = resolution_wh
    if spec.normalized_maximum is not None:
        maximum = spec.normalized_maximum
        sx, sy = w / maximum, h / maximum
    elif spec.resized:
        sx, sy = w / uploaded_wh[0], h / uploaded_wh[1]
    else:
        sx = sy = 1.0
    failed = False
    for entry in entries:
        try:
            label, boxes = _entry_boxes(entry, spec, mode, notes)
            for box in boxes:
                try:
                    first, second, third, fourth = _coordinates(box, spec, mode, notes)
                    if spec.y_first:
                        x1, y1, x2, y2 = second, first, fourth, third
                    else:
                        x1, y1, x2, y2 = first, second, third, fourth
                    scaled = [x1 * sx, y1 * sy, x2 * sx, y2 * sy]
                    if not all(math.isfinite(v) for v in scaled):
                        raise ValueError("invalid_coordinate")
                    boxes_out.append(scaled)
                    ids.append(class_index.setdefault(label, len(class_index)))
                    labels_out.append(label)
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
    if (mode == ParserMode.STRICT and failed) or not boxes_out:
        return Detections.empty()
    return Detections(
        xyxy=np.asarray(boxes_out, dtype=float),
        class_id=np.asarray(ids, dtype=int),
        confidence=np.asarray(confidences, dtype=float),
        data={"class_name": np.asarray(labels_out, dtype=object)},
    )
