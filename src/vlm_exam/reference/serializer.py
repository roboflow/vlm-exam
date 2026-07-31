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

from vlm_exam.reference.base import ReferencePrediction


def serialize_reference_prediction(
    prediction: ReferencePrediction,
    *,
    label_remap: dict[str, str] | None = None,
) -> str:
    """Serialize structured detections into benchmark JSON.

    Args:
        prediction: Adapter output with boxes in original-image pixels.
        label_remap: Optional mapping from prompt text back to canonical labels.

    Returns:
        JSON list string compatible with ``parse_prediction``.
    """
    entries: list[dict[str, object]] = []
    for index, label in enumerate(prediction.labels):
        x_min, y_min, x_max, y_max = (
            float(value) for value in prediction.boxes_xyxy[index]
        )
        canonical_label = label_remap.get(label, label) if label_remap else label
        entry: dict[str, object] = {
            "box_2d": [x_min, y_min, x_max, y_max],
            "label": canonical_label,
        }
        if prediction.confidences is not None:
            entry["confidence"] = round(float(prediction.confidences[index]), 6)
        entries.append(entry)
    return json.dumps(entries)
