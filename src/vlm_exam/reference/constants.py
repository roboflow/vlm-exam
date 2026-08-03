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

REFERENCE_EFFORT = "reference"
"""Effort label written into reference-run JSONL files."""

CANONICAL_REFERENCE_RUNS: tuple[tuple[str, str, str], ...] = (
    (
        "sam3",
        "class-names",
        "reference/results/detection_sam3_reference_20260803_093736.jsonl",
    ),
    (
        "sam3",
        "v1",
        "reference/results/detection_sam3_reference_20260803_095800.jsonl",
    ),
    (
        "sam3",
        "v2-none",
        "reference/results/detection_sam3_reference_20260803_101805.jsonl",
    ),
    (
        "sam3",
        "v2-overlay",
        "reference/results/detection_sam3_reference_20260803_104113.jsonl",
    ),
    (
        "yoloe-11l-seg",
        "class-names",
        "reference/results/detection_yoloe-11l-seg_reference_20260803_090104.jsonl",
    ),
    (
        "yoloe-11l-seg",
        "v1",
        "reference/results/detection_yoloe-11l-seg_reference_20260803_090240.jsonl",
    ),
    (
        "yoloe-11l-seg",
        "v2-none",
        "reference/results/detection_yoloe-11l-seg_reference_20260803_090418.jsonl",
    ),
    (
        "yoloe-11l-seg",
        "v2-overlay",
        "reference/results/detection_yoloe-11l-seg_reference_20260803_090556.jsonl",
    ),
    (
        "yoloe-26x-seg",
        "class-names",
        "reference/results/detection_yoloe-26x-seg_reference_20260803_092841.jsonl",
    ),
    (
        "yoloe-26x-seg",
        "v1",
        "reference/results/detection_yoloe-26x-seg_reference_20260803_093010.jsonl",
    ),
    (
        "yoloe-26x-seg",
        "v2-none",
        "reference/results/detection_yoloe-26x-seg_reference_20260803_093140.jsonl",
    ),
    (
        "yoloe-26x-seg",
        "v2-overlay",
        "reference/results/detection_yoloe-26x-seg_reference_20260803_093309.jsonl",
    ),
)
"""Canonical full-dataset reference runs by model and prompt mode."""
