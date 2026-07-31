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

CANONICAL_BEST_PAIRS: tuple[tuple[str, str, str], ...] = (
    (
        "yoloe-11l-seg",
        "reference/results/detection_yoloe-11l-seg_reference_20260729_224530.jsonl",
        "reference/results/detection_yoloe-11l-seg_reference_20260729_224941.jsonl",
    ),
    (
        "yoloe-26x-seg",
        "reference/results/detection_yoloe-26x-seg_reference_20260729_224745.jsonl",
        "reference/results/detection_yoloe-26x-seg_reference_20260729_225750.jsonl",
    ),
    (
        "sam3",
        "reference/results/detection_sam3_reference_20260730_103753.jsonl",
        "reference/results/detection_sam3_reference_20260730_121633.jsonl",
    ),
)
"""Canonical baseline and image-conditioned reference runs for 250-image eval."""
