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
from pathlib import Path

import numpy as np
import pytest
import supervision as sv

from vlm_exam.reference.analysis import (
    build_reference_analysis_report,
    object_count_bucket,
)
from vlm_exam.reference.base import ReferencePrediction
from vlm_exam.reference.prompts import (
    load_image_conditioned_prompt_set,
    validate_prompt_set_coverage,
)
from vlm_exam.reference.serializer import serialize_reference_prediction
from vlm_exam.results import RunResult, SampleResult
from vlm_exam.tasks.detection import (
    DetectionCoordinateFormat,
    DetectionSample,
    build_sample_index,
    parse_prediction,
)


class TestPromptAssets:
    def test_image_conditioned_resolution(self, tmp_path: Path) -> None:
        jsonl_path = tmp_path / "prompts.jsonl"
        jsonl_path.write_text(
            json.dumps(
                {
                    "image": "a.jpg",
                    "class_name": "potato",
                    "primary": "brown oval potato",
                    "variants": ["brown potato tuber"],
                    "generation_model": "gemini-3.5-flash",
                    "generation_prompt_version": "image_conditioned_v1",
                    "generated_at": "2026-07-29T00:00:00Z",
                }
            )
            + "\n"
        )
        loaded = load_image_conditioned_prompt_set(jsonl_path)
        assert loaded.image_conditioned is not None
        assert (
            loaded.image_conditioned.resolve("a.jpg", "potato") == "brown oval potato"
        )

    def test_duplicate_image_prompt_text_is_rejected(self, tmp_path: Path) -> None:
        jsonl_path = tmp_path / "prompts.jsonl"
        rows = [
            {
                "image": "a.jpg",
                "class_name": "one cent coin",
                "primary": "small copper coin",
            },
            {
                "image": "a.jpg",
                "class_name": "two cent coin",
                "primary": "small copper coin",
            },
        ]
        jsonl_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

        with pytest.raises(ValueError, match="Duplicate prompt"):
            load_image_conditioned_prompt_set(jsonl_path)

    def test_label_remap_round_trip(self) -> None:
        prediction = ReferencePrediction(
            boxes_xyxy=np.array([[10.0, 20.0, 50.0, 60.0]], dtype=np.float32),
            labels=("domestic cat",),
            confidences=np.array([0.91], dtype=np.float32),
        )
        serialized = serialize_reference_prediction(
            prediction,
            label_remap={"domestic cat": "cat"},
        )
        detections = parse_prediction(
            serialized,
            (100, 100),
            ["cat"],
            coordinate_format=DetectionCoordinateFormat.XYXY_ABSOLUTE_ORIGINAL_IMAGE,
        )
        assert detections.data["class_name"][0] == "cat"

    def test_coverage_validation(self, tmp_path: Path) -> None:
        jsonl_path = tmp_path / "prompts.jsonl"
        jsonl_path.write_text(
            json.dumps(
                {
                    "image": "a.jpg",
                    "class_name": "cat",
                    "primary": "striped house cat",
                    "variants": [],
                }
            )
            + "\n"
        )
        loaded = load_image_conditioned_prompt_set(jsonl_path)
        errors = validate_prompt_set_coverage(
            loaded,
            all_classes=("cat", "dog"),
            required_pairs=[("a.jpg", "cat"), ("a.jpg", "dog")],
        )
        assert errors


class TestReferenceAnalysis:
    def test_object_count_bucket(self) -> None:
        assert object_count_bucket(1).value == "1-2"
        assert object_count_bucket(41).value == "41+"

    def test_build_reference_analysis_report(self) -> None:
        classes = ("cat", "dog")
        ground_truth = sv.Detections(
            xyxy=np.array([[10, 10, 30, 30], [50, 50, 70, 70]], dtype=np.float32),
            class_id=np.array([0, 1], dtype=int),
        )
        sample = DetectionSample(
            image_path="/tmp/a.jpg",
            image_width=100,
            image_height=100,
            classes=classes,
            ground_truth=ground_truth,
        )
        sample_index = build_sample_index([sample])
        run = RunResult(
            model="yoloe-11l-seg",
            effort="reference",
            task="detection",
            timestamp="20260729_120000",
            samples=[
                SampleResult(
                    index=0,
                    image="a.jpg",
                    expected="",
                    predicted=(
                        '[{"box_2d": [10, 10, 30, 30], "label": "cat",'
                        ' "confidence": 0.9},'
                        ' {"box_2d": [50, 50, 70, 70], "label": "cat",'
                        ' "confidence": 0.8}]'
                    ),
                    correct=False,
                    input_tokens=0,
                    output_tokens=0,
                    metadata={
                        "coordinate_format": "xyxy_absolute_original_image",
                        "reference": True,
                    },
                )
            ],
        )
        report = build_reference_analysis_report(run, sample_index)
        assert report.map50 >= 0.0
        assert len(report.per_class) == 2
        assert report.recall_class_agnostic >= report.recall_class_aware
