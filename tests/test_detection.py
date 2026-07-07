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

from vlm_exam.tasks.detection import (
    DetectionSample,
    DetectionTask,
    build_sample_index,
    parse_prediction,
)

_COCO = {
    "categories": [
        {"id": 0, "name": "vlm-exam", "supercategory": "none"},
        {"id": 1, "name": "cat", "supercategory": "vlm-exam"},
        {"id": 2, "name": "dog", "supercategory": "vlm-exam"},
    ],
    "images": [
        {"id": 10, "file_name": "first.jpg", "width": 100, "height": 200},
        {"id": 11, "file_name": "second.jpg", "width": 50, "height": 50},
    ],
    "annotations": [
        {"id": 1, "image_id": 10, "category_id": 1, "bbox": [10, 20, 30, 40]},
        {"id": 2, "image_id": 10, "category_id": 2, "bbox": [0, 0, 5, 5]},
    ],
}


@pytest.fixture()
def dataset_directory(tmp_path: Path) -> str:
    annotations_path = tmp_path / "_annotations.coco.json"
    annotations_path.write_text(json.dumps(_COCO))
    return str(tmp_path)


def _make_sample(
    ground_truth: sv.Detections,
    classes: tuple[str, ...] = ("cat", "dog"),
) -> DetectionSample:
    return DetectionSample(
        image_path="/data/image.jpg",
        image_width=100,
        image_height=100,
        ground_truth=ground_truth,
        classes=classes,
    )


def _detections(xyxy: list[list[float]], class_ids: list[int]) -> sv.Detections:
    return sv.Detections(
        xyxy=np.array(xyxy, dtype=np.float32),
        class_id=np.array(class_ids, dtype=int),
    )


class TestLoadSamples:
    def test_skips_placeholder_category(self, dataset_directory: str) -> None:
        task = DetectionTask()
        task.load_samples(dataset_directory)
        assert task.classes == ["cat", "dog"]

    def test_converts_bbox_xywh_to_xyxy(self, dataset_directory: str) -> None:
        task = DetectionTask()
        samples = task.load_samples(dataset_directory)
        sample = build_sample_index(samples)["first.jpg"]
        np.testing.assert_allclose(
            sample.ground_truth.xyxy,
            np.array([[10, 20, 40, 60], [0, 0, 5, 5]], dtype=np.float32),
        )
        np.testing.assert_array_equal(sample.ground_truth.class_id, [0, 1])

    def test_image_without_annotations_has_empty_ground_truth(
        self, dataset_directory: str
    ) -> None:
        task = DetectionTask()
        samples = task.load_samples(dataset_directory)
        sample = build_sample_index(samples)["second.jpg"]
        assert len(sample.ground_truth) == 0

    def test_sample_dimensions(self, dataset_directory: str) -> None:
        task = DetectionTask()
        samples = task.load_samples(dataset_directory)
        sample = build_sample_index(samples)["first.jpg"]
        assert sample.image_width == 100
        assert sample.image_height == 200


class TestBuildPrompt:
    def test_image_mode_lists_only_present_classes(self) -> None:
        task = DetectionTask(prompt_classes="image")
        sample = _make_sample(_detections([[0, 0, 10, 10]], [1]))
        prompt = task.build_prompt(sample)
        assert "dog" in prompt
        assert "cat" not in prompt

    def test_all_mode_lists_every_class(self) -> None:
        task = DetectionTask(prompt_classes="all")
        sample = _make_sample(_detections([[0, 0, 10, 10]], [1]))
        prompt = task.build_prompt(sample)
        assert "cat" in prompt
        assert "dog" in prompt

    def test_image_mode_falls_back_to_all_classes_when_no_ground_truth(self) -> None:
        task = DetectionTask(prompt_classes="image")
        sample = _make_sample(sv.Detections.empty())
        prompt = task.build_prompt(sample)
        assert "cat" in prompt
        assert "dog" in prompt

    def test_invalid_mode_raises(self) -> None:
        with pytest.raises(ValueError):
            DetectionTask(prompt_classes="bogus")


class TestParsePrediction:
    def test_parses_fenced_json(self) -> None:
        prediction = '```json\n[{"box_2d": [100, 200, 300, 400], "label": "cat"}]\n```'
        detections = parse_prediction(prediction, (1000, 1000), ["cat", "dog"])
        assert len(detections) == 1
        assert detections.class_id is not None
        assert detections.class_id[0] == 0
        # box_2d is [y_min, x_min, y_max, x_max] normalized to 1000
        np.testing.assert_allclose(detections.xyxy[0], [200, 100, 400, 300])

    def test_malformed_json_returns_empty(self) -> None:
        detections = parse_prediction("not json at all", (100, 100), ["cat"])
        assert len(detections) == 0

    def test_unknown_labels_are_filtered(self) -> None:
        prediction = '```json\n[{"box_2d": [0, 0, 100, 100], "label": "unicorn"}]\n```'
        detections = parse_prediction(prediction, (100, 100), ["cat", "dog"])
        assert len(detections) == 0


class TestEvaluate:
    def test_empty_ground_truth_and_empty_prediction_is_correct(self) -> None:
        task = DetectionTask()
        sample = _make_sample(sv.Detections.empty())
        result = task.evaluate(sample, "[]")
        assert result.correct is True
        assert result.details is not None
        assert result.details["map50"] == 1.0

    def test_empty_prediction_with_ground_truth_is_incorrect(self) -> None:
        task = DetectionTask()
        sample = _make_sample(_detections([[0, 0, 10, 10]], [0]))
        result = task.evaluate(sample, "[]")
        assert result.correct is False
        assert result.details is not None
        assert result.details["map50"] == 0.0

    def test_prediction_with_empty_ground_truth_is_incorrect(self) -> None:
        task = DetectionTask()
        sample = _make_sample(sv.Detections.empty())
        prediction = '[{"box_2d": [0, 0, 100, 100], "label": "cat"}]'
        result = task.evaluate(sample, prediction)
        assert result.correct is False

    def test_perfect_prediction_scores_high(self) -> None:
        task = DetectionTask()
        sample = _make_sample(_detections([[10, 10, 50, 50]], [0]))
        # normalized to 1000 over a 100x100 image: [y1, x1, y2, x2]
        prediction = '[{"box_2d": [100, 100, 500, 500], "label": "cat"}]'
        result = task.evaluate(sample, prediction)
        assert result.correct is True
        assert result.details is not None
        assert result.details["map50"] > 0.99

    def test_details_include_prompt_classes_mode(self) -> None:
        task = DetectionTask(prompt_classes="all")
        sample = _make_sample(sv.Detections.empty())
        result = task.evaluate(sample, "[]")
        assert result.details is not None
        assert result.details["prompt_classes"] == "all"


class TestBuildSampleIndex:
    def test_indexes_by_basename(self) -> None:
        sample = _make_sample(sv.Detections.empty())
        index = build_sample_index([sample])
        assert index["image.jpg"] is sample
