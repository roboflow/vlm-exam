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

import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import supervision as sv
from PIL import Image

from vlm_exam import detection_v2_inference as inference
from vlm_exam import detection_v2_runner as runner
from vlm_exam.detection_v2 import (
    _record_problems,
    load_release,
    read_records,
    validate_release,
)
from vlm_exam.detection_v2_inference import _sample
from vlm_exam.detection_v2_publication import publish_recoveries, publish_run
from vlm_exam.providers import base
from vlm_exam.providers.anthropic import compute_resize_dimensions
from vlm_exam.results import RunResult, SampleResult
from vlm_exam.tasks.detection import (
    MAP_PASS_THRESHOLD,
    DetectionCoordinateFormat,
    DetectionSample,
    DetectionTask,
    build_sample_index,
    compute_dataset_map,
    compute_image_map50,
    filter_prediction_json,
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

    def test_prompt_defines_coordinate_convention(self) -> None:
        task = DetectionTask()
        sample = _make_sample(_detections([[0, 0, 10, 10]], [0]))
        prompt = task.build_prompt(sample)
        assert "[y_min, x_min, y_max, x_max]" in prompt
        assert "0 and 1000" in prompt

    def test_pixel_format_prompt_states_dimensions(self) -> None:
        task = DetectionTask(
            coordinate_format=DetectionCoordinateFormat.XYXY_ABSOLUTE_RESIZED_IMAGE
        )
        sample = _make_sample(_detections([[0, 0, 10, 10]], [0]))
        prompt = task.build_prompt(sample, uploaded_size=(100, 100))
        assert "[x_min, y_min, x_max, y_max]" in prompt
        assert "pixel coordinates" in prompt
        assert "100x100 pixel image" in prompt

    def test_pixel_format_prompt_states_uploaded_dimensions(self) -> None:
        task = DetectionTask(
            coordinate_format=DetectionCoordinateFormat.XYXY_ABSOLUTE_RESIZED_IMAGE
        )
        sample = DetectionSample(
            image_path="/data/large.jpg",
            image_width=4000,
            image_height=3000,
            ground_truth=_detections([[0, 0, 10, 10]], [0]),
            classes=("cat", "dog"),
        )
        uploaded = compute_resize_dimensions(4000, 3000)
        prompt = task.build_prompt(sample, uploaded_size=uploaded)
        assert f"{uploaded[0]}x{uploaded[1]} pixel image" in prompt

    def test_pixel_format_prompt_requires_uploaded_size(self) -> None:
        task = DetectionTask(
            coordinate_format=DetectionCoordinateFormat.XYXY_ABSOLUTE_RESIZED_IMAGE
        )
        sample = _make_sample(_detections([[0, 0, 10, 10]], [0]))
        with pytest.raises(ValueError, match="uploaded_size"):
            task.build_prompt(sample)

    def test_resized_bbox_prompt_uses_bbox_key(self) -> None:
        task = DetectionTask(
            coordinate_format=DetectionCoordinateFormat.XYXY_ABSOLUTE_RESIZED_IMAGE_BBOX
        )
        sample = _make_sample(_detections([[0, 0, 10, 10]], [0]))
        prompt = task.build_prompt(sample, uploaded_size=(100, 80))
        assert '"bbox"' in prompt
        assert "100x80 pixel image" in prompt
        assert "box_2d" not in prompt

    def test_invalid_coordinate_format_raises(self) -> None:
        with pytest.raises(ValueError):
            DetectionCoordinateFormat("bogus")

    def test_meta_flat_prompt_uses_named_coordinate_fields(self) -> None:
        task = DetectionTask(
            coordinate_format=(
                DetectionCoordinateFormat.XYXY_NORMALIZED_0_TO_1000_META_FLAT
            )
        )
        sample = _make_sample(_detections([[0, 0, 10, 10]], [0]))
        prompt = task.build_prompt(sample)
        assert "object grounding expert" in prompt
        assert '"x_min"' in prompt
        assert "0-1000 range" in prompt

    def test_meta_bbox_prompt_uses_official_object_name_schema(self) -> None:
        task = DetectionTask(
            coordinate_format=(
                DetectionCoordinateFormat.XYXY_NORMALIZED_0_TO_1000_META_BBOX
            )
        )
        sample = _make_sample(_detections([[0, 0, 10, 10]], [0]))
        prompt = task.build_prompt(sample)
        assert "object grounding expert" in prompt
        assert '"object_name"' in prompt
        assert '"bbox"' in prompt
        assert '"cat"' in prompt
        assert "0-1000 range" in prompt


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

    def test_parses_meta_flat_normalized_coordinates(self) -> None:
        prediction = (
            '[{"label": "cat", "x_min": 100, "y_min": 200, "x_max": 300, "y_max": 400}]'
        )
        detections = parse_prediction(
            prediction,
            (1000, 1000),
            ["cat", "dog"],
            coordinate_format=(
                DetectionCoordinateFormat.XYXY_NORMALIZED_0_TO_1000_META_FLAT
            ),
        )
        assert len(detections) == 1
        np.testing.assert_allclose(detections.xyxy[0], [100, 200, 300, 400])

    def test_parses_meta_bbox_normalized_coordinates(self) -> None:
        prediction = (
            '[{"object_name": "cat", "bbox": '
            '[{"x_min": 100, "y_min": 200, "x_max": 300, "y_max": 400}]}]'
        )
        detections = parse_prediction(
            prediction,
            (1000, 1000),
            ["cat", "dog"],
            coordinate_format=(
                DetectionCoordinateFormat.XYXY_NORMALIZED_0_TO_1000_META_BBOX
            ),
        )
        assert len(detections) == 1
        np.testing.assert_allclose(detections.xyxy[0], [100, 200, 300, 400])

    def test_parses_multiple_boxes_under_one_meta_bbox_object(self) -> None:
        prediction = (
            '[{"object_name": "cat", "bbox": ['
            '{"x_min": 0, "y_min": 0, "x_max": 10, "y_max": 10}, '
            '{"x_min": 20, "y_min": 20, "x_max": 30, "y_max": 30}]}]'
        )
        detections = parse_prediction(
            prediction,
            (1000, 1000),
            ["cat"],
            coordinate_format=(
                DetectionCoordinateFormat.XYXY_NORMALIZED_0_TO_1000_META_BBOX
            ),
        )
        assert len(detections) == 2

    def test_meta_bbox_missing_bbox_returns_empty(self) -> None:
        prediction = '[{"object_name": "cat"}]'
        detections = parse_prediction(
            prediction,
            (1000, 1000),
            ["cat"],
            coordinate_format=(
                DetectionCoordinateFormat.XYXY_NORMALIZED_0_TO_1000_META_BBOX
            ),
        )
        assert len(detections) == 0

    def test_meta_flat_missing_fields_returns_empty(self) -> None:
        prediction = '[{"label": "cat", "x_min": 100, "y_min": 200}]'
        detections = parse_prediction(
            prediction,
            (1000, 1000),
            ["cat"],
            coordinate_format=(
                DetectionCoordinateFormat.XYXY_NORMALIZED_0_TO_1000_META_FLAT
            ),
        )
        assert len(detections) == 0

    def test_unknown_labels_are_filtered(self) -> None:
        prediction = '```json\n[{"box_2d": [0, 0, 100, 100], "label": "unicorn"}]\n```'
        detections = parse_prediction(prediction, (100, 100), ["cat", "dog"])
        assert len(detections) == 0

    @pytest.mark.parametrize(
        ("coordinate_format", "prediction"),
        [
            (
                DetectionCoordinateFormat.XYXY_NORMALIZED_0_TO_1000,
                '[{"bbox_2d": [0, 0, 100, 100], "label": ["cat", "dog"]}]',
            ),
            (
                DetectionCoordinateFormat.XYXY_ABSOLUTE_ORIGINAL_IMAGE,
                '[{"bbox": [0, 0, 100, 100], "label": ["cat"]}]',
            ),
            (
                DetectionCoordinateFormat.XYXY_NORMALIZED_0_TO_100,
                '[{"bbox": [0, 0, 10, 10], "label": ["cat"]}]',
            ),
            (
                DetectionCoordinateFormat.XYXY_NORMALIZED_0_TO_1000_META_FLAT,
                '[{"label": ["cat"], "x_min": 0, "y_min": 0, "x_max": 1, "y_max": 1}]',
            ),
            (
                DetectionCoordinateFormat.XYXY_NORMALIZED_0_TO_1000_META_BBOX,
                '[{"object_name": ["cat"], "bbox": '
                '[{"x_min": 0, "y_min": 0, "x_max": 1, "y_max": 1}]}]',
            ),
        ],
    )
    def test_non_string_labels_are_skipped(
        self, coordinate_format: DetectionCoordinateFormat, prediction: str
    ) -> None:
        detections = parse_prediction(
            prediction, (1000, 1000), ["cat", "dog"], coordinate_format
        )
        assert len(detections) == 0

    def test_parses_prose_wrapped_json(self) -> None:
        prediction = (
            "Here are the detected objects:\n"
            '[{"box_2d": [100, 200, 300, 400], "label": "cat"}]\n'
            "Let me know if you need anything else."
        )
        detections = parse_prediction(prediction, (1000, 1000), ["cat", "dog"])
        assert len(detections) == 1
        np.testing.assert_allclose(detections.xyxy[0], [200, 100, 400, 300])

    def test_parses_bare_json_without_fences(self) -> None:
        prediction = '[{"box_2d": [0, 0, 500, 500], "label": "dog"}]'
        detections = parse_prediction(prediction, (100, 100), ["cat", "dog"])
        assert len(detections) == 1
        assert detections.class_id is not None
        assert detections.class_id[0] == 1


class TestParseObjectScanFallback:
    def test_parses_newline_delimited_meta_flat_objects(self) -> None:
        prediction = (
            '{"label": "cat", "x_min": 100, "y_min": 200, "x_max": 300, "y_max": 400}\n'
            '{"label": "dog", "x_min": 0, "y_min": 0, "x_max": 50, "y_max": 50}'
        )
        detections = parse_prediction(
            prediction,
            (1000, 1000),
            ["cat", "dog"],
            coordinate_format=(
                DetectionCoordinateFormat.XYXY_NORMALIZED_0_TO_1000_META_FLAT
            ),
        )
        assert len(detections) == 2
        np.testing.assert_allclose(detections.xyxy[0], [100, 200, 300, 400])

    def test_parses_comma_separated_meta_flat_objects(self) -> None:
        prediction = (
            '{"label": "cat", "x_min": 100, "y_min": 200, "x_max": 300, "y_max": 400}, '
            '{"label": "dog", "x_min": 0, "y_min": 0, "x_max": 50, "y_max": 50}'
        )
        detections = parse_prediction(
            prediction,
            (1000, 1000),
            ["cat", "dog"],
            coordinate_format=(
                DetectionCoordinateFormat.XYXY_NORMALIZED_0_TO_1000_META_FLAT
            ),
        )
        assert len(detections) == 2

    def test_parses_single_meta_flat_object_without_array(self) -> None:
        prediction = (
            '{"label": "cat", "x_min": 100, "y_min": 200, "x_max": 300, "y_max": 400}'
        )
        detections = parse_prediction(
            prediction,
            (1000, 1000),
            ["cat"],
            coordinate_format=(
                DetectionCoordinateFormat.XYXY_NORMALIZED_0_TO_1000_META_FLAT
            ),
        )
        assert len(detections) == 1
        np.testing.assert_allclose(detections.xyxy[0], [100, 200, 300, 400])

    def test_parses_bare_objects_for_normalized_xyxy_format(self) -> None:
        prediction = (
            '{"box_2d": [100, 200, 300, 400], "label": "cat"}\n'
            '{"box_2d": [0, 0, 100, 100], "label": "dog"}'
        )
        detections = parse_prediction(
            prediction,
            (1000, 1000),
            ["cat", "dog"],
            coordinate_format=DetectionCoordinateFormat.XYXY_NORMALIZED_0_TO_1000,
        )
        assert len(detections) == 2
        np.testing.assert_allclose(detections.xyxy[0], [100, 200, 300, 400])

    def test_object_scan_keeps_format_validation(self) -> None:
        prediction = '{"label": "cat", "x_min": 100, "y_min": 200}'
        detections = parse_prediction(
            prediction,
            (1000, 1000),
            ["cat"],
            coordinate_format=(
                DetectionCoordinateFormat.XYXY_NORMALIZED_0_TO_1000_META_FLAT
            ),
        )
        assert len(detections) == 0

    def test_object_scan_skips_default_supervision_format(self) -> None:
        prediction = '{"box_2d": [100, 200, 300, 400], "label": "cat"}'
        detections = parse_prediction(prediction, (1000, 1000), ["cat"])
        assert len(detections) == 0

    def test_garbage_with_braces_returns_empty(self) -> None:
        prediction = "the objects {are somewhere} in the image"
        detections = parse_prediction(
            prediction,
            (1000, 1000),
            ["cat"],
            coordinate_format=(
                DetectionCoordinateFormat.XYXY_NORMALIZED_0_TO_1000_META_FLAT
            ),
        )
        assert len(detections) == 0


class TestParseNormalizedPercentPrediction:
    def test_scales_xyxy_percentages_to_pixels(self) -> None:
        prediction = '[{"box_2d": [10, 20, 30, 40], "label": "cat"}]'
        detections = parse_prediction(
            prediction,
            (200, 100),
            ["cat", "dog"],
            coordinate_format=DetectionCoordinateFormat.XYXY_NORMALIZED_0_TO_100,
        )
        assert len(detections) == 1
        np.testing.assert_allclose(detections.xyxy[0], [20, 20, 60, 40])

    def test_accepts_description_label_alias(self) -> None:
        prediction = '[{"box_2d": [0, 0, 50, 50], "description": "dog"}]'
        detections = parse_prediction(
            prediction,
            (100, 100),
            ["cat", "dog"],
            coordinate_format=DetectionCoordinateFormat.XYXY_NORMALIZED_0_TO_100,
        )
        assert len(detections) == 1
        assert detections.class_id is not None
        assert detections.class_id[0] == 1

    def test_prompt_requests_percent_coordinates(self) -> None:
        task = DetectionTask(
            coordinate_format=DetectionCoordinateFormat.XYXY_NORMALIZED_0_TO_100
        )
        sample = _make_sample(_detections([[10, 10, 20, 20]], [0]))
        prompt = task.build_prompt(sample)
        assert "floats between 0 and 100" in prompt
        assert "[x_min, y_min, x_max, y_max]" in prompt


class TestParseNormalized999Prediction:
    def test_scales_the_0_to_999_grid_to_the_full_image(self) -> None:
        prediction = '[{"box_2d": [0, 0, 999, 999], "label": "cat"}]'
        detections = parse_prediction(
            prediction,
            (1998, 999),
            ["cat", "dog"],
            coordinate_format=DetectionCoordinateFormat.XYXY_NORMALIZED_0_TO_999,
        )
        assert len(detections) == 1
        np.testing.assert_allclose(detections.xyxy[0], [0, 0, 1998, 999])

    def test_prompt_requests_0_to_999_coordinates(self) -> None:
        task = DetectionTask(
            coordinate_format=DetectionCoordinateFormat.XYXY_NORMALIZED_0_TO_999
        )
        sample = _make_sample(_detections([[10, 10, 20, 20]], [0]))
        prompt = task.build_prompt(sample)
        assert "integers between 0 and 999" in prompt
        assert "[x_min, y_min, x_max, y_max]" in prompt


class TestParsePixelPrediction:
    def test_parses_pixel_coordinates_directly(self) -> None:
        prediction = '[{"box_2d": [10, 20, 30, 40], "label": "cat"}]'
        detections = parse_prediction(
            prediction,
            (100, 100),
            ["cat", "dog"],
            coordinate_format=DetectionCoordinateFormat.XYXY_ABSOLUTE_RESIZED_IMAGE,
        )
        assert len(detections) == 1
        assert detections.class_id is not None
        assert detections.class_id[0] == 0
        # box_2d is [x_min, y_min, x_max, y_max] in pixels; no resize at 100x100
        np.testing.assert_allclose(detections.xyxy[0], [10, 20, 30, 40])

    def test_parses_resized_bbox_key_and_ignores_box_2d(self) -> None:
        prediction = (
            '[{"bbox": [10, 20, 30, 40], "label": "cat"},'
            ' {"box_2d": [1, 2, 3, 4], "label": "dog"}]'
        )
        detections = parse_prediction(
            prediction,
            (200, 100),
            ["cat", "dog"],
            coordinate_format=DetectionCoordinateFormat.XYXY_ABSOLUTE_RESIZED_IMAGE_BBOX,
            uploaded_wh=(100, 50),
        )
        assert len(detections) == 1
        assert detections.class_id is not None
        assert detections.class_id[0] == 0
        np.testing.assert_allclose(detections.xyxy[0], [20, 40, 60, 80])

    def test_scales_from_uploaded_to_original_resolution(self) -> None:
        original_width, original_height = 4000, 3000
        uploaded_width, uploaded_height = compute_resize_dimensions(
            original_width, original_height
        )
        assert uploaded_width < original_width
        prediction = json.dumps(
            [
                {
                    "box_2d": [0, 0, uploaded_width, uploaded_height],
                    "label": "cat",
                }
            ]
        )
        detections = parse_prediction(
            prediction,
            (original_width, original_height),
            ["cat"],
            coordinate_format=DetectionCoordinateFormat.XYXY_ABSOLUTE_RESIZED_IMAGE,
        )
        assert len(detections) == 1
        np.testing.assert_allclose(
            detections.xyxy[0],
            [0, 0, original_width, original_height],
            rtol=1e-5,
        )

    def test_parses_fenced_pixel_json(self) -> None:
        prediction = '```json\n[{"box_2d": [10, 20, 30, 40], "label": "cat"}]\n```'
        detections = parse_prediction(
            prediction,
            (100, 100),
            ["cat"],
            coordinate_format=DetectionCoordinateFormat.XYXY_ABSOLUTE_RESIZED_IMAGE,
        )
        assert len(detections) == 1
        np.testing.assert_allclose(detections.xyxy[0], [10, 20, 30, 40])

    def test_unknown_labels_are_filtered(self) -> None:
        prediction = '[{"box_2d": [10, 20, 30, 40], "label": "unicorn"}]'
        detections = parse_prediction(
            prediction,
            (100, 100),
            ["cat"],
            coordinate_format=DetectionCoordinateFormat.XYXY_ABSOLUTE_RESIZED_IMAGE,
        )
        assert len(detections) == 0

    def test_malformed_entries_are_skipped(self) -> None:
        prediction = (
            '[{"box_2d": [10, 20, 30], "label": "cat"},'
            ' {"box_2d": [10, "a", 30, 40], "label": "cat"},'
            ' "not a dict",'
            ' {"box_2d": [10, 20, 30, 40], "label": "cat"}]'
        )
        detections = parse_prediction(
            prediction,
            (100, 100),
            ["cat"],
            coordinate_format=DetectionCoordinateFormat.XYXY_ABSOLUTE_RESIZED_IMAGE,
        )
        assert len(detections) == 1

    def test_sets_class_name_data(self) -> None:
        prediction = '[{"box_2d": [10, 20, 30, 40], "label": "dog"}]'
        detections = parse_prediction(
            prediction,
            (100, 100),
            ["cat", "dog"],
            coordinate_format=DetectionCoordinateFormat.XYXY_ABSOLUTE_RESIZED_IMAGE,
        )
        assert list(detections.data["class_name"]) == ["dog"]

    def test_clamps_out_of_bounds_coordinates_to_original(self) -> None:
        original_width, original_height = 4000, 3000
        uploaded_width, uploaded_height = compute_resize_dimensions(
            original_width, original_height
        )
        prediction = json.dumps(
            [
                {
                    "box_2d": [-50, -20, uploaded_width + 100, uploaded_height + 100],
                    "label": "cat",
                }
            ]
        )
        detections = parse_prediction(
            prediction,
            (original_width, original_height),
            ["cat"],
            coordinate_format=DetectionCoordinateFormat.XYXY_ABSOLUTE_RESIZED_IMAGE,
        )
        assert len(detections) == 1
        np.testing.assert_allclose(
            detections.xyxy[0],
            [0, 0, original_width, original_height],
            rtol=1e-5,
        )

    def test_prefers_recorded_uploaded_dimensions(self) -> None:
        prediction = '[{"box_2d": [0, 0, 100, 100], "label": "cat"}]'
        detections = parse_prediction(
            prediction,
            (400, 400),
            ["cat"],
            coordinate_format=DetectionCoordinateFormat.XYXY_ABSOLUTE_RESIZED_IMAGE,
            uploaded_wh=(100, 100),
        )
        assert len(detections) == 1
        np.testing.assert_allclose(detections.xyxy[0], [0, 0, 400, 400])

    def test_standard_tier_scales_with_smaller_upload(self) -> None:
        original_width, original_height = 4000, 3000
        standard = compute_resize_dimensions(
            original_width, original_height, 1568, 1568
        )
        high = compute_resize_dimensions(original_width, original_height)
        assert standard != high
        prediction = json.dumps(
            [{"box_2d": [0, 0, standard[0], standard[1]], "label": "cat"}]
        )
        detections = parse_prediction(
            prediction,
            (original_width, original_height),
            ["cat"],
            coordinate_format=DetectionCoordinateFormat.XYXY_ABSOLUTE_RESIZED_IMAGE,
            resolution_tier="standard",
        )
        assert len(detections) == 1
        np.testing.assert_allclose(
            detections.xyxy[0],
            [0, 0, original_width, original_height],
            rtol=1e-5,
        )


class TestParseNativePixelPrediction:
    def test_parses_pixel_native_coordinates_directly(self) -> None:
        prediction = '[{"box_2d": [10, 20, 30, 40], "label": "cat"}]'
        detections = parse_prediction(
            prediction,
            (100, 100),
            ["cat", "dog"],
            coordinate_format=DetectionCoordinateFormat.XYXY_ABSOLUTE_ORIGINAL_IMAGE,
        )
        assert len(detections) == 1
        np.testing.assert_allclose(detections.xyxy[0], [10, 20, 30, 40])

    def test_pixel_native_does_not_apply_resize_scaling(self) -> None:
        prediction = '[{"box_2d": [100, 200, 300, 400], "label": "cat"}]'
        detections = parse_prediction(
            prediction,
            (4000, 3000),
            ["cat"],
            coordinate_format=DetectionCoordinateFormat.XYXY_ABSOLUTE_ORIGINAL_IMAGE,
        )
        assert len(detections) == 1
        np.testing.assert_allclose(detections.xyxy[0], [100, 200, 300, 400])

    def test_parses_pixel_yxyx_native_coordinates(self) -> None:
        prediction = '[{"box_2d": [20, 10, 40, 30], "label": "cat"}]'
        detections = parse_prediction(
            prediction,
            (100, 100),
            ["cat"],
            coordinate_format=DetectionCoordinateFormat.YXYX_ABSOLUTE_ORIGINAL_IMAGE,
        )
        assert len(detections) == 1
        np.testing.assert_allclose(detections.xyxy[0], [10, 20, 30, 40])

    def test_pixel_native_prompt_uses_original_dimensions(self) -> None:
        task = DetectionTask(
            coordinate_format=DetectionCoordinateFormat.XYXY_ABSOLUTE_ORIGINAL_IMAGE
        )
        sample = DetectionSample(
            image_path="/data/large.jpg",
            image_width=4000,
            image_height=3000,
            ground_truth=_detections([[0, 0, 10, 10]], [0]),
            classes=("cat", "dog"),
        )
        prompt = task.build_prompt(sample)
        assert "4000x3000 pixel image" in prompt

    def test_pixel_yxyx_native_prompt_uses_yxyx_order(self) -> None:
        task = DetectionTask(
            coordinate_format=DetectionCoordinateFormat.YXYX_ABSOLUTE_ORIGINAL_IMAGE
        )
        sample = _make_sample(_detections([[0, 0, 10, 10]], [0]))
        prompt = task.build_prompt(sample)
        assert "[y_min, x_min, y_max, x_max]" in prompt
        assert "100x100 pixel image" in prompt


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

    def test_partial_match_below_threshold_is_incorrect(self) -> None:
        task = DetectionTask()
        sample = _make_sample(_detections([[10, 10, 50, 50], [60, 60, 90, 90]], [0, 0]))
        # only the first ground truth box is matched
        prediction = '[{"box_2d": [100, 100, 500, 500], "label": "cat"}]'
        result = task.evaluate(sample, prediction)
        assert result.details is not None
        assert 0.4 < result.details["map50"] < MAP_PASS_THRESHOLD
        assert result.correct is False

    def test_details_include_prompt_classes_mode(self) -> None:
        task = DetectionTask(prompt_classes="all")
        sample = _make_sample(sv.Detections.empty())
        result = task.evaluate(sample, "[]")
        assert result.details is not None
        assert result.details["prompt_classes"] == "all"

    def test_details_include_coordinate_format(self) -> None:
        task = DetectionTask(
            coordinate_format=DetectionCoordinateFormat.XYXY_ABSOLUTE_RESIZED_IMAGE
        )
        sample = _make_sample(sv.Detections.empty())
        result = task.evaluate(sample, "[]")
        assert result.details is not None
        assert result.details["coordinate_format"] == "xyxy_absolute_resized_image"

    def test_canonical_string_parses_to_enum_member(self) -> None:
        assert (
            DetectionCoordinateFormat("xyxy_absolute_resized_image")
            is DetectionCoordinateFormat.XYXY_ABSOLUTE_RESIZED_IMAGE
        )

    def test_pixel_format_perfect_prediction_scores_high(self) -> None:
        task = DetectionTask(
            coordinate_format=DetectionCoordinateFormat.XYXY_ABSOLUTE_RESIZED_IMAGE
        )
        sample = _make_sample(_detections([[10, 10, 50, 50]], [0]))
        prediction = '[{"box_2d": [10, 10, 50, 50], "label": "cat"}]'
        result = task.evaluate(sample, prediction)
        assert result.correct is True
        assert result.details is not None
        assert result.details["map50"] > 0.99


class TestBuildSampleIndex:
    def test_indexes_by_basename(self) -> None:
        sample = _make_sample(sv.Detections.empty())
        index = build_sample_index([sample])
        assert index["image.jpg"] is sample


def _make_run(image: str, predicted: str, metadata: dict | None = None) -> RunResult:
    return RunResult(
        model="test-model",
        effort="low",
        task="detection",
        timestamp="20260707_000000",
        samples=[
            SampleResult(
                index=0,
                image=image,
                expected="",
                predicted=predicted,
                correct=True,
                input_tokens=0,
                output_tokens=0,
                metadata=metadata or {},
            )
        ],
    )


class TestComputeDatasetMap:
    def test_perfect_run(self) -> None:
        sample = _make_sample(_detections([[10, 10, 50, 50]], [0]))
        index = build_sample_index([sample])
        run = _make_run(
            "image.jpg", '[{"box_2d": [100, 100, 500, 500], "label": "cat"}]'
        )
        result = compute_dataset_map(run, index)
        assert result is not None
        assert result.image_count == 1
        assert result.map50 > 0.99

    def test_no_matching_images_returns_none(self) -> None:
        sample = _make_sample(_detections([[10, 10, 50, 50]], [0]))
        index = build_sample_index([sample])
        run = _make_run("unknown.jpg", "[]")
        assert compute_dataset_map(run, index) is None

    def test_respects_pixel_coordinate_format_metadata(self) -> None:
        sample = _make_sample(_detections([[10, 10, 50, 50]], [0]))
        index = build_sample_index([sample])
        run = _make_run(
            "image.jpg",
            '[{"box_2d": [10, 10, 50, 50], "label": "cat"}]',
            metadata={"coordinate_format": "xyxy_absolute_resized_image"},
        )
        result = compute_dataset_map(run, index)
        assert result is not None
        assert result.map50 > 0.99

    def test_uses_recorded_uploaded_dimensions_metadata(self) -> None:
        sample = _make_sample(_detections([[0, 0, 100, 100]], [0]))
        index = build_sample_index([sample])
        run = _make_run(
            "image.jpg",
            '[{"box_2d": [0, 0, 50, 50], "label": "cat"}]',
            metadata={
                "coordinate_format": "xyxy_absolute_resized_image",
                "uploaded_width": 50,
                "uploaded_height": 50,
            },
        )
        result = compute_dataset_map(run, index)
        assert result is not None
        assert result.map50 > 0.99

    def test_non_positive_recorded_dimensions_fall_back(self) -> None:
        sample = _make_sample(_detections([[10, 10, 50, 50]], [0]))
        index = build_sample_index([sample])
        run = _make_run(
            "image.jpg",
            '[{"box_2d": [10, 10, 50, 50], "label": "cat"}]',
            metadata={
                "coordinate_format": "xyxy_absolute_resized_image",
                "uploaded_width": 0,
                "uploaded_height": 0,
            },
        )
        result = compute_dataset_map(run, index)
        assert result is not None
        assert result.map50 > 0.99

    def test_min_confidence_filters_low_scoring_boxes(self) -> None:
        sample = _make_sample(_detections([[10, 10, 50, 50]], [0]))
        index = build_sample_index([sample])
        run = _make_run(
            "image.jpg",
            json.dumps(
                [
                    {
                        "box_2d": [100, 100, 500, 500],
                        "label": "cat",
                        "confidence": 0.9,
                    },
                    {
                        "box_2d": [600, 600, 900, 900],
                        "label": "cat",
                        "confidence": 0.2,
                    },
                ]
            ),
            metadata={"coordinate_format": "yxyx_normalized_0_to_1000"},
        )
        unfiltered = compute_dataset_map(run, index)
        filtered = compute_dataset_map(run, index, min_confidence=0.5)
        assert unfiltered is not None
        assert filtered is not None
        assert filtered.map50 >= unfiltered.map50


class TestConfidenceFiltering:
    def test_filter_prediction_json_drops_subthreshold_entries(self) -> None:
        raw = json.dumps(
            [
                {"box_2d": [1, 1, 2, 2], "label": "cat", "confidence": 0.9},
                {"box_2d": [3, 3, 4, 4], "label": "cat", "confidence": 0.2},
            ]
        )
        filtered = json.loads(filter_prediction_json(raw, 0.5))
        assert len(filtered) == 1
        assert filtered[0]["confidence"] == 0.9

    def test_filter_prediction_json_keeps_entries_without_confidence(self) -> None:
        raw = json.dumps(
            [
                {"box_2d": [1, 1, 2, 2], "label": "cat"},
                {"box_2d": [3, 3, 4, 4], "label": "cat", "confidence": 0.2},
            ]
        )

        filtered = json.loads(filter_prediction_json(raw, 0.5))

        assert filtered == [{"box_2d": [1, 1, 2, 2], "label": "cat"}]

    def test_compute_image_map50_handles_empty_predictions(self) -> None:
        ground_truth = _detections([[10, 10, 50, 50]], [0])
        assert compute_image_map50(sv.Detections.empty(), ground_truth) == 0.0


class TestEvaluateUploadedSize:
    def test_evaluate_uses_provided_uploaded_size(self) -> None:
        task = DetectionTask(
            coordinate_format=DetectionCoordinateFormat.XYXY_ABSOLUTE_RESIZED_IMAGE
        )
        sample = _make_sample(_detections([[10, 10, 50, 50]], [0]))
        prediction = '[{"box_2d": [5, 5, 25, 25], "label": "cat"}]'
        result = task.evaluate(sample, prediction, uploaded_size=(50, 50))
        assert result.details is not None
        assert result.details["map50"] > 0.99

    def test_build_prompt_uses_provided_uploaded_size(self) -> None:
        task = DetectionTask(
            coordinate_format=DetectionCoordinateFormat.XYXY_ABSOLUTE_RESIZED_IMAGE
        )
        sample = _make_sample(_detections([[0, 0, 10, 10]], [0]))
        prompt = task.build_prompt(sample, uploaded_size=(321, 123))
        assert "321x123 pixel image" in prompt


class TestResolutionTierMath:
    def test_matches_documented_standard_tier_example(self) -> None:
        assert compute_resize_dimensions(1920, 1080, 1568, 1568) == (1456, 819)
        assert compute_resize_dimensions(1075, 1520, 1568, 1568) == (924, 1307)

    def test_high_tier_leaves_documented_scan_unresized(self) -> None:
        assert compute_resize_dimensions(1075, 1520) == (1075, 1520)


class TestLabelCollision:
    def test_sparse_boxes_do_not_collide(self) -> None:
        from vlm_exam.visualization.detection import _labels_collide

        detections = _detections([[0, 100, 50, 200], [500, 600, 600, 700]], [0, 1])
        assert _labels_collide(detections, ["cat", "dog"], (1000, 1000)) is False

    def test_stacked_boxes_collide(self) -> None:
        from vlm_exam.visualization.detection import _labels_collide

        boxes = [[100.0 + i, 100.0 + i, 200.0 + i, 200.0 + i] for i in range(10)]
        detections = _detections(boxes, [0] * 10)
        labels = ["cat"] * 10
        assert _labels_collide(detections, labels, (1000, 1000)) is True

    def test_few_stacked_boxes_stay_readable(self) -> None:
        from vlm_exam.visualization.detection import _labels_collide

        detections = _detections(
            [[100, 100, 200, 200], [105, 105, 205, 205], [110, 110, 210, 210]],
            [0, 1, 0],
        )
        labels = ["cat", "dog", "cat"]
        assert _labels_collide(detections, labels, (1000, 1000)) is False

    def test_many_boxes_always_collide(self) -> None:
        from vlm_exam.visualization.detection import _labels_collide

        boxes = [[i * 25.0, 0.0, i * 25.0 + 20, 20.0] for i in range(35)]
        detections = _detections(boxes, [0] * 35)
        labels = ["x"] * 35
        assert _labels_collide(detections, labels, (1000, 1000)) is True

    def test_single_box_never_collides(self) -> None:
        from vlm_exam.visualization.detection import _labels_collide

        detections = _detections([[0, 0, 100, 100]], [0])
        assert _labels_collide(detections, ["cat"], (1000, 1000)) is False


class TestConfidenceParsing:
    def test_absolute_pixel_predictions_keep_confidence(self) -> None:
        prediction = (
            '[{"box_2d": [10, 20, 50, 60], "label": "cat", "confidence": 0.87},'
            ' {"box_2d": [1, 2, 3, 4], "label": "dog", "confidence": 0.12}]'
        )
        detections = parse_prediction(
            prediction,
            (100, 100),
            ["cat", "dog"],
            coordinate_format=DetectionCoordinateFormat.XYXY_ABSOLUTE_ORIGINAL_IMAGE,
        )
        assert detections.confidence is not None
        assert detections.confidence.tolist() == pytest.approx([0.87, 0.12])

    def test_vlm_predictions_remain_without_confidence(self) -> None:
        prediction = '[{"box_2d": [100, 200, 300, 400], "label": "cat"}]'
        detections = parse_prediction(
            prediction,
            (1000, 1000),
            ["cat"],
            coordinate_format=DetectionCoordinateFormat.YXYX_NORMALIZED_0_TO_1000,
        )
        assert detections.confidence is None


class TestDetectionCard:
    def test_region_diff_tints_disjoint_and_overlapping_areas(self) -> None:
        import numpy as np

        from vlm_exam.visualization.detection import _region_diff_image

        image = np.full((100, 100, 3), 255, dtype=np.uint8)
        ground_truth = _detections([[0, 0, 40, 40], [60, 60, 90, 90]], [0, 0])
        predictions = _detections([[20, 20, 40, 40], [0, 60, 30, 90]], [0, 0])
        diff = _region_diff_image(image, ground_truth, predictions)
        only_expected = diff[10, 10]
        only_model = diff[70, 10]
        both = diff[30, 30]
        untouched = diff[50, 50]
        assert only_expected[1] > only_expected[0]
        assert only_model[0] > only_model[1]
        assert both[0] > both[1] and both[2] > both[1]
        assert (untouched == untouched[0]).all()

    def test_invalid_label_mode_raises(self) -> None:
        import numpy as np

        from vlm_exam.config import BenchmarkConfig
        from vlm_exam.visualization.detection import plot_detection_card

        with pytest.raises(ValueError):
            plot_detection_card(
                image=np.zeros((10, 10, 3), dtype=np.uint8),
                ground_truth=sv.Detections.empty(),
                predictions=sv.Detections.empty(),
                gt_labels=[],
                pred_labels=[],
                model_id="test-model",
                config=BenchmarkConfig(labs={}, models={}),
                label_mode="bogus",
            )

    def test_save_annotated_detection_writes_png(self, tmp_path: Path) -> None:
        import numpy as np

        from vlm_exam.visualization.detection import save_annotated_detection

        image = np.full((64, 64, 3), 200, dtype=np.uint8)
        detections = _detections([[10, 10, 40, 40]], [0])
        output_file = tmp_path / "annotated.png"
        save_annotated_detection(
            image,
            detections,
            ["pill"],
            output_file,
            label_mode="labels",
        )
        assert output_file.is_file()
        assert output_file.stat().st_size > 0


@pytest.mark.parametrize("maximum", [100, 999, 1000])
def test_text_normalized_coordinates_follow_detection_contract(maximum: int) -> None:
    from vlm_exam.tasks.text import TextSample, TextTask

    sample = TextSample(
        image_path="image.png",
        category="localization_recognition",
        question="Read",
        specification="",
        expected='[{"bbox": [0,0,100,200], "text": "a"}]',
        answer=[{"bbox": [0, 0, 100, 200], "text": "a"}],
        image_width=100,
        image_height=200,
        image_hash="image",
        identity="pair",
        dataset_hash="dataset",
        dataset_pairs=1,
        prefix="{}",
        subset="sample",
        scoring_profile="default",
    )
    task = TextTask(coordinate_format=f"xyxy_normalized_0_to_{maximum}")
    prediction = json.dumps([{"box_2d": [0, 0, maximum, maximum], "text": "a"}])
    assert f"{maximum}" in task.build_prompt(sample)
    assert task.evaluate(sample, prediction).score == 1.0


def test_v2_dense_scoring_has_no_100_object_cap() -> None:
    from vlm_exam.detection_v2 import score_subset

    objects = [
        {"box": [index * 3, 0, index * 3 + 2, 2], "label": "seed"}
        for index in range(150)
    ]
    result = score_subset(
        [{"id": "crowded", "objects": objects}],
        [{"case": "crowded", "detections": objects, "status": "returned"}],
    )
    assert result["map50"] == pytest.approx(1)
    assert result["map50_capped100"] < 0.7
    assert result["f1"] == 1
    assert result["tp"] == 150


def test_v2_f1_counts_unknown_classes_and_duplicate_predictions() -> None:
    from vlm_exam.detection_v2 import score_subset

    target = {"box": [0, 0, 10, 10], "label": "seed"}
    predictions = [target, target, {**target, "label": "stone"}]
    result = score_subset(
        [{"id": "a", "objects": [target]}],
        [{"case": "a", "detections": predictions, "status": "returned"}],
    )
    assert (result["tp"], result["fp"], result["fn"]) == (1, 2, 0)
    assert result["f1"] == 0.5


def test_v2_missing_response_stays_in_denominator() -> None:
    from vlm_exam.detection_v2 import score_subset

    target = {"box": [0, 0, 10, 10], "label": "seed"}
    result = score_subset(
        [{"id": "a", "objects": [target]}],
        [{"case": "a", "detections": [], "status": "missing"}],
    )
    assert result["images"] == result["missing"] == result["fn"] == 1
    assert result["map50"] == result["f1"] == 0


def test_v2_parser_preserves_unknown_labels_and_uploaded_coordinates() -> None:
    from vlm_exam.detection_v2_inference import parse_objects

    case = {"width": 1000, "height": 500, "objects": []}
    result = {
        "predicted": '[{"box_2d":[10,20,30,40],"label":"unexpected"}]',
        "uploaded_size": [500, 250],
    }
    parsed = parse_objects(case, result, "xyxy_absolute_resized_image")
    assert parsed == [
        {"box": [20.0, 40.0, 60.0, 80.0], "label": "unexpected", "confidence": 1.0}
    ]


def test_v2_openai_request_preserves_native_effort_and_token_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from PIL import Image

    from vlm_exam import detection_v2_inference as inference

    Image.new("RGB", (100, 50)).save(tmp_path / "a.png")
    response = SimpleNamespace(
        output_text='[{"box_2d":[1,2,3,4],"label":"seed"}]',
        status="completed",
        model="test-model",
        usage=SimpleNamespace(
            input_tokens=4,
            output_tokens=8,
            output_tokens_details=SimpleNamespace(reasoning_tokens=3),
        ),
        model_dump=lambda **kwargs: {},
    )
    client = MagicMock()
    stream = client.responses.stream.return_value.__enter__.return_value
    stream.get_final_response.return_value = response
    monkeypatch.setattr(inference, "_client", lambda provider: client)
    case = {
        "id": "a",
        "image": "a.png",
        "width": 100,
        "height": 50,
        "objects": [{"box": [1, 2, 3, 4], "label": "seed"}],
    }
    config = {
        "provider": "openai",
        "wire": "test-model",
        "maximum": "max",
        "format": "xyxy_absolute_resized_image",
        "max_output_tokens": 128000,
    }
    result = inference.infer(case, "test-model", "maximum", config, tmp_path)
    request = client.responses.stream.call_args.kwargs
    assert request["reasoning"] == {"effort": "max"}
    assert request["max_output_tokens"] == 128000
    assert request["store"] is False
    assert "100x50" in result["prompt"]
    assert result["native_effort"] == "max"
    assert result["detections"][0]["box"] == [1, 2, 3, 4]


def test_v2_resume_reuses_success_and_rejects_changed_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hashlib

    from PIL import Image

    from vlm_exam import detection_v2_inference as inference
    from vlm_exam import detection_v2_runner as runner

    images = tmp_path / "images"
    images.mkdir()
    path = images / "a.png"
    Image.new("RGB", (10, 10)).save(path)
    case = {
        "id": "a",
        "subset": "s",
        "image": "a.png",
        "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "objects": [{"box": [0, 0, 5, 5], "label": "seed"}],
    }
    policy = {
        "dataset_sha256": "first",
        "efforts": ["minimum", "maximum"],
        "configurations": {"profile": {"maximum": "max"}},
        "runs": [{"model": "test", "effort": "maximum", "configurations": ["profile"]}],
    }
    monkeypatch.setattr(runner, "load_release", lambda root: (policy, [case]))
    monkeypatch.setattr(runner, "validate_release", lambda root: [])
    calls = []

    def fake_infer(*args: object) -> dict:
        calls.append(args)
        return {
            "case": "a",
            "model": args[1],
            "arm": args[2],
            "repeat": 1,
            "predicted": "[]",
            "detections": [],
            "truncated": False,
        }

    monkeypatch.setattr(inference, "infer", fake_infer)
    output = tmp_path / "run"
    runner.run_release(tmp_path, images, "test", "maximum", "profile", output, 2)
    runner.run_release(tmp_path, images, "test", "maximum", "profile", output, 2)
    assert len(calls) == 1
    policy["dataset_sha256"] = "changed"
    with pytest.raises(ValueError, match="Resume rejected"):
        runner.run_release(tmp_path, images, "test", "maximum", "profile", output, 2)
    assert len(calls) == 1

    policy["runs"] = []
    profile = {
        "provider": "openai",
        "wire": "new-model",
        "format": "xyxy_absolute_resized_image",
        "minimum": "low",
        "maximum": "max",
        "max_output_tokens": 128000,
    }
    runner.run_release(
        tmp_path, images, "new-model", "maximum", None, tmp_path / "new-run", 2, profile
    )
    assert len(calls) == 2


@pytest.mark.parametrize(
    "provider,arm,effort",
    [
        (provider, arm, effort)
        for provider in ["openai", "xai", "anthropic", "google", "openrouter"]
        for arm, effort in [("minimum", "low"), ("maximum", "high")]
    ]
    + [
        ("google", "minimum", 128),
        ("google", "maximum", 32768),
        ("openrouter", "minimum", "disabled"),
        ("openrouter", "maximum", "enabled"),
    ],
)
@pytest.mark.parametrize("parser_failure", [False, True])
def test_native_request_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    arm: str,
    effort: str | int,
    parser_failure: bool,
) -> None:
    text = '[{"box_2d":[1,2,3,4],"label":"seed"}]'
    Image.new("RGB", (100, 50)).save(tmp_path / "a.png")
    case = {
        "id": "a",
        "image": "a.png",
        "width": 100,
        "height": 50,
        "objects": [{"box": [1, 2, 3, 4], "label": "seed"}],
    }
    config = {
        "provider": provider,
        "wire": "test-model",
        "minimum": "low",
        "maximum": "high",
        "format": "xyxy_absolute_resized_image",
        "max_output_tokens": 128000,
    }
    config[arm] = effort
    response = SimpleNamespace(
        output_text=text,
        status="completed",
        model="test-model",
        stop_reason="end_turn",
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(
            input_tokens=4, output_tokens=8, output_tokens_details=None
        ),
        model_dump=lambda **kwargs: {"output_text": text, "id": "response-id"},
    )
    client = MagicMock()
    responses = client.responses.stream.return_value.__enter__.return_value
    messages = client.messages.stream.return_value.__enter__.return_value
    responses.get_final_response.return_value = response
    messages.get_final_message.return_value = response
    chunk = SimpleNamespace(
        model="test-model",
        model_version="test-model",
        usage=None,
        usage_metadata=None,
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[SimpleNamespace(text=text, thought=False)]
                ),
                finish_reason="STOP",
            )
        ],
        choices=[
            SimpleNamespace(delta=SimpleNamespace(content=text), finish_reason="stop")
        ],
        model_dump=lambda **kwargs: {"text": text},
    )
    client.models.generate_content_stream.return_value = iter([chunk])
    client.chat.completions.create.return_value.__iter__.return_value = iter([chunk])
    monkeypatch.setattr(inference, "_client", lambda provider: client)
    if parser_failure:

        def broken_parser(*args: object, **kwargs: object) -> list[dict]:
            raise ValueError("simulated parser failure")

        monkeypatch.setattr(inference, "parse_objects", broken_parser)
        with pytest.raises(inference.ResponseParseError) as raised:
            inference.infer(case, "test-model", arm, config, tmp_path)
        result = raised.value.response_record
        assert not base.is_retryable_error(raised.value)
        assert raised.value.request_retryable is False
        assert result["predicted"] == text
        assert result["parse_error"]["error_type"] == "ValueError"
        assert result["uploaded_size"] == [100, 50]
        assert "input_tokens" in result and "output_tokens" in result
    else:
        result = inference.infer(case, "test-model", arm, config, tmp_path)
        assert result["detections"][0]["box"] == [1, 2, 3, 4]
    assert result["raw"]
    assert result["native_effort"] == effort
    if provider in {"openai", "xai"}:
        request = client.responses.stream.call_args.kwargs
        assert request["reasoning"] == {"effort": effort}
        assert request["max_output_tokens"] == 128000
        assert request["store"] is False
    elif provider == "anthropic":
        request = client.messages.stream.call_args.kwargs
        assert request["thinking"] == {"type": "adaptive"}
        assert request["output_config"] == {"effort": effort}
        assert request["max_tokens"] == 128000
    elif provider == "google":
        request = client.models.generate_content_stream.call_args.kwargs
        if isinstance(effort, int):
            assert request["config"].thinking_config.thinking_budget == effort
            assert request["config"].thinking_config.thinking_level is None
        else:
            assert (
                request["config"].thinking_config.thinking_level.value.lower() == effort
            )
        assert request["config"].max_output_tokens == 128000
    else:
        request = client.chat.completions.create.call_args.kwargs
        expected = (
            {"enabled": effort == "enabled"}
            if effort in {"enabled", "disabled"}
            else {"effort": effort}
        )
        assert request["extra_body"]["reasoning"] == expected
        assert request["max_tokens"] == 128000
        assert request["extra_body"]["provider"] == {"require_parameters": True}


def test_google_identity_uses_explicit_aliases() -> None:
    assert inference.google_response_matches("gemini-3.5-flash", "gemini-3.6-flash")
    assert inference.google_response_matches("gemini-3.7-flash", "gemini-3.8-flash")
    assert not inference.google_response_matches(
        "gemini-3.5-flash", "gemini-3.5-flash-lite"
    )
    assert not inference.google_response_matches("gemini-3.5-flash", None)


def test_rate_limits_opt_in_preserves_legacy_failover(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RateLimit(Exception):
        status_code = 429
        response = SimpleNamespace(headers={"retry-after": "9"})

    calls = []
    sleeps = []

    def operation() -> str:
        calls.append(1)
        if len(calls) == 1:
            raise RateLimit("temporary")
        return "ok"

    monkeypatch.setattr(base.time, "sleep", sleeps.append)
    monkeypatch.setattr(base.random, "uniform", lambda a, b: 0)
    with pytest.raises(RateLimit):
        base.call_with_retries(operation)
    calls.clear()
    result, stats = base.call_with_retries(operation, retry_rate_limits=True)
    assert result == "ok" and stats.attempts == 2
    assert sleeps == [9]


def test_error_diagnostics_redacts_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "private-value")
    message = base.error_diagnostics(
        ValueError("private-value Bearer abc123 api_key=xyz")
    )["message"]
    assert (
        "private-value" not in message
        and "abc123" not in message
        and "xyz" not in message
    )


def test_failed_run_is_incomplete_and_resumable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    images = tmp_path / "images"
    images.mkdir()
    image = images / "a.png"
    Image.new("RGB", (10, 10)).save(image)
    case = {
        "id": "a",
        "subset": "s",
        "image": "a.png",
        "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
        "objects": [{"box": [0, 0, 5, 5], "label": "seed"}],
    }
    policy = {
        "dataset_sha256": "dataset",
        "efforts": ["minimum", "maximum"],
        "configurations": {"profile": {"maximum": "max"}},
        "runs": [{"model": "test", "effort": "maximum", "configurations": ["profile"]}],
    }
    monkeypatch.setattr(runner, "load_release", lambda root: (policy, [case]))
    monkeypatch.setattr(runner, "validate_release", lambda root: [])

    def fail(*args: object) -> dict:
        raise ValueError("bad parameter")

    monkeypatch.setattr(inference, "infer", fail)
    output = tmp_path / "run"
    with pytest.raises(ValueError, match="Run incomplete"):
        runner.run_release(tmp_path, images, "test", "maximum", "profile", output, 1)
    assert json.loads((output / "status.json").read_text())["phase"] == "incomplete"
    row = json.loads((output / "responses/a.json").read_text())
    assert row["error"]["message"] == "bad parameter"
    assert (output / "summary.json").exists()
    monkeypatch.setattr(
        inference,
        "infer",
        lambda *args: {
            "case": "a",
            "model": "test",
            "arm": "maximum",
            "repeat": 1,
            "predicted": "[]",
            "detections": [],
            "truncated": False,
        },
    )
    runner.run_release(tmp_path, images, "test", "maximum", "profile", output, 1)
    assert json.loads((output / "status.json").read_text())["phase"] == "finished"


def test_current_endpoints_do_not_mutate_historical_profiles() -> None:
    historical = {"provider": "google", "wire": "gemini-4-carbon-eap"}
    current = inference.current_request_profile(historical)
    assert historical["wire"] == "gemini-4-carbon-eap"
    assert current["wire"] == "gemini-4-carbon"
    assert (
        inference.current_request_profile(
            {"provider": "google", "wire": "gemini-4-hydrogen-eap"}
        )["wire"]
        == "gemini-4-hydrogen"
    )
    assert (
        inference.current_request_profile(
            {"provider": "openrouter", "wire": "gemini-4-carbon-eap"}
        )["wire"]
        == "gemini-4-carbon-eap"
    )


def _publication_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _publication_release(root: Path) -> dict:
    case = dict(
        id="a",
        subset="one",
        image="a.png",
        image_sha256="image",
        width=10,
        height=10,
        objects=[dict(box=[0, 0, 5, 5], label="wire")],
    )
    profile = dict(
        provider="openai",
        wire="test",
        format="xyxy_absolute_resized_image",
        minimum="low",
        maximum="max",
        max_output_tokens=128000,
    )
    fingerprint = _publication_digest(profile)
    directory = root / "datasets/detection/v2"
    directory.mkdir(parents=True)
    manifest = directory / "manifest.json"
    manifest.write_text(json.dumps(dict(cases=[case])))
    row = dict(
        case="a",
        model="test",
        arm="minimum",
        repeat=1,
        configuration=fingerprint,
        status="missing",
        detections=[],
    )
    archive = root / "results/detection/v2/test__minimum.jsonl.gz"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(gzip.compress((json.dumps(row) + "\n").encode()))
    policy = dict(
        protocol="detection-v2",
        dataset=str(manifest.relative_to(root)),
        dataset_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        images=1,
        subsets=1,
        images_per_subset=1,
        models=1,
        repeats=1,
        efforts=["minimum", "maximum"],
        configurations={fingerprint: profile},
        missing=["test/minimum/a"],
        diagnostics=[],
        runs=[
            dict(
                model="test",
                effort="minimum",
                path=str(archive.relative_to(root)),
                sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                samples=1,
                status_counts={"missing": 1},
                configurations=[fingerprint],
            )
        ],
    )
    (directory / "release.json").write_text(json.dumps(policy))
    prompt = DetectionTask(
        prompt_classes="image", coordinate_format=profile["format"]
    ).build_prompt(_sample(case), uploaded_size=(10, 10))
    text = '[{"box_2d":[0,0,5,5],"label":"wire"}]'
    return dict(
        case="a",
        model="test",
        arm="minimum",
        repeat=1,
        configuration=fingerprint,
        provider="openai",
        wire_model="test",
        native_effort="low",
        coordinate_format=profile["format"],
        max_output_tokens=128000,
        prompt=prompt,
        prompt_sha256=_publication_digest(prompt),
        uploaded_size=[10, 10],
        predicted=text,
        prediction_sha256=hashlib.sha256(text.encode()).hexdigest(),
        status="returned",
        detections=[dict(box=[0.0, 0.0, 5.0, 5.0], label="wire", confidence=1.0)],
        dataset_sha256=policy["dataset_sha256"],
        image_sha256="image",
    )


def test_stale_class_prompt_is_rejected_even_when_prediction_is_parseable(
    tmp_path: Path,
) -> None:
    row = _publication_release(tmp_path)
    policy, cases = load_release(tmp_path)
    row["prompt"] = row["prompt"].replace(
        "labels: wire", "labels: thick bond wire, thin bond wire"
    )
    row["prompt_sha256"] = _publication_digest(row["prompt"])
    assert "Prompt differs from current manifest" in _record_problems(
        cases[0], row, policy["configurations"][row["configuration"]]
    )
    with pytest.raises(ValueError, match="Prompt differs"):
        publish_recoveries(tmp_path, [row], {}, "Recovered infrastructure error")


def test_recovery_preserves_previous_record_and_cannot_replace_terminal_response(
    tmp_path: Path,
) -> None:
    row = _publication_release(tmp_path)
    publish_recoveries(tmp_path, [row], {}, "Recovered infrastructure error")
    assert validate_release(tmp_path) == []
    policy, _ = load_release(tmp_path)
    assert not policy["missing"]
    assert len(policy["diagnostics"]) == 1
    assert (
        read_records(tmp_path / policy["diagnostics"][0]["path"])[0]["status"]
        == "missing"
    )
    summary = json.loads((tmp_path / "datasets/detection/v2/summary.json").read_text())
    assert summary["models"][0]["metrics"]["map50"] == pytest.approx(1)
    with pytest.raises(ValueError, match="only replace"):
        publish_recoveries(tmp_path, [row], {}, "Not a genuine recovery")


def test_recovery_rejects_changed_infrastructure_profile(tmp_path: Path) -> None:
    row = _publication_release(tmp_path)
    policy, _ = load_release(tmp_path)
    profile = dict(
        policy["configurations"][row["configuration"]], max_output_tokens=256000
    )
    row.update(configuration=_publication_digest(profile), max_output_tokens=256000)
    with pytest.raises(ValueError, match="changed its profile"):
        publish_recoveries(
            tmp_path, [row], {_publication_digest(profile): profile}, "Changed cap"
        )


def test_full_publication_rejects_partial_and_accepts_finished_canonical_run(
    tmp_path: Path,
) -> None:
    row = _publication_release(tmp_path)
    policy, _ = load_release(tmp_path)
    output = tmp_path / "run"
    output.mkdir()
    (output / "run.json").write_text(
        json.dumps(
            dict(
                dataset_sha256=policy["dataset_sha256"],
                model="test",
                effort="minimum",
                configuration=row["configuration"],
                settings=policy["configurations"][row["configuration"]],
            )
        )
    )
    (output / "status.json").write_text(json.dumps(dict(phase="finished")))
    archive = output / "responses.jsonl.gz"
    archive.write_bytes(gzip.compress(b""))
    with pytest.raises(ValueError, match="partial"):
        publish_run(tmp_path, output)
    archive.write_bytes(gzip.compress((json.dumps(row) + "\n").encode()))
    publish_run(tmp_path, output)
    assert validate_release(tmp_path) == []


def test_publication_rolls_back_on_scoring_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vlm_exam import detection_v2_publication

    row = _publication_release(tmp_path)
    original = (tmp_path / "datasets/detection/v2/release.json").read_bytes()

    def fail(root: Path) -> dict:
        raise RuntimeError("Scoring failed")

    monkeypatch.setattr(detection_v2_publication, "summarize_release", fail)
    with pytest.raises(RuntimeError, match="Scoring failed"):
        publish_recoveries(tmp_path, [row], {}, "Recovered infrastructure error")
    assert (tmp_path / "datasets/detection/v2/release.json").read_bytes() == original
    assert validate_release(tmp_path) == []


def test_full_publication_preserves_native_enabled_arm(tmp_path: Path) -> None:
    row = _publication_release(tmp_path)
    policy, _ = load_release(tmp_path)
    profile = dict(
        policy["configurations"][row["configuration"]],
        arms=["enabled"],
        enabled="enabled",
    )
    row.update(
        arm="enabled",
        native_effort="enabled",
        configuration=_publication_digest(profile),
    )
    output = tmp_path / "enabled-run"
    output.mkdir()
    (output / "run.json").write_text(
        json.dumps(
            dict(
                dataset_sha256=policy["dataset_sha256"],
                model="test",
                effort="enabled",
                configuration=row["configuration"],
                settings=profile,
            )
        )
    )
    (output / "status.json").write_text(json.dumps(dict(phase="finished")))
    (output / "responses.jsonl.gz").write_bytes(
        gzip.compress((json.dumps(row) + "\n").encode())
    )
    publish_run(tmp_path, output)
    policy, _ = load_release(tmp_path)
    assert [entry["effort"] for entry in policy["runs"]] == ["enabled", "minimum"]
    assert validate_release(tmp_path) == []


def _download_fixture(root: Path, cases: list[dict], source: dict) -> None:
    directory = root / "datasets/detection/v2"
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text(json.dumps({"cases": cases}))
    (directory / "download.json").write_text(json.dumps(source))


def test_v2_image_cache_needs_no_network_or_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vlm_exam import detection_v2_data as data

    destination = tmp_path / "images"
    destination.mkdir()
    path = destination / "a.png"
    Image.new("RGB", (10, 10), "red").save(path)
    case = {
        "id": "a",
        "image": "a.png",
        "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    _download_fixture(tmp_path, [case], {"images": {"a": "remote-a"}})
    monkeypatch.delenv("ROBOFLOW_API_KEY", raising=False)
    network = MagicMock(side_effect=AssertionError("No network allowed"))
    monkeypatch.setattr(data, "_get", network)
    assert data.ensure_images(tmp_path, destination) == destination
    network.assert_not_called()


def test_v2_image_cache_rejects_modified_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vlm_exam import detection_v2_data as data

    destination = tmp_path / "images"
    destination.mkdir()
    path = destination / "a.png"
    path.write_bytes(b"modified")
    case = {
        "id": "a",
        "image": "a.png",
        "image_sha256": hashlib.sha256(b"original").hexdigest(),
    }
    _download_fixture(tmp_path, [case], {"images": {"a": "remote-a"}})
    network = MagicMock(side_effect=AssertionError("No network allowed"))
    monkeypatch.setattr(data, "_get", network)
    with pytest.raises(ValueError, match="Modified cached image"):
        data.ensure_images(tmp_path, destination)
    assert path.read_bytes() == b"modified"
    network.assert_not_called()


def test_v2_shared_photo_preserves_distinct_cases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vlm_exam import detection_v2_data as data

    destination = tmp_path / "images"
    destination.mkdir()
    path = destination / "a.png"
    Image.new("RGB", (10, 10), "red").save(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    cases = [
        {"id": name, "image": name + ".png", "image_sha256": digest}
        for name in ("a", "b")
    ]
    _download_fixture(tmp_path, cases, {"images": {"a": "remote-a", "b": "remote-b"}})
    monkeypatch.delenv("ROBOFLOW_API_KEY", raising=False)
    network = MagicMock(side_effect=AssertionError("No network allowed"))
    monkeypatch.setattr(data, "_get", network)
    data.ensure_images(tmp_path, destination)
    assert (destination / "b.png").read_bytes() == path.read_bytes()
    network.assert_not_called()


def test_v2_lossless_transport_crop_preserves_canonical_pixels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import io

    from vlm_exam import detection_v2_data as data

    original = io.BytesIO()
    Image.new("RGB", (10, 10), "red").save(original, format="JPEG")
    canonical = Image.open(io.BytesIO(original.getvalue())).convert("RGB")
    prepared = io.BytesIO()
    canonical.save(prepared, format="PNG")
    transport = Image.new("RGB", (11, 10))
    transport.paste(canonical)
    transport.paste(canonical.crop((9, 0, 10, 10)), (10, 0))
    encoded = io.BytesIO()
    transport.save(encoded, format="PNG")
    payload = encoded.getvalue()
    case = {
        "id": "a",
        "image": "a.jpg",
        "width": 10,
        "height": 10,
        "image_sha256": hashlib.sha256(original.getvalue()).hexdigest(),
    }
    source = {
        "workspace": "workspace",
        "project": "project",
        "images": {"a": "remote-a"},
        "lossless_copies": {
            "a": {
                "image_id": "lossless-a",
                "remove_right_columns": 1,
                "download_sha256": hashlib.sha256(payload).hexdigest(),
                "sha256": hashlib.sha256(prepared.getvalue()).hexdigest(),
                "pixel_sha256": hashlib.sha256(canonical.tobytes()).hexdigest(),
            }
        },
    }
    _download_fixture(tmp_path, [case], source)
    monkeypatch.setenv("ROBOFLOW_API_KEY", "fake-test-key")
    network = MagicMock(
        side_effect=[
            SimpleNamespace(
                json=lambda: {
                    "image": {
                        "urls": {
                            "original": "https://storage.googleapis.com/test/image"
                        }
                    }
                }
            ),
            SimpleNamespace(content=payload),
        ]
    )
    monkeypatch.setattr(data, "_get", network)
    destination = tmp_path / "images"
    destination.mkdir()
    outside = tmp_path / "untouched.txt"
    outside.write_text("keep")
    (destination / "a.jpg.part").symlink_to(outside)
    data.ensure_images(tmp_path, destination)
    assert outside.read_text() == "keep"
    path = destination / "a.jpg"
    assert Image.open(path).tobytes() == canonical.tobytes()
    assert data.verify_image(case, path, tmp_path)
    assert hashlib.sha256(path.read_bytes()).hexdigest() != case["image_sha256"]
    source["lossless_copies"]["a"]["pixel_sha256"] = "0" * 64
    (tmp_path / "datasets/detection/v2/download.json").write_text(json.dumps(source))
    assert not data.verify_image(case, path, tmp_path)
    path.unlink()
    network.side_effect = [
        SimpleNamespace(
            json=lambda: {
                "image": {
                    "urls": {"original": "https://storage.googleapis.com/test/image"}
                }
            }
        ),
        SimpleNamespace(content=payload + b"modified"),
    ]
    with pytest.raises(ValueError, match="Modified lossless transport"):
        data.ensure_images(tmp_path, destination)
    assert not path.exists()


def test_v2_download_rejects_redirects(monkeypatch: pytest.MonkeyPatch) -> None:
    from vlm_exam import detection_v2_data as data

    request = MagicMock(return_value=SimpleNamespace(status_code=302))
    monkeypatch.setattr(data.requests, "get", request)
    with pytest.raises(ValueError, match="HTTP 302"):
        data._get("https://api.roboflow.com/project/images/a", {"api_key": "fake"})
    assert request.call_args.kwargs["allow_redirects"] is False
    assert request.call_count == 1


def test_v2_parser_failure_recovers_locally_without_new_api_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    images = tmp_path / "images"
    images.mkdir()
    image = images / "a.png"
    Image.new("RGB", (10, 10)).save(image)
    case = {
        "id": "a",
        "subset": "s",
        "image": "a.png",
        "width": 10,
        "height": 10,
        "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
        "objects": [{"box": [0, 0, 5, 5], "label": "seed"}],
    }
    config = {
        "provider": "openai",
        "wire": "test",
        "maximum": "high",
        "format": "xyxy_absolute_resized_image",
    }
    digest = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    policy = {
        "dataset_sha256": "dataset",
        "efforts": ["minimum", "maximum"],
        "configurations": {digest: config},
        "runs": [{"model": "test", "effort": "maximum", "configurations": [digest]}],
    }
    monkeypatch.setattr(runner, "load_release", lambda root: (policy, [case]))
    monkeypatch.setattr(runner, "validate_release", lambda root: [])
    prediction = '[{"box_2d":[0,0,5,5],"label":"seed"}]'
    record = {
        "case": "a",
        "model": "test",
        "arm": "maximum",
        "repeat": 1,
        "predicted": prediction,
        "raw": {"text": prediction},
        "uploaded_size": [10, 10],
        "parse_error": {"error_type": "ValueError", "message": "broken parser"},
    }
    api = MagicMock(
        side_effect=inference.ResponseParseError(record, ValueError("broken parser"))
    )
    monkeypatch.setattr(inference, "infer", api)
    output = tmp_path / "run"
    for _ in range(2):
        with pytest.raises(ValueError, match="Run incomplete"):
            runner.run_release(tmp_path, images, "test", "maximum", digest, output, 1)
    assert api.call_count == 1
    identity_before = (output / "run.json").read_bytes()
    assert runner.reparse_run(tmp_path, output) == 1
    assert api.call_count == 1
    assert (output / "run.json").read_bytes() == identity_before
    saved = json.loads((output / "responses/a.json").read_text())
    assert saved["status"] == "returned"
    assert saved["detections"][0]["box"] == [0, 0, 5, 5]
    assert saved["raw"] == {"text": prediction}
    assert "local_reparse" in saved and "parse_error" not in saved
    assert len(list((output / "diagnostics").glob("*.json"))) == 1
    assert json.loads((output / "status.json").read_text())["phase"] == "finished"


def test_publication_holds_run_lock_before_reading_until_publish_finishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import fcntl

    from vlm_exam import detection_v2_publication

    output = tmp_path / "run"
    output.mkdir()
    lock_path = output / "run.lock"
    with lock_path.open("a") as writer:
        fcntl.flock(writer, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="Run is active"):
            publish_run(tmp_path, output)

    row = _publication_release(tmp_path)
    policy, _ = load_release(tmp_path)
    (output / "run.json").write_text(
        json.dumps(
            dict(
                dataset_sha256=policy["dataset_sha256"],
                model="test",
                effort="minimum",
                configuration=row["configuration"],
                settings=policy["configurations"][row["configuration"]],
            )
        )
    )
    (output / "status.json").write_text(json.dumps(dict(phase="finished")))
    (output / "responses.jsonl.gz").write_bytes(
        gzip.compress((json.dumps(row) + "\n").encode())
    )
    published = []

    def publish_while_locked(*args: object) -> None:
        with lock_path.open("a") as competing_writer:
            with pytest.raises(BlockingIOError):
                fcntl.flock(competing_writer, fcntl.LOCK_EX | fcntl.LOCK_NB)
        published.append(args)

    monkeypatch.setattr(detection_v2_publication, "_publish", publish_while_locked)
    publish_run(tmp_path, output)
    assert len(published) == 1
    with lock_path.open("a") as next_writer:
        fcntl.flock(next_writer, fcntl.LOCK_EX | fcntl.LOCK_NB)


@pytest.mark.parametrize(
    "fmt",
    [
        "yxyx_normalized_0_to_1000",
        "xyxy_normalized_0_to_1000",
        "xyxy_absolute_original_image",
    ],
)
@pytest.mark.parametrize(
    "raw,count",
    [
        ('{"label":"cat","box_2d":[10,20,40,50]}', 1),
        ('[{"label":"cat","box_2d":[10,20,40,50]}', 1),
        (
            '[{"label":"cat","box_2d":[10,20,40,50]} '
            '{"label":"cat","box_2d":[50,60,80,90]}]',
            2,
        ),
        (
            '{"label":"cat","box_2d":[10,20,40,50]}\n{"label":"cat","box_2d":[50,60,80,90]}',
            2,
        ),
    ],
)
def test_v2_parser_recovery_and_strict(fmt: str, raw: str, count: int) -> None:
    from vlm_exam.detection_v2_parsing import parse_response

    assert len(parse_response(raw, (1000, 1000), ["cat"], fmt, (1000, 1000))) == count
    assert (
        len(parse_response(raw, (1000, 1000), ["cat"], fmt, (1000, 1000), "strict"))
        == 0
    )


def test_v2_parser_schema_and_points() -> None:
    from vlm_exam.detection_v2_parsing import parse_response, strict_format_error

    fmt = "yxyx_normalized_0_to_1000"
    assert strict_format_error("[]", fmt) is None
    raw = '[{"label":"cat","point_2d":[10,20]}]'
    assert strict_format_error(raw, fmt) == "invalid_box"
    assert len(parse_response(raw, (100, 100), ["cat"], fmt, (100, 100))) == 0
    valid = '[{"label":"cat","box_2d":[10,20,40,50]}]'
    assert strict_format_error(valid, fmt) is None
    assert (
        len(parse_response(valid, (100, 100), ["cat"], fmt, (100, 100), "strict")) == 1
    )
    assert (
        strict_format_error(
            '[{"label":"cat","box_2d":[1.5,2,3,4]}]', "xyxy_normalized_0_to_100"
        )
        is None
    )


def test_v2_google_recovers_good_entry_beside_bad_entry() -> None:
    from vlm_exam.detection_v2_parsing import parse_response

    raw = '[{"label":"cat","box_2d":[10,20,40,50]}, {"label":"cat","box_2d":[1]}]'
    assert (
        len(
            parse_response(
                raw, (100, 100), ["cat"], "yxyx_normalized_0_to_1000", (100, 100)
            )
        )
        == 1
    )


def test_v2_parser_comparison_scores_same_response_without_mutating_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pathlib import Path

    from vlm_exam import detection_v2

    case = dict(
        id="test",
        subset="subset",
        width=100,
        height=100,
        objects=[dict(label="cat", box=[10, 20, 40, 50])],
    )
    row = dict(
        case="test",
        status="invalid_json",
        uploaded_size=[100, 100],
        coordinate_format="xyxy_absolute_original_image",
        predicted='{"label":"cat","box_2d":[10,20,40,50]}',
        detections=[],
    )
    entry = dict(
        path="unused",
        model="test",
        effort="minimum",
        sha256="hash",
        status_counts={"invalid_json": 1},
        configurations=[],
    )
    monkeypatch.setattr(detection_v2, "read_records", lambda path: [row])
    report = detection_v2._compare_parser_run((Path("."), entry, [case], "both"))
    assert report["format_compliant"] == 0
    assert report["recovered_noncompliant_responses"] == 1
    assert report["modes"]["strict"]["metrics"]["f1"] == 0
    assert report["modes"]["recovery"]["metrics"]["f1"] == 1
    assert report["modes"]["recovery"]["metrics"]["map50"] > 0.99
    assert row["detections"] == []


def test_v2_parser_retains_unknown_meta_label() -> None:
    from vlm_exam.detection_v2_inference import parse_objects

    case = dict(width=100, height=100, objects=[dict(label="cat")])
    result = dict(
        uploaded_size=[100, 100],
        predicted=(
            '[{"object_name":"dog","bbox":'
            '[{"x_min":10,"y_min":20,"x_max":40,"y_max":50}]}]'
        ),
    )
    objects = parse_objects(
        case, result, "xyxy_normalized_0_to_1000_meta_bbox", "strict"
    )
    assert len(objects) == 1
    assert objects[0]["label"] == "dog"


@pytest.mark.parametrize("coordinate_format", list(DetectionCoordinateFormat))
def test_v2_strict_accepts_requested_schema_for_every_format(
    coordinate_format: DetectionCoordinateFormat,
) -> None:
    from vlm_exam.detection_v2_parsing import parse_response, strict_format_error

    value = coordinate_format.value
    entry = {"label": "cat", "box_2d": [10, 20, 40, 50]}
    if value.endswith("meta_bbox"):
        entry = {
            "object_name": "cat",
            "bbox": [{"x_min": 10, "y_min": 20, "x_max": 40, "y_max": 50}],
        }
    elif value.endswith("meta_flat"):
        entry = {"label": "cat", "x_min": 10, "y_min": 20, "x_max": 40, "y_max": 50}
    elif value.endswith("image_bbox"):
        entry = {"label": "cat", "bbox": [10, 20, 40, 50]}
    raw = json.dumps([entry])
    assert strict_format_error(raw, value) is None
    assert (
        len(parse_response(raw, (100, 100), ["cat"], value, (100, 100), "strict")) == 1
    )


@pytest.mark.parametrize("fmt", list(DetectionCoordinateFormat))
@pytest.mark.parametrize(
    "values",
    [
        [True, 2, 3, 4],
        [None, 2, 3, 4],
        [float("nan"), 2, 3, 4],
        [float("inf"), 2, 3, 4],
    ],
)
def test_bounded_parser_rejects_invalid_numbers_in_every_format(
    fmt: DetectionCoordinateFormat, values: list[object]
) -> None:
    from vlm_exam.detection_v2_parsing import parse_response

    if fmt.value.endswith("meta_bbox"):
        entry = {
            "object_name": "cat",
            "bbox": [dict(zip(("x_min", "y_min", "x_max", "y_max"), values))],
        }
    elif fmt.value.endswith("meta_flat"):
        entry = dict(
            label="cat", **dict(zip(("x_min", "y_min", "x_max", "y_max"), values))
        )
    else:
        entry = {
            "label": "cat",
            "bbox" if fmt.value.endswith("image_bbox") else "box_2d": values,
        }
    for mode in ("strict", "recovery"):
        assert (
            len(
                parse_response(
                    json.dumps([entry]), (100, 100), ["cat"], fmt, (100, 100), mode
                )
            )
            == 0
        )


@pytest.mark.parametrize(
    "raw",
    [
        '{"detections":[{"label":"cat","box_2d":[1,2,3,4]}]}',
        '[{"nested":{"label":"cat","box_2d":[1,2,3,4]}}]',
        '[{"label":"cat","box_2d":[1,2,3,4,5]}]',
        '[{"label":"cat","box_2d":[1,2,3]}]',
        '[{"label":"cat","box_2d":[1,2,3,4],"bbox_2d":[2,3,4,5]}]',
        '[{"label":"cat","label":"dog","box_2d":[1,2,3,4]}]',
        '[{"label":"cat","box_2d":[1,2,3,4]}] [{"label":"cat","box_2d":[5,6,7,8]}]',
        "The cat is at x=1, y=2, x2=3, y2=4.",
    ],
)
def test_bounded_parser_does_not_reconstruct_ambiguous_or_wrong_answers(
    raw: str,
) -> None:
    from vlm_exam.detection_v2_parsing import parse_response

    for fmt in (
        "yxyx_normalized_0_to_1000",
        "xyxy_normalized_0_to_1000",
        "xyxy_absolute_resized_image",
    ):
        assert (
            len(parse_response(raw, (100, 100), ["cat", "dog"], fmt, (100, 100))) == 0
        )


@pytest.mark.parametrize(
    "fmt",
    [
        "yxyx_normalized_0_to_1000",
        "xyxy_normalized_0_to_1000",
        "xyxy_normalized_0_to_100",
        "xyxy_absolute_resized_image",
    ],
)
def test_bounded_parser_common_aliases_strings_and_confidence(fmt: str) -> None:
    from vlm_exam.detection_v2_parsing import parse_response

    raw = (
        '[{"description":"cat","bbox_2d":["10","20","30","40"],"confidence":0},'
        ' {"label":"cat","box_2d":[10,20,30,40]}]'
    )
    diagnostic = {}
    parsed = parse_response(
        raw, (100, 100), ["cat"], fmt, (100, 100), diagnostics=diagnostic
    )
    assert len(parsed) == 2
    assert parsed.confidence.tolist() == [0, 1]
    assert diagnostic["operations"]["field_alias"] == 2
    assert diagnostic["operations"]["numeric_string"] == 4
    assert len(parse_response(raw, (100, 100), ["cat"], fmt, (100, 100), "strict")) == 0


def test_bounded_parser_meta_zero_and_no_schema_reconstruction() -> None:
    from vlm_exam.detection_v2_parsing import parse_response

    fmt = "xyxy_normalized_0_to_1000_meta_bbox"
    raw = (
        '[{"label":"cat","bbox":{"x_min":"10","y_min":"20","x_max":"30",'
        '"y_max":"40","confidence":0},"confidence":0.9}]'
    )
    parsed = parse_response(raw, (100, 100), ["cat"], fmt, (100, 100))
    assert len(parsed) == 1
    assert parsed.confidence.tolist() == [0]
    assert (
        len(
            parse_response(
                '[{"label":"cat","box_2d":[1,2,3,4]}]',
                (100, 100),
                ["cat"],
                fmt,
                (100, 100),
            )
        )
        == 0
    )


@pytest.mark.parametrize("mode", ["strict", "recovery"])
def test_bounded_parser_preserves_outside_and_reversed_geometry(mode: str) -> None:
    from vlm_exam.detection_v2_parsing import parse_response

    raw = (
        '[{"label":"cat","box_2d":[-10,-20,200,300]},'
        ' {"label":"cat","box_2d":[40,50,10,20]}]'
    )
    for fmt in ("xyxy_normalized_0_to_1000", "xyxy_absolute_resized_image"):
        parsed = parse_response(raw, (1000, 1000), ["cat"], fmt, (1000, 1000), mode)
        assert parsed.xyxy.tolist() == [[-10, -20, 200, 300], [40, 50, 10, 20]]


def test_bounded_parser_malformed_complete_entry_does_not_erase_other_objects() -> None:
    from vlm_exam.detection_v2_parsing import parse_response

    raw = (
        '[{"label":"cat","box_2d":[1,2,3,4]}, {"label":oops},'
        ' {"label":"cat","box_2d":[5,6,7,8]},]'
    )
    assert (
        len(
            parse_response(
                raw, (100, 100), ["cat"], "xyxy_absolute_resized_image", (100, 100)
            )
        )
        == 2
    )


def test_bounded_reversed_boxes_remain_false_positives() -> None:
    from vlm_exam.detection_v2 import score_subset
    from vlm_exam.detection_v2_inference import parse_objects

    case = dict(
        id="a", width=100, height=100, objects=[dict(label="cat", box=[10, 20, 40, 50])]
    )
    objects = parse_objects(
        case,
        dict(
            predicted='[{"label":"cat","box_2d":[40,50,10,20]}]',
            uploaded_size=[100, 100],
        ),
        "xyxy_absolute_resized_image",
    )
    score = score_subset(
        [case], [dict(case="a", status="returned", detections=objects)]
    )
    assert score["fp"] == 1
    assert score["fn"] == 1
    assert score["tp"] == 0


@pytest.mark.parametrize("fmt", list(DetectionCoordinateFormat))
def test_bounded_formats_decode_same_geometry(fmt: DetectionCoordinateFormat) -> None:
    from vlm_exam.detection_v2_parsing import parse_response

    expected = [100, 100, 400, 300]
    if "normalized" in fmt.value:
        maximum = fmt.normalized_maximum
        values = [
            100 / 1000 * maximum,
            100 / 500 * maximum,
            400 / 1000 * maximum,
            300 / 500 * maximum,
        ]
    elif "resized" in fmt.value:
        values = [20, 20, 80, 60]
    else:
        values = expected
    if fmt.value.startswith("yxyx"):
        values = [values[1], values[0], values[3], values[2]]
    if fmt.value.endswith("meta_bbox"):
        entry = {
            "object_name": "cat",
            "bbox": [dict(zip(("x_min", "y_min", "x_max", "y_max"), values))],
        }
    elif fmt.value.endswith("meta_flat"):
        entry = dict(
            label="cat", **dict(zip(("x_min", "y_min", "x_max", "y_max"), values))
        )
    else:
        entry = {
            "label": "cat",
            "bbox" if fmt.value.endswith("image_bbox") else "box_2d": values,
        }
    result = parse_response(json.dumps([entry]), (1000, 500), ["cat"], fmt, (200, 100))
    assert result.xyxy[0].tolist() == pytest.approx(expected)


@pytest.mark.parametrize(
    "wrapper", ["{} Done.", "Here: {}", "Here: {} Done.", "```json\n{}\n```"]
)
def test_bounded_parser_one_unambiguous_wrapper(wrapper: str) -> None:
    from vlm_exam.detection_v2_parsing import parse_response

    raw = wrapper.format('[{"label":"cat","box_2d":[1,2,3,4]}]')
    assert (
        len(
            parse_response(
                raw, (100, 100), ["cat"], "xyxy_absolute_resized_image", (100, 100)
            )
        )
        == 1
    )


def test_approved_timeout_is_scored_without_hiding_missing_status(
    tmp_path: Path,
) -> None:
    from vlm_exam.detection_v2 import load_release, summarize_release, validate_release

    _publication_release(tmp_path)
    initial = summarize_release(tmp_path)["models"][0]
    assert initial["metrics"]["map50"] is None
    policy, _ = load_release(tmp_path)
    entry = policy["runs"][0]
    entry["accepted_missing_as_empty"] = ["a"]
    entry["acceptance_reason"] = "User accepted the timeout as an empty prediction"
    path = tmp_path / "datasets/detection/v2/release.json"
    path.write_text(json.dumps(policy))
    assert validate_release(tmp_path) == []
    updated = summarize_release(tmp_path)["models"][0]
    assert updated["complete"]
    assert updated["accepted_failure_count"] == 1
    assert updated["response_counts"] == {"missing": 1}
    assert updated["metrics"]["map50"] == 0
    assert updated["metrics"]["f1"] == 0
    assert updated["metrics"]["seconds_per_image"] is None
    entry["accepted_missing_as_empty"] = ["not-a-case"]
    path.write_text(json.dumps(policy))
    assert any(
        "Invalid accepted failure policy" in e for e in validate_release(tmp_path)
    )
    entry["accepted_missing_as_empty"] = ["a"]
    del entry["acceptance_reason"]
    path.write_text(json.dumps(policy))
    assert any(
        "Invalid accepted failure policy" in e for e in validate_release(tmp_path)
    )
