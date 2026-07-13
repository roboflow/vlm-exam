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

from vlm_exam.providers.anthropic import compute_resize_dimensions
from vlm_exam.results import RunResult, SampleResult
from vlm_exam.tasks.detection import (
    MAP_PASS_THRESHOLD,
    DetectionCoordinateFormat,
    DetectionSample,
    DetectionTask,
    build_sample_index,
    compute_dataset_map,
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
        task = DetectionTask(coordinate_format="xyxy_absolute_provider_upload")
        sample = _make_sample(_detections([[0, 0, 10, 10]], [0]))
        prompt = task.build_prompt(sample)
        assert "[x_min, y_min, x_max, y_max]" in prompt
        assert "pixel coordinates" in prompt
        assert "100x100 pixel image" in prompt

    def test_pixel_format_prompt_uses_resized_dimensions(self) -> None:
        task = DetectionTask(coordinate_format="xyxy_absolute_provider_upload")
        sample = DetectionSample(
            image_path="/data/large.jpg",
            image_width=4000,
            image_height=3000,
            ground_truth=_detections([[0, 0, 10, 10]], [0]),
            classes=("cat", "dog"),
        )
        prompt = task.build_prompt(sample)
        uploaded_width, uploaded_height = compute_resize_dimensions(4000, 3000)
        assert uploaded_width < 4000
        assert f"{uploaded_width}x{uploaded_height} pixel image" in prompt

    def test_invalid_coordinate_format_raises(self) -> None:
        with pytest.raises(ValueError):
            DetectionTask(coordinate_format="bogus")


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


class TestParsePixelPrediction:
    def test_parses_pixel_coordinates_directly(self) -> None:
        prediction = '[{"box_2d": [10, 20, 30, 40], "label": "cat"}]'
        detections = parse_prediction(
            prediction,
            (100, 100),
            ["cat", "dog"],
            coordinate_format="xyxy_absolute_provider_upload",
        )
        assert len(detections) == 1
        assert detections.class_id is not None
        assert detections.class_id[0] == 0
        # box_2d is [x_min, y_min, x_max, y_max] in pixels; no resize at 100x100
        np.testing.assert_allclose(detections.xyxy[0], [10, 20, 30, 40])

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
            coordinate_format="xyxy_absolute_provider_upload",
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
            coordinate_format="xyxy_absolute_provider_upload",
        )
        assert len(detections) == 1
        np.testing.assert_allclose(detections.xyxy[0], [10, 20, 30, 40])

    def test_unknown_labels_are_filtered(self) -> None:
        prediction = '[{"box_2d": [10, 20, 30, 40], "label": "unicorn"}]'
        detections = parse_prediction(
            prediction,
            (100, 100),
            ["cat"],
            coordinate_format="xyxy_absolute_provider_upload",
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
            coordinate_format="xyxy_absolute_provider_upload",
        )
        assert len(detections) == 1

    def test_sets_class_name_data(self) -> None:
        prediction = '[{"box_2d": [10, 20, 30, 40], "label": "dog"}]'
        detections = parse_prediction(
            prediction,
            (100, 100),
            ["cat", "dog"],
            coordinate_format="xyxy_absolute_provider_upload",
        )
        assert list(detections.data["class_name"]) == ["dog"]


class TestParseNativePixelPrediction:
    def test_parses_pixel_native_coordinates_directly(self) -> None:
        prediction = '[{"box_2d": [10, 20, 30, 40], "label": "cat"}]'
        detections = parse_prediction(
            prediction,
            (100, 100),
            ["cat", "dog"],
            coordinate_format="xyxy_absolute_original_image",
        )
        assert len(detections) == 1
        np.testing.assert_allclose(detections.xyxy[0], [10, 20, 30, 40])

    def test_pixel_native_does_not_apply_resize_scaling(self) -> None:
        prediction = '[{"box_2d": [100, 200, 300, 400], "label": "cat"}]'
        detections = parse_prediction(
            prediction,
            (4000, 3000),
            ["cat"],
            coordinate_format="xyxy_absolute_original_image",
        )
        assert len(detections) == 1
        np.testing.assert_allclose(detections.xyxy[0], [100, 200, 300, 400])

    def test_parses_pixel_yxyx_native_coordinates(self) -> None:
        prediction = '[{"box_2d": [20, 10, 40, 30], "label": "cat"}]'
        detections = parse_prediction(
            prediction,
            (100, 100),
            ["cat"],
            coordinate_format="yxyx_absolute_original_image",
        )
        assert len(detections) == 1
        np.testing.assert_allclose(detections.xyxy[0], [10, 20, 30, 40])

    def test_pixel_native_prompt_uses_original_dimensions(self) -> None:
        task = DetectionTask(coordinate_format="xyxy_absolute_original_image")
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
        task = DetectionTask(coordinate_format="yxyx_absolute_original_image")
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
            coordinate_format=DetectionCoordinateFormat.XYXY_ABSOLUTE_PROVIDER_UPLOAD
        )
        sample = _make_sample(sv.Detections.empty())
        result = task.evaluate(sample, "[]")
        assert result.details is not None
        assert result.details["coordinate_format"] == "xyxy_absolute_provider_upload"

    def test_enum_member_and_canonical_string_are_equivalent(self) -> None:
        from_string = DetectionTask(coordinate_format="xyxy_absolute_original_image")
        from_member = DetectionTask(
            coordinate_format=DetectionCoordinateFormat.XYXY_ABSOLUTE_ORIGINAL_IMAGE
        )
        sample = _make_sample(sv.Detections.empty())
        assert (
            from_string.evaluate(sample, "[]").details["coordinate_format"]
            == from_member.evaluate(sample, "[]").details["coordinate_format"]
        )

    def test_pixel_format_perfect_prediction_scores_high(self) -> None:
        task = DetectionTask(
            coordinate_format=DetectionCoordinateFormat.XYXY_ABSOLUTE_PROVIDER_UPLOAD
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
            metadata={"coordinate_format": "xyxy_absolute_provider_upload"},
        )
        result = compute_dataset_map(run, index)
        assert result is not None
        assert result.map50 > 0.99


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
