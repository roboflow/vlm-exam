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

import pytest

from vlm_exam.reference.best_prompt import merge_best_prompt_run
from vlm_exam.reference.constants import REFERENCE_EFFORT
from vlm_exam.reference.visualization import (
    detection_labels_for_card,
    prompt_label_map_from_metadata,
)
from vlm_exam.results import RunResult, SampleResult
from vlm_exam.tasks.detection import (
    DetectionCoordinateFormat,
    DetectionTask,
    build_sample_index,
    compute_dataset_map,
    parse_prediction,
)


def _write_detection_dataset(dataset_directory: Path) -> None:
    annotations = {
        "categories": [
            {"id": 0, "name": "vlm-exam", "supercategory": "none"},
            {"id": 1, "name": "cat", "supercategory": "vlm-exam"},
            {"id": 2, "name": "dog", "supercategory": "vlm-exam"},
        ],
        "images": [
            {"id": 1, "file_name": "a.jpg", "width": 100, "height": 100},
            {"id": 2, "file_name": "b.jpg", "width": 100, "height": 100},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20]},
            {"id": 2, "image_id": 2, "category_id": 2, "bbox": [40, 40, 20, 20]},
        ],
    }
    dataset_directory.mkdir(parents=True, exist_ok=True)
    (dataset_directory / "_annotations.coco.json").write_text(json.dumps(annotations))
    for image_name in ("a.jpg", "b.jpg"):
        (dataset_directory / image_name).write_bytes(b"\xff\xd8\xff\xd9")


def _sample(
    index: int,
    image: str,
    predicted: str,
    *,
    map50: float,
    prompt_class_names: list[str] | None = None,
    prompt_texts: list[str] | None = None,
) -> SampleResult:
    metadata: dict[str, object] = {
        "coordinate_format": "xyxy_absolute_original_image",
        "reference": True,
        "map50": map50,
    }
    if prompt_class_names is not None:
        metadata["prompt_class_names"] = prompt_class_names
    if prompt_texts is not None:
        metadata["prompt_texts"] = prompt_texts
    return SampleResult(
        index=index,
        image=image,
        expected="",
        predicted=predicted,
        correct=True,
        input_tokens=0,
        output_tokens=0,
        metadata=metadata,
    )


class TestBestPromptMerge:
    def test_merge_selects_better_prompt_per_image(self, tmp_path: Path) -> None:
        dataset_directory = tmp_path / "dataset"
        _write_detection_dataset(dataset_directory)
        task = DetectionTask()
        sample_index = build_sample_index(task.load_samples(str(dataset_directory)))

        baseline_run = RunResult(
            model="sam3",
            effort=REFERENCE_EFFORT,
            task="detection",
            timestamp="20260729_120000",
            samples=[
                _sample(
                    0,
                    "a.jpg",
                    '[{"box_2d": [10, 10, 30, 30], "label": "cat", "confidence": 0.9}]',
                    map50=1.0,
                ),
                _sample(
                    1,
                    "b.jpg",
                    '[{"box_2d": [40, 40, 60, 60], "label": "cat", "confidence": 0.9}]',
                    map50=0.0,
                ),
            ],
        )
        image_conditioned_run = RunResult(
            model="sam3",
            effort=REFERENCE_EFFORT,
            task="detection",
            timestamp="20260729_130000",
            samples=[
                _sample(
                    0,
                    "a.jpg",
                    '[{"box_2d": [10, 10, 30, 30], "label": "cat", "confidence": 0.5}]',
                    map50=1.0,
                ),
                _sample(
                    1,
                    "b.jpg",
                    '[{"box_2d": [40, 40, 60, 60], "label": "dog", "confidence": 0.9}]',
                    map50=1.0,
                ),
            ],
        )

        merge_result = merge_best_prompt_run(
            baseline_run,
            image_conditioned_run,
            sample_index,
        )

        assert merge_result.baseline_wins == 0
        assert merge_result.image_conditioned_wins == 1
        assert merge_result.ties == 1

        by_image = {sample.image: sample for sample in merge_result.merged_run.samples}
        assert "cat" in by_image["a.jpg"].predicted
        assert "dog" in by_image["b.jpg"].predicted

        map_result = compute_dataset_map(merge_result.merged_run, sample_index)
        assert map_result is not None
        assert map_result.map50 == pytest.approx(1.0)
        assert map_result.image_count == 2

    def test_merged_run_labels_use_winning_prompt(self, tmp_path: Path) -> None:
        dataset_directory = tmp_path / "dataset"
        _write_detection_dataset(dataset_directory)
        task = DetectionTask()
        sample_index = build_sample_index(task.load_samples(str(dataset_directory)))

        cat_box = '[{"box_2d": [10, 10, 30, 30], "label": "cat", "confidence": 0.9}]'
        dog_box = '[{"box_2d": [40, 40, 60, 60], "label": "dog", "confidence": 0.9}]'
        baseline_run = RunResult(
            model="sam3",
            effort=REFERENCE_EFFORT,
            task="detection",
            timestamp="20260730_120000",
            samples=[
                _sample(
                    0,
                    "a.jpg",
                    cat_box,
                    map50=1.0,
                    prompt_class_names=["cat", "dog"],
                    prompt_texts=["cat", "dog"],
                ),
                _sample(
                    1,
                    "b.jpg",
                    "[]",
                    map50=0.0,
                    prompt_class_names=["cat", "dog"],
                    prompt_texts=["cat", "dog"],
                ),
            ],
        )
        image_conditioned_run = RunResult(
            model="sam3",
            effort=REFERENCE_EFFORT,
            task="detection",
            timestamp="20260730_130000",
            samples=[
                _sample(
                    0,
                    "a.jpg",
                    "[]",
                    map50=0.0,
                    prompt_class_names=["cat", "dog"],
                    prompt_texts=["striped tabby cat", "brown furry dog"],
                ),
                _sample(
                    1,
                    "b.jpg",
                    dog_box,
                    map50=1.0,
                    prompt_class_names=["cat", "dog"],
                    prompt_texts=["striped tabby cat", "brown furry dog"],
                ),
            ],
        )

        merge_result = merge_best_prompt_run(
            baseline_run,
            image_conditioned_run,
            sample_index,
        )
        by_image = {sample.image: sample for sample in merge_result.merged_run.samples}

        labels_by_image: dict[str, list[str]] = {}
        for image_name, merged_sample in by_image.items():
            dataset_sample = sample_index[image_name]
            predictions = parse_prediction(
                merged_sample.predicted,
                (dataset_sample.image_width, dataset_sample.image_height),
                list(dataset_sample.classes),
                coordinate_format=DetectionCoordinateFormat(
                    merged_sample.metadata["coordinate_format"]
                ),
            )
            labels_by_image[image_name] = detection_labels_for_card(
                predictions,
                list(dataset_sample.classes),
                label_classes="augmented",
                prompt_label_map=prompt_label_map_from_metadata(merged_sample.metadata),
            )

        assert by_image["a.jpg"].metadata["best_prompt_selected"] == "baseline"
        assert labels_by_image["a.jpg"] == ["cat"]
        assert by_image["b.jpg"].metadata["best_prompt_selected"] == (
            "image-conditioned"
        )
        assert labels_by_image["b.jpg"] == ["brown furry dog"]

    def test_merge_rejects_mismatched_image_sets(self, tmp_path: Path) -> None:
        dataset_directory = tmp_path / "dataset"
        _write_detection_dataset(dataset_directory)
        task = DetectionTask()
        sample_index = build_sample_index(task.load_samples(str(dataset_directory)))

        baseline_run = RunResult(
            model="sam3",
            effort=REFERENCE_EFFORT,
            task="detection",
            timestamp="20260729_120000",
            samples=[
                _sample(
                    0,
                    "a.jpg",
                    '[{"box_2d": [10, 10, 30, 30], "label": "cat", "confidence": 0.9}]',
                    map50=1.0,
                ),
            ],
        )
        image_conditioned_run = RunResult(
            model="sam3",
            effort=REFERENCE_EFFORT,
            task="detection",
            timestamp="20260729_130000",
            samples=[
                _sample(
                    0,
                    "b.jpg",
                    '[{"box_2d": [40, 40, 60, 60], "label": "dog", "confidence": 0.9}]',
                    map50=1.0,
                ),
            ],
        )

        with pytest.raises(ValueError, match="same images"):
            merge_best_prompt_run(
                baseline_run,
                image_conditioned_run,
                sample_index,
            )
