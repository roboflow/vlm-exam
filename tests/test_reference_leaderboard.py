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
from dataclasses import replace
from pathlib import Path

import pytest

from vlm_exam.config import load_config
from vlm_exam.reference.config import ReferenceConfig, load_reference_config
from vlm_exam.reference.constants import REFERENCE_EFFORT
from vlm_exam.reference.leaderboard import (
    YOLOE_GEMINI_FOCUS_VLM,
    build_mixed_detection_leaderboard,
    build_mixed_leaderboard_config,
    family_reference_keys,
    leaderboard_rows_for_family,
    leaderboard_rows_for_yoloe_gemini_focus,
    mixed_detection_leaderboard_payload,
    row_chart_label,
)
from vlm_exam.results import RunResult, SampleResult, save_results
from vlm_exam.tasks.detection import DetectionTask, build_sample_index


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


def _sample(index: int, image: str, predicted: str) -> SampleResult:
    return SampleResult(
        index=index,
        image=image,
        expected="",
        predicted=predicted,
        correct=True,
        input_tokens=0,
        output_tokens=0,
        metadata={
            "coordinate_format": "xyxy_absolute_original_image",
            "reference": True,
        },
    )


class TestMixedDetectionLeaderboard:
    def test_build_mixed_leaderboard_config_adds_reference_variants(self) -> None:
        vlm_config = load_config()
        reference_config = load_reference_config()
        chart_config = build_mixed_leaderboard_config(vlm_config, reference_config)
        for key in (
            "sam3-baseline",
            "sam3-image-conditioned",
            "sam3-best",
            "yoloe-11l-seg-baseline",
            "yoloe-11l-seg-image-conditioned",
            "yoloe-11l-seg-best",
            "yoloe-26x-seg-baseline",
            "yoloe-26x-seg-image-conditioned",
            "yoloe-26x-seg-best",
        ):
            assert key in chart_config.models
        assert chart_config.models["sam3-baseline"].name == "SAM 3 (base prompt)"
        assert chart_config.models["sam3-best"].name == "SAM 3 (best prompt)"
        assert chart_config.models["yoloe-11l-seg-baseline"].name == (
            "YOLOE-11l (base prompt)"
        )
        yoloe_x_best = chart_config.models["yoloe-26x-seg-best"].name
        assert yoloe_x_best == "YOLOE-26x (best prompt)"

    def test_unlisted_reference_without_lab_is_ignored(self) -> None:
        vlm_config = load_config()
        reference_config = load_reference_config()
        models = dict(reference_config.models)
        models["experimental"] = replace(models["sam3"], key="experimental", lab=None)

        chart_config = build_mixed_leaderboard_config(
            vlm_config,
            ReferenceConfig(models=models, labs=reference_config.labs),
        )

        assert "experimental-baseline" not in chart_config.models

    def test_family_reference_key_counts(self) -> None:
        assert len(family_reference_keys("sam3")) == 2
        assert len(family_reference_keys("yoloe")) == 4

    def test_reference_best_row_beats_baseline_in_mixed_leaderboard(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
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
                ),
                _sample(
                    1,
                    "b.jpg",
                    '[{"box_2d": [40, 40, 60, 60], "label": "cat", "confidence": 0.9}]',
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
                ),
                _sample(
                    1,
                    "b.jpg",
                    '[{"box_2d": [40, 40, 60, 60], "label": "dog", "confidence": 0.9}]',
                ),
            ],
        )
        baseline_path = tmp_path / "baseline.jsonl"
        image_conditioned_path = tmp_path / "image_conditioned.jsonl"
        save_results(baseline_run, baseline_path)
        save_results(image_conditioned_run, image_conditioned_path)

        vlm_results = tmp_path / "results"
        vlm_results.mkdir()
        save_results(
            RunResult(
                model="gemini-3.5-flash",
                effort="low",
                task="detection",
                timestamp="20260729_140000",
                samples=baseline_run.samples,
            ),
            vlm_results / "detection_gemini-3.5-flash_low_20260729_140000.jsonl",
        )

        from vlm_exam.reference import leaderboard as reference_leaderboard

        baseline_rel = baseline_path.relative_to(tmp_path)
        image_conditioned_rel = image_conditioned_path.relative_to(tmp_path)
        monkeypatch.setattr(
            reference_leaderboard,
            "CANONICAL_BEST_PAIRS",
            (("sam3", str(baseline_rel), str(image_conditioned_rel)),),
        )

        leaderboard = build_mixed_detection_leaderboard(
            vlm_results,
            sample_index,
            load_config(),
            load_reference_config(),
            repo_root=tmp_path,
        )
        rows_by_key = {row.key: row for row in leaderboard.rows}
        assert rows_by_key["sam3-best"].map50 > rows_by_key["sam3-baseline"].map50
        assert rows_by_key["sam3-best"].map50 == pytest.approx(1.0)
        assert "gemini-3.5-flash" in rows_by_key

        sam3_family = leaderboard_rows_for_family(leaderboard, "sam3")
        yoloe_family = leaderboard_rows_for_family(leaderboard, "yoloe")
        sam3_reference_rows = [row for row in sam3_family.rows if row.source != "vlm"]
        yoloe_reference_rows = [row for row in yoloe_family.rows if row.source != "vlm"]
        assert len(sam3_reference_rows) == 2
        assert len(yoloe_reference_rows) == 0
        assert "gemini-3.5-flash" in {row.key for row in sam3_family.rows}

        payload = mixed_detection_leaderboard_payload(leaderboard)
        assert "families" in payload
        assert "sam3" in payload["families"]
        assert "yoloe" in payload["families"]
        assert "yoloe_gemini_focus_class_names" in payload["families"]
        assert "yoloe_gemini_focus_augmented_prompt" in payload["families"]
        assert "rows" not in payload

    def test_yoloe_gemini_focus_has_one_vlm_and_two_reference_rows_per_prompt(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        dataset_directory = tmp_path / "dataset"
        _write_detection_dataset(dataset_directory)
        task = DetectionTask()
        sample_index = build_sample_index(task.load_samples(str(dataset_directory)))

        baseline_run = RunResult(
            model="yoloe-11l-seg",
            effort=REFERENCE_EFFORT,
            task="detection",
            timestamp="20260729_120000",
            samples=[
                _sample(
                    0,
                    "a.jpg",
                    '[{"box_2d": [10, 10, 30, 30], "label": "cat", "confidence": 0.9}]',
                ),
            ],
        )
        image_conditioned_run = RunResult(
            model="yoloe-11l-seg",
            effort=REFERENCE_EFFORT,
            task="detection",
            timestamp="20260729_130000",
            samples=baseline_run.samples,
        )
        baseline_path = tmp_path / "yoloe_baseline.jsonl"
        image_conditioned_path = tmp_path / "yoloe_image_conditioned.jsonl"
        save_results(baseline_run, baseline_path)
        save_results(image_conditioned_run, image_conditioned_path)

        vlm_results = tmp_path / "results"
        vlm_results.mkdir()
        save_results(
            RunResult(
                model=YOLOE_GEMINI_FOCUS_VLM,
                effort="low",
                task="detection",
                timestamp="20260729_140000",
                samples=baseline_run.samples,
            ),
            vlm_results
            / f"detection_{YOLOE_GEMINI_FOCUS_VLM}_low_20260729_140000.jsonl",
        )
        save_results(
            RunResult(
                model="gpt-5.6-sol",
                effort="low",
                task="detection",
                timestamp="20260729_140000",
                samples=baseline_run.samples,
            ),
            vlm_results / "detection_gpt-5.6-sol_low_20260729_140000.jsonl",
        )

        from vlm_exam.reference import leaderboard as reference_leaderboard

        baseline_rel = baseline_path.relative_to(tmp_path)
        image_conditioned_rel = image_conditioned_path.relative_to(tmp_path)
        monkeypatch.setattr(
            reference_leaderboard,
            "CANONICAL_BEST_PAIRS",
            (
                ("yoloe-11l-seg", str(baseline_rel), str(image_conditioned_rel)),
                ("yoloe-26x-seg", str(baseline_rel), str(image_conditioned_rel)),
            ),
        )

        leaderboard = build_mixed_detection_leaderboard(
            vlm_results,
            sample_index,
            load_config(),
            load_reference_config(),
            repo_root=tmp_path,
        )
        class_names = leaderboard_rows_for_yoloe_gemini_focus(
            leaderboard,
            "class_names",
        )
        augmented_prompt = leaderboard_rows_for_yoloe_gemini_focus(
            leaderboard,
            "augmented_prompt",
        )
        for focus in (class_names, augmented_prompt):
            assert len(focus.rows) == 3
            assert {row.key for row in focus.rows if row.source == "vlm"} == {
                YOLOE_GEMINI_FOCUS_VLM
            }
            assert len([row for row in focus.rows if row.source != "vlm"]) == 2
        assert row_chart_label(class_names.rows[1]).startswith("YOLO")
        assert " (" not in row_chart_label(class_names.rows[1])
