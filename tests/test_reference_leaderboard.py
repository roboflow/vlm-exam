# Copyright 2026 Roboflow, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
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
from vlm_exam.reference.constants import CANONICAL_REFERENCE_RUNS, REFERENCE_EFFORT
from vlm_exam.reference.leaderboard import (
    build_mixed_detection_leaderboard,
    build_mixed_leaderboard_config,
    family_reference_keys,
    leaderboard_rows_for_family,
    mixed_detection_leaderboard_payload,
)
from vlm_exam.reference.manifest import (
    build_run_manifest,
    load_manifest,
    manifest_path_for_results,
    save_manifest,
)
from vlm_exam.reference.prompts import file_sha256
from vlm_exam.reference.report import _latest_runs_by_key, build_reference_report_rows
from vlm_exam.results import (
    RunResult,
    SampleResult,
    is_failed_sample,
    load_results,
    save_results,
)
from vlm_exam.tasks.detection import DetectionTask, build_sample_index


def _write_detection_dataset(dataset_directory: Path) -> None:
    annotations = {
        "categories": [
            {"id": 0, "name": "vlm-exam", "supercategory": "none"},
            {"id": 1, "name": "cat", "supercategory": "vlm-exam"},
        ],
        "images": [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 100}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20]}
        ],
    }
    dataset_directory.mkdir(parents=True, exist_ok=True)
    (dataset_directory / "_annotations.coco.json").write_text(json.dumps(annotations))
    (dataset_directory / "a.jpg").write_bytes(b"\xff\xd8\xff\xd9")


def _run(model: str) -> RunResult:
    return RunResult(
        model=model,
        effort=REFERENCE_EFFORT,
        task="detection",
        timestamp="20260803_120000",
        samples=[
            SampleResult(
                index=0,
                image="a.jpg",
                expected="",
                predicted=(
                    '[{"box_2d": [10, 10, 30, 30], "label": "cat", "confidence": 0.9}]'
                ),
                correct=True,
                input_tokens=0,
                output_tokens=0,
                metadata={
                    "coordinate_format": "xyxy_absolute_original_image",
                    "reference": True,
                },
            )
        ],
    )


class TestMixedDetectionLeaderboard:
    def test_canonical_reference_asset_contract(self) -> None:
        root = Path(__file__).parents[1]
        assert len(CANONICAL_REFERENCE_RUNS) == 12
        assert len(_latest_runs_by_key(root / "reference/results")) == 12

        for model, mode, relative_path in CANONICAL_REFERENCE_RUNS:
            path = root / relative_path
            run = load_results(path)
            manifest = load_manifest(manifest_path_for_results(path))
            assert run.model == model
            assert manifest.model == model
            assert len(run.samples) == 250
            assert len({sample.image for sample in run.samples}) == 250
            assert not any(is_failed_sample(sample) for sample in run.samples)
            assert manifest.completed_sample_count == 250
            assert manifest.failed_sample_count == 0
            if mode == "class-names":
                assert manifest.prompt_asset_type == "none"
                continue
            assert manifest.prompt_set_version == mode
            prompt_path = root / str(manifest.prompt_set_path)
            assert prompt_path.exists()
            assert file_sha256(prompt_path) == manifest.prompt_set_sha256

    def test_build_config_adds_canonical_reference_modes(self) -> None:
        chart_config = build_mixed_leaderboard_config(
            load_config(),
            load_reference_config(),
        )

        for model in ("sam3", "yoloe-11l-seg", "yoloe-26x-seg"):
            for mode in ("class-names", "v1", "v2-none", "v2-overlay"):
                assert f"{model}-{mode}" in chart_config.models
        assert chart_config.models["sam3-class-names"].name == "SAM 3 (class names)"
        assert chart_config.models["sam3-v2-overlay"].name == "SAM 3 (v2 overlay)"

    def test_unlisted_reference_without_lab_is_ignored(self) -> None:
        reference_config = load_reference_config()
        models = dict(reference_config.models)
        models["experimental"] = replace(models["sam3"], key="experimental", lab=None)

        chart_config = build_mixed_leaderboard_config(
            load_config(),
            ReferenceConfig(models=models, labs=reference_config.labs),
        )

        assert "experimental-class-names" not in chart_config.models

    def test_family_reference_key_counts(self) -> None:
        assert len(family_reference_keys("sam3")) == 4
        assert len(family_reference_keys("yoloe")) == 8

    def test_canonical_modes_are_reported(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        dataset_directory = tmp_path / "dataset"
        _write_detection_dataset(dataset_directory)
        sample_index = build_sample_index(
            DetectionTask().load_samples(str(dataset_directory))
        )
        canonical_runs = []
        for mode in ("class-names", "v1", "v2-none", "v2-overlay"):
            path = tmp_path / f"detection_sam3_reference_{mode}.jsonl"
            save_results(_run("sam3"), path)
            manifest = build_run_manifest(
                model_config=load_reference_config().models["sam3"],
                effort=REFERENCE_EFFORT,
                task="detection",
                timestamp="20260803_120000",
                dataset_directory=str(dataset_directory),
                prompt_classes="image",
                classes_processed="individually",
                device="cpu",
                precision="float32",
            )
            manifest.dataset_image_count = 1
            manifest.completed_sample_count = 1
            if mode != "class-names":
                manifest.prompt_asset_type = "image_conditioned"
                manifest.prompt_set_version = mode
            save_manifest(manifest, manifest_path_for_results(path))
            canonical_runs.append(("sam3", mode, path.name))

        vlm_results = tmp_path / "results"
        vlm_results.mkdir()
        save_results(
            replace(_run("gemini-3.5-flash"), effort="low"),
            vlm_results / "detection_gemini-3.5-flash_low_20260803_120000.jsonl",
        )

        from vlm_exam.reference import leaderboard as reference_leaderboard

        monkeypatch.setattr(
            reference_leaderboard,
            "CANONICAL_REFERENCE_RUNS",
            tuple(canonical_runs),
        )
        leaderboard = build_mixed_detection_leaderboard(
            vlm_results,
            sample_index,
            load_config(),
            load_reference_config(),
            repo_root=tmp_path,
        )

        sam3 = leaderboard_rows_for_family(leaderboard, "sam3")
        reference_rows = [row for row in sam3.rows if row.source == "reference"]
        assert {row.key for row in reference_rows} == {
            "sam3-class-names",
            "sam3-v1",
            "sam3-v2-none",
            "sam3-v2-overlay",
        }
        assert "gemini-3.5-flash" in {row.key for row in sam3.rows}

        report_rows = build_reference_report_rows(
            vlm_results,
            tmp_path,
            str(dataset_directory),
            load_config(),
        )
        assert {row.model for row in report_rows if row.run_type == "reference"} == {
            "sam3 (class names)",
            "sam3 (v1)",
            "sam3 (v2 none)",
            "sam3 (v2 overlay)",
        }

        payload = mixed_detection_leaderboard_payload(leaderboard)
        assert set(payload["families"]) == {"sam3", "yoloe"}
