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
from click.testing import CliRunner

from vlm_exam.cli import main
from vlm_exam.config import load_config
from vlm_exam.metrics import build_latest_runs_index
from vlm_exam.reference.base import ReferencePrediction
from vlm_exam.reference.config import (
    ReferenceConfig,
    ReferenceInferenceConfig,
    ReferenceModelConfig,
    assert_no_vlm_model_overlap,
    load_reference_config,
)
from vlm_exam.reference.manifest import build_run_manifest, save_manifest
from vlm_exam.reference.serializer import serialize_reference_prediction
from vlm_exam.reference.validate import validate_reference_run
from vlm_exam.reference.visualization import (
    build_reference_card_config,
    detection_labels_for_card,
    prompt_label_map_from_metadata,
    resolve_card_confidence_threshold,
)
from vlm_exam.results import (
    RunResult,
    SampleResult,
    load_results_directory,
    save_results,
)
from vlm_exam.tasks.detection import (
    DetectionCoordinateFormat,
    parse_prediction,
)


class TestReferenceConfig:
    def test_reference_keys_do_not_overlap_vlm_models(self) -> None:
        reference_config = load_reference_config()
        vlm_config = load_config()
        assert_no_vlm_model_overlap(reference_config, set(vlm_config.models))

    def test_sam3_config_loads_with_null_inference_fields(self) -> None:
        reference_config = load_reference_config()
        sam3 = reference_config.models["sam3"]
        assert sam3.adapter == "sam3"
        assert sam3.checkpoint == "facebook/sam3"
        assert sam3.lab == "meta"
        assert sam3.card_confidence_threshold == pytest.approx(0.5)
        assert sam3.inference.iou is None
        assert sam3.inference.imgsz is None
        assert sam3.inference.max_det is None

    def test_yoloe_config_loads_card_confidence_threshold(self) -> None:
        reference_config = load_reference_config()
        yoloe = reference_config.models["yoloe-11l-seg"]
        assert yoloe.card_confidence_threshold == pytest.approx(0.01)
        assert yoloe.lab == "ultralytics"


class TestReferenceVisualization:
    def test_build_reference_card_config_uses_sam3_and_meta_lab(self) -> None:
        reference_config = load_reference_config()
        vlm_config = load_config()
        card_config = build_reference_card_config("sam3", reference_config, vlm_config)
        assert card_config.models["sam3"].name == "SAM 3"
        assert card_config.models["sam3"].lab == "meta"
        assert card_config.labs["meta"].name == "Meta"

    def test_build_reference_card_config_requires_lab(self) -> None:
        reference_config = ReferenceConfig(
            models={
                "no-lab": ReferenceModelConfig(
                    key="no-lab",
                    name="No Lab",
                    lab=None,
                    family="test",
                    adapter="test",
                    checkpoint="test",
                    coordinate_format=DetectionCoordinateFormat.XYXY_ABSOLUTE_ORIGINAL_IMAGE,
                    supported_devices=("cpu",),
                    inference=ReferenceInferenceConfig(
                        conf=0.1,
                        iou=None,
                        imgsz=None,
                        max_det=None,
                        agnostic_nms=None,
                    ),
                )
            }
        )
        vlm_config = load_config()
        with pytest.raises(ValueError, match="has no lab"):
            build_reference_card_config("no-lab", reference_config, vlm_config)

    def test_resolve_card_confidence_threshold_uses_config_default(self) -> None:
        reference_config = load_reference_config()
        sam3_threshold = resolve_card_confidence_threshold("sam3", reference_config)
        assert sam3_threshold == pytest.approx(0.5)
        yoloe_threshold = resolve_card_confidence_threshold(
            "yoloe-11l-seg", reference_config
        )
        assert yoloe_threshold == pytest.approx(0.01)

    def test_resolve_card_confidence_threshold_cli_override(self) -> None:
        reference_config = load_reference_config()
        assert resolve_card_confidence_threshold(
            "sam3", reference_config, override=0.25
        ) == pytest.approx(0.25)

    def test_prompt_label_map_from_metadata(self) -> None:
        prompt_map = prompt_label_map_from_metadata(
            {
                "prompt_class_names": ["barcode", "egg"],
                "prompt_texts": ["rectangular barcode sticker", "brown chicken egg"],
            }
        )
        assert prompt_map == {
            "barcode": "rectangular barcode sticker",
            "egg": "brown chicken egg",
        }

    def test_detection_labels_for_card_uses_augmented_prompts(self) -> None:
        import supervision as sv

        detections = sv.Detections(
            xyxy=np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32),
            class_id=np.array([0]),
            data={"class_name": np.array(["barcode"])},
        )
        labels = detection_labels_for_card(
            detections,
            ["barcode"],
            label_classes="augmented",
            prompt_label_map={"barcode": "rectangular barcode sticker"},
        )
        assert labels == ["rectangular barcode sticker"]

    def test_reference_detection_visualize_smoke(self, tmp_path: Path) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        results_file = (
            repo_root
            / "results-reference-smoke"
            / "detection_sam3_reference_20260729_123536.jsonl"
        )
        if not results_file.exists():
            pytest.skip("Smoke results file not available.")
        output_directory = tmp_path / "cards"
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "reference-detection-visualize",
                "--results-file",
                str(results_file),
                "--output-directory",
                str(output_directory),
                "--max-images",
                "1",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "card mAP uses native per-image scores" in result.output
        png_files = list(output_directory.glob("*.png"))
        assert len(png_files) == 1

    def test_reference_detection_visualize_confidence_threshold_smoke(
        self,
        tmp_path: Path,
    ) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        results_file = (
            repo_root
            / "results-reference-smoke"
            / "detection_sam3_reference_20260729_123536.jsonl"
        )
        if not results_file.exists():
            pytest.skip("Smoke results file not available.")
        output_directory = tmp_path / "cards-conf050"
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "reference-detection-visualize",
                "--results-file",
                str(results_file),
                "--output-directory",
                str(output_directory),
                "--max-images",
                "1",
                "--confidence-threshold",
                "0.5",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "card mAP uses native per-image scores" in result.output
        assert list(output_directory.glob("*.png"))


class TestReferenceSerializer:
    def test_round_trip_with_confidence(self) -> None:
        prediction = ReferencePrediction(
            boxes_xyxy=np.array([[10.0, 20.0, 50.0, 60.0]], dtype=np.float32),
            labels=("cat",),
            confidences=np.array([0.91], dtype=np.float32),
        )
        serialized = serialize_reference_prediction(prediction)
        detections = parse_prediction(
            serialized,
            (100, 100),
            ["cat"],
            coordinate_format=DetectionCoordinateFormat.XYXY_ABSOLUTE_ORIGINAL_IMAGE,
        )
        assert len(detections) == 1
        assert detections.confidence is not None
        assert float(detections.confidence[0]) == pytest.approx(0.91)


class _RecordingAdapter:
    def __init__(self) -> None:
        self.vocabulary: tuple[str, ...] = ()
        self.set_calls: list[tuple[str, ...]] = []

    @property
    def device(self) -> str:
        return "cpu"

    def set_vocabulary(self, class_names: tuple[str, ...]) -> None:
        if class_names == self.vocabulary:
            return
        self.vocabulary = class_names
        self.set_calls.append(class_names)

    def predict(self, image: object) -> ReferencePrediction:
        return ReferencePrediction(
            boxes_xyxy=np.array([[1.0, 1.0, 5.0, 5.0]], dtype=np.float32),
            labels=(self.vocabulary[0],),
            confidences=np.array([0.9], dtype=np.float32),
        )


class TestRunnerVocabulary:
    def test_repeated_vocabulary_is_reset_after_intervening_image(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from PIL import Image

        from vlm_exam.reference import runner as reference_runner

        dataset = tmp_path / "dataset"
        dataset.mkdir()
        annotations = {
            "categories": [
                {"id": 0, "name": "vlm-exam", "supercategory": "none"},
                {"id": 1, "name": "cat", "supercategory": "vlm-exam"},
                {"id": 2, "name": "dog", "supercategory": "vlm-exam"},
            ],
            "images": [
                {"id": 1, "file_name": "a.jpg", "width": 32, "height": 32},
                {"id": 2, "file_name": "b.jpg", "width": 32, "height": 32},
                {"id": 3, "file_name": "c.jpg", "width": 32, "height": 32},
            ],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 1, 4, 4]},
                {"id": 2, "image_id": 2, "category_id": 2, "bbox": [1, 1, 4, 4]},
                {"id": 3, "image_id": 3, "category_id": 1, "bbox": [1, 1, 4, 4]},
            ],
        }
        (dataset / "_annotations.coco.json").write_text(json.dumps(annotations))
        for image_name in ("a.jpg", "b.jpg", "c.jpg"):
            Image.new("RGB", (32, 32)).save(dataset / image_name)

        adapter = _RecordingAdapter()
        monkeypatch.setattr(
            reference_runner,
            "create_reference_adapter",
            lambda model_config, device: adapter,
        )

        model_config = ReferenceModelConfig(
            key="fake-reference",
            name="Fake Reference",
            lab=None,
            family="test",
            adapter="test",
            checkpoint="fake.pt",
            coordinate_format=DetectionCoordinateFormat.XYXY_ABSOLUTE_ORIGINAL_IMAGE,
            supported_devices=("cpu",),
            inference=ReferenceInferenceConfig(
                conf=0.1,
                iou=None,
                imgsz=None,
                max_det=None,
                agnostic_nms=None,
            ),
        )
        run, _ = reference_runner.run_reference_benchmark(
            model_config=model_config,
            dataset_directory=str(dataset),
            output_path=tmp_path / "run.jsonl",
            timestamp="20260731_120000",
            device="cpu",
            verbose=False,
        )

        assert run.timestamp == "20260731_120000"
        assert adapter.set_calls == [("cat",), ("dog",), ("cat",)]
        for sample in run.samples:
            predicted_labels = {box["label"] for box in json.loads(sample.predicted)}
            assert predicted_labels <= set(sample.metadata["prompt_class_names"])


class TestReferenceValidation:
    def test_valid_reference_run_passes(self, tmp_path: Path) -> None:
        dataset = tmp_path / "dataset"
        dataset.mkdir()
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
        (dataset / "_annotations.coco.json").write_text(json.dumps(annotations))

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
                        '[{"box_2d": [10, 10, 30, 30], "label": "cat", '
                        '"confidence": 0.5}]'
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
        results_path = (
            tmp_path / "detection_yoloe-11l-seg_reference_20260729_120000.jsonl"
        )
        save_results(run, results_path)
        reference_config = load_reference_config()
        manifest = build_run_manifest(
            model_config=reference_config.models["yoloe-11l-seg"],
            effort="reference",
            task="detection",
            timestamp="20260729_120000",
            dataset_directory=str(dataset),
            prompt_classes="image",
            device="cpu",
            precision="float32",
        )
        manifest.dataset_image_count = 1
        manifest.completed_sample_count = 1
        save_manifest(manifest, results_path.with_suffix(".manifest.json"))

        report = validate_reference_run(results_path, str(dataset))
        assert report.ok


class TestLeaderboardGuards:
    def test_reference_runs_are_ignored_by_vlm_leaderboard(
        self,
        tmp_path: Path,
    ) -> None:
        vlm_config = load_config()
        reference_run = RunResult(
            model="yoloe-11l-seg",
            effort="reference",
            task="detection",
            timestamp="20260729_120000",
            samples=[],
        )
        save_results(
            reference_run,
            tmp_path / "detection_yoloe-11l-seg_reference_20260729_120000.jsonl",
        )
        runs = load_results_directory(tmp_path)
        latest = build_latest_runs_index(runs, vlm_config)
        assert latest == {}
