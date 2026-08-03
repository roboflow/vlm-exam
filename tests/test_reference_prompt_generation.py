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
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import supervision as sv
from PIL import Image

from reference.scripts import generate_image_conditioned_prompts as generator
from reference.scripts import render_prompt_probe_review as renderer
from vlm_exam.tasks.detection import DetectionSample


def _sample(box_count: int = 1) -> DetectionSample:
    cat_boxes = [[index, index, index + 10, index + 10] for index in range(box_count)]
    boxes = np.array([*cat_boxes, [50, 50, 70, 70]], dtype=np.float32)
    class_ids = np.array([*[0] * box_count, 1], dtype=int)
    return DetectionSample(
        image_path="/tmp/a.jpg",
        image_width=100,
        image_height=100,
        classes=("cat", "dog"),
        ground_truth=sv.Detections(xyxy=boxes, class_id=class_ids),
    )


class TestConditioning:
    def test_none_mode_does_not_leak_instance_counts(self) -> None:
        _, context = generator._conditioning_input(
            Image.new("RGB", (100, 100)),
            _sample(box_count=20),
            ("cat", "dog"),
            "none",
        )

        assert "showing" not in context
        assert "20" not in context

    def test_overlay_uses_distinct_class_colors(self) -> None:
        image = Image.new("RGB", (100, 100), "white")
        overlay, context = generator._conditioning_input(
            image,
            _sample(),
            ("cat", "dog"),
            "overlay",
        )

        assert overlay.getpixel((0, 0)) != overlay.getpixel((50, 50))
        assert 'C1: "cat"' in context
        assert 'C2: "dog"' in context


class TestGeneration:
    def test_invalid_response_is_retried_with_feedback(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        responses = [
            {
                "prompts": [
                    {
                        "class_name": "cat",
                        "primary": "animal in box",
                        "variants": ["small cat"],
                    },
                    {
                        "class_name": "dog",
                        "primary": "animal in box",
                        "variants": ["small dog"],
                    },
                ]
            },
            {
                "prompts": [
                    {
                        "class_name": "cat",
                        "primary": "striped house cat",
                        "variants": ["small domestic cat"],
                    },
                    {
                        "class_name": "dog",
                        "primary": "shaggy brown dog",
                        "variants": ["floppy eared dog"],
                    },
                ]
            },
        ]
        prompts: list[str] = []

        class _Models:
            def generate_content(self, **kwargs: Any) -> SimpleNamespace:
                prompts.append(kwargs["contents"][1])
                return SimpleNamespace(text=json.dumps(responses.pop(0)))

        def call_once(function: Callable[[], Any]) -> tuple[Any, int]:
            return function(), 0

        client = SimpleNamespace(models=_Models())
        monkeypatch.setattr(
            generator,
            "call_with_retries",
            call_once,
        )

        entries, attempts = generator._complete_image(
            client,
            "gemini-3.5-flash",
            Image.new("RGB", (100, 100)),
            _sample(),
            ("cat", "dog"),
            "none",
        )

        assert attempts == 2
        assert entries["cat"][0] == "striped house cat"
        assert "previous response was invalid" in prompts[1]

    def test_numeric_identity_must_be_preserved(self) -> None:
        issues = generator._validate_phrase("small silver coin", "7 euro coin")

        assert any("identity token" in issue for issue in issues)

    def test_visual_description_without_exact_class_token_is_allowed(self) -> None:
        issues = generator._validate_phrase(
            "orange metal basketball hoop",
            "basket rim",
        )

        assert issues == []

    @pytest.mark.parametrize(
        ("phrase", "class_name"),
        [
            ("walking pedestrian from above", "person"),
            ("large box cargo truck", "truck"),
        ],
    )
    def test_viewpoint_and_box_truck_are_not_annotation_leaks(
        self,
        phrase: str,
        class_name: str,
    ) -> None:
        assert generator._validate_phrase(phrase, class_name) == []

    def test_bounding_box_reference_is_rejected(self) -> None:
        issues = generator._validate_phrase("cat in red bounding box", "cat")

        assert any("annotation reference" in issue for issue in issues)


class TestResumeValidation:
    def test_incompatible_manifest_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.json"
        expected = {
            "version": "v2",
            "generation_model": "gemini-3.5-flash",
            "generation_prompt_version": generator.GENERATION_PROMPT_VERSION,
            "dataset_annotations_sha256": "dataset",
            "prompt_classes": "image",
            "conditioning": "none",
            "selected_images_sha256": "images",
            "selected_image_contents_sha256": "contents",
            "generation_config_sha256": "config",
        }
        generator._save_json(path, expected)
        incompatible = {**expected, "conditioning": "overlay"}

        with pytest.raises(ValueError, match="conditioning"):
            generator._validate_existing_manifest(path, incompatible)


class TestReviewRendering:
    def test_duplicate_prompt_records_are_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "prompts.jsonl"
        record = {"image": "a.jpg", "class_name": "cat", "primary": "striped cat"}
        path.write_text(f"{json.dumps(record)}\n{json.dumps(record)}\n")

        with pytest.raises(ValueError, match="Duplicate prompt"):
            renderer._load_prompts(path)

    def test_mislabeled_probe_manifest_is_rejected(self, tmp_path: Path) -> None:
        prompt_path = tmp_path / "prompts.jsonl"
        prompt_path.write_text("")
        (tmp_path / "manifest.json").write_text(
            json.dumps(
                {
                    "version": "v2",
                    "generation_model": "gemini-3.5-flash",
                    "generation_prompt_version": "image_conditioned_v2",
                    "generation_config_sha256": "config",
                    "prompt_classes": "image",
                    "conditioning": "none",
                    "dataset_annotations_sha256": "dataset",
                    "selected_images_sha256": "images",
                    "selected_image_contents_sha256": "contents",
                }
            )
        )

        with pytest.raises(ValueError, match="conditioning"):
            renderer._validate_probe_manifest(
                prompt_path,
                mode="overlay",
                dataset_sha256="dataset",
                selected_images_sha256="images",
                selected_image_contents_sha256="contents",
            )

    def test_mislabeled_prompt_record_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "prompts.jsonl"
        path.write_text(
            json.dumps(
                {
                    "image": "a.jpg",
                    "class_name": "cat",
                    "primary": "striped cat",
                    "generation_model": "gemini-3.5-flash",
                    "generation_prompt_version": "image_conditioned_v2",
                    "conditioning": "none",
                }
            )
            + "\n"
        )

        with pytest.raises(ValueError, match="conditioning"):
            renderer._load_prompts(
                path,
                expected_metadata={
                    "generation_model": "gemini-3.5-flash",
                    "generation_prompt_version": "image_conditioned_v2",
                    "conditioning": "overlay",
                },
            )
