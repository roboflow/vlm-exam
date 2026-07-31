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

import hashlib
import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from vlm_exam.reference.config import (
    ReferenceInferenceConfig,
    ReferenceModelConfig,
)
from vlm_exam.tasks.detection import DetectionCoordinateFormat

_ADAPTER_PATH = (
    Path(__file__).resolve().parents[1]
    / "reference"
    / "yoloe"
    / "src"
    / "vlm_exam_yoloe"
    / "adapter.py"
)
_ADAPTER_SPEC = importlib.util.spec_from_file_location(
    "tested_yoloe_adapter", _ADAPTER_PATH
)
assert _ADAPTER_SPEC is not None and _ADAPTER_SPEC.loader is not None
_ADAPTER_MODULE = importlib.util.module_from_spec(_ADAPTER_SPEC)
_ADAPTER_SPEC.loader.exec_module(_ADAPTER_MODULE)
YoloeAdapter = _ADAPTER_MODULE.YoloeAdapter


def _model_config(checkpoint: Path, sha256: str) -> ReferenceModelConfig:
    return ReferenceModelConfig(
        key="yoloe",
        name="YOLO-E",
        lab="ultralytics",
        family="yoloe",
        adapter="yoloe",
        checkpoint=str(checkpoint),
        checkpoint_sha256=sha256,
        coordinate_format=DetectionCoordinateFormat.XYXY_ABSOLUTE_ORIGINAL_IMAGE,
        supported_devices=("cpu",),
        inference=ReferenceInferenceConfig(
            conf=0.01,
            iou=0.7,
            imgsz=640,
            max_det=300,
            agnostic_nms=None,
        ),
    )


def _install_ultralytics(
    monkeypatch: pytest.MonkeyPatch,
    model_factory: Callable[[str], Any],
    downloader: Callable[[str], str],
) -> None:
    ultralytics = ModuleType("ultralytics")
    ultralytics.YOLOE = model_factory
    utils = ModuleType("ultralytics.utils")
    downloads = ModuleType("ultralytics.utils.downloads")
    downloads.attempt_download_asset = downloader
    monkeypatch.setitem(sys.modules, "ultralytics", ultralytics)
    monkeypatch.setitem(sys.modules, "ultralytics.utils", utils)
    monkeypatch.setitem(sys.modules, "ultralytics.utils.downloads", downloads)


class TestYoloeCheckpointVerification:
    def test_hash_mismatch_prevents_model_construction(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        checkpoint = tmp_path / "model.pt"
        checkpoint.write_bytes(b"unexpected")
        constructed: list[str] = []

        def model_factory(path: str) -> object:
            constructed.append(path)
            return object()

        _install_ultralytics(
            monkeypatch,
            model_factory,
            lambda path: path,
        )

        with pytest.raises(ValueError, match="SHA256 mismatch"):
            YoloeAdapter(_model_config(checkpoint, "0" * 64), "cpu")

        assert constructed == []

    def test_downloaded_checkpoint_is_verified_before_loading(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        checkpoint = tmp_path / "downloaded.pt"
        checkpoint.write_bytes(b"verified")
        expected_sha256 = hashlib.sha256(b"verified").hexdigest()
        constructed: list[str] = []

        class _Model:
            def to(self, device: str) -> None:
                assert device == "cpu"

        def model_factory(path: str) -> _Model:
            constructed.append(path)
            return _Model()

        _install_ultralytics(
            monkeypatch,
            model_factory,
            lambda path: str(checkpoint),
        )
        missing = tmp_path / "missing.pt"

        YoloeAdapter(_model_config(missing, expected_sha256), "cpu")

        assert constructed == [str(checkpoint.resolve())]
