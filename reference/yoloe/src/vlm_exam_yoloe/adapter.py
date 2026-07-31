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

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from vlm_exam.reference.base import ReferencePrediction
from vlm_exam.reference.config import ReferenceModelConfig


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_checkpoint(checkpoint: str) -> Path | None:
    configured = Path(checkpoint).expanduser()
    candidates = [configured]
    if not configured.is_absolute():
        adapter_path = Path(__file__).resolve()
        candidates.extend(
            [
                adapter_path.parents[2] / configured,
                adapter_path.parents[4] / configured,
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


class YoloeAdapter:
    """Ultralytics YOLO-E reference adapter."""

    def __init__(self, model_config: ReferenceModelConfig, device: str) -> None:
        from ultralytics import YOLOE
        from ultralytics.utils.downloads import attempt_download_asset

        self._model_config = model_config
        self._device = device
        checkpoint = _resolve_checkpoint(model_config.checkpoint)
        if checkpoint is None:
            downloaded = Path(attempt_download_asset(model_config.checkpoint))
            if not downloaded.exists():
                raise FileNotFoundError(
                    f"YOLO-E checkpoint could not be downloaded: "
                    f"{model_config.checkpoint}"
                )
            checkpoint = downloaded.resolve()
        if model_config.checkpoint_sha256 is not None:
            actual_sha256 = _file_sha256(checkpoint)
            if actual_sha256 != model_config.checkpoint_sha256:
                raise ValueError(
                    f"Checkpoint SHA256 mismatch for {checkpoint}: "
                    f"expected {model_config.checkpoint_sha256}, got {actual_sha256}."
                )
        self._model = YOLOE(str(checkpoint))
        self._model.to(device)
        self._class_names: tuple[str, ...] = ()

    @property
    def classes_processed(self) -> str:
        """Return how class prompts are processed during inference."""
        return "together"

    @property
    def device(self) -> str:
        """Device backend used for inference."""
        return self._device

    def set_vocabulary(self, class_names: tuple[str, ...]) -> None:
        if class_names == self._class_names:
            return
        self._class_names = class_names
        names = list(class_names)
        if self._device == "mps":
            self._model.to("cpu")
            self._model.set_classes(names)
            self._model.to(self._device)
        else:
            self._model.set_classes(names)

    def predict(self, image: Image.Image) -> ReferencePrediction:
        inference = self._model_config.inference
        predict_kwargs: dict[str, object] = {
            "conf": inference.conf,
            "iou": inference.iou,
            "imgsz": inference.imgsz,
            "max_det": inference.max_det,
            "device": self._device,
            "verbose": False,
        }
        if inference.agnostic_nms is not None:
            predict_kwargs["agnostic_nms"] = inference.agnostic_nms

        results = self._model.predict(image, **predict_kwargs)
        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return ReferencePrediction(
                boxes_xyxy=np.empty((0, 4), dtype=np.float32),
                labels=(),
                confidences=np.empty((0,), dtype=np.float32),
                raw={"checkpoint": self._model_config.checkpoint},
            )

        boxes = result.boxes.xyxy.cpu().numpy().astype(np.float32)
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy().astype(np.float32)
        labels = tuple(self._class_names[class_id] for class_id in class_ids)
        return ReferencePrediction(
            boxes_xyxy=boxes,
            labels=labels,
            confidences=confidences,
            raw={
                "checkpoint": self._model_config.checkpoint,
                "num_boxes": len(labels),
            },
        )
