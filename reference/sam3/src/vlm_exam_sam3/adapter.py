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

import numpy as np
import torch
from PIL import Image

from vlm_exam.reference.base import ReferencePrediction
from vlm_exam.reference.config import ReferenceModelConfig


class Sam3Adapter:
    """Hugging Face SAM 3 reference adapter."""

    def __init__(self, model_config: ReferenceModelConfig, device: str) -> None:
        from transformers import Sam3Model, Sam3Processor

        self._model_config = model_config
        self._device = device
        self._threshold = model_config.inference.conf
        self._model = Sam3Model.from_pretrained(
            model_config.checkpoint,
            revision=model_config.checkpoint_revision,
        )
        self._processor = Sam3Processor.from_pretrained(
            model_config.checkpoint,
            revision=model_config.checkpoint_revision,
        )
        self._model.to(device)
        self._model.eval()
        self._prompt_texts: tuple[str, ...] = ()

    @property
    def classes_processed(self) -> str:
        """Return how class prompts are processed during inference."""
        return "individually"

    @property
    def device(self) -> str:
        """Device backend used for inference."""
        return self._device

    def set_vocabulary(self, class_names: tuple[str, ...]) -> None:
        if class_names == self._prompt_texts:
            return
        self._prompt_texts = class_names

    def predict(self, image: Image.Image) -> ReferencePrediction:
        if not self._prompt_texts:
            return ReferencePrediction(
                boxes_xyxy=np.empty((0, 4), dtype=np.float32),
                labels=(),
                confidences=np.empty((0,), dtype=np.float32),
                raw={"checkpoint": self._model_config.checkpoint},
            )

        boxes_list: list[np.ndarray] = []
        labels_list: list[str] = []
        scores_list: list[float] = []

        for prompt_text in self._prompt_texts:
            inputs = self._processor(
                images=image,
                text=prompt_text,
                return_tensors="pt",
            )
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._model(**inputs)

            original_sizes = inputs.get("original_sizes")
            target_sizes = (
                original_sizes.tolist()
                if original_sizes is not None
                else [[image.height, image.width]]
            )
            results = self._processor.post_process_instance_segmentation(
                outputs,
                threshold=self._threshold,
                mask_threshold=self._threshold,
                target_sizes=target_sizes,
            )[0]

            if len(results["boxes"]) == 0:
                continue

            boxes = results["boxes"].detach().cpu().numpy().astype(np.float32)
            scores = results["scores"].detach().cpu().numpy().astype(np.float32)
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, image.width)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, image.height)
            valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes = boxes[valid]
            scores = scores[valid]
            if len(boxes) == 0:
                continue
            boxes_list.append(boxes)
            scores_list.extend(float(score) for score in scores)
            labels_list.extend(prompt_text for _ in range(len(boxes)))

        if not boxes_list:
            return ReferencePrediction(
                boxes_xyxy=np.empty((0, 4), dtype=np.float32),
                labels=(),
                confidences=np.empty((0,), dtype=np.float32),
                raw={
                    "checkpoint": self._model_config.checkpoint,
                    "num_prompts": len(self._prompt_texts),
                },
            )

        return ReferencePrediction(
            boxes_xyxy=np.vstack(boxes_list),
            labels=tuple(labels_list),
            confidences=np.array(scores_list, dtype=np.float32),
            raw={
                "checkpoint": self._model_config.checkpoint,
                "num_prompts": len(self._prompt_texts),
                "num_boxes": len(labels_list),
            },
        )
