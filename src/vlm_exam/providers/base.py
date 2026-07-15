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

from abc import ABC, abstractmethod
from dataclasses import dataclass

from PIL import Image


@dataclass(frozen=True)
class Usage:
    """Token usage reported by a model after a single prediction."""

    input_tokens: int
    output_tokens: int


class Provider(ABC):
    """Abstract base for VLM inference providers."""

    @property
    @abstractmethod
    def model(self) -> str:
        """Model identifier this provider instance serves."""
        ...

    @abstractmethod
    def predict(
        self,
        image: Image.Image,
        prompt: str,
        effort: str,
    ) -> tuple[str, Usage]:
        """Run inference on an image with a text prompt.

        Args:
            image: Input image in RGB mode.
            prompt: Text prompt to send alongside the image.
            effort: Effort level (e.g. ``"low"``, ``"high"``).

        Returns:
            A tuple of ``(answer_text, usage)``.
        """
        ...

    def uploaded_image_size(self, image: Image.Image) -> tuple[int, int] | None:
        """Return the pixel dimensions the provider uploads for an image.

        Providers that resize an image before sending it override this so
        callers can map returned pixel coordinates back onto the original.

        Args:
            image: Input image in RGB mode.

        Returns:
            The ``(width, height)`` actually sent to the provider, or
            ``None`` when the provider sends the image unchanged.
        """
        return None
