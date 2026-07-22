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

import base64
import io
import math

import anthropic
from PIL import Image

from vlm_exam.providers.base import (
    REQUEST_TIMEOUT_SECONDS,
    Provider,
    RetryStats,
    Usage,
    call_with_retries,
)

_REFUSAL_TEXT = "[model refused to answer]"

RESOLUTION_TIERS: dict[str, tuple[int, int]] = {
    "standard": (1568, 1568),
    "high": (2576, 4784),
}
"""Per-tier ``(max_edge, max_tokens)`` image limits from Anthropic's docs."""

DEFAULT_RESOLUTION_TIER = "high"
"""Tier assumed when a model does not declare one."""

_MAX_EDGE, _MAX_TOKENS = RESOLUTION_TIERS[DEFAULT_RESOLUTION_TIER]


def resolution_tier_limits(tier: str) -> tuple[int, int]:
    """Return the ``(max_edge, max_tokens)`` limits for a resolution tier.

    Args:
        tier: Tier name, one of the keys in :data:`RESOLUTION_TIERS`.

    Returns:
        The maximum padded edge length and visual-token budget.

    Raises:
        ValueError: If ``tier`` is not a known resolution tier.
    """
    if tier not in RESOLUTION_TIERS:
        valid = ", ".join(sorted(RESOLUTION_TIERS))
        raise ValueError(f"Unknown resolution tier {tier!r}. Valid tiers: {valid}")
    return RESOLUTION_TIERS[tier]


def _count_image_tokens(width: int, height: int) -> int:
    return math.ceil(width / 28) * math.ceil(height / 28)


def compute_resize_dimensions(
    width: int,
    height: int,
    max_edge: int = _MAX_EDGE,
    max_tokens: int = _MAX_TOKENS,
) -> tuple[int, int]:
    """Compute the dimensions Claude resizes an image to before upload.

    Mirrors the reference implementation from the Anthropic vision
    coordinates documentation, so pixel coordinates returned by Claude
    map one-to-one onto an image pre-resized to these dimensions.

    Args:
        width: Original image width in pixels.
        height: Original image height in pixels.
        max_edge: Maximum padded edge length in pixels.
        max_tokens: Maximum visual token budget.

    Returns:
        The ``(width, height)`` Claude scales the image to.
    """

    def fits(candidate_width: int, candidate_height: int) -> bool:
        return (
            math.ceil(candidate_width / 28) * 28 <= max_edge
            and math.ceil(candidate_height / 28) * 28 <= max_edge
            and _count_image_tokens(candidate_width, candidate_height) <= max_tokens
        )

    if fits(width, height):
        return (width, height)

    if height > width:
        resized_height, resized_width = compute_resize_dimensions(
            height, width, max_edge, max_tokens
        )
        return (resized_width, resized_height)

    aspect_ratio = width / height
    low, high = 1, width
    while low + 1 < high:
        mid = (low + high) // 2
        if fits(mid, max(round(mid / aspect_ratio), 1)):
            low = mid
        else:
            high = mid
    return (low, max(round(low / aspect_ratio), 1))


class AnthropicProvider(Provider):
    """Anthropic Claude provider.

    Pre-resizes images to the dimensions Claude sees after its internal
    resize, so pixel coordinates the model returns map one-to-one onto
    the uploaded image. The resize target depends on the model's
    resolution tier (see :data:`RESOLUTION_TIERS`).
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        provider_model_id: str | None = None,
        resolution_tier: str = DEFAULT_RESOLUTION_TIER,
    ) -> None:
        self._model = model
        self._wire_model_id = provider_model_id or model
        self._max_edge, self._max_tokens = resolution_tier_limits(resolution_tier)
        self._client = anthropic.Anthropic(
            api_key=api_key,
            timeout=REQUEST_TIMEOUT_SECONDS,
            max_retries=0,
        )

    @property
    def model(self) -> str:
        return self._model

    def uploaded_image_size(self, image: Image.Image) -> tuple[int, int]:
        return compute_resize_dimensions(*image.size, self._max_edge, self._max_tokens)

    def _prepare_image(self, image: Image.Image) -> str:
        target_size = self.uploaded_image_size(image)
        if target_size != image.size:
            image = image.resize(target_size, Image.LANCZOS)

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.standard_b64encode(buffer.getvalue()).decode("utf-8")

    def predict(
        self,
        image: Image.Image,
        prompt: str,
        effort: str,
    ) -> tuple[str, Usage, RetryStats]:
        base64_data = self._prepare_image(image)

        message, retry_stats = call_with_retries(
            lambda: self._client.messages.create(
                model=self._wire_model_id,
                max_tokens=4096,
                output_config={"effort": effort},
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": base64_data,
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            )
        )

        text_blocks = [block.text for block in message.content if block.type == "text"]
        # Claude answers refusals with stop_reason "refusal" and no content
        # blocks; record that as a wrong answer instead of a run error so
        # resume logic does not retry it forever.
        answer = text_blocks[0].strip() if text_blocks else _REFUSAL_TEXT

        return (
            answer,
            Usage(
                input_tokens=message.usage.input_tokens,
                output_tokens=message.usage.output_tokens,
            ),
            retry_stats,
        )
