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

import base64
import io

from PIL import Image

OPENAI_MAX_EDGE_PIXELS = 2048
"""Maximum longest edge OpenAI uploads should use for stable detection."""

OPENROUTER_MAX_BASE64_BYTES = 9_500_000
"""Safety cap below Alibaba DashScope's 10,485,760-byte data-URI limit."""

OPENROUTER_JPEG_QUALITY = 90
"""JPEG quality used for OpenRouter uploads."""


def scale_dimensions_to_max_edge(
    width: int,
    height: int,
    max_edge: int,
) -> tuple[int, int]:
    """Return dimensions scaled down so the longest edge is at most ``max_edge``.

    Never upscales; preserves aspect ratio.

    Args:
        width: Original image width in pixels.
        height: Original image height in pixels.
        max_edge: Maximum allowed longest edge in pixels.

    Returns:
        Target ``(width, height)`` after scaling.
    """
    if max(width, height) <= max_edge:
        return (width, height)
    if width >= height:
        scaled_width = max_edge
        scaled_height = max(round(height * max_edge / width), 1)
    else:
        scaled_height = max_edge
        scaled_width = max(round(width * max_edge / height), 1)
    return (scaled_width, scaled_height)


def resize_image_to_max_edge(image: Image.Image, max_edge: int) -> Image.Image:
    """Resize ``image`` so its longest edge is at most ``max_edge``.

    Args:
        image: Input image in RGB mode.
        max_edge: Maximum allowed longest edge in pixels.

    Returns:
        The original image when already within bounds, otherwise a resized copy.
    """
    target_size = scale_dimensions_to_max_edge(*image.size, max_edge)
    if target_size == image.size:
        return image
    return image.resize(target_size, Image.LANCZOS)


def jpeg_data_url(image: Image.Image, quality: int = OPENROUTER_JPEG_QUALITY) -> str:
    """Encode ``image`` as a JPEG data URI."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True)
    base64_data = base64.standard_b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{base64_data}"


def jpeg_data_url_under_max_base64_bytes(
    image: Image.Image,
    max_base64_bytes: int,
    quality: int = OPENROUTER_JPEG_QUALITY,
) -> tuple[str, tuple[int, int]]:
    """Encode ``image`` as JPEG, downscaling until the data URI fits the cap.

    Args:
        image: Input image in RGB mode.
        max_base64_bytes: Maximum allowed base64 payload size in bytes.
        quality: JPEG quality passed to Pillow.

    Returns:
        A tuple of the data URI and the ``(width, height)`` actually encoded.
    """
    working = image
    while True:
        data_url = jpeg_data_url(working, quality=quality)
        if len(data_url.encode("utf-8")) <= max_base64_bytes:
            return data_url, working.size
        if working.size[0] <= 1 and working.size[1] <= 1:
            return data_url, working.size
        max_edge = max(max(working.size) * 9 // 10, 1)
        working = resize_image_to_max_edge(working, max_edge)
