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
import hashlib
import io
from typing import Any

from PIL import Image

_XAI_BASE64_LIMIT = 26_214_400
_SAFETY_MARGIN = 65_536


def _prepare_xai_image(
    image: Image.Image, limit: int = _XAI_BASE64_LIMIT, margin: int = _SAFETY_MARGIN
) -> tuple[str, tuple[int, int], dict[str, Any]]:
    original = image.convert("RGB")
    budget = limit - margin
    if budget < 1024:
        raise ValueError("Image payload budget is too small")

    def encode(im: Image.Image, fmt: str, **options: Any) -> bytes:
        buf = io.BytesIO()
        im.save(buf, format=fmt, **options)
        return buf.getvalue()

    def fits(data: bytes, fmt: str) -> bool:
        return (
            4 * ((len(data) + 2) // 3) + len(f"data:image/{fmt.lower()};base64,")
            <= budget
        )

    def finish(
        data: bytes, fmt: str, size: tuple[int, int], quality: int | None = None
    ) -> tuple[str, tuple[int, int], dict[str, Any]]:
        uri = f"data:image/{fmt.lower()};base64," + base64.b64encode(data).decode(
            "ascii"
        )
        assert len(uri) <= budget
        return (
            uri,
            size,
            dict(
                original_size=list(original.size),
                uploaded_size=list(size),
                encoding=fmt,
                quality=quality,
                chroma_subsampling=0 if fmt == "JPEG" else None,
                lossless=fmt == "PNG",
                resized=size != original.size,
                data_url_bytes=len(uri),
                encoded_bytes=len(data),
                budget_bytes=budget,
                source_pixel_sha256=hashlib.sha256(original.tobytes()).hexdigest(),
                encoded_sha256=hashlib.sha256(data).hexdigest(),
                policy="xai_payload_v1",
            ),
        )

    for options in ({}, {"optimize": True, "compress_level": 9}):
        data = encode(original, "PNG", **options)
        if fits(data, "PNG"):
            return finish(data, "PNG", original.size)
    for quality in (100, 95):
        data = encode(original, "JPEG", quality=quality, subsampling=0, optimize=True)
        if fits(data, "JPEG"):
            return finish(data, "JPEG", original.size, quality)
    width, height = original.size

    def at(edge: int) -> tuple[bytes, tuple[int, int]]:
        scale = edge / max(width, height)
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        im = original.resize(size, Image.Resampling.LANCZOS)
        return encode(im, "JPEG", quality=95, subsampling=0, optimize=True), size

    low, high = 1, max(width, height) - 1
    best = None
    while low <= high:
        edge = (low + high) // 2
        data, size = at(edge)
        if fits(data, "JPEG"):
            best = (data, size)
            low = edge + 1
        else:
            high = edge - 1
    if best is None:
        raise ValueError("Image exceeds payload budget even at minimum size")
    return finish(best[0], "JPEG", best[1], 95)
