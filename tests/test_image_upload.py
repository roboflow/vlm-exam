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

from PIL import Image

from vlm_exam.providers.image_upload import (
    OPENAI_MAX_EDGE_PIXELS,
    OPENROUTER_MAX_BASE64_BYTES,
    jpeg_data_url_under_max_base64_bytes,
    resize_image_to_max_edge,
    scale_dimensions_to_max_edge,
)
from vlm_exam.providers.openai import OpenAIProvider


def test_scale_dimensions_never_upscales() -> None:
    assert scale_dimensions_to_max_edge(800, 600, OPENAI_MAX_EDGE_PIXELS) == (800, 600)


def test_scale_dimensions_preserves_aspect_ratio() -> None:
    width, height = scale_dimensions_to_max_edge(4000, 2000, OPENAI_MAX_EDGE_PIXELS)
    assert max(width, height) == OPENAI_MAX_EDGE_PIXELS
    assert width / height == 2.0


def test_resize_image_to_max_edge() -> None:
    image = Image.new("RGB", (4000, 2000), color=(255, 0, 0))
    resized = resize_image_to_max_edge(image, OPENAI_MAX_EDGE_PIXELS)
    assert resized.size == (OPENAI_MAX_EDGE_PIXELS, 1024)


def test_openai_uploaded_image_size_matches_resize() -> None:
    provider = OpenAIProvider(model="gpt-5.5", api_key="test")
    image = Image.new("RGB", (3000, 1500))
    assert provider.uploaded_image_size(image) == (2048, 1024)


def test_jpeg_data_url_under_max_base64_bytes_fits_cap() -> None:
    image = Image.new("RGB", (4512, 3008), color=(128, 64, 32))
    data_url, uploaded_size = jpeg_data_url_under_max_base64_bytes(
        image,
        OPENROUTER_MAX_BASE64_BYTES,
    )
    assert len(data_url.encode("utf-8")) <= OPENROUTER_MAX_BASE64_BYTES
    assert max(uploaded_size) <= max(image.size)
