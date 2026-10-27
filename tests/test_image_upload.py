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

from types import SimpleNamespace

import pytest
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


@pytest.mark.parametrize(
    ("status", "reason", "finish"),
    [
        ("completed", None, "stop"),
        ("incomplete", "max_output_tokens", "length"),
        ("failed", None, "error"),
    ],
)
def test_direct_openai_limits_and_completion_telemetry(
    monkeypatch: pytest.MonkeyPatch, status: str, reason: str | None, finish: str
) -> None:
    provider = OpenAIProvider(
        "gpt-6.1-terra",
        api_key="test",
        options={"max_output_tokens": 65536, "store": False, "service_tier": "default"},
    )
    calls = []
    response = SimpleNamespace(
        output_text="answer",
        status=status,
        incomplete_details=SimpleNamespace(reason=reason),
        id="resp_test",
        model="gpt-6.1-terra-20261026",
        usage=SimpleNamespace(
            input_tokens=100,
            output_tokens=200,
            model_dump=lambda: {
                "input_tokens": 100,
                "output_tokens": 200,
                "output_tokens_details": {"reasoning_tokens": 150},
            },
        ),
    )
    monkeypatch.setattr(
        provider._client.responses,
        "create",
        lambda **kwargs: calls.append(kwargs) or response,
    )
    answer, usage, _ = provider.predict(Image.new("RGB", (3000, 1500)), "Read", "high")
    assert str(provider._client.base_url) == "https://api.openai.com/v1/"
    assert calls[0]["reasoning"] == {"effort": "high"}
    assert calls[0]["max_output_tokens"] == 65536
    assert calls[0]["store"] is False
    assert calls[0]["service_tier"] == "default"
    assert calls[0]["input"][0]["content"][0]["image_url"].startswith("data:image/png;")
    assert answer == "answer"
    assert usage.metadata["finish_reason"] == finish
    assert usage.metadata["requested_model"] == "gpt-6.1-terra"
    assert (
        usage.metadata["reported"]["output_tokens_details"]["reasoning_tokens"] == 150
    )


def test_jpeg_data_url_under_max_base64_bytes_fits_cap() -> None:
    image = Image.new("RGB", (4512, 3008), color=(128, 64, 32))
    data_url, uploaded_size = jpeg_data_url_under_max_base64_bytes(
        image,
        OPENROUTER_MAX_BASE64_BYTES,
    )
    assert len(data_url.encode("utf-8")) <= OPENROUTER_MAX_BASE64_BYTES
    assert max(uploaded_size) <= max(image.size)
