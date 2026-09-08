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

from __future__ import annotations

import base64
import io
import json
import os
import re
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import openai
import supervision as sv
from PIL import Image

from vlm_exam.providers.base import EMPTY_RESPONSE_TEXT, call_with_retries
from vlm_exam.providers.image_upload import (
    OPENAI_MAX_EDGE_PIXELS,
    OPENROUTER_JPEG_QUALITY,
    OPENROUTER_MAX_BASE64_BYTES,
    jpeg_data_url_under_max_base64_bytes,
    resize_image_to_max_edge,
    scale_dimensions_to_max_edge,
)
from vlm_exam.tasks.detection import (
    DetectionCoordinateFormat,
    DetectionSample,
    parse_prediction,
)

Box = tuple[float, float, float, float]
"""Absolute ``(x_min, y_min, x_max, y_max)`` box in original image pixels."""

PREDICTION_LABEL = "object"
"""Class-agnostic label requested from and assigned to every prediction."""

REQUEST_TIMEOUT_SECONDS = 300.0
"""Per-request timeout; box prompting answers are long and slow."""

SUPPORTED_MODELS = ("gpt-6-astra", "qwen-3.8-max")
"""Model keys with a box-prompting backend."""

_DASHSCOPE_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
_DASHSCOPE_MAX_OUTPUT_TOKENS = 16384
_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def select_matrix_images(
    sample_index: dict[str, DetectionSample],
    *,
    count: int,
    seed: int = 42,
) -> list[str]:
    """Fixed random image subset; smaller counts are prefixes of larger ones.

    Args:
        sample_index: Mapping of image basename to detection sample.
        count: Number of image names to return.
        seed: Shuffle seed.

    Returns:
        Image basenames in shuffled order.
    """
    names = sorted(sample_index)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(names))
    return [names[index] for index in order[:count]]


def completed_images(jsonl_path: Path) -> set[str]:
    """Images already collected successfully in an existing raw JSONL file.

    Args:
        jsonl_path: Raw collection file.

    Returns:
        Image basenames whose record has no error.
    """
    if not jsonl_path.exists():
        return set()
    done: set[str] = set()
    with open(jsonl_path) as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("error") is None:
                done.add(str(record.get("image", "")))
    return done


def _try_json(text: str) -> Any | None:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


def extract_json_payload(raw_output: str) -> tuple[Any | None, str]:
    """Pull a JSON payload out of a raw response.

    Args:
        raw_output: Raw model text.

    Returns:
        The parsed payload and a container label describing where the JSON
        lived: bare, fenced, embedded in prose, Python literal, or absent.
    """
    text = raw_output.strip()
    if not text:
        return None, "empty"
    for match in _FENCE_PATTERN.finditer(text):
        parsed = _try_json(match.group(1).strip())
        if parsed is not None:
            return parsed, "fenced_json"
    parsed = _try_json(text)
    if parsed is not None:
        return parsed, "bare_json"
    for opener, closer in (("[", "]"), ("{", "}")):
        start = text.find(opener)
        while start != -1:
            depth = 0
            for index in range(start, len(text)):
                character = text[index]
                if character == opener:
                    depth += 1
                elif character == closer:
                    depth -= 1
                    if depth == 0:
                        parsed = _try_json(text[start : index + 1])
                        if parsed is not None:
                            return parsed, "embedded_json"
                        break
            start = text.find(opener, start + 1)
    python_like = _try_json(text.replace("(", "[").replace(")", "]"))
    if python_like is not None:
        return python_like, "python_literal"
    return None, "no_json"


class BoxPromptingBackend(Protocol):
    """Model access used by the box-prompting experiments."""

    model_key: str
    provider_model_id: str
    coordinate_format: DetectionCoordinateFormat

    def uploaded_size(self, image: Image.Image) -> tuple[int, int]:
        """Dimensions the backend will upload for ``image``."""

    def call(
        self,
        *,
        images: list[Image.Image],
        prompt: str,
        effort: str,
    ) -> dict[str, Any]:
        """Send one multi-image request and return output plus telemetry."""


def _png_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    base64_data = base64.standard_b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{base64_data}"


class OpenAIResponsesBackend:
    """OpenAI Responses API backend; mirrors ``OpenAIProvider`` uploads.

    Images are resized to a 2048 pixel longest edge and sent as PNG, so
    the model answers in pixel coordinates of the uploaded image.
    """

    coordinate_format = DetectionCoordinateFormat.XYXY_ABSOLUTE_RESIZED_IMAGE

    def __init__(
        self,
        model_key: str,
        provider_model_id: str,
        api_key: str | None = None,
        timeout_seconds: float = REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        self.model_key = model_key
        self.provider_model_id = provider_model_id
        self._client = openai.OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            timeout=timeout_seconds,
            max_retries=0,
        )

    def uploaded_size(self, image: Image.Image) -> tuple[int, int]:
        return scale_dimensions_to_max_edge(*image.size, OPENAI_MAX_EDGE_PIXELS)

    def call(
        self,
        *,
        images: list[Image.Image],
        prompt: str,
        effort: str,
    ) -> dict[str, Any]:
        content: list[dict[str, Any]] = []
        uploaded_sizes: list[list[int]] = []
        for image in images:
            upload = resize_image_to_max_edge(image, OPENAI_MAX_EDGE_PIXELS)
            content.append({"type": "input_image", "image_url": _png_data_url(upload)})
            uploaded_sizes.append([upload.size[0], upload.size[1]])
        content.append({"type": "input_text", "text": prompt})

        response, retry_stats = call_with_retries(
            lambda: self._client.responses.create(
                model=self.provider_model_id,
                reasoning={"effort": effort},
                input=[{"role": "user", "content": content}],
            )
        )
        answer = (response.output_text or EMPTY_RESPONSE_TEXT).strip()
        usage = response.usage
        return {
            "raw_output": answer,
            "uploaded_sizes": uploaded_sizes,
            "input_tokens": usage.input_tokens if usage else None,
            "output_tokens": usage.output_tokens if usage else None,
            "inference_seconds": retry_stats.inference_seconds,
            "attempts": retry_stats.attempts,
        }


class DashScopeBackend:
    """DashScope OpenAI-compatible backend used by the original Qwen runs.

    Thinking is always enabled to match the committed Qwen3.8-Max
    benchmark behavior; ``effort`` is recorded but not forwarded.
    """

    coordinate_format = DetectionCoordinateFormat.XYXY_NORMALIZED_0_TO_1000

    def __init__(
        self,
        model_key: str,
        provider_model_id: str,
        api_key: str | None = None,
        timeout_seconds: float = REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        self.model_key = model_key
        self.provider_model_id = provider_model_id
        self._client = openai.OpenAI(
            base_url=_DASHSCOPE_BASE_URL,
            api_key=api_key or os.environ.get("DASHSCOPE_API_KEY"),
            timeout=timeout_seconds,
            max_retries=0,
        )

    def uploaded_size(self, image: Image.Image) -> tuple[int, int]:
        return image.size

    def call(
        self,
        *,
        images: list[Image.Image],
        prompt: str,
        effort: str,
    ) -> dict[str, Any]:
        content: list[dict[str, Any]] = []
        uploaded_sizes: list[list[int]] = []
        for image in images:
            data_url, uploaded_size = jpeg_data_url_under_max_base64_bytes(
                image,
                OPENROUTER_MAX_BASE64_BYTES,
                quality=OPENROUTER_JPEG_QUALITY,
            )
            content.append({"type": "image_url", "image_url": {"url": data_url}})
            uploaded_sizes.append([uploaded_size[0], uploaded_size[1]])
        content.append({"type": "text", "text": prompt})

        response, retry_stats = call_with_retries(
            lambda: self._client.chat.completions.create(
                model=self.provider_model_id,
                max_tokens=_DASHSCOPE_MAX_OUTPUT_TOKENS,
                messages=[{"role": "user", "content": content}],
                extra_body={"enable_thinking": True},
            )
        )
        if not response.choices:
            answer = EMPTY_RESPONSE_TEXT
        else:
            message = response.choices[0].message
            answer = (message.content or EMPTY_RESPONSE_TEXT).strip()
        usage = response.usage
        return {
            "raw_output": answer,
            "uploaded_sizes": uploaded_sizes,
            "input_tokens": usage.prompt_tokens if usage else None,
            "output_tokens": usage.completion_tokens if usage else None,
            "inference_seconds": retry_stats.inference_seconds,
            "attempts": retry_stats.attempts,
        }


def build_backend(
    model_key: str,
    api_key: str | None = None,
    timeout_seconds: float = REQUEST_TIMEOUT_SECONDS,
) -> BoxPromptingBackend:
    """Create the backend for a supported model key.

    Args:
        model_key: One of :data:`SUPPORTED_MODELS`.
        api_key: Optional provider API key overriding the environment.
        timeout_seconds: Per-request timeout.

    Returns:
        Configured backend.
    """
    if model_key == "gpt-6-astra":
        return OpenAIResponsesBackend(
            model_key, "gpt-6-astra", api_key, timeout_seconds
        )
    if model_key == "qwen-3.8-max":
        return DashScopeBackend(model_key, "qwen3.8-max", api_key, timeout_seconds)
    raise ValueError(f"Unsupported model key: {model_key!r}")


def required_api_key_name(model_key: str) -> str:
    """Environment variable holding the API key for a model key.

    Args:
        model_key: One of :data:`SUPPORTED_MODELS`.

    Returns:
        Environment variable name.
    """
    if model_key == "gpt-6-astra":
        return "OPENAI_API_KEY"
    if model_key == "qwen-3.8-max":
        return "DASHSCOPE_API_KEY"
    raise ValueError(f"Unsupported model key: {model_key!r}")


def prompt_box(
    box: Box,
    original_wh: tuple[int, int],
    coordinate_format: DetectionCoordinateFormat,
    uploaded_wh: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Express an absolute box in the coordinate space the model is prompted in.

    Args:
        box: Box in original image pixels.
        original_wh: Original image ``(width, height)``.
        coordinate_format: Coordinate convention used in the prompt.
        uploaded_wh: Uploaded image ``(width, height)``.

    Returns:
        Integer ``(x_min, y_min, x_max, y_max)`` in prompt coordinates.
    """
    width, height = original_wh
    if coordinate_format == DetectionCoordinateFormat.XYXY_ABSOLUTE_RESIZED_IMAGE:
        scale_x = uploaded_wh[0] / width
        scale_y = uploaded_wh[1] / height
        limits = (uploaded_wh[0], uploaded_wh[1])
    elif coordinate_format == DetectionCoordinateFormat.XYXY_NORMALIZED_0_TO_1000:
        scale_x = 1000 / width
        scale_y = 1000 / height
        limits = (1000, 1000)
    else:
        raise ValueError(f"Unsupported coordinate format: {coordinate_format!r}")
    x_min, y_min, x_max, y_max = box
    return (
        max(0, min(limits[0], round(x_min * scale_x))),
        max(0, min(limits[1], round(y_min * scale_y))),
        max(0, min(limits[0], round(x_max * scale_x))),
        max(0, min(limits[1], round(y_max * scale_y))),
    )


def format_prompt_boxes(
    boxes: tuple[Box, ...],
    original_wh: tuple[int, int],
    coordinate_format: DetectionCoordinateFormat,
    uploaded_wh: tuple[int, int],
) -> str:
    """Render boxes as a comma-separated list of ``[x, y, x, y]`` literals.

    Args:
        boxes: Boxes in original image pixels.
        original_wh: Original image ``(width, height)``.
        coordinate_format: Coordinate convention used in the prompt.
        uploaded_wh: Uploaded image ``(width, height)``.

    Returns:
        Prompt-ready text.
    """
    rendered = [
        prompt_box(box, original_wh, coordinate_format, uploaded_wh) for box in boxes
    ]
    return ", ".join(f"[{box[0]}, {box[1]}, {box[2]}, {box[3]}]" for box in rendered)


def box_convention_clause(
    coordinate_format: DetectionCoordinateFormat,
    uploaded_wh: tuple[int, int],
) -> str:
    """Describe the coordinate convention of ``[x_min, y_min, x_max, y_max]``.

    Args:
        coordinate_format: Coordinate convention used in the prompt.
        uploaded_wh: Uploaded image ``(width, height)``.

    Returns:
        Clause completing "given as ...".
    """
    if coordinate_format == DetectionCoordinateFormat.XYXY_ABSOLUTE_RESIZED_IMAGE:
        return (
            "absolute pixel coordinates of the "
            f"{uploaded_wh[0]}x{uploaded_wh[1]} pixel image"
        )
    if coordinate_format == DetectionCoordinateFormat.XYXY_NORMALIZED_0_TO_1000:
        return (
            "integers between 0 and 1000, normalized to the image width (x) "
            "and height (y)"
        )
    raise ValueError(f"Unsupported coordinate format: {coordinate_format!r}")


def output_clause(
    coordinate_format: DetectionCoordinateFormat,
    uploaded_wh: tuple[int, int],
) -> str:
    """Build the output-format instruction shared by every arm.

    Args:
        coordinate_format: Coordinate convention the model should answer in.
        uploaded_wh: Uploaded image ``(width, height)`` of the image the
            answer refers to.

    Returns:
        Prompt clause requesting the benchmark output format.
    """
    if coordinate_format == DetectionCoordinateFormat.XYXY_ABSOLUTE_RESIZED_IMAGE:
        convention = (
            "the top-left and bottom-right corners in absolute pixel "
            f"coordinates of the {uploaded_wh[0]}x{uploaded_wh[1]} pixel image. "
        )
    elif coordinate_format == DetectionCoordinateFormat.XYXY_NORMALIZED_0_TO_1000:
        convention = (
            "the top-left and bottom-right corners as integers between 0 and "
            "1000, normalized to the image width (x) and height (y). "
        )
    else:
        raise ValueError(f"Unsupported coordinate format: {coordinate_format!r}")
    return (
        "Output a JSON list where each entry contains the 2D bounding box "
        'in the key "box_2d" and the text label in the key "label". '
        'The "box_2d" value must be [x_min, y_min, x_max, y_max]: '
        + convention
        + f'Use the label "{PREDICTION_LABEL}" for every entry. '
        "Return only the JSON list, with no extra text."
    )


def record_uploaded_wh(record: dict[str, Any]) -> tuple[int, int] | None:
    """Uploaded size of the first image in a raw record, when stored.

    Args:
        record: Raw collection record.

    Returns:
        ``(width, height)`` or ``None``.
    """
    sizes = record.get("uploaded_sizes")
    if not sizes:
        return None
    first = sizes[0]
    return (int(first[0]), int(first[1]))


def parse_class_agnostic(
    raw_output: str,
    sample: DetectionSample,
    coordinate_format: DetectionCoordinateFormat,
    uploaded_wh: tuple[int, int] | None,
    key: str | None = None,
) -> tuple[sv.Detections, bool]:
    """Parse raw model text into class-agnostic detections.

    All labels are collapsed to :data:`PREDICTION_LABEL` before the
    benchmark parser runs.

    Args:
        raw_output: Raw model text.
        sample: Detection sample for coordinate scaling.
        coordinate_format: Coordinate convention of the answer.
        uploaded_wh: Uploaded image size for pixel formats.
        key: When set, the payload must be a JSON object and only the list
            under this key is parsed; otherwise a bare list or the first
            list value of an object is used.

    Returns:
        Parsed detections and a parse-failure flag.
    """
    payload, _ = extract_json_payload(raw_output)
    entries: list[Any] | None = None
    if key is not None:
        if isinstance(payload, dict) and isinstance(payload.get(key), list):
            entries = payload[key]
    elif isinstance(payload, list):
        entries = payload
    elif isinstance(payload, dict):
        for value in payload.values():
            if isinstance(value, list):
                entries = value
                break
    if entries is None:
        return sv.Detections.empty(), True
    if not entries:
        return sv.Detections.empty(), False
    collapsed = []
    for entry in entries:
        if isinstance(entry, dict):
            relabeled = dict(entry)
            relabeled["label"] = PREDICTION_LABEL
            collapsed.append(relabeled)
    detections = parse_prediction(
        json.dumps(collapsed),
        (sample.image_width, sample.image_height),
        [PREDICTION_LABEL],
        coordinate_format=coordinate_format,
        uploaded_wh=uploaded_wh,
    )
    return detections, len(detections) == 0
