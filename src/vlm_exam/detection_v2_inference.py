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
import json
import os
import re
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import anthropic
import httpx
import numpy as np
import openai
from google import genai
from google.genai import types
from PIL import Image, ImageOps
from supervision import Detections

from vlm_exam.detection_v2_upload import _prepare_xai_image
from vlm_exam.providers.anthropic import compute_resize_dimensions
from vlm_exam.providers.image_upload import (
    OPENAI_MAX_EDGE_PIXELS,
    OPENROUTER_JPEG_QUALITY,
    OPENROUTER_MAX_BASE64_BYTES,
    jpeg_data_url_under_max_base64_bytes,
    resize_image_to_max_edge,
)
from vlm_exam.providers.openai import _png_data_url
from vlm_exam.tasks.detection import DetectionSample, DetectionTask, parse_prediction

_LOCAL = threading.local()


class ImagePayloadError(ValueError):
    """An image cannot be encoded without violating the frozen upload policy."""


def _claude_image_payload(image: Image.Image) -> tuple[str, str, dict[str, Any]]:
    limit = 10 * 1024 * 1024
    pixels = hashlib.sha256(image.tobytes()).hexdigest()
    for fmt, media, options in [
        ("PNG", "image/png", {}),
        ("WEBP", "image/webp", {"lossless": True, "method": 6}),
    ]:
        buffer = io.BytesIO()
        image.save(buffer, format=fmt, **options)
        data = buffer.getvalue()
        payload = base64.standard_b64encode(data).decode("ascii")
        if len(payload) <= limit:
            decoded = Image.open(io.BytesIO(data)).convert("RGB")
            if (
                decoded.size != image.size
                or hashlib.sha256(decoded.tobytes()).hexdigest() != pixels
            ):
                raise ImagePayloadError(
                    "Invalid image: lossless payload pixel verification failed"
                )
            return (
                payload,
                media,
                {
                    "media_type": media,
                    "base64_bytes": len(payload),
                    "pixel_sha256": pixels,
                    "encoded_sha256": hashlib.sha256(data).hexdigest(),
                    "size": list(image.size),
                    "lossless": True,
                },
            )
    raise ImagePayloadError(
        "Image exceeds base64 limit after lossless encoding; "
        "no resizing or lossy fallback applied"
    )


def _client(provider: str) -> Any:
    if not hasattr(_LOCAL, "clients"):
        _LOCAL.clients = {}
    if provider not in _LOCAL.clients:
        if provider == "google":
            value = genai.Client(
                api_key=os.environ["GOOGLE_API_KEY"],
                http_options=types.HttpOptions(
                    timeout=600000, retry_options=types.HttpRetryOptions(attempts=1)
                ),
            )
        elif provider == "anthropic":
            value = anthropic.Anthropic(
                api_key=os.environ["ANTHROPIC_API_KEY"],
                timeout=httpx.Timeout(600, connect=30, pool=30),
                max_retries=0,
            )
        else:
            args = {
                "timeout": httpx.Timeout(600, connect=30, pool=30),
                "max_retries": 0,
            }
            if provider == "xai":
                args.update(
                    base_url="https://api.x.ai/v1", api_key=os.environ["XAI_API_KEY"]
                )
            elif provider == "openrouter":
                args.update(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=os.environ["OPENROUTER_API_KEY"],
                )
            else:
                args["api_key"] = os.environ["OPENAI_API_KEY"]
            value = openai.OpenAI(**args)
        _LOCAL.clients[provider] = value
    return _LOCAL.clients[provider]


def infer(
    case: dict[str, Any],
    model: str,
    arm: str,
    config: dict[str, Any],
    images: Path,
    heartbeat: Callable[[], None] = lambda: None,
) -> dict[str, Any]:
    """Generate one response with explicit native effort, format and token settings."""
    effort = config[arm]
    provider = config["provider"]
    started = time.time()
    image = ImageOps.exif_transpose(Image.open(images / case["image"])).convert("RGB")
    if provider == "xai":
        payload, size, image_payload = _prepare_xai_image(image)
    elif provider == "openai":
        image = resize_image_to_max_edge(image, OPENAI_MAX_EDGE_PIXELS)
        payload = _png_data_url(image)
        size = image.size
    elif provider == "anthropic":
        size = compute_resize_dimensions(*image.size)
        image = (
            image.resize(size, Image.Resampling.LANCZOS)
            if size != image.size
            else image
        )
        payload, media_type, image_payload = _claude_image_payload(image)
    elif provider == "openrouter":
        payload, size = jpeg_data_url_under_max_base64_bytes(
            image, OPENROUTER_MAX_BASE64_BYTES, quality=OPENROUTER_JPEG_QUALITY
        )
    else:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        payload = buffer.getvalue()
        size = image.size
    task = DetectionTask(prompt_classes="image", coordinate_format=config["format"])
    prompt = task.build_prompt(_sample(case), uploaded_size=size)
    api = _client(provider)
    raw = []
    text = ""
    reasoning_tokens = 0
    input_tokens = 0
    output_tokens = 0
    finish = None
    response_model = None
    if provider in ["openai", "xai"]:
        kwargs = {
            "model": config["wire"],
            "reasoning": {"effort": effort},
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_image", "image_url": payload},
                        {"type": "input_text", "text": prompt},
                    ],
                }
            ],
        }
        kwargs.update(store=False)
        if "max_output_tokens" in config:
            kwargs["max_output_tokens"] = config["max_output_tokens"]
        if provider == "xai":
            kwargs["input"][0]["content"][0]["detail"] = "high"
        if model == "gpt-6.1-terra":
            kwargs.update(store=False, service_tier="default")
        with api.responses.stream(**kwargs) as stream:
            for event in stream:
                heartbeat()
            response = stream.get_final_response()
        raw = response.model_dump(mode="json")
        text = response.output_text or ""
        finish = response.status
        response_model = response.model
        if response.usage:
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            reasoning_tokens = (
                getattr(response.usage.output_tokens_details, "reasoning_tokens", 0)
                or 0
            )
    elif provider == "anthropic":
        with api.messages.stream(
            model=config["wire"],
            max_tokens=config["max_output_tokens"],
            thinking={"type": "adaptive"},
            output_config={"effort": effort},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": payload,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        ) as stream:
            for event in stream:
                heartbeat()
            response = stream.get_final_message()
        raw = response.model_dump(mode="json")
        text = "".join(block.text for block in response.content if block.type == "text")
        finish = response.stop_reason
        response_model = response.model
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
    elif provider == "google":
        chunks = []
        for chunk in api.models.generate_content_stream(
            model=config["wire"],
            contents=[
                types.Part.from_bytes(data=payload, mime_type="image/png"),
                prompt,
            ],
            config=types.GenerateContentConfig(
                max_output_tokens=config.get("max_output_tokens"),
                thinking_config=types.ThinkingConfig(
                    **(
                        {"thinking_budget": effort}
                        if isinstance(effort, int)
                        else {"thinking_level": effort}
                    )
                ),
            ),
        ):
            heartbeat()
            raw.append(chunk.model_dump(mode="json", exclude_none=True))
            for candidate in chunk.candidates or []:
                if candidate.content:
                    for part in candidate.content.parts or []:
                        if part.text and not part.thought:
                            chunks.append(part.text)
                if candidate.finish_reason:
                    finish = str(candidate.finish_reason)
            if chunk.usage_metadata:
                usage = chunk.usage_metadata
                input_tokens = usage.prompt_token_count or 0
                reasoning_tokens = usage.thoughts_token_count or 0
                output_tokens = (usage.candidates_token_count or 0) + reasoning_tokens
            response_model = getattr(chunk, "model_version", None) or response_model
        text = "".join(chunks)
        if not response_model or not response_model.startswith(config["wire"]):
            raise RuntimeError("Model identity mismatch: " + str(response_model))
    else:
        chunks = []
        stream = api.chat.completions.create(
            model=config["wire"],
            **(
                {"max_tokens": config["max_output_tokens"]}
                if "max_output_tokens" in config
                else {}
            ),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": payload}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            extra_body={
                "reasoning": (
                    {"enabled": True}
                    if effort == "enabled"
                    else {"enabled": False}
                    if effort == "disabled"
                    else {"effort": effort}
                ),
                "provider": config.get("routing", {"require_parameters": True}),
            },
            stream=True,
            stream_options={"include_usage": True},
        )
        try:
            for chunk in stream:
                heartbeat()
                raw.append(chunk.model_dump(mode="json", exclude_none=True))
                response_model = chunk.model
                for choice in chunk.choices:
                    if choice.delta.content:
                        chunks.append(choice.delta.content)
                    if choice.finish_reason:
                        finish = choice.finish_reason
                if chunk.usage:
                    input_tokens = chunk.usage.prompt_tokens
                    output_tokens = chunk.usage.completion_tokens
                    reasoning_tokens = (
                        getattr(
                            getattr(chunk.usage, "completion_tokens_details", None),
                            "reasoning_tokens",
                            0,
                        )
                        or 0
                    )
        finally:
            stream.close()
        text = "".join(chunks)
    result = {
        "case": case["id"],
        "model": model,
        "wire_model": config["wire"],
        "response_model": response_model,
        "provider": provider,
        "arm": arm,
        "native_effort": effort,
        "repeat": 1,
        "predicted": text.strip(),
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(
            json.dumps(prompt, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "uploaded_size": list(size),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "finish_reason": finish,
        "elapsed_seconds": time.time() - started,
        "raw": raw,
        "timestamp": time.time(),
        "coordinate_format": config["format"],
        "max_output_tokens": config.get("max_output_tokens"),
        "truncated": finish
        in ["length", "incomplete", "FinishReason.MAX_TOKENS", "max_tokens"],
    }
    if provider in ["anthropic", "xai"]:
        result["image_payload"] = image_payload
    result["routing"] = config.get("routing")
    result["upstream_providers"] = (
        sorted(
            {
                chunk.get("provider")
                for chunk in raw
                if isinstance(chunk, dict) and chunk.get("provider")
            }
        )
        if isinstance(raw, list)
        else []
    )
    result["detections"] = parse_objects(case, result, config["format"])
    result.pop("raw", None)
    return result


def _sample(case: dict[str, Any]) -> DetectionSample:
    labels = sorted({obj["label"] for obj in case["objects"]})
    truth = Detections(
        xyxy=np.array([obj["box"] for obj in case["objects"]], dtype=float).reshape(
            -1, 4
        ),
        class_id=np.array(
            [labels.index(obj["label"]) for obj in case["objects"]], dtype=int
        ),
        confidence=np.ones(len(case["objects"])),
    )
    return DetectionSample(
        image_path=case["image"],
        ground_truth=truth,
        classes=tuple(labels),
        image_width=case["width"],
        image_height=case["height"],
    )


def parse_objects(
    case: dict[str, Any], result: dict[str, Any], coordinate_format: str
) -> list[dict[str, Any]]:
    """Decode boxes while retaining unknown labels as potential false positives."""
    labels = sorted({obj["label"] for obj in case["objects"]})
    for token in re.findall(r'"label"\s*:\s*("(?:[^"\\]|\\.)*")', result["predicted"]):
        try:
            label = json.loads(token, strict=False)
            if label not in labels:
                labels.append(label)
        except ValueError:
            pass
    parsed = parse_prediction(
        result["predicted"],
        (case["width"], case["height"]),
        labels,
        coordinate_format=coordinate_format,
        uploaded_wh=tuple(result["uploaded_size"]),
    )
    confidence = (
        parsed.confidence if parsed.confidence is not None else np.ones(len(parsed))
    )
    return [
        dict(box=box.tolist(), label=labels[int(index)], confidence=float(score))
        for box, index, score in zip(parsed.xyxy, parsed.class_id, confidence)
    ]
