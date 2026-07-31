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

import io
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import click
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image, ImageOps

from vlm_exam.providers.base import REQUEST_TIMEOUT_SECONDS, call_with_retries
from vlm_exam.reference.prompts import MAX_PROMPT_WORDS, MIN_PROMPT_WORDS
from vlm_exam.tasks.detection import DetectionSample, DetectionTask

GENERATION_PROMPT_VERSION = "image_conditioned_v1"
GENERATION_MODEL = "gemini-3.5-flash"
_SPATIAL_PATTERN = re.compile(
    r"\b(left|right|top|bottom|corner|near|next to|above|below|behind|front of)\b",
    re.IGNORECASE,
)

_GENERATION_PROMPT = (
    "You are helping evaluate open-vocabulary object detection models. "
    "Given a benchmark class name and an image, write concise text prompts that "
    "describe how the target object class appears in this image.\n\n"
    "Requirements:\n"
    "- Return JSON with keys: primary (string), variants (array of 1-2 strings).\n"
    "- Each phrase must be 2-6 words.\n"
    "- Focus on visual attributes: color, shape, material, parts, "
    "distinctive appearance.\n"
    "- Do not use spatial references (left, top, near, etc.).\n"
    "- Do not mention counts or background details.\n"
    "- Stay on the requested class; do not rename it to a different category.\n"
    "- The phrase must help a detector find instances of the class in this image.\n\n"
    "Class name: {class_name}"
)


def _image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _validate_phrase(phrase: str, class_name: str) -> list[str]:
    issues: list[str] = []
    words = phrase.split()
    if len(words) < MIN_PROMPT_WORDS or len(words) > MAX_PROMPT_WORDS:
        issues.append(f"word count out of range: {phrase!r}")
    if _SPATIAL_PATTERN.search(phrase):
        issues.append(f"spatial reference: {phrase!r}")
    class_tokens = [
        token for token in class_name.lower().split() if not token.isdigit()
    ]
    if class_tokens and not any(token in phrase.lower() for token in class_tokens):
        issues.append(f"missing class token: {phrase!r}")
    return issues


def _fallback_description(class_name: str) -> tuple[str, tuple[str, ...]]:
    tokens = [token for token in class_name.split() if not token.isdigit()]
    if len(tokens) >= 2:
        primary = " ".join(tokens[-2:])
        return primary, (f"{tokens[-1]} object",)
    token = tokens[0] if tokens else class_name
    return token, (f"{token} object",)


def _prompt_classes_for_sample(
    sample: DetectionSample, prompt_classes: str
) -> tuple[str, ...]:
    if (
        prompt_classes == "image"
        and sample.ground_truth.class_id is not None
        and len(sample.ground_truth) > 0
    ):
        present_ids = set(sample.ground_truth.class_id)
        return tuple(sample.classes[class_id] for class_id in sorted(present_ids))
    return sample.classes


def _load_existing(path: Path) -> dict[tuple[str, str], dict[str, object]]:
    if not path.exists():
        return {}
    existing: dict[tuple[str, str], dict[str, object]] = {}
    with open(path) as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            key = (str(record["image"]), str(record["class_name"]))
            existing[key] = record
    return existing


def _parse_json_response(text: str) -> dict[str, object]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = (
            cleaned.removeprefix("```json")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
    decoder = json.JSONDecoder()
    payload, _index = decoder.raw_decode(cleaned)
    if not isinstance(payload, dict):
        raise ValueError("Gemini response was not a JSON object.")
    return payload


def _complete_pair(
    client: genai.Client,
    model: str,
    image: Image.Image,
    class_name: str,
) -> tuple[str, tuple[str, ...]]:
    prompt = _GENERATION_PROMPT.format(class_name=class_name)
    png_bytes = _image_to_png_bytes(image)
    response, _ = call_with_retries(
        lambda: client.models.generate_content(
            model=model,
            contents=[
                types.Part.from_bytes(data=png_bytes, mime_type="image/png"),
                prompt,
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            ),
        )
    )
    try:
        if not response.text:
            return _fallback_description(class_name)
        payload = _parse_json_response(response.text)
        primary = str(payload.get("primary", "")).strip()
        variants = tuple(str(item).strip() for item in payload.get("variants", []))
        candidates = [primary, *variants]
        valid = [
            phrase
            for phrase in candidates
            if phrase and not _validate_phrase(phrase, class_name)
        ]
        if not valid:
            return _fallback_description(class_name)
        primary_text = valid[0]
        extra = tuple(item for item in valid[1:3] if item != primary_text)
        return primary_text, extra
    except (ValueError, TypeError, json.JSONDecodeError):
        return _fallback_description(class_name)


@click.command()
@click.option(
    "--dataset-directory",
    default="data/detection/train",
    type=click.Path(exists=True),
    help="Detection dataset directory.",
)
@click.option(
    "--output-directory",
    default="reference/prompts/image_conditioned/v1",
    type=click.Path(),
    help="Directory for image-conditioned prompt JSONL output.",
)
@click.option(
    "--model",
    "model_name",
    default=GENERATION_MODEL,
    help="Gemini model id.",
)
@click.option(
    "--prompt-classes",
    default="image",
    type=click.Choice(["image", "all"]),
    help="Generate prompts for per-image GT classes or all classes.",
)
@click.option(
    "--max-pairs",
    default=None,
    type=int,
    help="Optional cap on generated class-image pairs.",
)
def main(
    dataset_directory: str,
    output_directory: str,
    model_name: str,
    prompt_classes: str,
    max_pairs: int | None,
) -> None:
    """Generate image-conditioned class descriptions with Gemini."""
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise click.ClickException(
            "GOOGLE_API_KEY or GEMINI_API_KEY must be set in .env"
        )

    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "prompts.jsonl"
    manifest_path = output_dir / "manifest.json"

    task = DetectionTask(prompt_classes=prompt_classes)
    samples = task.load_samples(dataset_directory)
    pairs: list[tuple[str, str, str]] = []
    for sample in samples:
        assert isinstance(sample, DetectionSample)
        image_name = Path(sample.image_path).name
        for class_name in _prompt_classes_for_sample(sample, prompt_classes):
            pairs.append((image_name, class_name, sample.image_path))
    if max_pairs is not None:
        pairs = pairs[:max_pairs]

    existing = _load_existing(jsonl_path)
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=int(REQUEST_TIMEOUT_SECONDS * 1000),
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )

    generated = 0
    with open(jsonl_path, "a") as file:
        for index, (image_name, class_name, image_path) in enumerate(pairs, start=1):
            if (image_name, class_name) in existing:
                continue
            image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
            primary, variants = _complete_pair(client, model_name, image, class_name)
            timestamp = datetime.now(timezone.utc).isoformat()
            record = {
                "image": image_name,
                "class_name": class_name,
                "primary": primary,
                "variants": list(variants),
                "generation_model": model_name,
                "generation_prompt_version": GENERATION_PROMPT_VERSION,
                "generated_at": timestamp,
                "manual_edit": None,
            }
            file.write(json.dumps(record) + "\n")
            file.flush()
            existing[(image_name, class_name)] = record
            generated += 1
            if index % 25 == 0 or index == len(pairs):
                click.echo(f"Progress {index}/{len(pairs)} (generated {generated} new)")

    manifest = {
        "version": "v1",
        "generation_model": model_name,
        "generation_prompt_version": GENERATION_PROMPT_VERSION,
        "dataset_directory": dataset_directory,
        "prompt_classes": prompt_classes,
        "pair_count": len(existing),
        "output_file": str(jsonl_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    click.echo(f"Wrote {jsonl_path} ({len(existing)} pairs)")
    click.echo(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
