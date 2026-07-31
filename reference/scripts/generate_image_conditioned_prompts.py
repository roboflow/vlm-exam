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

import hashlib
import io
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image, ImageDraw, ImageOps

from vlm_exam.providers.base import REQUEST_TIMEOUT_SECONDS, call_with_retries
from vlm_exam.reference.prompts import (
    MAX_PROMPT_WORDS,
    MIN_PROMPT_WORDS,
    file_sha256,
    prompt_classes_for_sample,
)
from vlm_exam.tasks.detection import DetectionSample, DetectionTask

GENERATION_PROMPT_VERSION = "image_conditioned_v2"
GENERATION_MODEL = "gemini-3.5-flash"
MAX_BOXES_PER_CLASS = 12
MAX_GENERATION_ATTEMPTS = 3
CONDITIONING_MODES = ("none", "overlay", "coords", "overlay_coords")
_CLASS_COLORS = (
    "#ff3b30",
    "#007aff",
    "#34c759",
    "#ff9500",
    "#af52de",
    "#00c7be",
    "#ff2d55",
    "#5856d6",
    "#a2845e",
    "#32ade6",
    "#ffcc00",
    "#64d2ff",
)
_SPATIAL_PATTERN = re.compile(
    r"\b(left|right|top|bottom|corner|near|next to|above|below|behind|front of)\b",
    re.IGNORECASE,
)
_ANNOTATION_PATTERN = re.compile(
    r"\b(bounding box|colored box|boxes|boxed|box outline|outline|outlined|"
    r"highlight|highlighted|"
    r"annotation|annotated|marker|marked|border|c\d+)\b",
    re.IGNORECASE,
)
_VIEWPOINT_PATTERN = re.compile(r"\bfrom\s+(above|below)\b", re.IGNORECASE)
_TOKEN_PATTERN = re.compile(r"\b[\w]+\b", re.UNICODE)
_NUMBER_WORDS = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
    "10": "ten",
    "20": "twenty",
    "50": "fifty",
}
_IDENTITY_WORDS = {
    *_NUMBER_WORDS.values(),
    "ace",
    "jack",
    "queen",
    "king",
    "cent",
    "cents",
    "euro",
    "euros",
}
_GENERATION_PROMPT = (
    "You are writing text queries for an open-vocabulary object detector. "
    "Describe how every listed benchmark class visually appears in this image. "
    "Consider all classes together so each query remains clearly distinguishable "
    "from the other classes.\n\n"
    "Requirements:\n"
    "- Return one entry for every class, using the exact class_name supplied.\n"
    "- Each entry has class_name and primary string fields.\n"
    "- Each phrase must be 2-6 words.\n"
    "- Focus on visual attributes: color, shape, material, parts, "
    "markings, relative size, and distinctive appearance.\n"
    "- Preserve identity-bearing details from the class name, including card "
    "ranks, coin denominations, numbers, and symbols.\n"
    "- For fine-grained classes, state visible differences such as size, color, "
    "and drawings or markings.\n"
    "- Do not use spatial references (left, top, near, etc.).\n"
    "- Do not mention counts or background details.\n"
    "- Annotation marks only identify targets. Never mention boxes, outlines, "
    "markers, annotation colors, class codes, or highlighting.\n"
    "- Stay on the requested class; do not rename it to a different category.\n"
    "- Primary phrases for different classes must not be identical.\n"
    "- The phrase must help a detector find instances of the class in this image."
)

_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "prompts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "class_name": {"type": "string"},
                    "primary": {"type": "string"},
                },
                "required": ["class_name", "primary"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["prompts"],
    "additionalProperties": False,
}


def _image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _validate_phrase(
    phrase: str,
    class_name: str,
) -> list[str]:
    issues: list[str] = []
    words = phrase.split()
    if len(words) < max(2, MIN_PROMPT_WORDS) or len(words) > MAX_PROMPT_WORDS:
        issues.append(f"word count out of range: {phrase!r}")
    spatial_text = _VIEWPOINT_PATTERN.sub("", phrase)
    if _SPATIAL_PATTERN.search(spatial_text):
        issues.append(f"spatial reference: {phrase!r}")
    if _ANNOTATION_PATTERN.search(phrase):
        issues.append(f"annotation reference: {phrase!r}")
    class_tokens = set(_TOKEN_PATTERN.findall(class_name.casefold()))
    phrase_tokens = set(_TOKEN_PATTERN.findall(phrase.casefold()))
    identity_requirements: list[set[str]] = []
    for token in class_tokens:
        if token in _NUMBER_WORDS:
            identity_requirements.append({token, _NUMBER_WORDS[token]})
        elif token in _IDENTITY_WORDS:
            identity_requirements.append({token})
    for requirement in identity_requirements:
        if phrase_tokens.isdisjoint(requirement):
            issues.append(f"missing identity token {sorted(requirement)!r}: {phrase!r}")
    return issues


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


def _validate_existing_records(
    records: dict[tuple[str, str], dict[str, object]],
    *,
    model: str,
    conditioning: str,
) -> None:
    entries_by_image: dict[
        str,
        dict[str, tuple[str, tuple[str, ...]]],
    ] = {}
    for key, record in records.items():
        image, class_name = key
        expected_fields = {
            "image": image,
            "class_name": class_name,
            "generation_model": model,
            "generation_prompt_version": GENERATION_PROMPT_VERSION,
            "conditioning": conditioning,
        }
        mismatches = [
            name for name, value in expected_fields.items() if record.get(name) != value
        ]
        if mismatches:
            raise ValueError(
                f"Existing prompt record {key!r} has incompatible fields: "
                f"{', '.join(mismatches)}"
            )
        primary = record.get("primary")
        variants = record.get("variants")
        if not isinstance(primary, str) or not isinstance(variants, list):
            raise ValueError(f"Existing prompt record {key!r} is malformed.")
        entries_by_image.setdefault(image, {})[class_name] = (
            primary,
            tuple(str(variant) for variant in variants),
        )

    for image, entries in entries_by_image.items():
        issues = _validate_entries(entries, tuple(entries))
        if issues:
            raise ValueError(
                f"Existing prompts for image {image!r} are invalid: {'; '.join(issues)}"
            )


def _validate_existing_manifest(
    path: Path,
    expected: dict[str, object],
) -> None:
    if not path.exists():
        raise ValueError(f"Existing prompts require a manifest: {path}")
    with open(path) as file:
        existing = json.load(file)
    fields = (
        "version",
        "generation_model",
        "generation_prompt_version",
        "dataset_annotations_sha256",
        "prompt_classes",
        "conditioning",
        "selected_images_sha256",
        "selected_image_contents_sha256",
        "generation_config_sha256",
    )
    mismatches = [
        field for field in fields if existing.get(field) != expected.get(field)
    ]
    if mismatches:
        raise ValueError(
            "Existing prompt manifest is incompatible with this run: "
            + ", ".join(mismatches)
        )


def _selected_image_contents_sha256(
    sample_entries: list[tuple[DetectionSample, tuple[str, ...]]],
) -> str:
    digest = hashlib.sha256()
    for sample, _ in sample_entries:
        image_path = Path(sample.image_path)
        digest.update(image_path.name.encode())
        digest.update(b"\0")
        digest.update(file_sha256(image_path).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _save_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            json.dump(payload, file, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _save_records(
    path: Path,
    records: dict[tuple[str, str], dict[str, object]],
    pair_order: dict[tuple[str, str], int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            ordered = sorted(records.items(), key=lambda item: pair_order[item[0]])
            for _, record in ordered:
                file.write(json.dumps(record) + "\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


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


def _selected_boxes(
    sample: DetectionSample,
    class_names: tuple[str, ...],
) -> dict[str, list[list[float]]]:
    selected: dict[str, list[list[float]]] = {}
    if sample.ground_truth.class_id is None:
        return {class_name: [] for class_name in class_names}
    for class_name in class_names:
        class_id = sample.classes.index(class_name)
        boxes = [
            [float(value) for value in box]
            for box, box_class_id in zip(
                sample.ground_truth.xyxy,
                sample.ground_truth.class_id,
                strict=True,
            )
            if int(box_class_id) == class_id
        ]
        boxes.sort(
            key=lambda box: (box[2] - box[0]) * (box[3] - box[1]),
            reverse=True,
        )
        selected[class_name] = boxes[:MAX_BOXES_PER_CLASS]
    return selected


def _draw_box_overlay(
    image: Image.Image,
    class_names: tuple[str, ...],
    boxes_by_class: dict[str, list[list[float]]],
) -> Image.Image:
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    width = max(2, min(image.size) // 250)
    for class_index, class_name in enumerate(class_names):
        color = _CLASS_COLORS[class_index % len(_CLASS_COLORS)]
        class_code = f"C{class_index + 1}"
        for box in boxes_by_class[class_name]:
            draw.rectangle(tuple(box), outline=color, width=width)
            x_min, y_min = int(box[0]), int(box[1])
            draw.text(
                (x_min + width, y_min + width),
                class_code,
                fill=color,
                stroke_width=max(1, width // 2),
                stroke_fill="black",
            )
    return overlay


def _normalized_box(
    box: list[float],
    image_width: int,
    image_height: int,
) -> list[int]:
    x_min, y_min, x_max, y_max = box
    return [
        round(max(0.0, min(1.0, y_min / image_height)) * 1000),
        round(max(0.0, min(1.0, x_min / image_width)) * 1000),
        round(max(0.0, min(1.0, y_max / image_height)) * 1000),
        round(max(0.0, min(1.0, x_max / image_width)) * 1000),
    ]


def _conditioning_input(
    image: Image.Image,
    sample: DetectionSample,
    class_names: tuple[str, ...],
    conditioning: str,
) -> tuple[Image.Image, str]:
    boxes_by_class = _selected_boxes(sample, class_names)
    use_overlay = conditioning in {"overlay", "overlay_coords"}
    use_coordinates = conditioning in {"coords", "overlay_coords"}
    conditioned_image = (
        _draw_box_overlay(image, class_names, boxes_by_class) if use_overlay else image
    )
    lines = ["Classes and annotation codes:" if use_overlay else "Classes:"]
    if use_coordinates:
        lines.append(
            "box_2d uses [y_min, x_min, y_max, x_max] normalized from 0 to 1000."
        )
    ground_truth_class_ids = (
        sample.ground_truth.class_id if sample.ground_truth.class_id is not None else ()
    )
    for class_index, class_name in enumerate(class_names):
        class_code = f"C{class_index + 1}"
        total = sum(
            1
            for class_id in ground_truth_class_ids
            if int(class_id) == sample.classes.index(class_name)
        )
        prefix = f"{class_code}: " if use_overlay else ""
        line = f'- {prefix}"{class_name}"'
        if use_coordinates:
            normalized = [
                _normalized_box(box, image.width, image.height)
                for box in boxes_by_class[class_name]
            ]
            line += f"; box_2d={normalized}"
        if (use_overlay or use_coordinates) and total > MAX_BOXES_PER_CLASS:
            line += f"; showing {MAX_BOXES_PER_CLASS} of {total} instances"
        lines.append(line)
    return conditioned_image, "\n".join(lines)


def _parse_prompt_entries(
    payload: dict[str, object],
) -> dict[str, tuple[str, tuple[str, ...]]]:
    raw_entries = payload.get("prompts")
    if not isinstance(raw_entries, list):
        raise ValueError("Gemini response has no prompts array.")
    entries: dict[str, tuple[str, tuple[str, ...]]] = {}
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, dict):
            raise ValueError("Gemini prompt entry is not an object.")
        class_name = str(raw_entry.get("class_name", "")).strip()
        if class_name in entries:
            raise ValueError(f"Gemini returned class {class_name!r} more than once.")
        primary = str(raw_entry.get("primary", "")).strip()
        raw_variants = raw_entry.get("variants", [])
        if not isinstance(raw_variants, list):
            raise ValueError(f"Variants for {class_name!r} are not an array.")
        variants = tuple(str(item).strip() for item in raw_variants)
        entries[class_name] = (primary, variants)
    return entries


def _validate_entries(
    entries: dict[str, tuple[str, tuple[str, ...]]],
    class_names: tuple[str, ...],
) -> list[str]:
    issues: list[str] = []
    expected = set(class_names)
    returned = set(entries)
    if returned != expected:
        issues.append(
            f"class mismatch: missing={sorted(expected - returned)}, "
            f"unexpected={sorted(returned - expected)}"
        )
    prompt_owners: dict[str, str] = {}
    for class_name in class_names:
        entry = entries.get(class_name)
        if entry is None:
            continue
        primary, variants = entry
        if not primary:
            issues.append(f"empty primary for {class_name!r}")
        else:
            issues.extend(
                f"{class_name!r}: {issue}"
                for issue in _validate_phrase(primary, class_name)
            )
        for variant in variants:
            issues.extend(
                f"{class_name!r}: {issue}"
                for issue in _validate_phrase(variant, class_name)
            )
        normalized = primary.casefold()
        previous_owner = prompt_owners.get(normalized)
        if previous_owner is not None and previous_owner != class_name:
            issues.append(
                f"duplicate primary {primary!r}: {previous_owner!r} and {class_name!r}"
            )
        prompt_owners[normalized] = class_name
    return issues


def _complete_image(
    client: genai.Client,
    model: str,
    image: Image.Image,
    sample: DetectionSample,
    class_names: tuple[str, ...],
    conditioning: str,
) -> tuple[dict[str, tuple[str, tuple[str, ...]]], int]:
    conditioned_image, context = _conditioning_input(
        image,
        sample,
        class_names,
        conditioning,
    )
    feedback = ""
    last_entries: dict[str, tuple[str, tuple[str, ...]]] = {}
    last_issues: list[str] = []
    for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
        prompt = f"{_GENERATION_PROMPT}\n\n{context}{feedback}"
        response, _ = call_with_retries(
            lambda: client.models.generate_content(
                model=model,
                contents=[
                    types.Part.from_bytes(
                        data=_image_to_png_bytes(conditioned_image),
                        mime_type="image/png",
                    ),
                    prompt,
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_json_schema=_RESPONSE_SCHEMA,
                    temperature=0.2,
                ),
            )
        )
        try:
            if not response.text:
                raise ValueError("Gemini returned no text.")
            payload = _parse_json_response(response.text)
            last_entries = _parse_prompt_entries(payload)
            last_issues = _validate_entries(last_entries, class_names)
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            last_entries = {}
            last_issues = [str(error)]
        if not last_issues:
            return last_entries, attempt
        feedback = (
            "\n\nYour previous response was invalid. Correct every issue:\n- "
            + "\n- ".join(last_issues)
        )

    raise ValueError(
        f"Gemini failed prompt validation after {MAX_GENERATION_ATTEMPTS} "
        f"attempts: {'; '.join(last_issues)}"
    )


@click.command()
@click.option(
    "--dataset-directory",
    default="data/detection/train",
    type=click.Path(exists=True),
    help="Detection dataset directory.",
)
@click.option(
    "--output-directory",
    default="reference/prompts/image_conditioned/v2",
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
    "--conditioning",
    default="none",
    type=click.Choice(CONDITIONING_MODES),
    help="Ground-truth localization signal supplied to Gemini.",
)
@click.option(
    "--image-list",
    default=None,
    type=click.Path(exists=True),
    help="Optional text file containing image basenames to generate.",
)
@click.option(
    "--max-images",
    default=None,
    type=click.IntRange(min=1),
    help="Optional cap on generated images after filtering.",
)
def main(
    dataset_directory: str,
    output_directory: str,
    model_name: str,
    prompt_classes: str,
    conditioning: str,
    image_list: str | None,
    max_images: int | None,
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
    if image_list is not None:
        requested_images = {
            line.strip()
            for line in Path(image_list).read_text().splitlines()
            if line.strip()
        }
        known_images = {Path(sample.image_path).name for sample in samples}
        unknown_images = requested_images - known_images
        if unknown_images:
            raise click.ClickException(
                "Image list contains unknown files: "
                + ", ".join(sorted(unknown_images)[:10])
            )
        samples = [
            sample
            for sample in samples
            if Path(sample.image_path).name in requested_images
        ]
    if max_images is not None:
        samples = samples[:max_images]

    sample_entries: list[tuple[DetectionSample, tuple[str, ...]]] = []
    pairs: list[tuple[str, str]] = []
    for sample in samples:
        assert isinstance(sample, DetectionSample)
        class_names = prompt_classes_for_sample(sample, prompt_classes)
        sample_entries.append((sample, class_names))
        image_name = Path(sample.image_path).name
        pairs.extend((image_name, class_name) for class_name in class_names)

    selected_images = [Path(sample.image_path).name for sample, _ in sample_entries]
    selected_images_sha256 = hashlib.sha256(
        "\n".join(selected_images).encode()
    ).hexdigest()
    manifest: dict[str, object] = {
        "version": "v2",
        "generation_model": model_name,
        "generation_prompt_version": GENERATION_PROMPT_VERSION,
        "dataset_directory": dataset_directory,
        "dataset_annotations_sha256": file_sha256(
            Path(dataset_directory) / "_annotations.coco.json"
        ),
        "prompt_classes": prompt_classes,
        "conditioning": conditioning,
        "image_list": image_list,
        "selected_images_sha256": selected_images_sha256,
        "selected_image_contents_sha256": _selected_image_contents_sha256(
            sample_entries
        ),
        "generation_config_sha256": file_sha256(Path(__file__)),
        "image_count": len(sample_entries),
        "pair_count": 0,
        "output_file": str(jsonl_path),
    }
    existing = _load_existing(jsonl_path)
    pair_order = {pair: index for index, pair in enumerate(pairs)}
    unexpected_pairs = set(existing) - set(pair_order)
    if unexpected_pairs:
        raise click.ClickException(
            "Existing prompts contain pairs outside the selected images: "
            + ", ".join(repr(pair) for pair in sorted(unexpected_pairs)[:10])
        )
    try:
        if manifest_path.exists() or existing:
            _validate_existing_manifest(manifest_path, manifest)
        _validate_existing_records(
            existing,
            model=model_name,
            conditioning=conditioning,
        )
    except ValueError as error:
        raise click.ClickException(str(error)) from error
    manifest["pair_count"] = len(existing)
    _save_json(manifest_path, manifest)

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=int(REQUEST_TIMEOUT_SECONDS * 1000),
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )

    generated = 0
    for index, (sample, class_names) in enumerate(sample_entries, start=1):
        image_name = Path(sample.image_path).name
        image_keys = [(image_name, class_name) for class_name in class_names]
        if all(key in existing for key in image_keys):
            continue
        image = ImageOps.exif_transpose(Image.open(sample.image_path)).convert("RGB")
        entries, attempts = _complete_image(
            client,
            model_name,
            image,
            sample,
            class_names,
            conditioning,
        )
        timestamp = datetime.now(timezone.utc).isoformat()
        for class_name in class_names:
            primary, variants = entries[class_name]
            record = {
                "image": image_name,
                "class_name": class_name,
                "primary": primary,
                "variants": list(variants),
                "generation_model": model_name,
                "generation_prompt_version": GENERATION_PROMPT_VERSION,
                "generated_at": timestamp,
                "manual_edit": None,
                "conditioning": conditioning,
                "generation_attempts": attempts,
            }
            existing[(image_name, class_name)] = record
            generated += 1
        _save_records(jsonl_path, existing, pair_order)
        manifest["pair_count"] = len(existing)
        _save_json(manifest_path, manifest)
        if index % 10 == 0 or index == len(sample_entries):
            click.echo(
                f"Progress {index}/{len(sample_entries)} images "
                f"(generated {generated} pairs)"
            )

    click.echo(f"Wrote {jsonl_path} ({len(existing)} pairs)")
    click.echo(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
