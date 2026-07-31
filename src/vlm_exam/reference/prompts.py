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
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

MAX_PROMPT_WORDS = 6
MIN_PROMPT_WORDS = 1


class PromptAssetType(str, Enum):
    """Kind of reference-model prompt asset."""

    NONE = "none"
    IMAGE_CONDITIONED = "image_conditioned"


@dataclass(frozen=True)
class ImageConditionedEntry:
    """One image-conditioned description for a class-image pair."""

    image: str
    class_name: str
    primary: str
    variants: tuple[str, ...]
    generation_model: str
    generation_prompt_version: str
    generated_at: str
    manual_edit: dict[str, Any] | None = None


@dataclass(frozen=True)
class ImageConditionedPromptSet:
    """Image-conditioned descriptions keyed by image and class."""

    version: str
    path: Path
    sha256: str
    entries: dict[tuple[str, str], ImageConditionedEntry]

    def resolve(self, image: str, class_name: str) -> str:
        """Return the primary description for one class-image pair."""
        entry = self.entries.get((image, class_name))
        if entry is None:
            return class_name
        return entry.primary


@dataclass(frozen=True)
class LoadedPromptSet:
    """Resolved prompt asset with metadata for manifests."""

    asset_type: PromptAssetType
    version: str
    path: Path
    sha256: str
    image_conditioned: ImageConditionedPromptSet | None = None


def file_sha256(path: Path) -> str:
    """Return the SHA256 digest of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_prompt_phrase(phrase: str) -> None:
    words = phrase.split()
    if len(words) < MIN_PROMPT_WORDS or len(words) > MAX_PROMPT_WORDS:
        raise ValueError(
            f"Prompt phrase must be {MIN_PROMPT_WORDS}-{MAX_PROMPT_WORDS} words: "
            f"{phrase!r}"
        )


def load_image_conditioned_prompt_set(path: Path) -> LoadedPromptSet:
    """Load an image-conditioned prompt JSONL set.

    Args:
        path: Path to a versioned JSONL file of class-image descriptions.

    Returns:
        Loaded prompt set with per-pair resolution helpers.
    """
    entries: dict[tuple[str, str], ImageConditionedEntry] = {}
    prompt_owners: dict[tuple[str, str], str] = {}
    version = path.parent.name if path.parent.name.startswith("v") else path.stem
    with open(path) as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            image = str(raw["image"])
            class_name = str(raw["class_name"])
            variants = tuple(str(item) for item in raw.get("variants", []))
            primary = str(raw.get("primary", variants[0] if variants else class_name))
            for phrase in (primary, *variants):
                _validate_prompt_phrase(phrase)
            prompt_key = (image, primary)
            previous_owner = prompt_owners.get(prompt_key)
            if previous_owner is not None and previous_owner != class_name:
                raise ValueError(
                    f"Duplicate prompt {primary!r} for image {image!r}: "
                    f"{previous_owner!r} and {class_name!r}."
                )
            prompt_owners[prompt_key] = class_name
            entries[(image, class_name)] = ImageConditionedEntry(
                image=image,
                class_name=class_name,
                primary=primary,
                variants=variants,
                generation_model=str(raw.get("generation_model", "")),
                generation_prompt_version=str(raw.get("generation_prompt_version", "")),
                generated_at=str(raw.get("generated_at", "")),
                manual_edit=raw.get("manual_edit"),
            )

    prompt_set = ImageConditionedPromptSet(
        version=version,
        path=path,
        sha256=file_sha256(path),
        entries=entries,
    )
    return LoadedPromptSet(
        asset_type=PromptAssetType.IMAGE_CONDITIONED,
        version=version,
        path=path,
        sha256=prompt_set.sha256,
        image_conditioned=prompt_set,
    )


def load_prompt_set(path: Path) -> LoadedPromptSet:
    """Load an image-conditioned prompt asset.

    Args:
        path: JSONL image-conditioned prompt file.

    Returns:
        Loaded prompt set.
    """
    if path.suffix != ".jsonl":
        raise ValueError(
            "Reference prompt assets must be image-conditioned JSONL files."
        )
    return load_image_conditioned_prompt_set(path)


def resolve_prompt_texts(
    prompt_set: LoadedPromptSet | None,
    *,
    image: str,
    canonical_classes: tuple[str, ...],
) -> tuple[tuple[str, ...], dict[str, str]]:
    """Map canonical class names to prompt texts for one image.

    Args:
        prompt_set: Optional loaded prompt asset.
        image: Image basename for image-conditioned lookup.
        canonical_classes: Canonical benchmark class names for this sample.

    Returns:
        Tuple of prompt texts in the same order as ``canonical_classes``, and a
        mapping from prompt text back to canonical class name for label remapping.
    """
    prompt_to_canonical: dict[str, str] = {}
    prompt_texts: list[str] = []
    for class_name in canonical_classes:
        if prompt_set is None:
            prompt_text = class_name
        else:
            assert prompt_set.image_conditioned is not None
            prompt_text = prompt_set.image_conditioned.resolve(image, class_name)
        prompt_texts.append(prompt_text)
        prompt_to_canonical[prompt_text] = class_name
    return tuple(prompt_texts), prompt_to_canonical


def validate_prompt_set_coverage(
    prompt_set: LoadedPromptSet,
    *,
    all_classes: tuple[str, ...],
    required_pairs: list[tuple[str, str]] | None = None,
) -> list[str]:
    """Return validation errors for missing prompt coverage.

    Args:
        prompt_set: Loaded prompt asset.
        all_classes: Full benchmark class list.
        required_pairs: Required image/class pairs for image-conditioned sets.

    Returns:
        Human-readable validation errors; empty when coverage is complete.
    """
    del all_classes
    assert prompt_set.image_conditioned is not None
    if required_pairs is None:
        return []
    missing_pairs = [
        pair
        for pair in required_pairs
        if pair not in prompt_set.image_conditioned.entries
    ]
    if missing_pairs:
        return [f"Missing image-conditioned prompts for {len(missing_pairs)} pairs."]
    return []
