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

import json
import os
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from vlm_exam.tasks.detection import DetectionSample

GROUP_SIZE = 5
"""Images per cross-image group: one prompt image plus four targets."""


@dataclass(frozen=True)
class ImageGroup:
    """Five images that share a visual concept.

    Attributes:
        group_id: Stable zero-padded identifier.
        images: Image basenames, sorted.
        classes_by_image: Ground-truth classes present in each image.
        shared_classes: Classes present in every image of the group.
        origin: How the group was formed: ``"component"`` (images linked
            by shared class names), ``"split"`` (a larger class-sharing
            component split by class-set similarity), or ``"merged"``
            (small leftover components merged by class-name and file-name
            similarity; no class is shared).
    """

    group_id: str
    images: tuple[str, ...]
    classes_by_image: dict[str, tuple[str, ...]]
    shared_classes: tuple[str, ...]
    origin: str


def present_classes(sample: DetectionSample) -> tuple[str, ...]:
    """Class names that have at least one ground-truth box in a sample.

    Args:
        sample: Detection sample.

    Returns:
        Sorted class names.
    """
    class_ids = sample.ground_truth.class_id
    if class_ids is None or len(class_ids) == 0:
        return ()
    return tuple(sorted(sample.classes[int(index)] for index in np.unique(class_ids)))


def _class_components(classes_by_image: dict[str, tuple[str, ...]]) -> list[list[str]]:
    parent = {name: name for name in classes_by_image}

    def find(name: str) -> str:
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    by_class: dict[str, list[str]] = defaultdict(list)
    for name in sorted(classes_by_image):
        for class_name in classes_by_image[name]:
            by_class[class_name].append(name)
    for members in by_class.values():
        root = find(members[0])
        for member in members[1:]:
            parent[find(member)] = root
    components: dict[str, list[str]] = defaultdict(list)
    for name in sorted(classes_by_image):
        components[find(name)].append(name)
    return sorted(components.values(), key=lambda names: names[0])


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


def _split_component(
    names: list[str],
    classes_by_image: dict[str, tuple[str, ...]],
) -> list[list[str]]:
    """Split a class-sharing component into equal chunks by class-set
    similarity, falling back to sorted-name chunks when every image has
    the same class set."""
    chunk_count = len(names) // GROUP_SIZE
    class_sets = {name: frozenset(classes_by_image[name]) for name in names}
    distinct = {class_sets[name] for name in names}
    if len(distinct) == 1:
        ordered = sorted(names)
        return [
            ordered[index : index + GROUP_SIZE]
            for index in range(0, len(ordered), GROUP_SIZE)
        ]

    seeds = [sorted(names)[0]]
    while len(seeds) < chunk_count:
        farthest = max(
            (name for name in sorted(names) if name not in seeds),
            key=lambda name: min(
                1.0 - _jaccard(class_sets[name], class_sets[seed]) for seed in seeds
            ),
        )
        seeds.append(farthest)

    chunks: list[list[str]] = [[seed] for seed in seeds]
    remaining = [name for name in sorted(names) if name not in seeds]

    def confidence(name: str) -> tuple[float, str]:
        similarities = sorted(
            (_jaccard(class_sets[name], class_sets[seed]) for seed in seeds),
            reverse=True,
        )
        margin = similarities[0] - (similarities[1] if len(similarities) > 1 else 0.0)
        return (-margin, name)

    for name in sorted(remaining, key=confidence):
        candidates = [
            index for index, chunk in enumerate(chunks) if len(chunk) < GROUP_SIZE
        ]
        best = max(
            candidates,
            key=lambda index: (
                _jaccard(class_sets[name], class_sets[seeds[index]]),
                -index,
            ),
        )
        chunks[best].append(name)
    return [sorted(chunk) for chunk in chunks]


def _tokens(class_name: str) -> frozenset[str]:
    return frozenset(class_name.lower().split())


def _leftover_similarity(
    left: list[str],
    right: list[str],
    classes_by_image: dict[str, tuple[str, ...]],
) -> tuple[float, int]:
    token_similarity = max(
        _jaccard(_tokens(left_class), _tokens(right_class))
        for left_name in left
        for right_name in right
        for left_class in classes_by_image[left_name]
        for right_class in classes_by_image[right_name]
    )
    prefix_length = max(
        len(os.path.commonprefix([left_name, right_name]))
        for left_name in left
        for right_name in right
    )
    return (token_similarity, prefix_length)


def _merge_leftovers(
    components: list[list[str]],
    classes_by_image: dict[str, tuple[str, ...]],
) -> list[list[str]]:
    clusters = [sorted(component) for component in components]
    while True:
        best_pair: tuple[int, int] | None = None
        best_score: tuple[float, int] | None = None
        for left_index in range(len(clusters)):
            for right_index in range(left_index + 1, len(clusters)):
                if len(clusters[left_index]) + len(clusters[right_index]) > GROUP_SIZE:
                    continue
                score = _leftover_similarity(
                    clusters[left_index], clusters[right_index], classes_by_image
                )
                if best_score is None or score > best_score:
                    best_score = score
                    best_pair = (left_index, right_index)
        if best_pair is None:
            return clusters
        left_index, right_index = best_pair
        merged = sorted(clusters[left_index] + clusters[right_index])
        clusters = [
            cluster for index, cluster in enumerate(clusters) if index not in best_pair
        ]
        clusters.append(merged)


def build_image_groups(sample_index: dict[str, DetectionSample]) -> list[ImageGroup]:
    """Partition the dataset into groups of :data:`GROUP_SIZE` images.

    Images are first linked by shared ground-truth class names. Components
    of exactly :data:`GROUP_SIZE` images become groups; larger components
    are split by class-set similarity; smaller ones are merged greedily by
    class-name token overlap, then file-name prefix, until every group has
    :data:`GROUP_SIZE` images.

    Args:
        sample_index: Mapping of image basename to detection sample.

    Returns:
        Groups ordered by their first image name.

    Raises:
        ValueError: If the images cannot be partitioned into full groups.
    """
    if len(sample_index) % GROUP_SIZE != 0:
        raise ValueError(
            f"{len(sample_index)} images cannot form groups of {GROUP_SIZE}."
        )
    classes_by_image = {
        name: present_classes(sample) for name, sample in sample_index.items()
    }
    members: list[tuple[list[str], str]] = []
    leftovers: list[list[str]] = []
    for component in _class_components(classes_by_image):
        if len(component) == GROUP_SIZE:
            members.append((component, "component"))
        elif len(component) > GROUP_SIZE and len(component) % GROUP_SIZE == 0:
            members.extend(
                (chunk, "split")
                for chunk in _split_component(component, classes_by_image)
            )
        else:
            leftovers.append(component)
    for cluster in _merge_leftovers(leftovers, classes_by_image):
        if len(cluster) != GROUP_SIZE:
            raise ValueError(
                f"Leftover cluster of {len(cluster)} images could not be "
                f"completed: {cluster}"
            )
        members.append((cluster, "merged"))

    members.sort(key=lambda item: item[0][0])
    groups: list[ImageGroup] = []
    for index, (names, origin) in enumerate(members):
        class_sets = [set(classes_by_image[name]) for name in names]
        shared = set.intersection(*class_sets) if class_sets else set()
        groups.append(
            ImageGroup(
                group_id=f"group_{index + 1:02d}",
                images=tuple(names),
                classes_by_image={name: classes_by_image[name] for name in names},
                shared_classes=tuple(sorted(shared)),
                origin=origin,
            )
        )
    return groups


def group_label(group: ImageGroup) -> str:
    """Short human-readable label: the class present in most images.

    Args:
        group: Image group.

    Returns:
        Class name, ties broken alphabetically.
    """
    counts: Counter[str] = Counter()
    for classes in group.classes_by_image.values():
        counts.update(classes)
    return sorted(counts, key=lambda name: (-counts[name], name))[0]


def write_groups(groups: list[ImageGroup], path: Path) -> None:
    """Serialize groups to JSON.

    Args:
        groups: Image groups.
        path: Destination file; parent directories are created.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = []
    for group in groups:
        entry = asdict(group)
        entry["label"] = group_label(group)
        payload.append(entry)
    with open(path, "w") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")


def load_groups(path: Path) -> list[ImageGroup]:
    """Load groups written by :func:`write_groups`.

    Args:
        path: JSON file.

    Returns:
        Image groups in file order.
    """
    with open(path) as file:
        payload = json.load(file)
    return [
        ImageGroup(
            group_id=entry["group_id"],
            images=tuple(entry["images"]),
            classes_by_image={
                name: tuple(classes)
                for name, classes in entry["classes_by_image"].items()
            },
            shared_classes=tuple(entry["shared_classes"]),
            origin=entry["origin"],
        )
        for entry in payload
    ]
