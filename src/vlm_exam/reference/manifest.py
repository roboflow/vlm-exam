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
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vlm_exam.reference.config import ReferenceModelConfig


@dataclass
class RunManifest:
    """Reproducibility metadata for a reference benchmark run."""

    run_type: str
    model: str
    model_name: str
    checkpoint: str
    adapter: str
    effort: str
    task: str
    timestamp: str
    dataset_directory: str
    dataset_annotations_sha256: str
    dataset_image_count: int
    prompt_classes: str
    classes_processed: str
    coordinate_format: str
    device: str
    precision: str
    inference: dict[str, Any]
    benchmark_commit: str | None
    python_version: str
    platform: str
    dependency_versions: dict[str, str]
    deviations: list[str] = field(default_factory=list)
    total_seconds: float | None = None
    failed_sample_count: int = 0
    completed_sample_count: int = 0
    benchmark_dirty: bool | None = None
    checkpoint_revision: str | None = None
    checkpoint_sha256: str | None = None
    prompt_asset_type: str = "none"
    prompt_set_path: str | None = None
    prompt_set_version: str | None = None
    prompt_set_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert the manifest to a JSON-serializable mapping."""
        return asdict(self)


def _git_commit() -> str | None:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return output.strip() or None


def _git_dirty() -> bool | None:
    try:
        output = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return bool(output.strip())


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _collect_dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {"python": platform.python_version()}
    for package in ("vlm_exam", "ultralytics", "torch", "numpy", "supervision"):
        try:
            module = __import__(package)
        except ImportError:
            continue
        version = getattr(module, "__version__", None)
        if version is not None:
            versions[package] = str(version)
    return versions


def build_run_manifest(
    *,
    model_config: ReferenceModelConfig,
    effort: str,
    task: str,
    timestamp: str,
    dataset_directory: str,
    prompt_classes: str,
    device: str,
    precision: str,
    deviations: list[str] | None = None,
) -> RunManifest:
    """Build an initial manifest before inference starts.

    Args:
        model_config: Reference model configuration.
        effort: Effort label for the run file.
        task: Benchmark task name.
        timestamp: Run timestamp string.
        dataset_directory: Dataset root path.
        prompt_classes: ``image`` or ``all`` class listing mode.
        device: Resolved device backend.
        precision: Numeric precision used for inference.
        deviations: Documented deviations from the standard procedure.

    Returns:
        Manifest populated with static reproducibility fields.
    """
    annotations_path = Path(dataset_directory) / "_annotations.coco.json"
    inference = {
        "conf": model_config.inference.conf,
        "iou": model_config.inference.iou,
        "imgsz": model_config.inference.imgsz,
        "max_det": model_config.inference.max_det,
        "agnostic_nms": model_config.inference.agnostic_nms,
    }
    return RunManifest(
        run_type="reference",
        model=model_config.key,
        model_name=model_config.name,
        checkpoint=model_config.checkpoint,
        adapter=model_config.adapter,
        effort=effort,
        task=task,
        timestamp=timestamp,
        dataset_directory=str(dataset_directory),
        dataset_annotations_sha256=_file_sha256(annotations_path),
        dataset_image_count=0,
        prompt_classes=prompt_classes,
        classes_processed="together",
        coordinate_format=model_config.coordinate_format.value,
        device=device,
        precision=precision,
        inference=inference,
        benchmark_commit=_git_commit(),
        python_version=sys.version,
        platform=platform.platform(),
        dependency_versions=_collect_dependency_versions(),
        deviations=list(deviations or []),
        benchmark_dirty=_git_dirty(),
        checkpoint_revision=model_config.checkpoint_revision,
        checkpoint_sha256=model_config.checkpoint_sha256,
    )


def save_manifest(manifest: RunManifest, path: Path) -> None:
    """Write a manifest JSON file.

    Args:
        manifest: Run manifest to persist.
        path: Output file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as file:
        json.dump(manifest.to_dict(), file, indent=2)
        file.write("\n")


def load_manifest(path: Path) -> RunManifest:
    """Load a manifest JSON file.

    Args:
        path: Path to a manifest written by :func:`save_manifest`.

    Returns:
        Parsed run manifest.
    """
    with open(path) as file:
        raw = json.load(file)
    known_fields = {item.name for item in fields(RunManifest)}
    return RunManifest(
        **{key: value for key, value in raw.items() if key in known_fields}
    )


def manifest_path_for_results(results_path: Path) -> Path:
    """Derive the manifest path from a reference results JSONL path."""
    return results_path.with_suffix(".manifest.json")


def resolve_dataset_directory_from_manifest(
    results_path: Path,
    manifest_path: Path | None = None,
) -> Path:
    """Resolve the dataset directory recorded in a reference run manifest.

    Manifests may store paths relative to the process cwd at run time rather
    than relative to the manifest file, so several candidates are tried.

    Args:
        results_path: Reference results JSONL path.
        manifest_path: Optional explicit manifest path.

    Returns:
        Existing dataset directory path.

    Raises:
        FileNotFoundError: When no manifest exists or no candidate path exists.
        ValueError: When the manifest omits ``dataset_directory``.
    """
    path = manifest_path or manifest_path_for_results(results_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing manifest: {path}")

    manifest = load_manifest(path)
    raw_directory = manifest.dataset_directory
    if not raw_directory:
        raise ValueError(f"Manifest {path} has no dataset_directory.")

    dataset_path = Path(raw_directory)
    candidates: list[Path] = [
        dataset_path,
        path.parent / dataset_path,
        Path.cwd() / dataset_path,
    ]

    repo_root = _find_repo_root(path)
    if repo_root is not None:
        parts = dataset_path.parts
        if "data" in parts:
            data_index = parts.index("data")
            candidates.append(repo_root / Path(*parts[data_index:]))

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved

    raise FileNotFoundError(
        f"Dataset directory not found for manifest {path}: tried "
        f"{', '.join(str(item) for item in seen)}"
    )


def _find_repo_root(start: Path) -> Path | None:
    for parent in (start.resolve(), *start.resolve().parents):
        if (parent / "pyproject.toml").exists() and (
            parent / "src" / "vlm_exam"
        ).exists():
            return parent
    return None


def new_run_timestamp() -> str:
    """Return a UTC timestamp string for a new run."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
