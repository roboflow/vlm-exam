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

import os
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from vlm_exam.reference.base import create_reference_adapter
from vlm_exam.reference.config import ReferenceModelConfig
from vlm_exam.reference.constants import REFERENCE_EFFORT
from vlm_exam.reference.manifest import (
    RunManifest,
    build_run_manifest,
    load_manifest,
    manifest_path_for_results,
    new_run_timestamp,
    save_manifest,
)
from vlm_exam.reference.prompts import LoadedPromptSet, resolve_prompt_texts
from vlm_exam.reference.serializer import serialize_reference_prediction
from vlm_exam.results import (
    RunResult,
    SampleResult,
    is_failed_sample,
    load_results,
    merge_resumed_runs,
    save_results,
)
from vlm_exam.tasks.detection import DetectionSample, DetectionTask


def resolve_device(
    requested: str,
    supported_devices: tuple[str, ...],
) -> str:
    """Resolve an automatic or explicit device choice.

    Args:
        requested: ``auto``, ``mps``, ``cpu``, or ``cuda``.
        supported_devices: Devices declared for the model.

    Returns:
        Resolved device string.

    Raises:
        ValueError: When the requested device is unsupported.
    """
    if requested == "auto":
        if "mps" in supported_devices:
            try:
                import torch

                if torch.backends.mps.is_available():
                    return "mps"
            except ImportError:
                pass
        if "cuda" in supported_devices:
            try:
                import torch

                if torch.cuda.is_available():
                    return "cuda"
            except ImportError:
                pass
        if "cpu" in supported_devices:
            return "cpu"
        raise ValueError("No supported device backend is available.")

    if requested not in supported_devices:
        supported = ", ".join(supported_devices)
        raise ValueError(
            f"Device {requested!r} is not supported for this model. "
            f"Supported devices: {supported}."
        )
    return requested


def _prompt_classes_for_sample(
    sample: DetectionSample,
    prompt_classes: str,
) -> tuple[str, ...]:
    if (
        prompt_classes == "image"
        and sample.ground_truth.class_id is not None
        and len(sample.ground_truth) > 0
    ):
        present_ids = set(sample.ground_truth.class_id)
        return tuple(sample.classes[class_id] for class_id in sorted(present_ids))
    return sample.classes


def _images_to_rerun(previous: RunResult | None) -> set[str]:
    if previous is None:
        return set()
    return {sample.image for sample in previous.samples if is_failed_sample(sample)}


def _validate_resume_configuration(
    resume_file: Path,
    current: RunManifest,
    *,
    max_samples: int | None,
    image_filter: str | None,
) -> None:
    if max_samples is not None or image_filter is not None:
        raise ValueError("Resuming with --max-samples or --image is not supported.")
    previous_path = manifest_path_for_results(resume_file)
    if not previous_path.exists():
        raise ValueError(f"Resume manifest not found: {previous_path}")
    previous = load_manifest(previous_path)
    fields = (
        "model",
        "checkpoint",
        "checkpoint_revision",
        "checkpoint_sha256",
        "adapter",
        "dataset_annotations_sha256",
        "prompt_classes",
        "coordinate_format",
        "inference",
        "device",
        "precision",
        "prompt_asset_type",
        "prompt_set_sha256",
    )
    mismatches = [
        name for name in fields if getattr(previous, name) != getattr(current, name)
    ]
    if mismatches:
        raise ValueError(
            "Resume configuration does not match the original run: "
            + ", ".join(mismatches)
        )


def run_reference_benchmark(
    *,
    model_config: ReferenceModelConfig,
    dataset_directory: str,
    output_path: Path,
    timestamp: str | None = None,
    prompt_classes: str = "image",
    device: str = "auto",
    precision: str = "float32",
    max_samples: int | None = None,
    image_filter: str | None = None,
    resume_file: Path | None = None,
    verbose: bool = True,
    deviations: list[str] | None = None,
    prompt_set: LoadedPromptSet | None = None,
) -> tuple[RunResult, RunManifest]:
    """Run a reference detection benchmark with incremental persistence.

    Args:
        model_config: Reference model configuration entry.
        dataset_directory: Path to the COCO dataset directory.
        output_path: JSONL output path for the run.
        timestamp: Run timestamp used by the output filename and stored metadata.
        prompt_classes: ``image`` or ``all`` class listing mode.
        device: Device backend or ``auto``.
        precision: Numeric precision label recorded in the manifest.
        max_samples: Optional cap on processed samples.
        image_filter: Optional image basename to run exclusively.
        resume_file: Optional partial JSONL to resume from.
        verbose: Whether to print progress to stdout.
        deviations: Documented deviations from the standard procedure.
        prompt_set: Optional image-conditioned prompt asset.

    Returns:
        Completed run result and final manifest.
    """
    resolved_device = resolve_device(device, model_config.supported_devices)
    task = DetectionTask(
        prompt_classes=prompt_classes,
        coordinate_format=model_config.coordinate_format,
    )
    all_samples = task.load_samples(dataset_directory)
    if image_filter is not None:
        all_samples = [
            sample
            for sample in all_samples
            if os.path.basename(sample.image_path) == image_filter
        ]
    if max_samples is not None:
        all_samples = all_samples[:max_samples]

    previous_run = load_results(resume_file) if resume_file is not None else None
    run_timestamp = (
        previous_run.timestamp if previous_run else (timestamp or new_run_timestamp())
    )
    if previous_run is not None and timestamp not in (None, previous_run.timestamp):
        raise ValueError("Resume timestamp must match the original run.")
    failed_images = _images_to_rerun(previous_run)
    if previous_run is not None and previous_run.model != model_config.key:
        raise ValueError(
            f"Resume file is for {previous_run.model!r}, "
            f"but requested model is {model_config.key!r}."
        )

    samples_to_process = [
        sample
        for sample in all_samples
        if os.path.basename(sample.image_path) in failed_images
        or previous_run is None
        or not any(
            existing.image == os.path.basename(sample.image_path)
            and not is_failed_sample(existing)
            for existing in previous_run.samples
        )
    ]
    if previous_run is not None and verbose:
        kept = len(all_samples) - len(samples_to_process)
        print(
            f"Resuming {model_config.key}: keeping {kept} samples, "
            f"re-running {len(samples_to_process)} samples."
        )

    manifest = build_run_manifest(
        model_config=model_config,
        effort=REFERENCE_EFFORT,
        task="detection",
        timestamp=run_timestamp,
        dataset_directory=dataset_directory,
        prompt_classes=prompt_classes,
        device=resolved_device,
        precision=precision,
        deviations=deviations,
    )
    manifest.dataset_image_count = len(all_samples)
    if prompt_set is not None:
        manifest.prompt_asset_type = prompt_set.asset_type.value
        manifest.prompt_set_path = str(prompt_set.path)
        manifest.prompt_set_version = prompt_set.version
        manifest.prompt_set_sha256 = prompt_set.sha256
    if resume_file is not None:
        _validate_resume_configuration(
            resume_file,
            manifest,
            max_samples=max_samples,
            image_filter=image_filter,
        )
    manifest_path = manifest_path_for_results(output_path)
    save_manifest(manifest, manifest_path)

    adapter = create_reference_adapter(model_config, device=resolved_device)
    resumed_samples: list[SampleResult] = []
    total = len(samples_to_process)
    run_start = time.perf_counter()
    image_to_index = {
        os.path.basename(sample.image_path): position
        for position, sample in enumerate(all_samples)
    }
    partial_run = RunResult(
        model=model_config.key,
        effort=REFERENCE_EFFORT,
        task="detection",
        timestamp=run_timestamp,
        samples=[],
    )

    for offset, sample in enumerate(samples_to_process):
        assert isinstance(sample, DetectionSample)
        image_name = os.path.basename(sample.image_path)
        index = image_to_index[image_name]
        class_names = _prompt_classes_for_sample(sample, prompt_classes)
        prompt_texts, prompt_to_canonical = resolve_prompt_texts(
            prompt_set,
            image=image_name,
            canonical_classes=class_names,
        )
        adapter.set_vocabulary(prompt_texts)

        elapsed_seconds: float | None = None
        prediction: str
        metadata: dict[str, Any] = {
            "prompt_classes": prompt_classes,
            "coordinate_format": model_config.coordinate_format.value,
            "reference": True,
            "prompt_class_names": list(class_names),
            "prompt_texts": list(prompt_texts),
            "classes_processed": "together",
            "device": resolved_device,
            "checkpoint": model_config.checkpoint,
        }

        try:
            image = ImageOps.exif_transpose(Image.open(sample.image_path)).convert(
                "RGB"
            )
            start_time = time.perf_counter()
            structured = adapter.predict(image)
            elapsed_seconds = time.perf_counter() - start_time
            prediction = serialize_reference_prediction(
                structured,
                label_remap=prompt_to_canonical,
            )
            metadata["num_predictions"] = len(structured.labels)
            metadata["num_ground_truth"] = len(sample.ground_truth)
            if structured.raw:
                metadata["adapter_raw"] = structured.raw
        except Exception as error:
            prediction = f"ERROR: {error}"
            metadata["num_predictions"] = 0
            metadata["num_ground_truth"] = len(sample.ground_truth)

        evaluation = task.evaluate(sample, prediction)
        if evaluation.details:
            metadata.update(evaluation.details)

        resumed_samples.append(
            SampleResult(
                index=index,
                image=image_name,
                expected="",
                predicted=prediction,
                correct=evaluation.correct,
                input_tokens=0,
                output_tokens=0,
                elapsed_seconds=elapsed_seconds,
                metadata=metadata,
            )
        )

        partial_run = RunResult(
            model=model_config.key,
            effort=REFERENCE_EFFORT,
            task="detection",
            timestamp=run_timestamp,
            samples=resumed_samples,
        )
        if previous_run is not None:
            partial_run = merge_resumed_runs(previous_run, partial_run)
        save_results(partial_run, output_path)
        save_manifest(manifest, manifest_path)

        if verbose:
            status = "ok" if evaluation.correct else "miss"
            map50 = metadata.get("map50")
            time_string = f"{elapsed_seconds:.1f}s" if elapsed_seconds else "N/A"
            if map50 is not None:
                print(
                    f"[{offset + 1}/{total}] {status}  {time_string}"
                    f"  mAP@50={map50:.3f}"
                    f"  pred={metadata.get('num_predictions', 0)}"
                    f"  gt={metadata.get('num_ground_truth', 0)}"
                )
            else:
                print(f"[{offset + 1}/{total}] {status}  {time_string}  {image_name}")

    if previous_run is not None and resumed_samples:
        final_run = merge_resumed_runs(previous_run, partial_run)
    elif previous_run is not None:
        final_run = previous_run
    elif resumed_samples:
        final_run = partial_run
    else:
        final_run = RunResult(
            model=model_config.key,
            effort=REFERENCE_EFFORT,
            task="detection",
            timestamp=run_timestamp,
            samples=[],
        )

    final_run = RunResult(
        model=final_run.model,
        effort=final_run.effort,
        task=final_run.task,
        timestamp=final_run.timestamp,
        samples=[
            SampleResult(
                index=position,
                image=sample.image,
                expected=sample.expected,
                predicted=sample.predicted,
                correct=sample.correct,
                input_tokens=sample.input_tokens,
                output_tokens=sample.output_tokens,
                elapsed_seconds=sample.elapsed_seconds,
                metadata=sample.metadata,
            )
            for position, sample in enumerate(final_run.samples)
        ],
    )
    save_results(final_run, output_path)

    manifest.completed_sample_count = len(final_run.samples)
    manifest.failed_sample_count = sum(
        1 for sample in final_run.samples if is_failed_sample(sample)
    )
    manifest.total_seconds = time.perf_counter() - run_start
    save_manifest(manifest, manifest_path)
    return final_run, manifest
