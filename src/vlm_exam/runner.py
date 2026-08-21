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

import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from PIL import Image, ImageOps

from vlm_exam.providers.base import Provider, RetryStats, Usage
from vlm_exam.results import RunResult, SampleResult
from vlm_exam.tasks.base import Sample, Task

if TYPE_CHECKING:
    from vlm_exam.judge import Judge

_logger = logging.getLogger(__name__)

_VERBOSE_TEXT_LIMIT = 60


def _truncate(text: str, limit: int = _VERBOSE_TEXT_LIMIT) -> str:
    flattened = " ".join(text.split())
    if len(flattened) <= limit:
        return flattened
    return flattened[: limit - 3] + "..."


def _warn_on_dimension_mismatch(sample: Sample, image: Image.Image) -> None:
    annotated_width = getattr(sample, "image_width", None)
    annotated_height = getattr(sample, "image_height", None)
    if annotated_width is None or annotated_height is None:
        return
    if (annotated_width, annotated_height) != image.size:
        _logger.warning(
            "Image %s is %dx%d on disk but annotated as %dx%d; pixel "
            "coordinate scaling may be off.",
            os.path.basename(sample.image_path),
            image.size[0],
            image.size[1],
            annotated_width,
            annotated_height,
        )


def _evaluate_sample(
    index: int,
    sample: Sample,
    task: Task,
    provider: Provider,
    effort: str,
    match_mode: str,
    judge: Judge | None,
) -> SampleResult:
    image = ImageOps.exif_transpose(Image.open(sample.image_path)).convert("RGB")
    _warn_on_dimension_mismatch(sample, image)
    uploaded_size = provider.uploaded_image_size(image)
    prompt = task.build_prompt(sample, uploaded_size=uploaded_size)

    usage = Usage(input_tokens=0, output_tokens=0)
    elapsed_seconds: float | None = None
    total_seconds: float | None = None
    retry_stats: RetryStats | None = None
    prediction: str
    predict_succeeded = False

    try:
        start_time = time.perf_counter()
        prediction, usage, retry_stats = provider.predict(image, prompt, effort)
        total_seconds = time.perf_counter() - start_time
        elapsed_seconds = retry_stats.inference_seconds
        predict_succeeded = True
    except Exception as error:
        prediction = f"ERROR: {error}"

    evaluation = task.evaluate(
        sample,
        prediction,
        match_mode=match_mode,
        judge=judge,
        uploaded_size=uploaded_size,
    )
    image_name = os.path.basename(sample.image_path)

    metadata: dict[str, Any] = task.sample_metadata(sample)
    if evaluation.match_method is not None:
        metadata["match_method"] = evaluation.match_method
    if evaluation.score is not None:
        metadata["score"] = round(evaluation.score, 4)
    if evaluation.details:
        metadata.update(evaluation.details)
    if predict_succeeded and uploaded_size is not None:
        metadata["uploaded_width"] = uploaded_size[0]
        metadata["uploaded_height"] = uploaded_size[1]
    if predict_succeeded and retry_stats is not None:
        metadata["attempts"] = retry_stats.attempts
        metadata["retries"] = retry_stats.attempts - 1
        if total_seconds is not None:
            metadata["total_seconds"] = round(total_seconds, 4)
        if retry_stats.transient_error_types:
            metadata["transient_errors"] = list(retry_stats.transient_error_types)
    expected = task.expected_text(sample)

    return SampleResult(
        index=index,
        image=image_name,
        expected=expected,
        predicted=prediction,
        correct=evaluation.correct,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        elapsed_seconds=elapsed_seconds,
        metadata=metadata,
    )


def _print_sample_progress(
    sample_result: SampleResult,
    total: int,
) -> None:
    status = "\u2705" if sample_result.correct else "\u274c"
    elapsed_seconds = sample_result.elapsed_seconds
    metadata = sample_result.metadata
    retries = int(metadata.get("retries", 0))
    total_seconds = metadata.get("total_seconds")
    if elapsed_seconds is None:
        time_string = "N/A"
    elif retries > 0 and total_seconds is not None:
        time_string = (
            f"{elapsed_seconds:.1f}s (total {total_seconds:.1f}s, {retries} retries)"
        )
    else:
        time_string = f"{elapsed_seconds:.1f}s"
    if "map50" in metadata:
        map50 = metadata["map50"]
        n_pred = metadata.get("num_predictions", 0)
        n_gt = metadata.get("num_ground_truth", 0)
        print(
            f"[{sample_result.index + 1}/{total}] {status}  {time_string}"
            f"  mAP@50={map50:.3f}"
            f"  pred={n_pred} gt={n_gt}"
        )
        return
    if "score" in metadata:
        print(
            f"[{sample_result.index + 1}/{total}] {status}  {time_string}"
            f"  similarity={metadata['score']:.3f}"
            f"  expected: {_truncate(sample_result.expected)!r}"
            f"  model: {_truncate(sample_result.predicted)!r}"
        )
        return
    print(
        f"[{sample_result.index + 1}/{total}] {status}  {time_string}"
        f"  expected: {_truncate(sample_result.expected)!r}"
        f"  model: {_truncate(sample_result.predicted)!r}"
    )


def run_benchmark(
    task: Task,
    provider: Provider,
    samples: list[Sample],
    effort: str,
    task_name: str,
    verbose: bool = True,
    match_mode: str = "strict",
    judge: Judge | None = None,
    concurrency: int = 1,
) -> RunResult:
    """Run a benchmark across all samples with a single provider.

    Args:
        task: The task instance that builds prompts and evaluates.
        provider: The provider instance to call for predictions.
        samples: List of samples to evaluate.
        effort: Effort level (e.g. ``"low"``, ``"high"``).
        task_name: Name of the task for result metadata.
        verbose: Whether to print progress to stdout.
        match_mode: ``"strict"`` or ``"judge"``.
        judge: Optional LLM judge instance for ``"judge"`` mode.
        concurrency: Number of in-flight provider calls. ``1`` keeps the
            original sequential loop.

    Returns:
        A complete run result with per-sample outcomes.
    """
    if concurrency < 1:
        raise ValueError(f"concurrency must be >= 1, got {concurrency}")

    total = len(samples)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    if verbose:
        print(f"\n{'=' * 60}")
        suffix = f", concurrency={concurrency}" if concurrency > 1 else ""
        print(f"  Running: {provider.model}  (effort={effort}{suffix})")
        print(f"{'=' * 60}\n")

    if concurrency == 1:
        sample_results = []
        for index, sample in enumerate(samples):
            sample_result = _evaluate_sample(
                index,
                sample,
                task,
                provider,
                effort,
                match_mode,
                judge,
            )
            sample_results.append(sample_result)
            if verbose:
                _print_sample_progress(sample_result, total)
    else:
        completed: dict[int, SampleResult] = {}
        print_lock = threading.Lock()
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(
                    _evaluate_sample,
                    index,
                    sample,
                    task,
                    provider,
                    effort,
                    match_mode,
                    judge,
                ): index
                for index, sample in enumerate(samples)
            }
            for future in as_completed(futures):
                sample_result = future.result()
                completed[sample_result.index] = sample_result
                if verbose:
                    with print_lock:
                        _print_sample_progress(sample_result, total)
        sample_results = [completed[index] for index in range(total)]

    return RunResult(
        model=provider.model,
        effort=effort,
        task=task_name,
        timestamp=timestamp,
        samples=sample_results,
    )
