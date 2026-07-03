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

import os
import time
from datetime import datetime, timezone

from PIL import Image

from vlm_exam.providers.base import Provider, Usage
from vlm_exam.results import RunResult, SampleResult
from vlm_exam.tasks.base import Sample, Task


def run_benchmark(
    task: Task,
    provider: Provider,
    samples: list[Sample],
    effort: str,
    task_name: str = "vqa",
    verbose: bool = True,
) -> RunResult:
    """Run a benchmark across all samples with a single provider.

    Args:
        task: The task instance that builds prompts and evaluates.
        provider: The provider instance to call for predictions.
        samples: List of samples to evaluate.
        effort: Effort level (e.g. ``"low"``, ``"high"``).
        task_name: Name of the task for result metadata.
        verbose: Whether to print progress to stdout.

    Returns:
        A complete run result with per-sample outcomes.
    """
    total = len(samples)
    sample_results: list[SampleResult] = []
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"  Running: {provider.model}  (effort={effort})")
        print(f"{'=' * 60}\n")

    for index, sample in enumerate(samples):
        image = Image.open(sample.image_path).convert("RGB")
        prompt = task.build_prompt(sample)

        usage = Usage(input_tokens=0, output_tokens=0)
        elapsed_seconds: float | None = None
        prediction: str

        try:
            start_time = time.perf_counter()
            prediction, usage = provider.predict(image, prompt, effort)
            elapsed_seconds = time.perf_counter() - start_time
        except Exception as error:
            prediction = f"ERROR: {error}"

        evaluation = task.evaluate(sample, prediction)
        image_name = os.path.basename(sample.image_path)

        metadata: dict[str, str] = {}
        if hasattr(sample, "question"):
            metadata["question"] = sample.question
        expected = getattr(sample, "expected_answer", "")

        sample_results.append(
            SampleResult(
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
        )

        if verbose:
            status = "\u2705" if evaluation.correct else "\u274c"
            time_string = (
                f"{elapsed_seconds:.1f}s"
                if elapsed_seconds is not None
                else "N/A"
            )
            print(
                f"[{index + 1}/{total}] {status}  {time_string}"
                f"  expected: {expected!r}"
                f"  model: {prediction!r}"
            )

    return RunResult(
        model=provider.model,
        effort=effort,
        task=task_name,
        timestamp=timestamp,
        samples=sample_results,
    )
