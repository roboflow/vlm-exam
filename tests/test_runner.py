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

import threading
import time
from dataclasses import dataclass
from pathlib import Path

import pytest
from PIL import Image

from vlm_exam.judge import Judge
from vlm_exam.providers.base import Provider, RetryStats, Usage
from vlm_exam.runner import run_benchmark
from vlm_exam.tasks.base import EvaluationResult, Sample, Task


@dataclass(frozen=True)
class _StubSample(Sample):
    label: str = ""


class _StubTask(Task):
    def load_samples(self, data_directory: str) -> list[Sample]:
        return []

    def build_prompt(
        self,
        sample: Sample,
        *,
        uploaded_size: tuple[int, int] | None = None,
    ) -> str:
        assert isinstance(sample, _StubSample)
        return sample.label

    def evaluate(
        self,
        sample: Sample,
        prediction: str,
        *,
        judge: Judge | None = None,
        uploaded_size: tuple[int, int] | None = None,
    ) -> EvaluationResult:
        assert isinstance(sample, _StubSample)
        return EvaluationResult(
            correct=prediction == sample.label,
            strict_correct=prediction == sample.label,
            judge_correct=prediction == sample.label,
        )

    def expected_text(self, sample: Sample) -> str:
        assert isinstance(sample, _StubSample)
        return sample.label


class _StubProvider(Provider):
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._inflight = 0
        self.max_inflight = 0

    @property
    def model(self) -> str:
        return "stub"

    def predict(
        self,
        image: Image.Image,
        prompt: str,
        effort: str,
    ) -> tuple[str, Usage, RetryStats]:
        with self._lock:
            self._inflight += 1
            self.max_inflight = max(self.max_inflight, self._inflight)
        time.sleep(0.05)
        with self._lock:
            self._inflight -= 1
        return prompt, Usage(1, 1), RetryStats(attempts=1, inference_seconds=0.05)


def _make_samples(directory: Path, count: int) -> list[_StubSample]:
    image_path = directory / "pixel.png"
    Image.new("RGB", (8, 8), "white").save(image_path)
    return [
        _StubSample(image_path=str(image_path), label=f"sample-{index}")
        for index in range(count)
    ]


def test_run_benchmark_rejects_invalid_concurrency(
    tmp_path: Path,
) -> None:
    samples = _make_samples(tmp_path, 1)
    with pytest.raises(ValueError, match="concurrency"):
        run_benchmark(
            task=_StubTask(),
            provider=_StubProvider(),
            samples=samples,
            effort="low",
            task_name="identification",
            verbose=False,
            concurrency=0,
        )


def test_run_benchmark_parallel_preserves_order_and_overlaps(
    tmp_path: Path,
) -> None:
    samples = _make_samples(tmp_path, 8)
    provider = _StubProvider()
    result = run_benchmark(
        task=_StubTask(),
        provider=provider,
        samples=samples,
        effort="low",
        task_name="identification",
        verbose=False,
        concurrency=4,
    )

    assert [sample.index for sample in result.samples] == list(range(8))
    assert [sample.predicted for sample in result.samples] == [
        f"sample-{index}" for index in range(8)
    ]
    assert all(sample.correct for sample in result.samples)
    assert provider.max_inflight >= 2
