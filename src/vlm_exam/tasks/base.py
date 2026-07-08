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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vlm_exam.judge import Judge


@dataclass(frozen=True)
class Sample:
    """Single benchmark sample with an image and task-specific data."""

    image_path: str


@dataclass(frozen=True)
class EvaluationResult:
    """Outcome of evaluating a model prediction against ground truth."""

    correct: bool
    match_method: str | None = None
    details: dict[str, Any] | None = None
    score: float | None = None


class Task(ABC):
    """Abstract base for benchmark tasks (OCR, counting, detection, etc.)."""

    def expected_text(self, sample: Sample) -> str:
        """Return the ground-truth text recorded in results.

        Args:
            sample: The sample being evaluated.

        Returns:
            The expected answer text, or an empty string for tasks
            whose ground truth is not textual.
        """
        return ""

    def sample_metadata(self, sample: Sample) -> dict[str, Any]:
        """Return task-specific metadata recorded per sample result.

        Args:
            sample: The sample being evaluated.

        Returns:
            Metadata entries to merge into the sample result.
        """
        return {}

    @abstractmethod
    def load_samples(self, data_directory: str) -> list[Sample]:
        """Load all samples from a dataset directory.

        Args:
            data_directory: Path to the directory containing images
                and annotation files.

        Returns:
            List of samples ready for evaluation.
        """
        ...

    @abstractmethod
    def build_prompt(self, sample: Sample) -> str:
        """Build the text prompt for a given sample.

        Args:
            sample: The sample to build a prompt for.

        Returns:
            Formatted prompt string.
        """
        ...

    @abstractmethod
    def evaluate(
        self,
        sample: Sample,
        prediction: str,
        *,
        match_mode: str = "strict",
        judge: Judge | None = None,
    ) -> EvaluationResult:
        """Evaluate a model prediction against the sample's ground truth.

        Args:
            sample: The original sample with expected answer.
            prediction: Raw text output from the model.
            match_mode: ``"strict"`` or ``"judge"``.
            judge: Optional LLM judge instance for non-strict matching.

        Returns:
            Evaluation result indicating correctness.
        """
        ...
