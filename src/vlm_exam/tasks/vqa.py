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

import json
import os
import re
from dataclasses import dataclass

from vlm_exam.tasks.base import EvaluationResult, Sample, Task

_ARTICLES_PATTERN = re.compile(r"^(a|an|the)\s+", re.IGNORECASE)
_MARKDOWN_PATTERN = re.compile(r"(\*{1,2}|`{1,3}|_{1,2})")

_PROMPT_TEMPLATE = (
    "{question}\n\n"
    "Answer as concisely as possible. Return ONLY the answer, "
    "no explanation or extra text."
)


@dataclass(frozen=True)
class VQASample(Sample):
    """A VQA benchmark sample with a question and expected answer."""

    question: str
    expected_answer: str


def normalize_answer(text: str) -> str:
    """Normalize an answer string for comparison.

    Strips markdown formatting, leading articles, and excess whitespace,
    then lowercases the result.

    Args:
        text: Raw answer text.

    Returns:
        Normalized lowercase string.
    """
    text = _MARKDOWN_PATTERN.sub("", text)
    text = text.strip()
    text = _ARTICLES_PATTERN.sub("", text)
    text = " ".join(text.split())
    return text.lower()


def answers_match(expected: str, predicted: str) -> bool:
    """Check whether two answers are semantically equivalent.

    Compares normalized forms using exact match, space-stripped match,
    and substring containment.

    Args:
        expected: Ground-truth answer.
        predicted: Model-produced answer.

    Returns:
        ``True`` if the answers are considered equivalent.
    """
    normalized_expected = normalize_answer(expected)
    normalized_predicted = normalize_answer(predicted)

    if normalized_expected == normalized_predicted:
        return True

    if normalized_expected.replace(" ", "") == normalized_predicted.replace(
        " ", ""
    ):
        return True

    if (
        normalized_expected in normalized_predicted
        or normalized_predicted in normalized_expected
    ):
        return True

    return False


class VQATask(Task):
    """Visual Question Answering / OCR benchmark task."""

    def load_samples(self, data_directory: str) -> list[Sample]:
        """Load VQA samples from a JSONL annotations file.

        Expects the directory to contain an ``annotations.jsonl`` file
        where each line has ``image``, ``prefix`` (question), and
        ``suffix`` (answer) fields.

        Args:
            data_directory: Path to the dataset directory.

        Returns:
            List of VQA samples.
        """
        annotations_path = os.path.join(data_directory, "annotations.jsonl")
        samples: list[Sample] = []

        with open(annotations_path) as file:
            for line in file:
                entry = json.loads(line)
                samples.append(
                    VQASample(
                        image_path=os.path.join(
                            data_directory, entry["image"]
                        ),
                        question=entry["prefix"],
                        expected_answer=entry["suffix"],
                    )
                )

        return samples

    def build_prompt(self, sample: Sample) -> str:
        """Build a VQA prompt from a sample.

        Args:
            sample: A ``VQASample`` instance.

        Returns:
            Formatted prompt string.
        """
        assert isinstance(sample, VQASample)
        return _PROMPT_TEMPLATE.format(question=sample.question)

    def evaluate(self, sample: Sample, prediction: str) -> EvaluationResult:
        """Evaluate a VQA prediction against the expected answer.

        Args:
            sample: A ``VQASample`` with the ground-truth answer.
            prediction: Raw model output text.

        Returns:
            Evaluation result with correctness flag.
        """
        assert isinstance(sample, VQASample)
        correct = answers_match(sample.expected_answer, prediction)
        return EvaluationResult(correct=correct)
