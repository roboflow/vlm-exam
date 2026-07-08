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

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vlm_exam.tasks.base import EvaluationResult, Sample, Task

if TYPE_CHECKING:
    from vlm_exam.judge import Judge

_logger = logging.getLogger(__name__)

_ARTICLES_PATTERN = re.compile(r"^(a|an|the)\s+", re.IGNORECASE)
_MARKDOWN_PATTERN = re.compile(r"(\*{1,2}|`{1,3}|_{1,2})")
_CODE_FENCE_OPEN_PATTERN = re.compile(r"^```[a-zA-Z]*\n")
_CODE_FENCE_CLOSE_PATTERN = re.compile(r"\n```$")
_INTEGER_PATTERN = re.compile(r"\d+")
_WORD_PATTERN = re.compile(r"[a-z]+(?:-[a-z]+)?")

_CONCISE_PROMPT_TEMPLATE = (
    "{question}\n\n"
    "Answer as concisely as possible. Return ONLY the answer, "
    "no explanation or extra text."
)

OCR_CORRECT_THRESHOLD = 0.95
"""Similarity above which an OCR transcription counts as correct."""

_UNIT_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}

_TENS_WORDS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}


@dataclass(frozen=True)
class QASample(Sample):
    """A question-answering benchmark sample with an expected answer."""

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


def strict_match(expected: str, predicted: str) -> bool:
    """Check whether two answers match using strict deterministic rules.

    Compares normalized forms using exact match and space-stripped match.
    Does NOT use substring containment.

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

    if normalized_expected.replace(" ", "") == normalized_predicted.replace(" ", ""):
        return True

    return False


def answers_match(
    expected: str,
    predicted: str,
    *,
    question: str = "",
    match_mode: str = "strict",
    judge: Judge | None = None,
    guidance: str = "",
) -> tuple[bool, str]:
    """Check whether two answers are equivalent.

    In ``"strict"`` mode only deterministic normalization rules are used.
    In ``"judge"`` mode an LLM judge is consulted when strict rules fail.

    Args:
        expected: Ground-truth answer.
        predicted: Model-produced answer.
        question: Original question text (used by judge for context).
        match_mode: ``"strict"`` or ``"judge"``.
        judge: A :class:`~vlm_exam.judge.Judge` instance (required when
            *match_mode* is ``"judge"``).
        guidance: Optional task-specific instructions for the judge.

    Returns:
        A tuple of (correct, match_method) where match_method is
        ``"strict"`` or ``"judge"``.
    """
    if strict_match(expected, predicted):
        return True, "strict"

    if match_mode == "judge" and judge is not None:
        return judge.evaluate(
            question=question,
            expected=expected,
            predicted=predicted,
            guidance=guidance,
        ), "judge"

    if match_mode == "judge" and judge is None:
        _logger.warning(
            "Judge mode requested but no judge instance provided; "
            "falling back to strict."
        )

    return False, "strict"


def _words_to_int(text: str) -> int | None:
    if text in _UNIT_WORDS:
        return _UNIT_WORDS[text]
    if text in _TENS_WORDS:
        return _TENS_WORDS[text]
    if "-" in text:
        tens_part, _, unit_part = text.partition("-")
        if tens_part in _TENS_WORDS and unit_part in _UNIT_WORDS:
            unit_value = _UNIT_WORDS[unit_part]
            if 1 <= unit_value <= 9:
                return _TENS_WORDS[tens_part] + unit_value
    return None


def parse_count(text: str) -> int | None:
    """Extract an integer count from an answer string.

    Handles plain digits, spelled-out number words up to ninety-nine,
    and answers with a single embedded integer (e.g. ``"4 bars"``).

    Args:
        text: Raw answer text.

    Returns:
        The parsed count, or ``None`` when no unambiguous integer is
        found.
    """
    cleaned = normalize_answer(text)
    if not cleaned:
        return None

    try:
        return int(cleaned)
    except ValueError:
        pass

    word_value = _words_to_int(cleaned)
    if word_value is not None:
        return word_value

    digits = _INTEGER_PATTERN.findall(cleaned)
    if len(digits) == 1:
        return int(digits[0])
    if digits:
        return None

    tokens = _WORD_PATTERN.findall(cleaned)
    word_values = [
        value
        for value in (_words_to_int(token) for token in tokens)
        if value is not None
    ]
    if len(word_values) == 1:
        return word_values[0]

    return None


def normalize_transcription(text: str) -> str:
    """Normalize a transcription for similarity comparison.

    Unifies line endings, strips a wrapping markdown code fence, and
    removes trailing whitespace on each line. Casing, punctuation, and
    line structure are preserved because they are part of the OCR task.

    Args:
        text: Raw transcription text.

    Returns:
        Normalized transcription.
    """
    text = text.replace("\r\n", "\n").strip()
    text = _CODE_FENCE_OPEN_PATTERN.sub("", text)
    text = _CODE_FENCE_CLOSE_PATTERN.sub("", text)
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def transcription_similarity(expected: str, predicted: str) -> float:
    """Compute character-level similarity between two transcriptions.

    Uses normalized Levenshtein similarity (1 minus normalized edit
    distance) on lightly normalized text.

    Args:
        expected: Ground-truth transcription.
        predicted: Model-produced transcription.

    Returns:
        Similarity in the range [0, 1].
    """
    from rapidfuzz.distance import Levenshtein

    normalized_expected = normalize_transcription(expected)
    normalized_predicted = normalize_transcription(predicted)
    return Levenshtein.normalized_similarity(normalized_expected, normalized_predicted)


class QATask(Task):
    """Base class for question-answering benchmark tasks."""

    judge_guidance = ""
    """Task-specific instructions forwarded to the LLM judge."""

    @staticmethod
    def _qa_sample(sample: Sample) -> QASample:
        assert isinstance(sample, QASample)
        return sample

    def expected_text(self, sample: Sample) -> str:
        """Return the sample's ground-truth answer.

        Args:
            sample: A ``QASample`` instance.

        Returns:
            The expected answer text.
        """
        return self._qa_sample(sample).expected_answer

    def sample_metadata(self, sample: Sample) -> dict[str, Any]:
        """Return the sample's question for result metadata.

        Args:
            sample: A ``QASample`` instance.

        Returns:
            Metadata dict holding the question text.
        """
        return {"question": self._qa_sample(sample).question}

    def load_samples(self, data_directory: str) -> list[Sample]:
        """Load QA samples from a JSONL annotations file.

        Expects the directory to contain an ``annotations.jsonl`` file
        where each line has ``image``, ``prefix`` (question), and
        ``suffix`` (answer) fields.

        Args:
            data_directory: Path to the dataset directory.

        Returns:
            List of QA samples.
        """
        annotations_path = os.path.join(data_directory, "annotations.jsonl")
        samples: list[Sample] = []

        with open(annotations_path) as file:
            for line in file:
                entry = json.loads(line)
                samples.append(
                    QASample(
                        image_path=os.path.join(data_directory, entry["image"]),
                        question=entry["prefix"],
                        expected_answer=entry["suffix"],
                    )
                )

        return samples

    def build_prompt(self, sample: Sample) -> str:
        """Build a short-answer prompt from a sample.

        Args:
            sample: A ``QASample`` instance.

        Returns:
            Formatted prompt string.
        """
        qa_sample = self._qa_sample(sample)
        return _CONCISE_PROMPT_TEMPLATE.format(question=qa_sample.question)

    def evaluate(
        self,
        sample: Sample,
        prediction: str,
        *,
        match_mode: str = "strict",
        judge: Judge | None = None,
    ) -> EvaluationResult:
        """Evaluate a prediction against the expected answer.

        Args:
            sample: A ``QASample`` with the ground-truth answer.
            prediction: Raw model output text.
            match_mode: ``"strict"`` or ``"judge"``.
            judge: Optional LLM judge for non-strict matching.

        Returns:
            Evaluation result with correctness flag.
        """
        qa_sample = self._qa_sample(sample)
        correct, match_method = answers_match(
            qa_sample.expected_answer,
            prediction,
            question=qa_sample.question,
            match_mode=match_mode,
            judge=judge,
            guidance=self.judge_guidance,
        )
        return EvaluationResult(correct=correct, match_method=match_method)


class OCRTask(QATask):
    """Full-text transcription task scored by character similarity."""

    def build_prompt(self, sample: Sample) -> str:
        """Return the OCR instruction block verbatim.

        OCR questions are self-contained instruction sets, so no
        conciseness suffix is added.

        Args:
            sample: A ``QASample`` instance.

        Returns:
            The question text unchanged.
        """
        return self._qa_sample(sample).question

    def evaluate(
        self,
        sample: Sample,
        prediction: str,
        *,
        match_mode: str = "strict",
        judge: Judge | None = None,
    ) -> EvaluationResult:
        """Score a transcription by normalized character similarity.

        Args:
            sample: A ``QASample`` with the ground-truth transcription.
            prediction: Raw model output text.
            match_mode: Ignored; OCR is always scored by similarity.
            judge: Ignored; OCR is always scored by similarity.

        Returns:
            Evaluation result with a [0, 1] similarity score and a
            correctness flag at the ``OCR_CORRECT_THRESHOLD``.
        """
        qa_sample = self._qa_sample(sample)
        score = transcription_similarity(qa_sample.expected_answer, prediction)
        return EvaluationResult(
            correct=score >= OCR_CORRECT_THRESHOLD,
            match_method="similarity",
            score=score,
        )


class ExtractionTask(QATask):
    """Single-field data extraction task."""

    judge_guidance = (
        "This is a data extraction task: the question asks for one specific "
        "value and often specifies an exact output format. Treat answers as "
        "equivalent only when they carry the same value; ignore purely "
        "cosmetic formatting differences the question does not forbid."
    )


class CountingTask(QATask):
    """Object counting task requiring an exact integer answer."""

    def evaluate(
        self,
        sample: Sample,
        prediction: str,
        *,
        match_mode: str = "strict",
        judge: Judge | None = None,
    ) -> EvaluationResult:
        """Compare parsed integer counts; the count must be exact.

        Args:
            sample: A ``QASample`` with the ground-truth count.
            prediction: Raw model output text.
            match_mode: Ignored unless the expected answer is not an
                integer, in which case strict matching applies.
            judge: Ignored; counts are compared deterministically.

        Returns:
            Evaluation result with correctness flag.
        """
        qa_sample = self._qa_sample(sample)
        expected_count = parse_count(qa_sample.expected_answer)
        if expected_count is None:
            correct = strict_match(qa_sample.expected_answer, prediction)
            return EvaluationResult(correct=correct, match_method="strict")

        predicted_count = parse_count(prediction)
        correct = predicted_count is not None and predicted_count == expected_count
        return EvaluationResult(correct=correct, match_method="count")


class IdentificationTask(QATask):
    """Entity identification task returning a name or descriptor."""

    judge_guidance = (
        "This is an identification task: the answer names an entity, type, "
        "color, material, or label. Accept synonyms or equally specific "
        "names for the same entity, unless the question demands an exact "
        "word or format."
    )


class ReasoningTask(QATask):
    """Visual reasoning task requiring inference beyond direct reading."""

    judge_guidance = (
        "This is a reasoning task: the answer is typically a number or a "
        "short phrase derived by inference. The underlying value must match "
        "exactly; only formatting may differ."
    )
