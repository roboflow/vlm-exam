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

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace

from vlm_exam.judge import Judge
from vlm_exam.providers.base import EMPTY_RESPONSE_TEXT
from vlm_exam.results import RunResult, SampleResult, is_failed_sample
from vlm_exam.tasks import create_task
from vlm_exam.tasks.qa import QATask, judge_answer

JUDGE_TASK_NAMES: tuple[str, ...] = (
    "counting",
    "extraction",
    "identification",
    "reasoning",
)
"""Tasks that report strict accuracy and LLM judge accuracy side by side."""

_LEGACY_METADATA_KEYS = ("match_method",)


@dataclass(frozen=True)
class RescoreSummary:
    """Outcome of re-scoring one stored run.

    Attributes:
        scored: Samples that were (re)scored.
        judge_calls: Samples sent to the judge (real model output only).
        strict_before: Strict-correct samples before re-scoring, when the
            run already carried strict verdicts.
        strict_after: Strict-correct samples after re-scoring.
        judge_before: Judge-correct samples before re-scoring, when the
            run already carried judge verdicts.
        judge_after: Judge-correct samples after re-scoring.
        total: Samples in the run.
    """

    scored: int
    judge_calls: int
    strict_before: int | None
    strict_after: int
    judge_before: int | None
    judge_after: int
    total: int


def has_both_verdicts(sample: SampleResult) -> bool:
    """Report whether a stored sample already carries both verdicts.

    Args:
        sample: A sample loaded from a run file.

    Returns:
        True when ``strict_correct`` and ``judge_correct`` are both
        recorded and no legacy scoring metadata remains.
    """
    metadata = sample.metadata
    if any(key in metadata for key in _LEGACY_METADATA_KEYS):
        return False
    return isinstance(metadata.get("strict_correct"), bool) and isinstance(
        metadata.get("judge_correct"), bool
    )


def is_scorable(sample: SampleResult) -> bool:
    """Report whether a sample holds real model output worth judging.

    Args:
        sample: A sample loaded from a run file.

    Returns:
        False for provider errors and empty-content markers.
    """
    return not is_failed_sample(sample) and sample.predicted != EMPTY_RESPONSE_TEXT


def _count(samples: list[SampleResult], key: str) -> int | None:
    values = [sample.metadata.get(key) for sample in samples]
    if not all(isinstance(value, bool) for value in values):
        return None
    return sum(1 for value in values if value)


def rescore_run(
    run: RunResult,
    judge: Judge,
    *,
    concurrency: int = 1,
    force: bool = False,
) -> tuple[RunResult, RescoreSummary]:
    """Score every sample of a stored run with the strict rule and the judge.

    The strict verdict is recomputed offline from the stored prediction;
    the judge is asked about every real prediction. Samples that already
    carry both verdicts are skipped unless ``force`` is set. Provider
    errors and empty responses get both verdicts set to false without a
    judge call. Legacy ``match_method`` metadata is dropped.

    Args:
        run: A stored run for one of the ``JUDGE_TASK_NAMES`` tasks.
        judge: Judge producing the LLM verdicts.
        concurrency: Number of in-flight judge calls.
        force: Re-score samples that already carry both verdicts.

    Returns:
        The rescored run and a summary of what changed.

    Raises:
        ValueError: If the run's task is not judge-scored.
    """
    if run.task not in JUDGE_TASK_NAMES:
        raise ValueError(
            f"Task {run.task!r} is not judge-scored; "
            f"expected one of {', '.join(JUDGE_TASK_NAMES)}."
        )
    task = create_task(run.task)
    assert isinstance(task, QATask)

    def score(sample: SampleResult) -> SampleResult:
        strict = False
        judged = False
        if is_scorable(sample):
            strict = task.strict_correct(sample.expected, sample.predicted)
            judged = judge_answer(
                sample.expected,
                sample.predicted,
                question=str(sample.metadata.get("question", "")),
                judge=judge,
                guidance=task.judge_guidance,
            )
        metadata = {
            key: value
            for key, value in sample.metadata.items()
            if key not in _LEGACY_METADATA_KEYS
        }
        metadata["strict_correct"] = strict
        metadata["judge_correct"] = judged
        return replace(sample, correct=judged, metadata=metadata)

    positions = [
        position
        for position, sample in enumerate(run.samples)
        if force or not has_both_verdicts(sample)
    ]
    samples = list(run.samples)
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
        scored = executor.map(score, (run.samples[p] for p in positions))
        for position, sample in zip(positions, scored):
            samples[position] = sample

    summary = RescoreSummary(
        scored=len(positions),
        judge_calls=sum(1 for p in positions if is_scorable(run.samples[p])),
        strict_before=_count(run.samples, "strict_correct"),
        strict_after=_count(samples, "strict_correct") or 0,
        judge_before=_count(run.samples, "judge_correct"),
        judge_after=_count(samples, "judge_correct") or 0,
        total=len(samples),
    )
    return replace(run, samples=samples), summary
