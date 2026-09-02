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
from vlm_exam.tasks.qa import QATask, answers_match

JUDGE_TASK_NAMES: tuple[str, ...] = ("extraction", "identification", "reasoning")
"""Tasks scored with strict matching plus an LLM judge fallback."""


@dataclass(frozen=True)
class RescoreSummary:
    """Outcome of re-judging one stored run.

    Attributes:
        judged: Samples sent to the judge (strict failures that had not
            been judged before).
        rescued: Judged samples the judge marked correct.
        correct_before: Correct samples before re-judging.
        correct_after: Correct samples after re-judging.
        total: Samples in the run.
    """

    judged: int
    rescued: int
    correct_before: int
    correct_after: int
    total: int


def needs_judge(sample: SampleResult) -> bool:
    """Report whether a stored sample still awaits a judge verdict.

    A sample needs the judge when strict matching failed, it has not
    been judged before, and its prediction is real model output rather
    than a provider error or empty-content marker.

    Args:
        sample: A sample loaded from a run file.

    Returns:
        True when the judge should be consulted for this sample.
    """
    if sample.correct or is_failed_sample(sample):
        return False
    if sample.predicted == EMPTY_RESPONSE_TEXT:
        return False
    return sample.metadata.get("match_method") != "judge"


def rescore_run(
    run: RunResult,
    judge: Judge,
    concurrency: int = 1,
) -> tuple[RunResult, RescoreSummary]:
    """Apply the judge fallback to every unjudged strict failure in a run.

    Samples already judged, already correct, or holding provider errors
    are left untouched, so re-running on a judge-mode file is a no-op.

    Args:
        run: A stored run for one of the ``JUDGE_TASK_NAMES`` tasks.
        judge: Judge used for the fallback verdicts.
        concurrency: Number of in-flight judge calls.

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
    guidance = task.judge_guidance

    def judge_sample(sample: SampleResult) -> SampleResult:
        correct, match_method = answers_match(
            sample.expected,
            sample.predicted,
            question=str(sample.metadata.get("question", "")),
            match_mode="judge",
            judge=judge,
            guidance=guidance,
        )
        return replace(
            sample,
            correct=correct,
            metadata={**sample.metadata, "match_method": match_method},
        )

    pending_positions = [
        position for position, sample in enumerate(run.samples) if needs_judge(sample)
    ]
    samples = list(run.samples)
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
        judged = executor.map(judge_sample, (run.samples[p] for p in pending_positions))
        for position, sample in zip(pending_positions, judged):
            samples[position] = sample

    correct_before = sum(1 for sample in run.samples if sample.correct)
    correct_after = sum(1 for sample in samples if sample.correct)
    summary = RescoreSummary(
        judged=len(pending_positions),
        rescued=correct_after - correct_before,
        correct_before=correct_before,
        correct_after=correct_after,
        total=len(samples),
    )
    return replace(run, samples=samples), summary
