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

from typing import Any

import pytest

from vlm_exam.judge import Judge
from vlm_exam.providers.base import EMPTY_RESPONSE_TEXT
from vlm_exam.rescore import has_both_verdicts, is_scorable, rescore_run
from vlm_exam.results import RunResult, SampleResult


class _StubJudge(Judge):
    def __init__(self, verdicts: dict[str, bool]) -> None:
        self._verdicts = verdicts
        self.calls: list[dict[str, str]] = []

    def evaluate(
        self, *, question: str, expected: str, predicted: str, guidance: str = ""
    ) -> bool:
        self.calls.append(
            {
                "question": question,
                "expected": expected,
                "predicted": predicted,
                "guidance": guidance,
            }
        )
        return self._verdicts[predicted]


def _sample(
    index: int,
    expected: str,
    predicted: str,
    correct: bool,
    **metadata: Any,
) -> SampleResult:
    return SampleResult(
        index=index,
        image=f"{index}.jpg",
        expected=expected,
        predicted=predicted,
        correct=correct,
        input_tokens=1,
        output_tokens=1,
        metadata={"question": f"q{index}", **metadata},
    )


def _run(task: str, samples: list[SampleResult]) -> RunResult:
    return RunResult(
        model="m", effort="low", task=task, timestamp="20260101_000000", samples=samples
    )


def test_has_both_verdicts_requires_both_flags_and_no_legacy_keys() -> None:
    assert has_both_verdicts(
        _sample(0, "7", "7", True, strict_correct=True, judge_correct=True)
    )
    assert not has_both_verdicts(_sample(1, "7", "7", True, strict_correct=True))
    assert not has_both_verdicts(_sample(2, "7", "7", True, match_method="strict"))
    assert not has_both_verdicts(
        _sample(
            3, "7", "7", True, strict_correct=True, judge_correct=True, match_method="x"
        )
    )


def test_is_scorable_rejects_errors_and_empty_output() -> None:
    assert is_scorable(_sample(0, "7", "seven", False))
    assert not is_scorable(_sample(1, "7", "ERROR: boom", False))
    assert not is_scorable(_sample(2, "7", EMPTY_RESPONSE_TEXT, False))


def test_rescore_run_scores_every_legacy_sample_with_both_rules() -> None:
    judge = _StubJudge({"7": True, "seven": True, "eight": False, "nine": False})
    run = _run(
        "reasoning",
        [
            _sample(0, "7", "7", True, match_method="strict"),
            _sample(1, "7", "seven", False, match_method="strict"),
            _sample(2, "7", "eight", False, match_method="judge"),
            _sample(3, "7", "nine", False, match_method="strict"),
            _sample(4, "7", "ERROR: boom", False, match_method="strict"),
        ],
    )

    rescored, summary = rescore_run(run, judge, concurrency=2)

    assert [sample.correct for sample in rescored.samples] == [
        True,
        True,
        False,
        False,
        False,
    ]
    assert [sample.metadata["strict_correct"] for sample in rescored.samples] == [
        True,
        False,
        False,
        False,
        False,
    ]
    assert [sample.metadata["judge_correct"] for sample in rescored.samples] == [
        True,
        True,
        False,
        False,
        False,
    ]
    assert all("match_method" not in sample.metadata for sample in rescored.samples)
    assert all(sample.metadata["question"] for sample in rescored.samples)
    assert summary.scored == 5
    assert summary.judge_calls == 4
    assert summary.strict_before is None
    assert summary.judge_before is None
    assert summary.strict_after == 1
    assert summary.judge_after == 2
    assert summary.total == 5
    assert {call["predicted"] for call in judge.calls} == {
        "7",
        "seven",
        "eight",
        "nine",
    }
    assert all("reasoning task" in call["guidance"] for call in judge.calls)
    assert run.samples[1].correct is False


def test_rescore_run_uses_counting_rule_and_guidance() -> None:
    judge = _StubJudge({"I see 4 or 5": True})
    sample = _sample(0, "4", "I see 4 or 5", False, match_method="count")
    run = _run("counting", [sample])

    rescored, _ = rescore_run(run, judge)

    assert rescored.samples[0].metadata["strict_correct"] is False
    assert rescored.samples[0].metadata["judge_correct"] is True
    assert rescored.samples[0].correct is True
    assert judge.calls[0]["guidance"].startswith("This is a counting task")


def test_rescore_run_skips_samples_with_both_verdicts_unless_forced() -> None:
    judge = _StubJudge({"a": False})
    run = _run(
        "extraction",
        [_sample(0, "a", "a", True, strict_correct=True, judge_correct=True)],
    )

    unchanged, summary = rescore_run(run, judge)
    assert unchanged.samples == run.samples
    assert summary.scored == 0
    assert summary.strict_before == 1
    assert summary.judge_before == 1
    assert judge.calls == []

    forced, forced_summary = rescore_run(run, judge, force=True)
    assert forced.samples[0].metadata["judge_correct"] is False
    assert forced.samples[0].correct is False
    assert forced_summary.scored == 1
    assert forced_summary.judge_after == 0


def test_rescore_run_rejects_non_judge_tasks() -> None:
    with pytest.raises(ValueError, match="not judge-scored"):
        rescore_run(_run("ocr", []), _StubJudge({}))
