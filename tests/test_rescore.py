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
from vlm_exam.rescore import needs_judge, rescore_run
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
    match_method: str = "strict",
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
        metadata={"question": f"q{index}", "match_method": match_method, **metadata},
    )


def _run(task: str, samples: list[SampleResult]) -> RunResult:
    return RunResult(
        model="m", effort="low", task=task, timestamp="20260101_000000", samples=samples
    )


def test_needs_judge_only_for_unjudged_strict_failures() -> None:
    assert needs_judge(_sample(0, "7", "seven", False))
    assert not needs_judge(_sample(1, "7", "7", True))
    assert not needs_judge(_sample(2, "7", "eight", False, match_method="judge"))
    assert not needs_judge(_sample(3, "7", "ERROR: boom", False))
    assert not needs_judge(_sample(4, "7", EMPTY_RESPONSE_TEXT, False))


def test_rescore_run_rewrites_only_pending_samples() -> None:
    judge = _StubJudge({"seven": True, "eight": False})
    run = _run(
        "reasoning",
        [
            _sample(0, "7", "7", True),
            _sample(1, "7", "seven", False),
            _sample(2, "7", "eight", False),
            _sample(3, "7", "nine", False, match_method="judge"),
            _sample(4, "7", "ERROR: boom", False),
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
    assert [sample.metadata["match_method"] for sample in rescored.samples] == [
        "strict",
        "judge",
        "judge",
        "judge",
        "strict",
    ]
    assert summary.judged == 2
    assert summary.rescued == 1
    assert summary.correct_before == 1
    assert summary.correct_after == 2
    assert summary.total == 5
    assert {call["predicted"] for call in judge.calls} == {"seven", "eight"}
    assert all("reasoning task" in call["guidance"] for call in judge.calls)
    assert judge.calls[0]["question"].startswith("q")
    assert run.samples[1].correct is False


def test_rescore_run_is_noop_on_judge_mode_run() -> None:
    judge = _StubJudge({})
    run = _run(
        "extraction",
        [
            _sample(0, "a", "a", True),
            _sample(1, "a", "b", False, match_method="judge"),
        ],
    )

    rescored, summary = rescore_run(run, judge)

    assert rescored.samples == run.samples
    assert summary.judged == 0
    assert judge.calls == []


def test_rescore_run_rejects_non_judge_tasks() -> None:
    with pytest.raises(ValueError, match="not judge-scored"):
        rescore_run(_run("counting", []), _StubJudge({}))
