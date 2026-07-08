# Copyright 2026 Roboflow, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from vlm_exam.results import (
    RunResult,
    SampleResult,
    is_failed_sample,
    merge_resumed_runs,
)


def _sample(index: int, image: str, predicted: str, correct: bool) -> SampleResult:
    return SampleResult(
        index=index,
        image=image,
        expected="",
        predicted=predicted,
        correct=correct,
        input_tokens=10,
        output_tokens=10,
    )


def _run(samples: list[SampleResult], timestamp: str = "20260707_000000") -> RunResult:
    return RunResult(
        model="test-model",
        effort="low",
        task="detection",
        timestamp=timestamp,
        samples=samples,
    )


class TestIsFailedSample:
    def test_error_prediction_is_failed(self) -> None:
        assert is_failed_sample(_sample(0, "a.jpg", "ERROR: boom", False)) is True

    def test_regular_prediction_is_not_failed(self) -> None:
        assert is_failed_sample(_sample(0, "a.jpg", "[]", True)) is False


class TestMergeResumedRuns:
    def test_replaces_failed_samples_and_keeps_successful(self) -> None:
        previous = _run(
            [
                _sample(0, "a.jpg", "[]", True),
                _sample(1, "b.jpg", "ERROR: credit balance", False),
                _sample(2, "c.jpg", "[]", True),
                _sample(3, "d.jpg", "ERROR: credit balance", False),
            ]
        )
        resumed = _run(
            [
                _sample(0, "b.jpg", '[{"box_2d": [0, 0, 1, 1], "label": "x"}]', True),
                _sample(1, "d.jpg", "[]", False),
            ],
            timestamp="20260707_111111",
        )

        merged = merge_resumed_runs(previous, resumed)

        assert [sample.image for sample in merged.samples] == [
            "a.jpg",
            "b.jpg",
            "c.jpg",
            "d.jpg",
        ]
        assert [sample.index for sample in merged.samples] == [0, 1, 2, 3]
        assert not any(is_failed_sample(sample) for sample in merged.samples)
        assert merged.samples[1].correct is True
        assert merged.timestamp == "20260707_111111"

    def test_failed_sample_without_replacement_is_kept(self) -> None:
        previous = _run([_sample(0, "a.jpg", "ERROR: boom", False)])
        resumed = _run([], timestamp="20260707_111111")

        merged = merge_resumed_runs(previous, resumed)

        assert len(merged.samples) == 1
        assert is_failed_sample(merged.samples[0]) is True

    def test_successful_sample_is_not_replaced(self) -> None:
        previous = _run([_sample(0, "a.jpg", "[]", True)])
        resumed = _run(
            [_sample(0, "a.jpg", "ERROR: should not appear", False)],
            timestamp="20260707_111111",
        )

        merged = merge_resumed_runs(previous, resumed)

        assert merged.samples[0].predicted == "[]"
