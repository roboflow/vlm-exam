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

from pathlib import Path
from typing import Any

import pytest

from vlm_exam.config import (
    BenchmarkConfig,
    LabConfig,
    ModelConfig,
    PricingConfig,
    RouteConfig,
)
from vlm_exam.metrics import BENCHMARK_TASK_NAMES
from vlm_exam.results import RunResult, SampleResult, save_results
from vlm_exam.summary import (
    _TASK_DEFINITIONS,
    build_summary,
    summary_to_dict,
)
from vlm_exam.tasks.detection import DetectionCoordinateFormat


def _model(model_id: str) -> ModelConfig:
    return ModelConfig(
        name=model_id,
        lab="openai",
        routes=(RouteConfig("openai"),),
        pricing=PricingConfig(1.0, 2.0),
        detection_coordinate_format=DetectionCoordinateFormat.XYXY_ABSOLUTE_ORIGINAL_IMAGE,
    )


def _config(*model_ids: str) -> BenchmarkConfig:
    return BenchmarkConfig(
        labs={"openai": LabConfig("OpenAI", "#000", "https://example.com/logo.svg")},
        models={model_id: _model(model_id) for model_id in model_ids},
    )


def _sample(
    index: int = 0,
    correct: bool = True,
    metadata: dict[str, Any] | None = None,
) -> SampleResult:
    return SampleResult(
        index=index,
        image=f"{index}.jpg",
        expected="",
        predicted="",
        correct=correct,
        input_tokens=100,
        output_tokens=50,
        elapsed_seconds=1.0,
        metadata=metadata or {},
    )


def _run(
    model: str,
    task: str,
    timestamp: str = "20260707_000000",
    effort: str = "low",
    samples: list[SampleResult] | None = None,
) -> RunResult:
    return RunResult(
        model=model,
        effort=effort,
        task=task,
        timestamp=timestamp,
        samples=samples if samples is not None else [_sample()],
    )


def _save(run: RunResult, directory: Path) -> None:
    filename = f"{run.task}_{run.model}_{run.effort}_{run.timestamp}.jsonl"
    save_results(run, directory / filename)


class TestTaskRegistry:
    def test_covers_all_benchmark_tasks(self) -> None:
        assert set(BENCHMARK_TASK_NAMES) <= set(_TASK_DEFINITIONS)


class TestBuildSummary:
    def test_one_entry_per_model_effort(self, tmp_path: Path) -> None:
        config = _config("alpha")
        _save(_run("alpha", "counting", effort="low"), tmp_path)
        _save(_run("alpha", "counting", effort="high"), tmp_path)

        summary = build_summary(tmp_path, config)

        assert [(model.id, model.effort) for model in summary.models] == [
            ("alpha:low", "low"),
            ("alpha:high", "high"),
        ]
        assert summary.efforts == ("low", "high")

    def test_effort_filter_keeps_single_effort(self, tmp_path: Path) -> None:
        config = _config("alpha")
        _save(_run("alpha", "counting", effort="low"), tmp_path)
        _save(_run("alpha", "counting", effort="high"), tmp_path)

        summary = build_summary(tmp_path, config, effort="high")

        assert [model.id for model in summary.models] == ["alpha:high"]
        assert summary.efforts == ("high",)

    def test_newest_run_wins(self, tmp_path: Path) -> None:
        config = _config("alpha")
        old = _run(
            "alpha",
            "counting",
            timestamp="20260701_000000",
            samples=[_sample(correct=False)],
        )
        new = _run(
            "alpha",
            "counting",
            timestamp="20260702_000000",
            samples=[_sample(correct=True)],
        )
        _save(old, tmp_path)
        _save(new, tmp_path)

        summary = build_summary(tmp_path, config)

        counting = summary.models[0].tasks["counting"]
        assert counting.metrics == {"accuracy": 100.0}
        assert counting.timestamp == "20260702_000000"

    def test_qa_reports_accuracy(self, tmp_path: Path) -> None:
        config = _config("alpha")
        samples = [
            _sample(index=0, correct=True),
            _sample(index=1, correct=True),
            _sample(index=2, correct=False),
        ]
        _save(_run("alpha", "reasoning", samples=samples), tmp_path)

        summary = build_summary(tmp_path, config)

        result = summary.models[0].tasks["reasoning"]
        assert result.metrics == {"accuracy": pytest.approx(200 / 3)}
        assert result.primary_metric is not None
        assert result.primary_metric.name == "accuracy"
        assert result.evaluated_sample_count is None

    def test_ocr_reports_only_similarity(self, tmp_path: Path) -> None:
        config = _config("alpha")
        samples = [
            _sample(index=0, correct=False, metadata={"score": 0.5}),
            _sample(index=1, correct=True, metadata={"score": 1.0}),
        ]
        _save(_run("alpha", "ocr", samples=samples), tmp_path)

        summary = build_summary(tmp_path, config)

        result = summary.models[0].tasks["ocr"]
        assert result.metrics == {"similarity": pytest.approx(75.0)}
        assert result.primary_metric is not None
        assert result.primary_metric.name == "similarity"

    def test_detection_without_index_omits_quality(self, tmp_path: Path) -> None:
        config = _config("alpha")
        _save(_run("alpha", "detection"), tmp_path)

        summary = build_summary(tmp_path, config)

        result = summary.models[0].tasks["detection"]
        assert result.metrics == {}
        assert result.primary_metric is None
        assert result.evaluated_sample_count is None
        assert result.tokens.total == 150

    def test_skips_unregistered_tasks_with_warning(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        config = _config("alpha")
        _save(_run("alpha", "counting"), tmp_path)
        _save(_run("alpha", "vqa"), tmp_path)

        summary = build_summary(tmp_path, config)

        assert [task.key for task in summary.tasks] == ["counting"]
        assert list(summary.models[0].tasks) == ["counting"]
        assert "unregistered task" in capsys.readouterr().out

    def test_generated_at_is_deterministic(self, tmp_path: Path) -> None:
        config = _config("alpha")
        _save(_run("alpha", "counting", timestamp="20260701_120000"), tmp_path)
        _save(_run("alpha", "ocr", timestamp="20260703_060000"), tmp_path)

        first = build_summary(tmp_path, config)
        second = build_summary(tmp_path, config)

        assert first.generated_at == "2026-07-03T06:00:00Z"
        assert summary_to_dict(first) == summary_to_dict(second)

    def test_empty_results_directory(self, tmp_path: Path) -> None:
        config = _config("alpha")

        summary = build_summary(tmp_path, config)

        assert summary.generated_at is None
        assert summary.efforts == ()
        assert summary.tasks == []
        assert summary.models == []

    def test_task_metadata_shape(self, tmp_path: Path) -> None:
        config = _config("alpha")
        _save(_run("alpha", "detection"), tmp_path)

        summary = build_summary(tmp_path, config)

        (task,) = summary.tasks
        assert task.key == "detection"
        assert task.primary_metric == "map50"
        assert [metric.key for metric in task.metrics] == [
            "map50",
            "map75",
            "map50_95",
        ]


class TestSummaryToDict:
    def test_payload_shape(self, tmp_path: Path) -> None:
        config = _config("alpha")
        samples = [
            _sample(index=0, correct=True),
            _sample(index=1, correct=True),
            _sample(index=2, correct=False),
        ]
        _save(
            _run("alpha", "counting", timestamp="20260710_073333", samples=samples),
            tmp_path,
        )

        payload = summary_to_dict(build_summary(tmp_path, config))

        assert list(payload) == ["generated_at", "efforts", "tasks", "models"]
        assert payload["generated_at"] == "2026-07-10T07:33:33Z"
        assert payload["efforts"] == ["low"]

        (task,) = payload["tasks"]
        assert list(task) == ["key", "name", "primary_metric", "metrics"]
        assert task["metrics"] == [
            {"key": "accuracy", "label": "Accuracy", "unit": "percent"}
        ]

        (model,) = payload["models"]
        assert model["id"] == "alpha:low"
        assert model["key"] == "alpha"
        assert model["effort"] == "low"
        assert "pricing" not in model

        counting = model["tasks"]["counting"]
        assert counting["metrics"] == {"accuracy": 66.67}
        assert counting["primary_metric"] == {"name": "accuracy", "value": 66.67}
        assert counting["evaluated_sample_count"] is None
        assert counting["timestamp"] == "2026-07-10T07:33:33Z"
        assert model["overall"]["sample_count"] == 3
