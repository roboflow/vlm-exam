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

import pytest
import yaml

from vlm_exam.config import (
    BenchmarkConfig,
    LabConfig,
    ModelConfig,
    PricingConfig,
    RouteConfig,
    load_leaderboard_groups,
)
from vlm_exam.metrics import (
    RepeatedMetric,
    aggregate_efficiency_by_model,
    aggregate_metric,
    build_latest_runs_index,
    group_runs,
    parse_model_filter,
    resolve_leaderboard_model_list,
    run_accuracy,
)
from vlm_exam.results import RunResult, SampleResult, save_results
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


def _sample() -> SampleResult:
    return SampleResult(
        index=0,
        image="a.jpg",
        expected="",
        predicted="",
        correct=True,
        input_tokens=100,
        output_tokens=50,
        elapsed_seconds=1.0,
    )


def _run(
    model: str,
    task: str,
    timestamp: str,
    effort: str = "low",
) -> RunResult:
    return RunResult(
        model=model,
        effort=effort,
        task=task,
        timestamp=timestamp,
        samples=[_sample()],
    )


class TestParseModelFilter:
    def test_rejects_unknown_model(self) -> None:
        config = _config("alpha")
        with pytest.raises(ValueError, match="Unknown model"):
            parse_model_filter("alpha,beta", config)

    def test_rejects_empty_string(self) -> None:
        config = _config("alpha")
        with pytest.raises(ValueError, match="at least one model"):
            parse_model_filter("", config)


class TestResolveLeaderboardModelList:
    def test_load_leaderboard_groups_preserves_order(self, tmp_path: Path) -> None:
        groups_path = tmp_path / "leaderboard_groups.yaml"
        groups_path.write_text(
            yaml.dump({"frontier": ["alpha", "beta"]}),
            encoding="utf-8",
        )
        groups = load_leaderboard_groups(groups_path)
        assert groups["frontier"] == ("alpha", "beta")

    def test_group_overrides_models(self) -> None:
        from vlm_exam.config import load_config

        config = load_config()
        assert resolve_leaderboard_model_list(
            config,
            models="claude-sonnet-5",
            group="alternative",
        ) == [
            "gemini-3.5-flash",
            "gpt-5.5",
            "gpt-5.6-sol",
            "gpt-5.6-terra",
            "gpt-5.6-luna",
            "claude-fable-5",
            "claude-fable-5-1",
        ]


class TestBuildLatestRunsIndex:
    def test_keeps_newest_run_per_task_effort_model(self) -> None:
        config = _config("alpha", "beta")
        runs = [
            _run("alpha", "counting", "20260707_000000"),
            _run("alpha", "counting", "20260707_120000"),
            _run("beta", "counting", "20260707_000000"),
        ]
        index = build_latest_runs_index(runs, config)
        assert index[("counting", "low", "alpha")].timestamp == "20260707_120000"
        assert ("counting", "low", "beta") in index

    def test_respects_model_filter(self) -> None:
        config = _config("alpha", "beta")
        runs = [
            _run("alpha", "counting", "20260707_000000"),
            _run("beta", "counting", "20260707_000000"),
        ]
        index = build_latest_runs_index(runs, config, models={"alpha"})
        assert list(index) == [("counting", "low", "alpha")]


class TestGroupRuns:
    def test_groups_repeats_oldest_first(self) -> None:
        config = _config("alpha", "beta")
        runs = [
            _run("alpha", "counting", "20260707_120000"),
            _run("alpha", "counting", "20260707_000000"),
            _run("alpha", "counting", "20260707_060000", effort="high"),
            _run("beta", "counting", "20260707_000000"),
            _run("gamma", "counting", "20260707_000000"),
        ]
        groups = group_runs(runs, config)
        assert set(groups) == {
            ("counting", "low", "alpha"),
            ("counting", "high", "alpha"),
            ("counting", "low", "beta"),
        }
        assert [run.timestamp for run in groups[("counting", "low", "alpha")]] == [
            "20260707_000000",
            "20260707_120000",
        ]

    def test_filters_by_model_effort_and_task(self) -> None:
        config = _config("alpha", "beta")
        runs = [
            _run("alpha", "counting", "20260707_000000"),
            _run("alpha", "ocr", "20260707_000000"),
            _run("alpha", "counting", "20260707_000000", effort="high"),
            _run("beta", "counting", "20260707_000000"),
        ]
        groups = group_runs(
            runs, config, models={"alpha"}, effort="low", tasks=("counting",)
        )
        assert list(groups) == [("counting", "low", "alpha")]


class TestRepeatedMetric:
    def test_mean_and_spread(self) -> None:
        metric = RepeatedMetric(values=(40.0, 50.0, 60.0))
        assert metric.mean == pytest.approx(50.0)
        assert metric.run_count == 3
        assert metric.minimum == 40.0
        assert metric.maximum == 60.0
        assert metric.spread == pytest.approx(20.0)

    def test_single_run_has_zero_spread(self) -> None:
        metric = RepeatedMetric(values=(72.5,))
        assert metric.mean == 72.5
        assert metric.spread == 0.0

    def test_aggregate_metric_skips_missing_values(self) -> None:
        runs = [
            _run("alpha", "counting", "20260707_000000"),
            _run("alpha", "counting", "20260707_010000"),
        ]
        assert aggregate_metric(runs, run_accuracy) == RepeatedMetric((100.0, 100.0))
        assert aggregate_metric(runs, lambda run: None) is None


class TestAggregateEfficiencyByModel:
    def test_totals_are_per_run_means(self, tmp_path: Path) -> None:
        config = _config("alpha", "beta")
        for stamp in ("20260707_000000", "20260707_010000", "20260707_020000"):
            save_results(
                _run("alpha", "counting", stamp),
                tmp_path / f"counting_alpha_low_{stamp}.jsonl",
            )
        save_results(
            _run("beta", "counting", "20260707_000000"),
            tmp_path / "counting_beta_low_20260707_000000.jsonl",
        )

        alpha, beta = aggregate_efficiency_by_model(tmp_path, config)

        assert alpha.sample_count == beta.sample_count == 1
        assert alpha.total_cost == pytest.approx(beta.total_cost)
        assert alpha.total_time_seconds == pytest.approx(beta.total_time_seconds)
        assert alpha.average_tokens == beta.average_tokens == 150

    def test_respects_model_filter(self, tmp_path: Path) -> None:
        config = _config("alpha", "beta")
        save_results(
            _run("alpha", "counting", "20260707_000000"),
            tmp_path / "counting_alpha_low_20260707_000000.jsonl",
        )
        save_results(
            _run("beta", "counting", "20260707_000000"),
            tmp_path / "counting_beta_low_20260707_000000.jsonl",
        )

        rows = aggregate_efficiency_by_model(
            tmp_path,
            config,
            models={"alpha"},
        )
        assert [row.model for row in rows] == ["alpha"]
