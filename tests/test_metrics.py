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
    aggregate_efficiency_by_model,
    build_latest_runs_index,
    parse_model_filter,
    resolve_leaderboard_model_list,
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


class TestAggregateEfficiencyByModel:
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
