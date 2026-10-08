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
    summary_drift,
    summary_to_dict,
)
from vlm_exam.tasks.detection import DetectionCoordinateFormat


def _model(model_id: str, benchmark_protocol: str = "full") -> ModelConfig:
    return ModelConfig(
        name=model_id,
        lab="openai",
        routes=(RouteConfig("openai"),),
        pricing=PricingConfig(1.0, 2.0),
        detection_coordinate_format=DetectionCoordinateFormat.XYXY_ABSOLUTE_ORIGINAL_IMAGE,
        benchmark_protocol=benchmark_protocol,
    )


def _config(*model_ids: str, legacy: tuple[str, ...] = ()) -> BenchmarkConfig:
    return BenchmarkConfig(
        labs={"openai": LabConfig("OpenAI", "#000", "https://example.com/logo.svg")},
        models={
            model_id: _model(model_id, "legacy" if model_id in legacy else "full")
            for model_id in model_ids
        },
    )


def _sample(
    index: int = 0,
    correct: bool = True,
    metadata: dict[str, Any] | None = None,
    strict_correct: bool | None = None,
) -> SampleResult:
    if metadata is None:
        metadata = {
            "strict_correct": correct if strict_correct is None else strict_correct,
            "judge_correct": correct,
        }
    return SampleResult(
        index=index,
        image=f"{index}.jpg",
        expected="",
        predicted="",
        correct=correct,
        input_tokens=100,
        output_tokens=50,
        elapsed_seconds=1.0,
        metadata=metadata,
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
    def test_protocol_status_counts_runs_across_efforts(self, tmp_path: Path) -> None:
        config = _config("alpha", "old", legacy=("old",))
        for task in BENCHMARK_TASK_NAMES:
            for effort in ("low", "high"):
                for repeat in range(3):
                    _save(
                        _run(
                            "alpha",
                            task,
                            effort=effort,
                            timestamp=f"2026070{repeat + 1}_000000",
                        ),
                        tmp_path,
                    )
        _save(_run("old", "counting"), tmp_path)

        summary = build_summary(tmp_path, config, effort="low")

        by_key = {model.key: model for model in summary.models}
        assert by_key["alpha"].protocol.status == "complete"
        assert by_key["alpha"].protocol.runs_present == 36
        assert by_key["old"].protocol.status == "legacy"
        assert by_key["old"].protocol.runs_present == 1
        assert by_key["old"].protocol.name == "legacy"

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

    def test_repeats_are_averaged(self, tmp_path: Path) -> None:
        config = _config("alpha")
        first = _run(
            "alpha",
            "counting",
            timestamp="20260701_000000",
            samples=[_sample(index=0, correct=False), _sample(index=1, correct=True)],
        )
        second = _run(
            "alpha",
            "counting",
            timestamp="20260702_000000",
            samples=[_sample(index=0, correct=True), _sample(index=1, correct=True)],
        )
        third = _run(
            "alpha",
            "counting",
            timestamp="20260703_000000",
            samples=[_sample(index=0, correct=True), _sample(index=1, correct=True)],
        )
        _save(third, tmp_path)
        _save(first, tmp_path)
        _save(second, tmp_path)

        summary = build_summary(tmp_path, config)

        counting = summary.models[0].tasks["counting"]
        assert counting.run_count == 3
        assert counting.metrics["accuracy_judge"] == pytest.approx(250 / 3)
        assert counting.metric_runs["accuracy_judge"] == (50.0, 100.0, 100.0)
        assert counting.timestamps == (
            "20260701_000000",
            "20260702_000000",
            "20260703_000000",
        )
        assert counting.timestamp == "20260703_000000"
        assert counting.sample_count == 2
        assert counting.tokens.total == 300
        assert counting.cost.total_usd == pytest.approx(2 * (100 * 1 + 50 * 2) / 1e6)
        assert summary.models[0].overall.sample_count == 2
        assert summary.generated_at == "2026-07-03T00:00:00Z"

    def test_failed_samples_emit_warning(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        config = _config("alpha")
        failed = SampleResult(
            index=0,
            image="a.jpg",
            expected="4",
            predicted="ERROR: boom",
            correct=False,
            input_tokens=0,
            output_tokens=0,
            elapsed_seconds=None,
            metadata={"strict_correct": False, "judge_correct": False},
        )
        _save(_run("alpha", "counting", samples=[failed]), tmp_path)

        summary = build_summary(tmp_path, config)

        assert summary.models[0].tasks["counting"].failed_sample_count == 1
        assert "--resume-file" in capsys.readouterr().out

    def test_qa_reports_judge_and_strict_accuracy(self, tmp_path: Path) -> None:
        config = _config("alpha")
        samples = [
            _sample(index=0, correct=True),
            _sample(index=1, correct=True, strict_correct=False),
            _sample(index=2, correct=False),
        ]
        _save(_run("alpha", "reasoning", samples=samples), tmp_path)

        summary = build_summary(tmp_path, config)

        result = summary.models[0].tasks["reasoning"]
        assert result.metrics == {
            "accuracy_judge": pytest.approx(200 / 3),
            "accuracy_strict": pytest.approx(100 / 3),
        }
        assert result.primary_metric is not None
        assert result.primary_metric.name == "accuracy_judge"
        assert result.primary_metric.value == pytest.approx(200 / 3)
        assert result.evaluated_sample_count is None

    def test_qa_run_without_verdicts_is_rejected(self, tmp_path: Path) -> None:
        config = _config("alpha")
        legacy = _sample(index=0, correct=True, metadata={"match_method": "strict"})
        _save(_run("alpha", "reasoning", samples=[legacy]), tmp_path)

        with pytest.raises(ValueError, match="vlm-exam rescore"):
            build_summary(tmp_path, config)

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

        assert list(payload) == [
            "generated_at",
            "efforts",
            "scoring",
            "protocol",
            "tasks",
            "models",
        ]
        assert payload["generated_at"] == "2026-07-10T07:33:33Z"
        assert payload["efforts"] == ["low"]
        assert payload["scoring"] == {
            "judge_model": "gemini-3.5-flash",
            "judge_metric": "accuracy_judge",
            "strict_metric": "accuracy_strict",
        }
        assert payload["protocol"] == {
            "repeats": 3,
            "efforts": ["low", "high"],
            "tasks": [
                "ocr",
                "extraction",
                "counting",
                "identification",
                "reasoning",
                "detection",
            ],
            "runs_per_model": 36,
        }

        (task,) = payload["tasks"]
        assert list(task) == ["key", "name", "primary_metric", "metrics"]
        assert task["primary_metric"] == "accuracy_judge"
        assert task["metrics"] == [
            {
                "key": "accuracy_judge",
                "label": "Accuracy (LLM judge)",
                "unit": "percent",
            },
            {
                "key": "accuracy_strict",
                "label": "Accuracy (strict match)",
                "unit": "percent",
            },
        ]

        (model,) = payload["models"]
        assert model["id"] == "alpha:low"
        assert model["key"] == "alpha"
        assert model["effort"] == "low"
        assert "pricing" not in model

        counting = model["tasks"]["counting"]
        assert counting["metrics"] == {
            "accuracy_judge": 66.67,
            "accuracy_strict": 66.67,
        }
        assert counting["primary_metric"] == {"name": "accuracy_judge", "value": 66.67}
        assert counting["metric_runs"] == {
            "accuracy_judge": [66.67],
            "accuracy_strict": [66.67],
        }
        assert counting["run_count"] == 1
        assert counting["timestamps"] == ["2026-07-10T07:33:33Z"]
        assert counting["evaluated_sample_count"] is None
        assert counting["timestamp"] == "2026-07-10T07:33:33Z"
        assert model["overall"]["sample_count"] == 3
        assert model["protocol"] == {
            "name": "full",
            "status": "incomplete",
            "runs_present": 1,
            "runs_required": 36,
        }


class TestSummaryDrift:
    def _payload(self, map50: float, cost: float) -> dict[str, Any]:
        return {
            "models": [
                {
                    "key": "alpha",
                    "tasks": {
                        "detection": {
                            "primary_metric": {"name": "map50", "value": map50},
                            "metrics": {"map50": map50},
                            "metric_runs": {"map50": [map50]},
                            "evaluated_sample_count": 250,
                            "cost": {"total_usd": cost},
                        },
                        "ocr": {"metrics": {"similarity": 90.0}},
                    },
                }
            ]
        }

    def test_identical_payloads_have_no_drift(self) -> None:
        assert (
            summary_drift(
                self._payload(60.0, 1.0),
                self._payload(60.0, 1.0),
                ignore_detection_quality=False,
            )
            == []
        )

    def test_detection_quality_is_ignored_only_when_requested(self) -> None:
        committed = self._payload(60.0, 1.0)
        fresh = self._payload(0.0, 1.0)

        assert summary_drift(committed, fresh, ignore_detection_quality=True) == []
        assert summary_drift(committed, fresh, ignore_detection_quality=False)

    def test_other_changes_still_drift(self) -> None:
        drift = summary_drift(
            self._payload(60.0, 1.0),
            self._payload(60.0, 2.0),
            ignore_detection_quality=True,
        )

        assert any(line.startswith("+") and "2.0" in line for line in drift)


@pytest.fixture
def text_release(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    import hashlib
    import json

    from vlm_exam import text_release as release

    directory = tmp_path / "results"
    directory.mkdir()
    categories = (
        "single_string",
        "transcription",
        "structured",
        "localization_recognition",
    )
    samples = [
        _sample(
            index=index,
            metadata={
                "sample_id": str(index),
                "dataset_hash": "snapshot",
                "dataset_pairs": 4,
                "text_protocol": "mixed-text-v2",
                "coordinate_format": "xyxy_normalized_0_to_1000",
                "inference_hash": "frozen",
                "image_sha256": str(index),
                "question": "Read",
                "category": category,
                "subset": category,
                "score": 0.75 if index != 2 else None,
            },
        )
        for index, category in enumerate(categories)
    ]
    samples[2].predicted = "ERROR: accepted timeout"
    samples[2].correct = False
    samples[2].elapsed_seconds = None
    run = _run(
        "alpha", "text", timestamp="2026-10-08T06:38:54.178352+00:00", samples=samples
    )
    save_results(run, directory / "text_alpha_low.jsonl")
    policy = {
        "name": "test-release",
        "status": "preliminary",
        "models": ["alpha"],
        "efforts": ["low"],
        "repeats": 1,
        "pairs": 4,
        "dataset_hash": "snapshot",
        "protocol": "mixed-text-v2",
        "selection_hash": hashlib.sha256(
            json.dumps({"sample_ids": ("0", "1", "2", "3"), "pairs": 4}).encode()
        ).hexdigest(),
        "configurations": {
            "alpha/low": {
                "inference_hash": "frozen",
                "coordinate_format": "xyxy_normalized_0_to_1000",
                "allowed_failures": ["2"],
            }
        },
    }
    monkeypatch.setattr(release, "load_text_release_policy", lambda: policy)
    return directory


def test_text_extends_existing_web_contract_without_changing_overview(
    tmp_path: Path, text_release: Path
) -> None:
    import hashlib

    legacy = tmp_path / "results"
    legacy.mkdir(exist_ok=True)
    save_results(_run("alpha", "counting"), legacy / "counting.jsonl")
    baseline = summary_to_dict(build_summary(legacy, _config("alpha"), effort="high"))
    assert baseline["models"] == []
    payload = summary_to_dict(build_summary(legacy, _config("alpha")))
    model = payload["models"][0]
    assert len(payload["models"]) == 1
    assert model["id"] == "alpha:low"
    assert payload["overview_tasks"] == ["counting"]
    assert model["overall"]["sample_count"] == 1
    assert model["overall"]["tokens"]["total"] == 150
    assert model["tasks"]["counting"]["primary_metric"]["value"] == 100
    assert set(model["tasks"]) == {"counting", "text"}
    text = model["tasks"]["text"]
    assert text["primary_metric"] is None
    assert text["metrics"] == {
        "single_string": 75,
        "transcription": 75,
        "localization_recognition": 75,
    }
    assert text["metric_runs"] == {
        key: [value] for key, value in text["metrics"].items()
    }
    assert text["run_count"] == 1
    assert "protocol" not in text
    assert text["timestamp"] == "2026-10-08T06:38:54Z"
    assert payload["generated_at"] == text["timestamp"]
    assert text["tokens"]["total"] == 600
    assert text["cost"]["total_usd"] == 0.0008
    assert text["speed"]["total_seconds"] == 3
    assert (
        text["provenance"]["result_files"][0]["sha256"]
        == hashlib.sha256(
            (text_release / "text_alpha_low.jsonl").read_bytes()
        ).hexdigest()
    )
    gap = text["coverage"][0]["by_metric"]["structured"]
    assert gap["score"] is None
    assert gap["evaluated_sample_count"] == 0
    assert gap["failed_sample_count"] == 1
    assert gap["score_bounds"] == [0, 100]
    assert text["sample_count"] == 4
    assert text["evaluated_sample_count"] == 3
    assert text["failed_sample_count"] == 1
    assert text["coverage"][0]["score_bounds"] == [56.25, 81.25]
    definition = next(t for t in payload["tasks"] if t["key"] == "text")
    assert definition["include_in_overall"] is False
    assert "protocol" not in definition
    assert definition["primary_metric"] == "overall"
    assert [metric["key"] for metric in definition["metrics"]] == [
        "overall",
        "single_string",
        "transcription",
        "structured",
        "localization_recognition",
    ]
    assert len(payload["tasks"]) == 2


def test_text_overall_weights_pairs_instead_of_categories(
    tmp_path: Path, text_release: Path
) -> None:
    import hashlib
    import json
    from copy import deepcopy

    from vlm_exam.results import load_results
    from vlm_exam.text_release import load_text_release_policy

    path = text_release / "text_alpha_low.jsonl"
    run = load_results(path)
    run.samples[2].predicted = "valid response"
    run.samples[2].metadata["score"] = 0.75
    run.samples[2].elapsed_seconds = 1.0
    extra = deepcopy(run.samples[0])
    extra.index = 4
    extra.metadata.update(sample_id="4", score=0.0)
    run.samples.append(extra)
    for sample in run.samples:
        sample.metadata["dataset_pairs"] = 5
    policy = load_text_release_policy()
    policy["pairs"] = 5
    policy["selection_hash"] = hashlib.sha256(
        json.dumps({"sample_ids": ("0", "1", "2", "3", "4"), "pairs": 5}).encode()
    ).hexdigest()
    policy["configurations"]["alpha/low"]["allowed_failures"] = []
    save_results(run, path)
    legacy = tmp_path / "results"
    legacy.mkdir(exist_ok=True)
    payload = summary_to_dict(build_summary(legacy, _config("alpha")))
    tasks = payload["models"][0]["tasks"]
    total = tasks["text"]
    assert total["primary_metric"] == {"name": "overall", "value": 60.0}
    assert total["metrics"] == {
        "overall": 60.0,
        "single_string": 37.5,
        "transcription": 75.0,
        "structured": 75.0,
        "localization_recognition": 75.0,
    }
    assert total["metric_runs"] == {
        key: [value] for key, value in total["metrics"].items()
    }
    assert total["sample_count"] == total["evaluated_sample_count"] == 5
    assert total["failed_sample_count"] == 0
    assert total["tokens"]["total"] == 750
    assert total["cost"]["total_usd"] == pytest.approx(0.001)
    assert total["speed"]["total_seconds"] == 5
    assert "protocol" not in total
    category_scores = [
        value for key, value in total["metrics"].items() if key != "overall"
    ]
    assert sum(category_scores) / len(category_scores) == 65.625
    assert "text" not in payload["overview_tasks"]


def test_unified_summary_checks_text_inventory_and_result_bytes(
    tmp_path: Path, text_release: Path
) -> None:
    legacy = tmp_path / "results"
    legacy.mkdir(exist_ok=True)
    config = _config("alpha")
    before = summary_to_dict(build_summary(legacy, config))
    path = text_release / "text_alpha_low.jsonl"
    path.write_text(
        path.read_text().replace('"predicted": ""', '"predicted": "different"')
    )
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError):
        build_summary(legacy, config)
    path.write_text(path.read_text().rstrip() + "\n")
    path.write_text(
        path.read_text().replace('"predicted":""', '"predicted":"different"')
    )
    after = summary_to_dict(build_summary(legacy, config))
    assert summary_drift(before, after, ignore_detection_quality=True)
    path.unlink()
    with pytest.raises(ValueError, match="Expected 1 runs, found 0"):
        build_summary(legacy, config)


def test_standard_summary_command_checks_text_and_has_no_sidecar_exports(
    tmp_path: Path, text_release: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    from click.testing import CliRunner

    from vlm_exam import cli
    from vlm_exam import text_release as release

    directory = tmp_path / "results"
    directory.mkdir(exist_ok=True)
    output = tmp_path / "web" / "benchmark_summary.json"
    monkeypatch.setattr(cli, "load_config", lambda path: _config("alpha"))
    checked = []
    monkeypatch.setattr(
        release, "check_text_charts", lambda *args: checked.append(args[0])
    )
    arguments = [
        "summary",
        "--results-directory",
        str(directory),
        "--output-file",
        str(output),
    ]
    runner = CliRunner()
    result = runner.invoke(cli.main, arguments)
    assert result.exit_code == 0, result.output
    assert list(output.parent.iterdir()) == [output]
    assert "text-summary" not in cli.main.commands
    assert "text-publish" not in cli.main.commands
    result = runner.invoke(cli.main, arguments + ["--check"])
    assert result.exit_code == 0, result.output
    assert checked == [text_release]
    payload = json.loads(output.read_text())
    payload["models"][0]["tasks"]["text"]["metrics"]["single_string"] = 99
    output.write_text(json.dumps(payload))
    result = runner.invoke(cli.main, arguments + ["--check"])
    assert result.exit_code == 1
    assert "out of date" in result.output


def test_shared_validation_selects_text_without_ignoring_missing_inventory(
    text_release: Path,
) -> None:
    from vlm_exam.text_release import (
        load_text_release_policy,
        text_result_paths,
        validate_text_release,
    )
    from vlm_exam.validation import validate_results

    save_results(_run("alpha", "counting"), text_release / "counting.jsonl")
    assert len(text_result_paths(text_release)) == 1
    assert validate_text_release(text_release, load_text_release_policy()) == []
    report = validate_results(text_release, _config("alpha"))
    assert not [problem for problem in report.orphans if problem.task == "text"]
    path = text_release / "text_alpha_low.jsonl"
    renamed = text_release / "renamed.jsonl"
    path.rename(renamed)
    assert text_result_paths(text_release) == [renamed]
    renamed.unlink()
    report = validate_results(text_release, _config("alpha"))
    assert any(
        problem.task == "text" and "Expected 1 runs, found 0" in problem.message
        for problem in report.errors
    )
    path.write_text("not JSON\n")
    assert any(
        problem.kind == "text_release"
        for problem in validate_text_release(text_release, load_text_release_policy())
    )


def test_shared_leaderboard_dispatches_text_to_existing_renderer(
    text_release: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import click
    from click.testing import CliRunner

    from vlm_exam import cli

    received = []

    @click.command()
    def renderer(**kwargs: object) -> None:
        received.append(kwargs)

    monkeypatch.setattr(cli, "load_config", lambda path: _config("alpha"))
    monkeypatch.setitem(cli.main.commands, "text-leaderboard", renderer)
    result = CliRunner().invoke(
        cli.main,
        ["leaderboard", "--results-directory", str(text_release), "--models", "alpha"],
    )
    assert result.exit_code == 0, result.output
    assert received[0]["results_directory"] == str(text_release)
    assert received[0]["models"] == "alpha"
    assert "No leaderboard renderer for task 'text'" not in result.output
    save_results(_run("beta", "counting"), text_release / "counting_beta.jsonl")
    monkeypatch.setattr(cli, "load_config", lambda path: _config("alpha", "beta"))
    monkeypatch.setattr(cli, "_resolve_model_filter", lambda *args: {"beta"})
    monkeypatch.setattr("vlm_exam.metrics.group_runs", lambda *args, **kwargs: {})
    result = CliRunner().invoke(
        cli.main,
        ["leaderboard", "--results-directory", str(text_release), "--models", "beta"],
    )
    assert result.exit_code == 0, result.output
    assert len(received) == 1
