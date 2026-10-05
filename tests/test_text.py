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
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner
from PIL import Image

from vlm_exam import cli
from vlm_exam.providers.base import Provider, RetryStats, Usage
from vlm_exam.results import load_results, save_results
from vlm_exam.runner import run_benchmark
from vlm_exam.tasks.detection import DetectionCoordinateFormat
from vlm_exam.tasks.text import TextTask
from vlm_exam.tasks.text_scoring import field_score, parse_json, region_score
from vlm_exam.text_benchmark import summarize_text, validate_resume


def _dataset(path: Path) -> list[dict[str, str]]:
    path.mkdir(exist_ok=True)
    Image.new("RGB", (200, 100), "white").save(path / "shared.png")
    answers: dict[str, Any] = {
        "single_string": "001A",
        "transcription": "Hello\nWorld",
        "structured": {"a": "01", "b": None},
        "localization_recognition": [{"bbox": [20, 10, 100, 50], "text": "R1"}],
    }
    rows = [
        {
            "image": "shared.png",
            "prefix": json.dumps({"task": task, "question": task}),
            "suffix": answer if isinstance(answer, str) else json.dumps(answer),
        }
        for task, answer in answers.items()
    ]
    (path / "annotations.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    return rows


class _Provider(Provider):
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    @property
    def model(self) -> str:
        return "fake"

    def predict(
        self, image: Image.Image, prompt: str, effort: str
    ) -> tuple[str, Usage, RetryStats]:
        if self.fail and prompt == "single_string":
            raise RuntimeError("temporary failure")
        answers = {
            "single_string": "001A",
            "transcription": "Hello\nWorld",
            "structured": '{"a":"01","b":null}',
        }
        answer = answers.get(prompt, '[{"box_2d":[100,100,500,500],"text":"R1"}]')
        return answer, Usage(2, 3), RetryStats(1, 0.1)


def test_end_to_end_shared_image_and_repeat_summary(tmp_path: Path) -> None:
    _dataset(tmp_path)
    task = TextTask()
    samples = task.load_samples(str(tmp_path))
    assert len(samples) == 4
    assert len({s.identity for s in samples}) == 4
    assert len({s.image_path for s in samples}) == 1
    result = run_benchmark(
        task, _Provider(), samples, "low", "text", verbose=False, concurrency=2
    )
    assert all(s.metadata["score"] == 1 for s in result.samples)
    assert result.samples[0].metadata["resolved_prompt"] == "single_string"
    output = tmp_path / "results"
    for repeat in range(3):
        save_results(
            replace(result, timestamp=str(repeat)), output / f"run{repeat}.jsonl"
        )
    report = summarize_text(output)["configurations"][0]
    assert report["complete"] and report["mean"] == 1
    assert len(report["by_category"]) == 4
    assert report["input_tokens"] == 8
    assert (
        load_results(output / "run0.jsonl").samples[0].metadata["sample_id"]
        == samples[0].identity
    )


def test_failed_calls_are_unscored_and_partial_runs_not_complete(
    tmp_path: Path,
) -> None:
    _dataset(tmp_path)
    task = TextTask()
    samples = task.load_samples(str(tmp_path))
    result = run_benchmark(
        task, _Provider(fail=True), samples, "low", "text", verbose=False
    )
    output = tmp_path / "results"
    save_results(result, output / "run.jsonl")
    report = summarize_text(output)["configurations"][0]
    assert report["mean"] is None and report["coverage"][0]["failed"] == 1
    assert report["coverage"][0]["observed_mean"] == 1
    assert report["coverage"][0]["bounds"] == [0.75, 1.0]


@pytest.mark.parametrize("profile", list(DetectionCoordinateFormat))
def test_all_model_coordinate_profiles(
    tmp_path: Path, profile: DetectionCoordinateFormat
) -> None:
    _dataset(tmp_path)
    task = TextTask(profile)
    sample = task.load_samples(str(tmp_path))[-1]
    sample = replace(sample, answer=[{"bbox": [20, 20, 100, 70], "text": "R1"}])
    name = profile.value
    upload = (100, 50)
    if "normalized" in name:
        scale = 100 if name == "xyxy_normalized_0_to_100" else 1000
        box = [scale * 0.1, scale * 0.2, scale * 0.5, scale * 0.7]
    elif "resized" in name:
        box = [10, 10, 50, 35]
    else:
        box = [20, 20, 100, 70]
    if name.startswith("yxyx"):
        box = [box[1], box[0], box[3], box[2]]
    if "meta_flat" in name or "meta_bbox" in name:
        coords = dict(zip(["x_min", "y_min", "x_max", "y_max"], box))
        prediction = (
            {**coords, "text": "R1"}
            if "meta_flat" in name
            else {"bbox": [coords], "text": "R1"}
        )
    else:
        prediction = {"bbox" if name.endswith("_bbox") else "box_2d": box, "text": "R1"}
    assert (
        task.evaluate(sample, json.dumps([prediction]), uploaded_size=upload).score == 1
    )
    prompt = task.build_prompt(sample, uploaded_size=upload)
    assert "text" in prompt and "localization_recognition" in prompt
    assert "sample_id" not in prompt


def test_matching_is_one_to_one_and_invalid_regions_penalized() -> None:
    gt = [{"bbox": [0, 0, 10, 10], "text": "A"}]
    assert region_score(gt, gt + gt)["score"] == pytest.approx(2 / 3)
    assert region_score(gt, gt + [None])["score"] == pytest.approx(2 / 3)
    assert region_score([], [])["score"] == 1
    assert region_score(gt, [{"bbox": [0, 0, 10, 10], "text": "a"}])["score"] == 0


def test_matching_finds_augmenting_path() -> None:
    gt = [{"bbox": [0, 0, 10, 10], "text": "A"}, {"bbox": [3, 0, 13, 10], "text": "A"}]
    pred = [
        {"bbox": [1, 0, 11, 10], "text": "A"},
        {"bbox": [-1, 0, 9, 10], "text": "A"},
    ]
    assert region_score(gt, pred)["score"] == 1


def test_structured_null_missing_extra_and_types() -> None:
    assert field_score({"a": None}, {"a": None})["score"] == 1
    assert field_score({"a": None, "b": "1"}, {"b": "1"})["score"] == pytest.approx(
        2 / 3
    )
    assert field_score({"a": "01"}, {"a": 1})["score"] == 0
    assert field_score({"a": True}, {"a": 1})["score"] == 0
    assert field_score({"a": "x"}, {"a": "x", "b": None})["score"] == pytest.approx(
        2 / 3
    )


@pytest.mark.parametrize(
    "prediction", ['{"a":1,"a":2}', '{"a":NaN}', "prefix {}", "{} suffix"]
)
def test_parser_rejects_repairs(prediction: str) -> None:
    with pytest.raises(ValueError):
        parse_json(prediction)


def test_invalid_json_is_scored_zero_and_fence_flagged(tmp_path: Path) -> None:
    _dataset(tmp_path)
    task = TextTask()
    sample = task.load_samples(str(tmp_path))[2]
    assert task.evaluate(sample, "not JSON").score == 0
    result = task.evaluate(sample, '```json\n{"a":"01","b":null}\n```')
    assert result.score == 1 and not result.details["format_compliant"]


def test_validation_rejects_traversal_duplicate_and_unknown_task(
    tmp_path: Path,
) -> None:
    rows = _dataset(tmp_path)
    for bad in [
        {**rows[0], "image": "../shared.png"},
        {**rows[0], "prefix": '{"task":"unknown","question":"x"}'},
    ]:
        (tmp_path / "annotations.jsonl").write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            TextTask().load_samples(str(tmp_path))
    (tmp_path / "annotations.jsonl").write_text(
        "\n".join(json.dumps(rows[0]) for _ in range(2))
    )
    with pytest.raises(ValueError, match="Duplicate"):
        TextTask().load_samples(str(tmp_path))


def test_resume_rejects_changed_ground_truth(tmp_path: Path) -> None:
    from vlm_exam.config import load_config

    rows = _dataset(tmp_path)
    task = TextTask()
    samples = task.load_samples(str(tmp_path))
    run = run_benchmark(task, _Provider(), samples, "low", "text", verbose=False)
    model = replace(
        next(iter(load_config().models.values())),
        detection_coordinate_format=task.coordinate_format,
    )
    validate_resume(run, samples, model)
    rows[0]["suffix"] = "changed"
    (tmp_path / "annotations.jsonl").write_text("\n".join(map(json.dumps, rows)))
    with pytest.raises(ValueError, match="changed"):
        validate_resume(run, task.load_samples(str(tmp_path)), model)


def test_cli_validate_and_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.test_cli_run import _config

    _dataset(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        cli.main,
        [
            "text-validate",
            "--dataset-directory",
            str(tmp_path),
            "--expected-pairs",
            "4",
        ],
    )
    assert result.exit_code == 0, result.output
    monkeypatch.setattr(cli, "load_config", lambda _: _config())
    monkeypatch.setattr(cli, "build_model_provider", lambda *args: _Provider())
    result = runner.invoke(
        cli.main,
        [
            "run",
            "--task",
            "text",
            "--models",
            "alpha",
            "--effort",
            "low",
            "--dataset-directory",
            str(tmp_path),
            "--output-directory",
            str(tmp_path / "results"),
        ],
    )
    assert result.exit_code == 0, repr(result.exception)
    run = load_results(next((tmp_path / "results").glob("*.jsonl")))
    assert len(run.samples) == 4
    assert run.samples[0].metadata["resolved_prompt"] == "single_string"


def test_cli_resume_only_failed_question(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.test_cli_run import _config
    from vlm_exam.tasks.text import TextTask

    _dataset(tmp_path)
    config = _config()
    task = TextTask(config.models["alpha"].detection_coordinate_format)
    samples = task.load_samples(str(tmp_path))
    previous = run_benchmark(
        task, _Provider(fail=True), samples, "low", "text", verbose=False
    )
    previous = replace(previous, model="alpha")
    result_path = tmp_path / "old.jsonl"
    save_results(previous, result_path)
    seen: list[str] = []

    class RecordingProvider(_Provider):
        def predict(
            self, image: Image.Image, prompt: str, effort: str
        ) -> tuple[str, Usage, RetryStats]:
            seen.append(prompt)
            return super().predict(image, prompt, effort)

    monkeypatch.setattr(cli, "load_config", lambda _: config)
    monkeypatch.setattr(cli, "build_model_provider", lambda *args: RecordingProvider())
    monkeypatch.setattr(RecordingProvider, "model", property(lambda self: "alpha"))
    response = CliRunner().invoke(
        cli.main,
        [
            "run",
            "--task",
            "text",
            "--models",
            "alpha",
            "--effort",
            "low",
            "--dataset-directory",
            str(tmp_path),
            "--output-directory",
            str(tmp_path / "resumed"),
            "--resume-file",
            str(result_path),
        ],
    )
    assert response.exit_code == 0, repr(response.exception)
    assert seen == ["single_string"]
    result = load_results(next((tmp_path / "resumed").glob("*.jsonl")))
    assert len(result.samples) == 4
    assert not result_path.exists()


def test_cli_provider_failure_is_nonzero_after_saving(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.test_cli_run import _config

    _dataset(tmp_path)
    monkeypatch.setattr(cli, "load_config", lambda _: _config())
    monkeypatch.setattr(cli, "build_model_provider", lambda *args: _Provider(fail=True))
    result = CliRunner().invoke(
        cli.main,
        [
            "run",
            "--task",
            "text",
            "--models",
            "alpha",
            "--effort",
            "low",
            "--dataset-directory",
            str(tmp_path),
            "--output-directory",
            str(tmp_path / "results"),
        ],
    )
    assert result.exit_code == 1
    assert len(list((tmp_path / "results").glob("*.jsonl"))) == 1


def test_text_protocol_plans_six_runs_without_changing_legacy() -> None:
    from vlm_exam.orchestrate import plan_jobs
    from vlm_exam.protocol import PROTOCOL
    from vlm_exam.text_benchmark import TEXT_BENCHMARK_PROTOCOL

    jobs = plan_jobs(["fake"], protocol=TEXT_BENCHMARK_PROTOCOL)
    assert len(jobs) == 6
    assert {job.effort for job in jobs} == {"low", "high"}
    assert "text" not in PROTOCOL.tasks
