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

import click
import pytest
import yaml
from click.testing import CliRunner

from vlm_exam import (
    cli,
    detection_v2,
    detection_v2_data,
    detection_v2_runner,
    orchestrate,
)
from vlm_exam import detection_v2_workflow as workflow
from vlm_exam.config import load_config


def registry(tmp_path: Path, *, invalid_high: bool = False) -> Path:
    models = {}
    for key in ("alpha", "beta"):
        models[key] = {
            "name": key,
            "lab": "openai",
            "provider": "openai",
            "provider_model_id": key,
            "pricing": {"input_per_million_tokens": 1, "output_per_million_tokens": 2},
            "detection_coordinate_format": "xyxy_absolute_original_image",
            "detection_v2": {
                "provider": "openai",
                "wire": key,
                "format": "xyxy_absolute_original_image",
                "minimum": "low",
                "maximum": "high",
                "max_output_tokens": 128000,
            },
        }
    if invalid_high:
        models["beta"]["detection_v2"]["arms"] = ["minimum"]
    path = tmp_path / "models.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "labs": {
                    "openai": {"name": "OpenAI", "color": "#ffffff", "logo_url": ""}
                },
                "models": models,
            }
        )
    )
    return path


@pytest.fixture
def empty_release(monkeypatch: pytest.MonkeyPatch) -> dict:
    policy = {"runs": [], "configurations": {}, "efforts": ["minimum", "maximum"]}
    monkeypatch.setattr(detection_v2, "load_release", lambda root: (policy, []))
    return policy


def test_run_uses_each_models_registry_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, empty_release: dict
) -> None:
    path = registry(tmp_path)
    calls = []
    monkeypatch.setattr(detection_v2_data, "ensure_images", lambda root, images: images)
    monkeypatch.setattr(
        detection_v2_runner, "run_release", lambda *a, **kw: calls.append((a, kw))
    )
    result = CliRunner().invoke(
        cli.main,
        [
            "run",
            "--task",
            "detection",
            "--models",
            "alpha,beta",
            "--effort",
            "low",
            "--config",
            str(path),
            "--max-samples",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    assert [(a[2], kw["profile"]["wire"]) for a, kw in calls] == [
        ("alpha", "alpha"),
        ("beta", "beta"),
    ]
    assert all(a[3] == "minimum" for a, _ in calls)


def test_benchmark_forwards_registry_to_all_jobs_and_defaults_v2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, empty_release: dict
) -> None:
    path = registry(tmp_path)
    scheduled = []
    monkeypatch.setattr(
        orchestrate, "run_jobs", lambda jobs, **kw: scheduled.extend(jobs) or []
    )
    result = CliRunner().invoke(
        cli.main,
        [
            "benchmark",
            "--dataset-root",
            str(tmp_path),
            "--models",
            "alpha,beta",
            "--tasks",
            "counting,detection",
            "--config",
            str(path),
            "--max-samples",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    detection = [job for job in scheduled if job.task == "detection"]
    assert len(detection) == 4
    for job in scheduled:
        assert job.command[job.command.index("--config") + 1] == str(path)
    assert all(
        job.command[job.command.index("--detection-version") + 1] == "v2"
        for job in detection
    )


def test_benchmark_preflights_all_arms_before_any_task_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, empty_release: dict
) -> None:
    path = registry(tmp_path, invalid_high=True)
    scheduled = []
    monkeypatch.setattr(
        orchestrate, "run_jobs", lambda jobs, **kw: scheduled.extend(jobs) or []
    )
    result = CliRunner().invoke(
        cli.main,
        [
            "benchmark",
            "--dataset-root",
            str(tmp_path),
            "--models",
            "alpha,beta",
            "--tasks",
            "counting,detection",
            "--config",
            str(path),
        ],
    )
    assert result.exit_code != 0
    assert "requested arm" in result.output
    assert not scheduled


def test_explicit_single_profile_rejects_multiple_models(
    tmp_path: Path, empty_release: dict
) -> None:
    config = load_config(registry(tmp_path))
    profile = config.models["alpha"].detection_v2
    with pytest.raises(click.UsageError, match="only.*one model"):
        workflow.resolve_detection_profiles(
            "alpha,beta", "minimum", empty_release, config, settings=profile
        )
    resolved = workflow.resolve_detection_profiles(
        "alpha", "minimum", empty_release, config, settings=profile
    )
    assert resolved == [("alpha", None, profile)]


def test_v1_benchmark_remains_explicit_and_ignores_v2_profiles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = registry(tmp_path, invalid_high=True)
    scheduled = []
    monkeypatch.setattr(
        orchestrate, "run_jobs", lambda jobs, **kw: scheduled.extend(jobs) or []
    )
    result = CliRunner().invoke(
        cli.main,
        [
            "benchmark",
            "--dataset-root",
            str(tmp_path),
            "--models",
            "beta",
            "--tasks",
            "detection",
            "--config",
            str(path),
            "--detection-version",
            "v1",
        ],
    )
    assert result.exit_code == 0, result.output
    assert len(scheduled) == 6
    assert all(
        job.command[job.command.index("--detection-version") + 1] == "v1"
        for job in scheduled
    )


def test_unpinned_openrouter_registry_fails_preflight(
    tmp_path: Path, empty_release: dict
) -> None:
    path = registry(tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw["models"]["alpha"]["detection_v2"].update(
        provider="openrouter", wire="qwen/new-model"
    )
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="routing.only"):
        workflow.resolve_detection_profiles(
            "alpha", "minimum", empty_release, load_config(path)
        )
