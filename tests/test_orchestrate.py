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

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from vlm_exam.orchestrate import (
    Job,
    JobOutcome,
    _launch,
    format_outcomes,
    plan_jobs,
    run_jobs,
)
from vlm_exam.protocol import PROTOCOL, TASK_CONCURRENCY, run_command


class TestRunCommand:
    def test_builds_run_command_with_task_concurrency(self) -> None:
        command = run_command("detection", "alpha", "high", repeats=3)

        assert command[:2] == ["vlm-exam", "run"]
        assert "--dataset-directory" in command
        assert command[command.index("--dataset-directory") + 1] == str(
            Path("data/detection/train")
        )
        assert command[command.index("--concurrency") + 1] == str(
            TASK_CONCURRENCY["detection"]
        )
        assert command[-2:] == ["--repeats", "3"]

    def test_single_repeat_omits_repeats_flag(self) -> None:
        assert "--repeats" not in run_command("ocr", "alpha", "low")


class TestPlanJobs:
    def test_full_protocol_for_one_model(self) -> None:
        jobs = plan_jobs(["alpha"])

        assert len(jobs) == PROTOCOL.required_runs
        assert [job.effort for job in jobs[:18]] == ["low"] * 18
        assert [job.effort for job in jobs[18:]] == ["high"] * 18
        assert jobs[0].log_path == Path("logs/ocr_alpha_low_r1.log")
        assert jobs[2].repeat == 3
        assert "--max-samples" not in jobs[0].command

    def test_filters_and_first_repeat(self, tmp_path: Path) -> None:
        jobs = plan_jobs(
            ["alpha", "beta"],
            tasks=("counting",),
            efforts=("high",),
            repeats=2,
            first_repeat=2,
            dataset_root=tmp_path,
            output_directory=tmp_path / "out",
            log_directory=tmp_path / "logs",
            max_samples=5,
        )

        assert [(j.model, j.repeat) for j in jobs] == [
            ("alpha", 2),
            ("alpha", 3),
            ("beta", 2),
            ("beta", 3),
        ]
        command = jobs[0].command
        assert command[command.index("--dataset-directory") + 1] == str(
            tmp_path / "counting" / "train"
        )
        assert command[command.index("--output-directory") + 1] == str(tmp_path / "out")
        assert command[-2:] == ("--max-samples", "5")
        assert jobs[0].log_path == tmp_path / "logs" / "counting_alpha_high_r2.log"


class _FakeProcess:
    def __init__(self, return_code: int) -> None:
        self.returncode = return_code

    def poll(self) -> int:
        return self.returncode


def _job(task: str, repeat: int, tmp_path: Path) -> Job:
    return Job(
        model="alpha",
        task=task,
        effort="low",
        repeat=repeat,
        command=("vlm-exam", "run"),
        log_path=tmp_path / f"{task}_r{repeat}.log",
    )


class TestRunJobs:
    def test_runs_all_jobs_and_reports_failures(self, tmp_path: Path) -> None:
        jobs = [_job("ocr", 1, tmp_path), _job("ocr", 2, tmp_path)]
        launched: list[Job] = []
        echoed: list[str] = []

        def launch(job: Job) -> _FakeProcess:
            launched.append(job)
            return _FakeProcess(0 if job.repeat == 1 else 3)

        outcomes = run_jobs(
            jobs,
            max_parallel=1,
            stagger_seconds=0,
            poll_seconds=0,
            echo=echoed.append,
            launch=launch,  # type: ignore[arg-type]
        )

        assert launched == jobs
        assert [outcome.ok for outcome in outcomes] == [True, False]
        assert outcomes[1].return_code == 3
        assert echoed[0].startswith("Launching 2 runs (max 1 in parallel)")
        assert str(jobs[0].log_path) in echoed[1]
        assert any(line.startswith("FAILED (3) ocr/low r2") for line in echoed)

        table = format_outcomes(outcomes)
        assert "1 of 2 runs finished cleanly; 1 failed" in table
        assert "exit 3" in table

    def test_rejects_non_positive_parallelism(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="max_parallel"):
            run_jobs([_job("ocr", 1, tmp_path)], max_parallel=0)


class TestLaunch:
    def test_writes_command_header_and_output_to_log(self, tmp_path: Path) -> None:
        job = Job(
            model="alpha",
            task="ocr",
            effort="low",
            repeat=1,
            command=(sys.executable, "-c", "print('hello from run')"),
            log_path=tmp_path / "nested" / "ocr.log",
        )

        process = _launch(job)
        process.wait(timeout=30)

        assert isinstance(process, subprocess.Popen)
        assert process.returncode == 0
        text = job.log_path.read_text()
        assert text.splitlines()[0].endswith("print('hello from run')")
        assert "hello from run" in text


class TestBenchmarkCommand:
    def test_plans_jobs_and_exits_nonzero_on_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from click.testing import CliRunner

        import vlm_exam.orchestrate as orchestrate
        from vlm_exam.cli import main

        captured: dict[str, object] = {}

        def fake_run_jobs(jobs: list[Job], **kwargs: object) -> list[JobOutcome]:
            captured["jobs"] = jobs
            captured["kwargs"] = kwargs
            return [
                JobOutcome(job=job, return_code=index, elapsed_seconds=1.0)
                for index, job in enumerate(jobs)
            ]

        monkeypatch.setattr(orchestrate, "run_jobs", fake_run_jobs)

        result = CliRunner().invoke(
            main,
            [
                "benchmark",
                "--models",
                "claude-fable-5-1",
                "--tasks",
                "counting,ocr",
                "--efforts",
                "low",
                "--repeats",
                "2",
                "--dataset-root",
                str(tmp_path),
                "--log-directory",
                str(tmp_path / "logs"),
                "--max-parallel",
                "4",
            ],
        )

        assert result.exit_code == 1, result.output
        jobs = captured["jobs"]
        assert isinstance(jobs, list)
        assert [(job.task, job.repeat) for job in jobs] == [
            ("counting", 1),
            ("counting", 2),
            ("ocr", 1),
            ("ocr", 2),
        ]
        assert captured["kwargs"]["max_parallel"] == 4  # type: ignore[index]
        assert "1 of 4 runs finished cleanly; 3 failed" in result.output
        assert "Next: vlm-exam validate" in result.output


def test_text_benchmark_defaults_to_one_run_per_effort_in_shared_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from click.testing import CliRunner

    from vlm_exam import text_benchmark
    from vlm_exam.cli import main

    captured: list[Job] = []

    def run(jobs: list[Job], **kwargs: object) -> list[JobOutcome]:
        captured.extend(jobs)
        return [JobOutcome(job=job, return_code=0, elapsed_seconds=1) for job in jobs]

    monkeypatch.setattr(text_benchmark, "run_jobs", run)
    monkeypatch.setattr(text_benchmark.TextTask, "load_samples", lambda *args: [])
    result = CliRunner().invoke(
        main,
        [
            "text-benchmark",
            "--models",
            "claude-fable-5-1",
            "--dataset-root",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert [(job.task, job.effort, job.repeat) for job in captured] == [
        ("text", "low", 1),
        ("text", "high", 1),
    ]
    for job in captured:
        index = job.command.index("--output-directory")
        assert job.command[index + 1] == "results"


def test_campaign_checkpoint_rejects_changed_manifest(tmp_path: Path) -> None:
    import pytest

    from vlm_exam.campaign import CampaignStore

    path = tmp_path / "checkpoints.sqlite"
    store = CampaignStore(path, {"model": "one", "prompt": "frozen"})
    store.save("sample", "prediction", {"answer": "saved"})
    store.charge({"generation_id": "one", "reported": {"cost": 0.25}})
    store.charge({"generation_id": "one", "reported": {"cost": 0.25}})
    resumed = CampaignStore(path, {"model": "one", "prompt": "frozen"})
    assert resumed.get("sample")["prediction"] is not None
    assert resumed.spent() == 0.25
    with pytest.raises(ValueError, match="Frozen campaign changed"):
        CampaignStore(path, {"model": "one", "prompt": "changed"})


def test_campaign_judge_failure_preserves_paid_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    import pytest
    from PIL import Image

    from vlm_exam.campaign import CampaignStore, _evaluate
    from vlm_exam.providers.base import RetryStats, Usage
    from vlm_exam.tasks.base import EvaluationResult, Sample

    path = tmp_path / "image.png"
    Image.new("RGB", (8, 8)).save(path)
    calls = []

    class _Provider:
        def uploaded_image_size(self, image: Image.Image) -> None:
            return None

        def predict(
            self, image: Image.Image, prompt: str, effort: str
        ) -> tuple[str, Usage, RetryStats]:
            calls.append(prompt)
            return (
                "saved",
                Usage(
                    10,
                    20,
                    {
                        "generation_id": "one",
                        "reported": {"cost": 0.1},
                        "finish_reason": "stop",
                    },
                ),
                RetryStats(1, 2),
            )

    class _Task:
        requires_judge = True
        failed = False

        def build_prompt(self, sample: Sample, **kwargs: Any) -> str:
            return "prompt"

        def evaluate(
            self, sample: Sample, answer: str, **kwargs: Any
        ) -> EvaluationResult:
            if not self.failed:
                self.failed = True
                raise TimeoutError("judge timeout")
            return EvaluationResult(True, strict_correct=True, judge_correct=True)

        def sample_metadata(self, sample: Sample) -> dict[str, Any]:
            return {}

        def expected_text(self, sample: Sample) -> str:
            return "saved"

    monkeypatch.setattr(
        "vlm_exam.campaign.build_model_provider", lambda *args: _Provider()
    )
    store = CampaignStore(tmp_path / "checkpoint.sqlite", {})
    task = _Task()
    arguments = (
        "sample",
        0,
        Sample(str(path)),
        task,
        "model",
        "low",
        None,
        store,
        SimpleNamespace(usage={}),
    )
    with pytest.raises(TimeoutError):
        _evaluate(*arguments)
    result = _evaluate(*arguments)
    assert calls == ["prompt"]
    assert result.correct
    assert store.spent() == 0.1
    assert store.get("sample")["result"] is not None


def _campaign_plan(tmp_path: Path) -> Path:
    import json

    from PIL import Image

    dataset = tmp_path / "data" / "ocr" / "train"
    dataset.mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(dataset / "one.png")
    (dataset / "annotations.jsonl").write_text(
        json.dumps({"image": "one.png", "prefix": "Read", "suffix": "one"}) + "\n"
    )
    config = tmp_path / "models.yaml"
    config.write_text(
        "labs: {}\nmodels:\n  sample:\n    name: Sample\n    lab: test\n"
        "    routes:\n      - provider: openrouter\n"
        "        provider_model_id: test/paid\n        options:\n"
        "          max_tokens: 65536\n"
        "          provider:\n            only: [test]\n"
        "            allow_fallbacks: false\n            data_collection: deny\n"
        "            require_parameters: true\n"
        "          reasoning:\n            low: {effort: low}\n"
        "            high: {effort: high}\n"
        "    detection_coordinate_format: xyxy_normalized_0_to_1000\n"
        "    pricing:\n      input_per_million_tokens: 1\n"
        "      output_per_million_tokens: 2\n"
    )
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "config": str(config),
                "models": ["sample"],
                "tasks": {"ocr": 1},
                "dataset_root": str(tmp_path / "data"),
                "output_directory": str(tmp_path / "results"),
                "timestamp": "20261026_120000",
            }
        )
    )
    return plan


def test_campaign_empty_balance_pauses_without_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    from vlm_exam.campaign import run_campaign

    plan = _campaign_plan(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setattr("vlm_exam.campaign._balance", lambda: 0)
    monkeypatch.setattr(
        "vlm_exam.campaign._evaluate",
        lambda *args: pytest.fail("No inference is allowed without funds"),
    )
    directory = tmp_path / "campaign"
    with pytest.raises(RuntimeError, match="paused_budget"):
        run_campaign(plan, directory, budget_usd=100)
    assert (directory / "checkpoints.sqlite").exists()
    assert json.loads((directory / "status.json").read_text())["completed"] == 0
    assert not list((tmp_path / "results").glob("*.jsonl"))


@pytest.mark.parametrize(
    "failure",
    [TimeoutError("one request timed out"), ValueError("Truncated completion")],
)
def test_campaign_transient_failure_retries_without_stopping_other_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    import itertools
    import json
    from dataclasses import asdict

    from vlm_exam.campaign import CampaignStore, run_campaign
    from vlm_exam.results import SampleResult

    plan = _campaign_plan(tmp_path)
    calls: list[str] = []
    ticks = itertools.count(step=10)

    def evaluate(identity: str, *args: Any) -> SampleResult:
        calls.append(identity)
        if ":low:" in identity and calls.count(identity) == 1:
            raise failure
        store: CampaignStore = args[-2]
        result = SampleResult(0, "one.png", "one", "one", True, 10, 10, 1.0)
        store.save(identity, "result", asdict(result))
        return result

    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setattr("vlm_exam.campaign._balance", lambda: 100)
    monkeypatch.setattr("vlm_exam.campaign._evaluate", evaluate)
    monkeypatch.setattr("vlm_exam.campaign.time.monotonic", lambda: next(ticks))
    directory = tmp_path / "campaign"
    run_campaign(plan, directory, budget_usd=100)
    status = json.loads((directory / "status.json").read_text())
    assert status["status"] == "complete"
    assert status["completed"] == 2
    assert len(calls) == 3
    assert len(list((tmp_path / "results").glob("*.jsonl"))) == 2
    run_campaign(plan, directory, budget_usd=100)
    assert len(calls) == 3


def test_campaign_rate_limit_keeps_other_provider_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import copy
    import json
    from dataclasses import asdict

    import yaml

    from vlm_exam.campaign import CampaignStore, run_campaign
    from vlm_exam.results import SampleResult

    plan_path = _campaign_plan(tmp_path)
    plan = json.loads(plan_path.read_text())
    config_path = Path(plan["config"])
    config = yaml.safe_load(config_path.read_text())
    config["models"]["other"] = copy.deepcopy(config["models"]["sample"])
    config["models"]["other"]["routes"][0]["options"]["provider"]["only"] = ["other"]
    config_path.write_text(yaml.safe_dump(config))
    plan["models"].append("other")
    plan["tasks"]["ocr"] = 50
    plan_path.write_text(json.dumps(plan))
    calls: list[str] = []
    ticks = 0
    other_at_retry = []

    class _RateLimit(Exception):
        status_code = 429

    def clock() -> float:
        nonlocal ticks
        ticks += 1
        other = sum(identity.startswith("other:") for identity in calls)
        return 1000.0 if other >= 80 or ticks > 1000 else 0.0

    def evaluate(identity: str, *args: Any) -> SampleResult:
        calls.append(identity)
        if identity == "sample:ocr:low:0:0":
            if calls.count(identity) == 1:
                raise _RateLimit("upstream provider rate limit")
            other_at_retry.append(sum(item.startswith("other:") for item in calls))
        store: CampaignStore = args[-2]
        result = SampleResult(0, "one.png", "one", "one", True, 10, 10, 1.0)
        store.save(identity, "result", asdict(result))
        return result

    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setattr("vlm_exam.campaign._balance", lambda: 100)
    monkeypatch.setattr("vlm_exam.campaign._evaluate", evaluate)
    monkeypatch.setattr("vlm_exam.campaign.time.monotonic", clock)
    monkeypatch.setattr("vlm_exam.campaign.time.sleep", lambda _: None)
    directory = tmp_path / "campaign"
    run_campaign(plan_path, directory, budget_usd=100, concurrency=48)
    assert other_at_retry and other_at_retry[0] >= 80
    assert len(calls) == 201
    assert json.loads((directory / "status.json").read_text())["completed"] == 200


def test_campaign_repeated_truncation_pauses_after_bounded_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json
    from dataclasses import asdict

    from vlm_exam.campaign import CampaignStore, run_campaign
    from vlm_exam.results import SampleResult

    plan = _campaign_plan(tmp_path)
    calls: list[str] = []

    def evaluate(identity: str, *args: Any) -> SampleResult:
        calls.append(identity)
        if ":low:" in identity:
            raise ValueError("Truncated completion")
        store: CampaignStore = args[-2]
        result = SampleResult(0, "one.png", "one", "one", True, 10, 10, 1.0)
        store.save(identity, "result", asdict(result))
        return result

    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setattr("vlm_exam.campaign._balance", lambda: 100)
    monkeypatch.setattr("vlm_exam.campaign._evaluate", evaluate)
    directory = tmp_path / "campaign"
    with pytest.raises(RuntimeError, match="paused_configuration"):
        run_campaign(plan, directory, budget_usd=100)
    assert sum(":low:" in identity for identity in calls) == 2
    assert sum(":high:" in identity for identity in calls) == 1
    assert json.loads((directory / "status.json").read_text())["completed"] == 1


def test_campaign_deferred_sample_does_not_block_other_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json
    from dataclasses import asdict

    from vlm_exam.campaign import CampaignStore, run_campaign
    from vlm_exam.results import SampleResult

    plan = _campaign_plan(tmp_path)
    directory = tmp_path / "campaign"
    directory.mkdir()
    deferred = "sample:ocr:low:0:0"
    (directory / "control.json").write_text(
        json.dumps({"concurrency": 32, "deferred_samples": [deferred]})
    )
    calls: list[str] = []

    def evaluate(identity: str, *args: Any) -> SampleResult:
        calls.append(identity)
        store: CampaignStore = args[-2]
        result = SampleResult(0, "one.png", "one", "one", True, 10, 10, 1.0)
        store.save(identity, "result", asdict(result))
        return result

    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setattr("vlm_exam.campaign._balance", lambda: 100)
    monkeypatch.setattr("vlm_exam.campaign._evaluate", evaluate)
    with pytest.raises(RuntimeError, match="paused_deferred_samples"):
        run_campaign(plan, directory, budget_usd=100)
    assert calls == ["sample:ocr:high:0:0"]
    (directory / "control.json").write_text(json.dumps({"concurrency": 32}))
    run_campaign(plan, directory, budget_usd=100)
    assert calls == ["sample:ocr:high:0:0", deferred]
    assert json.loads((directory / "status.json").read_text())["completed"] == 2


def test_campaign_format_import_preserves_other_models_and_all_charges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib.util
    import json
    import sqlite3
    from dataclasses import dataclass
    from types import SimpleNamespace

    specification = importlib.util.spec_from_file_location(
        "campaign_import", Path("scripts/import_unaffected_campaign.py")
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)

    @dataclass
    class Model:
        detection_coordinate_format: str

    monkeypatch.setattr(
        module,
        "load_config",
        lambda: SimpleNamespace(
            models={"changed": Model("1000"), "other": Model("999")}
        ),
    )
    source = tmp_path / "source"
    source.mkdir()
    manifest = {
        "models": {
            "changed": {"detection_coordinate_format": "999"},
            "other": {"detection_coordinate_format": "999"},
        },
        "code": {},
    }
    (source / "manifest.json").write_text(json.dumps(manifest))
    database = sqlite3.connect(source / "checkpoints.sqlite")
    database.executescript(
        "CREATE TABLE manifest(payload TEXT);"
        "CREATE TABLE samples(identity TEXT PRIMARY KEY, prediction TEXT, result TEXT);"
        "CREATE TABLE charges(generation TEXT PRIMARY KEY, payload TEXT);"
    )
    database.execute("INSERT INTO manifest VALUES (?)", (json.dumps(manifest),))
    rows = [
        ("changed:detection:high:0:0", "prediction", "result"),
        ("changed:text:low:0:0", "prediction", None),
        ("changed:counting:low:0:0", "prediction", "result"),
        ("other:detection:high:0:0", "prediction", "result"),
    ]
    database.executemany("INSERT INTO samples VALUES (?,?,?)", rows)
    database.execute("INSERT INTO charges VALUES ('generation', 'paid')")
    database.commit()
    with pytest.raises(ValueError, match="Affected-task"):
        module.prepare(source, tmp_path / "rejected", apply=True)
    assert not (tmp_path / "rejected").exists()
    destination = tmp_path / "destination"
    receipt = module.prepare(
        source, destination, apply=True, invalidate_models=("changed",)
    )
    target = sqlite3.connect(destination / "checkpoints.sqlite")
    assert (
        target.execute("SELECT * FROM samples ORDER BY identity").fetchall() == rows[2:]
    )
    assert database.execute("SELECT * FROM samples").fetchall() == rows
    assert target.execute("SELECT * FROM charges").fetchall() == [
        ("generation", "paid")
    ]
    assert receipt["completed_results"] == 2
    assert receipt["invalidated_completed_results"] == 1
    target.close()
    database.close()


def test_campaign_live_export_omits_partial_runs(tmp_path: Path) -> None:
    import importlib.util
    import json
    import sqlite3
    from dataclasses import asdict

    from vlm_exam.results import SampleResult, load_results

    specification = importlib.util.spec_from_file_location(
        "campaign_publish", Path("scripts/campaign_publish.py")
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    directory = tmp_path / "campaign"
    directory.mkdir()
    output = tmp_path / "results"
    output.mkdir()
    manifest = {
        "plan": {
            "models": ["sample"],
            "tasks": {"ocr": 1},
            "timestamp": "20261026_120000",
            "output_directory": str(output),
        }
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))
    connection = sqlite3.connect(directory / "checkpoints.sqlite")
    connection.execute("CREATE TABLE manifest(payload TEXT)")
    connection.execute("INSERT INTO manifest VALUES (?)", (json.dumps(manifest),))
    connection.execute("CREATE TABLE samples(identity TEXT, result TEXT)")
    result = SampleResult(0, "one.png", "one", "one", True, 10, 10, 1.0)
    connection.execute(
        "INSERT INTO samples VALUES (?, ?)",
        (
            "sample:ocr:low:0:0",
            json.dumps(asdict(result)),
        ),
    )
    connection.commit()
    connection.close()
    assert module.publish(directory, {"ocr": 1}) == 1
    paths = list(output.glob("*.jsonl"))
    assert len(paths) == 1 and load_results(paths[0]).effort == "low"
    before = paths[0].read_bytes()
    assert module.publish(directory, {"ocr": 1}) == 0
    assert paths[0].read_bytes() == before


def test_campaign_publisher_refreshes_external_exports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib.util
    import json

    specification = importlib.util.spec_from_file_location(
        "campaign_publish", Path("scripts/campaign_publish.py")
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    output = tmp_path / "results"
    output.mkdir()
    (output / "existing.jsonl").write_text("{}\n")
    (tmp_path / "manifest.json").write_text(
        json.dumps({"plan": {"tasks": {}, "output_directory": str(output)}})
    )
    calls = []
    sleeps = 0

    def pause(seconds: int) -> None:
        nonlocal sleeps
        sleeps += 1
        if sleeps == 1:
            (output / "runner-export.jsonl").write_text("{}\n")
        if sleeps == 3:
            raise StopIteration

    monkeypatch.setattr(sys, "argv", ["campaign_publish", str(tmp_path), "--watch"])
    monkeypatch.setattr(module, "publish", lambda directory, counts: 0)
    monkeypatch.setattr(
        module.subprocess, "run", lambda *args, **kwargs: calls.append(args)
    )
    monkeypatch.setattr(module.time, "sleep", pause)
    with pytest.raises(StopIteration):
        module.main()
    assert len(calls) == 2


@pytest.mark.parametrize("finish", ["error", "content_filter", None])
@pytest.mark.parametrize("cached", [False, True])
def test_campaign_rejects_incomplete_provider_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, finish: str | None, cached: bool
) -> None:
    from dataclasses import asdict
    from types import SimpleNamespace

    from PIL import Image

    from vlm_exam.campaign import CampaignStore, _evaluate
    from vlm_exam.providers.base import RetryStats, Usage
    from vlm_exam.tasks.base import Sample

    path = tmp_path / "image.png"
    Image.new("RGB", (8, 8)).save(path)
    usage = Usage(
        10,
        20,
        {"generation_id": "failed", "reported": {"cost": 0.1}, "finish_reason": finish},
    )
    calls = []

    class _Provider:
        def uploaded_image_size(self, image: Image.Image) -> None:
            return None

        def predict(
            self, image: Image.Image, prompt: str, effort: str
        ) -> tuple[str, Usage, RetryStats]:
            calls.append(prompt)
            return "partial answer", usage, RetryStats(1, 2)

    class _Task:
        requires_judge = True

        def build_prompt(self, sample: Sample, **kwargs: Any) -> str:
            return "prompt"

        def evaluate(self, sample: Sample, answer: str, **kwargs: Any) -> None:
            raise AssertionError("Incomplete answers must not be scored or judged")

    monkeypatch.setattr(
        "vlm_exam.campaign.build_model_provider", lambda *args: _Provider()
    )
    store = CampaignStore(tmp_path / "checkpoints.sqlite", {})
    if cached:
        store.save(
            "sample", "prediction", {"answer": "partial answer", "usage": asdict(usage)}
        )
        store.charge(usage.metadata)
    with pytest.raises(ValueError, match="Incomplete provider response"):
        _evaluate(
            "sample",
            0,
            Sample(str(path)),
            _Task(),
            "model",
            "low",
            None,
            store,
            SimpleNamespace(usage={}),
        )
    assert calls == ([] if cached else ["prompt"])
    assert store.spent() == 0.1
    assert store.get("sample")["result"] is None
    if not cached:
        assert store.get("sample")["prediction"] is None


@pytest.mark.parametrize(
    ("input_tokens", "expected_cost"), [(1000, 0.00175), (300000, 0.75075)]
)
def test_campaign_native_cost_accounts_for_cache_and_long_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    input_tokens: int,
    expected_cost: float,
) -> None:
    from types import SimpleNamespace

    from PIL import Image

    from vlm_exam.campaign import CampaignStore, _evaluate
    from vlm_exam.providers.base import RetryStats, Usage
    from vlm_exam.tasks.base import Sample

    path = tmp_path / "image.png"
    Image.new("RGB", (8, 8)).save(path)
    usage = Usage(
        input_tokens,
        100,
        {
            "generation_id": "native-response",
            "api_endpoint": "https://api.openai.com/v1/responses",
            "reported": {},
            "finish_reason": "stop",
        },
    )

    class _Provider:
        def uploaded_image_size(self, image: Image.Image) -> None:
            return None

        def predict(
            self, image: Image.Image, prompt: str, effort: str
        ) -> tuple[str, Usage, RetryStats]:
            return "answer", usage, RetryStats(1, 2)

    class _Task:
        requires_judge = False

        def build_prompt(self, sample: Sample, **kwargs: Any) -> str:
            return "prompt"

        def evaluate(self, sample: Sample, answer: str, **kwargs: Any) -> None:
            raise TimeoutError("scoring interrupted after paid prediction")

    monkeypatch.setattr(
        "vlm_exam.campaign.build_model_provider", lambda *args: _Provider()
    )
    configuration = SimpleNamespace(
        pricing=SimpleNamespace(input_per_million_tokens=1, output_per_million_tokens=5)
    )
    store = CampaignStore(tmp_path / "checkpoints.sqlite", {})
    with pytest.raises(TimeoutError):
        _evaluate(
            "sample",
            0,
            Sample(str(path)),
            _Task(),
            "model",
            "high",
            configuration,
            store,
            SimpleNamespace(usage={}),
        )
    assert store.spent() == pytest.approx(expected_cost)
    assert usage.metadata["cost_source"] == "conservative_token_estimate"
    assert store.get("sample")["prediction"] is not None


@pytest.mark.parametrize("case", ["valid", "dataset_mismatch", "invalid_count"])
def test_campaign_counted_native_inputs_preserve_budget_and_frozen_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    import json
    from dataclasses import asdict

    import yaml

    from vlm_exam.campaign import CampaignStore, run_campaign
    from vlm_exam.results import SampleResult

    plan_path = _campaign_plan(tmp_path)
    plan = json.loads(plan_path.read_text())
    config_path = Path(plan["config"])
    config = yaml.safe_load(config_path.read_text())
    config["models"]["sample"]["routes"] = [
        {
            "provider": "openai",
            "provider_model_id": "native-model",
            "options": {
                "max_output_tokens": 65536,
                "store": False,
                "service_tier": "default",
            },
        }
    ]
    config_path.write_text(yaml.safe_dump(config))
    calls = []

    def evaluate(identity: str, *args: Any) -> SampleResult:
        calls.append(identity)
        store: CampaignStore = args[-2]
        result = SampleResult(0, "one.png", "one", "one", True, 100, 10, 1.0)
        store.save(identity, "result", asdict(result))
        return result

    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setattr("vlm_exam.campaign._balance", lambda: 100)
    monkeypatch.setattr("vlm_exam.campaign._evaluate", evaluate)
    directory = tmp_path / "campaign"
    with pytest.raises(RuntimeError, match="paused_budget"):
        run_campaign(plan_path, directory, budget_usd=0.5, concurrency=48)
    assert calls == []
    manifest = json.loads((directory / "manifest.json").read_text())
    counted = {key: manifest[key] for key in ("models", "datasets", "code")}
    counted["counts"] = {
        f"sample:ocr:{effort}:0": {
            "input_tokens": 0 if case == "invalid_count" else 100
        }
        for effort in ("low", "high")
    }
    if case == "dataset_mismatch":
        counted["datasets"] = {}
    count_file = tmp_path / "counts.json"
    count_file.write_text(json.dumps(counted))
    (directory / "control.json").write_text(
        json.dumps({"concurrency": 48, "input_token_counts_file": str(count_file)})
    )
    if case != "valid":
        with pytest.raises(ValueError):
            run_campaign(plan_path, directory, budget_usd=0.5, concurrency=48)
        assert calls == []
    else:
        run_campaign(plan_path, directory, budget_usd=0.5, concurrency=48)
        assert len(calls) == 2
        assert json.loads((directory / "status.json").read_text())["completed"] == 2
