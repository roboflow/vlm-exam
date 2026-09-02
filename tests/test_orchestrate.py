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
