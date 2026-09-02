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
from click.testing import CliRunner

from vlm_exam.cli import main
from vlm_exam.config import (
    BenchmarkConfig,
    LabConfig,
    ModelConfig,
    PricingConfig,
    RouteConfig,
)
from vlm_exam.protocol import PROTOCOL, BenchmarkProtocol
from vlm_exam.results import RunResult, SampleResult, save_results
from vlm_exam.tasks.detection import DetectionCoordinateFormat
from vlm_exam.validation import (
    ERROR,
    KIND_FAILED_SAMPLES,
    KIND_ORPHAN,
    KIND_PARTIAL,
    KIND_RUNS,
    KIND_VERDICTS,
    WARNING,
    format_github_annotations,
    format_github_summary,
    format_report,
    validate_results,
)

SMALL = BenchmarkProtocol(repeats=2, efforts=("low", "high"), tasks=("counting", "ocr"))


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


def _sample(index: int, *, failed: bool = False, verdicts: bool = True) -> SampleResult:
    metadata = {"strict_correct": True, "judge_correct": True} if verdicts else {}
    return SampleResult(
        index=index,
        image=f"{index}.jpg",
        expected="1",
        predicted="ERROR: boom" if failed else "1",
        correct=not failed,
        input_tokens=10,
        output_tokens=5,
        elapsed_seconds=1.0,
        metadata=metadata,
    )


def _save(
    directory: Path,
    model: str,
    task: str,
    effort: str,
    repeat: int,
    *,
    sample_count: int = 3,
    failed: int = 0,
    verdicts: bool = True,
) -> Path:
    samples = [
        _sample(index, failed=index < failed, verdicts=verdicts)
        for index in range(sample_count)
    ]
    run = RunResult(
        model=model,
        effort=effort,
        task=task,
        timestamp=f"2026090{repeat}_000000",
        samples=samples,
    )
    path = directory / f"{task}_{model}_{effort}_{run.timestamp}.jsonl"
    save_results(run, path)
    return path


def _complete(directory: Path, model: str, protocol: BenchmarkProtocol = SMALL) -> None:
    for task, effort in protocol.configurations:
        for repeat in range(1, protocol.repeats + 1):
            _save(directory, model, task, effort, repeat)


def _kinds(problems: tuple) -> list[str]:
    return [problem.kind for problem in problems]


class TestValidateResults:
    def test_complete_model_passes(self, tmp_path: Path) -> None:
        _complete(tmp_path, "alpha")

        report = validate_results(tmp_path, _config("alpha"), protocol=SMALL)

        assert report.ok
        (entry,) = report.coverage
        assert entry.status == "complete"
        assert entry.runs_present == entry.runs_required == 8
        assert entry.problems == ()

    def test_missing_effort_fails_full_protocol_model(self, tmp_path: Path) -> None:
        for task in SMALL.tasks:
            for repeat in (1, 2):
                _save(tmp_path, "alpha", task, "low", repeat)

        report = validate_results(tmp_path, _config("alpha"), protocol=SMALL)

        assert not report.ok
        (entry,) = report.coverage
        assert entry.status == "incomplete"
        assert entry.runs_present == 4
        scopes = {(p.task, p.effort) for p in entry.errors}
        assert scopes == {("counting", "high"), ("ocr", "high")}
        fix = entry.errors[0].fix
        assert fix is not None
        assert "--effort high" in fix
        assert "--repeats 2" in fix

    def test_missing_repeat_reports_remaining_count(self, tmp_path: Path) -> None:
        _complete(tmp_path, "alpha")
        (tmp_path / "counting_alpha_low_20260902_000000.jsonl").unlink()

        report = validate_results(tmp_path, _config("alpha"), protocol=SMALL)

        (problem,) = report.errors
        assert problem.kind == KIND_RUNS
        assert problem.message == "1 of 2 runs"
        assert problem.fix is not None
        assert problem.fix.endswith("--concurrency 2")
        assert "--repeats" not in problem.fix

    def test_surplus_run_is_reported(self, tmp_path: Path) -> None:
        _complete(tmp_path, "alpha")
        _save(tmp_path, "alpha", "counting", "low", 3)

        report = validate_results(tmp_path, _config("alpha"), protocol=SMALL)

        (problem,) = report.errors
        assert problem.kind == KIND_RUNS
        assert "surplus" in problem.message

    def test_partial_run_is_detected_against_largest_run(self, tmp_path: Path) -> None:
        _complete(tmp_path, "alpha")
        (tmp_path / "ocr_alpha_high_20260902_000000.jsonl").unlink()
        _save(tmp_path, "alpha", "ocr", "high", 2, sample_count=1)

        report = validate_results(tmp_path, _config("alpha"), protocol=SMALL)

        (problem,) = report.errors
        assert problem.kind == KIND_PARTIAL
        assert "1 of 3 samples" in problem.message
        assert "--max-samples" in (problem.fix or "")

    def test_failed_samples_are_warnings_with_resume_command(
        self, tmp_path: Path
    ) -> None:
        _complete(tmp_path, "alpha")
        (tmp_path / "counting_alpha_high_20260901_000000.jsonl").unlink()
        path = _save(tmp_path, "alpha", "counting", "high", 1, failed=2)

        report = validate_results(tmp_path, _config("alpha"), protocol=SMALL)

        assert report.ok
        (problem,) = report.warnings
        assert problem.kind == KIND_FAILED_SAMPLES
        assert "2 failed sample(s)" in problem.message
        assert problem.fix is not None
        assert f"--resume-file {path}" in problem.fix

    def test_missing_verdicts_fail_even_for_legacy(self, tmp_path: Path) -> None:
        _save(tmp_path, "old", "counting", "low", 1, verdicts=False)

        report = validate_results(
            tmp_path, _config("old", legacy=("old",)), protocol=SMALL
        )

        assert not report.ok
        (problem,) = report.errors
        assert problem.kind == KIND_VERDICTS
        assert problem.fix is not None
        assert problem.fix.startswith("vlm-exam rescore ")

    def test_ocr_runs_do_not_need_verdicts(self, tmp_path: Path) -> None:
        _complete(tmp_path, "alpha")
        (tmp_path / "ocr_alpha_low_20260901_000000.jsonl").unlink()
        _save(tmp_path, "alpha", "ocr", "low", 1, verdicts=False)

        report = validate_results(tmp_path, _config("alpha"), protocol=SMALL)

        assert report.ok

    def test_legacy_gaps_are_warnings_unless_strict(self, tmp_path: Path) -> None:
        _save(tmp_path, "old", "counting", "low", 1)
        config = _config("old", legacy=("old",))

        report = validate_results(tmp_path, config, protocol=SMALL)
        strict = validate_results(tmp_path, config, protocol=SMALL, strict=True)

        assert report.ok
        (entry,) = report.coverage
        assert entry.status == "legacy"
        assert {p.severity for p in entry.problems} == {WARNING}
        assert len(entry.gaps) == 4
        assert not strict.ok
        assert {p.severity for p in strict.coverage[0].problems} == {ERROR}

    def test_model_without_any_runs_is_incomplete(self, tmp_path: Path) -> None:
        report = validate_results(tmp_path, _config("alpha"), protocol=SMALL)

        (entry,) = report.coverage
        assert entry.status == "incomplete"
        assert entry.runs_present == 0
        assert all(p.message == "0 of 2 runs" for p in entry.errors)

    def test_orphan_model_is_an_error(self, tmp_path: Path) -> None:
        _complete(tmp_path, "alpha")
        _save(tmp_path, "ghost", "counting", "low", 1)

        report = validate_results(tmp_path, _config("alpha"), protocol=SMALL)

        assert not report.ok
        (problem,) = report.orphans
        assert problem.kind == KIND_ORPHAN
        assert problem.severity == ERROR
        assert problem.model == "ghost"

    def test_unprotocol_task_and_effort_are_warnings(self, tmp_path: Path) -> None:
        _complete(tmp_path, "alpha")
        _save(tmp_path, "alpha", "vqa", "low", 1)
        _save(tmp_path, "alpha", "counting", "max", 1)

        report = validate_results(tmp_path, _config("alpha"), protocol=SMALL)

        assert report.ok
        assert _kinds(report.orphans) == [KIND_ORPHAN, KIND_ORPHAN]
        assert {p.severity for p in report.orphans} == {WARNING}
        assert report.coverage[0].runs_present == 8

    def test_default_protocol_requires_thirty_six_runs(self, tmp_path: Path) -> None:
        report = validate_results(tmp_path, _config("alpha"))

        assert report.protocol is PROTOCOL
        assert report.coverage[0].runs_required == 36


class TestFormatting:
    def test_report_lists_fixes_and_collapses_legacy(self, tmp_path: Path) -> None:
        _save(tmp_path, "old", "counting", "low", 1)
        _save(tmp_path, "new", "counting", "low", 1)
        config = _config("old", "new", legacy=("old",))
        report = validate_results(tmp_path, config, protocol=SMALL)

        text = format_report(report)

        assert "LEGACY  old" in text
        assert "FAIL    new" in text
        assert "7 missing run(s) across 4 configuration(s)" in text
        assert "vlm-exam benchmark --models old" in text
        assert text.count("  [error]") == 4
        assert text.count("  [warning]") == 1
        assert "-> vlm-exam run --task ocr --models new --effort high" in text
        assert text.endswith("Fix the [error] items above.")

        verbose = format_report(report, verbose=True)
        assert verbose.count("  [warning]") == 4

    def test_github_annotations_collapse_warnings_per_model(
        self, tmp_path: Path
    ) -> None:
        _save(tmp_path, "old", "counting", "low", 1)
        _save(tmp_path, "new", "counting", "low", 1, failed=1)
        config = _config("old", "new", legacy=("old",))
        report = validate_results(tmp_path, config, protocol=SMALL)

        lines = format_github_annotations(report).splitlines()

        errors = [line for line in lines if line.startswith("::error ")]
        warnings = [line for line in lines if line.startswith("::warning ")]
        assert len(errors) == 4
        assert errors[0].startswith("::error title=new counting/low::")
        assert len(warnings) == 2
        assert any("title=old::7 missing run(s)" in line for line in warnings)
        assert any("title=new::" in line and "failed" in line for line in warnings)

    def test_github_summary_has_table_and_verdict(self, tmp_path: Path) -> None:
        _complete(tmp_path, "alpha")
        report = validate_results(tmp_path, _config("alpha"), protocol=SMALL)

        markdown = format_github_summary(report)

        assert markdown.startswith("## Benchmark protocol validation: PASS")
        assert "| OK | `alpha` | full | 8/8 | 0 | 0 |" in markdown
        assert "### Findings" not in markdown


class TestValidateCommand:
    def _write_config(self, tmp_path: Path, config: BenchmarkConfig) -> Path:
        raw = {
            "labs": {
                key: {"name": lab.name, "color": lab.color, "logo_url": lab.logo_url}
                for key, lab in config.labs.items()
            },
            "models": {
                key: {
                    "name": model.name,
                    "lab": model.lab,
                    "provider": model.provider,
                    "detection_coordinate_format": (
                        model.detection_coordinate_format.value
                    ),
                    "benchmark_protocol": model.benchmark_protocol,
                    "pricing": {
                        "input_per_million_tokens": 1.0,
                        "output_per_million_tokens": 2.0,
                    },
                }
                for key, model in config.models.items()
            },
        }
        path = tmp_path / "models.yaml"
        path.write_text(yaml.safe_dump(raw))
        return path

    def test_exit_code_and_step_summary(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        results = tmp_path / "results"
        results.mkdir()
        _save(results, "new", "counting", "low", 1)
        config_path = self._write_config(tmp_path, _config("new"))
        step_summary = tmp_path / "summary.md"
        monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(step_summary))

        result = CliRunner().invoke(
            main,
            [
                "validate",
                "--results-directory",
                str(results),
                "--config",
                str(config_path),
                "--format",
                "github",
            ],
        )

        assert result.exit_code == 1
        assert "::error title=new counting/high::" in result.output
        assert step_summary.read_text().startswith(
            "## Benchmark protocol validation: FAIL"
        )

    def test_legacy_only_repository_passes(self, tmp_path: Path) -> None:
        results = tmp_path / "results"
        results.mkdir()
        _save(results, "old", "counting", "low", 1)
        config_path = self._write_config(tmp_path, _config("old", legacy=("old",)))

        result = CliRunner().invoke(
            main,
            [
                "validate",
                "--results-directory",
                str(results),
                "--config",
                str(config_path),
            ],
        )

        assert result.exit_code == 0, result.output
        assert "PASS: no protocol violations" in result.output

        strict = CliRunner().invoke(
            main,
            [
                "validate",
                "--results-directory",
                str(results),
                "--config",
                str(config_path),
                "--strict",
            ],
        )
        assert strict.exit_code == 1
