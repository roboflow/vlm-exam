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

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from vlm_exam import cli
from vlm_exam.cli import _unique_result_path
from vlm_exam.config import (
    BenchmarkConfig,
    LabConfig,
    ModelConfig,
    PricingConfig,
    RouteConfig,
)
from vlm_exam.judge import Judge
from vlm_exam.results import RunResult, SampleResult, load_results, save_results
from vlm_exam.tasks.base import EvaluationResult, Sample, Task
from vlm_exam.tasks.detection import DetectionCoordinateFormat


@dataclass(frozen=True)
class _StubSample(Sample):
    label: str = ""


class _StubTask(Task):
    def load_samples(self, data_directory: str) -> list[Sample]:
        return [
            _StubSample(image_path=f"{data_directory}/{name}", label=name)
            for name in ("a.jpg", "b.jpg", "c.jpg")
        ]

    def build_prompt(
        self,
        sample: Sample,
        *,
        uploaded_size: tuple[int, int] | None = None,
    ) -> str:
        return ""

    def evaluate(
        self,
        sample: Sample,
        prediction: str,
        *,
        judge: Judge | None = None,
        uploaded_size: tuple[int, int] | None = None,
    ) -> EvaluationResult:
        return EvaluationResult(correct=True)

    def expected_text(self, sample: Sample) -> str:
        assert isinstance(sample, _StubSample)
        return sample.label


def _config() -> BenchmarkConfig:
    return BenchmarkConfig(
        labs={"openai": LabConfig("OpenAI", "#000", "https://example.com/logo.svg")},
        models={
            "alpha": ModelConfig(
                name="alpha",
                lab="openai",
                routes=(RouteConfig("openai"),),
                pricing=PricingConfig(1.0, 2.0),
                detection_coordinate_format=(
                    DetectionCoordinateFormat.XYXY_ABSOLUTE_ORIGINAL_IMAGE
                ),
            )
        },
    )


def _sample(index: int, image: str, predicted: str) -> SampleResult:
    return SampleResult(
        index=index,
        image=image,
        expected=image,
        predicted=predicted,
        correct=not predicted.startswith("ERROR"),
        input_tokens=1,
        output_tokens=1,
        elapsed_seconds=0.1,
    )


@pytest.fixture
def stubbed_run(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    timestamps = iter(f"20260707_00000{index}" for index in range(10))

    def fake_run_benchmark(**kwargs: Any) -> RunResult:
        calls.append(kwargs)
        samples = [
            _sample(index, Path(sample.image_path).name, "ok")
            for index, sample in enumerate(kwargs["samples"])
        ]
        return RunResult(
            model="alpha",
            effort=kwargs["effort"],
            task=kwargs["task_name"],
            timestamp=next(timestamps),
            samples=samples,
        )

    monkeypatch.setattr(cli, "load_config", lambda path: _config())
    monkeypatch.setattr(cli, "create_task", lambda name, **kwargs: _StubTask())
    monkeypatch.setattr(cli, "build_model_provider", lambda model, config: object())
    monkeypatch.setattr(cli, "run_benchmark", fake_run_benchmark)
    return calls


def _invoke(tmp_path: Path, *extra: str) -> Any:
    return CliRunner().invoke(
        cli.main,
        [
            "run",
            "--task",
            "ocr",
            "--models",
            "alpha",
            "--effort",
            "low",
            "--dataset-directory",
            str(tmp_path),
            "--output-directory",
            str(tmp_path / "results"),
            *extra,
        ],
        catch_exceptions=False,
    )


class TestUniqueResultPath:
    def test_suffixes_when_file_exists(self, tmp_path: Path) -> None:
        stem = "ocr_alpha_low_20260707_000000"
        (tmp_path / f"{stem}.jsonl").touch()
        (tmp_path / f"{stem}_2.jsonl").touch()

        path = _unique_result_path(tmp_path, "ocr", "alpha", "low", "20260707_000000")

        assert path.name == f"{stem}_3.jsonl"


class TestRunRepeats:
    def test_writes_one_file_per_repeat(
        self, tmp_path: Path, stubbed_run: list[dict[str, Any]]
    ) -> None:
        result = _invoke(tmp_path, "--repeats", "3", "--concurrency", "4")

        assert result.exit_code == 0, result.output
        files = sorted((tmp_path / "results").glob("ocr_alpha_low_*.jsonl"))
        assert len(files) == 3
        assert len(stubbed_run) == 3
        assert all(call["concurrency"] == 4 for call in stubbed_run)

    def test_repeats_cannot_combine_with_resume(
        self, tmp_path: Path, stubbed_run: list[dict[str, Any]]
    ) -> None:
        previous = tmp_path / "previous.jsonl"
        save_results(RunResult("alpha", "low", "ocr", "20260706_000000", []), previous)

        result = CliRunner().invoke(
            cli.main,
            [
                "run",
                "--task",
                "ocr",
                "--models",
                "alpha",
                "--effort",
                "low",
                "--dataset-directory",
                str(tmp_path),
                "--resume-file",
                str(previous),
                "--repeats",
                "2",
            ],
        )

        assert result.exit_code != 0
        assert "--repeats" in result.output


class TestRunResume:
    def test_resume_replaces_failures_and_deletes_source(
        self, tmp_path: Path, stubbed_run: list[dict[str, Any]]
    ) -> None:
        output = tmp_path / "results"
        output.mkdir()
        previous_path = output / "ocr_alpha_low_20260706_000000.jsonl"
        save_results(
            RunResult(
                "alpha",
                "low",
                "ocr",
                "20260706_000000",
                [
                    _sample(0, "a.jpg", "fine"),
                    _sample(1, "b.jpg", "ERROR: timeout"),
                    _sample(2, "c.jpg", "fine"),
                ],
            ),
            previous_path,
        )

        result = _invoke(tmp_path, "--resume-file", str(previous_path))

        assert result.exit_code == 0, result.output
        assert [Path(s.image_path).name for s in stubbed_run[0]["samples"]] == ["b.jpg"]
        assert not previous_path.exists()
        (merged_path,) = output.glob("*.jsonl")
        merged = load_results(merged_path)
        assert [sample.predicted for sample in merged.samples] == [
            "fine",
            "ok",
            "fine",
        ]
        assert "Removed resumed file" in result.output


@pytest.mark.parametrize("prediction", ["", '{"total":'])
def test_invalid_json_card_keeps_reference(
    prediction: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from PIL import Image

    from vlm_exam.visualization import text as cards

    font = FontProperties()
    monkeypatch.setattr(cards.theme, "load_fonts", lambda: SimpleNamespace(bold=font))
    monkeypatch.setattr(cards, "draw_verdict", lambda *args: None)
    monkeypatch.setattr(cards.theme, "draw_legend_chip", lambda *args: 0.5)
    monkeypatch.setattr(cards.theme, "draw_brand_footer", lambda *args: None)

    def frame(*args: Any) -> tuple[Any, Any]:
        figure = plt.figure()
        return figure, figure.add_axes([0, 0, 1, 1])

    monkeypatch.setattr(cards, "_frame", frame)
    drawn = []
    monkeypatch.setattr(
        cards,
        "draw_diff_line",
        lambda rail, x, y, spans, *args: drawn.extend(spans),
    )
    figures = cards._json_cards(
        Image.new("RGB", (20, 20)),
        SimpleNamespace(answer={"total": "42.00"}),
        SimpleNamespace(predicted=prediction, correct=False),
        "model",
        None,
    )
    try:
        text = "".join(span[0] for span in drawn)
        assert '"total": "42.00"' in text
        assert "INVALID JSON" in text
        assert prediction in text if prediction else "(empty response)" in text
    finally:
        for figure in figures:
            plt.close(figure)


@pytest.mark.parametrize("count", [1, 10])
def test_leaderboard_export_has_fixed_visible_padding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, count: int
) -> None:
    from types import SimpleNamespace

    import numpy as np
    from matplotlib.font_manager import FontProperties
    from PIL import Image

    from vlm_exam.config import DisplayConfig, ModelDisplay
    from vlm_exam.visualization import charts

    font = FontProperties()
    monkeypatch.setattr(
        charts,
        "load_fonts",
        lambda: SimpleNamespace(display=font, bold=font, medium=font),
    )
    monkeypatch.setattr(charts, "_add_model_label", lambda *args, **kwargs: None)
    config = DisplayConfig(
        labs={"lab": LabConfig("Lab", "#ff0000", "")},
        models={f"m{i}": ModelDisplay(f"Model {i}", "lab") for i in range(count)},
    )
    scores = {key: 80.0 for key in config.models}
    figure = charts.plot_accuracy_chart(
        scores, config, "Example leaderboard", run_counts={key: 3 for key in scores}
    )
    assert not any(
        "mean over" in text.get_text() for axes in figure.axes for text in axes.texts
    )
    path = tmp_path / "chart.png"
    charts.save_leaderboard_chart(figure, path)
    pixels = np.asarray(Image.open(path).convert("RGB")).astype(np.int16)
    first = np.nonzero(np.any(np.max(abs(pixels - pixels[0, 0]), axis=2) > 30, axis=1))[
        0
    ][0]
    assert first == 118
