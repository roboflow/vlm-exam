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


@pytest.fixture()
def card_fonts(monkeypatch: pytest.MonkeyPatch) -> None:
    from matplotlib.font_manager import FontProperties
    from PIL import Image

    from vlm_exam.visualization import theme

    monkeypatch.setattr(
        theme, "fetch_logo", lambda *args, **kwargs: Image.new("RGBA", (30, 30))
    )
    font = FontProperties(family="DejaVu Sans")
    fonts = theme.FontSet(font, font, font, font, font)
    monkeypatch.setattr(theme, "load_fonts", lambda: fonts)


@pytest.mark.parametrize("prediction", ["", '{"total":', '{"x":NaN}', '{"x":1,"x":2}'])
def test_invalid_json_card_keeps_reference(
    prediction: str, monkeypatch: pytest.MonkeyPatch, card_fonts: None
) -> None:
    from PIL import ImageDraw

    from vlm_exam.visualization import structured

    drawn: list[str] = []
    original = ImageDraw.ImageDraw.text

    def capture(self: Any, position: Any, text: str, *args: Any, **kwargs: Any) -> Any:
        drawn.append(text)
        return original(self, position, text, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", capture)
    panel = structured._panel({"total": "42.00"}, prediction)
    assert panel.size == (879, 604)
    assert "total" in drawn and '"42.00"' in drawn
    assert "Invalid JSON" in drawn if prediction else "Empty response" in drawn
    assert prediction in drawn if prediction else "(empty response)" in drawn


@pytest.mark.parametrize("score", [0.0, 0.8, 0.95, None])
def test_large_json_is_one_card_with_native_field_score(
    score: float | None, card_fonts: None
) -> None:
    from types import SimpleNamespace

    import matplotlib.pyplot as plt
    from PIL import Image

    from vlm_exam.config import DisplayConfig, ModelDisplay
    from vlm_exam.visualization import text as cards

    answer = {"rows": [[str(i + j) for j in range(10)] for i in range(62)]}
    config = DisplayConfig(
        labs={"lab": LabConfig("Lab", "#000000", "")},
        models={"test": ModelDisplay("Test", "lab")},
    )
    result = SimpleNamespace(predicted=json.dumps(answer), metadata={"score": score})
    figures = cards._json_cards(
        Image.new("RGB", (20, 20)),
        SimpleNamespace(answer=answer),
        result,
        "test",
        config,
    )
    try:
        assert len(figures) == 1
        rail = figures[0].axes[1]
        texts = {text.get_text(): text for text in rail.texts}
        value = f"{score * 100:.1f}%" if score is not None else "—"
        assert texts["FIELD F1"].get_position() == (0.5, 0.212)
        assert texts[value].get_position() == (0.5, 0.142)
        assert texts[value].get_color() == (
            cards.theme.score_color(score)
            if score is not None
            else cards.theme.TEXT_SECONDARY
        )
        assert len(rail.images) == 1
    finally:
        for figure in figures:
            plt.close(figure)


def test_json_fields_preserve_types_and_missing_extra_paths() -> None:
    from vlm_exam.visualization.structured import _fields, _path, _value

    fields = _fields(
        {"zero": 0, "null": None, "empty": [], "missing": "x"},
        {"zero": False, "null": "null", "empty": {}, "extra": True},
    )
    assert not any(field.matches for field in fields)
    assert [field.path for field in fields][-1] == ("extra",)
    assert _value(fields[3].predicted) == "[missing]"
    assert _value("null") != _value(None)
    assert _value("1.5") != _value(1.5)
    assert _path(("a.b",)) != _path(("a", "b"))


def test_json_map_is_conditional_and_prioritizes_differences(
    monkeypatch: pytest.MonkeyPatch, card_fonts: None
) -> None:
    from vlm_exam.visualization import structured

    drawn: list[Any] = []
    maps: list[Any] = []
    draw_tiles = structured._draw_tiles
    map_layout = structured._map_layout

    def tiles(draw: Any, items: list[Any]) -> None:
        drawn.extend(items)
        assert all(tile.y + tile.height <= 604 for tile in items)
        draw_tiles(draw, items)

    def cells(fields: list[Any], height: int) -> list[Any]:
        result = map_layout(fields, height)
        maps.extend(result)
        return result

    monkeypatch.setattr(structured, "_draw_tiles", tiles)
    monkeypatch.setattr(structured, "_map_layout", cells)
    structured._panel({"a": None}, '{"a":null}')
    assert not maps
    drawn.clear()
    expected = {"rows": [["ok"] * 10 for _ in range(62)]}
    predicted = json.loads(json.dumps(expected))
    predicted["rows"][-1][-1] = "wrong"
    structured._panel(expected, json.dumps(predicted))
    assert len(maps) == 620
    assert sum(not match for _, _, _, match in maps) == 1
    assert drawn[0].field.path == ("rows", 61, 9)
    assert len(drawn) < 620
    assert all(
        0 <= x < x + size <= 879 and 0 <= y < y + size <= 222 for x, y, size, _ in maps
    )


@pytest.mark.parametrize("fields", [1, 2, 4])
def test_long_json_strings_remain_within_card(
    fields: int, monkeypatch: pytest.MonkeyPatch, card_fonts: None
) -> None:
    from vlm_exam.visualization import structured

    original = structured._draw_tiles

    def check(draw: Any, tiles: list[Any]) -> None:
        assert tiles
        assert all(tile.y + tile.height <= 604 for tile in tiles)
        original(draw, tiles)

    monkeypatch.setattr(structured, "_draw_tiles", check)
    expected = {str(i): "abc " * 2000 for i in range(fields)}
    predicted = {str(i): "xyz " * 2000 for i in range(fields)}
    assert structured._panel(expected, json.dumps(predicted)).size == (879, 604)


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


def test_text_run_uses_shared_output_default(
    tmp_path: Path, stubbed_run: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
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
        ],
    )
    assert result.exit_code == 0, result.output
    assert len(list((tmp_path / "results").glob("text_*.jsonl"))) == 1
