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
def test_invalid_json_card_has_centered_failure_message(
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
    assert drawn == ["Could not parse JSON" if prediction else "No response"]
    background = panel.getpixel((0, 0))
    changed = [
        (x, y)
        for y in range(panel.height)
        for x in range(panel.width)
        if panel.getpixel((x, y)) != background
    ]
    assert (
        abs((min(x for x, _ in changed) + max(x for x, _ in changed)) / 2 - 439.5) <= 1
    )
    assert (
        abs((min(y for _, y in changed) + max(y for _, y in changed)) / 2 - 301.5) <= 1
    )


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


@pytest.mark.parametrize("expected", [{}, [], {"long": "x" * 20000}])
def test_json_hybrid_small_values_omit_map(
    expected: Any, monkeypatch: pytest.MonkeyPatch, card_fonts: None
) -> None:
    from vlm_exam.visualization import structured

    labels: list[str] = []
    original = structured._header

    def header(draw: Any, title: str, y: int, *args: Any) -> None:
        labels.append(title)
        original(draw, title, y, *args)

    monkeypatch.setattr(structured, "_header", header)
    structured._panel(expected, json.dumps(expected))
    assert labels == ["FIELD VALUES"]


def test_json_hybrid_map_counts_and_full_width(
    monkeypatch: pytest.MonkeyPatch, card_fonts: None
) -> None:
    from vlm_exam.visualization import structured

    headers: list[tuple[Any, ...]] = []
    rectangles: list[tuple[int, ...]] = []
    original = structured._header
    from PIL import ImageDraw

    rectangle = ImageDraw.ImageDraw.rectangle

    def header(draw: Any, title: str, y: int, *args: Any) -> None:
        headers.append((title, *args))
        original(draw, title, y, *args)

    def record(draw: Any, box: Any, *args: Any, **kwargs: Any) -> None:
        if kwargs.get("fill") in (structured._MAP_MATCH, structured._MAP_CHANGED):
            rectangles.append(box)
        rectangle(draw, box, *args, **kwargs)

    monkeypatch.setattr(structured, "_header", header)
    monkeypatch.setattr(ImageDraw.ImageDraw, "rectangle", record)
    expected = {"rows": list(range(620))}
    predicted = {"rows": list(range(619)) + [-1]}
    structured._panel(expected, json.dumps(predicted))
    assert headers == [("JSON MAP", 620, 1), ("SELECTED VALUES",)]
    assert min(box[0] for box in rectangles) == 0
    assert max(box[2] for box in rectangles) == 878


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


def test_json_wrapping_keeps_the_measured_font(
    monkeypatch: pytest.MonkeyPatch, card_fonts: None
) -> None:
    from matplotlib import font_manager
    from PIL import ImageDraw, ImageFont

    from vlm_exam.visualization import structured

    primary = font_manager.findfont("DejaVu Sans Mono")
    fallback = font_manager.findfont("DejaVu Sans")

    def choose(value: str, size: int) -> Any:
        return ImageFont.truetype(fallback if "日本語" in value else primary, size)

    original = ImageDraw.ImageDraw.text
    widths: list[float] = []

    def capture(self: Any, position: Any, text: str, *args: Any, **kwargs: Any) -> Any:
        font = kwargs["font"]
        if font.size == 21:
            widths.append(font.getlength(text))
            assert font.path == fallback
        return original(self, position, text, *args, **kwargs)

    monkeypatch.setattr(structured, "_text_font", choose)
    monkeypatch.setattr(ImageDraw.ImageDraw, "text", capture)
    answer = {"value": "I" * 100 + "日本語"}
    structured._panel(answer, json.dumps(answer))
    assert len(widths) > 1
    assert max(widths) <= 402


def test_json_fallback_preserves_supported_text(
    monkeypatch: pytest.MonkeyPatch, card_fonts: None
) -> None:
    from matplotlib.font_manager import FontEntry

    from vlm_exam.visualization import structured

    monkeypatch.setattr(
        structured.font_manager.fontManager,
        "ttflist",
        [
            FontEntry(fname="placeholder", name="LastResortHE"),
            FontEntry(fname="cjk", name="Noto Sans CJK"),
        ],
    )
    structured._fallbacks.cache_clear()
    try:
        assert structured._fallbacks() == ("cjk",)
        monkeypatch.setattr(
            structured.font_manager, "findfont", lambda *args: "primary"
        )
        coverage = {
            "primary": frozenset(map(ord, "Invoice paid ")),
            "cjk": frozenset(map(ord, "日本語")),
        }
        monkeypatch.setattr(structured, "_coverage", coverage.__getitem__)
        monkeypatch.setattr(structured, "_font", lambda path, size: path)
        assert structured._text_font("Invoice paid 🚀", 21) == "primary"
        assert structured._text_font("日本語😀", 21) == "cjk"
    finally:
        structured._fallbacks.cache_clear()


def test_deep_json_and_json_like_strings_stay_bounded(card_fonts: None) -> None:
    from vlm_exam.visualization import structured

    nested: Any = 1
    for _ in range(2000):
        nested = [nested]
    fields = structured._fields({"a": 1}, {"a": nested})
    assert fields[-1].path == ("a",) + (0,) * 2000
    for response in (
        '{"a":' + "[" * 1000 + "1" + "]" * 1000 + "}",
        json.dumps({"a": "[" * 10000 + "1" + "]" * 10000}),
    ):
        assert structured._panel({"a": "ok"}, response).size == (879, 604)


@pytest.mark.parametrize("content", ["ELL\nJAIL\nVOW", "a\n\nb", "日本語\nInvoice"])
def test_transcription_preserves_line_breaks(
    content: str, monkeypatch: pytest.MonkeyPatch, card_fonts: None
) -> None:
    from PIL import ImageDraw

    from vlm_exam.visualization import transcription

    positions = []
    original = ImageDraw.ImageDraw.text

    def capture(self: Any, position: Any, value: str, *args: Any, **kwargs: Any) -> Any:
        if value in content.split("\n") and value:
            positions.append(position)
        return original(self, position, value, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", capture)
    transcription._panel(content, content)
    assert [pos[1] for pos in positions] == [
        36 + index * 23 for index, line in enumerate(content.split("\n")) if line
    ]
    assert all(pos[0] == 0 for pos in positions)


def test_transcription_overflow_is_bounded(
    card_fonts: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PIL import ImageDraw

    from vlm_exam.visualization import transcription

    drawn = []
    original = ImageDraw.ImageDraw.text

    def capture(self: Any, position: Any, value: str, *args: Any, **kwargs: Any) -> Any:
        drawn.append((position, value))
        return original(self, position, value, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", capture)
    image = transcription._panel("word\n" * 1000, "word\n" * 1000)
    assert image.size == (879, 604)
    assert drawn[-1][1] == "…"
    assert len(drawn) < 40
    assert all(0 <= position[1] < 604 for position, _ in drawn)
    assert not any("MAP" in value for _, value in drawn)


@pytest.mark.parametrize("same", [True, False])
def test_single_value_comparison_is_bounded(
    same: bool, monkeypatch: pytest.MonkeyPatch, card_fonts: None
) -> None:
    from PIL import ImageDraw

    from vlm_exam.visualization import single_value

    drawn = []
    original = ImageDraw.ImageDraw.text

    def capture(self: Any, position: Any, value: str, *args: Any, **kwargs: Any) -> Any:
        drawn.append((position, value, kwargs.get("font")))
        return original(self, position, value, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", capture)
    expected = "ABC" * 2000
    predicted = expected if same else "XYZ" * 2000
    panel = single_value._panel("Read " * 1000, expected, predicted)
    assert panel.size == (879, 604)
    assert any("…" in value for _, value, _ in drawn)
    assert len(drawn) < 20
    assert all(0 <= position[1] < 580 for position, _, _ in drawn)
    assert all(
        position[0] + font.getlength(value) <= 880 for position, value, font in drawn
    )


def test_single_value_question_wraps_at_words(card_fonts: None) -> None:
    from vlm_exam.visualization import single_value

    question = (
        "Read the letters and numbers imprinted on the pill in this image. "
        "Return only the text, preserving capitalization and separating lines "
        "with a single space. Do not explain."
    )
    assert " ".join(single_value._question_lines(question)) == question
    assert len(single_value._question_lines("X" * 1000)) > 1
