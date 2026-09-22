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
from unittest.mock import Mock

import pytest
from click.testing import CliRunner

from vlm_exam import cli, visualization
from vlm_exam.results import RunResult
from vlm_exam.tasks import detection
from vlm_exam.tasks.detection import DatasetMapResult


@pytest.mark.parametrize("has_dataset", [True, False])
@pytest.mark.parametrize("all_unmatched", [True, False])
def test_detection_leaderboard_evaluates_each_repeat_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    has_dataset: bool,
    all_unmatched: bool,
) -> None:
    model = next(iter(cli.load_config().models))
    # Equal timestamps must not collapse distinct repeats.
    runs = [
        RunResult(model=model, task="detection", effort=effort, timestamp="same")
        for effort in ("low", "low", "low", "high")
    ]
    monkeypatch.setattr(cli, "load_results_directory", lambda path: runs)
    load_samples = Mock(return_value=[])
    monkeypatch.setattr(detection.DetectionTask, "load_samples", load_samples)
    scores = {
        id(runs[0]): DatasetMapResult(0.2, 0.1, 0.05, 1),
        id(runs[1]): DatasetMapResult(0.8, 0.5, 0.35, 1),
        id(runs[2]): None,
        id(runs[3]): DatasetMapResult(0.9, 0.7, 0.6, 1),
    }
    if all_unmatched:
        scores = {key: None for key in scores}
    compute = Mock(side_effect=lambda run, index: scores[id(run)])
    monkeypatch.setattr(detection, "compute_dataset_map", compute)
    figures = []

    def plot(*args: object, **kwargs: object) -> Mock:
        figure = Mock()
        figures.append(figure)
        return figure

    chart = Mock(side_effect=plot)
    monkeypatch.setattr(visualization, "plot_metric_chart", chart)
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "close", Mock())
    arguments = [
        "leaderboard",
        "--results-directory",
        str(tmp_path),
        "--output-directory",
        str(tmp_path / "charts"),
    ]
    if has_dataset:
        arguments += ["--dataset-directory", str(tmp_path)]
    result = CliRunner().invoke(cli.main, arguments)
    assert result.exit_code == 0, result.output
    if not has_dataset:
        compute.assert_not_called()
        chart.assert_not_called()
        load_samples.assert_not_called()
        assert "skipping detection leaderboards" in result.output
        return

    assert compute.call_count == len(runs)
    assert {id(call.args[0]) for call in compute.call_args_list} == set(scores)
    if all_unmatched:
        assert result.output.count("skipping that run") == len(runs)
        chart.assert_not_called()
        assert "No leaderboard charts generated" in result.output
        return
    assert result.output.count("skipping that run") == 1
    load_samples.assert_called_once()
    assert chart.call_count == 6
    expected = {
        "high": [(0.9, None), (0.7, None), (0.6, None)],
        "low": [(0.5, (0.2, 0.8)), (0.3, (0.1, 0.5)), (0.2, (0.05, 0.35))],
    }
    for effort_index, effort in enumerate(("high", "low")):
        for metric_index, metric in enumerate(("map50", "map75", "map50_95")):
            index = effort_index * 3 + metric_index
            call = chart.call_args_list[index]
            mean, spread = expected[effort][metric_index]
            assert call.args[0] == {model: pytest.approx(mean)}
            assert call.kwargs["spread"] == ({model: spread} if spread else {})
            assert call.kwargs["run_counts"] == {model: 2 if spread else 1}
            figures[index].savefig.assert_called_once_with(
                str(tmp_path / "charts" / f"detection_{metric}_{effort}.png"), dpi=150
            )
