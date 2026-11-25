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

import gzip
import hashlib
import json
from pathlib import Path

import pytest

from tests.test_detection import _publication_release
from tests.test_summary import _config, _release
from vlm_exam.config import load_config
from vlm_exam.detection_v2 import load_release, read_records, validate_release
from vlm_exam.detection_v2_identity import canonical_model_id, release_model_id
from vlm_exam.detection_v2_publication import publish_recoveries
from vlm_exam.detection_v2_workflow import detection_charts, resolve_detection_profiles
from vlm_exam.summary import build_summary

PUBLIC = "gemini-4-hydrogen"
ARCHIVE = "gemini-4-hydrogen-eap"


def test_both_names_resolve_existing_profiles():
    policy = {
        "runs": [{"model": ARCHIVE, "effort": "minimum", "configurations": ["p"]}],
        "configurations": {
            "p": {
                "provider": "google",
                "wire": PUBLIC,
                "minimum": "low",
                "max_output_tokens": 128000,
            }
        },
    }
    for name in (PUBLIC, ARCHIVE):
        assert canonical_model_id(name) == PUBLIC
        assert resolve_detection_profiles(name, "minimum", policy, load_config()) == [
            (ARCHIVE, "p", None)
        ]


def test_conflicting_alias_inventory_is_rejected():
    with pytest.raises(ValueError, match="Conflicting release identities"):
        release_model_id({"runs": [{"model": PUBLIC}, {"model": ARCHIVE}]}, PUBLIC)


def test_publication_reuses_historical_identity_and_preserves_input(tmp_path):
    row = _publication_release(tmp_path)
    policy, _ = load_release(tmp_path)
    entry = policy["runs"][0]
    archive = tmp_path / entry["path"]
    previous = read_records(archive)
    previous[0]["model"] = ARCHIVE
    archive.write_bytes(gzip.compress((json.dumps(previous[0]) + "\n").encode()))
    entry.update(model=ARCHIVE, sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    policy["missing"] = [f"{ARCHIVE}/minimum/a"]
    (tmp_path / "datasets/detection/v2/release.json").write_text(json.dumps(policy))
    row["model"] = PUBLIC
    publish_recoveries(tmp_path, [row], {}, "Alias recovery")
    updated, _ = load_release(tmp_path)
    assert len(updated["runs"]) == 1
    assert updated["runs"][0]["model"] == ARCHIVE
    published = read_records(archive)[0]
    assert published["submitted_model_key"] == PUBLIC
    assert published["model"] == ARCHIVE
    assert published["wire_model"] == row["wire_model"]
    assert row["model"] == PUBLIC
    assert validate_release(tmp_path) == []


def hydrogen_summary(root: Path) -> None:
    _release(root)
    directory = root / "datasets/detection/v2"
    for name in ("release.json", "summary.json", "model_labels.json"):
        p = directory / name
        p.write_text(p.read_text().replace('"alpha"', json.dumps(ARCHIVE)))


def test_coverage_counts_historical_lane_under_public_key(tmp_path):
    (tmp_path / "results").mkdir()
    hydrogen_summary(tmp_path)
    result = build_summary(tmp_path / "results", _config(PUBLIC))
    assert len(result.models) == 1
    model = result.models[0]
    assert model.key == PUBLIC
    assert model.protocol.runs_present == 1
    assert model.tasks["detection"].metrics["map50"] == 42


@pytest.mark.parametrize("name", [PUBLIC, ARCHIVE])
def test_chart_filter_accepts_either_alias(tmp_path, monkeypatch, name):
    import matplotlib.pyplot as plt

    from vlm_exam.visualization import charts

    hydrogen_summary(tmp_path)
    captured = []
    monkeypatch.setattr(
        charts,
        "plot_accuracy_chart",
        lambda scores, *args: captured.append(scores) or plt.figure(),
    )
    monkeypatch.setattr(charts, "save_leaderboard_chart", lambda *args: None)
    detection_charts(tmp_path, tmp_path / "charts", {name})
    assert captured == [{ARCHIVE: 42.0}]


def test_duplicate_aliases_do_not_launch_two_runs():
    import click

    with pytest.raises(click.UsageError, match="Duplicate model identities"):
        resolve_detection_profiles(
            f"{PUBLIC},{ARCHIVE}", "minimum", {"runs": []}, load_config()
        )
