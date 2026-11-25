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
import json
from dataclasses import replace

import pytest

from vlm_exam.summary import (
    BenchmarkSummary,
    CostSummary,
    MetricValue,
    ModelOverall,
    ModelProtocolSummary,
    ModelSummary,
    ModelTaskResult,
    ProtocolSummary,
    SpeedSummary,
    TaskSummary,
    TokenSummary,
    summary_to_dict,
)
from vlm_exam.summary_metadata import (
    detection_configurations,
    historical_detection,
    profile_details,
    task_metadata,
    unknown_configurations,
    write_historical_snapshot,
)


def profile(**changes: object) -> dict:
    return {
        "provider": "google",
        "wire": "example",
        "minimum": "low",
        "maximum": "high",
        "max_output_tokens": 128000,
        **changes,
    }


@pytest.mark.parametrize(
    "provider,value,kind,parameters",
    [
        ("google", "high", "level", {"thinking_config.thinking_level": "high"}),
        ("google", 128, "budget", {"thinking_config.thinking_budget": 128}),
        ("openrouter", "disabled", "enabled", {"reasoning.enabled": False}),
        ("openrouter", "enabled", "enabled", {"reasoning.enabled": True}),
        (
            "anthropic",
            "max",
            "level",
            {"thinking.type": "adaptive", "output_config.effort": "max"},
        ),
        ("openai", "xhigh", "level", {"reasoning.effort": "xhigh"}),
    ],
)
def test_native_reasoning_parameters(provider, value, kind, parameters):
    details = profile_details(profile(provider=provider, maximum=value), "maximum")
    assert details["reasoning"]["kind"] == kind
    assert details["reasoning"]["api_parameters"] == parameters
    assert details["max_output_tokens"] == {"status": "explicit", "value": 128000}


def test_unknown_is_not_omitted_or_unlimited():
    omitted = profile_details(profile(max_output_tokens=None), "minimum")
    missing = profile()
    del missing["max_output_tokens"]
    assert omitted["max_output_tokens"] == {"status": "omitted", "value": None}
    assert (
        profile_details(missing, "minimum")["max_output_tokens"]["status"] == "unknown"
    )
    unknown = unknown_configurations(["old-hash", "old-hash", None])
    assert unknown["status"] == "unknown"
    assert unknown["record_count"] == 3
    assert all(v["reasoning"]["kind"] == "unknown" for v in unknown["variants"])


def test_mixed_profiles_count_primary_records_not_unique_images(tmp_path):
    rows = [{"configuration": "a"}] * 3 + [{"configuration": "b", "status": "missing"}]
    path = tmp_path / "records.gz"
    path.write_bytes(gzip.compress("\n".join(json.dumps(r) for r in rows).encode()))
    policy = {
        "runs": [{"model": "example", "effort": "minimum", "path": path.name}],
        "configurations": {"a": profile(), "b": profile(max_output_tokens=65536)},
    }
    details = detection_configurations(tmp_path, policy, "example", "minimum")
    assert details["status"] == "mixed"
    assert details["record_count"] == 4
    assert [v["record_count"] for v in details["variants"]] == [3, 1]


def task(details=None):
    return ModelTaskResult(
        primary_metric=MetricValue("map50", 70),
        metrics={"map50": 70, "f1": 75},
        metric_runs={"map50": (70,), "f1": (75,)},
        run_count=1,
        sample_count=2,
        failed_sample_count=0,
        tokens=TokenSummary(1, 2, 3, 1.5),
        cost=CostSummary(1, 0.5),
        speed=SpeedSummary(2, 1),
        timestamp="2026-01-01T00:00:00Z",
        timestamps=(),
        configuration_details=details,
    )


def test_per_task_settings_and_historical_are_isolated():
    detection = task(
        {"status": "uniform", "variants": [profile_details(profile(), "minimum")]}
    )
    text = task(
        {
            "status": "uniform",
            "variants": [profile_details(profile(minimum=128), "minimum")],
        }
    )
    model = ModelSummary(
        "example:low",
        "example",
        "Example",
        "google",
        "low",
        {"detection": detection, "text": text},
        ModelOverall(2, 4, detection.tokens, detection.cost, detection.speed),
        ModelProtocolSummary("full", "complete", 2, 2),
    )
    summary = BenchmarkSummary(
        None,
        ("low",),
        [TaskSummary("detection", "Detection", "map50", ())],
        [model],
        protocol=ProtocolSummary(3, ("low",), ("detection",), 1),
    )
    current = summary_to_dict(summary)
    combined = summary_to_dict(
        replace(summary, historical={"detection": {"v1": {"models": []}}})
    )
    assert current["models"] == combined["models"]
    tasks = combined["models"][0]["tasks"]
    assert (
        tasks["detection"]["configuration_details"]["variants"][0]["reasoning"]["value"]
        == "low"
    )
    assert (
        tasks["text"]["configuration_details"]["variants"][0]["reasoning"]["value"]
        == 128
    )
    assert tasks["detection"]["metrics"]["f1"] == 75
    assert combined["schema_version"] == 2
    assert combined["tasks"][0]["benchmark_id"] == "detection-v2"
    json.dumps(combined, allow_nan=False)


def test_text_composition_and_detection_aggregation():
    text = task_metadata("text", detection_v2=True)
    assert len(text["subtasks"]) == 4
    assert text["aggregation"]["components"] == [s["metric"] for s in text["subtasks"]]
    assert text["aggregation"]["within_run"] == "mean_sample_score"
    assert text["aggregation"]["incomplete_run_policy"] == "withhold_score"
    assert (
        task_metadata("detection", detection_v2=True)["aggregation"]["f1"]["within_run"]
        == "micro_tp_fp_fn"
    )
    assert (
        task_metadata("detection", detection_v2=False)["aggregation"]["map50"][
            "within_run"
        ]
        == "pooled_dataset_ap"
    )
    assert task_metadata("ocr", detection_v2=True) == {}


def test_historical_snapshot_filters_and_detects_modified_archives(tmp_path):
    results = tmp_path / "results"
    directory = results / "detection/v1"
    directory.mkdir(parents=True)
    archive = directory / "old.jsonl"
    archive.write_text(json.dumps({"model": "example", "effort": "low"}) + "\n")
    source = tmp_path / "old-summary.json"
    source.write_text(
        json.dumps(
            {
                "generated_at": "then",
                "protocol": {"repeats": 3, "efforts": ["low", "high"]},
                "tasks": [
                    {"key": "detection", "name": "Detection", "primary_metric": "map50"}
                ],
                "models": [
                    {
                        "id": "example:low",
                        "key": "example",
                        "name": "Example",
                        "lab": "google",
                        "effort": "low",
                        "overall": {"task_count": 6},
                        "tasks": {
                            "detection": {
                                "primary_metric": {"name": "map50", "value": 90}
                            },
                            "ocr": {},
                        },
                    }
                ],
            }
        )
    )
    target = tmp_path / "datasets/detection/v1/web_snapshot.json"
    write_historical_snapshot(results, source, target, "pinned-commit")
    view = historical_detection(results, None, None)["detection"]["v1"]
    assert "overall" not in view["models"][0]
    assert list(view["models"][0]["tasks"]) == ["detection"]
    assert (
        view["models"][0]["tasks"]["detection"]["configuration_details"]["status"]
        == "unknown"
    )
    assert (
        historical_detection(results, "high", None)["detection"]["v1"]["models"] == []
    )
    assert (
        historical_detection(results, None, {"other"})["detection"]["v1"]["models"]
        == []
    )
    archive.write_text(archive.read_text() + "\n")
    with pytest.raises(ValueError, match="stale"):
        historical_detection(results, None, None)
