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

"""Version and request metadata for the public leaderboard export."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def unknown_configurations(hashes: Iterable[str | None]) -> dict[str, Any]:
    """Describe recorded identities without guessing settings from today's registry."""
    counts = Counter(hashes)
    return {
        "status": "unknown",
        "record_count": sum(counts.values()),
        "variants": [
            {
                "id": identity,
                "record_count": count,
                "provider": None,
                "model_id": None,
                "max_output_tokens": {"status": "unknown", "value": None},
                "reasoning": {"kind": "unknown", "value": None, "api_parameters": {}},
                "evidence": "identity_only" if identity else "not_recorded",
            }
            for identity, count in sorted(
                counts.items(), key=lambda item: item[0] or ""
            )
        ],
    }


def profile_details(profile: dict[str, Any], arm: str) -> dict[str, Any]:
    """Describe a recorded V2 native request, not the public low/high lane."""
    provider = profile["provider"]
    value = profile.get(arm)
    params = {}
    kind = "unknown"
    if value is not None:
        kind = "level"
        if provider == "google":
            kind = "budget" if type(value) is int else "level"
            key = "thinking_budget" if kind == "budget" else "thinking_level"
            params = {f"thinking_config.{key}": value}
        elif provider == "anthropic":
            params = {"thinking.type": "adaptive", "output_config.effort": value}
        elif provider == "openrouter" and value in ("enabled", "disabled"):
            kind = "enabled"
            value = value == "enabled"
            params = {"reasoning.enabled": value}
        else:
            params = {"reasoning.effort": value}
    limit = profile.get("max_output_tokens")
    limit_status = (
        "unknown"
        if "max_output_tokens" not in profile
        else "omitted"
        if limit is None
        else "explicit"
    )
    result = {
        "provider": provider,
        "model_id": profile["wire"],
        "max_output_tokens": {"status": limit_status, "value": limit},
        "reasoning": {"kind": kind, "value": value, "api_parameters": params},
        "evidence": "recorded_request_profile",
    }
    routing = profile.get("routing")
    if routing is not None:
        result["routing"] = {
            "requested_provider_ids": routing.get("only"),
            "allow_fallbacks": routing.get("allow_fallbacks"),
        }
    return result


def detection_configurations(
    root: Path, policy: dict[str, Any], model: str, arm: str
) -> dict[str, Any]:
    """Count primary records per pinned request profile, including missing records."""
    from vlm_exam.detection_v2 import read_records

    entry = next(
        (r for r in policy["runs"] if (r["model"], r["effort"]) == (model, arm)), None
    )
    if entry is None:
        return unknown_configurations([])
    counts = Counter(row["configuration"] for row in read_records(root / entry["path"]))
    variants = [
        {
            "id": identity,
            "record_count": count,
            **profile_details(policy["configurations"][identity], arm),
        }
        for identity, count in sorted(counts.items())
    ]
    return {
        "status": "uniform" if len(variants) == 1 else "mixed",
        "record_count": sum(counts.values()),
        "variants": variants,
    }


def task_metadata(key: str, *, detection_v2: bool) -> dict[str, Any]:
    """Expose score composition without changing existing metric keys."""
    if key == "text":
        subtasks = {
            "single_string": "Single String Extraction",
            "transcription": "Text Transcription",
            "structured": "Structured Text Extraction",
            "localization_recognition": "Text Localization and Recognition",
        }
        return {
            "benchmark_id": "RF-OCRv2",
            "subtasks": [
                {"key": k, "name": name, "metric": k} for k, name in subtasks.items()
            ],
            "aggregation": {
                "metric": "overall",
                "components": list(subtasks),
                "within_run": "mean_sample_score",
                "across_runs": "mean_run_score",
                "incomplete_run_policy": "withhold_score",
            },
        }
    if key == "detection":
        return {
            "benchmark_id": "RF-ODv2" if detection_v2 else "RF-ODv1",
            "aggregation": {
                "map50": {
                    "within_run": "mean_subset_ap"
                    if detection_v2
                    else "pooled_dataset_ap",
                    "across_runs": "mean_run_score",
                    "max_detections": "uncapped" if detection_v2 else 100,
                },
                **(
                    {"f1": {"within_run": "micro_tp_fp_fn", "iou_threshold": 0.5}}
                    if detection_v2
                    else {}
                ),
            },
        }
    if key == "ocr":
        return {"benchmark_id": "RF-OCRv1"}
    return {}


def historical_detection(
    results: Path, effort: str | None, models: set[str] | None
) -> dict[str, Any]:
    """Load a verified frozen V1 view independently of current scores and settings."""
    source = results.resolve().parent / "datasets/detection/v1/web_snapshot.json"
    if not source.exists():
        return {}
    snapshot = json.loads(source.read_text())
    directory = results / "detection/v1"
    actual = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in directory.glob("*.jsonl")
    }
    if actual != snapshot["result_sha256"]:
        raise ValueError("Historical detection snapshot is stale: V1 archives changed")
    view = snapshot["view"]
    view.update(benchmark_id="RF-ODv1", name="RF-ODv1")
    for task in view["tasks"]:
        task.update(name="RF-ODv1", benchmark_id="RF-ODv1")
    view["models"] = [
        m
        for m in view["models"]
        if (effort is None or m["effort"] == effort)
        and (models is None or m["key"] in models)
    ]
    view["efforts"] = sorted({m["effort"] for m in view["models"]})
    return {"detection": {"v1": view}}


def write_historical_snapshot(
    results: Path, source: Path, destination: Path, source_revision: str
) -> None:
    """Freeze a verified V1 summary with hashes of the archives it describes.

    The source must be a V1 export scored against the original dataset, not a
    current V2 summary. This function does not recalculate historical scores.
    """
    payload = json.loads(source.read_text())
    definition = next(t for t in payload["tasks"] if t["key"] == "detection")
    if (
        definition.get("benchmark_id") == "detection-v2"
        or "v2" in definition["name"].lower()
    ):
        raise ValueError("Historical source must describe detection V1")
    groups: dict[tuple[str, str], list[str | None]] = {}
    hashes = {}
    for path in sorted((results / "detection/v1").glob("*.jsonl")):
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        for line in path.read_text().splitlines():
            row = json.loads(line)
            groups.setdefault((row["model"], row["effort"]), []).append(
                row.get("metadata", {}).get("inference_hash")
            )
    models = []
    for row in payload["models"]:
        if "detection" not in row["tasks"]:
            continue
        task = row["tasks"]["detection"]
        if task["primary_metric"] is None:
            raise ValueError("V1 snapshot requires scored detection results")
        task["configuration_details"] = unknown_configurations(
            groups[row["key"], row["effort"]]
        )
        models.append(
            {
                **{k: row[k] for k in ("id", "key", "name", "lab", "effort")},
                "tasks": {"detection": task},
            }
        )
    if set(groups) != {(m["key"], m["effort"]) for m in models}:
        raise ValueError("Historical summary model inventory differs from V1 archives")
    view = {
        "benchmark_id": "detection-v1",
        "name": "Object Detection — V1",
        "status": "archived",
        "generated_at": payload["generated_at"],
        "efforts": sorted({m["effort"] for m in models}),
        "scoring": {
            "source_revision": source_revision,
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        },
        "protocol": {
            "tasks": ["detection"],
            "repeats": payload["protocol"]["repeats"],
            "efforts": payload["protocol"]["efforts"],
        },
        "tasks": [{**definition, **task_metadata("detection", detection_v2=False)}],
        "models": models,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps({"result_sha256": hashes, "view": view}, indent=2) + "\n"
    )
