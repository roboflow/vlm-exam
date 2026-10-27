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

import argparse
import html
import json
from datetime import datetime, timezone
from pathlib import Path

from vlm_exam.config import load_config
from vlm_exam.metrics import group_runs
from vlm_exam.results import is_failed_sample, load_results_directory
from vlm_exam.summary import (
    _load_detection_index,
    _model_task_result,
    _task_result_dict,
    _text_result,
)
from vlm_exam.tasks import create_task
from vlm_exam.text_release import load_text_release_policy


def main() -> None:
    """Publish standard metrics for complete campaign configurations offline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    plan = json.loads((args.directory / "manifest.json").read_text())["plan"]
    config = load_config(Path(plan["config"]))
    groups = group_runs(
        load_results_directory(Path(plan["output_directory"])),
        config,
        models=set(plan["models"]),
    )
    detection_index = _load_detection_index(
        Path(plan["dataset_root"]) / "detection/train"
    )
    policy = load_text_release_policy()
    expected = {
        task: len(
            create_task(task).load_samples(
                str(Path(plan["dataset_root"]) / task / "train")
            )
        )
        for task in plan["tasks"]
    }
    rows = []
    for model in plan["models"]:
        for effort in ("low", "high"):
            for task, repeats in plan["tasks"].items():
                runs = groups.get((task, effort, model), [])
                complete = bool(
                    len(runs) == repeats
                    and all(
                        len(run.samples) == expected[task]
                        and len({sample.index for sample in run.samples})
                        == expected[task]
                        and not any(is_failed_sample(s) for s in run.samples)
                        for run in runs
                    )
                )
                result = None
                if complete:
                    print(
                        f"Scoring {model} {effort} {task} ({repeats} runs)", flush=True
                    )
                    scored = (
                        _text_result(runs, config.models[model], policy, [])
                        if task == "text"
                        else _model_task_result(
                            runs, config.models[model], detection_index
                        )
                    )
                    result = _task_result_dict(scored)
                    assert result["primary_metric"] is not None
                    if task in {"detection", "text"}:
                        assert result["evaluated_sample_count"] == expected[task]
                rows.append(
                    {
                        "model": model,
                        "name": config.models[model].name,
                        "effort": effort,
                        "task": task,
                        "status": "complete" if complete else "pending",
                        "runs_present": len(runs),
                        "runs_required": repeats,
                        "samples_per_run": expected[task],
                        "result": result if complete else None,
                    }
                )
    updated = datetime.now(timezone.utc).isoformat()
    notes = (
        "Scores are 0–100. Original tasks average three complete runs; Text uses "
        "one complete 600-pair run. Pending configurations have no published score. "
        "Cost is the repository's token-price estimate for one run, averaged over "
        "repeats; it excludes judge charges, retries and cache discounts. Latency "
        "is mean inference seconds per sample, not concurrent campaign wall time."
    )
    payload = {"updated": updated, "notes": notes, "rows": rows}
    (args.directory / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    table = []
    for row in rows:
        result = row["result"]
        primary = result["primary_metric"] if result else None
        scores = (
            "; ".join(f"{key}: {value:.2f}" for key, value in result["metrics"].items())
            if result
            else "Pending"
        )
        cells = [
            row["name"],
            row["effort"].title(),
            row["task"],
            f"{row['runs_present']}/{row['runs_required']}",
            f"{primary['value']:.2f}" if primary else "—",
            scores,
            f"${result['cost']['total_usd']:.4f}" if result else "—",
            f"{result['speed']['average_seconds_per_sample']:.2f}" if result else "—",
        ]
        table.append(
            "<tr>" + "".join(f"<td>{html.escape(v)}</td>" for v in cells) + "</tr>"
        )
    completed = sum(row["status"] == "complete" for row in rows)
    document = (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>Completed benchmark metrics</title><style>"
        "body{font:15px system-ui;background:#101723;color:#e8edf5;margin:28px}"
        "a{color:#8bc9ff}table{border-collapse:collapse;width:100%}"
        "td,th{padding:10px;text-align:left;border-bottom:1px solid #314057}"
        "th{position:sticky;top:0;background:#1c293b}td:nth-child(6){max-width:460px}"
        "</style><h1>Completed benchmark metrics</h1>"
        '<p><a href="dashboard.html">Campaign dashboard</a> · '
        '<a href="metrics.json">Download full metrics and per-run scores (JSON)</a></p>'
        f"<p>{completed}/{len(rows)} configurations complete · Snapshot: {updated}</p>"
        f"<p>{html.escape(notes)}</p><table><thead><tr>"
        "<th>Model</th><th>Effort</th><th>Task</th><th>Runs</th><th>Score</th>"
        "<th>All metrics (%)</th><th>Estimated $/run</th><th>Seconds/sample</th>"
        "</tr></thead><tbody>" + "".join(table) + "</tbody></table></html>"
    )
    (args.directory / "metrics.html").write_text(document)
    print(
        f"Published {completed}/{len(rows)} completed configurations "
        f"to {args.directory}/metrics.html"
    )


if __name__ == "__main__":
    main()
