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
import json
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from vlm_exam.results import RunResult, SampleResult, save_results
from vlm_exam.tasks import create_task


def publish(directory: Path, counts: dict[str, int]) -> int:
    """Export only complete checkpoint groups without mutating the live database."""
    manifest = json.loads((directory / "manifest.json").read_text())
    plan = manifest["plan"]
    connection = sqlite3.connect(
        f"file:{directory / 'checkpoints.sqlite'}?mode=ro", uri=True
    )
    connection.execute("BEGIN")
    assert (
        json.loads(connection.execute("SELECT payload FROM manifest").fetchone()[0])
        == manifest
    )
    saved = dict(
        connection.execute(
            "SELECT identity, result FROM samples WHERE result IS NOT NULL"
        )
    )
    connection.close()
    exported = 0
    for task, repeats in plan["tasks"].items():
        for model in plan["models"]:
            for effort in ("low", "high"):
                for repeat in range(repeats):
                    timestamp = (
                        datetime.strptime(plan["timestamp"], "%Y%m%d_%H%M%S")
                        + timedelta(seconds=repeat)
                    ).strftime("%Y%m%d_%H%M%S")
                    path = (
                        Path(plan["output_directory"])
                        / f"{task}_{model}_{effort}_{timestamp}.jsonl"
                    )
                    if path.exists():
                        continue
                    rows = [
                        saved.get(f"{model}:{task}:{effort}:{repeat}:{i}")
                        for i in range(counts[task])
                    ]
                    if not all(rows):
                        continue
                    result = RunResult(
                        model=model,
                        task=task,
                        effort=effort,
                        timestamp=timestamp,
                        samples=[SampleResult(**json.loads(row)) for row in rows],
                    )
                    temporary = directory / (path.name + ".export")
                    save_results(result, temporary)
                    if path.exists():
                        assert path.read_bytes() == temporary.read_bytes()
                        temporary.unlink()
                    else:
                        temporary.replace(path)
                        exported += 1
                        print(f"Exported {path}", flush=True)
    return exported


def main() -> None:
    """Watch completed runs and refresh the dashboard using standard metrics."""
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    plan = json.loads((args.directory / "manifest.json").read_text())["plan"]
    counts = {
        task: len(
            create_task(task).load_samples(
                str(Path(plan["dataset_root"]) / task / "train")
            )
        )
        for task in plan["tasks"]
    }
    previous = None
    while True:
        publish(args.directory, counts)
        inventory = tuple(
            (path.name, path.stat().st_mtime_ns, path.stat().st_size)
            for path in sorted(Path(plan["output_directory"]).glob("*.jsonl"))
        )
        if inventory != previous:
            subprocess.run(
                [sys.executable, "scripts/campaign_metrics.py", str(args.directory)],
                check=True,
            )
            previous = inventory
        if not args.watch:
            break
        time.sleep(45)


if __name__ == "__main__":
    main()
