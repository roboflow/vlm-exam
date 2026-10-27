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
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from vlm_exam.tasks import create_task


def publish(directory: Path, totals: dict[str, int], models: list[str]) -> None:
    """Publish read-only checkpoint coverage without contacting any provider."""
    connection = sqlite3.connect(
        f"file:{directory / 'checkpoints.sqlite'}?mode=ro", uri=True
    )
    completed: Counter[tuple[str, str]] = Counter()
    errors: Counter[tuple[str, str]] = Counter()
    predictions: Counter[tuple[str, str]] = Counter()
    for identity, prediction, result, error in connection.execute(
        "SELECT identity, prediction IS NOT NULL, result IS NOT NULL, error "
        "FROM samples"
    ):
        model, task, *_ = identity.split(":")
        completed[(model, task)] += int(result)
        predictions[(model, task)] += int(prediction and not result)
        errors[(model, task)] += int(bool(error) and not result)
    connection.close()
    payload = {
        "updated": datetime.now(timezone.utc).isoformat(),
        "tasks": list(totals),
        "models": [
            {
                "model": model,
                "tasks": {
                    task: {
                        "completed": completed[(model, task)],
                        "total": total,
                        "outstanding_errors": errors[(model, task)],
                        "saved_predictions_pending_score": predictions[(model, task)],
                    }
                    for task, total in totals.items()
                },
            }
            for model in models
        ],
    }
    temporary = directory / "coverage.tmp"
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(directory / "coverage.json")


def main() -> None:
    """Publish task coverage once or continuously while a campaign is running."""
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    plan = json.loads((args.directory / "manifest.json").read_text())["plan"]
    totals = {
        task: len(
            create_task(task).load_samples(
                str(Path(plan["dataset_root"]) / task / "train")
            )
        )
        * repeats
        * 2
        for task, repeats in plan["tasks"].items()
    }
    print(
        "Publishing task coverage to " + str(args.directory / "coverage.json"),
        flush=True,
    )
    while True:
        publish(args.directory, totals, plan["models"])
        if not args.watch:
            break
        time.sleep(5)


if __name__ == "__main__":
    main()
