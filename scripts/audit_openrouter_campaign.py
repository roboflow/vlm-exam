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
from collections import Counter
from pathlib import Path
from typing import Any

from vlm_exam.tasks import create_task


def audit(
    directory: Path, complete: bool = False, recompute_text: bool = False
) -> dict[str, Any]:
    """Audit saved campaign results and billing without making API requests.

    Args:
        directory: Campaign directory containing its manifest and SQLite store.
        complete: Require every planned sample to have a valid final result.
        recompute_text: Recompute deterministic Text scores from saved responses.

    Returns:
        Counts, reported charges and any violated provenance invariants.
    """
    manifest = json.loads((directory / "manifest.json").read_text())
    plan = manifest["plan"]
    expected_providers = {
        "together": "together",
        "minimax": "minimax",
        "moonshotai/mxfp4": "moonshot ai",
        "crusoe/bf16": "crusoe",
    }
    expected = set()
    references = {}
    tasks = {}
    for task, repeats in plan["tasks"].items():
        samples = create_task(task).load_samples(
            str(Path(plan["dataset_root"]) / task / "train")
        )
        references[task] = samples
        for model in plan["models"]:
            arguments = {}
            if task in {"detection", "text"}:
                arguments["coordinate_format"] = manifest["models"][model][
                    "detection_coordinate_format"
                ]
            tasks[(model, task)] = create_task(task, **arguments)
            for effort in ("low", "high"):
                for repeat in range(repeats):
                    for index in range(len(samples)):
                        expected.add(f"{model}:{task}:{effort}:{repeat}:{index}")
    connection = sqlite3.connect(
        f"file:{directory / 'checkpoints.sqlite'}?mode=ro", uri=True
    )
    connection.execute("BEGIN")
    completed = set()
    generations = set()
    errors = []
    counts: Counter[str] = Counter()
    recomputed_text = 0
    for identity, payload in connection.execute(
        "SELECT identity,result FROM samples WHERE result IS NOT NULL"
    ):
        row = json.loads(payload)
        completed.add(identity)
        model, task, effort, repeat, index = identity.split(":")
        counts[model] += 1
        route = manifest["models"][model]["routes"][0]
        usage = row["metadata"]["usage"]
        direct = route["provider"] == "openai"
        provider = (
            "openai"
            if direct
            else expected_providers[route["options"]["provider"]["only"][0]]
        )
        checks = {
            "planned identity": identity in expected,
            "provider": usage["provider"].lower() == provider,
            "model": usage.get("requested_model", usage["model"])
            == route["provider_model_id"],
            "normal completion": usage["finish_reason"] == "stop",
            "nonempty prediction": bool(row["predicted"].strip()),
            "input tokens": row["input_tokens"] == usage["reported"]["prompt_tokens"],
            "output tokens": row["output_tokens"]
            == usage["reported"]["completion_tokens"],
            "generation uniqueness": usage["generation_id"] not in generations,
            "image identity": row["image"]
            == Path(references[task][int(index)].image_path).name,
            "reported cost": isinstance(usage["reported"].get("cost"), (int, float)),
        }
        if direct:
            checks["direct API endpoint"] = (
                usage.get("api_endpoint") == "https://api.openai.com/v1/responses"
            )
            checks["native response completed"] = (
                usage.get("response_status") == "completed"
            )
            checks["native cost labeled estimate"] = (
                usage.get("cost_source") == "conservative_token_estimate"
            )
        sample = references[task][int(index)]
        uploaded_size = (
            row["metadata"]["uploaded_width"],
            row["metadata"]["uploaded_height"],
        )
        checks["frozen prompt"] = row["metadata"]["resolved_prompt"] == tasks[
            (model, task)
        ].build_prompt(sample, uploaded_size=uploaded_size)
        checks["reference answer"] = row["expected"] == tasks[
            (model, task)
        ].expected_text(sample)
        if task in {"detection", "text"}:
            checks["frozen coordinate format"] = (
                row["metadata"]["coordinate_format"]
                == manifest["models"][model]["detection_coordinate_format"]
            )
        if task == "text" and recompute_text:
            evaluation = tasks[(model, task)].evaluate(
                sample, row["predicted"], uploaded_size=uploaded_size
            )
            checks["recomputed text score"] = (
                evaluation.score == row["metadata"]["score"]
            )
            checks["recomputed text verdict"] = evaluation.correct == row["correct"]
            recomputed_text += 1
        generations.add(usage["generation_id"])
        if task in {"extraction", "counting", "identification", "reasoning"}:
            checks["judge and strict verdicts"] = all(
                isinstance(row["metadata"].get(key), bool)
                for key in ("strict_correct", "judge_correct")
            )
            checks["headline verdict"] = (
                row["correct"] == row["metadata"]["judge_correct"]
            )
        errors.extend(
            {"sample": identity, "check": key}
            for key, valid in checks.items()
            if not valid
        )
    charges = [
        json.loads(row[0]) for row in connection.execute("SELECT payload FROM charges")
    ]
    connection.close()
    missing = expected - completed
    if complete and missing:
        errors.append({"check": "complete inventory", "missing": len(missing)})
    return {
        "complete": not missing,
        "completed": len(completed),
        "expected": len(expected),
        "by_model": dict(counts),
        "reported_charges_usd": sum(
            row.get("reported", {}).get("cost", 0) or 0
            for row in charges
            if row.get("cost_source") != "conservative_token_estimate"
        ),
        "conservative_estimated_charges_usd": sum(
            row.get("reported", {}).get("cost", 0) or 0
            for row in charges
            if row.get("cost_source") == "conservative_token_estimate"
        ),
        "unique_billed_generations": len(charges),
        "recomputed_text_samples": recomputed_text,
        "errors": errors,
    }


def main() -> None:
    """Validate a live snapshot or a completed campaign without modifying it."""
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--complete", action="store_true")
    parser.add_argument("--recompute-text", action="store_true")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = audit(arguments.directory, arguments.complete, arguments.recompute_text)
    encoded = json.dumps(result, indent=2)
    if arguments.output:
        arguments.output.write_text(encoded + "\n")
    print(encoded)
    raise SystemExit(bool(result["errors"]))


if __name__ == "__main__":
    main()
