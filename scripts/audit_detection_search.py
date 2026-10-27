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

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def audit(directory: Path, complete: bool = False) -> dict[str, Any]:
    """Check frozen search jobs and response provenance without API calls.

    Args:
        directory: Search directory with manifest and saved responses.
        complete: Require every matrix job to have a saved response.

    Returns:
        Coverage, provider and billing checks, including failed responses.
    """
    manifest = json.loads((directory / "matrix-manifest.json").read_text())
    jobs = {job["id"]: job for job in manifest["jobs"]}
    providers = {
        "openai": "openai",
        "together": "together",
        "minimax": "minimax",
        "moonshotai/mxfp4": "moonshot ai",
        "crusoe/bf16": "crusoe",
    }
    model_ids = {
        "gpt-6.1-terra": "openai/gpt-6.1-terra",
        "muse-spark-open": "meta/muse-spark-open",
        "deepseek-v4.1-flash-1017": "deepseek/deepseek-v4.1-flash-1017",
        "minimax-m3.1-flash": "minimax/minimax-m3.1-flash",
        "kimi-k3.1-pro": "moonshotai/kimi-k3.1-pro",
        "nemotron-3.5-super-vl": "nvidia/nemotron-3.5-super-vl",
    }
    errors = []
    generations = set()
    coverage: Counter[str] = Counter()
    failures: Counter[str] = Counter()
    costs: Counter[str] = Counter()
    seen = set()
    for path in sorted((directory / "responses").glob("*.json")):
        row = json.loads(path.read_text())
        if row["format"] == "native":
            continue
        identity = row["id"]
        if identity not in jobs:
            errors.append(f"Unknown job: {identity}")
            continue
        job = jobs[identity]
        seen.add(identity)
        for key, expected in job.items():
            if row.get(key) != expected:
                errors.append(f"Changed {key}: {identity}")
        group = f"{row['model']}:{row['format']}:{row['effort']}"
        coverage[group] += 1
        metadata = row.get("usage", {}).get("metadata", {})
        if row.get("error"):
            failures["api"] += 1
        else:
            route = job["options"]["provider"]["only"][0]
            if str(metadata.get("provider", "")).casefold() != providers[route]:
                errors.append(f"Unexpected provider: {identity}")
            if metadata.get("model") != model_ids[row["model"]]:
                errors.append(f"Unexpected model ID: {identity}")
            generation = metadata.get("generation_id")
            if not generation or generation in generations:
                errors.append(f"Missing or duplicate generation: {identity}")
            generations.add(generation)
            if metadata.get("finish_reason") == "length":
                failures["truncated"] += 1
            if (
                not row.get("answer", "").strip()
                or row.get("answer") == "[model returned no content]"
            ):
                failures["empty"] += 1
            cost = metadata.get("reported", {}).get("cost")
            if cost is None:
                errors.append(f"Missing reported cost: {identity}")
            else:
                costs[row["model"]] += cost
    for name, digest in manifest["source_sha256"].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            errors.append(f"Changed source: {name}")
    if complete and failures["api"]:
        errors.append(f"Unresolved API failures: {failures['api']}")
    if complete and seen != set(jobs):
        errors.append(f"Missing {len(set(jobs) - seen)} matrix responses")
    if len(jobs) != len(manifest["jobs"]):
        errors.append("Duplicate manifest jobs")
    for group, count in Counter(
        f"{j['model']}:{j['format']}:{j['effort']}" for j in jobs.values()
    ).items():
        if count != 50:
            errors.append(f"Expected 50 images: {group} has {count}")
    return {
        "checked": len(seen),
        "planned": len(jobs),
        "complete": seen == set(jobs),
        "errors": errors,
        "response_failures": dict(failures),
        "coverage": dict(coverage),
        "model_cost_usd": dict(costs),
    }


def main() -> None:
    """Write a partial or complete audit report for a format search."""
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--complete", action="store_true")
    args = parser.parse_args()
    result = audit(args.directory, args.complete)
    (args.directory / "audit.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if result["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
