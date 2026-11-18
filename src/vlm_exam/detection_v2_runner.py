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

import fcntl
import gzip
import hashlib
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from vlm_exam.detection_v2 import (
    _summarize_run,
    load_release,
    response_status,
    validate_release,
)
from vlm_exam.providers.base import call_with_retries


def _write(path: Path, value: Any) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, allow_nan=False) + "\n")
    temporary.replace(path)


def _inference_fingerprint() -> str:
    package = Path(__file__).parent
    sources = [
        "detection_v2_inference.py",
        "detection_v2_upload.py",
        "tasks/detection.py",
        "providers/image_upload.py",
        "providers/anthropic.py",
        "providers/openai.py",
    ]
    digest = hashlib.sha256()
    for name in sources:
        digest.update(name.encode())
        digest.update((package / name).read_bytes())
    return digest.hexdigest()


def run_release(
    root: Path,
    images: Path,
    model: str,
    effort: str,
    configuration: str,
    output: Path,
    concurrency: int,
) -> None:
    """Evaluate or resume a full release using one explicit native request profile.

    Args:
        root: Repository root containing the frozen release.
        images: Hash-verified original images prepared by prepare-images.
        model: Model identity recorded in this release.
        effort: Minimum or maximum configuration.
        configuration: Reviewed request profile hash, never inferred from a name.
        output: New working directory outside the published result tree.
        concurrency: Maximum number of simultaneous image requests.
    """
    from vlm_exam.detection_v2_inference import infer

    problems = validate_release(root)
    if problems:
        raise ValueError("\n".join(problems))
    policy, cases = load_release(root)
    if not 1 <= concurrency <= 60 or effort not in policy["efforts"]:
        raise ValueError("Invalid concurrency or effort")
    entry = next(
        (
            row
            for row in policy["runs"]
            if (row["model"], row["effort"]) == (model, effort)
        ),
        None,
    )
    if entry is None or configuration not in entry["configurations"]:
        raise ValueError(
            "Choose a measured model/effort and one of its explicit profiles"
        )
    config = policy["configurations"][configuration]
    if output.resolve().is_relative_to((root / "results").resolve()):
        raise ValueError("New runs must be outside the frozen results tree")
    for case in cases:
        path = images / case["image"]
        if (
            not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != case["image_sha256"]
        ):
            raise ValueError(f"Missing or modified original image: {case['id']}")
    identity = {
        "protocol": "detection-v2",
        "dataset_sha256": policy["dataset_sha256"],
        "model": model,
        "effort": effort,
        "configuration": configuration,
        "settings": config,
        "inference_sha256": _inference_fingerprint(),
    }
    output.mkdir(parents=True, exist_ok=True)
    with (output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity_path = output / "run.json"
        if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
            raise ValueError(
                "Resume rejected: dataset, model, effort or profile changed"
            )
        _write(identity_path, identity)
        records_path = output / "responses"
        records_path.mkdir(exist_ok=True)
        existing = {}
        for case in cases:
            path = records_path / f"{case['id']}.json"
            if path.exists():
                row = json.loads(path.read_text())
                if (row["case"], row["model"], row["arm"], row["configuration"]) != (
                    case["id"],
                    model,
                    effort,
                    configuration,
                ):
                    raise ValueError("Resume rejected: response identity mismatch")
                existing[case["id"]] = row
        pending = [
            case
            for case in cases
            if case["id"] not in existing or existing[case["id"]]["status"] == "missing"
        ]

        def _perform(case: dict[str, Any]) -> dict[str, Any]:
            try:
                row, retries = call_with_retries(
                    lambda: infer(case, model, effort, config, images)
                )
                row["attempts"] = retries.attempts
                row["status"] = response_status(row)
                row["prediction_sha256"] = hashlib.sha256(
                    row["predicted"].encode()
                ).hexdigest()
            except Exception as error:
                row = {
                    "case": case["id"],
                    "model": model,
                    "arm": effort,
                    "repeat": 1,
                    "status": "missing",
                    "error_type": type(error).__name__,
                    "detections": [],
                }
            row["configuration"] = configuration
            row["dataset_sha256"] = policy["dataset_sha256"]
            row["image_sha256"] = case["image_sha256"]
            row["inference_sha256"] = identity["inference_sha256"]
            _write(records_path / f"{case['id']}.json", row)
            return row

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(_perform, case): case["id"] for case in pending}
            _write(
                output / "status.json",
                {
                    "phase": "running",
                    "completed": len(existing),
                    "pending": len(pending),
                    "concurrency": concurrency,
                    "updated": time.time(),
                },
            )
            try:
                for future in as_completed(futures):
                    row = future.result()
                    existing[row["case"]] = row
                    _write(
                        output / "status.json",
                        {
                            "phase": "running",
                            "completed": len(existing),
                            "response_counts": dict(
                                Counter(row["status"] for row in existing.values())
                            ),
                            "updated": time.time(),
                        },
                    )
            except BaseException:
                for future in futures:
                    future.cancel()
                _write(
                    output / "status.json",
                    {"phase": "interrupted", "updated": time.time()},
                )
                raise
        records = sorted(existing.values(), key=lambda row: row["case"])
        archive = output / "responses.jsonl.gz"
        archive.write_bytes(
            gzip.compress(
                "".join(json.dumps(row) + "\n" for row in records).encode(), mtime=0
            )
        )
        frozen = {
            "model": model,
            "effort": effort,
            "path": archive.name,
            "status_counts": dict(Counter(row["status"] for row in records)),
            "configurations": [configuration],
        }
        _write(output / "summary.json", _summarize_run((output, frozen, cases)))
        _write(
            output / "status.json",
            {
                "phase": "finished",
                "completed": len(records),
                "response_counts": frozen["status_counts"],
                "updated": time.time(),
            },
        )
