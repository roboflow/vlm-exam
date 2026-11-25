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
from vlm_exam.detection_v2_data import verify_image
from vlm_exam.providers.base import call_with_retries, error_diagnostics


def _write(path: Path, value: Any) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, allow_nan=False) + "\n")
    temporary.replace(path)


def _inference_fingerprint() -> str:
    package = Path(__file__).parent
    sources = [
        "detection_v2_inference.py",
        "detection_v2_parsing.py",
        "detection_v2_runner.py",
        "providers/base.py",
        "detection_v2_upload.py",
        "detection_v2_data.py",
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
    configuration: str | None,
    output: Path,
    concurrency: int,
    profile: dict[str, Any] | None = None,
    *,
    max_samples: int | None = None,
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
        profile: Explicit native settings for a new model, instead of a frozen hash.
        max_samples: Optional smoke-test prefix; cannot be published as a full run.
    """
    from vlm_exam.detection_v2_inference import current_request_profile, infer
    from vlm_exam.providers.openrouter_routing import validate_openrouter_routing

    problems = validate_release(root)
    if problems:
        raise ValueError("\n".join(problems))
    policy, cases = load_release(root)
    if max_samples is not None:
        if type(max_samples) is not int or not 1 <= max_samples <= len(cases):
            raise ValueError("max_samples must be between one and the dataset size")
        cases = cases[:max_samples]
    declared_arms = set(policy["efforts"]) | {
        run["effort"] for run in policy["runs"] if run["model"] == model
    }
    if isinstance(profile, dict):
        declared_arms.update(profile.get("arms", []))
    if not 1 <= concurrency <= 60 or effort not in declared_arms:
        raise ValueError("Invalid concurrency or effort")
    source_configuration = configuration
    if profile is None:
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
        config = current_request_profile(policy["configurations"][configuration])
        if config != policy["configurations"][configuration]:
            configuration = hashlib.sha256(
                json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
    else:
        from vlm_exam.tasks.detection import DetectionCoordinateFormat

        if configuration is not None:
            raise ValueError("Choose either --configuration or --profile, not both")
        if not isinstance(profile, dict):
            raise ValueError("Profile must be a JSON object")
        arms = profile.get("arms", policy["efforts"])
        if (
            not isinstance(arms, list)
            or not arms
            or not set(arms).issubset({"minimum", "maximum", "enabled"})
        ):
            raise ValueError("Profile arms must be minimum, maximum or enabled")
        required = {"provider", "wire", "format", "max_output_tokens", *arms}
        if not required.issubset(profile) or set(profile) - required - {
            "routing",
            "routing_exception",
            "arms",
            "minimum",
            "maximum",
            "enabled",
        }:
            raise ValueError(
                "Profile must explicitly specify provider, wire, format, "
                "declared native efforts and max_output_tokens"
            )
        if profile["provider"] not in {
            "openai",
            "google",
            "anthropic",
            "openrouter",
            "xai",
        }:
            raise ValueError("Unsupported provider")
        DetectionCoordinateFormat(profile["format"])
        if not isinstance(profile["wire"], str) or not profile["wire"]:
            raise ValueError("Profile requires an upstream model identifier")
        limit = profile["max_output_tokens"]
        if limit is not None and (type(limit) is not int or limit <= 0):
            raise ValueError("max_output_tokens must be a positive integer or null")
        if profile["provider"] == "anthropic" and limit is None:
            raise ValueError("Anthropic requires an explicit positive output limit")
        if effort not in profile.get("arms", policy["efforts"]):
            raise ValueError("This profile does not define the requested arm")
        config = profile
        configuration = hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    validate_openrouter_routing(config)
    if output.resolve().is_relative_to((root / "results").resolve()):
        raise ValueError("New runs must be outside the frozen results tree")
    actual_image_hashes = {}
    for case in cases:
        path = images / case["image"]
        if not verify_image(case, path, root):
            raise ValueError(f"Missing or modified original image: {case['id']}")
        actual_image_hashes[case["id"]] = hashlib.sha256(path.read_bytes()).hexdigest()
    identity = {
        "protocol": "detection-v2",
        "dataset_sha256": policy["dataset_sha256"],
        "model": model,
        "effort": effort,
        "configuration": configuration,
        "source_configuration": source_configuration,
        "settings": config,
        "sample_selection": [case["id"] for case in cases],
        "sample_limit": max_samples,
        "inference_sha256": _inference_fingerprint(),
    }
    output.mkdir(parents=True, exist_ok=True)
    with (output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity_path = output / "run.json"
        if identity_path.exists():
            previous_identity = json.loads(identity_path.read_text())
            previous_effective = {
                key: value
                for key, value in previous_identity.items()
                if key != "source_configuration"
            }
            effective = {
                key: value
                for key, value in identity.items()
                if key != "source_configuration"
            }
            if previous_effective != effective:
                raise ValueError(
                    "Resume rejected: dataset, model, effort or profile changed"
                )
            identity = previous_identity
            source_configuration = identity.get("source_configuration")
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
                for key, expected in {
                    "dataset_sha256": policy["dataset_sha256"],
                    "image_sha256": case["image_sha256"],
                    "actual_image_sha256": actual_image_hashes[case["id"]],
                    "inference_sha256": identity["inference_sha256"],
                }.items():
                    if row.get(key) != expected:
                        raise ValueError(f"Resume rejected: response {key} mismatch")
                existing[case["id"]] = row
        pending = [
            case
            for case in cases
            if case["id"] not in existing
            or (
                existing[case["id"]]["status"] == "missing"
                and existing[case["id"]].get("request_retryable") is not False
            )
        ]

        def _perform(case: dict[str, Any]) -> dict[str, Any]:
            try:
                row, retries = call_with_retries(
                    lambda: infer(case, model, effort, config, images),
                    retry_rate_limits=True,
                )
                row["attempts"] = retries.attempts
                row["status"] = response_status(row)
                row["prediction_sha256"] = hashlib.sha256(
                    row["predicted"].encode()
                ).hexdigest()
            except Exception as error:
                row = {
                    **getattr(error, "response_record", {}),
                    "case": case["id"],
                    "model": model,
                    "arm": effort,
                    "repeat": 1,
                    "status": "missing",
                    "error_type": type(error).__name__,
                    "error": error_diagnostics(error),
                    "request_retryable": getattr(error, "request_retryable", True),
                    "detections": [],
                }
            if "predicted" in row:
                row["prediction_sha256"] = hashlib.sha256(
                    row["predicted"].encode()
                ).hexdigest()
            row["configuration"] = configuration
            row["source_configuration"] = source_configuration
            row["dataset_sha256"] = policy["dataset_sha256"]
            row["image_sha256"] = case["image_sha256"]
            row["actual_image_sha256"] = actual_image_hashes[case["id"]]
            row["inference_sha256"] = identity["inference_sha256"]
            _write(records_path / f"{case['id']}.json", row)
            return row

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(_perform, case): case["id"] for case in pending}
            _write(
                output / "status.json",
                {
                    "phase": "running",
                    "completed": sum(
                        row["status"] != "missing" for row in existing.values()
                    ),
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
                            "completed": sum(
                                row["status"] != "missing" for row in existing.values()
                            ),
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
        _finalize_run(output, identity, cases, list(existing.values()))


def _finalize_run(
    output: Path,
    identity: dict[str, Any],
    cases: list[dict[str, Any]],
    records: list[dict[str, Any]],
) -> None:
    records = sorted(records, key=lambda row: row["case"])
    archive = output / "responses.jsonl.gz"
    archive.write_bytes(
        gzip.compress(
            "".join(json.dumps(row) + "\n" for row in records).encode(), mtime=0
        )
    )
    counts = dict(Counter(row["status"] for row in records))
    entry = {
        "model": identity["model"],
        "effort": identity["effort"],
        "path": archive.name,
        "status_counts": counts,
        "configurations": [identity["configuration"]],
    }
    _write(output / "summary.json", _summarize_run((output, entry, cases)))
    _write(
        output / "status.json",
        {
            "phase": "incomplete" if counts.get("missing") else "finished",
            "completed": sum(row["status"] != "missing" for row in records),
            "attempted": len(records),
            "response_counts": counts,
            "updated": time.time(),
        },
    )
    if counts.get("missing"):
        raise ValueError(
            f"Run incomplete: {counts['missing']} infrastructure/processing failures. "
            "Responses are preserved. Resume for request failures; use reparse-run "
            "after fixing a parser failure (no additional API request)."
        )


def reparse_run(root: Path, output: Path) -> int:
    """Recover saved parser failures locally without repeating paid inference.

    Original inference identities and failed records remain preserved. Only records
    explicitly marked as parser failures are eligible; terminal model responses and
    missing network responses are never changed by this command.

    Args:
        root: Repository containing the canonical manifest and release.
        output: Existing runner directory with saved raw responses.

    Returns:
        Number of recovered parser failures.
    """
    from vlm_exam.detection_v2_inference import parse_objects

    problems = validate_release(root)
    if problems:
        raise ValueError("\n".join(problems))
    policy, all_cases = load_release(root)
    with (output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = json.loads((output / "run.json").read_text())
        config = identity["settings"]
        digest = hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if (
            identity["dataset_sha256"] != policy["dataset_sha256"]
            or digest != identity["configuration"]
        ):
            raise ValueError("Reparse rejected: dataset or configuration changed")
        selected = identity["sample_selection"]
        cases = [case for case in all_cases if case["id"] in selected]
        if len(cases) != len(selected) or len(set(selected)) != len(selected):
            raise ValueError("Reparse rejected: unknown or duplicate case selection")
        records = []
        recovered = 0
        for case in cases:
            path = output / "responses" / f"{case['id']}.json"
            row = json.loads(path.read_text())
            expected = {
                "case": case["id"],
                "model": identity["model"],
                "arm": identity["effort"],
                "configuration": identity["configuration"],
                "dataset_sha256": identity["dataset_sha256"],
                "image_sha256": case["image_sha256"],
                "inference_sha256": identity["inference_sha256"],
            }
            if any(row.get(key) != value for key, value in expected.items()):
                raise ValueError("Reparse rejected: response provenance changed")
            if row["status"] == "missing" and "parse_error" in row:
                if "raw" not in row or hashlib.sha256(
                    row["predicted"].encode()
                ).hexdigest() != row.get("prediction_sha256"):
                    raise ValueError(
                        "Reparse rejected: missing or modified raw response"
                    )
                previous_digest = hashlib.sha256(path.read_bytes()).hexdigest()
                diagnostics = output / "diagnostics"
                diagnostics.mkdir(exist_ok=True)
                backup = diagnostics / f"{case['id']}-{previous_digest}.json"
                if not backup.exists():
                    backup.write_bytes(path.read_bytes())
                try:
                    detections = parse_objects(case, row, config["format"])
                except Exception:
                    records.append(row)
                    continue
                row["detections"] = detections
                row["status"] = response_status(row)
                row["local_reparse"] = {
                    "parser_sha256": _inference_fingerprint(),
                    "previous_record_sha256": previous_digest,
                    "timestamp": time.time(),
                }
                for field in (
                    "parse_error",
                    "error",
                    "error_type",
                    "request_retryable",
                ):
                    row.pop(field, None)
                _write(path, row)
                recovered += 1
            records.append(row)
        _finalize_run(output, identity, cases, records)
        return recovered
