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
import re
from pathlib import Path
from typing import Any

from vlm_exam.detection_v2 import (
    _record_problems,
    load_release,
    public_summary,
    read_records,
    summarize_release,
    validate_release,
)
from vlm_exam.detection_v2_identity import release_model_id


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _archive(records: list[dict[str, Any]]) -> bytes:
    return gzip.compress(
        "".join(json.dumps(row, allow_nan=False) + "\n" for row in records).encode(),
        mtime=0,
    )


def _atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def _json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def _publish(
    root: Path,
    records: list[dict[str, Any]],
    configurations: dict[str, dict[str, Any]],
    reason: str,
    recovery: bool,
) -> None:
    from collections import Counter, defaultdict

    from vlm_exam.detection_v2_inference import parse_objects

    if not reason.strip() or not records:
        raise ValueError("Publication requires records and an audit reason")
    with (root / ".detection-v2-publication.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        errors = validate_release(root)
        if errors:
            raise ValueError("\n".join(errors))
        policy, cases = load_release(root)
        case_index = {case["id"]: case for case in cases}
        for digest, config in configurations.items():
            if _digest(config) != digest:
                raise ValueError("Configuration fingerprint mismatch")
        profiles = policy["configurations"] | configurations
        lanes: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for original in records:
            row = dict(original)
            stored_model = release_model_id(policy, row.get("model", ""))
            if stored_model != row.get("model"):
                row["submitted_model_key"] = row["model"]
                row["model"] = stored_model
            case = case_index.get(row.get("case"))
            config = profiles.get(row.get("configuration"))
            if case is None or config is None or row.get("status") == "missing":
                raise ValueError("Unknown case/profile or missing response")
            if row.get("repeat") != 1 or row.get("arm") not in set(
                policy["efforts"]
            ) | set(config.get("arms", [])):
                raise ValueError("Invalid repeat or effort")
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", row.get("model", "")):
                raise ValueError("Unsafe model identity")
            errors = _record_problems(case, row, config)
            if hashlib.sha256(row["predicted"].encode()).hexdigest() != row.get(
                "prediction_sha256"
            ):
                errors.append("Prediction checksum mismatch")
            if parse_objects(case, row, row["coordinate_format"]) != row["detections"]:
                errors.append("Parsed detections differ from raw response")
            if errors:
                raise ValueError(f"{row['case']}: " + "; ".join(errors))
            lanes[row["model"], row["arm"]].append(dict(row))
        writes: dict[Path, bytes] = {}
        for (model, arm), incoming in lanes.items():
            incoming_index = {row["case"]: row for row in incoming}
            if len(incoming_index) != len(incoming):
                raise ValueError("Duplicate publication case")
            entry = next(
                (
                    e
                    for e in policy["runs"]
                    if (e["model"], e["effort"]) == (model, arm)
                ),
                None,
            )
            previous = read_records(root / entry["path"]) if entry else []
            previous_index = {row["case"]: row for row in previous}
            if recovery:
                for case_id, row in incoming_index.items():
                    old = previous_index.get(case_id)
                    if old is None or old["status"] != "missing":
                        raise ValueError(
                            "Recovery may only replace missing/stale cases"
                        )
                    if (
                        old.get("missing_reason") != "stale_prompt"
                        and old["configuration"] != row["configuration"]
                    ):
                        raise ValueError("Infrastructure recovery changed its profile")
                combined = previous_index | incoming_index
            else:
                combined = incoming_index
            if set(combined) != set(case_index):
                raise ValueError(
                    "Publication must retain the entire canonical selection"
                )
            if not recovery and any(
                row["status"] == "missing" for row in combined.values()
            ):
                raise ValueError("A full publication cannot contain missing responses")
            superseded = [row for row in previous if row["case"] in incoming_index]
            if superseded:
                data = _archive(superseded)
                digest = hashlib.sha256(data).hexdigest()
                archive_path = (
                    f"results/detection/v2/diagnostics/superseded-{digest}.jsonl.gz"
                )
                if not any(e["path"] == archive_path for e in policy["diagnostics"]):
                    writes[root / archive_path] = data
                    policy["diagnostics"].append(
                        dict(path=archive_path, sha256=digest, samples=len(superseded))
                    )
            for row in incoming_index.values():
                row["publication"] = {
                    "reason": reason,
                    "kind": (
                        "protocol_correction"
                        if recovery
                        and previous_index[row["case"]].get("missing_reason")
                        == "stale_prompt"
                        else "infrastructure_recovery"
                        if recovery
                        else "full_run"
                    ),
                    "replaced_record_sha256": _digest(previous_index[row["case"]])
                    if row["case"] in previous_index
                    else None,
                }
            path = (
                entry["path"]
                if entry
                else f"results/detection/v2/{model}__{arm}.jsonl.gz"
            )
            data = _archive(sorted(combined.values(), key=lambda row: row["case"]))
            writes[root / path] = data
            updated = dict(
                model=model,
                effort=arm,
                path=path,
                sha256=hashlib.sha256(data).hexdigest(),
                samples=len(combined),
                status_counts=dict(Counter(row["status"] for row in combined.values())),
                configurations=sorted(
                    {row["configuration"] for row in combined.values()}
                ),
            )
            if entry:
                entry.clear()
                entry.update(updated)
            else:
                policy["runs"].append(updated)
        policy["configurations"] = profiles
        policy["runs"].sort(key=lambda entry: (entry["model"], entry["effort"]))
        policy["models"] = len({entry["model"] for entry in policy["runs"]})
        policy["missing"] = []
        for entry in policy["runs"]:
            data = writes.get(root / entry["path"])
            rows = (
                [json.loads(line) for line in gzip.decompress(data).splitlines()]
                if data is not None
                else read_records(root / entry["path"])
            )
            policy["missing"].extend(
                f"{row['model']}/{row['arm']}/{row['case']}"
                for row in rows
                if row["status"] == "missing"
            )
        policy["missing"].sort()
        writes[root / "datasets/detection/v2/release.json"] = _json(policy)
        summary_paths = [
            root / "datasets/detection/v2/summary.json",
            root / "web/detection_v2_summary.json",
        ]
        backups = {
            path: path.read_bytes() if path.exists() else None
            for path in [*writes, *summary_paths]
        }
        try:
            for path, data in writes.items():
                _atomic(path, data)
            summary = (
                summarize_release(root, workers=8)
                if len(cases) > 10
                else summarize_release(root)
            )
            _atomic(summary_paths[0], _json(summary))
            _atomic(summary_paths[1], _json(public_summary(summary)))
        except BaseException:
            for path, data in backups.items():
                if data is None:
                    path.unlink(missing_ok=True)
                else:
                    _atomic(path, data)
            raise


def publish_run(root: Path, output: Path) -> None:
    """Publish a complete, finished run and refresh both detection summaries.

    Args:
        root: Repository root containing the canonical release.
        output: Runner output directory; partial smoke runs are rejected.
    """
    with (output / "run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(
                "Run is active; wait for execution or reparse to finish"
            ) from error
        identity = json.loads((output / "run.json").read_text())
        status = json.loads((output / "status.json").read_text())
        policy, cases = load_release(root)
        if (
            status.get("phase") != "finished"
            or identity.get("dataset_sha256") != policy["dataset_sha256"]
        ):
            raise ValueError(
                "Only a finished run on the current dataset can be published"
            )
        rows = read_records(output / "responses.jsonl.gz")
        config = identity["settings"]
        fingerprint = _digest(config)
        if fingerprint != identity["configuration"] or len(rows) != len(cases):
            raise ValueError("Invalid profile or partial run")
        case_index = {case["id"]: case for case in cases}
        for row in rows:
            for field in ("inference_sha256", "source_configuration"):
                if field in identity and row.get(field) != identity[field]:
                    raise ValueError(f"Mixed run identity: {field}")
            if (row.get("model"), row.get("arm"), row.get("configuration")) != (
                identity["model"],
                identity["effort"],
                fingerprint,
            ) or row.get("dataset_sha256") != policy["dataset_sha256"]:
                raise ValueError("Mixed run identity")
            if row.get("image_sha256") != case_index.get(row.get("case"), {}).get(
                "image_sha256"
            ):
                raise ValueError("Image identity mismatch")
        _publish(
            root,
            rows,
            {fingerprint: config},
            "Completed canonical runner output",
            False,
        )


def publish_recoveries(
    root: Path,
    records: list[dict[str, Any]],
    configurations: dict[str, dict[str, Any]],
    reason: str,
) -> None:
    """Promote reviewed recoveries without replacing any terminal model response.

    Args:
        root: Repository containing the canonical release.
        records: Responses replacing explicit missing/stale placeholders only.
        configurations: Fingerprinted native settings, including new rerun profiles.
        reason: Human-readable justification retained in every promoted record.
    """
    _publish(root, records, configurations, reason, True)
