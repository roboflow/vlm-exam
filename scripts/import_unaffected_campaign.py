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
import shutil
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Any

from vlm_exam.config import load_config


def _verify_response_validation_change(previous: bytes, current: bytes) -> None:
    old = b"""        if usage.metadata.get("finish_reason") == "length":
            raise ValueError("Truncated completion; configuration needs review")
        if not answer.strip() or answer == EMPTY_RESPONSE_TEXT:
            raise ValueError("Empty completion")
        store.save(identity, "prediction", prediction)
"""
    new = b"""    finish = prediction["usage"]["metadata"].get("finish_reason")
    if finish == "length":
        raise ValueError("Truncated completion; configuration needs review")
    if finish != "stop":
        raise ValueError(f"Incomplete provider response: finish_reason={finish!r}")
    if not prediction["answer"].strip() or prediction["answer"] == EMPTY_RESPONSE_TEXT:
        raise ValueError("Empty completion")
    if not prior["prediction"]:
        store.save(identity, "prediction", prediction)
"""
    if previous.count(old) != 1 or previous.replace(old, new, 1) != current:
        raise ValueError("Changes exceed the reviewed response-validation fix")


def prepare(
    source: Path,
    destination: Path,
    apply: bool = False,
    scheduler_source: Path | None = None,
    invalidate_models: tuple[str, ...] = (),
    response_validation_source: Path | None = None,
) -> dict[str, Any]:
    """Import unaffected checkpoints into a newly frozen campaign directory.

    Args:
        source: Stopped campaign with its original manifest and SQLite store.
        destination: New directory; existing destinations are rejected.
        apply: Create the new campaign after validating every invariant.
        scheduler_source: Original campaign source for a scheduler-only migration.
            Its digest must match the old manifest; all code outside the queue
            dispatch section must remain byte-identical.

        invalidate_models: Explicitly authorized models whose changed-format
            Detection/Text rows must be discarded in the new copy only.

        response_validation_source: Original source for the exact reviewed
            incomplete-response rejection fix; invalid responses are archived
            in the source and removed only from the new copy.

    Returns:
        An import receipt with both manifest hashes and checkpoint counts.
    """
    if destination.exists():
        raise ValueError("Destination must not exist")
    old = json.loads((source / "manifest.json").read_text())
    configuration = load_config()
    new = json.loads(json.dumps(old))
    changes = {}
    for model, previous in old["models"].items():
        current = json.loads(json.dumps(asdict(configuration.models[model])))
        changed = {key for key in previous if previous[key] != current[key]}
        if changed - {"detection_coordinate_format"}:
            raise ValueError(f"Non-format model change: {model}: {changed}")
        if changed:
            changes[model] = {
                "old": previous["detection_coordinate_format"],
                "new": current["detection_coordinate_format"],
            }
        new["models"][model] = current
    source_root = Path("src/vlm_exam")
    scheduler_change = None
    response_validation_change = None
    for relative, digest in old["code"].items():
        current = (source_root / relative).read_bytes()
        current_digest = hashlib.sha256(current).hexdigest()
        if current_digest == digest:
            continue
        if relative == "campaign.py" and response_validation_source is not None:
            previous = response_validation_source.read_bytes()
            if hashlib.sha256(previous).hexdigest() != digest:
                raise ValueError("Original response source does not match manifest")
            _verify_response_validation_change(previous, current)
            new["code"][relative] = current_digest
            response_validation_change = {
                "previous_sha256": digest,
                "current_sha256": current_digest,
                "scope": "Reject incomplete fresh/cached responses before scoring; "
                "requests, valid response scoring and all other code unchanged.",
            }
            continue
        if relative != "campaign.py" or scheduler_source is None:
            raise ValueError(f"Source changed: {relative}")
        previous = scheduler_source.read_bytes()
        if hashlib.sha256(previous).hexdigest() != digest:
            raise ValueError("Original scheduler source does not match old manifest")
        start = b"    limit = concurrency\n"
        end = b'    output = Path(plan["output_directory"])\n'
        if any(
            content.count(marker) != 1
            for content in (previous, current)
            for marker in (start, end)
        ):
            raise ValueError("Scheduler boundaries are ambiguous")
        if (
            previous.split(start)[0] != current.split(start)[0]
            or previous.split(end)[1] != current.split(end)[1]
        ):
            raise ValueError("Inference, scoring, setup or export code changed")
        new["code"][relative] = current_digest
        scheduler_change = {
            "previous_sha256": digest,
            "current_sha256": current_digest,
            "scope": (
                "Queue dispatch, per-provider cooldowns and status only; frozen "
                "inference, scoring, configuration, dataset loading and exports "
                "unchanged."
            ),
        }
    database = sqlite3.connect(
        f"file:{source / 'checkpoints.sqlite'}?mode=ro", uri=True
    )
    database.execute("BEGIN")
    frozen = json.loads(database.execute("SELECT payload FROM manifest").fetchone()[0])
    if frozen != old:
        raise ValueError("File and database manifests differ")
    rows = database.execute(
        "SELECT identity, prediction, result FROM samples"
    ).fetchall()
    if set(invalidate_models) - changes.keys():
        raise ValueError("Invalidation requires an actual format change")
    invalidated = []
    for identity, prediction, result in rows:
        if response_validation_change and prediction:
            saved_prediction = json.loads(prediction)
            metadata = saved_prediction["usage"]["metadata"]
            answer = saved_prediction["answer"]
            if (
                metadata.get("finish_reason") != "stop"
                or not answer.strip()
                or answer == "[model returned no content]"
            ):
                invalidated.append(identity)
                continue
        model, task = identity.split(":")[:2]
        if model not in changes or task not in {"detection", "text"}:
            continue
        if model in invalidate_models:
            invalidated.append(identity)
        elif prediction or result:
            raise ValueError(f"Affected-task inference already exists: {identity}")

    def _encode(value: Any) -> bytes:
        return json.dumps(value, sort_keys=True).encode()

    receipt = {
        "source": str(source.resolve()),
        "destination": str(destination.resolve()),
        "source_manifest_sha256": hashlib.sha256(_encode(old)).hexdigest(),
        "destination_manifest_sha256": hashlib.sha256(_encode(new)).hexdigest(),
        "format_changes": changes,
        "scheduler_change": scheduler_change,
        "response_validation_change": response_validation_change,
        "invalidated_sample_ids": invalidated,
        "invalidated_completed_results": sum(
            bool(row[2]) for row in rows if row[0] in invalidated
        ),
        "sample_rows": len(rows) - len(invalidated),
        "completed_results": sum(
            bool(row[2]) for row in rows if row[0] not in invalidated
        ),
        "preservation": (
            "Unaffected sample rows and ALL charges copied verbatim; original "
            "unchanged. Explicitly invalidated format/incomplete-response rows remain "
            "archived in the source and must be inferred anew."
        ),
        "applied": apply,
    }
    if apply:
        destination.mkdir(parents=True)
        target = sqlite3.connect(destination / "checkpoints.sqlite")
        database.backup(target)
        target.execute(
            "UPDATE manifest SET payload=?", (json.dumps(new, sort_keys=True),)
        )
        target.executemany(
            "DELETE FROM samples WHERE identity=?",
            [(identity,) for identity in invalidated],
        )
        target.commit()
        for table in ["samples", "charges"]:
            source_rows = database.execute(
                f"SELECT * FROM {table} ORDER BY 1"
            ).fetchall()
            if table == "samples":
                source_rows = [row for row in source_rows if row[0] not in invalidated]
            target_rows = target.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()
            if source_rows != target_rows:
                raise ValueError(f"Checkpoint copy verification failed: {table}")
        target.close()
        (destination / "manifest.json").write_text(json.dumps(new, indent=2))
        (destination / "import-receipt.json").write_text(json.dumps(receipt, indent=2))
        (destination / "control.json").write_text(
            json.dumps({"pause": True, "concurrency": 48})
        )
        cache = source / "judge-cache.json"
        if cache.exists():
            shutil.copy2(cache, destination / cache.name)
    database.close()
    return receipt


def main() -> None:
    """Validate or apply a bounded checkpoint import after stopping workers."""
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--scheduler-source", type=Path)
    parser.add_argument("--response-validation-source", type=Path)
    parser.add_argument("--invalidate-model", action="append", default=[])
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                args.source,
                args.destination,
                args.apply,
                args.scheduler_source,
                tuple(args.invalidate_model),
                args.response_validation_source,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
