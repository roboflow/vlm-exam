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

import contextlib
import gzip
import hashlib
import io
import json
import statistics
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import click
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from supervision import Detections, box_iou_batch
from supervision.metrics import MeanAveragePrecision
from supervision.metrics.mean_average_precision import COCOEvaluator, EvaluationDataset

from vlm_exam.detection_v2_identity import release_model_id

DETECTION_V2_PROTOCOL = "detection-v2"
"""Frozen 600-image detection release: one run per native minimum/maximum effort."""


def read_records(path: Path) -> list[dict[str, Any]]:
    """Read a portable compressed response archive without executing inference."""
    with gzip.open(path, "rt", encoding="utf-8") as source:
        return [json.loads(line) for line in source]


def load_release(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load the pinned release policy and its canonical image/annotation selection."""
    policy = json.loads((root / "datasets/detection/v2/release.json").read_text())
    manifest = json.loads((root / policy["dataset"]).read_text())
    return policy, manifest["cases"]


def _record_problems(
    case: dict[str, Any], row: dict[str, Any], profile: dict[str, Any]
) -> list[str]:
    from vlm_exam.tasks.detection import DetectionSample, DetectionTask

    if row.get("status") == "missing":
        return [] if not row.get("detections") else ["Missing response has detections"]
    problems = []
    try:
        for field, setting in (
            ("native_effort", row["arm"]),
            ("coordinate_format", "format"),
            ("wire_model", "wire"),
            ("provider", "provider"),
        ):
            if row.get(field) != profile.get(setting):
                problems.append(f"Response/profile mismatch: {field}")
        for field in ("max_output_tokens", "routing"):
            if field in row and row[field] != profile.get(field):
                problems.append(f"Response/profile mismatch: {field}")
        labels = sorted({obj["label"] for obj in case["objects"]})
        sample = DetectionSample(
            image_path=case["image"],
            ground_truth=_detections(case["objects"], labels),
            classes=tuple(labels),
            image_width=case["width"],
            image_height=case["height"],
        )
        prompt = DetectionTask(
            prompt_classes="image", coordinate_format=row["coordinate_format"]
        ).build_prompt(sample, uploaded_size=tuple(row["uploaded_size"]))
        if row.get("prompt") != prompt:
            problems.append("Prompt differs from current manifest")
        digest = hashlib.sha256(
            json.dumps(prompt, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if row.get("prompt_sha256") != digest:
            problems.append("Prompt checksum mismatch")
        if "image_sha256" in row and row["image_sha256"] != case["image_sha256"]:
            problems.append("Response image checksum differs from manifest")
        if row.get("status") != response_status(row):
            problems.append("Response status disagrees with saved response")
    except (KeyError, TypeError, ValueError) as error:
        problems.append(f"Invalid response metadata: {type(error).__name__}")
    return problems


def validate_release(root: Path) -> list[str]:
    """Check content hashes, exact coverage, identities and frozen configurations.

    Missing responses remain explicit records; they cannot be dropped to improve
    scores. Diagnostics never count as additional repeats or primary responses.
    """
    policy, cases = load_release(root)
    problems = []
    if (
        hashlib.sha256((root / policy["dataset"]).read_bytes()).hexdigest()
        != policy["dataset_sha256"]
    ):
        problems.append("Dataset checksum mismatch")
    identities = {case["id"] for case in cases}
    case_index = {case["id"]: case for case in cases}
    sizes = Counter(case["subset"] for case in cases)
    if len(cases) != policy["images"] or len(identities) != len(cases):
        problems.append("Missing or duplicate case identities")
    if len(sizes) != policy["subsets"] or set(sizes.values()) != {
        policy["images_per_subset"]
    }:
        problems.append("Unexpected subset selection")
    expected_paths = {
        root / entry["path"] for entry in policy["runs"] + policy["diagnostics"]
    }
    if set((root / "results/detection/v2").rglob("*.jsonl.gz")) != expected_paths:
        problems.append("Response archives differ from the frozen inventory")
    lanes = set()
    missing = []
    for digest, profile in policy["configurations"].items():
        encoded = json.dumps(profile, sort_keys=True, separators=(",", ":")).encode()
        if hashlib.sha256(encoded).hexdigest() != digest:
            problems.append(f"Configuration checksum mismatch: {digest}")
    for model in {row["model"] for row in policy["runs"]}:
        try:
            release_model_id(policy, model)
        except ValueError as error:
            problems.append(str(error))
    for entry in policy["runs"] + policy["diagnostics"]:
        path = root / entry["path"]
        if (
            not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]
        ):
            problems.append(f"Result checksum mismatch: {entry['path']}")
            continue
        records = read_records(path)
        if len(records) != entry["samples"]:
            problems.append(f"Incorrect record count: {entry['path']}")
        if "model" in entry:
            lane = (entry["model"], entry["effort"])
            if lane in lanes:
                problems.append(f"Duplicate configuration: {lane}")
            lanes.add(lane)
            accepted = entry.get("accepted_missing_as_empty", [])
            missing_ids = {r["case"] for r in records if r["status"] == "missing"}
            if (
                len(set(accepted)) != len(accepted)
                or not set(accepted).issubset(missing_ids)
                or (accepted and not entry.get("acceptance_reason"))
            ):
                problems.append(f"Invalid accepted failure policy: {lane}")
            if (
                len(records) != len(identities)
                or {row["case"] for row in records} != identities
            ):
                problems.append(f"Incorrect case selection: {lane}")
            if (
                dict(Counter(row["status"] for row in records))
                != entry["status_counts"]
            ):
                problems.append(f"Changed response status inventory: {lane}")
            if "configurations" in entry and set(entry["configurations"]) != {
                row["configuration"] for row in records
            }:
                problems.append(f"Run configuration inventory mismatch: {lane}")
            if any(
                (row["model"], row["arm"]) != lane or row["repeat"] != 1
                for row in records
            ):
                problems.append(f"Mixed run identity: {lane}")
        for row in records:
            if row["case"] not in identities:
                problems.append(f"Unknown case: {row['case']}")
                continue
            if "model" in entry and row["status"] == "missing":
                missing.append(f"{row['model']}/{row['arm']}/{row['case']}")
            if row["configuration"] not in policy["configurations"]:
                problems.append(f"Unknown configuration: {row['case']}")
            elif "model" in entry:
                problems.extend(
                    f"{row['model']}/{row['arm']}/{row['case']}: {message}"
                    for message in _record_problems(
                        case_index[row["case"]],
                        row,
                        policy["configurations"][row["configuration"]],
                    )
                )
            if (
                row.get("predicted") is not None
                and hashlib.sha256(row["predicted"].encode()).hexdigest()
                != row["prediction_sha256"]
            ):
                problems.append(f"Prediction checksum mismatch: {row['case']}")
    if sorted(missing) != policy["missing"]:
        problems.append("Missing-response identities differ from the frozen inventory")
    if len({model for model, effort in lanes}) != policy["models"]:
        problems.append("Model inventory mismatch")
    return problems


def _detections(objects: list[dict[str, Any]], labels: list[str]) -> Detections:
    return Detections(
        xyxy=np.asarray([obj["box"] for obj in objects], dtype=float).reshape(-1, 4),
        class_id=np.asarray([labels.index(obj["label"]) for obj in objects], dtype=int),
        confidence=np.asarray(
            [obj.get("confidence", 1.0) for obj in objects], dtype=float
        ),
    )


def response_status(result: dict[str, Any]) -> str:
    """Describe generation completeness and strict JSON syntax without repair."""
    text = result.get("predicted", "").strip()
    if not text:
        return "empty"
    if result.get("truncated"):
        return "truncated"
    try:
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        return (
            "returned" if isinstance(json.loads(text), (list, dict)) else "invalid_json"
        )
    except (ValueError, IndexError):
        return "invalid_json"


def score_subset(
    cases: list[dict[str, Any]], records: list[dict[str, Any]]
) -> dict[str, Any]:
    """Compute dense COCO AP@50 and class-aware maximum-matching F1@50.

    All images remain in the denominator. Absent responses contribute no
    predictions and are also reported as coverage gaps. Scores are fractions.
    """
    index = {row["case"]: row for row in records}
    labels = sorted(
        {obj["label"] for case in cases for obj in case["objects"]}
        | {obj["label"] for row in records for obj in row["detections"]}
    )
    predictions, targets = [], []
    true_positive = false_positive = false_negative = 0
    for case in cases:
        predicted = _detections(index[case["id"]]["detections"], labels)
        target = _detections(case["objects"], labels)
        matched = 0
        if len(predicted) and len(target):
            edges = (box_iou_batch(predicted.xyxy, target.xyxy) >= 0.5) & (
                predicted.class_id[:, None] == target.class_id[None, :]
            )
            matched = int(
                (
                    maximum_bipartite_matching(csr_matrix(edges), perm_type="column")
                    >= 0
                ).sum()
            )
        true_positive += matched
        false_positive += len(predicted) - matched
        false_negative += len(target) - matched
        predictions.append(predicted)
        targets.append(target)
    metric = MeanAveragePrecision()
    limit = max([101, *[len(predicted) for predicted in predictions]])
    with contextlib.redirect_stdout(io.StringIO()):
        truth = EvaluationDataset(targets=metric._prepare_targets(targets))
        evaluator = COCOEvaluator(
            truth, truth.load_predictions(metric._prepare_predictions(predictions))
        )
        evaluator.params.max_dets = [1, 10, 100, limit]
        evaluator.params.iou_thrs = np.array([0.5])
        evaluator.evaluate()
    precision = evaluator.results["precision"][0, :, :, 0, :]
    dense, capped = precision[:, :, -1], precision[:, :, -2]
    return {
        "images": len(cases),
        "map50": float(dense[dense > -1].mean()) if np.any(dense > -1) else 0.0,
        "map50_capped100": float(capped[capped > -1].mean())
        if np.any(capped > -1)
        else 0.0,
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "f1": 2
        * true_positive
        / max(1, 2 * true_positive + false_positive + false_negative),
        "missing": sum(row["status"] == "missing" for row in records),
    }


def _summarize_run(
    job: tuple[Path, dict[str, Any], list[dict[str, Any]]],
) -> dict[str, Any]:
    root, entry, cases = job
    records = read_records(root / entry["path"])
    return _summarize_records(entry, cases, records)


def _summarize_records(
    entry: dict[str, Any],
    cases: list[dict[str, Any]],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    by_case = {row["case"]: row for row in records}
    by_subset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        by_subset[case["subset"]].append(case)
    subsets = {
        name: score_subset(group, [by_case[case["id"]] for case in group])
        for name, group in sorted(by_subset.items())
    }
    true_positive = sum(value["tp"] for value in subsets.values())
    false_positive = sum(value["fp"] for value in subsets.values())
    false_negative = sum(value["fn"] for value in subsets.values())
    accepted = set(entry.get("accepted_missing_as_empty", []))
    unaccepted = sum(
        row["status"] == "missing" and row["case"] not in accepted for row in records
    )
    latency = [
        row["elapsed_seconds"]
        for row in records
        if isinstance(row.get("elapsed_seconds"), (int, float))
    ]
    return {
        "model": entry["model"],
        "effort": entry["effort"],
        "images": len(cases),
        "run_count": 1,
        "complete": unaccepted == 0,
        "accepted_failure_count": len(accepted),
        "response_counts": entry["status_counts"],
        "metrics": {
            "map50": statistics.mean(value["map50"] for value in subsets.values())
            if not unaccepted
            else None,
            "map50_missing_as_empty": statistics.mean(
                value["map50"] for value in subsets.values()
            ),
            "map50_capped100": statistics.mean(
                value["map50_capped100"] for value in subsets.values()
            )
            if not unaccepted
            else None,
            "f1": 2
            * true_positive
            / max(1, 2 * true_positive + false_positive + false_negative)
            if not unaccepted
            else None,
            "precision": true_positive / max(1, true_positive + false_positive)
            if not unaccepted
            else None,
            "recall": true_positive / max(1, true_positive + false_negative)
            if not unaccepted
            else None,
            "seconds_per_image": statistics.mean(latency)
            if len(latency) == len(cases)
            else None,
            "input_tokens": sum(row.get("input_tokens", 0) or 0 for row in records),
            "output_tokens": sum(row.get("output_tokens", 0) or 0 for row in records),
        },
        "configurations": entry["configurations"],
        "subsets": subsets,
    }


def _compare_parser_run(
    job: tuple[Path, dict[str, Any], list[dict[str, Any]], str],
) -> dict[str, Any]:
    from vlm_exam.detection_v2_inference import parse_objects
    from vlm_exam.detection_v2_parsing import effective_format, strict_format_error

    root, entry, cases, mode = job
    records = read_records(root / entry["path"])
    by_case = {case["id"]: case for case in cases}
    reasons: Counter[str] = Counter()
    compliant = recovered = 0
    variants: dict[str, Any] = {}
    audit: dict[str, list[dict[str, Any]]] = {}
    modes = ("strict", "recovery") if mode == "both" else (mode,)
    for row in records:
        if row["status"] == "missing":
            reasons["missing_response"] += 1
            continue
        error = strict_format_error(
            row["predicted"],
            effective_format(row["coordinate_format"], row.get("prompt", "")),
        )
        if error:
            reasons[error] += 1
        else:
            compliant += 1
    for variant in modes:
        parsed_records = []
        audit[variant] = []
        for row in records:
            if row["status"] == "missing":
                parsed_records.append(row)
                continue
            details: dict[str, Any] = {}
            detections = parse_objects(
                by_case[row["case"]],
                row,
                row["coordinate_format"],
                parser_mode=variant,
                diagnostics=details,
            )
            if details.get("operations") or details.get("rejected"):
                audit[variant].append(dict(case=row["case"], **details))
            if (
                variant == "recovery"
                and detections
                and strict_format_error(
                    row["predicted"],
                    effective_format(row["coordinate_format"], row.get("prompt", "")),
                )
            ):
                recovered += 1
            parsed_records.append(dict(row, detections=detections))
        variants[variant] = _summarize_records(entry, cases, parsed_records)
    return {
        "model": entry["model"],
        "effort": entry["effort"],
        "source_sha256": entry["sha256"],
        "format_compliant": compliant,
        "format_compliance_rate": compliant / len(cases),
        "format_errors": dict(reasons),
        "recovered_noncompliant_responses": recovered if "recovery" in modes else None,
        "modes": variants,
        "parser_audit": audit,
    }


def compare_parsers(root: Path, mode: str = "both", workers: int = 1) -> dict[str, Any]:
    """Score strict and recovered interpretations of identical saved responses."""
    from vlm_exam.detection_v2_parsing import PARSER_VERSION

    if mode not in {"both", "strict", "recovery"}:
        raise ValueError(f"Unknown parser mode: {mode}")
    problems = validate_release(root)
    if problems:
        raise ValueError("\n".join(problems))
    policy, cases = load_release(root)
    jobs = [(root, entry, cases, mode) for entry in policy["runs"]]
    if workers == 1:
        rows = list(map(_compare_parser_run, jobs))
    else:
        with ProcessPoolExecutor(workers) as pool:
            rows = list(pool.map(_compare_parser_run, jobs))
    return {
        "parser_version": PARSER_VERSION,
        "dataset_sha256": policy["dataset_sha256"],
        "images": len(cases),
        "mode": mode,
        "models": rows,
    }


def summarize_release(root: Path, workers: int = 1) -> dict[str, Any]:
    """Recompute every frozen configuration from stored predictions, without APIs."""
    problems = validate_release(root)
    if problems:
        raise ValueError("\n".join(problems))
    policy, cases = load_release(root)
    jobs = [(root, entry, cases) for entry in policy["runs"]]
    if workers == 1:
        rows = list(map(_summarize_run, jobs))
    else:
        with ProcessPoolExecutor(workers) as pool:
            rows = list(pool.map(_summarize_run, jobs))
    for effort in policy["efforts"]:
        ranked = sorted(
            [row for row in rows if row["effort"] == effort and row["complete"]],
            key=lambda row: -row["metrics"]["map50"],
        )
        for row in ranked:
            row["rank"] = 1 + sum(
                other["metrics"]["map50"] > row["metrics"]["map50"] for other in ranked
            )
    return {
        "version": "v2",
        "protocol": DETECTION_V2_PROTOCOL,
        "dataset_sha256": policy["dataset_sha256"],
        "images": len(cases),
        "subset_count": policy["subsets"],
        "repeats": 1,
        "scoring": {
            "primary_metric": "map50",
            "aggregation": "macro mean across subsets",
            "prediction_cap": None,
            "f1": "micro, class-aware maximum bipartite matching at IoU >= 0.5",
            "units": "fraction",
        },
        "models": rows,
    }


def public_summary(result: dict[str, Any]) -> dict[str, Any]:
    """Remove subset and provenance details from the public website payload."""
    return {
        key: value
        for key, value in result.items()
        if key not in {"dataset_sha256", "models"}
    } | {
        "models": [
            {
                key: value
                for key, value in row.items()
                if key not in {"subsets", "configurations"}
            }
            for row in result["models"]
        ]
    }


def register_detection_commands(main: click.Group) -> None:
    """Register versioned detection commands without changing the v1 protocol."""
    legacy = click.Group(
        "detection-v1", help="Original detection protocol and saved results."
    )
    legacy.add_command(main.commands["detection-report"], "report")
    legacy.add_command(main.commands["detection-visualize"], "visualize")
    main.add_command(legacy)

    @main.group("detection-v2")
    def detection_v2() -> None:
        """Inspect, validate and score the independent detection v2 release."""

    @detection_v2.command("validate")
    @click.option("--root", type=click.Path(path_type=Path), default=".")
    @click.option(
        "--reparse", is_flag=True, help="Check frozen boxes against raw responses."
    )
    def validate(root: Path, reparse: bool) -> None:
        """Validate the complete frozen inventory without private image access."""
        problems = validate_release(root)
        if problems:
            raise click.ClickException("\n".join(problems))
        policy, cases = load_release(root)
        if reparse:
            from vlm_exam.detection_v2_inference import parse_objects

            case_index = {case["id"]: case for case in cases}
            for entry in policy["runs"]:
                for row in read_records(root / entry["path"]):
                    if row["status"] == "missing":
                        continue
                    parsed = parse_objects(
                        case_index[row["case"]], row, row["coordinate_format"]
                    )
                    if parsed != row["detections"]:
                        raise click.ClickException(
                            f"Parser drift: {entry['path']} / {row['case']}"
                        )
            click.echo("Raw responses match all frozen detections")
        click.echo(
            f"Valid: {policy['models']} models, {len(policy['runs'])} configurations, "
            f"{policy['images']} images; "
            f"{len(policy['missing'])} explicit missing responses."
        )

    @detection_v2.command("evaluate-parsers")
    @click.option("--root", type=click.Path(path_type=Path), default=".")
    @click.option(
        "--mode", type=click.Choice(["strict", "recovery", "both"]), default="both"
    )
    @click.option("--workers", type=click.IntRange(1, 16), default=1)
    @click.option("--output", required=True, type=click.Path(path_type=Path))
    def evaluate_parsers(root: Path, mode: str, workers: int, output: Path) -> None:
        """Compare parser modes offline without changing published scores."""
        if output.exists():
            raise click.ClickException(
                "Choose a new report path; existing artifacts are preserved"
            )
        result = compare_parsers(root, mode, workers)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        click.echo(f"Saved parser evaluation: {output}")

    @detection_v2.command("summary")
    @click.option("--root", type=click.Path(path_type=Path), default=".")
    @click.option("--workers", type=click.IntRange(1, 16), default=1)
    @click.option("--check", is_flag=True)
    def summary(root: Path, workers: int, check: bool) -> None:
        """Recompute the v2 summary offline; check rejects publication drift."""
        result = summarize_release(root, workers)
        outputs = {
            root / "web/detection_v2_summary.json": public_summary(result),
            root / "datasets/detection/v2/summary.json": result,
        }
        for path, value in outputs.items():
            if check:
                if not path.exists() or json.loads(path.read_text()) != value:
                    raise click.ClickException(f"V2 summary is out of date: {path}")
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        click.echo("V2 summaries are up to date" if check else "Saved v2 summaries")

    @detection_v2.command("publish")
    @click.option("--root", type=click.Path(path_type=Path), default=".")
    @click.option(
        "--output", required=True, type=click.Path(exists=True, path_type=Path)
    )
    def publish(root: Path, output: Path) -> None:
        """Publish a finished full run, retaining the previous responses for audit."""
        from vlm_exam.detection_v2_publication import publish_run

        try:
            publish_run(root, output)
        except ValueError as error:
            raise click.ClickException(str(error)) from error
        click.echo("Published complete run and refreshed detection summaries")

    @detection_v2.command("prepare-images")
    @click.option("--root", type=click.Path(path_type=Path), default=".")
    @click.option(
        "--source", required=True, type=click.Path(exists=True, path_type=Path)
    )
    @click.option("--destination", required=True, type=click.Path(path_type=Path))
    def prepare_images(root: Path, source: Path, destination: Path) -> None:
        """Copy authorized local originals by SHA256 into the portable case layout."""
        import shutil

        _, cases = load_release(root)
        wanted = {case["image_sha256"] for case in cases}
        found = {}
        for path in source.rglob("*"):
            if path.is_file() and path.suffix.lower() in {
                ".png",
                ".jpg",
                ".jpeg",
                ".webp",
            }:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest in wanted:
                    found[digest] = path
        missing = wanted - found.keys()
        if missing:
            raise click.ClickException(
                f"Missing {len(missing)} original image hashes; no files copied"
            )
        destination.mkdir(parents=True, exist_ok=True)
        for case in cases:
            target = destination / case["image"]
            if (
                target.exists()
                and hashlib.sha256(target.read_bytes()).hexdigest()
                != case["image_sha256"]
            ):
                raise click.ClickException(f"Conflicting image: {target}")
        for case in cases:
            target = destination / case["image"]
            if not target.exists():
                shutil.copyfile(found[case["image_sha256"]], target)
        click.echo(
            f"Prepared {len(cases)} case images; shared photos retain separate task IDs"
        )

    @detection_v2.command("visualize")
    @click.option("--root", type=click.Path(path_type=Path), default=".")
    @click.option(
        "--images", required=True, type=click.Path(exists=True, path_type=Path)
    )
    @click.option("--model", required=True)
    @click.option(
        "--effort",
        type=click.Choice(["minimum", "maximum", "enabled", "disabled"]),
        required=True,
    )
    @click.option("--case", "case_id", required=True)
    @click.option("--output", required=True, type=click.Path(path_type=Path))
    def visualize(
        root: Path, images: Path, model: str, effort: str, case_id: str, output: Path
    ) -> None:
        """Render a saved response and canonical GT using the shared detection card."""
        import matplotlib.pyplot as plt
        from PIL import Image

        from vlm_exam.config import load_display_config
        from vlm_exam.visualization.detection import plot_detection_card

        policy, cases = load_release(root)
        model = release_model_id(policy, model)
        entry = next(
            (
                row
                for row in policy["runs"]
                if (row["model"], row["effort"]) == (model, effort)
            ),
            None,
        )
        case = next((row for row in cases if row["id"] == case_id), None)
        if entry is None or case is None:
            raise click.ClickException("Unknown measured configuration or case")
        row = next(
            row for row in read_records(root / entry["path"]) if row["case"] == case_id
        )
        image_path = images / case["image"]
        if hashlib.sha256(image_path.read_bytes()).hexdigest() != case["image_sha256"]:
            raise click.ClickException("Image checksum mismatch")
        image = np.array(Image.open(image_path).convert("RGB"))[:, :, ::-1]
        labels = sorted({obj["label"] for obj in case["objects"] + row["detections"]})
        figure = plot_detection_card(
            image=image,
            ground_truth=_detections(case["objects"], labels),
            predictions=_detections(row["detections"], labels),
            gt_labels=[obj["label"] for obj in case["objects"]],
            pred_labels=[obj["label"] for obj in row["detections"]],
            model_id=model,
            config=load_display_config(
                model_labels=root / "datasets/detection/v2/model_labels.json"
            ),
            map_score=score_subset([case], [row])["map50"]
            if row["status"] != "missing"
            else None,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=150, facecolor=figure.get_facecolor())
        plt.close(figure)
        click.echo(f"Saved {output}")

    @detection_v2.command("leaderboard")
    @click.option("--root", type=click.Path(path_type=Path), default=".")
    @click.option(
        "--effort",
        type=click.Choice(["minimum", "maximum", "enabled", "disabled"]),
        required=True,
    )
    @click.option("--top", type=click.IntRange(1), default=15)
    @click.option("--output", required=True, type=click.Path(path_type=Path))
    def leaderboard(root: Path, effort: str, top: int, output: Path) -> None:
        """Render v2 mAP@50 from its generated summary, never from v1 results."""
        import matplotlib.pyplot as plt

        from vlm_exam.config import load_display_config
        from vlm_exam.visualization.charts import plot_accuracy_chart

        payload = json.loads((root / "datasets/detection/v2/summary.json").read_text())
        policy, _ = load_release(root)
        if payload["dataset_sha256"] != policy["dataset_sha256"]:
            raise click.ClickException("Summary uses a different dataset selection")
        rows = sorted(
            [
                row
                for row in payload["models"]
                if row["effort"] == effort and row["complete"]
            ],
            key=lambda row: row["rank"],
        )[:top]
        values = {row["model"]: row["metrics"]["map50"] * 100 for row in rows}
        figure = plot_accuracy_chart(
            values,
            load_display_config(
                model_labels=root / "datasets/detection/v2/model_labels.json"
            ),
            f"Detection v2 mAP@50 · {effort.title()}",
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=150, facecolor=figure.get_facecolor())
        plt.close(figure)
        click.echo(f"Saved {output}")

    @detection_v2.command("reparse-run")
    @click.option("--root", type=click.Path(path_type=Path), default=".")
    @click.option(
        "--output", required=True, type=click.Path(exists=True, path_type=Path)
    )
    def reparse_saved_run(root: Path, output: Path) -> None:
        """Recover parser errors locally without changing inference provenance."""
        from vlm_exam.detection_v2_runner import reparse_run

        try:
            recovered = reparse_run(root, output)
        except ValueError as error:
            raise click.ClickException(str(error)) from error
        click.echo(f"Locally recovered {recovered} responses; no API requests sent")

    @detection_v2.command("run")
    @click.option("--root", type=click.Path(path_type=Path), default=".")
    @click.option(
        "--images", required=True, type=click.Path(exists=True, path_type=Path)
    )
    @click.option("--model", required=True)
    @click.option(
        "--effort",
        type=click.Choice(["minimum", "maximum", "enabled", "disabled"]),
        required=True,
    )
    @click.option(
        "--configuration",
        required=False,
        help="Explicit frozen profile hash from release.json.",
    )
    @click.option(
        "--profile",
        type=click.Path(exists=True, path_type=Path),
        help="Explicit JSON request profile for a new model.",
    )
    @click.option("--output", required=True, type=click.Path(path_type=Path))
    @click.option("--concurrency", type=click.IntRange(1, 60), default=30)
    def run(
        root: Path,
        images: Path,
        model: str,
        effort: str,
        configuration: str | None,
        profile: Path | None,
        output: Path,
        concurrency: int,
    ) -> None:
        """Run all 600 cases with a pinned profile; resume only unfinished cases.

        This command sends paid API requests. New outputs never replace the
        reviewed release automatically. Native efforts are not mapped to v1's
        low/high settings. Read status.json in the output directory for progress.
        """
        from vlm_exam.detection_v2_runner import run_release

        try:
            run_release(
                root,
                images,
                model,
                effort,
                configuration,
                output,
                concurrency,
                json.loads(profile.read_text()) if profile else None,
            )
        except ValueError as error:
            raise click.ClickException(str(error)) from error
        click.echo(f"Saved run and summary to {output}")
