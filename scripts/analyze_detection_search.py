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
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import supervision as sv
from supervision.metrics import MeanAveragePrecision

from vlm_exam.tasks.detection import (
    DetectionCoordinateFormat,
    DetectionTask,
    parse_prediction,
)


def _score(predictions: list[Any], targets: list[Any]) -> tuple[float, float]:
    metric = MeanAveragePrecision()
    metric.update(predictions, targets)
    result = metric.compute()
    return float(result.map50), float(result.map50_95)


def analyze(directory: Path, resamples: int = 300) -> dict[str, Any]:
    """Compare complete formats and bootstrap paired image-level score gaps.

    Args:
        directory: Frozen format search directory.
        resamples: Number of deterministic paired bootstrap resamples.

    Returns:
        Complete format scores and uncertainty for the top two per model.
    """
    manifest = json.loads((directory / "matrix-manifest.json").read_text())
    samples = DetectionTask().load_samples("data/detection/train")[:50]
    targets = [sample.ground_truth for sample in samples]
    groups = defaultdict(list)
    for job in manifest["jobs"]:
        path = directory / "responses" / (job["id"] + ".json")
        if not path.exists():
            raise ValueError(f"Incomplete search: {job['id']}")
        row = json.loads(path.read_text())
        groups[(row["model"], row["format"])].append(row)
    predictions = {}
    scores = defaultdict(list)
    for (model, coordinate_format), rows in groups.items():
        rows.sort(key=lambda row: row["index"])
        if [row["index"] for row in rows] != list(range(50)):
            raise ValueError("Expected the same 50 distinct image indices")
        boxes = []
        for sample, row in zip(samples, rows):
            finish = row.get("usage", {}).get("metadata", {}).get("finish_reason")
            if row.get("error") or finish == "length":
                boxes.append(sv.Detections.empty())
                continue
            boxes.append(
                parse_prediction(
                    row.get("answer", ""),
                    (sample.image_width, sample.image_height),
                    list(sample.classes),
                    coordinate_format=DetectionCoordinateFormat(coordinate_format),
                    uploaded_wh=row["uploaded_size"],
                )
            )
        predictions[(model, coordinate_format)] = boxes
        map50, map50_95 = _score(boxes, targets)
        scores[model].append(
            {
                "format": coordinate_format,
                "map50": map50,
                "map50_95": map50_95,
                "n": len(rows),
            }
        )
    comparisons = {}
    generator = np.random.default_rng(20261026)
    indices = generator.integers(0, 50, size=(resamples, 50))
    for model, formats in scores.items():
        formats.sort(key=lambda row: (-row["map50"], -row["map50_95"], row["format"]))
        first, second = formats[:2]
        gaps = []
        for selected in indices:
            expected = [targets[index] for index in selected]
            a = _score(
                [predictions[(model, first["format"])][index] for index in selected],
                expected,
            )[0]
            b = _score(
                [predictions[(model, second["format"])][index] for index in selected],
                expected,
            )[0]
            gaps.append(a - b)
        interval = np.quantile(gaps, [0.025, 0.975]).tolist()
        comparisons[model] = {
            "best_observed": first["format"],
            "runner_up": second["format"],
            "gap": first["map50"] - second["map50"],
            "paired_bootstrap_95_percent_interval": interval,
            "resamples": resamples,
            "positive_interval": interval[0] > 0,
        }
        print(json.dumps({"model": model, **comparisons[model]}), flush=True)
    return {
        "effort": "high",
        "images": 50,
        "scores": dict(scores),
        "comparisons": comparisons,
        "limitations": (
            "Exploratory image bootstrap on the search subset, conditional on one "
            "response per image and format. Does not capture generation variance "
            "or correct for selecting winners among multiple formats; not a claim "
            "of global optimality or independent held-out validation."
        ),
    }


def main() -> None:
    """Save scores and paired uncertainty after every search job is complete."""
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--resamples", type=int, default=300)
    args = parser.parse_args()
    report = analyze(args.directory, args.resamples)
    (args.directory / "comparison.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
