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

import json
import shutil
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import cv2

from vlm_exam.tasks.detection import (
    DetectionSample,
    DetectionTask,
    build_sample_index,
    detection_labels,
)
from vlm_exam.visualization.detection import save_annotated_detection

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR = _REPO_ROOT / "data"
_DOCS_DIR = Path(__file__).resolve().parent
_SUMMARY_SOURCE = _REPO_ROOT / "web" / "benchmark_summary.json"
_STAGE_DIR = _REPO_ROOT / "web" / "homepage-assets"
_ZIP_PATH = _REPO_ROOT / "web" / "homepage-assets.zip"
_EXAMPLES_ROOT = _STAGE_DIR / "examples"

_TASK_DISPLAY_NAMES = {
    "ocr": "OCR",
    "extraction": "Data Extraction",
    "counting": "Counting",
    "identification": "Identification",
    "reasoning": "Reasoning",
    "detection": "Object Detection",
}

_QA_TASK_ORDER = ("ocr", "extraction", "counting", "identification", "reasoning")

_QA_PICKS: dict[str, list[tuple[str, str | None, str]]] = {
    "ocr": [
        (
            "4f770728-7b1f-4bf3-b877-e3d33fcc135d_png.rf.dffcaaeb26fd7fe4fd14891cb00a0adf.jpg",
            None,
            "handwriting (recipe)",
        ),
        (
            "c05c3288-ac14-47d6-8756-9ad484ae9696_png.rf.0ff67a7a39c5d5c7c9739988d851c032.jpg",
            None,
            "printed table (cost of living)",
        ),
        (
            "5cf55f37-8adb-4adb-9520-148340548028_png.rf.d09a7340877434f1cbdb25f1eccd4774.jpg",
            None,
            "invoice with table (HTML structure)",
        ),
        (
            "2e43ea51-fc34-4d42-8b0f-66b2172d81f7_png.rf.345ac976ac43b542da24279e8683876b.jpg",
            None,
            "math exam (LaTeX)",
        ),
        (
            "00099ca8-7388-4ccb-a9ae-0def0f38e1d4_png.rf.87454ebaa2a188c18aadac0664ef70f6.jpg",
            None,
            "social media post",
        ),
    ],
    "extraction": [
        (
            "d2029347-fd3c-40be-8d89-b0e1abf64e9b_input-input-1781204084098-88xr6eavh_png_jpg.rf.21d8156058ed4f8ed2cffba4507eedee.jpg",
            "license plate number on the rear of the black SUV",
            "license plate",
        ),
        (
            "f0d4c0bf-3a21-41a5-b0c2-c33b94feadb4_input-input-1771859752511-bk56l0pe1_png_jpg.rf.6c4ddb547fa7d14bea520ceb83d179fe.jpg",
            "complete two-line identification text",
            "container serial number",
        ),
        (
            "39fe721b-44c4-4a0c-aeac-2897520dfa68_input-input-1781394102861-00bvuicbi_png_jpg.rf.363242f15c7e98f02eea7cb38d065c13.jpg",
            "large alphanumeric tire size sequence",
            "tire sidewall spec",
        ),
    ],
    "counting": [
        (
            "db35c41c-a9ed-41bc-96cf-60186d715477_input-input-1774653477270-a4j6j7v6e_png_jpg.rf.81a4c1ac6254af08518ea58b5137ed9c.jpg",
            "standing or kneeling directly on the rebar grid",
            "people (occlusion)",
        ),
        (
            "c9b00dcb-e093-43d9-baf8-f1971ff328af_input-input-1769103177177-3x5hfzwbp_png_jpg.rf.5036bc0df858e9911be4fa08e6b80a40.jpg",
            "intact capsules remain",
            "pills in a blister pack",
        ),
        (
            "21a6d806-d16e-4452-a2e3-7fa520bd671d_input-input-1769360430457-50a88jpis_png_jpg.rf.686e67d06dd1f487be5dc88fff5e7bc3.jpg",
            "plastic bin labeled",
            "screws in a labeled bin",
        ),
        (
            "1478f318-0372-4bbb-bf28-6cab8c447a59_input-input-1771868885484-ua18hynnb_png_jpg.rf.48494081054afe07b3a29e3cf54058fe.jpg",
            "M&M's packages are visible in the machine",
            "dense small objects (vending)",
        ),
        (
            "26aea39a-beea-43d5-bbd5-c716b698ebaf_png_jpg.rf.b43c70658245fd13f38d8b30e5bdc44d.jpg",
            "shrink-wrapped packs of coke",
            "packs on a pallet (logistics)",
        ),
    ],
    "identification": [
        (
            "cc68304c-144a-4975-a93a-d8efa85db6ed_input-input-1769095484858-rohdg9mmc_png_jpg.rf.af1b26038a5e163c931dc4e983f53a56.jpg",
            "brand name is printed on the yogurt",
            "brand name",
        ),
        (
            "097f7d52-0fd1-4385-9ec4-65a9ee83d81b_input-input-1776852419339-bqf3nz0wx_png_jpg.rf.40fa273eb7f690671e9c0e59d85e7be5.jpg",
            "immediately to the left of the bright green car",
            "color (relative position)",
        ),
        (
            "4f2b6aab-f1c1-4f16-b3ca-bbe4d399a897_input-input-1773381300920-0uaf4zno9_png_jpg.rf.cafad4b6eb9a3ec94667457e32e8b7f5.jpg",
            "specific type of red fruit",
            "object type (fruit)",
        ),
    ],
    "reasoning": [
        (
            "6479e64d-04e1-4a4f-8b52-a25f229c4b6d_input-input-1773260224462-12gc1mat5_png_jpg.rf.5db91ad0f0d494c6342881f9641bb9ef.jpg",
            "abnormal or defective candies",
            "defect inspection (yes/no)",
        ),
        (
            "1ad1c779-b18f-4aaa-9cf0-6155cb19879b_input-input-1769444672197-ug2bdacps_png_jpg.rf.63d95c1d0be80a188822c14bbb724d70.jpg",
            "total milligrams",
            "multi-step (dosage x pill count)",
        ),
        (
            "25e6ed9c-2d37-4533-8021-977770aa97d1_input-input-1780691294811-oql9y0w9f_png_jpg.rf.7ff054e9f71f1f19f68b7c2d986394ce.jpg",
            "completely full",
            "spatial capacity (grid if full)",
        ),
        (
            "ac750e7d-de93-4813-828d-409d16bd176b_png_jpg.rf.c3a8340123de2edb1fa67616ed0ff9c9.jpg",
            "difference between the original total number of events",
            "printed vs handwritten revision",
        ),
    ],
}

_DETECTION_PICKS: list[tuple[str, str]] = [
    (
        "basketball-player-detection_boston-celtics-new-york-knicks-game-4-q1-00-57-00-54-0091_png.rf.83258ba0ac1e70470effeb141e679662.jpg",
        "sports (basketball)",
    ),
    (
        "drone-vision_frame_000080_jpg.rf.d9e4df80c876c60246588e0fdb8b17e3.jpg",
        "aerial / drone (traffic)",
    ),
    (
        "damage-severity_7664036-hyundai-hyundai_elantra_2016-2019_16_sx_mt-2019-minor_scratch-dents_jpeg_jpg.rf.349db7fdbf0b7bb30a3d30fa4e6c7073.jpg",
        "automotive (damage detection)",
    ),
    (
        "bone-fracture_23_jpg.rf.3f7ca22f850d50ee64232ed53ffd4a44.jpg",
        "medical (X-ray fracture)",
    ),
    (
        "circuit-voltages_89_jpg.rf.7515d5c7fbc223c0efa06b2e27c10eaf.jpg",
        "electronics (circuit schematic)",
    ),
]


def _load_qa_rows(task: str) -> list[dict[str, Any]]:
    path = _DATA_DIR / task / "train" / "annotations.jsonl"
    rows: list[dict[str, Any]] = []
    with open(path) as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _select_row(
    rows: list[dict[str, Any]], image: str, match: str | None
) -> dict[str, Any]:
    candidates = [
        row
        for row in rows
        if row["image"] == image and (match is None or match in row["prefix"])
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one row for image={image!r} match={match!r}, "
            f"found {len(candidates)}."
        )
    return candidates[0]


def _reset_stage() -> None:
    if _STAGE_DIR.exists():
        shutil.rmtree(_STAGE_DIR)
    _EXAMPLES_ROOT.mkdir(parents=True, exist_ok=True)


def _build_qa_examples() -> dict[str, list[dict[str, Any]]]:
    examples_by_task: dict[str, list[dict[str, Any]]] = {}
    for task in _QA_TASK_ORDER:
        rows = _load_qa_rows(task)
        images_dir = _EXAMPLES_ROOT / task / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        task_examples: list[dict[str, Any]] = []
        for index, (image, match, subtask) in enumerate(_QA_PICKS[task], start=1):
            row = _select_row(rows, image, match)
            shutil.copy2(_DATA_DIR / task / "train" / image, images_dir / image)
            task_examples.append(
                {
                    "id": f"{task}-{index}",
                    "task": task,
                    "subtask": subtask,
                    "image": f"images/{image}",
                    "question": row["prefix"],
                    "answer": row["suffix"],
                }
            )
        examples_by_task[task] = task_examples
    return examples_by_task


def _build_detection_examples() -> list[dict[str, Any]]:
    detection_dir = _DATA_DIR / "detection" / "train"
    task = DetectionTask()
    samples = task.load_samples(str(detection_dir))
    index = build_sample_index(samples)

    images_dir = _EXAMPLES_ROOT / "detection" / "images"
    annotated_dir = _EXAMPLES_ROOT / "detection" / "annotated"
    images_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)

    examples: list[dict[str, Any]] = []
    for position, (image, subtask) in enumerate(_DETECTION_PICKS, start=1):
        sample = index.get(image)
        if not isinstance(sample, DetectionSample):
            raise ValueError(f"Detection sample not found for image={image!r}.")

        shutil.copy2(detection_dir / image, images_dir / image)

        labels = detection_labels(sample.ground_truth, list(sample.classes))
        annotated_name = f"{Path(image).stem}.png"
        scene = cv2.imread(str(detection_dir / image))
        save_annotated_detection(
            scene,
            sample.ground_truth,
            labels,
            annotated_dir / annotated_name,
            label_mode="auto",
        )

        per_class = Counter(labels)
        present_classes = [
            sample.classes[class_id]
            for class_id in sorted(set(sample.ground_truth.class_id))
        ]
        examples.append(
            {
                "id": f"detection-{position}",
                "task": "detection",
                "subtask": subtask,
                "image": f"images/{image}",
                "annotated_image": f"annotated/{annotated_name}",
                "prompt": task.build_prompt(sample),
                "classes": present_classes,
                "answer": {
                    "total_objects": int(len(sample.ground_truth)),
                    "per_class": dict(sorted(per_class.items())),
                },
            }
        )
    return examples


def _write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _rebase(row: dict[str, Any], task: str) -> dict[str, Any]:
    rebased = dict(row)
    rebased["image"] = f"{task}/{row['image']}"
    if "annotated_image" in row:
        rebased["annotated_image"] = f"{task}/{row['annotated_image']}"
    return rebased


def _fence(text: str) -> str:
    longest = 0
    current = 0
    for char in text:
        if char == "`":
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return "`" * max(3, longest + 1)


def _detection_answer_line(answer: dict[str, Any]) -> str:
    parts = [f"{name} x{count}" for name, count in answer["per_class"].items()]
    return f"{answer['total_objects']} objects - " + ", ".join(parts)


def _append_fenced(lines: list[str], label: str, text: str) -> None:
    fence = _fence(text)
    lines.append(f"**{label}:**")
    lines.append("")
    lines.append(fence)
    lines.append(text)
    lines.append(fence)
    lines.append("")


def _write_preview(index_rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# VLM-Exam - example gallery",
        "",
        "A rough preview of five examples per task (image, question, answer).",
        "Question and answer text is shown in full; the same data is also in the",
        "`examples.jsonl` files and in `index.jsonl`.",
        "",
    ]
    by_task: dict[str, list[dict[str, Any]]] = {}
    for row in index_rows:
        by_task.setdefault(row["task"], []).append(row)

    for task in (*_QA_TASK_ORDER, "detection"):
        lines.append(f"## {_TASK_DISPLAY_NAMES[task]}")
        lines.append("")
        for row in by_task.get(task, []):
            lines.append(f"### {row['id'].upper()} - {row['subtask']}")
            lines.append("")
            lines.append(f"![]({row['image']})")
            lines.append("")
            if task == "detection":
                lines.append("Ground truth (boxes + labels):")
                lines.append("")
                lines.append(f"![]({row['annotated_image']})")
                lines.append("")
                lines.append(f"**Detect:** {', '.join(row['classes'])}")
                lines.append("")
                lines.append(f"**Answer:** {_detection_answer_line(row['answer'])}")
                lines.append("")
            else:
                _append_fenced(lines, "Q", row["question"])
                _append_fenced(lines, "A", row["answer"])
    path.write_text("\n".join(lines) + "\n")


def _copy_docs_and_summary() -> None:
    shutil.copy2(_DOCS_DIR / "task_definitions.md", _STAGE_DIR / "task_definitions.md")
    shutil.copy2(
        _DOCS_DIR / "summary_json_schema.md", _STAGE_DIR / "summary_json_schema.md"
    )
    if not _SUMMARY_SOURCE.exists():
        raise FileNotFoundError(
            f"Summary JSON not found at {_SUMMARY_SOURCE}; run vlm-exam summary first."
        )
    shutil.copy2(_SUMMARY_SOURCE, _STAGE_DIR / "benchmark_summary.json")


def _write_zip() -> None:
    if _ZIP_PATH.exists():
        _ZIP_PATH.unlink()
    with zipfile.ZipFile(_ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as archive:
        for file_path in sorted(_STAGE_DIR.rglob("*")):
            if file_path.is_file():
                archive.write(file_path, file_path.relative_to(_STAGE_DIR))


def main() -> None:
    """Build the homepage assets zip for the design and frontend team."""
    _reset_stage()

    qa_examples = _build_qa_examples()
    detection_examples = _build_detection_examples()

    index_rows: list[dict[str, Any]] = []
    for task in _QA_TASK_ORDER:
        rows = qa_examples[task]
        _write_jsonl(rows, _EXAMPLES_ROOT / task / "examples.jsonl")
        index_rows.extend(_rebase(row, task) for row in rows)

    _write_jsonl(detection_examples, _EXAMPLES_ROOT / "detection" / "examples.jsonl")
    index_rows.extend(_rebase(row, "detection") for row in detection_examples)

    _write_jsonl(index_rows, _EXAMPLES_ROOT / "index.jsonl")
    _write_preview(index_rows, _EXAMPLES_ROOT / "PREVIEW.md")
    _copy_docs_and_summary()
    _write_zip()

    print(f"Staged {len(index_rows)} examples in {_STAGE_DIR}")
    print(f"Wrote {_ZIP_PATH}")


if __name__ == "__main__":
    main()
