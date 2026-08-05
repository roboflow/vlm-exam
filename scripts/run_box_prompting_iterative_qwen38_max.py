# Copyright 2026 Roboflow, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import click
from dotenv import load_dotenv

from vlm_exam.box_prompting import load_case_image
from vlm_exam.box_prompting_incontext import (
    PROVIDER_MODEL_ID,
    SETS,
    TARGET_CLASS,
    call_messages,
    class_boxes,
    correction_text,
    draw_sent_boxes,
    image_map50,
    iterative_next_text,
    iterative_turn1_text,
    parse_entries,
    payload_from_output,
    resolve_image_name,
    result_panel,
    user_message,
)
from vlm_exam.box_prompting_round2 import build_round2_client
from vlm_exam.providers.image_upload import resize_image_to_max_edge
from vlm_exam.tasks.detection import DetectionTask, build_sample_index

_ARMS = ("corrected", "control")


def _collect_conversation(
    *,
    set_name: str,
    arm: str,
    sample_index: dict[str, Any],
    output_directory: Path,
    client: Any,
    force: bool,
) -> None:
    raw_path = output_directory / set_name / "raw" / f"{arm}.json"
    if raw_path.exists() and not force:
        return
    class_name = TARGET_CLASS[set_name]
    names = [resolve_image_name(name, sample_index) for name in SETS[set_name]]
    example_sample = sample_index[names[0]]
    target_names = names[1:]

    marked_example = draw_sent_boxes(
        load_case_image(example_sample),
        class_boxes(example_sample, class_name),
        (),
    )
    record: dict[str, Any] = {
        "model": PROVIDER_MODEL_ID,
        "set": set_name,
        "arm": arm,
        "class_name": class_name,
        "example_image": names[0],
        "target_images": target_names,
        "turns": [],
        "error": None,
    }
    messages: list[dict[str, Any]] = []
    try:
        for turn_index, target_name in enumerate(target_names):
            target_sample = sample_index[target_name]
            if turn_index == 0:
                message, _ = user_message(
                    [marked_example, load_case_image(target_sample)],
                    iterative_turn1_text(),
                )
            else:
                correction = None
                if arm == "corrected":
                    previous = sample_index[target_names[turn_index - 1]]
                    correction = correction_text(
                        class_boxes(previous, class_name),
                        previous.image_width,
                        previous.image_height,
                    )
                message, _ = user_message(
                    [load_case_image(target_sample)],
                    iterative_next_text(correction),
                )
            messages.append(message)
            result = call_messages(client, messages)
            messages.append({"role": "assistant", "content": result["raw_output"]})
            record["turns"].append(
                {
                    "turn": turn_index + 1,
                    "target_image": target_name,
                    "raw_output": result["raw_output"],
                    "input_tokens": result["input_tokens"],
                    "output_tokens": result["output_tokens"],
                    "inference_seconds": result["inference_seconds"],
                }
            )
            print(
                f"[{set_name}/{arm}] turn {turn_index + 1}/{len(target_names)} done",
                flush=True,
            )
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(record, indent=2))
    print(
        f"[{set_name}/{arm}] conversation done ({record['error'] or 'ok'})", flush=True
    )


def _analyze_and_render(
    *,
    sample_index: dict[str, Any],
    output_directory: Path,
    max_edge: int,
) -> dict[str, dict[str, list[float]]]:
    scores: dict[str, dict[str, list[float]]] = {}
    for set_name in SETS:
        class_name = TARGET_CLASS[set_name]
        set_lines = [
            f"# Iterative correction: {set_name}",
            "",
            f"- Target class: {class_name}",
            "",
            "| Turn | Target image | Targets | corrected mAP@50 | control mAP@50 |",
            "|---:|---|---:|---:|---:|",
        ]
        scores[set_name] = {}
        per_arm_turn: dict[str, list[tuple[int, str, int, float]]] = {}
        for arm in _ARMS:
            raw_path = output_directory / set_name / "raw" / f"{arm}.json"
            if not raw_path.exists():
                continue
            record = json.loads(raw_path.read_text())
            renders = output_directory / set_name / "renders" / arm
            renders.mkdir(parents=True, exist_ok=True)
            rows: list[tuple[int, str, int, float]] = []
            for turn in record.get("turns", []):
                target_name = turn["target_image"]
                sample = sample_index[target_name]
                payload = payload_from_output(turn.get("raw_output", ""))
                detections, _ = parse_entries(payload, "target", sample)
                boxes = class_boxes(sample, class_name)
                score = image_map50(detections, boxes)
                rows.append((turn["turn"], target_name, len(boxes), score))
                overlay = resize_image_to_max_edge(
                    result_panel(load_case_image(sample), detections), max_edge
                )
                overlay.save(
                    renders / f"turn{turn['turn']}_{Path(target_name).stem}.png"
                )
            per_arm_turn[arm] = rows
            scores[set_name][arm] = [row[3] for row in rows]

        corrected_rows = {row[0]: row for row in per_arm_turn.get("corrected", [])}
        control_rows = {row[0]: row for row in per_arm_turn.get("control", [])}
        for turn_number in sorted(set(corrected_rows) | set(control_rows)):
            corrected = corrected_rows.get(turn_number)
            control = control_rows.get(turn_number)
            reference = corrected or control
            name = reference[1]
            targets = reference[2]
            corrected_cell = f"{corrected[3] * 100:.0f}%" if corrected else "-"
            control_cell = f"{control[3] * 100:.0f}%" if control else "-"
            set_lines.append(
                f"| {turn_number} | {name} | {targets} | {corrected_cell} | "
                f"{control_cell} |"
            )
        report_path = output_directory / set_name / "report.md"
        report_path.write_text("\n".join(set_lines) + "\n")
    return scores


def _mean_by_turn(
    scores: dict[str, dict[str, list[float]]],
    arm: str,
) -> list[float]:
    columns: dict[int, list[float]] = {}
    for set_scores in scores.values():
        for turn_index, value in enumerate(set_scores.get(arm, [])):
            columns.setdefault(turn_index, []).append(value)
    return [sum(columns[i]) / len(columns[i]) for i in sorted(columns)]


@click.command()
@click.option(
    "--dataset-directory",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("data/detection/train"),
    show_default=True,
)
@click.option(
    "--output-directory",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("results-box-prompting-qwen38-max-iterative"),
    show_default=True,
)
@click.option("--max-workers", type=click.IntRange(min=1), default=6, show_default=True)
@click.option(
    "--max-edge", type=click.IntRange(min=200), default=1600, show_default=True
)
@click.option("--force/--no-force", default=False, show_default=True)
def main(
    dataset_directory: Path,
    output_directory: Path,
    max_workers: int,
    max_edge: int,
    force: bool,
) -> None:
    """Iterative correction: multi-turn conversation with ground-truth
    corrections (corrected) versus no feedback (control)."""
    load_dotenv()
    if not os.getenv("DASHSCOPE_API_KEY"):
        raise click.ClickException("DASHSCOPE_API_KEY is required.")
    sample_index = build_sample_index(
        DetectionTask().load_samples(str(dataset_directory))
    )
    client = build_round2_client()
    jobs = [(set_name, arm) for set_name in SETS for arm in _ARMS]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _collect_conversation,
                set_name=set_name,
                arm=arm,
                sample_index=sample_index,
                output_directory=output_directory,
                client=client,
                force=force,
            )
            for set_name, arm in jobs
        ]
        for future in as_completed(futures):
            future.result()

    scores = _analyze_and_render(
        sample_index=sample_index,
        output_directory=output_directory,
        max_edge=max_edge,
    )
    corrected_trend = _mean_by_turn(scores, "corrected")
    control_trend = _mean_by_turn(scores, "control")
    lines = [
        "# Iterative correction: summary (mean mAP@50 across sets by turn)",
        "",
        "| Turn | corrected | control |",
        "|---:|---:|---:|",
    ]
    for turn_index in range(max(len(corrected_trend), len(control_trend))):
        corrected_cell = (
            f"{corrected_trend[turn_index] * 100:.0f}%"
            if turn_index < len(corrected_trend)
            else "-"
        )
        control_cell = (
            f"{control_trend[turn_index] * 100:.0f}%"
            if turn_index < len(control_trend)
            else "-"
        )
        lines.append(f"| {turn_index + 1} | {corrected_cell} | {control_cell} |")
    summary_path = output_directory / "summary.md"
    summary_path.write_text("\n".join(lines) + "\n")
    click.echo("\n".join(lines))
    click.echo(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
