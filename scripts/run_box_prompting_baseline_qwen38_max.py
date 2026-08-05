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
    build_baseline_prompt,
    call_messages,
    class_boxes,
    draw_sent_boxes,
    image_map50,
    parse_entries,
    payload_from_output,
    prompt_panel,
    resolve_image_name,
    result_panel,
    select_reference_boxes,
    stack_horizontal,
    user_message,
)
from vlm_exam.box_prompting_round2 import build_round2_client
from vlm_exam.providers.image_upload import resize_image_to_max_edge
from vlm_exam.tasks.detection import DetectionTask, build_sample_index

_ARMS = ("single", "multi")

_CLASS_OVERRIDES = {
    "set2_football": "referee",
    "set5_basketball": "jersey number",
}
_PROMPT_OVERRIDES = {
    "set4_technical_drawing": (
        "V000078_0_0_jpeg_jpg.rf.ea93b54e17810ab845506f60bf69b7dd.jpg"
    ),
}


def _baseline_config(
    set_name: str,
    sample_index: dict[str, Any],
) -> tuple[str, str, list[str]]:
    class_name = _CLASS_OVERRIDES.get(set_name, TARGET_CLASS[set_name])
    names = [resolve_image_name(name, sample_index) for name in SETS[set_name]]
    if set_name in _PROMPT_OVERRIDES:
        prompt_name = resolve_image_name(_PROMPT_OVERRIDES[set_name], sample_index)
        target_names = [name for name in names if name != prompt_name]
    else:
        prompt_name = names[0]
        target_names = names[1:]
    return class_name, prompt_name, target_names


def _collect_arm(
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
    class_name, prompt_name, target_names = _baseline_config(set_name, sample_index)
    prompt_sample = sample_index[prompt_name]
    target_samples = [sample_index[name] for name in target_names]

    positives, negatives, negative_classes = select_reference_boxes(
        prompt_sample, class_name, arm
    )
    prompt = build_baseline_prompt(arm, len(target_samples), bool(negatives))
    marked = draw_sent_boxes(load_case_image(prompt_sample), positives, negatives)
    images = [marked] + [load_case_image(sample) for sample in target_samples]
    message, sizes = user_message(images, prompt)

    record: dict[str, Any] = {
        "model": PROVIDER_MODEL_ID,
        "set": set_name,
        "arm": arm,
        "class_name": class_name,
        "prompt_image": prompt_name,
        "target_images": target_names,
        "positive_xyxy": [list(box) for box in positives],
        "negative_xyxy": [list(box) for box in negatives],
        "negative_classes": list(negative_classes),
        "prompt": prompt,
        "uploaded_sizes": sizes,
        "error": None,
    }
    try:
        record.update(call_messages(client, [message]))
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(record, indent=2))
    status = record["error"] or f"{record.get('output_tokens')} tok"
    print(f"[{set_name}/{arm}] done ({status})", flush=True)


def _analyze_and_render(
    *,
    sample_index: dict[str, Any],
    output_directory: Path,
    max_edge: int,
) -> list[tuple[str, str, float]]:
    summary: list[tuple[str, str, float]] = []
    for set_name in SETS:
        class_name, prompt_name, target_names = _baseline_config(set_name, sample_index)
        set_lines = [
            f"# Baseline cross-image prompting: {set_name}",
            "",
            f"- Target class: {class_name}",
            f"- Prompt image: `{prompt_name}`",
            "",
            "| Arm | Target image | Targets | Predicted | mAP@50 | Parse failed |",
            "|---|---|---:|---:|---:|---|",
        ]
        for arm in _ARMS:
            raw_path = output_directory / set_name / "raw" / f"{arm}.json"
            if not raw_path.exists():
                continue
            record = json.loads(raw_path.read_text())
            renders = output_directory / set_name / "renders" / arm
            renders.mkdir(parents=True, exist_ok=True)
            prompt_sample = sample_index[prompt_name]
            positives = [tuple(box) for box in record["positive_xyxy"]]
            negatives = [tuple(box) for box in record["negative_xyxy"]]
            panel = resize_image_to_max_edge(
                prompt_panel(
                    load_case_image(prompt_sample),
                    tuple(positives),
                    tuple(negatives),
                ),
                max_edge,
            )
            panel.save(renders / "prompt.png")

            payload = payload_from_output(record.get("raw_output", ""))
            arm_scores = []
            for index, name in enumerate(target_names):
                sample = sample_index[name]
                detections, failed = parse_entries(
                    payload, f"image_{index + 2}", sample
                )
                boxes = class_boxes(sample, class_name)
                score = image_map50(detections, boxes)
                arm_scores.append(score)
                set_lines.append(
                    f"| {arm} | {name} | {len(boxes)} | {len(detections)} | "
                    f"{score * 100:.0f}% | {'yes' if failed else 'no'} |"
                )
                overlay = resize_image_to_max_edge(
                    result_panel(load_case_image(sample), detections), max_edge
                )
                stem = Path(name).stem
                overlay.save(renders / f"{stem}.png")
                stack_horizontal(panel, overlay).save(renders / f"stack_{stem}.png")
            mean_score = sum(arm_scores) / len(arm_scores) if arm_scores else 0.0
            summary.append((set_name, arm, mean_score))
        report_path = output_directory / set_name / "report.md"
        report_path.write_text("\n".join(set_lines) + "\n")
    return summary


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
    default=Path("results-box-prompting-qwen38-max-baseline"),
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
    """Baseline single vs multi cross-image box prompting over all sets."""
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
                _collect_arm,
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

    summary = _analyze_and_render(
        sample_index=sample_index,
        output_directory=output_directory,
        max_edge=max_edge,
    )
    lines = [
        "# Baseline cross-image prompting: summary",
        "",
        "| Set | Arm | Mean target mAP@50 |",
        "|---|---|---:|",
    ]
    for set_name, arm, score in summary:
        lines.append(f"| {set_name} | {arm} | {score * 100:.0f}% |")
    summary_path = output_directory / "summary.md"
    summary_path.write_text("\n".join(lines) + "\n")
    click.echo("\n".join(lines))
    click.echo(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
