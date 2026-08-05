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
from PIL import Image

from vlm_exam.box_prompting import load_case_image
from vlm_exam.box_prompting_incontext import (
    PROVIDER_MODEL_ID,
    SETS,
    TARGET_CLASS,
    build_count_prompt,
    call_messages,
    class_boxes,
    draw_sent_boxes,
    image_map50,
    parse_entries,
    payload_from_output,
    prompt_panel,
    resolve_image_name,
    result_panel,
    stack_horizontal,
    user_message,
)
from vlm_exam.box_prompting_round2 import build_round2_client
from vlm_exam.providers.image_upload import resize_image_to_max_edge
from vlm_exam.tasks.detection import DetectionTask, build_sample_index

_EXAMPLE_COUNTS = (1, 2, 3, 4)


def _collect_count(
    *,
    set_name: str,
    example_count: int,
    sample_index: dict[str, Any],
    output_directory: Path,
    client: Any,
    force: bool,
) -> None:
    raw_path = output_directory / set_name / "raw" / f"k{example_count}.json"
    if raw_path.exists() and not force:
        return
    class_name = TARGET_CLASS[set_name]
    names = [resolve_image_name(name, sample_index) for name in SETS[set_name]]
    example_names = names[:example_count]
    target_name = names[-1]
    target_sample = sample_index[target_name]

    example_images = []
    for name in example_names:
        sample = sample_index[name]
        example_images.append(
            draw_sent_boxes(
                load_case_image(sample), class_boxes(sample, class_name), ()
            )
        )
    images = example_images + [load_case_image(target_sample)]
    prompt = build_count_prompt(example_count)
    message, sizes = user_message(images, prompt)

    record: dict[str, Any] = {
        "model": PROVIDER_MODEL_ID,
        "set": set_name,
        "example_count": example_count,
        "class_name": class_name,
        "example_images": example_names,
        "target_image": target_name,
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
    print(f"[{set_name}/k{example_count}] done ({status})", flush=True)


def _analyze_and_render(
    *,
    sample_index: dict[str, Any],
    output_directory: Path,
    max_edge: int,
) -> dict[str, dict[int, float]]:
    scores: dict[str, dict[int, float]] = {}
    for set_name in SETS:
        class_name = TARGET_CLASS[set_name]
        names = [resolve_image_name(name, sample_index) for name in SETS[set_name]]
        target_name = names[-1]
        target_sample = sample_index[target_name]
        target_boxes = class_boxes(target_sample, class_name)
        renders = output_directory / set_name / "renders"
        renders.mkdir(parents=True, exist_ok=True)
        set_lines = [
            f"# Example-count sweep: {set_name}",
            "",
            f"- Target class: {class_name}",
            f"- Held-out target image: `{target_name}` ({len(target_boxes)} targets)",
            "",
            "| Examples (K) | Predicted | mAP@50 | Parse failed | Out tokens | Sec |",
            "|---:|---:|---:|---|---:|---:|",
        ]
        scores[set_name] = {}
        for example_count in _EXAMPLE_COUNTS:
            raw_path = output_directory / set_name / "raw" / f"k{example_count}.json"
            if not raw_path.exists():
                continue
            record = json.loads(raw_path.read_text())
            payload = payload_from_output(record.get("raw_output", ""))
            detections, failed = parse_entries(payload, "target", target_sample)
            score = image_map50(detections, target_boxes)
            scores[set_name][example_count] = score
            set_lines.append(
                f"| {example_count} | {len(detections)} | {score * 100:.0f}% | "
                f"{'yes' if failed else 'no'} | {record.get('output_tokens')} | "
                f"{record.get('inference_seconds', 0):.0f} |"
            )
            overlay = resize_image_to_max_edge(
                result_panel(load_case_image(target_sample), detections), max_edge
            )
            overlay.save(renders / f"k{example_count}.png")
        report_path = output_directory / set_name / "report.md"
        report_path.write_text("\n".join(set_lines) + "\n")

        example_panel = resize_image_to_max_edge(
            prompt_panel(
                load_case_image(sample_index[names[0]]),
                class_boxes(sample_index[names[0]], class_name),
                (),
            ),
            max_edge,
        )
        example_panel.save(renders / "example_1.png")
        if scores[set_name]:
            best = max(scores[set_name], key=lambda key: scores[set_name][key])
            best_overlay = renders / f"k{best}.png"
            if best_overlay.exists():
                stack_horizontal(example_panel, Image.open(best_overlay)).save(
                    renders / "stack_example_vs_best.png"
                )
    return scores


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
    default=Path("results-box-prompting-qwen38-max-count"),
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
    """Example-count sweep: K=1..4 annotated examples plus one held-out target."""
    load_dotenv()
    if not os.getenv("DASHSCOPE_API_KEY"):
        raise click.ClickException("DASHSCOPE_API_KEY is required.")
    sample_index = build_sample_index(
        DetectionTask().load_samples(str(dataset_directory))
    )
    client = build_round2_client()
    jobs = [
        (set_name, example_count)
        for set_name in SETS
        for example_count in _EXAMPLE_COUNTS
    ]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _collect_count,
                set_name=set_name,
                example_count=example_count,
                sample_index=sample_index,
                output_directory=output_directory,
                client=client,
                force=force,
            )
            for set_name, example_count in jobs
        ]
        for future in as_completed(futures):
            future.result()

    scores = _analyze_and_render(
        sample_index=sample_index,
        output_directory=output_directory,
        max_edge=max_edge,
    )
    header = "| Set | " + " | ".join(f"K={k}" for k in _EXAMPLE_COUNTS) + " |"
    divider = "|---|" + "|".join("---:" for _ in _EXAMPLE_COUNTS) + "|"
    lines = ["# Example-count sweep: summary (mAP@50)", "", header, divider]
    for set_name in SETS:
        set_scores = scores.get(set_name, {})
        cells = [
            f"{set_scores[k] * 100:.0f}%" if k in set_scores else "-"
            for k in _EXAMPLE_COUNTS
        ]
        lines.append(f"| {set_name} | " + " | ".join(cells) + " |")
    summary_path = output_directory / "summary.md"
    summary_path.write_text("\n".join(lines) + "\n")
    click.echo("\n".join(lines))
    click.echo(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
