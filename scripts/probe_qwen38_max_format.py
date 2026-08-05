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

"""Probe the native detection output format of Qwen3.8-Max on DashScope."""

from __future__ import annotations

import json
import os
from pathlib import Path

import click
from dotenv import load_dotenv
from PIL import Image, ImageOps

from vlm_exam.format_matrix import select_matrix_images
from vlm_exam.format_probe import (
    PROMPT_VARIANTS,
    analyze_record,
    build_probe_prompt,
    completed_images,
    format_probe_report,
    summarize_job,
)
from vlm_exam.providers.dashscope import DashScopeProvider
from vlm_exam.tasks.detection import DetectionTask, build_sample_index
from vlm_exam.workflows_comparison import image_classes

MODEL_KEY = "qwen-3.8-max"
PROVIDER_MODEL_ID = "qwen3.8-max"


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
    default=Path("results-format-probe-qwen38-max"),
    show_default=True,
)
@click.option(
    "--image-count", type=click.IntRange(min=1), default=20, show_default=True
)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option("--effort", default="low", show_default=True)
def main(
    dataset_directory: Path,
    output_directory: Path,
    image_count: int,
    seed: int,
    effort: str,
) -> None:
    """Collect and analyze format-free detection responses from Qwen3.8-Max."""
    load_dotenv()
    if not os.getenv("DASHSCOPE_API_KEY"):
        raise click.ClickException("DASHSCOPE_API_KEY is required.")

    sample_index = build_sample_index(
        DetectionTask().load_samples(str(dataset_directory))
    )
    selected = select_matrix_images(sample_index, count=image_count, seed=seed)
    samples = [sample_index[name] for name in selected]

    raw_directory = output_directory / "raw"
    raw_directory.mkdir(parents=True, exist_ok=True)

    for variant in PROMPT_VARIANTS:
        provider = DashScopeProvider(MODEL_KEY, provider_model_id=PROVIDER_MODEL_ID)
        output_path = raw_directory / f"{PROVIDER_MODEL_ID}__{variant}.jsonl"
        done = completed_images(output_path)
        with open(output_path, "a", buffering=1) as file:
            for index, sample in enumerate(samples, start=1):
                image_name = Path(sample.image_path).name
                if image_name in done:
                    continue
                classes = image_classes(sample)
                prompt = build_probe_prompt(variant, classes)
                record = {
                    "model": PROVIDER_MODEL_ID,
                    "variant": variant,
                    "reasoning_effort": effort,
                    "image": image_name,
                    "original_width": sample.image_width,
                    "original_height": sample.image_height,
                    "classes": classes,
                    "prompt": prompt,
                    "error": None,
                }
                try:
                    image = ImageOps.exif_transpose(
                        Image.open(sample.image_path)
                    ).convert("RGB")
                    uploaded = provider.uploaded_image_size(image) or image.size
                    answer, usage, retry_stats = provider.predict(image, prompt, effort)
                    record.update(
                        {
                            "raw_output": answer,
                            "uploaded_width": uploaded[0],
                            "uploaded_height": uploaded[1],
                            "input_tokens": usage.input_tokens,
                            "output_tokens": usage.output_tokens,
                            "inference_seconds": retry_stats.inference_seconds,
                            "attempts": retry_stats.attempts,
                        }
                    )
                except Exception as error:
                    record["error"] = f"{type(error).__name__}: {error}"
                file.write(json.dumps(record) + "\n")
                click.echo(f"Probed {variant} on {image_name} ({index}/{len(samples)})")

    summaries = {}
    for jsonl_path in sorted(raw_directory.glob("*.jsonl")):
        analyses = []
        with open(jsonl_path) as file:
            for line in file:
                if not line.strip():
                    continue
                record = json.loads(line)
                sample = sample_index.get(record["image"])
                if sample is None:
                    continue
                analyses.append(analyze_record(record, sample))
        if analyses:
            summaries[jsonl_path.stem] = summarize_job(analyses)

    report = format_probe_report(summaries)
    analysis_directory = output_directory / "analysis"
    analysis_directory.mkdir(parents=True, exist_ok=True)
    with open(analysis_directory / "format_summary.json", "w") as file:
        json.dump(summaries, file, indent=2)
    with open(analysis_directory / "report.md", "w") as file:
        file.write(report + "\n")
    click.echo(report)


if __name__ == "__main__":
    main()
