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

import hashlib
import json
import textwrap
from pathlib import Path

import click
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from PIL import Image, ImageOps

from vlm_exam.reference.prompts import file_sha256, prompt_classes_for_sample
from vlm_exam.tasks.detection import DetectionSample, DetectionTask

_MODES = ("v1", "none", "overlay", "coords", "overlay_coords")
_MODE_LABELS = {
    "v1": "v1 per-class",
    "none": "all classes",
    "overlay": "box overlay",
    "coords": "box coordinates",
    "overlay_coords": "overlay + coordinates",
}


def _load_prompts(
    path: Path,
    expected_metadata: dict[str, str] | None = None,
) -> dict[tuple[str, str], str]:
    prompts: dict[tuple[str, str], str] = {}
    with open(path) as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            key = (str(record["image"]), str(record["class_name"]))
            if key in prompts:
                raise ValueError(f"Duplicate prompt record in {path}: {key!r}")
            if expected_metadata is not None:
                mismatches = [
                    field
                    for field, value in expected_metadata.items()
                    if record.get(field) != value
                ]
                if mismatches:
                    raise ValueError(
                        f"Prompt record {key!r} in {path} is incompatible: "
                        f"{', '.join(mismatches)}"
                    )
            prompts[key] = str(record["primary"])
    return prompts


def _validate_probe_manifest(
    path: Path,
    *,
    mode: str,
    dataset_sha256: str,
    selected_images_sha256: str,
    selected_image_contents_sha256: str,
    generation_config_sha256: str,
) -> None:
    manifest_path = path.with_name("manifest.json")
    if not manifest_path.exists():
        raise ValueError(f"Missing prompt manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    expected = {
        "version": "v2",
        "generation_model": "gemini-3.5-flash",
        "generation_prompt_version": "image_conditioned_v2",
        "prompt_classes": "image",
        "conditioning": mode,
        "dataset_annotations_sha256": dataset_sha256,
        "selected_images_sha256": selected_images_sha256,
        "selected_image_contents_sha256": selected_image_contents_sha256,
        "generation_config_sha256": generation_config_sha256,
    }
    mismatches = [
        field for field, value in expected.items() if manifest.get(field) != value
    ]
    if mismatches:
        raise ValueError(
            f"Prompt manifest {manifest_path} is incompatible: {', '.join(mismatches)}"
        )


def _selected_image_contents_sha256(samples: list[DetectionSample]) -> str:
    digest = hashlib.sha256()
    for sample in samples:
        image_path = Path(sample.image_path)
        digest.update(image_path.name.encode())
        digest.update(b"\0")
        digest.update(file_sha256(image_path).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _wrap(value: str, width: int = 24) -> str:
    return "\n".join(textwrap.wrap(value, width=width))


def _draw_ground_truth(
    axis: plt.Axes,
    image: Image.Image,
    sample: DetectionSample,
    class_names: tuple[str, ...],
) -> None:
    axis.imshow(image)
    axis.set_axis_off()
    colors = plt.get_cmap("tab20")
    labeled_classes: set[int] = set()
    if sample.ground_truth.class_id is None:
        return
    prompted_ids = {sample.classes.index(class_name) for class_name in class_names}
    for box, raw_class_id in zip(
        sample.ground_truth.xyxy,
        sample.ground_truth.class_id,
        strict=True,
    ):
        class_id = int(raw_class_id)
        if class_id not in prompted_ids:
            continue
        color = colors(class_id % 20)
        x_min, y_min, x_max, y_max = (float(value) for value in box)
        axis.add_patch(
            Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                fill=False,
                edgecolor=color,
                linewidth=1.5,
            )
        )
        if class_id in labeled_classes:
            continue
        labeled_classes.add(class_id)
        axis.text(
            x_min,
            y_min,
            sample.classes[class_id],
            color="white",
            fontsize=8,
            verticalalignment="bottom",
            bbox={"facecolor": color, "alpha": 0.85, "pad": 1, "edgecolor": "none"},
        )


def _build_figure(
    sample: DetectionSample,
    prompts: dict[str, dict[tuple[str, str], str]],
) -> Figure:
    image_path = Path(sample.image_path)
    image_name = image_path.name
    image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    class_names = prompt_classes_for_sample(sample, "image")

    figure = plt.figure(figsize=(24, 10), constrained_layout=True)
    grid = figure.add_gridspec(1, 2, width_ratios=(1.1, 1.9))
    image_axis = figure.add_subplot(grid[0, 0])
    table_axis = figure.add_subplot(grid[0, 1])
    _draw_ground_truth(image_axis, image, sample, class_names)
    image_axis.set_title("Ground truth: canonical classes", fontsize=13)

    columns = ["Base class", *(_MODE_LABELS[mode] for mode in _MODES)]
    rows = [
        [
            _wrap(class_name),
            *(_wrap(prompts[mode][(image_name, class_name)]) for mode in _MODES),
        ]
        for class_name in class_names
    ]
    table_axis.set_axis_off()
    table = table_axis.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="left",
        colLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 2.5)
    for column in range(len(columns)):
        table[(0, column)].set_text_props(weight="bold")
    figure.suptitle(image_name, fontsize=14)
    return figure


@click.command()
@click.option(
    "--dataset-directory",
    default="data/detection/train",
    type=click.Path(exists=True),
)
@click.option("--image-list", required=True, type=click.Path(exists=True))
@click.option(
    "--v1-prompts",
    default="reference/prompts/image_conditioned/v1/prompts.jsonl",
    type=click.Path(exists=True),
)
@click.option("--probe-directory", required=True, type=click.Path(exists=True))
@click.option("--output-directory", required=True, type=click.Path())
def main(
    dataset_directory: str,
    image_list: str,
    v1_prompts: str,
    probe_directory: str,
    output_directory: str,
) -> None:
    """Render canonical boxes and generated prompts for visual review."""
    selected_images = {
        line.strip()
        for line in Path(image_list).read_text().splitlines()
        if line.strip()
    }
    task = DetectionTask(prompt_classes="image")
    samples = [
        sample
        for sample in task.load_samples(dataset_directory)
        if Path(sample.image_path).name in selected_images
    ]
    found_images = {Path(sample.image_path).name for sample in samples}
    unknown_images = selected_images - found_images
    if unknown_images:
        raise click.ClickException(
            "Image list contains unknown files: "
            + ", ".join(sorted(unknown_images)[:10])
        )
    ordered_images = [Path(sample.image_path).name for sample in samples]
    selected_images_sha256 = hashlib.sha256(
        "\n".join(ordered_images).encode()
    ).hexdigest()
    dataset_sha256 = file_sha256(Path(dataset_directory) / "_annotations.coco.json")
    selected_image_contents_sha256 = _selected_image_contents_sha256(samples)
    probe_path = Path(probe_directory)
    generator_sha256 = file_sha256(
        Path(__file__).with_name("generate_image_conditioned_prompts.py")
    )
    try:
        for mode in _MODES:
            if mode == "v1":
                continue
            _validate_probe_manifest(
                probe_path / mode / "prompts.jsonl",
                mode=mode,
                dataset_sha256=dataset_sha256,
                selected_images_sha256=selected_images_sha256,
                selected_image_contents_sha256=selected_image_contents_sha256,
                generation_config_sha256=generator_sha256,
            )
    except ValueError as error:
        raise click.ClickException(str(error)) from error
    try:
        prompts = {
            "v1": _load_prompts(
                Path(v1_prompts),
                expected_metadata={
                    "generation_model": "gemini-3.5-flash",
                    "generation_prompt_version": "image_conditioned_v1",
                },
            ),
            **{
                mode: _load_prompts(
                    probe_path / mode / "prompts.jsonl",
                    expected_metadata={
                        "generation_model": "gemini-3.5-flash",
                        "generation_prompt_version": "image_conditioned_v2",
                        "conditioning": mode,
                    },
                )
                for mode in _MODES
                if mode != "v1"
            },
        }
    except ValueError as error:
        raise click.ClickException(str(error)) from error
    required_pairs = {
        (Path(sample.image_path).name, class_name)
        for sample in samples
        for class_name in prompt_classes_for_sample(sample, "image")
    }
    coverage_issues: list[str] = []
    for mode, mode_prompts in prompts.items():
        prompt_pairs = set(mode_prompts)
        missing = required_pairs - prompt_pairs
        extra = prompt_pairs - required_pairs if mode != "v1" else set()
        if missing or extra:
            coverage_issues.append(
                f"{mode}: missing={len(missing)}, extra={len(extra)}"
            )
    if coverage_issues:
        raise click.ClickException(
            "Prompt coverage is invalid: " + "; ".join(coverage_issues)
        )
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    pdf_path = output_path / "prompt_probe_review.pdf"
    with PdfPages(pdf_path) as pdf:
        for index, sample in enumerate(samples, start=1):
            assert isinstance(sample, DetectionSample)
            figure = _build_figure(sample, prompts)
            image_name = Path(sample.image_path).stem
            png_path = output_path / f"{index:03d}_{image_name}.png"
            figure.savefig(png_path, dpi=150, bbox_inches="tight")
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)
            click.echo(f"[{index}/{len(samples)}] {png_path}")
    click.echo(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
