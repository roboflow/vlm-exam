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

import math
from pathlib import Path

import click
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

_CRITIQUE_PROMPT = (
    "You are a senior visual designer reviewing an automatically generated "
    "benchmark result card. All cards share one 16:9 hero layout with a "
    "light Roboflow-style theme: purple accent stripe on top; the "
    "benchmark input image fills the left half; the right rail stacks a "
    "model identity row (lab logo, model name, lab name, purple task tag "
    "such as OCR or COUNTING), a compact status line, the main content "
    "section, and a footer with the Roboflow Playground brand. The rail "
    "content varies by task:\n\n"
    "1. OCR cards: status line is a similarity percentage with a slim "
    "progress bar; the content is a text diff (large EXPECTED vs MODEL "
    "for short answers, FULL TEXT DIFF wrapped monospace, or DIFFS "
    "fragments), with green highlights marking text only in the expected "
    "transcription and red marking text only in the model output.\n\n"
    "2. QA cards (extraction, counting, identification, reasoning): "
    "status line is a green CORRECT or red INCORRECT verdict; the content "
    "is a QUESTION section followed by the answer: a single green model "
    "answer when correct, or stacked EXPECTED and MODEL blocks with "
    "diff highlights when incorrect; the footer notes the evaluation "
    "method.\n\n"
    "3. Object detection cards: the prediction-annotated image fills "
    "the left half; the right rail stacks the identity row, a "
    "PREDICTION VS GROUND TRUTH section with legend chips and a faded "
    "copy of the image tinted green where only ground-truth boxes "
    "cover, red where only predicted boxes cover, and purple where "
    "both agree, then a prominent mAP@50 score; the footer shows "
    "expected versus predicted object counts. Colored boxes and class "
    "chips drawn on the images are the annotation payload and are "
    "expected to be dense or overlapping; do not critique them.\n\n"
    "Design intent: the diff, answer, or annotated images are the hero "
    "and should get the most space; score and verdict are deliberately "
    "compact.\n\n"
    "Critique ONLY the generated layout and typography, NOT the content of "
    "the benchmark input image itself (it is user data and may be blurry, "
    "dark, or oddly cropped).\n\n"
    "Report every issue you can find, focusing on:\n"
    "- Overlapping, clipped, or truncated text\n"
    "- Elements colliding with or crowding each other\n"
    "- Misaligned elements or inconsistent margins\n"
    "- Large areas of wasted whitespace\n"
    "- Readability problems: font too small, poor contrast\n"
    "- Confusing visual hierarchy or unpolished details\n\n"
    "Format: one numbered line per issue, starting with a severity tag "
    "[HIGH], [MEDIUM], or [LOW], followed by a specific, actionable "
    "description that names the affected element and what to change. "
    "HIGH means broken (overlap, clipping, unreadable), MEDIUM means "
    "clearly unpolished, LOW means nitpick.\n\n"
    "If the card looks clean, reply with exactly: NO ISSUES"
)

_ANNOTATION_CRITIQUE_PROMPT = (
    "You are a senior visual designer reviewing object-detection benchmark "
    "result cards. Each card shows, on its left half, an input image "
    "annotated with predicted bounding boxes drawn in distinct colors. When "
    "per-box text labels would be too crowded, the card instead draws a "
    "class color legend: a translucent panel (usually top-left, one or two "
    "columns) with a colored swatch next to each class name, so viewers can "
    "map each box color to its class.\n\n"
    "You are given a MONTAGE that tiles several cards of DIFFERENT source "
    "image resolutions together. Judge ONLY these two things:\n"
    "1. Legend: when a card has no per-box text labels, is a color legend "
    "present, readable, and unambiguous? Flag missing legends, low contrast "
    "between swatch/text and the panel, swatches whose color is hard to tell "
    "apart from the box color they represent, clipped or truncated class "
    "names, a panel that overlaps important image content, or an oversized "
    "or undersized panel.\n"
    "2. Consistency: across the tiled cards, do the bounding-box line "
    "thickness and the label/legend font sizes look VISUALLY CONSISTENT "
    "regardless of source resolution? Flag any card whose boxes look "
    "noticeably thicker or thinner, or whose text looks larger or smaller, "
    "than the others.\n\n"
    "Do NOT critique the underlying photo content, the box positions, the "
    "right-hand rail, or detection accuracy. Focus solely on legend "
    "readability and cross-card consistency of box thickness and font size.\n\n"
    "Format: one numbered line per issue, starting with a severity tag "
    "[HIGH], [MEDIUM], or [LOW], then a specific, actionable description that "
    "names the affected card (by grid position, e.g. top-left) and what to "
    "change. If both aspects look clean and consistent, reply with exactly: "
    "NO ISSUES"
)

_PROMPTS = {
    "layout": _CRITIQUE_PROMPT,
    "annotations": _ANNOTATION_CRITIQUE_PROMPT,
}


def critique_image(
    client: genai.Client,
    model: str,
    image_path: Path,
    prompt: str = _CRITIQUE_PROMPT,
) -> str:
    """Ask a vision model to critique one rendered card image.

    Args:
        client: Configured google-genai client.
        model: Gemini model identifier.
        image_path: Path to the PNG card to review.
        prompt: Critique instruction sent alongside the image.

    Returns:
        The model's critique text.
    """
    image_part = types.Part.from_bytes(
        data=image_path.read_bytes(), mime_type="image/png"
    )
    response = client.models.generate_content(
        model=model,
        contents=[image_part, prompt],
        config=types.GenerateContentConfig(temperature=0.0),
    )
    return (response.text or "EMPTY RESPONSE").strip()


def build_montage(
    image_paths: list[Path],
    output_path: Path,
    columns: int = 2,
    cell_width: int = 900,
    gap: int = 24,
) -> Path:
    """Tile card PNGs into a single montage for cross-card comparison.

    Args:
        image_paths: Card PNGs to tile, in reading order.
        output_path: Destination PNG path for the montage.
        columns: Number of grid columns.
        cell_width: Width each card is resized to before tiling.
        gap: Pixel gap between cells and around the grid.

    Returns:
        The montage path written.
    """
    resized: list[Image.Image] = []
    for path in image_paths:
        image = Image.open(path).convert("RGB")
        width, height = image.size
        new_height = round(height * cell_width / width)
        resized.append(image.resize((cell_width, new_height)))

    rows = math.ceil(len(resized) / columns)
    row_heights = [
        max(
            (image.size[1] for image in resized[row * columns : (row + 1) * columns]),
            default=0,
        )
        for row in range(rows)
    ]
    total_width = gap + columns * (cell_width + gap)
    total_height = gap + sum(row_heights) + rows * gap
    canvas = Image.new("RGB", (total_width, total_height), (255, 255, 255))

    y = gap
    for row in range(rows):
        for column in range(columns):
            index = row * columns + column
            if index >= len(resized):
                break
            x = gap + column * (cell_width + gap)
            canvas.paste(resized[index], (x, y))
        y += row_heights[row] + gap

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


@click.command()
@click.option(
    "--images-directory",
    required=True,
    type=click.Path(exists=True),
    help="Directory containing card PNGs to review.",
)
@click.option(
    "--model",
    default="gemini-3.5-flash",
    help="Vision model used as the design critic.",
)
@click.option(
    "--report-file",
    default=None,
    type=click.Path(),
    help="Markdown report path (default: critique.md next to the images).",
)
@click.option(
    "--focus",
    default="layout",
    type=click.Choice(sorted(_PROMPTS)),
    help="Critique focus: 'layout' (general polish) or 'annotations' "
    "(detection legend and cross-card box/font consistency).",
)
@click.option(
    "--montage/--no-montage",
    default=False,
    help="Tile all cards into one montage and critique that instead of "
    "each card separately (needed for cross-card consistency checks).",
)
def main(
    images_directory: str,
    model: str,
    report_file: str | None,
    focus: str,
    montage: bool,
) -> None:
    """Review rendered benchmark cards with a vision-model design critic."""
    load_dotenv()
    client = genai.Client()

    images_path = Path(images_directory)
    image_files = sorted(images_path.glob("*.png"))
    image_files = [path for path in image_files if path.name != "_montage.png"]
    if not image_files:
        click.echo(f"No .png files found in {images_directory}")
        return

    prompt = _PROMPTS[focus]
    report_path = Path(report_file) if report_file else images_path / "critique.md"
    sections: list[str] = [f"# Card critique ({model}, focus={focus})\n"]

    if montage:
        montage_path = build_montage(image_files, images_path / "_montage.png")
        click.echo(f"\n=== montage of {len(image_files)} cards ===")
        feedback = critique_image(client, model, montage_path, prompt)
        click.echo(feedback)
        sections.append(f"## montage ({len(image_files)} cards)\n\n{feedback}\n")
    else:
        for image_path in image_files:
            click.echo(f"\n=== {image_path.name} ===")
            feedback = critique_image(client, model, image_path, prompt)
            click.echo(feedback)
            sections.append(f"## {image_path.name}\n\n{feedback}\n")

    report_path.write_text("\n".join(sections))
    click.echo(f"\nReport written to {report_path}")


if __name__ == "__main__":
    main()
