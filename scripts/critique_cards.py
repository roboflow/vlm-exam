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

from pathlib import Path

import click
from dotenv import load_dotenv
from google import genai
from google.genai import types

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


def critique_image(client: genai.Client, model: str, image_path: Path) -> str:
    """Ask a vision model to critique one rendered card image.

    Args:
        client: Configured google-genai client.
        model: Gemini model identifier.
        image_path: Path to the PNG card to review.

    Returns:
        The model's critique text.
    """
    image_part = types.Part.from_bytes(
        data=image_path.read_bytes(), mime_type="image/png"
    )
    response = client.models.generate_content(
        model=model,
        contents=[image_part, _CRITIQUE_PROMPT],
        config=types.GenerateContentConfig(temperature=0.0),
    )
    return (response.text or "EMPTY RESPONSE").strip()


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
def main(images_directory: str, model: str, report_file: str | None) -> None:
    """Review rendered benchmark cards with a vision-model design critic."""
    load_dotenv()
    client = genai.Client()

    images_path = Path(images_directory)
    image_files = sorted(images_path.glob("*.png"))
    if not image_files:
        click.echo(f"No .png files found in {images_directory}")
        return

    report_path = Path(report_file) if report_file else images_path / "critique.md"
    sections: list[str] = [f"# Card critique ({model})\n"]

    for image_path in image_files:
        click.echo(f"\n=== {image_path.name} ===")
        feedback = critique_image(client, model, image_path)
        click.echo(feedback)
        sections.append(f"## {image_path.name}\n\n{feedback}\n")

    report_path.write_text("\n".join(sections))
    click.echo(f"\nReport written to {report_path}")


if __name__ == "__main__":
    main()
