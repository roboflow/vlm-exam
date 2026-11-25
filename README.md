# vlm-exam

Benchmark suite for Vision Language Models. Compare accuracy, cost, and
speed across frontier VLMs on standardized visual tasks.

## Active benchmarks

- **RF-ODv2** (`detection`): object detection, one run per native configuration.
- **RF-OCRv2** (`text`): mixed OCR, one run at Low and High.

RF-ODv1 and RF-OCRv1 are historical benchmarks published separately in the web
JSON. Extraction, Counting, Identification and Reasoning remain runnable and
retain their raw results, but are no longer active or exported to the website.

## Historical QA leaderboards

The `results/` directory holds the raw benchmark outputs (one JSONL file per
run) and is the single source of truth for the numbers below. Regenerate the
charts at any time with `vlm-exam leaderboard`.

The QA charts show LLM-judge accuracy; each links to the strict-match
variant of the same leaderboard. See "Run a VQA benchmark" below for how
the two metrics differ. Charts below are at `--effort low`; every chart also
has a `_high.png` counterpart in `visualizations/leaderboards/`. Models
benchmarked under the three-run protocol (see "Benchmark a model end to
end") show a min-max whisker over their runs.

### Counting

![Counting accuracy leaderboard](visualizations/leaderboards/counting_accuracy_low.png)

[Strict-match variant](visualizations/leaderboards/counting_accuracy_strict_low.png)

### Extraction

![Extraction accuracy leaderboard](visualizations/leaderboards/extraction_accuracy_low.png)

[Strict-match variant](visualizations/leaderboards/extraction_accuracy_strict_low.png)

### Identification

![Identification accuracy leaderboard](visualizations/leaderboards/identification_accuracy_low.png)

[Strict-match variant](visualizations/leaderboards/identification_accuracy_strict_low.png)

### Reasoning

![Reasoning accuracy leaderboard](visualizations/leaderboards/reasoning_accuracy_low.png)

[Strict-match variant](visualizations/leaderboards/reasoning_accuracy_strict_low.png)

### Object Detection

![Object Detection mAP@50 leaderboard](visualizations/leaderboards/detection_map50_low.png)

Reference open-vocabulary detectors (YOLO-E, etc.) are benchmarked separately.
See [reference/README.md](reference/README.md) for setup and commands.

Historical v1 (different dataset/protocol):
[mAP@50](visualizations/leaderboards/detection/v1/detection_map50_low.png) |
[mAP@75](visualizations/leaderboards/detection/v1/detection_map75_low.png) |
[mAP@50:95](visualizations/leaderboards/detection/v1/detection_map50_95_low.png)

## Supported providers

- Anthropic (Claude)
- Google (Gemini)
- OpenAI (GPT)
- OpenRouter (any OpenAI-compatible vision model, e.g. Qwen3.7 Plus,
  GLM 5V Turbo)

## Installation

```bash
pip install vlm-exam
```

Or install from source:

```bash
git clone https://github.com/roboflow/vlm-exam.git
cd vlm-exam
pip install -e ".[dev]"
```

## Quick start

Set your API keys (or place them in a `.env` file):

```bash
export ANTHROPIC_API_KEY=...
export DASHSCOPE_API_KEY=...
export GOOGLE_API_KEY=...
export OPENAI_API_KEY=...
export OPENROUTER_API_KEY=...
export XAI_API_KEY=...
```

### Run a VQA benchmark

Expects a dataset directory containing an `annotations.jsonl` file with
`image`, `prefix` (question), and `suffix` (answer) fields.

```bash
vlm-exam run \
    --task vqa \
    --models claude-fable-5,gemini-3.5-flash,gpt-5.5 \
    --effort high \
    --dataset-directory data/vqa/train
```

Counting, extraction, identification, and reasoning are scored two ways at
once, and both numbers are reported for every model:

- **strict**: a deterministic rule (normalized exact match, or integer
  equality for counting). Measures correctness and format compliance.
- **judge**: an LLM judge (`gemini-3.5-flash`, temperature 0) scores every
  sample independently of the strict rule. Measures correctness while
  tolerating phrasing.

The judge runs automatically and needs `GOOGLE_API_KEY`; pick a different
judge model with `--judge-model`. The judge number is the headline on the
leaderboard charts and the website; the strict number is exported next to
it. A run whose stored predictions lack either verdict can be backfilled in
place without re-running the model:

```bash
vlm-exam rescore results/reasoning_gpt-5.5_low_20260725_101010.jsonl
```

### Benchmark a model end to end

The default protocol is RF-ODv2 and RF-OCRv2 at Low and High, with one run
per configuration (four jobs for a model with two detection arms):

```bash
vlm-exam benchmark --models MODEL
```

Use `--tasks`, `--efforts`, and `--max-parallel` to select work. Logs are written
under `logs/`. Historical tasks require explicit selection and default to three
repeats; selecting detection v1 changes only the detection task:

```bash
vlm-exam benchmark --models MODEL --tasks counting,ocr
vlm-exam benchmark --models MODEL --tasks detection --detection-version v1
vlm-exam leaderboard --tasks counting,ocr
vlm-exam report --tasks reasoning
```

`text` remains the internal/CLI identity of RF-OCRv2; `ocr` means RF-OCRv1.
`text-benchmark` remains available as an OCRv2-only launcher. `--repeats` can
explicitly select repeats for non-v2-detection tasks; detection v2 always uses
one run per arm.

Low/High are categories, not identical API parameters across benchmarks.
Detection maps them to reviewed minimum/maximum profiles; Gemini 4 retains its
approved native low/high stability exception. OCRv2 uses the existing provider
adapters. Recorded native parameters remain in `configuration_details` in the
JSON; unknown historical settings remain unknown. This change does not rerun
inference or modify any historical scores or request settings.

### Validate the results directory

`vlm-exam validate` checks active OCRv2 coverage and the pinned detection v2
release. Missing archived tasks do not affect current completion. Use
`validate --detection-version v1` for historical six-task coverage. Frozen
release integrity checks remain enforced.

```bash
vlm-exam validate
```

Models benchmarked before the protocol are marked
`benchmark_protocol: legacy` in `models.yaml`; their gaps are reported as
warnings rather than failures until they are backfilled (`--strict` fails on
them too, `--verbose` lists every missing configuration). `--format github`
adds workflow annotations and a step summary; the repository's GitHub
Action runs it, together with `vlm-exam summary --check`, on every pull
request.

### Run a detection benchmark

Expects a COCO-format dataset directory containing an
`_annotations.coco.json` file alongside the images.

```bash
vlm-exam run \
    --task detection \
    --models gemini-3.5-flash,gpt-5.5,claude-fable-5 \
    --effort low \
    --dataset-directory data/detection/train
```

Useful options:

- `--max-samples 10` limits the number of processed images (handy for smoke tests).
- `--prompt-classes image` (default) lists only the classes present in each
  image's ground truth; `--prompt-classes all` lists every dataset class.

### Summarize results

Accuracy, token usage, and cost tables across all saved runs. Each row is
one `(task, model, effort)` configuration with its run count; metrics are
the mean over runs with half the min-max spread after `+/-`, and cost is
per run:

```bash
vlm-exam report --results-directory results
```

Dataset-level mAP@50, mAP@75, and mAP@50:95 for every detection run, with
the mean per configuration:

```bash
vlm-exam detection-report \
    --results-directory results \
    --dataset-directory data/detection/train
```

### Visualize detection predictions

Side-by-side ground truth vs. prediction cards with per-image mAP@50:

```bash
vlm-exam detection-visualize \
    --results-file results/detection/v1/detection_gemini-3.5-flash_low_20260721_114044.jsonl \
    --dataset-directory data/detection/train \
    --output-directory visualizations/detection \
    --max-images 20
```

`--label-mode` controls box labeling: `labels` draws class names on the boxes,
`boxes` draws boxes with a class color legend overlaid on the image (so colors
still map to class names when per-box labels would be too crowded), and `auto`
(default) picks between the two based on label density.

### Generate leaderboards

Regenerates leaderboard charts for all locally saved runs (VQA accuracy plus
detection mAP@50 / mAP@75 / mAP@50:95 per effort level). Bars show the mean
over a model's repeated runs; models with more than one run get a whisker
spanning their lowest and highest run. Use `--group <name>` or `--models`
to filter to a subset.

```bash
vlm-exam leaderboard \
    --results-directory results \
    --dataset-directory data/detection/train \
    --output-directory visualizations/leaderboards
```

### Compile a summary for the web

`vlm-exam summary` generates `web/benchmark_summary.json` (schema version 3).
The current `tasks`, `models[].tasks`, overview totals and completion status
contain only RF-ODv2 (`detection`) and RF-OCRv2 (`text`). Both require one run
per Low/High arm. Historical results never fill missing current coverage.

The payload also contains `historical.detection.v1` (RF-ODv1) and
`historical.ocr.v1` (RF-OCRv1), with archived metrics, repeat counts and recorded
configuration details. Other tasks are excluded from the payload, but their
raw results and explicit execution/reporting paths remain in the repository.

```bash
vlm-exam summary
vlm-exam summary --check
```

Generation is offline and deterministic. Detection scores come from the pinned
release summary; historical detection retains its hash-verified snapshot.
Filter by `--models`, `--group`, or `--effort`. The frontend must read the two
historical sections separately and use `benchmark_id`/`name` for public names.
See [the JSON contract](scripts/homepage_assets/summary_json_schema.md).

### Python

```python
from vlm_exam import load_config, create_provider, create_task, run_benchmark

config = load_config()
task = create_task("vqa")
samples = task.load_samples("/path/to/vqa/dataset")
provider = create_provider("anthropic", model="claude-fable-5")

results = run_benchmark(task=task, provider=provider, samples=samples, effort="high")
```

## Configuration

Model definitions, pricing, lab branding, detection coordinate formats,
and optional fallback routes live in `src/vlm_exam/configs/models.yaml`.
Add a new model by editing this file -- no code changes required for
single-route models.

Each model must declare `detection_coordinate_format` for its native
grounding convention. Valid values are defined by
`DetectionCoordinateFormat` in `src/vlm_exam/tasks/detection.py`:
`yxyx_normalized_0_to_1000`, `xyxy_normalized_0_to_1000`,
`xyxy_normalized_0_to_999`,
`xyxy_normalized_0_to_100`, `xyxy_normalized_0_to_1000_meta_flat`,
`xyxy_normalized_0_to_1000_meta_bbox`, `xyxy_absolute_resized_image`,
`xyxy_absolute_resized_image_bbox`, `xyxy_absolute_original_image`, and
`yxyx_absolute_original_image`.

For rate-limit resilience, list multiple `routes` in priority order.
`FallbackProvider` fails over on 429/quota errors and sticks to the next
route for the rest of the run. Example:

```yaml
  gemini-3.1-pro-preview:
    detection_coordinate_format: yxyx_normalized_0_to_1000
    routes:
      - provider: google
      - provider: openrouter
        provider_model_id: google/gemini-3.1-pro-preview
```

For OpenRouter-only models, set `provider: openrouter` (legacy syntax) or
a single `routes` entry, plus `provider_model_id` with the OpenRouter slug
(e.g. `qwen/qwen3-vl-235b-a22b-instruct`). The short YAML key appears in
result filenames and leaderboards.

## License

Apache 2.0. See [LICENSE](LICENSE).

## Mixed text benchmark

Text evaluation covers single-string extraction, transcription, structured JSON
extraction, and localization/recognition. The frozen dataset contains 600
image/question pairs referencing 576 images. The published evaluation includes
48 models at Low and High effort, with one run per configuration.

Raw runs live in [results/](results/), alongside the other tasks. The existing
[web summary](web/benchmark_summary.json) publishes one `text` task with `overall`
as its primary metric and the four category metrics. Scores use a 0–100 scale;
overall is the mean of all 600 pair scores. Accepted missing responses remain
explicitly unscored, with coverage reported for each metric.

See [AGENTS.md](AGENTS.md#mixed-text-benchmark) for dataset import, inference,
coordinate-format requirements, validation and rendering commands. Charts use the
shared [leaderboard directory](visualizations/leaderboards/).

Structured JSON cards show one image per sample, including large responses.
Small objects show all fields; larger responses add a field agreement map grouped
by JSON parent paths and prioritize mismatches in the value tiles. Purple marks
differences, with the expected value struck through above the model value.
The full-width JSON MAP includes total field and difference counts; the values
header identifies the struck-through Reference and purple Model.
FIELD F1 uses the shared detection-score styling. Long values may be shortened
with an ellipsis; full responses remain in the source result file.

Unparseable JSON uses a centered “Could not parse JSON” message on a pale purple
background. Empty responses show “No response”; the saved FIELD F1 remains visible.

Transcription cards use a single text-only panel with 18 px inline comparisons,
muted struck reference text and purple model differences. Original line breaks
are preserved; long lines wrap and overflow ends with an ellipsis. There is no
word map or pagination. The saved full-response score remains visible, labeled
Character Similarity or Exact Match according to the recorded scoring profile.

Single-value cards retain the question above a centered 24 px answer comparison.
Correct values appear once in neutral text; differences show the muted struck
reference above the purple model value. The saved Exact Match score uses the
same bottom metric position as the other text cards.

Single-value comparisons display embedded newlines and tabs as visible markers
before measuring the text. Transcription keeps line breaks and marks only changed
whitespace (newline, tab or space), using the same insertion/deletion colors.

## Detection benchmark versions

Detection **v2** is the default for run, summary and leaderboard commands.
Use `--detection-version v1` explicitly for the original protocol and historical
results. V1 results do not enter the current detection leaderboard; the combined
JSON exposes them separately under `historical.detection.v1`.

V2 contains 600 images across 120 subsets and results for 56 models. The public
`low`/`high` options select pinned native profiles; Gemini 4 uses native low/high,
not xhigh. This is an intentional stability exception: higher-effort trials
produced too many failed or unusable responses to complete the benchmarks
reliably, so low/high was approved for the Gemini 4 comparison. These results
should not be described as absolute maximum effort.

Missing images download automatically using `ROBOFLOW_API_KEY` into
`data/detection/v2`, with hashes verified against the approved manifest.

```bash
vlm-exam run --task detection --models MODEL --effort low --concurrency 30
vlm-exam detection-v2 validate
vlm-exam detection-v2 summary --workers 4 --check
```

Runs resume from `working-runs/detection/v2` only when their configuration and
inputs match. Use `--max-samples 1` for a non-publishing smoke test. New models
define their native settings in a `detection_v2` mapping in `models.yaml`; both
`run` and `benchmark` use it, with `--config` supported for custom registries.
Standalone `--profile` remains a single-model override. Configuration and publication rules are in
`AGENTS.md`. Versioned results and audit records live in `datasets/detection/v2/`
and `results/detection/v2/`.
