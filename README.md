# vlm-exam

Benchmark suite for Vision Language Models. Compare accuracy, cost, and
speed across frontier VLMs on standardized visual tasks.

## Supported tasks

- **VQA / OCR** -- visual question answering and optical character recognition
- **Object Detection** -- bounding box prediction evaluated with COCO-style mAP

## Leaderboard

The `results/` directory holds the raw benchmark outputs (one JSONL file per
run) and is the single source of truth for the numbers below. Regenerate the
charts at any time with `vlm-exam leaderboard`.

### Counting

![Counting accuracy leaderboard](visualizations/leaderboards/counting_accuracy_low.png)

### Extraction

![Extraction accuracy leaderboard](visualizations/leaderboards/extraction_accuracy_low.png)

### Identification

![Identification accuracy leaderboard](visualizations/leaderboards/identification_accuracy_low.png)

### Reasoning

![Reasoning accuracy leaderboard](visualizations/leaderboards/reasoning_accuracy_low.png)

### Object Detection

![Object Detection mAP@50 leaderboard](visualizations/leaderboards/detection_map50_low.png)

Stricter IoU thresholds:
[mAP@75](visualizations/leaderboards/detection_map75_low.png) |
[mAP@50:95](visualizations/leaderboards/detection_map50_95_low.png)

## Supported providers

- Anthropic (Claude)
- Google (Gemini)
- OpenAI (GPT)
- OpenRouter (any OpenAI-compatible vision model, e.g. Qwen 3.7 Plus,
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
export GOOGLE_API_KEY=...
export OPENAI_API_KEY=...
export OPENROUTER_API_KEY=...
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

Use an LLM judge as a fallback when strict answer matching fails:

```bash
vlm-exam run \
    --task vqa \
    --models gpt-5.5 \
    --effort low \
    --dataset-directory data/vqa/train \
    --match-mode judge \
    --judge-model gemini-3.5-flash
```

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

Accuracy, token usage, and cost tables across all saved runs:

```bash
vlm-exam report --results-directory results
```

Dataset-level mAP@50, mAP@75, and mAP@50:95 for detection runs:

```bash
vlm-exam detection-report \
    --results-directory results \
    --dataset-directory data/detection/train
```

### Visualize detection predictions

Side-by-side ground truth vs. prediction cards with per-image mAP@50:

```bash
vlm-exam detection-visualize \
    --results-file results/detection_gemini-3.5-flash_low_20260707_122136.jsonl \
    --dataset-directory data/detection/train \
    --output-directory visualizations/detection \
    --max-images 20
```

`--label-mode` controls box labeling: `labels` draws class names on the boxes,
`legend` draws boxes only with a color legend below the images, and `auto`
(default) picks based on label density.

### Generate leaderboards

Regenerates leaderboard charts for all locally saved runs (VQA accuracy plus
detection mAP@50 / mAP@75 / mAP@50:95 per effort level). Use
`--group <name>` or `--models` to filter to a subset.

```bash
vlm-exam leaderboard \
    --results-directory results \
    --dataset-directory data/detection/train \
    --output-directory visualizations/leaderboards
```

### Compile a summary for the web

Compiles the newest run per `(task, effort, model)` in `results/` (older
runs are ignored) into a single JSON payload for the benchmark website. It
carries per-task metadata and, for each model, its per-task quality plus
token spend, cost, and inference speed. Each entry is one `(model, effort)`
pair with a unique `id`, so the same model appears once per effort; the
top-level `efforts` array lists every effort present. Pass
`--dataset-directory` to include detection mAP (otherwise detection quality
metrics are omitted while its efficiency metrics are kept). Filter with
`--group <name>` or `--models`, and restrict to a single effort with
`--effort` (defaults to all efforts). The frontend supplies its own lab
logos and live pricing, so those are intentionally left out. The output is
deterministic: `generated_at` is derived from the newest included run, so
regenerating without new results produces a byte-identical file.

```bash
vlm-exam summary \
    --results-directory results \
    --dataset-directory data/detection/train \
    --output-file web/benchmark_summary.json
```

The output has this shape (abbreviated):

```json
{
  "generated_at": "2026-07-10T08:11:31Z",
  "efforts": ["low"],
  "tasks": [
    {
      "key": "ocr",
      "name": "OCR",
      "primary_metric": "similarity",
      "metrics": [
        { "key": "similarity", "label": "Mean Similarity", "unit": "percent" }
      ]
    },
    {
      "key": "detection",
      "name": "Detection",
      "primary_metric": "map50",
      "metrics": [
        { "key": "map50", "label": "mAP@50", "unit": "percent" },
        { "key": "map75", "label": "mAP@75", "unit": "percent" },
        { "key": "map50_95", "label": "mAP@50:95", "unit": "percent" }
      ]
    }
  ],
  "models": [
    {
      "id": "gpt-5.6-sol:low",
      "key": "gpt-5.6-sol",
      "name": "GPT-5.6 Sol",
      "lab": "openai",
      "effort": "low",
      "tasks": {
        "ocr": {
          "primary_metric": { "name": "similarity", "value": 90.73 },
          "metrics": { "similarity": 90.73 },
          "sample_count": 37,
          "evaluated_sample_count": null,
          "failed_sample_count": 0,
          "tokens": { "input": 46626, "output": 31917, "total": 78543, "average_per_sample": 2122.8 },
          "cost": { "total_usd": 1.19064, "average_per_sample_usd": 0.032179 },
          "speed": { "total_seconds": 496.636, "average_seconds_per_sample": 13.423 },
          "timestamp": "2026-07-10T07:33:33Z"
        },
        "detection": {
          "primary_metric": { "name": "map50", "value": 46.23 },
          "metrics": { "map50": 46.23, "map75": 20.85, "map50_95": 23.26 },
          "sample_count": 250,
          "evaluated_sample_count": 250,
          "failed_sample_count": 0,
          "tokens": { "input": 629173, "output": 180400, "total": 809573, "average_per_sample": 3238.3 },
          "cost": { "total_usd": 8.557865, "average_per_sample_usd": 0.034231 },
          "speed": { "total_seconds": 3632.994, "average_seconds_per_sample": 14.532 },
          "timestamp": "2026-07-09T21:50:44Z"
        }
      },
      "overall": {
        "task_count": 6,
        "sample_count": 513,
        "tokens": { "input": 976102, "output": 237183, "total": 1213285, "average_per_sample": 2365.1 },
        "cost": { "total_usd": 11.996, "average_per_sample_usd": 0.023384 },
        "speed": { "total_seconds": 5094.531, "average_seconds_per_sample": 9.931 }
      }
    }
  ]
}
```

All quality metrics are percentages (0-100). QA tasks report `accuracy`,
OCR reports `similarity`, and detection reports `map50` (primary), `map75`,
and `map50_95`. No-data contract: a task absent from a model's `tasks`
means the model was not benchmarked on it; `primary_metric: null` with
empty `metrics` means quality could not be computed (e.g. detection without
`--dataset-directory`) while efficiency numbers remain valid.
`evaluated_sample_count` is detection-only (null elsewhere): the number of
images the mAP was actually computed on, which should equal `sample_count`.
The `cost` numbers are estimates from token usage and the config's static
pricing; the site should recompute cost from its own live pricing feed.

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
`xyxy_absolute_provider_upload`, `xyxy_absolute_original_image`, and
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
