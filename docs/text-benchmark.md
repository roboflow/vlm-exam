# Mixed text benchmark

The independent `text` task reads Roboflow JSONL exports. Run files belong in
`results-text/`; its low/high, three-repeat protocol does not change the legacy
six-task results or web summary. Do not commit smoke runs or private early-access
outputs.

### Dataset and scoring

Rows contain exactly `image`, `prefix`, and `suffix`, all strings. `prefix` is
JSON with `task`, `question`, and, for `localization_recognition`, optional
`specification`. Supported tasks are `single_string`, `transcription`,
`structured`, and `localization_recognition`. `suffix` contains the reference
text or JSON. Localization references are arrays of `{"bbox": [x1,y1,x2,y2],
"text": "..."}` in original-image pixels; text may be null.

Read all questions and answers from the dataset. There is no repository subset
registry. Optional `subset` and `scoring_profile` fields inside the prefix are
benchmark metadata and are never sent to the model. Without a subset label,
reports group by a stable prefix hash. Two subsets with identical prefixes
cannot be distinguished unless the export supplies their labels.

Default scoring is deterministic: exact string match, transcription character
similarity, typed nested JSON field F1, or joint box+text F1. Exact strings
normalize NFC, line endings, and outer whitespace only. JSON field F1 penalizes
missing and extra fields and distinguishes strings, numbers, booleans, and null.
Localization requires one-to-one matches with IoU strictly greater than 0.5 and
exact text. Invalid regions count as unmatched predictions. JSON fences are
parsed but marked noncompliant; prose, duplicate keys, and non-finite numbers
are rejected. Provider failures are unscored. Fresh runs require recovery before release;
the frozen preliminary release retains its explicitly accepted gaps.

A transcription prefix may explicitly request `scoring_profile: "exact"` or
`"italian_soft_wraps"` (join single line wraps, preserving paragraphs). The
existing 600-pair export does not contain these fields. It therefore receives
uniform category defaults under `mixed-text-v2`, including passport MRZ and
Italian documents. To retain the earlier exploratory scoring exceptions, put
them into the dataset before exporting. Do not infer rules from question hashes
or compare these fresh runs with earlier curated scores as one protocol.

The loader preserves multiple questions on one image and checks images, JSON,
paths, and reference boxes. Images must be upright with EXIF orientation 1.
Roboflow may re-encode images, so every compared model must use the same export.
Dataset bytes, prompts, references, scoring, coordinate conventions, and
inference settings are frozen in run provenance. Changing any of these requires
fresh repeats. Resuming rejects changed snapshots or model preprocessing.

### Import, run, and resume

Install with `uv sync --extra dev`, then configure provider credentials as for
existing tasks. The deterministic text task does not need an LLM judge key.
Download the Roboflow JSONL ZIP and validate its expected pair count:

```bash
uv run vlm-exam text-import ~/Downloads/DATASET.zip --expected-pairs 600
uv run vlm-exam text-validate --dataset-directory data/text/train --expected-pairs 600
```

Import refuses to overwrite a snapshot; use `--output-directory` for another
export. The checked export has 600 pairs referencing 576 images. A small smoke
run, followed by a complete low/high benchmark with three repeats each:

```bash
uv run vlm-exam run --task text --models gpt-6.1-sol --effort low \
  --dataset-directory data/text/train --max-samples 4 \
  --output-directory results-text-smoke --concurrency 2
uv run vlm-exam text-benchmark --models gpt-6.1-sol --max-parallel 2
```

Use `--dataset-root` for a different root containing `text/train`.
`text-benchmark` prints its per-run log paths under `logs/text/` before starting.
To resume failed image/question pairs in one saved run:

```bash
uv run vlm-exam run --task text --models gpt-6.1-sol --effort low \
  --dataset-directory data/text/train --output-directory results-text \
  --resume-file results-text/RESULT.jsonl
```

The summary keeps incompatible snapshots, selections, and inference profiles
separate. It reports category/subset means and coverage. An overall mean mixes
several metrics and must not be described as uniform accuracy.

### Render saved results

These commands make no model calls. Cards use the existing Playground frame:
strings show answer differences, JSON shows whole-token differences with extra
pages when needed, and localization shows region overlays and text crops,
including missing and extra regions. Full responses remain in the result files.

```bash
uv run vlm-exam text-visualize results-text/RESULT.jsonl \
  --dataset-directory data/text/train --category single_string --max-samples 10
uv run vlm-exam text-visualize results-text/FIRST.jsonl \
  --compare-file results-text/SECOND.jsonl --dataset-directory data/text/train \
  --category structured --output-directory visualizations/json-comparisons
uv run vlm-exam text-visualize results-text/RESULT.jsonl \
  --dataset-directory data/text/train --category localization_recognition
uv run vlm-exam text-leaderboard --results-directory results-text
```

Cards default to `visualizations/`; use separate output directories
for separate selections. Comparison PNGs stack the first run's native cards
above the second run's cards with no added title, padding, or footer. They must
share effort, snapshot, and image/question selection. Failed pairs are skipped.
Rendering verifies image hashes and recomputed scores using recorded upload
dimensions and coordinate formats.

Leaderboards default to `visualizations/leaderboards/`, using
`text_<category>_<effort>.png` alongside the existing task charts. A command
accepts one snapshot/selection; use explicit alternate output directories for
other snapshots or filtered selections. They render each category at low/high effort and
require three complete scored repeats. `--allow-incomplete` permits full-dataset
runs with fewer repeats and visibly labels their charts preliminary. Partial
and failed runs are never ranked. Their compact manifest fingerprints the source summary, display identities,
renderer and PNGs. It does not duplicate the summary.

Before updating the PR, run Ruff check/format, the existing test suite,
`vlm-exam validate`, and `vlm-exam summary --check`. Validate a real export and
visually inspect offline render fixtures for strings, JSON, localization,
empty/invalid responses, and multi-page JSON. New test files are intentionally
outside this PR at the user's request.

The committed `results-text/` collection is an explicitly single-run release.
Keep its preliminary labels and failure coverage; do not present it as the
three-repeat protocol. Rebuild its charts with the display-only
`--model-labels results-text/model-labels.json` option and `--allow-incomplete`.

All percentage leaderboards use `plot_accuracy_chart`, a thin adapter over
`plot_metric_chart`; task code supplies values and labels only. Keep layout,
fonts, margins, row spacing and bars in the shared renderer, and use
`save_leaderboard_chart` for the common 150 DPI PNG export.

Use `_layout_leaderboard` for fixed vertical padding (approximately 118 px
above the title at 150 DPI for full leaderboards), capped for short charts.
Do not scale vertical padding as a percentage of a growing leaderboard.

## Shared rendering and publication

Cards and leaderboards accept the same `--config` and `--model-labels` options.
The latter adds display-only names and lab affiliations, never provider routes
or prices. For the committed release, pass `--model-labels
results-text/model-labels.json` to either rendering command. Leaderboards also
accept the existing `--models` and `--group` filters, plus `--effort`.

`text-benchmark` supports `--config`, `--efforts`, `--repeats`, and
`--first-repeat`, using the same job planner as the existing benchmark. These
controls do not authorize rerunning the frozen release. An explicit output
path passed to `run` is always honored; only the omitted text-task default
selects `results-text`.

The frozen text source files remain in `results-text/`; the website receives
one payload, `web/benchmark_summary.json`. There is no separate text summary or
file-index endpoint. The existing commands discover the sibling `results-text/`
directory and validate its reviewed release policy before JSON publication:

```bash
uv run vlm-exam validate
uv run vlm-exam summary --dataset-directory data/detection/train
uv run vlm-exam summary --check
```

To refresh only the text PNGs after a text-only change, use
`text-leaderboard --allow-incomplete --model-labels results-text/model-labels.json`.
The PNGs remain unchanged in this JSON migration. Their existing eligibility
rule requires all 600 scored pairs. The unified JSON exposes category coverage
independently so consumers can rank every fully scored category.
`summary --check` checks the unified JSON and compact chart manifest without
inference. CI uses the existing validate and summary steps.

The independent `configs/text_release.json` policy pins the 48 public models,
both efforts, one run per configuration, all 600 pair IDs, frozen inference
hashes, and exact accepted failure IDs. Changing result files does not redefine
that policy. Native fresh benchmarking still defaults to three repeats.

## Website JSON contract

The existing `generated_at`, `efforts`, `tasks`, and `models` envelope is retained.
Each existing `model:effort` row gains four ordinary entries under `tasks`:
`text_single_string`, `text_transcription`, `text_structured`, and
`text_localization_recognition`. Existing OCR and extraction entries are unchanged.

Each text task uses the established `primary_metric`, `metrics`, `metric_runs`,
`run_count`, `timestamps`, sample-count, token, cost and speed fields. The primary
metric is `score` (mean normalized task score, percent 0–100), not uniform answer
accuracy. Token and speed aggregation reuse the existing summary helpers. Cost
is an estimate from the registered token prices, as for the other tasks.

Additive per-task fields retain the evidence:

- `protocol`: frozen release name, `repeats: 1`, `preliminary: true`, and
  `status: complete` or `complete_with_gaps`. This is independent of the existing
  six-task protocol block; a complete single-run release is not three repeats.
- `provenance`: dataset, selection, inference and scoring identities, coordinate
  convention, and source result filenames with SHA-256 hashes.
- `coverage`: per-run expected, scored and failed counts, observed score and
  score bounds, all quality values on the same 0–100 scale.
- `subsets`: the corresponding coverage and scores for each prompt subset.

A category with accepted missing responses has `primary_metric: null` and empty
`metrics`/`metric_runs`. Its observed score remains diagnostic coverage metadata;
it is not a full-coverage ranking value. Fully scored categories remain rankable.

The current six-task overview is preserved. `overview_tasks` explicitly names
its task selection; the new task metadata sets `include_in_overall: false`.
Existing model `overall` cost/token/speed totals and six-task protocol status do
not change. Consumers must use that overview selection for scores as well as
pooled efficiency instead of averaging every newly registered task.

### CV Model Playground integration

The existing `refresh-benchmarks` script can keep fetching the same file and
reconciling model mappings. `getBenchmarkSummary()` keeps its current loader.
The regular task leaderboard needs no alternate data source or normalization.
To expose new pages, extend `lib/benchmarks/types.ts` task keys, slugs and
`features/evals/benchmark/copy.ts`; use the same task tables, ranking and pricing
helpers. Extend the existing result type for protocol/coverage metadata and show
preliminary run and coverage labels. Use `overview_tasks` in overview averaging,
with the legacy six-task list as the fallback for older snapshots.

Task, model-comparison, and model-profile views may show all ten task keys, while
the overview mean and its efficiency pool remain the six original tasks.
A JSON refresh alone does not register the new website routes. Dataset example
images are handled by the existing examples workflow, not embedded in this JSON.

The imported release preserves its recorded inference profiles. Some differ
from the current bundled model defaults, and display identities do not add
inference support. Fresh runs with bundled defaults are separate experiments;
they must not resume or be pooled with incompatible frozen runs. The runner
checks inference hashes on resume and publication rejects profiles outside the reviewed release policy.

All leaderboard types use a fixed 118 px title gap at the standard 150 DPI
export, independent of model count. The averaging footnote is omitted from PNGs;
repeat counts and statistics remain in the summaries, and spread whiskers remain.
