# Agent Guidelines

Coding standards for AI agents working on this codebase.

## License header

Every `.py` file must begin with this Apache 2.0 header:

```python
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
```

## Type annotations

All function and method signatures must have full type annotations for every
parameter and the return type. No exceptions.

## Documentation

- Use Google-style docstrings on all **public** classes, functions, and
  constants.
- Do NOT add docstrings to private/internal symbols (prefixed with `_`).
- Do NOT add file-level module docstrings. Packages, files, functions,
  variables, and classes should be named clearly enough to be
  self-documenting.

## Comments

Do NOT write code comments unless documenting a non-obvious hack,
workaround, or performance trick (e.g. "exploiting numpy broadcast to
avoid a loop"). Never narrate what the code does.

## Naming

- Names must be short, concise, and easy to understand.
- No abbreviations. Favor clarity over brevity.
- Prefix private symbols with `_` (functions, classes, constants, methods).
- Use `snake_case` for functions, methods, and variables.
- Use `PascalCase` for classes.
- Use `UPPER_SNAKE_CASE` for module-level constants.

## Style

- No emoji in code or documentation.
- Run `ruff check` and `ruff format` before committing.
- Keep imports sorted (enforced by ruff `I` rules).

## Benchmark protocol

The six original tasks in `results/` follow the protocol defined once
as `PROTOCOL` in `src/vlm_exam/protocol.py` and read by `validate`,
`benchmark`, and the historical leaderboards. The website overview uses the
five-task selection described under "Web summary". Text currently uses one run per
model/effort, as described under "Mixed text benchmark" below:

1. **Three runs per configuration.** Each `(task, effort)` is run three
   times and all three result files are committed. A single run is not a
   benchmark entry; model output is stochastic and one file cannot tell a
   real gap from run-to-run noise.
2. **Report the mean.** Quality metrics (judge and strict accuracy, OCR
   similarity, mAP) are the arithmetic mean of the per-run values. Token,
   cost, and time totals are the mean of one complete run, so three
   repeats are not reported as three times the cost. Failed sample counts
   are summed.
3. **Low and high on every task.** Every task (OCR, extraction, counting,
   identification, reasoning, detection) is run at both `--effort low` and
   `--effort high`. That is 6 tasks x 2 efforts x 3 runs = 36 result files
   per model. Both effort levels have their own leaderboard PNGs.
4. **All of it in the web JSON.** `web/benchmark_summary.json` carries the
   overview protocol (`protocol.repeats`, `protocol.repeats_by_task`,
   `protocol.efforts`, `protocol.tasks`, `protocol.runs_per_model`), one entry
   per `(model, effort)`, per-task
   `metrics` (means), `metric_runs` (the per-run values), `run_count`, and
   `timestamps`, and per model a `protocol` block with `name`
   (`full` or `legacy`), `status` (`complete`, `incomplete`, `legacy`),
   `runs_present`, and `runs_required`.

### Rules for `results/`

- `results/` is the single source of truth; every command globs every
  file in it. Repeats are plain files: every file sharing a
  `(task, model, effort)` triple is one repeat. There is no repeat field or
  manifest. Never delete or cherry-pick repeats to move a number; if a run
  is broken, fix the cause, re-run it, and remove only the broken file.
- Only commit full-dataset runs. Never commit partial or smoke runs (any
  run produced with `--max-samples`); `validate` flags a file with fewer
  samples than the largest run of its task as a partial run.
- Only average runs produced under one protocol. A change to prompts,
  coordinate formats, judge settings, or image preprocessing invalidates
  the existing repeats of the affected configurations. Obtain authorization
  for fresh runs under that task's protocol; never silently rewrite provenance.
- Images are EXIF-transposed on load before being sent to any provider.
  Datasets whose images carry EXIF orientation tags will therefore produce
  runs that are not comparable to runs made before this behavior existed;
  re-run all models on such a dataset rather than mixing old and new runs.
- `vlm-exam run --resume-file <file>` re-runs only the failed samples,
  writes the merged complete file, and deletes the source file, so a
  partial run and its completion are never both counted as repeats.
  `validate` and `summary` warn about any run with failed samples and
  print the resume command; resume it before committing unless the exact gaps
  are explicitly accepted by the text release policy. Never retry accepted gaps
  without new authorization.

### Legacy models (`benchmark_protocol: legacy`)

- Models benchmarked before this protocol have one low run per task and
  one high run for reasoning (7 of 36 files). They are marked
  `benchmark_protocol: legacy` in `configs/models.yaml`. The field is
  temporary: it is a backlog of models to backfill, not a second tier.
- `validate` reports coverage for every model, legacy or not, and emits
  GitHub warnings for legacy gaps so they stay visible on every PR. It
  fails only for models without the marker. `--strict` fails on legacy
  gaps too, for the day the backfill is done.
- Never add `benchmark_protocol: legacy` to a new model. A new model must
  ship all 36 files (or not be added yet). To backfill a legacy model, run
  its missing files with `vlm-exam benchmark --models <key>`, then delete
  the `benchmark_protocol` line.
- Legacy rows stay in the leaderboards with their `run_count` and
  `protocol.status: legacy` until re-run; do not drop them.

### Commands

- Run the whole protocol for a new model (36 processes, parallel,
  staggered 2 s so timestamps never collide, one log each under `logs/`,
  low first and high as low finishes):

```bash
vlm-exam benchmark --models <key>
```

  `--tasks`, `--efforts`, `--repeats`, and `--first-repeat` narrow it
  (e.g. top up two missing high repeats of detection with
  `--tasks detection --efforts high --repeats 2 --first-repeat 2`).
  `--max-samples` exists for smoke tests only. The command prints every
  log path up front and a per-run status table at the end. Under the
  hood each process is `vlm-exam run --concurrency <n>` with the per-task
  table in `TASK_CONCURRENCY` (detection 6, reasoning 4, OCR 3,
  extraction, counting, identification 2), which keeps a full effort
  level under 15 minutes with Claude-class latency; drop detection to 4
  if the provider returns 429s. Running `vlm-exam run --repeats 3` by
  hand is equivalent; filenames carry a second-resolution timestamp and
  the CLI suffixes `_2`, `_3` on collision, so parallel launches are safe.
- Check the repository before committing:

```bash
vlm-exam validate
```

  It prints a coverage table (`36/36`, `7/36 legacy`, ...) followed by
  every problem with the exact command that fixes it: missing or surplus
  runs, missing effort levels, partial runs, missing strict/judge
  verdicts, failed samples, and files for models absent from
  `models.yaml`. Exit code 1 on any violation. `--verbose` expands legacy
  gaps; `--format github` adds annotations and a step summary.
- CI (`.github/workflows/ci.yml`) runs `vlm-exam validate --format github`
  and `vlm-exam summary --check` on every pull request and push to `main`,
  plus `ruff check`, `ruff format --check`, and `pytest`. A PR that adds a
  model without all 36 files, or that changes `results/` without
  regenerating `web/benchmark_summary.json`, fails.

## Scoring: strict and judge are two independent metrics

- Counting, extraction, identification, and reasoning report two accuracy
  numbers for every model, past and future. Both are always computed and
  stored; neither is a fallback for the other.
  - `strict`: the deterministic rule (`strict_match` normalization for
    extraction, identification, reasoning; `parse_count` equality for
    counting). It measures answer correctness and format compliance
    together.
  - `judge`: the LLM judge (`gemini-3.5-flash`, temperature 0, the prompt
    in `src/vlm_exam/judge.py` plus the task's `judge_guidance`) scores
    every sample, including ones that pass strict. It measures answer
    correctness while tolerating phrasing.
- Per sample, `metadata.strict_correct` and `metadata.judge_correct` are
  both recorded; the `correct` column equals `judge_correct`. Provider
  errors and empty responses are false under both rules and never reach
  the judge. `metadata.match_method` is legacy and must not appear in
  these four tasks' files.
- `vlm-exam run` always produces both verdicts for these tasks and needs
  `GOOGLE_API_KEY` for the judge; there is no strict-only mode. OCR
  (similarity) and detection (mAP) are single-metric and ignore the judge.
- Never commit a run for these tasks whose samples lack either flag.
  `report`, `leaderboard`, and `summary` refuse such runs. Backfill with
  `vlm-exam rescore <file-or-directory>`: it recomputes strict offline,
  judges every stored prediction, and skips samples that already carry
  both verdicts (use `--force` to re-judge).
- Do not change the judge model, temperature, prompt, or task guidance
  without re-scoring every committed run with `vlm-exam rescore --force`;
  judge numbers are only comparable under one fixed protocol.
- Headline versus secondary: `accuracy_judge` is the `primary_metric` in
  `web/benchmark_summary.json` and the `{task}_accuracy_{effort}.png`
  charts; `accuracy_strict` is exported alongside it in the same payload
  and rendered as `{task}_accuracy_strict_{effort}.png`. The payload's
  top-level `scoring` block names the judge model and both metric keys.

## Reference models

- SAM 3, YOLO-E, and future local reference detectors are comparison
  baselines, not VLM benchmark entries. Their code, environments, results,
  prompts, documentation, and rendered leaderboards live under `reference/`
  or `src/vlm_exam/reference/`.
- Never put reference run files in `results/`, add reference model keys to
  `src/vlm_exam/configs/models.yaml`, or include reference rows in the main
  VLM leaderboard or `web/benchmark_summary.json`.
- Full reference runs use effort `reference` and belong in
  `reference/results/`. Partial and smoke runs remain local.
- The committed reference prompt modes are class names, image-conditioned v1,
  v2 none, and v2 overlay. Treat other prompt-generation experiments as local
  scratch work unless their scope is explicitly approved.
- Regenerate the separate mixed comparison charts with
  `vlm-exam reference-detection-leaderboard`; output belongs in
  `reference/leaderboards/`.
- Keep model-specific dependencies and adapters in each model's isolated
  project under `reference/<model>/`. The main package must not depend on
  PyTorch, Transformers, Ultralytics, or model weights.

## Web summary

- `overview_tasks` is `["text", "counting", "identification", "reasoning",
  "detection"]`. Text replaces OCR and Data Extraction in the overview; keep
  their individual task results available. Overall token, cost and elapsed-time
  totals include only the five selected tasks, averaging repeats within each
  task before summing. Recompute per-sample averages from the selected totals.
- The web protocol and every model's run status use the same five tasks across
  Low and High: Text requires one run per effort, the other four require three,
  for 26 files per complete model. `protocol.repeats` remains the default of 3;
  `protocol.repeats_by_task` explicitly records the per-task requirements.
  Status describes run inventory; response coverage and accepted gaps remain
  explicit in the task metrics and coverage fields. Keep legacy status for
  legacy models with missing runs. Missing Text runs cannot be filled by OCR
  or Extraction runs. This publication selection does not change the historical
  six-task inference planner or authorize additional runs.
- Omit `subsets` and `provenance` from the web JSON. Retain them in raw results
  and internal summaries for audit and validation. Preserve task metrics,
  metric runs, counts and coverage in the website payload.
- Regenerate `web/benchmark_summary.json` and commit it in every PR so the
  website payload never drifts from `results/` and `configs/models.yaml`.
- Rebuild it with the detection dataset so detection mAP is included:

```bash
vlm-exam summary --dataset-directory data/detection/train
```

- The command compiles all efforts by default, emitting one entry per
  `(model, effort)` pair; pass `--effort` only to restrict to one level.
  Each task entry averages every repeat of that configuration (see
  "Benchmark protocol" above).
- The output is deterministic given `results/`: `generated_at` derives
  from the newest included run, so an unchanged diff after regeneration
  means the results did not change.
- `vlm-exam summary --check` rebuilds the payload in memory and exits 1 if
  the committed file differs. CI runs it without the detection dataset, so
  detection mAP fields are excluded from that comparison; everything else
  must match.
- The file is a generated artifact; never hand-edit it.

## Leaderboard charts

- Regenerate the leaderboard charts in `visualizations/leaderboards/` and
  commit them in every PR that changes `results/`, so the tracked PNGs
  never drift from the underlying runs. Both effort levels are tracked
  (`*_low.png` and `*_high.png`):

```bash
vlm-exam leaderboard --dataset-directory data/detection/train
vlm-exam efficiency-report --effort low
vlm-exam efficiency-report --effort high
```

  `leaderboard` renders every effort present in `results/`;
  `efficiency-report` renders one effort per call.

## Adding and benchmarking models

- In `configs/models.yaml`, each model has an ordered `routes` list (or
  legacy single `provider` field). The vlm-exam model **key** is used in
  result filenames and leaderboards. Each route's `provider_model_id` is
  the upstream API id; when omitted, the model key is used.
- Before adding any model, research its capabilities online using **official
  provider documentation** (API reference, cookbooks, model cards). Record
  what you find, especially for detection: the provider's **native prompt
  wording**, **output JSON field names**, axis order, and coordinate space.
- Do not assume an existing `box_2d` prompt variant matches a provider's
  documented schema (e.g. separate `x_min`/`y_min`/`x_max`/`y_max` keys
  versus a four-number `box_2d` array). Map to an enum value only after a
  local format probe confirms mAP on a 50-image detection subset
  (`--max-samples 50`, written outside `results/`), compared against an
  already-benchmarked model of the same family on the same images.
- Set the required `detection_coordinate_format` per model after that
  research and probe. Valid values are the `DetectionCoordinateFormat` enum
  strings in `src/vlm_exam/tasks/detection.py`: `yxyx_normalized_0_to_1000`,
  `xyxy_normalized_0_to_1000`, `xyxy_normalized_0_to_999`,
  `xyxy_normalized_0_to_100`,
  `xyxy_normalized_0_to_1000_meta_flat`,
  `xyxy_normalized_0_to_1000_meta_bbox`, `xyxy_absolute_resized_image`,
  `xyxy_absolute_resized_image_bbox`, `xyxy_absolute_original_image`, and
  `yxyx_absolute_original_image`. The
  format follows the model, not the route
  -- the same weights use the same box convention on Google direct and
  OpenRouter. Cite the source URL when choosing a format (in the PR
  description).
- Add fallback routes when a provider has tight rate limits. Example:
  `gemini-3.1-pro-preview` uses Google first, then OpenRouter on 429.

## Running long jobs (logging)

- ALWAYS tee long-running command output to a tailable log file (e.g.
  `logs/<task>_<models>_<effort>.log`) using unbuffered output
  (`PYTHONUNBUFFERED=1 ... 2>&1 | tee <logfile>`), so progress can be
  followed independently.
- ALWAYS give the user the log file path as soon as processing starts, so
  they can `tail -f` it without asking. Never make the user request
  progress; the link must be provided up front.

## Git workflow

- Do not commit or push directly to `main`. Create a branch from the
  current `main` with a short descriptive name (e.g.
  `feat/provider-image-preprocessing`) and open a pull request, unless
  the user explicitly instructs otherwise.
- Before creating the branch and pushing, ask the user to confirm that
  workflow (branch name and intent to open a PR).

## Mixed text benchmark

### Dataset, runs and coordinates

- Text is one task with four scored categories: `single_string`, `transcription`,
  `structured`, and `localization_recognition`. Its raw JSONL files belong in
  **`results/`**, alongside all other VLM tasks. Use the existing model registry,
  `summary`, `validate`, card primitives and shared chart layout. Do not create
  separate result roots, web payloads, display registries or task documentation.
- **For now, run each text model once at Low and once at High effort.**
  `TEXT_BENCHMARK_PROTOCOL` and `text-benchmark` default to one run per
  `(task, model, effort)`. The original six tasks retain three repeats. Do not
  start additional text repeats or rerun completed configurations without
  authorization. Single-run chart labels remain preliminary.
- The frozen export is Roboflow project `roboflow-jvuqo/vlm-exam-text`, version 2:
  <https://app.roboflow.com/roboflow-jvuqo/vlm-exam-text/2>. It contains 600 pairs
  referencing 576 images: 230 single-string, 50 transcription, 230 structured,
  and 90 localization/recognition pairs. Use the same export for every model.
- Import rows have exactly `image`, `prefix`, and `suffix` string fields. Prefix
  JSON contains the category in `task`, `question`, and optional localization
  `specification`; suffix contains the reference text or JSON. Read prompts and
  references from the dataset. Optional `subset` and `scoring_profile` are
  scoring metadata, never model instructions. Do not infer subset-specific rules
  from filenames or question hashes. Images must be upright (EXIF orientation 1).
- **Before running localization/recognition, establish the model's native
  bounding-box format.** Research official provider evidence and perform the
  same 50-image detection-format comparison described under "Adding and
  benchmarking models". Pin `detection_coordinate_format` in the shared model
  configuration: axis order, normalized scale versus absolute pixels, JSON
  shape, and whether pixels refer to the original or uploaded/resized image.
  Do not guess a format or substitute another model. Keep unresolved formats
  pending. The text runner uses that setting to build prompts and convert
  predictions back to original-image coordinates.
- Localization ground truth uses `{"bbox": [x1, y1, x2, y2], "text": ...}` in
  original-image pixels; text can be null. Record actual upload dimensions for
  resized-image formats. Pin provider routing, preprocessing, token limits and
  reasoning settings before inference. Changing dataset bytes, prompts,
  references, coordinates or inference settings creates an incompatible run.

### Commands

Install with `uv sync --extra dev` and configure the relevant provider keys.
Text scoring is deterministic and does not require a judge API key.

```bash
uv run vlm-exam text-import ~/Downloads/DATASET.zip --expected-pairs 600
uv run vlm-exam text-validate --dataset-directory data/text/train --expected-pairs 600
uv run vlm-exam text-benchmark --models <key> --max-parallel 2
```

Import refuses to overwrite an existing snapshot. `text-benchmark` uses the
shared job planner, prints each log path under `logs/text/`, and writes to
`results/`. `--dataset-root`, `--config`, and `--efforts` select the dataset,
configuration and effort levels. Put smoke runs outside `results/`:

```bash
uv run vlm-exam run --task text --models <key> --effort low \
  --dataset-directory data/text/train --max-samples 4 \
  --output-directory /tmp/text-smoke --concurrency 2
```

For authorized recovery of failed pairs, resume the existing run:

```bash
uv run vlm-exam run --task text --models <key> --effort low \
  --dataset-directory data/text/train --resume-file results/RESULT.jsonl
```

Resume verifies the snapshot and inference profile and preserves successes.
Do not resume frozen files with incompatible bundled defaults. The reviewed
`src/vlm_exam/configs/text_release.json` pins the 48-model publication inventory,
both efforts, one run each, 600 pair IDs, inference hashes and exact accepted
failure IDs. The release has nine accepted gaps: Sonnet 5.5 High one, Grok 4.7
High five, and Muse Spark 1.3 Low two / High one. Keep them unscored; do not retry
or change the policy to make validation pass. New publications require an
explicitly reviewed inventory change. Private results stay outside this repo.

### Scoring and publication

Default scores are exact string match, transcription character similarity,
typed JSON field F1, and joint box/text F1. Localization uses one-to-one matches
with IoU strictly greater than 0.5 and exact text. Invalid regions count as
unmatched predictions. Provider failures are unscored. Explicit transcription
profiles can request `exact` or `italian_soft_wraps`; never silently change
recorded scoring profiles or merge incompatible runs.

Publish through `vlm-exam summary --dataset-directory data/detection/train` to
`web/benchmark_summary.json`. One `models[].tasks.text` entry has `overall` as
its primary metric and four category metrics, following detection's `metrics`
and `metric_runs` contract. Scores are 0–100; overall averages all 600 pair
scores, not the four category means equally. Preserve standard run statistics,
per-metric coverage. Keep source checksums and subset metadata in internal
summaries and raw results, outside the web payload. Missing responses
omit the affected ranking metrics; coverage retains observed scores and bounds.
`run_count` records repeats; do not add another text protocol block. Preserve
the five-task overview and its explicitly selected efficiency pool.

### Rendering and checks

Render text cards or just text charts from saved runs without inference:

```bash
uv run vlm-exam text-visualize results/RESULT.jsonl \
  --dataset-directory data/text/train --category localization_recognition
uv run vlm-exam text-leaderboard
```

The shared `leaderboard` command also dispatches text charts to the same
renderer. Leaderboards belong in `visualizations/leaderboards/`; cards default
to `visualizations/`. Both use the shared model configuration and layout. Apply
the common 118 px title gap at 150 DPI and omit the averaging footnote without
changing statistics or spread whiskers. Text PNG rankings require all 600
scored pairs; JSON exposes category coverage independently. Preserve source,
renderer and PNG fingerprints in the compact chart manifest.

Before updating this PR, run Ruff check/format, existing tests, `validate`, and
`summary --check` (including text coverage and chart freshness). Verify cards
for strings, JSON, localization, empty/invalid responses and pagination when
changing rendering. Add focused regressions to existing test modules; do not
create new test files. Moving unchanged result files does not require rebuilding
PNGs or recomputing inference.
