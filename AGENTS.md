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

Detection v2 is the default. The historical six-task `PROTOCOL` remains the
source of QA defaults and the explicit v1 contract. Current runs use:

- OCR, extraction, counting, identification and reasoning: three repeats at
  Low and High, unchanged. Report means across repeats; token/cost/time totals
  describe one mean full run, not three times the cost.
- Detection v2: one run per recorded native minimum/maximum configuration.
  `low`/`high` CLI aliases select those native profiles. Enabled-only models
  remain neutral single-arm measurements; never duplicate them into two lanes.
- Text: one run at Low and High, under the frozen mixed-text release policy.
- The website overview uses Text, counting, identification, reasoning and
  Detection v2. Its per-task repeat requirements are 1, 3, 3, 3 and 1 respectively.
  Historical detection v1 remains available only with `--detection-version v1`.

### Rules for `results/`

- QA flat JSONL files retain their repeat-pool semantics. Detection v1 files
  live in `results/detection/v1/` and are excluded from current results.
  V2 uses the hash-pinned manifest and inventory in `datasets/detection/v2/`;
  its compressed primary archives and diagnostics must never be pooled.
- Only publish full-dataset runs. `--max-samples` is smoke-only and cannot
  publish. Keep partial checkpoints outside the frozen results tree.
- Never cherry-pick by score or mix prompts, formats, ground truths,
  preprocessing, or native efforts without explicit authorization and provenance.
- Images are EXIF-transposed before provider submission. Preserve original
  image identities and approved coordinate frames; do not silently alter pixels.
- QA `--resume-file` retries failures and merges a complete replacement.
  V2 resumes by working directory and only republishes via strict validation,
  preserving superseded records. Frozen terminal outcomes and accepted gaps
  require authorization before retry or promotion.

### Legacy models (`benchmark_protocol: legacy`)

Historical QA gaps remain visible as warnings, or errors under `--strict`.
Do not add a legacy exemption simply to make a new model pass. Detection v2
publication is governed by its explicit measured inventory, not by pretending
that historical v1 repeats satisfy v2 coverage. The web protocol reports missing
current coverage without filling it from older datasets.

### Commands

```bash
vlm-exam benchmark --models <key>
vlm-exam run --task detection --models <key> --effort low --concurrency 30
vlm-exam validate
vlm-exam summary --check
```

`benchmark` schedules three QA repeats and one v2 detection run per arm, with
per-job logs. Text has its own documented launcher. `--tasks`, `--efforts`, and
QA repeat options narrow the schedule. V2 uses native pinned profiles; ambiguous
profiles require an explicit `--configuration`, and new models use `--profile`.
Do not pass custom `--config` to a v2 inference run and assume it changes the
request profile. `--max-samples` never publishes.

Default validation checks QA repeat coverage separately from the complete v2
release inventory, hashes, case identities, profiles and response coverage.
`validate --detection-version v1` verifies the archived six-task protocol.
`--format github` emits annotations and a step summary. CI also checks both
summary artifacts, Ruff and pytest. Do not silence a failing validation check;
fix the artifact or obtain authorization for an explicit policy change.

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
  Low and High: Text and Detection v2 require one run per effort; counting,
  identification and reasoning require three, for 22 runs per complete model. `protocol.repeats` remains the default of 3;
  `protocol.repeats_by_task` explicitly records the per-task requirements.
  Status describes run inventory; response coverage and accepted gaps remain
  explicit in the task metrics and coverage fields. Keep legacy status for
  legacy models with missing runs. Missing Text runs cannot be filled by OCR
  or Extraction runs. This publication selection does not change the historical
  QA task definitions or authorize additional runs.
- Omit per-subset detail and response-level provenance from the web JSON.
  Detection retains compact protocol/dataset/profile provenance so its version
  cannot be confused with v1. Retain full detail in raw/internal results. Preserve task metrics,
  metric runs, counts and coverage in the website payload.
- Regenerate `web/benchmark_summary.json` and commit it in every PR so the
  website payload never drifts from `results/` and `configs/models.yaml`.
- Rebuild it offline; v2 scores come from the approved release summary:

```bash
vlm-exam summary
```

- The command compiles all efforts by default, emitting one entry per
  `(model, effort)` pair; pass `--effort` only to restrict to one level.
  Each task entry averages every repeat of that configuration (see
  "Benchmark protocol" above).
- The output is deterministic given `results/`: `generated_at` derives
  from the newest included run, so an unchanged diff after regeneration
  means the results did not change.
- `vlm-exam summary --check` rebuilds the payload in memory and exits 1 if
  the committed file differs. V2 quality is included in this check without
  private pixels. Separately run `detection-v2 summary --check` to recompute
  quality from the pinned responses and annotations.
- The file is a generated artifact; never hand-edit it.

## Leaderboard charts

- Regenerate the leaderboard charts in `visualizations/leaderboards/` and
  commit them in every PR that changes `results/`, so the tracked PNGs
  never drift from the underlying runs. Both effort levels are tracked
  (`*_low.png` and `*_high.png`). Historical detection charts are isolated
  under `visualizations/leaderboards/detection/v1/`; current default charts
  never show archived mAP@75 or mAP@50:95 as v2 metrics:


```bash
vlm-exam leaderboard
vlm-exam efficiency-report --effort low
vlm-exam efficiency-report --effort high
```

  `leaderboard` renders every effort present in `results/`;
  `efficiency-report` renders one effort per call.

## Adding and benchmarking models

- For every new model, explicitly set the output token limit to **128,000**
  tokens for every task and effort level, including smoke tests and retries.
  Here, 128K means 128000, not 131072. Never omit the limit, pass null, or rely
  on a provider default. In detection v2 profiles set `max_output_tokens: 128000`;
  ensure the adapter sends the corresponding provider parameter on the wire.
  Verify acceptance with a small request before launching the full benchmark.
  If the provider rejects this limit or documents a lower hard ceiling, record
  the evidence and report the limitation before running with a different cap;
  never silently lower it. Document whether the cap includes reasoning tokens.
  This rule applies prospectively: preserve historical profiles and results;
  changing their limits requires a separately authorized rerun.
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
  `(task, model, effort)`. QA tasks and explicit detection v1 retain three repeats; default detection
  v2 uses one. Do not
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
failure IDs. The release has six accepted gaps: Sonnet 5.5 High one and Grok 4.7
High five. Muse Spark 1.3 Low and High are complete after the authorized
October 13 recovery of three failed pairs. Keep the remaining gaps unscored;
do not retry or change the policy to make validation pass. New publications require an
explicitly reviewed inventory change. Private results stay outside this repo.

### Scoring and publication

Default scores are exact string match, transcription character similarity,
typed JSON field F1, and joint box/text F1. Localization uses one-to-one matches
with IoU strictly greater than 0.5 and exact text. Invalid regions count as
unmatched predictions. Provider failures are unscored. Explicit transcription
profiles can request `exact` or `italian_soft_wraps`; never silently change
recorded scoring profiles or merge incompatible runs.

Publish through `vlm-exam summary` to
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
for strings, JSON, localization, empty/invalid responses and bounded single-card JSON when
changing rendering. Add focused regressions to existing test modules; do not
create new test files. Moving unchanged result files does not require rebuilding
PNGs or recomputing inference.


## Detection versions

- Standard `run --task detection`, `benchmark`, `summary` and `leaderboard` use
  **v2**. `low` and `high` select the recorded native minimum and maximum settings;
  they do not override a model's native effort values. V2 has one run per arm.
- Use `--detection-version v1` explicitly to run or summarize the historical
  pipeline. Historical flat files and `results/detection/v1/*.jsonl` are preserved
  byte-for-byte, but never enter the default overview JSON or detection charts.
- V2 has 600 cases, 120 subsets, five images each. Its exact approved selection,
  response inventory and configuration hashes live in `datasets/detection/v2/`.
  Never pool versions, configurations, diagnostics, or smoke runs.
- With no image directory, standard detection runs use `data/detection/v2` and
  fetch missing canonical originals from the private Roboflow project using the
  pinned image IDs. Verify exact hashes. Do not use re-encoded COCO images or the
  padded duplicate Blokus uploads; both Blokus tasks share the original photo
  but retain separate case IDs and annotations from the canonical manifest.
- New executions go to `working-runs/detection/v2/`; resume uses that directory.
  Full completed runs are promoted through the validated publication helper,
  preserving superseded responses. `--max-samples` is smoke-only and never
  publishes. Ambiguous historical request profiles require `--configuration`;
  new models require explicit `--profile` JSON rather than guessed parameters.
- V2 archives retain raw prediction text, canonical detections and provenance.
  Keep private pixels and credentials out of Git. Treat terminal invalid output
  as a model outcome; promote infrastructure recoveries only under explicit
  policy, never by score.
- `detection-v2 summary` rebuilds its detailed/compact outputs. Standard `summary`
  incorporates v2 detection in `web/benchmark_summary.json`; standard detection
  charts also use v2. Run release validation, both summary freshness checks and
  regression tests before publishing. Legacy results remain available opt-in.

### Detection v2 parser recovery

A local parser exception is an infrastructure gap, not a zero-scoring model
response. Preserve the raw response and usage, and do not resend the paid request.
After fixing parsing, use `vlm-exam detection-v2 reparse-run --output RUN_DIRECTORY`;
this validates provenance, retains the original inference hash and records the
new parser fingerprint. Publish only after the run passes full validation.


### V2 strict and recovery parser evaluation

New v2 inference uses uniform complete-object recovery, including Google. V1
parsing is unchanged. Run `vlm-exam detection-v2 evaluate-parsers --mode both
--workers 4 --output /tmp/parser-comparison.json` to score saved responses in
both modes without API calls. `--mode strict` requires a JSON array and the
requested box schema; any malformed entry rejects the whole response.
`--mode recovery` accepts complete boxes from recoverable malformed output.
Neither mode invents boxes from points. Unknown labels remain false positives.
Reports include schema compliance, failure reasons, recovered response counts,
and per-mode mAP@50/F1@50. Valid empty arrays are compliant. Missing API responses
stay missing, separately from model formatting errors. Reports must be written
to new paths and do not replace release archives or published scores. A frozen
release `validate --reparse` may expose intentional parser-version drift after
this change; do not rewrite historical predictions to hide it. Review a new
scoring revision explicitly before publishing it.


### Bounded parser policy (v2-bounded-recovery-4)

All v2 models now share the decoder in `detection_v2_parsing.py`; the v1
Supervision/provider parsers are not used by v2. The format adapter specifies
only the requested box structure, coordinate order, and scaling. Recovery may
remove one unambiguous JSON wrapper, read complete direct objects from JSONL or
an incomplete array, accept explicit field aliases and decimal numeric strings,
and skip malformed entries. It must never recursively search nested structures,
rebuild a foreign schema, resolve conflicting fields, complete partial boxes,
guess coordinate order/scale, clip geometry, reorder corners, or deduplicate.
Exactly four finite non-boolean coordinates are required. Native named-coordinate
box containers may use a single dictionary in recovery, but flat coordinate
arrays are not converted into that foreign schema. Strict requires the requested
schema and rejects the entire response if an entry is invalid. Both modes use
the same non-repairing coordinate decoder and confidence policy: a valid numeric
confidence in [0,1] (including zero) is preserved, otherwise the fallback is 1.
Unknown labels remain potential false positives. New inference records carry
`parser` version/mode/operation counts; offline comparisons include per-case
`parser_audit`. Published raw responses/scores are never rewritten implicitly.
Existing parser comparison reports describe their recorded parser version and
must not be presented as measurements of a later version without recalculation.

Use `ParserMode.STRICT` / `ParserMode.RECOVERY` and
`DetectionCoordinateFormat` in Python calls. Serialized string values remain
accepted at API boundaries and are validated as enums. Version 3 also accepts
one orphan closing Markdown fence after a complete JSON array in recovery mode,
recording `orphan_closing_fence`; strict mode rejects it. This does not authorize
repairing incomplete JSON, competing payloads, duplicate keys, or foreign schemas.
Existing release scores retain their recorded parser versions until an explicitly
approved offline reparse and scoring revision.

Parser version 4 shares explicit format specifications and entry validation
between strict compliance and decoding. Add a format to the enum and `_FORMATS`,
then extend the format geometry fixtures; the coverage test rejects missing
adapters. Keep recovery strategies shared across models. Decoded string labels
are preserved in `Detections.data["class_name"]`, even if absent from ground truth;
`classes` only seeds IDs. JSON constants such as NaN/Infinity are rejected, not
silently treated as valid metadata. Recovery records accepted fractional
coordinates for integer-coordinate formats, without rounding them. Existing
running evaluations must retain their pinned runtime and parser revision.

The November 20 offline scoring revision applies version 4 in recovery mode to
all 64,193 available primary responses across 107 configurations. Seven missing
responses and their accepted-failure policy are unchanged. Original archives
are retained as `diagnostics/parser-before-v4-*.jsonl.gz`. The release records
parser source hashes; `parser_v4_rescore.json` records before/after mAP and F1,
separating substantive detection changes from sub-millipixel precision changes.
No inference requests, prompts, settings, or raw response text changed.

### Gemini 4 detection refresh (November 20, 2026)

The four Gemini 4 release entries use the approved first-pass low/high runs with
128000 output tokens. High here is native `high`, not `xhigh` or an assertion of
the model's absolute maximum. The release lanes retain their historical
minimum/maximum names for API compatibility; inspect pinned native profiles.
Do not substitute later retry-round responses. `gemini4_refresh.json` records
source hashes and identity normalization; superseded archives remain diagnostics.
Carbon High has seven explicitly accepted 504 failures. Its run entry lists their
exact IDs in `accepted_missing_as_empty` with an approval reason. They retain
`missing` status and diagnostics but count as empty predictions in quality scores.
Unapproved missing cases still block ranking. Missing latency/usage is not invented.

### Explicit 128,000-token profiles (November 20, 2026)

`current_request_profile` now sets `max_output_tokens: 128000` for the 20
previously uncapped Google, OpenAI and OpenRouter profiles enumerated in
`_TOKEN_LIMIT_128000_PROFILES`. Standard future V2 executions derive a new
configuration hash; historical configuration hashes are not relabeled. Explicit
`--profile` files must also set 128000 under the new-model policy above. Existing
working directories with a different request identity cannot be resumed under
these settings; start a new working directory. This does not authorize full
reruns. The approved recovery is one request per each of the 13 audited
incomplete/empty/error outcomes, with no automatic retries. Preserve superseded
responses and promote completed valid replacements regardless of score.

The authorized 128K recovery completed with 13/13 valid responses. The primary
archives contain those replacements; `token_128k_recovery.json` lists the exact
cases and old/new configuration hashes. The original 13 records are preserved
in `diagnostics/token-128k-before-*.jsonl.gz`. Other responses keep their original
settings. This is a mixed historical run with targeted 128K recovery, not a claim
that every historical request used 128000 tokens. Both detection summaries,
the website summary and the low/high detection charts were regenerated.
