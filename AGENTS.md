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

## Benchmark results

- Only commit full-dataset benchmark runs to `results/`. It is the single
  source of truth aggregated by `report`, `leaderboard`, and
  `detection-report`, which glob every file in the directory.
- Never commit partial or smoke runs (e.g. any run produced with
  `--max-samples`). Their noisy, non-comparable numbers would corrupt the
  leaderboards. Keep such runs local or write them elsewhere.
- Keep `--effort` consistent with existing runs when adding a model to a
  task, so results stay comparable on shared leaderboards. All committed
  runs currently use `--effort low`; do not mix effort levels within a
  task's leaderboard unless the run is explicitly an effort comparison.
- Images are EXIF-transposed on load before being sent to any provider.
  Datasets whose images carry EXIF orientation tags will therefore produce
  runs that are not comparable to runs made before this behavior existed;
  re-run all models on such a dataset rather than mixing old and new runs.

## Repeated runs: three per configuration, report the mean

- Every new model is benchmarked three times per `(task, effort)`
  configuration and all three result files are committed to `results/`.
  A single run is not a benchmark entry; model output is stochastic and
  one file cannot tell a real gap from run-to-run noise.
- Repeats are plain result files. There is no repeat field or manifest:
  every file under `results/` sharing a `(task, model, effort)` triple is
  one repeat, and `report`, `leaderboard`, `efficiency-report`,
  `detection-report`, and `summary` average across all of them. Never
  delete or cherry-pick repeats to move a number; if a run is broken, fix
  the cause, re-run it, and remove only the broken file.
- Reported numbers: quality metrics (judge and strict accuracy, OCR
  similarity, mAP) are the arithmetic mean of the per-run values. Token,
  cost, and time totals are the mean of one complete run, so a model with
  three repeats is not reported as three times more expensive. Failed
  sample counts are summed over repeats.
- `web/benchmark_summary.json` keeps `metrics` as the flat mean so the
  site keeps working, and adds `metric_runs` (the per-run values behind
  each mean), `run_count`, and `timestamps` per task entry, plus a
  top-level `protocol.repeats` block. Leaderboard PNGs draw a min-max
  whisker on bars backed by more than one run and note the run count in a
  footnote; `report` shows a `Runs` column and `mean +/- half-spread`.
- Models benchmarked before this rule have one run per configuration.
  They stay in the leaderboards with `run_count: 1` until re-run; do not
  drop them, and prefer adding their missing repeats over re-running the
  existing file.
- Only average runs produced under one protocol. A change to prompts,
  coordinate formats, judge settings, or image preprocessing invalidates
  the existing repeats of the affected configurations; re-run all three.
- Run repeats with `vlm-exam run --repeats 3` (sequential) or as separate
  parallel processes writing to the same `--output-directory`; filenames
  carry a second-resolution timestamp and the CLI suffixes `_2`, `_3` on
  collision, so parallel launches are safe. Use `--concurrency` to evaluate
  samples in parallel within one run; size it by task length so long tasks
  finish alongside short ones. With Claude-class latency (5-25 s per
  request), detection (250 images) at 6, reasoning (151) at 3-4, OCR and
  extraction at 2-3, counting and identification at 1-2 keeps a full
  effort level (six tasks, two repeats) under 15 minutes while staying
  near 30 in-flight requests. Drop detection to 4 if the provider starts
  returning 429s.
- `vlm-exam run --resume-file <file>` re-runs only the failed samples,
  writes the merged complete file, and deletes the source file, so a
  partial run and its completion are never both counted as repeats.
  `summary` warns about any run with failed samples; resume it before
  committing.

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

- Regenerate `web/benchmark_summary.json` and commit it in every PR so the
  website payload never drifts from `results/` and `configs/models.yaml`.
- Rebuild it with the detection dataset so detection mAP is included:

```bash
vlm-exam summary --dataset-directory data/detection/train
```

- The command compiles all efforts by default, emitting one entry per
  `(model, effort)` pair; pass `--effort` only to restrict to one level.
  Each task entry averages every repeat of that configuration (see
  "Repeated runs" above).
- The output is deterministic given `results/`: `generated_at` derives
  from the newest included run, so an unchanged diff after regeneration
  means the results did not change.
- The file is a generated artifact; never hand-edit it.

## Leaderboard charts

- Regenerate the leaderboard charts in `visualizations/leaderboards/` and
  commit them in every PR that changes `results/`, so the tracked PNGs
  never drift from the underlying runs:

```bash
vlm-exam leaderboard --dataset-directory data/detection/train
vlm-exam efficiency-report
```

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
  local format probe confirms mAP on a ~20-image detection subset.
- Set the required `detection_coordinate_format` per model after that
  research and probe. Valid values are the `DetectionCoordinateFormat` enum
  strings in `src/vlm_exam/tasks/detection.py`: `yxyx_normalized_0_to_1000`,
  `xyxy_normalized_0_to_1000`, `xyxy_normalized_0_to_100`,
  `xyxy_normalized_0_to_1000_meta_flat`, `xyxy_absolute_resized_image`,
  `xyxy_absolute_original_image`, and `yxyx_absolute_original_image`. The
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
