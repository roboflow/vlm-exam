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

## Web summary

- Regenerate `web/benchmark_summary.json` and commit it in every PR so the
  website payload never drifts from `results/` and `configs/models.yaml`.
- Rebuild it with the detection dataset so detection mAP is included:

```bash
vlm-exam summary --dataset-directory data/detection/train
```

- The command compiles all efforts by default, emitting one entry per
  `(model, effort)` pair; pass `--effort` only to restrict to one level.
- The output is deterministic given `results/`: `generated_at` derives
  from the newest included run, so an unchanged diff after regeneration
  means the results did not change.
- The file is a generated artifact; never hand-edit it.

## Adding and benchmarking models

- In `configs/models.yaml`, each model has an ordered `routes` list (or
  legacy single `provider` field). The vlm-exam model **key** is used in
  result filenames and leaderboards. Each route's `provider_model_id` is
  the upstream API id; when omitted, the model key is used.
- Set the required `detection_coordinate_format` per model after researching
  its native grounding convention (GitHub, forums, papers, official docs).
  Valid values are the `DetectionCoordinateFormat` enum strings in
  `src/vlm_exam/tasks/detection.py`: `yxyx_normalized_0_to_1000`,
  `xyxy_normalized_0_to_1000`, `xyxy_absolute_provider_upload`,
  `xyxy_absolute_original_image`, and `yxyx_absolute_original_image`. The
  format follows the model, not the route -- the same weights use the same
  box convention on Google direct and OpenRouter.
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
