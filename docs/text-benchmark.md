# Mixed text benchmark

The new `text` task runs the 600-pair mixed OCR dataset alongside the existing
benchmark. Legacy tasks, committed results and the six-task protocol remain
unchanged. Text runs live in `results-text/` and have a separate summary.

## Import and validate

Install the repository normally (`uv sync --extra dev`). Download the Roboflow
JSONL ZIP and run:

```bash
uv run vlm-exam text-import ~/Downloads/vlm-exam-text-test2.v1i.jsonl.zip
uv run vlm-exam text-validate --dataset-directory data/text/train --expected-pairs 600
```

Import refuses to overwrite an existing snapshot. Use `--output-directory` for
a new snapshot. The verified export contains 600 questions referencing 576
images: 230 single-string, 230 structured, 90 localization/recognition and 50
transcription pairs. The same image filename can appear in multiple rows.
Images are local dataset files, not committed in this PR.

Each line has exactly three string fields:

```json
{"image":"image.jpg","prefix":"{\"task\":\"single_string\",\"question\":\"Read the identifier. Do not explain.\"}","suffix":"00123"}
```

Decoded prefixes have `task`, `question`, and optional `specification` only for
`localization_recognition`. Other task values are `single_string`,
`transcription`, and `structured`. The suffix is plain text for text tasks and
JSON-encoded content for structured/localization tasks. Nothing inside the
metadata envelope is sent to the model except the question and specification.
Ground-truth localization objects contain `bbox` (original-image pixel xyxy)
and `text` (string or null).

The validator rejects missing/unsafe image paths, duplicate image/question
pairs, unknown categories, malformed ground truth, non-finite coordinates,
out-of-frame boxes and non-upright EXIF orientation. Export upright images so
pixel coordinates cannot silently rotate. It hashes image bytes, prompts,
answers and scoring profiles; changes create a different result snapshot.

Roboflow may re-encode images. Comparisons must use the same downloaded snapshot.
Upload one file per unique image and reference it repeatedly for multiple tasks.
Uploading duplicate image bytes under different filenames previously lost
secondary annotations; the shared-filename re-upload retained all 600 pairs.

## Run

Configure provider credentials as for existing VLM Exam runs. No judge key is
required by this deterministic text protocol. A small connectivity smoke test:

```bash
uv run vlm-exam run --task text --models gpt-6.1-sol --effort low \
  --dataset-directory data/text/train --max-samples 4 \
  --output-directory results-text-smoke --concurrency 2
```

A single full configuration:

```bash
uv run vlm-exam run --task text --models gpt-6.1-sol --effort low \
  --dataset-directory data/text/train --output-directory results-text --concurrency 3
```

The complete repeated protocol (low/high, three repeats each):

```bash
uv run vlm-exam text-benchmark --models gpt-6.1-sol --max-parallel 2
uv run vlm-exam text-summary --results-directory results-text --output text-summary.json
```

`text-benchmark` reuses the existing job scheduler and prints the path to every
per-run log under `logs/text/` before dispatch. Providers, effort handling,
fallbacks, token usage and timing are the existing runner's implementations.
Use `--dataset-root` for a root containing `text/train`. Smoke runs remain
separate and cannot count as complete runs in text-summary.

Resume a run containing provider failures with the standard command:

```bash
uv run vlm-exam run --task text --models gpt-6.1-sol --effort low \
  --dataset-directory data/text/train --output-directory results-text \
  --resume-file results-text/REPLACE_WITH_RESULT_FILE.jsonl
```

Resume checks the dataset, protocol and coordinate profile and selects failed
image/question pairs, not every question for a failed image. The established
save/merge logic replaces the old result only after writing its successor.

## Prompt and scoring protocol

Single-string, transcription and structured questions are sent verbatim.
Localization appends the versioned JSON output contract to question and
specification. The model's existing `detection_coordinate_format` determines
key, order, scale and original-versus-uploaded frame. Provider-upload dimensions
are required for resized pixel formats. Predictions are converted to original
image pixels before matching; no format guessing or JSON repair is performed.

| Category | Primary score |
| --- | --- |
| Single string | Exact typed text, NFC, line endings and outer whitespace only |
| Transcription | Normalized character edit similarity |
| Structured | Exact typed nested leaf-value F1; missing/extra keys penalized |
| Localization/recognition | One-to-one joint F1, IoU strictly greater than 0.5 and exact text |

Structured null values earn credit; non-null accuracy and complete-object match
are also recorded. Localization records detection-only F1 and invalid regions.
Invalid content scores zero. Provider failures are unscored and summaries show
observed means and coverage bounds rather than presenting incomplete results as
final. A complete JSON code fence is parsed but flagged as noncompliant; duplicate
keys, non-finite values and surrounding prose are rejected.

The versioned `configs/text_subsets.json` maps known prefixes to subset labels
and the existing passport exact-match / Italian soft-wrap scoring exceptions.
Two night-plate subsets share a prefix and are disambiguated by a hash of the
prefix and expected answer. These hashes are local grouping metadata, never
model input. Unknown prompts receive a stable hash label and category-default
scoring; register new exceptions explicitly. Weather tables and election records
are routed as structured because their answers are JSON. Circuit LaTeX remains
strict text in this protocol; the experimental rendered-LaTeX judge is not used.

Results include raw response, expected answer, resolved prompt, pair identity,
image hash, whole-dataset hash, category, subset, scoring version and coordinate
profile. `text-summary` groups only compatible snapshots and pair sets, averages
per-run scores and token counts, and provides category and subset breakdowns.
An overall mean combines different task metrics and is not uniform accuracy.
Three complete repeats are required per effort; the old web leaderboard is not
silently extended or re-scored.

Changing prompts, normalization, ground truths, coordinates or rendering requires
a new protocol/snapshot and fresh repeats. The previous exploratory scores,
including hypothetical nutrition rescoring, are not results for these new runs.
