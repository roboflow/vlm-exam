# Detection v2

Detection now has two independently versioned protocols. No additional inference
was performed to produce this migration.

| | v1 | v2 |
|---|---|---|
| Selection | Original detection dataset | Approved final 600 cases, 120 subsets, 5 per subset |
| Repeats | Existing low/high repeats, unchanged | One measured run per configuration |
| Effort | Historical low/high | Native minimum/maximum, recorded per response |
| Primary metric | Historical pooled mAP@50 | Mean of the 120 subset dense mAP@50 scores |
| Prediction cap | Historical COCO settings | No 100-prediction cap |
| Secondary metrics | Unchanged | Capped-100 mAP@50, micro F1@50, precision, recall, latency and tokens |
| Results | `results/detection/v1/*.jsonl` | `results/detection/v2/*.jsonl.gz` |
| Web payload | Existing `web/benchmark_summary.json` | `web/detection_v2_summary.json` |

These scores are not a controlled dataset-only comparison: the selection,
aggregation, native reasoning effort, output limits and some formats changed.
Do not average v1 and v2 runs or count diagnostic retries as repeats. Single-arm
models have one measured entry, not a synthetic second measurement.

## Frozen inventory

`datasets/detection/v2/release.json` pins all 55 models and 105 measured
configurations. Each archive contains exactly the 600 selected case IDs. There
are 62,989 saved primary responses and 11 explicit missing-response records for
Kimi K3. Configurations with missing responses have no headline rank; the
missing-as-empty score remains available for diagnosis.

All returned responses remain included in scoring, including empty, truncated,
and non-strict JSON responses. `invalid_json` describes strict JSON syntax,
not necessarily an empty parser result. The historical parser can extract boxes
from prose or partial output. Status counts are exposed rather than silently
filtering these cases. Stored canonical detections preserve the historical parser
output; raw prediction text and its checksum are also retained for auditing.

The additional 221 responses under `diagnostics/` are separate recovery
experiments, including successful Kimi recoveries. They were explicitly stored
without automatic promotion in the local experiments. This migration retains that
boundary. Promoting them requires an explicit inventory change, retaining the
superseded responses and provenance; it must not select a response by score.

Every primary record carries its native configuration, format, upstream model,
source checksum and relative source identifier. Fields absent from the original
record remain absent; unknown historical token limits are not filled with current
defaults. Some configurations vary between subsets following authorized ground
truth reruns. These variations remain visible through profile hashes and native
per-response settings. Token/latency totals describe saved final responses, not
billing for all attempts. Raw SDK envelopes, credentials and machine-specific
absolute paths are excluded.

## Ground truths and images

The portable manifest contains the approved boxes, class names, case identities,
dimensions and image/annotation hashes. It has 40,410 objects. Source images remain
private and are not committed. The Roboflow project is
[vlm-exam-detection-v2](https://app.roboflow.com/roboflow-jvuqo/vlm-exam-detection-v2/browse).
No public download or generated dataset-version availability is implied.

Two Blokus tasks share five original photos but have different ground truths.
They retain distinct case IDs. The five one-pixel-padded Roboflow upload copies
are not canonical benchmark inputs. Always use hash-matching original files;
do not substitute re-encoded downloads or padded copies.

Prepare authorized local originals, preserving separate identities for shared
photos:

```bash
uv run vlm-exam detection-v2 prepare-images \
  --source /path/to/original-images --destination data/detection-v2/images
```

## Offline validation, scoring and rendering

```bash
uv run vlm-exam detection-v2 validate
uv run vlm-exam detection-v2 summary --workers 4
uv run vlm-exam detection-v2 summary --workers 4 --check
uv run vlm-exam detection-v2 leaderboard --effort minimum \
  --output visualizations/leaderboards/detection_v2_minimum.png
uv run vlm-exam detection-v2 leaderboard --effort maximum \
  --output visualizations/leaderboards/detection_v2_maximum.png
uv run vlm-exam detection-v2 visualize --model gpt-6.2-sol --effort maximum \
  --case OLD031-001 --images data/detection-v2/images \
  --output visualizations/detection_v2_example.png
```

Validation and scoring need no API credentials or private images. Rendering
images requires authorized originals. Charts use the shared VLM Exam renderer;
fonts and logos may require network access on first use.

Scores in the v2 JSON are fractions, not percentages. Dense AP uses COCO's
101 recall thresholds at IoU 0.5, evaluated per subset with enough prediction
slots for every returned box. F1 uses class-aware maximum bipartite matching at
IoU >= 0.5; duplicate and unknown-class predictions count as false positives.
F1, precision and recall aggregate TP/FP/FN across all cases. COCO AP's treatment
of classes without ground truth is unchanged; F1 explicitly exposes those FPs.

The generated web JSON intentionally omits per-subset scores and response-level
provenance. Those remain in `datasets/detection/v2/summary.json` and raw archives.
The existing overview remains explicitly v1 until its consuming website opts into
v2. It never silently substitutes missing v2 results with v1 scores.

## New inference and resume

The v2 runner uses the same native SDK families as the local experiments:
OpenAI/xAI Responses, Anthropic streaming Messages, Gemini streaming content,
and OpenRouter chat completions with explicit reasoning and routing. It does not
inherit v1's effort mappings or output-token defaults.

Choose an explicit profile hash listed for the desired model/effort in
`release.json`. Review its native effort, upstream identifier, token cap and
routing before running; a model may have several historical profiles. The chosen
profile applies to every image in the new run. This is a new standardized run,
not an assertion that all historical responses used identical settings.

```bash
uv run vlm-exam detection-v2 run --model gpt-6.2-sol --effort maximum \
  --configuration PROFILE_SHA256 --images data/detection-v2/images \
  --output runs/detection-v2/gpt-6.2-sol-maximum --concurrency 30
```

This command sends paid requests. It checks every image hash before any request,
uses the profile's coordinate format and image-conditioned class list, records
native parameters, bounds API timeouts, retries transient transport/server errors,
and writes progress to `status.json`. Concurrency is bounded at 60. Resume reuses
completed responses only when the dataset, configuration and inference implementation match. Re-running the
same command retries missing/API-error responses, not terminal truncated or empty
generations. New runs never overwrite the frozen release. `summary.json` next to
the new run reports its results. Publication is a separate reviewed operation.

## v1 compatibility

`run --task detection`, `detection-report`, `detection-visualize`, `validate`,
`summary`, and legacy leaderboards continue to use v1. Default detection outputs
now go to `results/detection/v1`. The directory loader reads old flat files and
the v1 archive but deliberately never recurses into v2 or diagnostic folders.
Explicit `detection-v1 report` and `detection-v1 visualize` aliases are available.
Every migrated v1 result is byte-for-byte identical to its source file.
