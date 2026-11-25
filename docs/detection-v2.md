# Detection v2

Detection has two independently versioned protocols. The migration preserves
historical responses and includes ten authorized Carbon/S196 corrections plus
eleven previously saved Kimi infrastructure recoveries. Integration smoke tests
are kept outside the publication inventory.

| | v1 | v2 |
|---|---|---|
| Selection | Original detection dataset | Approved final 600 cases, 120 subsets, 5 per subset |
| Repeats | Existing low/high repeats, unchanged | One measured run per configuration |
| Effort | Historical low/high | Native minimum/maximum, recorded per response |
| Primary metric | Historical pooled mAP@50 | Mean of the 120 subset dense mAP@50 scores |
| Prediction cap | Historical COCO settings | No 100-prediction cap |
| Secondary metrics | Unchanged | Capped-100 mAP@50, micro F1@50, precision, recall, latency and tokens |
| Results | `results/detection/v1/*.jsonl` | `results/detection/v2/*.jsonl.gz` |
| Web payload | Explicit historical summary only | `web/benchmark_summary.json` plus `web/detection_v2_summary.json` |

These scores are not a controlled dataset-only comparison: the selection,
aggregation, native reasoning effort, output limits and some formats changed.
Do not average v1 and v2 runs or count diagnostic retries as repeats. Single-arm
models have one measured entry, not a synthetic second measurement.

## Frozen inventory

`datasets/detection/v2/release.json` pins the measured models, request profiles,
primary archives, diagnostics and missing-response identities. Every primary
archive covers exactly the 600 selected case IDs. Configurations with missing
responses have no headline rank; missing-as-empty scores remain diagnostic.

All returned responses remain included in scoring, including empty, truncated,
and non-strict JSON responses. `invalid_json` describes strict JSON syntax,
not necessarily an empty parser result. The historical parser can extract boxes
from prose or partial output. Status counts are exposed rather than silently
filtering these cases. Stored canonical detections preserve the historical parser
output; raw prediction text and its checksum are also retained for auditing.

Diagnostic responses remain separate from primary results. Explicit validated
publication can promote approved infrastructure recoveries while archiving the
superseded responses. It must never select responses by their quality score.

Every primary record carries its native configuration, format, upstream model,
source checksum and relative source identifier. Fields absent from the original
record remain absent; unknown historical token limits are not filled with current
defaults. Some configurations vary between subsets following authorized ground
truth reruns. These variations remain visible through profile hashes and native
per-response settings. Token/latency totals describe saved final responses, not
billing for all attempts. Imported archives exclude original SDK envelopes. New runs retain raw responses
for parser recovery. Credentials and machine-specific absolute paths are excluded
from published records.

## Ground truths and images

The portable manifest contains the approved boxes, class names, case identities,
dimensions and image/annotation hashes. It has 40,410 objects. Source images remain
private and are not committed. The Roboflow project is
[vlm-exam-detection-v2](https://app.roboflow.com/roboflow-jvuqo/vlm-exam-detection-v2/browse).
The normal run downloads missing original images using the pinned private
Roboflow image IDs and `ROBOFLOW_API_KEY`. It verifies their SHA256 hashes against
the canonical manifest or the explicit lossless transport inventory. Generated COCO exports can re-encode images or change
annotations and are therefore not the canonical source.

Two Blokus tasks share five original photos but have different ground truths.
They retain distinct case IDs. The five one-pixel-padded Roboflow upload copies
are not canonical benchmark inputs. Always use hash-matching original files;
do not substitute re-encoded downloads or padded copies. The download manifest
points both Blokus tasks to the same original Roboflow photo.

The original-image API was verified for every case: 594 files match the original
bytes, and six require auxiliary lossless PNG transport copies. Roboflow's
perceptual deduplication requires a replicated rightmost column on these transport
copies. The downloader verifies the transport checksum, removes that column, and
checks the prepared file checksum, dimensions and canonical upright RGB hash. All
600 prepared images have exactly the approved pixels. The six auxiliary images
are not additional benchmark cases. Encoding changes fail closed rather than
silently changing model inputs. The pinned version 1 COCO export remains a
browsing/export artifact; the canonical manifest retains all 40,410 annotations,
including five annotations absent from that export.

Prepare authorized local originals, preserving separate identities for shared
photos:

```bash
uv run vlm-exam detection-v2 prepare-images \
  --source /path/to/original-images --destination data/detection/v2
```

## Standard workflow (v2 by default)

```bash
uv run vlm-exam run --task detection --models MODEL --effort low --concurrency 30
uv run vlm-exam run --task detection --models MODEL --effort high --concurrency 30
uv run vlm-exam summary
uv run vlm-exam leaderboard
```

The public CLI retains `low`/`high`; v2 maps these to the native minimum/maximum
profiles. When historical settings differ, the command lists explicit
`--configuration HASH` choices. A new model uses `--profile FILE` containing all
native request settings. `--max-samples 1` is a non-publishing smoke test.
Full runs are validated and published; existing directory checkpoints resume
without resending successful cases. The default image cache is
`data/detection/v2`, and working results live in `working-runs/detection/v2`.

To access archived v1, explicitly add `--detection-version v1` to `run`,
`benchmark`, `summary`, or `leaderboard`. Historical visualization commands
remain available under `detection-v1`.

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
  --case OLD031-001 --images data/detection/v2 \
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
The standard overview includes v2 under its existing detection task key.
It never substitutes missing v2 results with v1 scores. Native minimum/maximum
are exposed through the existing low/high effort keys with provenance; neutral
enabled-only measurements remain distinct, without fabricated low/high copies.

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
  --configuration PROFILE_SHA256 --images data/detection/v2 \
  --output runs/detection-v2/gpt-6.2-sol-maximum --concurrency 30
```

For a model not yet in the release, supply `--profile request-profile.json` instead
of `--configuration`. The JSON must explicitly define `provider`, `wire`,
`format`, `minimum`, `maximum`, and `max_output_tokens`; optional `routing` and
`arms` pin provider routing and measured arms. A null output limit means omit the
explicit limit, not unlimited output. Anthropic requires a positive limit.
The runner fingerprints this profile and its inference implementation for resume;
it does not require adding an unevaluated model to the published inventory first.

This command sends paid requests. It checks every image hash before any request,
uses the profile's coordinate format and image-conditioned class list, records
native parameters, bounds API timeouts, retries transient transport/server errors,
and writes progress to `status.json`. Concurrency is bounded at 60. Resume reuses
completed responses only when the dataset, configuration and inference implementation match. Re-running the
same command retries missing/API-error responses, not terminal truncated or empty
generations. The explicit `detection-v2 run` command keeps working results separate;
`detection-v2 publish --output RUN_DIRECTORY` validates and promotes a full run,
retaining superseded responses. Standard `run --task detection` performs that
validated publication after a complete run. Smoke runs are never promoted.

A parser exception preserves the paid raw response and marks the run incomplete.
It does not resend that request on resume. After correcting the parser, recover
locally with `vlm-exam detection-v2 reparse-run --output RUN_DIRECTORY`, then
publish the complete run. This records the parser fingerprint and retains the
original inference provenance; no API request is sent.

## v1 compatibility

`run`, `benchmark`, `summary`, `leaderboard`, `detection-report`, and
`detection-visualize` default to v2. Explicit `--detection-version v1` preserves
historical behavior and writes legacy inference to `results/detection/v1`.
The `detection-v1 report` and `detection-v1 visualize` aliases also opt into v1.
Historical validation still checks the archived results. Default result loaders
exclude v1; code that intentionally compares historical datasets must explicitly
set `include_detection_v1=True`. All 254 migrated v1 result files are byte-for-byte
identical to their original versions.

## Migration evidence

`migration_audit.json` records archive preservation, exact parser replay and live
integration checks. All 55 models were exercised on one image, including four
initial failures subsequently resolved with verified endpoints and alias policy.
A standard CLI smoke also completed. These calls prove integration, not statistical
equivalence of stochastic scores or live coverage of every effort. Offline
payload contracts cover provider families and both effort arms. Canonical scoring
is replayed from all saved responses; v1 and v2 scores are not directly equivalent.

## OpenRouter routing for new runs

Every new V2 OpenRouter request requires `routing.only` with exactly one provider
ID, `allow_fallbacks: false`, and `require_parameters: true`. Missing routing is a
preflight error, including for new runs based on historical unpinned profiles.
Supply a reviewed `--profile` with verified routing instead. Historical release
validation and offline rescoring do not apply this new-request rule retroactively.

Prefer the model developer's native provider. For a non-native endpoint, the
profile must include `routing_exception` with nonempty `reason` and `evidence`
strings documenting why no suitable native endpoint is available. This exception
is part of the configuration hash and is not sent to OpenRouter. The validator
checks that the evidence is recorded; the operator must verify its accuracy and
currency before starting. Unknown model families must be registered explicitly
in `providers/openrouter_routing.py`. Endpoint variants such as `xiaomi/fp8`
are recognized under their native provider family. Network errors never trigger
an automatic provider switch.
