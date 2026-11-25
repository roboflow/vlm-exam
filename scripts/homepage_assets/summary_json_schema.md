# Benchmark summary JSON schema

The website pulls a single file, `benchmark_summary.json`, to render every
leaderboard and model/task view. This document describes its shape so the
frontend can consume it without reading the Python source.

## How it is generated

```bash
vlm-exam summary
```

- Output path defaults to `web/benchmark_summary.json`.
- Current detection V2 scores come from the approved release summary. Historical
  detection V1 scores come from an archive-hash-verified frozen website snapshot;
  neither requires private images for the default export.
- Explicit `--detection-version v1` remains available for rescoring historical
  results with `--dataset-directory data/detection/train`.
- By default every effort level is compiled, emitting one model entry per
  `(model, effort)` pair. Pass `--effort low` to restrict to one level.
- The output is deterministic given the contents of `results/`:
  `generated_at` derives from the newest included run, so an unchanged diff
  after regeneration means the underlying results did not change.
- The file is a generated artifact and must never be hand-edited.

## Top level

| Field | Type | Description |
|---|---|---|
| `generated_at` | string \| null | ISO-8601 UTC timestamp of the newest included run; `null` when there are no runs. |
| `efforts` | string[] | Distinct effort levels present, ordered `low`, `medium`, `high`. |
| `tasks` | Task[] | Metadata for each benchmarked task (see below). |
| `models` | Model[] | One entry per `(model, effort)` pair (see below). |

## Task

Describes a task and the metrics reported for it.

| Field | Type | Description |
|---|---|---|
| `key` | string | Stable task id: `ocr`, `extraction`, `counting`, `identification`, `reasoning`, `detection`. |
| `name` | string | Display name, e.g. `OCR`, `Data Extraction`, `Detection`. |
| `primary_metric` | string | The metric key to sort/headline by for this task. |
| `metrics` | Metric[] | All metrics reported for the task. |

### Metric

| Field | Type | Description |
|---|---|---|
| `key` | string | Metric id used inside each model's task `metrics` map. |
| `label` | string | Human label, e.g. `Mean Similarity`, `mAP@50`, `Accuracy`. |
| `unit` | string | Always `percent` (values are 0-100). |

Metric keys by task:

- `ocr` -> `similarity` (primary)
- `extraction`, `counting`, `identification`, `reasoning` -> `accuracy` (primary)
- `detection` -> `map50` (primary), `map75`, `map50_95`

## Model

One benchmarked model at one effort level.

| Field | Type | Description |
|---|---|---|
| `id` | string | Unique row id, `"{key}:{effort}"`, e.g. `claude-fable-5:low`. |
| `key` | string | Model key used in result filenames. |
| `name` | string | Display name, e.g. `Claude Fable 5`. |
| `lab` | string | Lab/vendor key, e.g. `anthropic`. |
| `effort` | string | Effort level: `low`, `medium`, or `high`. |
| `tasks` | map<string, TaskResult> | Per-task results, keyed by task `key`. A task is absent when the model has no run for it. |
| `overall` | Overall | Efficiency pooled across all the model's benchmarked tasks. |

### TaskResult

| Field | Type | Description |
|---|---|---|
| `primary_metric` | `{name: string, value: number}` \| null | The task's headline metric value (percent, 0-100). `null` when unavailable (e.g. detection mAP omitted). |
| `metrics` | map<string, number> | All metric values for this task (percent, 0-100). |
| `sample_count` | number | Number of samples in the run. |
| `evaluated_sample_count` | number \| null | Samples actually scored for quality; set for detection mAP, `null` otherwise. |
| `failed_sample_count` | number | Samples that errored / produced no usable output. |
| `tokens` | Tokens | Token usage for this task. |
| `cost` | Cost | Estimated USD cost for this task. |
| `speed` | Speed | Wall-clock inference time for this task. |
| `timestamp` | string | ISO-8601 UTC timestamp of the run. |

### Overall

| Field | Type | Description |
|---|---|---|
| `task_count` | number | Number of tasks pooled. |
| `sample_count` | number | Total samples pooled across tasks. |
| `tokens` | Tokens | Pooled token usage. |
| `cost` | Cost | Pooled USD cost. |
| `speed` | Speed | Pooled inference time. |

### Tokens / Cost / Speed

```jsonc
"tokens": { "input": 0, "output": 0, "total": 0, "average_per_sample": 0.0 }
"cost":   { "total_usd": 0.0, "average_per_sample_usd": 0.0 }
"speed":  { "total_seconds": 0.0, "average_seconds_per_sample": 0.0 }
```

## Rendering a leaderboard

1. Choose an effort (from `efforts`) and a task (from `tasks`).
2. Filter `models` to entries whose `effort` matches.
3. For each model, read `model.tasks[taskKey].primary_metric.value` (percent).
4. Sort descending; the task's `primary_metric` names the field to sort on.
5. For a cross-task "overall efficiency" view, use `model.overall`
   (tokens, cost, speed) rather than any single task.

All quality values are percentages in `[0, 100]`. Cost is USD; speed is
seconds. Guard against `null` `primary_metric` (notably detection when the
summary was built without `--dataset-directory`).

## Minimal example

```jsonc
{
  "generated_at": "2026-07-10T08:11:31Z",
  "efforts": ["low"],
  "tasks": [
    {
      "key": "ocr",
      "name": "OCR",
      "primary_metric": "similarity",
      "metrics": [{ "key": "similarity", "label": "Mean Similarity", "unit": "percent" }]
    }
  ],
  "models": [
    {
      "id": "claude-fable-5:low",
      "key": "claude-fable-5",
      "name": "Claude Fable 5",
      "lab": "anthropic",
      "effort": "low",
      "tasks": {
        "ocr": {
          "primary_metric": { "name": "similarity", "value": 94.03 },
          "metrics": { "similarity": 94.03 },
          "sample_count": 37,
          "evaluated_sample_count": null,
          "failed_sample_count": 0,
          "tokens": { "input": 51849, "output": 18602, "total": 70451, "average_per_sample": 1904.1 },
          "cost": { "total_usd": 1.44859, "average_per_sample_usd": 0.039151 },
          "speed": { "total_seconds": 402.169, "average_seconds_per_sample": 10.869 },
          "timestamp": "2026-07-10T06:53:34Z"
        }
      },
      "overall": {
        "task_count": 6,
        "sample_count": 513,
        "tokens": { "input": 894745, "output": 119490, "total": 1014235, "average_per_sample": 1977.1 },
        "cost": { "total_usd": 14.92195, "average_per_sample_usd": 0.029088 },
        "speed": { "total_seconds": 4299.149, "average_seconds_per_sample": 8.38 }
      }
    }
  ]
}
```

## Schema version 2 additions

`schema_version` is `2`. Existing fields retain their types. Root `tasks` and
`models` describe current leaderboards. `historical.detection.v1` has its own
`tasks`, `models`, `efforts`, scoring source and historical protocol. Historical
model rows have only `tasks.detection`; they have no current `overall` or overall
completion status. Select the view first, then reuse the same task-table component.
Model IDs are scoped to a view; include benchmark identity in cross-view UI keys.

`overview_tasks` lists the tasks included in current efficiency totals and
completion requirements. It does not define a cross-task quality average.
Text replaces OCR/extraction in this list; those old task results remain separate.

Task definitions expose `benchmark_id` for detection and mixed text, and
`aggregation` describes score composition. Text `subtasks` maps the four IDs
`single_string`, `transcription`, `structured`, `localization_recognition` to
their display labels and existing metric keys. Text overall is the mean sample
score within each complete run, then the mean run score. It is not the equal
mean of four categories. Incomplete runs withhold their score. Existing coverage
contains per-category counts and failures. V2 mAP@50 is subset-macro AP; F1@50
uses aggregated class-aware TP/FP/FN at IoU 0.5. Historical V1 mAP remains pooled
dataset AP with the original prediction cap. Metric values remain percentages.

Every model task result adds `configuration_details`:

- `status`: `uniform`, `mixed`, or `unknown`. It describes recorded profile
  identity; inspect field statuses for missing individual settings.
- `record_count`: number of primary evaluation records across repeats, including
  missing-result placeholders. Not unique images, attempts, or billing requests.
- `variants`: inline profiles, each with `id`, `record_count`, `provider`,
  `model_id`, `max_output_tokens`, `reasoning`, and `evidence`.
- `max_output_tokens`: `{ "status": "explicit", "value": 128000 }`; statuses
  `omitted` and `unknown` use null. Null never means unlimited.
- `reasoning`: `kind` (`level`, `budget`, `enabled`, `unknown`), typed `value`,
  and `api_parameters` mapping exact parameter paths to values. For example,
  Anthropic includes both `thinking.type` and `output_config.effort`.
- Optional `routing` reports requested provider IDs and fallback policy only;
  these must not be presented as observed upstream providers.

`effort` remains the display category. A model's text and detection settings can
differ in the same Low row. Mixed settings must show all variants; do not choose
the first one as representative or invent per-variant scores. A profile hash
without its historical snapshot cannot establish native settings: export unknown.
The existing `provenance.configurations` remains an array of hash strings.

Frontend access: `model.tasks[taskKey].primary_metric`, `.metrics.f1` for V2
detection F1, and `.configuration_details`. Historical view selection is
`payload.historical.detection.v1`; current selection is `payload`.
