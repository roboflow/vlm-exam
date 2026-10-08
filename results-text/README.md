# Mixed text evaluation — frozen 600-pair snapshot

This collection contains 48 models at Low and High effort: 96 runs, one per configuration, on the same frozen dataset. Of 57,600 pairs, 57,591 have successful responses. These are **preliminary single-run results**, not three-repeat averages. The existing six-task benchmark and its protocol are unchanged.

The snapshot contains 600 image/question pairs referencing 576 images: 230 structured JSON, 230 single-string extraction, 50 transcription, and 90 localization/recognition pairs. Dataset hash: `6a6cbeadcbf0966327cc19aacd165e9b6335d0cf4ee3584f3e0cc18e400141c7`. Scoring protocol: `mixed-text-v2`. Prompts, references, and inference/coordinate provenance are stored per pair.

Scores are on a 0–100 scale. Overall scores average the 600 per-pair scores, mixing exact string match, transcription similarity, structured field F1, and joint box/text F1; they are not uniform accuracy. All configurations have one run.

## Coverage

Nine gaps are retained as failed, unscored records: Sonnet 5.5 High has one content-filter gap; Grok 4.7 High has five exhausted timeouts; Muse Spark 1.3 has two Low and one High exhausted provider errors. These runs are not ranked in the PNGs. Their observed scores and coverage bounds remain available in the machine-readable summary. No failed response is replaced with a successful score, and no additional inference was performed for this export.

| Model | Low mean | Low coverage | High mean | High coverage |
|---|---:|---:|---:|---:|
| Claude Fable 5 | 72.28 | 600/600 | 74.16 | 600/600 |
| Claude Fable 5.1 | 72.88 | 600/600 | 75.26 | 600/600 |
| Claude Haiku 5.5 | 61.21 | 600/600 | 65.28 | 600/600 |
| Claude Opus 4.8 | 61.45 | 600/600 | 62.12 | 600/600 |
| Claude Opus 5 | 67.96 | 600/600 | 65.82 | 600/600 |
| Claude Opus 5.5 | 78.26 | 600/600 | 79.92 | 600/600 |
| Claude Sonnet 5 | 59.37 | 600/600 | 59.45 | 600/600 |
| Claude Sonnet 5.5 | 66.25 | 600/600 | — | 599/600 |
| DeepSeek V4 Flash | 43.98 | 600/600 | 44.68 | 600/600 |
| GLM 5.3 Flash | 55.36 | 600/600 | 55.31 | 600/600 |
| GLM 5V Turbo | 51.54 | 600/600 | 55.96 | 600/600 |
| GPT-5.4 mini | 51.12 | 600/600 | 55.75 | 600/600 |
| GPT-5.5 | 58.40 | 600/600 | 60.57 | 600/600 |
| GPT-5.6 Luna | 51.78 | 600/600 | 55.72 | 600/600 |
| GPT-5.6 Sol | 63.59 | 600/600 | 65.28 | 600/600 |
| GPT-5.6 Terra | 56.27 | 600/600 | 56.91 | 600/600 |
| GPT-6 Astra | 77.84 | 600/600 | 77.63 | 600/600 |
| GPT-6 Luna | 59.77 | 600/600 | 62.54 | 600/600 |
| GPT-6 Sol | 66.68 | 600/600 | 68.77 | 600/600 |
| GPT-6.1 Sol | 75.75 | 600/600 | 75.84 | 600/600 |
| Gemini 2.5 Pro | 65.21 | 600/600 | 64.56 | 600/600 |
| Gemini 3 Flash | 63.92 | 600/600 | 73.07 | 600/600 |
| Gemini 3.1 Pro | 72.63 | 600/600 | 73.14 | 600/600 |
| Gemini 3.5 Flash | 76.69 | 600/600 | 75.55 | 600/600 |
| Gemini 3.5 Flash-Lite | 65.80 | 600/600 | 71.17 | 600/600 |
| Gemini 3.6 Flash | 71.76 | 600/600 | 77.42 | 600/600 |
| Gemini 3.7 Flash | 73.89 | 600/600 | 78.77 | 600/600 |
| Gemini 3.8 Flash | 72.75 | 600/600 | 78.66 | 600/600 |
| Grok 4.5 | 57.14 | 600/600 | 57.55 | 600/600 |
| Grok 4.6 | 60.59 | 600/600 | 61.16 | 600/600 |
| Grok 4.7 | 63.62 | 600/600 | — | 595/600 |
| Kimi K2.6 | 54.40 | 600/600 | 52.45 | 600/600 |
| Kimi K3 | 58.59 | 600/600 | 61.81 | 600/600 |
| MiMo V2.6 Flash | 49.73 | 600/600 | 50.40 | 600/600 |
| MiMo V2.6 Pro | 48.60 | 600/600 | 49.91 | 600/600 |
| Mistral Large 4 | 46.07 | 600/600 | 54.54 | 600/600 |
| Muse Glimmer 30B | 60.70 | 600/600 | 60.08 | 600/600 |
| Muse Spark 1.1 | 68.78 | 600/600 | 67.53 | 600/600 |
| Muse Spark 1.2 | 68.70 | 600/600 | 68.60 | 600/600 |
| Muse Spark 1.3 | — | 598/600 | — | 599/600 |
| Qwen3 VL 235B A22B Instruct | 56.36 | 600/600 | 55.45 | 600/600 |
| Qwen3.5-27B | 60.34 | 600/600 | 61.70 | 600/600 |
| Qwen3.6 27B | 60.50 | 600/600 | 59.80 | 600/600 |
| Qwen3.7 Flash | 54.31 | 600/600 | 62.26 | 600/600 |
| Qwen3.7 Plus | 60.25 | 600/600 | 65.54 | 600/600 |
| Qwen3.8 27B | 54.34 | 600/600 | 57.81 | 600/600 |
| Qwen3.8 Flash | 58.87 | 600/600 | 62.93 | 600/600 |
| Qwen3.8 Max | 74.02 | 600/600 | 71.78 | 600/600 |

## Regenerate from the committed results

```bash
uv run vlm-exam validate
uv run vlm-exam summary --dataset-directory data/detection/train
uv run vlm-exam summary --check
```

`web/benchmark_summary.json` is the single website payload. Its existing
model/effort rows include the four text categories and `text_overall`, their 0–100
scores, per-category coverage, preliminary single-run protocol, and source result
checksums.
`model-labels.json` supplies card/chart names and lab affiliations only; it never
changes inference provenance. The compact chart manifest fingerprints rendering
inputs and PNGs. The standard validation and summary checks cover the exact
release inventory, accepted gaps and generated artifacts.
See [the workflow](../docs/text-benchmark.md) for rendering individual cards
and for the distinction between frozen imported profiles and fresh native runs.

Provider request/account identifiers have been removed from error diagnostics, generation identifiers omitted, and internal experiment labels made self-contained. Predictions, references, scores, image hashes, model routes, inference hashes, retry counts and failure kinds are preserved.

## Leaderboards

Only configurations with all 600 scored pairs are plotted: 47 Low and 45 High.
These existing PNGs are unchanged by the unified JSON migration. The JSON
preserves scores and coverage per category, including fully scored categories
from configurations with an accepted gap elsewhere. All 48 models remain in the
coverage table and summary.

### Single-string extraction

![Single-string extraction, Low](../visualizations/leaderboards/text_single_string_low.png)

[High effort](../visualizations/leaderboards/text_single_string_high.png)

### Transcription

![Transcription, Low](../visualizations/leaderboards/text_transcription_low.png)

[High effort](../visualizations/leaderboards/text_transcription_high.png)

### Structured JSON extraction

![Structured JSON extraction, Low](../visualizations/leaderboards/text_structured_low.png)

[High effort](../visualizations/leaderboards/text_structured_high.png)

### Text localization and recognition

![Text localization and recognition, Low](../visualizations/leaderboards/text_localization_recognition_low.png)

[High effort](../visualizations/leaderboards/text_localization_recognition_high.png)
