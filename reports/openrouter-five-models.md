> Historical report for the original shared five-model campaign. Muse Spark Open and GPT-6.1 Terra are published by the prerequisite two-model PR. This follow-up adds only DeepSeek, MiniMax, Kimi and Nemotron. See [the follow-up scope](four-model-follow-up.md).

# Five-model OpenRouter benchmark

Completed 25,230 samples in 190 full result files: 5,046 samples and 38 files per model. Each original task has three repetitions at Low and High; Text has one 600-pair run per effort. All 70 task/effort configurations are complete.

## Quality

Scores are percentages. OCR is similarity; extraction, counting, identification and reasoning use the existing judge accuracy; detection is mAP@50; Text is the mean of all 600 deterministic pair scores. Original-task values average three complete repetitions. Different task metrics should not be averaged into an invented overall score.

| Model | Effort | OCR | Extraction | Counting | Identification | Reasoning | Detection | Text |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Muse Spark Open | low | 93.44 | 88.66 | 75.68 | 82.29 | 63.80 | 57.78 | 67.64 |
| Muse Spark Open | high | 92.02 | 88.32 | 75.23 | 80.21 | 65.12 | 60.87 | 68.59 |
| DeepSeek V4.1 Flash 1017 | low | 89.10 | 84.19 | 65.32 | 86.46 | 63.36 | 49.43 | 55.21 |
| DeepSeek V4.1 Flash 1017 | high | 89.29 | 80.41 | 61.71 | 82.29 | 65.56 | 50.24 | 55.88 |
| MiniMax M3.1 Flash | low | 88.81 | 76.29 | 65.77 | 78.12 | 59.82 | 55.70 | 50.75 |
| MiniMax M3.1 Flash | high | 86.78 | 76.29 | 67.57 | 82.29 | 63.58 | 57.09 | 51.97 |
| Kimi K3.1 Pro | low | 91.95 | 89.69 | 77.48 | 89.58 | 81.90 | 72.16 | 69.79 |
| Kimi K3.1 Pro | high | 92.76 | 89.00 | 77.03 | 90.62 | 80.79 | 70.37 | 70.93 |
| NVIDIA Nemotron 3.5 Super VL | low | 87.17 | 77.66 | 71.62 | 84.38 | 56.73 | 55.69 | 58.61 |
| NVIDIA Nemotron 3.5 Super VL | high | 88.49 | 76.98 | 65.32 | 81.25 | 56.07 | 54.53 | 55.87 |

Kimi has the highest detection, reasoning and Text scores among these five models at both effort levels, with the highest reported model spending. High effort does not consistently improve quality. These observations are descriptive; the benchmark does not establish statistical significance for every difference.

## Actual campaign spending and observed speed

The table includes all recorded model generations, including retries and invalidated runs. Latency describes completed samples across both efforts and all repetitions; it is not campaign wall time. The standard efficiency charts separately use the repository's task pool and mean cost of one full run.

| Model | Completed samples | Reported model cost | Mean latency / sample | Median latency / sample |
|---|---:|---:|---:|---:|
| muse-spark-open | 5,046 | $36.70 | 14.15s | 6.53s |
| deepseek-v4.1-flash-1017 | 5,046 | $22.27 | 12.70s | 4.12s |
| minimax-m3.1-flash | 5,046 | $7.38 | 19.73s | 7.99s |
| nemotron-3.5-super-vl | 5,046 | $9.50 | 23.89s | 10.17s |
| kimi-k3.1-pro | 5,046 | $105.61 | 53.58s | 20.20s |

Shared judge charges were $3.93, bringing reported full-campaign spending to **$185.39**. Detection search and intervening calibration added $23.74; Terra calibration added $1.93. Earlier pre-budget calibration was separate. Eight campaign generations and one Terra calibration generation have unresolved reported usage; lookup 404s also occurred for a known billed control, so they must not be interpreted as free requests. One earlier Kimi transport/decode failure also lacks a usable billing record. These totals are reported ledger costs, not an exact account-debit reconciliation.

## Frozen configurations and provenance

| Model | Paid route | Detection format |
|---|---|---|
| Muse Spark Open | Together | XYXY, normalized 0–1000, Meta flat JSON shape |
| DeepSeek V4.1 Flash 1017 | Together | XYXY, normalized 0–1000 |
| MiniMax M3.1 Flash | MiniMax | XYXY, original-image pixels |
| Kimi K3.1 Pro | Moonshot AI MXFP4 | XYXY, normalized 0–1000 |
| NVIDIA Nemotron 3.5 Super VL | Crusoe BF16 | XYXY, normalized 0–1000 |

Routes prohibit training, disable provider fallback, and require supported parameters. All model requests use explicit Low/High reasoning and a 65,536-token output cap. Nemotron's route accepts these settings; this is not independent attestation of its backend reasoning implementation. The judge uses the existing prompt and temperature on the pinned Gemini 3.5 Flash dated model through Google Vertex.

The High-only coordinate search used the same 50 images for each candidate. Search images overlap the evaluation dataset; there is no held-out format-selection split. Some pixel prompt variants produced identical requests, and Nemotron sometimes disregarded requested conventions. The critical-review report documents these limitations. The chosen Nemotron scale of 1,000 follows the user's explicit decision. Its earlier scale-999 results were invalidated and archived, with charges retained.

Recorded model generations include eight DeepSeek token-limit completions and 53 Nemotron token-limit completions plus 23 Nemotron error completions. Nemotron also has 84 normal completions from the invalidated scale-999 configuration. These are kept in the billing ledger, not included as extra scored repeats. Technical failures were retried with the same frozen settings, preserving every successful answer regardless of correctness. Non-normal provider completions, including a discovered partial error response, were rejected. The migration preserved valid checkpoint rows and all recorded charges. Retrying until a normal response measures quality conditional on a completed response; recorded failure/retry costs and the recovery reports expose that limitation.

## Validation and evidence

- [Final validation commands and outcomes](openrouter-five-models-validation.json), including independent detection recomputation and summary/chart freshness checks.
- [Complete checkpoint and Text-score audit](openrouter-text-score-audit.json): all 25,230 results, all 6,000 Text scores independently recomputed, no errors.
- [Task metrics and per-run values](openrouter-five-models-metrics.json).
- [Actual spending, tokens and latency](openrouter-five-models-operational-summary.json) and [billing ledger evidence](openrouter-five-models-billing-evidence.zip).
- [Frozen manifest](openrouter-five-models-manifest.json) and [evaluation plan](openrouter-five-models-plan.json).
- [Detection search critical review](openrouter-detection-search-critical-review.json) and [archived evidence](openrouter-detection-search-evidence.zip).
- [Response-validation checkpoint migration](openrouter-five-models-validation-import.json).
- [Nemotron deferred-response recovery](openrouter-nemotron-deferred-recovery-review.json).

Repository protocol validation passes with 266 pre-existing historical warnings. Ruff check and format pass. The final implementation test suite passed 356 tests after the response-validation and dashboard-publication fixes.

## GPT-6.1 Terra addition

The additional Terra benchmark is complete: 5,046 samples in 38 full result files, using the direct OpenAI Responses API as requested. The user approved up to $50 within the $300 overall cap and waived further coordinate-format assessment. The frozen format is XYXY in uploaded-image pixels, following the existing OpenAI preprocessing protocol.

The native run accounted for $25.05 plus $0.12 preflight; nine transport retries have no returned usage, with a maximum additional model cost of $2.98 under the frozen token limits. Native charges are conservative token-based estimates, not an invoice reconciliation. Earlier OpenRouter calibration remains archived separately and is excluded from full results.

See the [six-model quality, cost and speed comparison](six-model-benchmark.md), [native result audit](openai-terra-result-audit.json), and [native operational summary](openai-terra-operational-summary.json). Historical OpenRouter calibration remains available in the [calibration report](openrouter-gpt-6.1-terra-calibration.json) and [evidence archive](openrouter-gpt-6.1-terra-calibration.zip).
