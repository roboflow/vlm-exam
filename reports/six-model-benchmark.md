# Six-model benchmark

Completed 30,276 samples in 228 full result files. The original five models use pinned paid OpenRouter routes; GPT-6.1 Terra uses the direct OpenAI Responses API. Original tasks have three runs per effort; Text has one 600-pair run at each effort.

Scores are percentages: OCR similarity; judge accuracy for extraction, counting, identification and reasoning; detection mAP@50; deterministic Text mean. These different metrics should not be averaged into an invented overall score.

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
| GPT-6.1 Terra | low | 92.21 | 91.75 | 80.18 | 96.88 | 85.43 | 84.74 | 81.92 |
| GPT-6.1 Terra | high | 91.42 | 92.10 | 84.23 | 96.88 | 86.98 | 84.66 | 84.42 |

## Recorded cost and observed speed

| Model | Model cost | Cost basis | Mean latency | Median latency |
|---|---:|---|---:|---:|
| muse-spark-open | $36.70 | Reported | 14.15s | 6.53s |
| deepseek-v4.1-flash-1017 | $22.27 | Reported | 12.70s | 4.12s |
| minimax-m3.1-flash | $7.38 | Reported | 19.73s | 7.99s |
| nemotron-3.5-super-vl | $9.50 | Reported | 23.89s | 10.17s |
| kimi-k3.1-pro | $105.61 | Reported | 53.58s | 20.20s |
| GPT-6.1 Terra | $23.22 | Conservative estimate | 10.70s | 6.41s |

Costs include recorded model attempts across all repetitions; judges and calibration are separate. Standard efficiency charts instead compare the mean cost of a single full run under the repository task pool.

## Terra protocol and cost

The user explicitly requested direct OpenAI API access and waived additional box-format assessment. Terra uses `gpt-6.1-terra`, explicit Low/High reasoning, the standard service tier, `store=false`, and a 65,536-token output limit. Detection and Text localization use XYXY in uploaded-image pixels, with the existing OpenAI PNG preprocessing and 2,048-pixel maximum edge. The separate earlier OpenRouter calibration is not mixed into these results.

Terra's complete evaluation accounted for $25.05: $23.22 conservatively estimated native inference and $1.83 reported judge charges. The native preflight added $0.12. Mean successful-sample latency was 10.70s; median 6.41s. Latency is not concurrent campaign wall time.

There were 9 transport retry attempts without returned usage. Their charges are unknown, not assumed zero; the maximum additional model cost under the frozen token limits is $2.98. Completed predictions were not repeated during the scheduler migration.

Native prices and reasoning support: [OpenAI model documentation](https://developers.openai.com/api/docs/models/gpt-6.1-terra). See [native preflight and approval](openai-terra-preflight.json), [frozen manifest](openai-terra-manifest.json), [quality metrics](openai-terra-metrics.json), [operational metrics](openai-terra-operational-summary.json), and [billing evidence](openai-terra-billing-evidence.zip). The [five-model report](openrouter-five-models.md) retains the original providers, detection-search limitations, retries and reported costs. Standard shared leaderboards compare quality, cost and speed with all existing models.

The shared pool was configured for 48 workers. Initial full-context budget reservations restricted admission; a scheduler-only checkpoint migration then used exact OpenAI input-token counts while retaining the full output-token reservation. See [input-count audit](openai-terra-input-count-audit.json) and [migration receipt](openai-terra-scheduler-import.json).
