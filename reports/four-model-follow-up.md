# Four-model benchmark follow-up

This PR is based on `feat/muse-terra-benchmarks`. Muse Spark Open and GPT-6.1 Terra are inherited from that base; this diff publishes only DeepSeek V4.1 Flash 1017, MiniMax M3.1 Flash, Kimi K3.1 Pro, and NVIDIA Nemotron 3.5 Super VL.

The four models add 152 complete result files and 20,184 samples. Six original tasks have three runs per effort; Text has one 600-pair run per effort. Existing results, prompts, provider settings, scores, and source timestamps are preserved. Nemotron remains Crusoe BF16, XYXY 0–1000, per the user's explicit choice.

| Model | Effort | Text | Counting | Identification | Reasoning | Detection mAP@50 |
|---|---|---:|---:|---:|---:|---:|
| DeepSeek V4.1 Flash 1017 | low | 55.21 | 65.32 | 86.46 | 63.36 | 49.43 |
| DeepSeek V4.1 Flash 1017 | high | 55.88 | 61.71 | 82.29 | 65.56 | 50.24 |
| MiniMax M3.1 Flash | low | 50.75 | 65.77 | 78.12 | 59.82 | 55.70 |
| MiniMax M3.1 Flash | high | 51.97 | 67.57 | 82.29 | 63.58 | 57.09 |
| Kimi K3.1 Pro | low | 69.79 | 77.48 | 89.58 | 81.90 | 72.16 |
| Kimi K3.1 Pro | high | 70.93 | 77.03 | 90.62 | 80.79 | 70.37 |
| NVIDIA Nemotron 3.5 Super VL | low | 58.61 | 71.62 | 84.38 | 56.73 | 55.69 |
| NVIDIA Nemotron 3.5 Super VL | high | 55.87 | 65.32 | 81.25 | 56.07 | 54.53 |

The [original campaign report](openrouter-five-models.md) and its evidence archives retain shared historical provenance and accounting, including the Muse runs that have moved to the base PR. They do not introduce duplicate benchmark entries. The summary and charts on this branch contain the base models plus all four follow-up models.

After the two-model PR is merged, retarget this PR to `main` and resolve/rebase any generated-artifact changes before merging. Do not merge this PR first.
