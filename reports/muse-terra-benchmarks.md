# Muse Spark Open and GPT-6.1 Terra benchmarks

This publication adds only Muse Spark Open and GPT-6.1 Terra to the existing main-branch models. Both are complete: 38 full result files and 5,046 samples per model (76 files / 10,092 samples total). The six original tasks have three runs at Low and High; Text has one 600-pair run at each effort. Scores, datasets, prompts, and judge verdicts are unchanged from the completed campaigns. No paid inference was repeated for this split.

## Quality

Values are percentages; Detection uses mAP@50, OCR uses similarity, Text uses its overall score, and the other tasks use judge accuracy. Original-task values average the three run scores. Text is preliminary single-run evidence.

| Model | Effort | Text | Counting | Identification | Reasoning | Detection | OCR | Extraction |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Muse Spark Open | low | 67.64 | 75.68 | 82.29 | 63.80 | 57.78 | 93.44 | 88.66 |
| Muse Spark Open | high | 68.59 | 75.23 | 80.21 | 65.12 | 60.87 | 92.02 | 88.32 |
| GPT-6.1 Terra | low | 81.92 | 80.18 | 96.88 | 85.43 | 84.74 | 92.21 | 91.75 |
| GPT-6.1 Terra | high | 84.42 | 84.23 | 96.88 | 86.98 | 84.66 | 91.42 | 92.10 |

## Frozen request settings

- Muse Spark Open: `meta/muse-spark-open` through OpenRouter, pinned to Together with no fallback and `data_collection: deny`. Low/High reasoning, 65,536 output-token cap. Detection and Text localization use flat named XYXY integer coordinates normalized to 0–1000. [Model endpoint](https://openrouter.ai/meta/muse-spark-open), [official model card](https://huggingface.co/meta-models/Muse-Spark-Open-816B-A42B).
- GPT-6.1 Terra: `gpt-6.1-terra` through the native OpenAI Responses API, Low/High reasoning, `max_output_tokens: 65536`, `store: false`, `service_tier: default`. Existing OpenAI preprocessing and XYXY absolute uploaded-image pixels. The user explicitly waived a new Terra coordinate-format search. [Official model documentation](https://developers.openai.com/api/docs/models/gpt-6.1-terra).

Muse's High-effort search used the same 50 images for three candidates. Flat named 0–1000 scored 62.22 mAP@50, `box_2d` XYXY 0–1000 scored 61.29, and nested Meta bbox scored 60.51. This is a small observed advantage, not proof of statistical superiority; Low optimality was not separately established. The selected prompt was frozen before the full Detection and Text runs. [Comparison](muse-spark-open-detection-search.json) and [raw search evidence, text probes, and family baselines](muse-spark-open-detection-evidence.zip).

## Cost and speed

Shared quality and efficiency charts are regenerated in `visualizations/leaderboards/`, and the website payload is regenerated in `web/benchmark_summary.json`. Per-task cost uses the repository's input/output-token prices and averages one complete run over repeats; latency is inference seconds per sample, not concurrent wall time. These chart estimates exclude judge costs, retry charges, and cache discounts. The website overview retains the existing five-task pool; standalone efficiency charts retain their existing task selection.

Terra's operational accounting is $25.0535355 for the full campaign including judging, plus $0.1183665 native preflight. Native API costs are conservative estimates rather than invoice reconciliation; nine retries without returned usage have an additional upper estimate of $2.9812825. See [operational accounting](openai-terra-operational-summary.json) and [billing evidence](openai-terra-billing-evidence.zip). Deployment-block smoke tests are separate from these benchmark results.

## Provenance and validation

- [Byte-for-byte result-copy audit](muse-terra-source-audit.json): 76 files / 10,092 samples; original result timestamps retained.
- Frozen manifests: [Muse](muse-spark-open-manifest.json), [Terra](openai-terra-manifest.json).
- Terra: [checkpoint audit](openai-terra-result-audit.json), [export audit](openai-terra-export-audit.json), [input-token verification](openai-terra-input-count-audit.json). Original operational evidence is preserved; its historical source paths refer to the campaign checkout.
- This PR includes only provider option/usage support needed by these two configurations. Campaign scheduling, broader recovery tooling and the four other models remain in the follow-up PR.
- Validation results are recorded in [the split validation report](muse-terra-validation.json).
