# MiniMax M3.1 Flash benchmark

This evaluation targets the full VLM Exam protocol: three full runs at Low and
High for OCR, extraction, counting, identification, reasoning and detection,
plus one full mixed-text-v2 run at each effort (38 result files, 5,046 model
requests before retries).

## Inference settings

- OpenRouter model: `minimax/minimax-m3.1-flash:free`.
- Provider: `minimax`, with no provider fallback.
- Data collection: `deny`; storing prompts is acceptable, training on them is not.
- Reasoning: explicit `low` or `high`, never disabled.
- Output budget: 16,384 tokens, including reasoning.
- Existing OpenRouter image preprocessing, datasets, prompts and scoring.
- The shared inference pool starts at 32 requests and can grow to 64. Multiple
  task/effort/repeat configurations share this pool. There is no fixed request
  start delay. HTTP 429 halves concurrency (minimum two) and applies a shared
  60-second cooldown; concurrency recovers by two after 120 seconds without a
  new rate limit. Gate waits are excluded from measured inference duration.

The paid model currently has no advertised endpoint and returned HTTP 404 in
preflight. The working free route has a 1,000-request daily allowance on the
benchmark account. There is no automatic switch to another route or model.
The scheduler reserves requests from the available daily allowance and stops
submitting when that allowance is exhausted, then drains in-flight calls. It
checkpoints each completed sample.
Only complete, successful full-dataset runs are copied into `results/`.

The launch route costs zero. Published token-cost estimates must use the actual
route price rather than presenting MiniMax's future list price as a billed cost.
The ordinary published price is $0.15 input and $0.60 output per million tokens.

## Coordinate selection

No official native detection schema was found. Compare these explicitly prompted
conventions on the same first 50 detection images at Low effort:

- `xyxy_normalized_0_to_1000`
- `yxyx_normalized_0_to_1000`
- `xyxy_absolute_original_image`

The 50-image comparison selected `xyxy_absolute_original_image`: mAP@50:95
32.19%, versus 28.75% for normalized YXYX and 26.23% for normalized XYXY.
Its mAP@50 is 53.42% and mAP@75 is 33.18%.
Keep all probe predictions outside `results/`. This establishes an empirical
prompt convention, not a claim about a documented native format. No existing
MiniMax-family baseline is registered, so the usual same-family comparison is
unavailable. Freeze the selected convention before any full run.

The model registry, text release inventory, summary and leaderboards remain
unchanged until complete benchmark results are ready. This draft is not a
published benchmark entry, and must not be marked as a legacy model.

## Sources

- [OpenRouter model and launch terms](https://openrouter.ai/minimax/minimax-m3.1-flash:free)
- [MiniMax capabilities and mandatory reasoning](https://agent.minimax.io/tools/m3.1-flash)
- [Provider training policies](https://openrouter.ai/Providers)
- [OpenRouter data-policy routing](https://openrouter.ai/docs/guides/privacy/provider-logging)

## Live monitoring

The local dashboard shows per-job checkpoints, actual in-flight API calls,
current concurrency and ceiling, 429 cooldowns, the remaining account quota,
and both the coordinate-probe and full-benchmark logs. It refreshes every
three seconds. API metadata refreshes every thirty seconds.

The previous four-worker runner drained at a job boundary before acceleration.
Its two complete OCR Low runs (74 answers) were preserved without resubmission.
Model inputs, output token budget, reasoning, coordinate format and scoring did
not change during the scheduler migration.
