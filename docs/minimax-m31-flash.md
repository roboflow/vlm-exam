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
- Requests are paced to at most one start per 3.2 seconds, with four in flight.
  Pacing is outside the provider's measured inference duration.

The paid model currently has no advertised endpoint and returned HTTP 404 in
preflight. The working free route has a 1,000-request daily allowance on the
benchmark account. There is no automatic switch to another route or model.
The runner checks quota before each job and checkpoints each completed sample.
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

Select by dataset mAP@50:95, using mAP@50 and mAP@75 as additional diagnostics.
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
