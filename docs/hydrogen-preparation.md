# Gemini 4 Hydrogen

Use the production Gemini API model through the opt-in configuration:

```sh
vlm-exam run --config src/vlm_exam/configs/previews/hydrogen.yaml \
  --task text --models gemini-4-hydrogen --effort low \
  --dataset-directory data/text/train --max-samples 4 \
  --output-directory /tmp/hydrogen-smoke --concurrency 2
```

Set `GOOGLE_API_KEY` before running. The configuration uses Gemini API `v1beta`,
`gemini-4-hydrogen`, and a 131,072-token output limit, including thinking.
Benchmark Low maps to native `low`, Medium to `medium`, and High to `xhigh`.
Google's native `high` is a different setting from benchmark High here.
Existing models retain their previous settings unless route options are supplied.

Python 3.11+ and Google Gen AI SDK 3.2.0 or later are required for automatic continuation handling.
An unresolved `CONTINUATION` response is rejected instead of being scored as a
completed answer. Python 3.10 retains its existing SDK compatibility for older
models; run Hydrogen on Python 3.11 or later. Results record the requested thinking level and token limit,
candidate/thinking token counts, and terminal finish reasons. `MAX_TOKENS` may
still produce an empty answer, even at the larger budget.

The detection coordinate format is Gemini's `box_2d` array in
`[y_min, x_min, y_max, x_max]` order, normalized to 0..1000. This matches the
Gemini image-understanding contract and the earlier Hydrogen detection probes.

The production endpoint was smoke-tested through this provider at Low and High
on November 12, 2026. Both returned the expected image text with `STOP`.
Smoke tests verify API integration; they are not published benchmark runs.

The opt-in file is deliberately outside the default model inventory until the
complete publication protocol is available. No benchmark results or synthetic
leaderboard entries are added. Historical `gemini-4-hydrogen-eap` results must
keep their original identity and must not be relabeled as production results.
The existing summary and release inventory remain unchanged.

Sources:

- https://blog.google/innovation-and-ai/models-and-research/gemini-models/introducing-gemini-4-hydrogen/
- https://ai.google.dev/gemini-api/docs/image-understanding#object-detection
- https://googleapis.github.io/python-genai/
