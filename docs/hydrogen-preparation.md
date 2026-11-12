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

## Published benchmark

The default registry publishes `gemini-4-hydrogen-eap`, preserving the endpoint
used for the measurements. It includes three repeats at Low and High for each
of OCR, extraction, counting, identification, reasoning and detection, plus one
complete frozen 600-pair Text run per effort. Benchmark High is native `xhigh`.
The original tasks and Text Low use the original provider-default output budget
(65,536 tokens). Text High is the explicitly authorized full replacement at
131,072 tokens, with its original inference hash pinned in the Text release
inventory. It is not pooled with the earlier 65,536-token Text High run.

The full Text High replacement preserves terminal responses from its first pass
and completes unresolved continuation responses using a continuation-capable
SDK. Empty terminal model responses remain scored outcomes; none are retried or
removed to improve the results. All published source files are copied unchanged.

Later diagnostics used 625 detection images and only 550 Text pairs, excluding
transcription. Those experiments are separate from this publication protocol and
are not included in `results/`, the website summary or these leaderboards.

The production `gemini-4-hydrogen` opt-in configuration above remains separate.
Historical EAP runs are never relabeled as production measurements. Published
costs use launch-price estimates ($1.50 input / $7.50 output per million tokens),
not actual early-access billing. Default registry settings reproduce the original
65,536-token runs; reproducing Text High requires explicitly setting the Google
route option `max_output_tokens: 131072` in a custom configuration.

Sources:

- https://blog.google/innovation-and-ai/models-and-research/gemini-models/introducing-gemini-4-hydrogen/
- https://ai.google.dev/gemini-api/docs/image-understanding#object-detection
- https://googleapis.github.io/python-genai/
