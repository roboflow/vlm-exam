# Reference detectors

SAM 3 and YOLO-E are local open-vocabulary detector baselines evaluated on the
same 250-image detection dataset as the hosted VLMs. They are comparison
references, not entries in the main VLM benchmark.

Reference model keys never belong in `src/vlm_exam/configs/models.yaml`, raw
reference runs never belong in `results/`, and reference rows are excluded from
the website payload and main leaderboard charts.

## Results

| Model | Class names | v1 | v2 none | v2 overlay |
| --- | ---: | ---: | ---: | ---: |
| SAM 3 | 0.3915 | 0.4892 | 0.5046 | **0.5205** |
| YOLOE-11l | 0.1829 | 0.2240 | 0.2279 | **0.2412** |
| YOLOE-26x | 0.2020 | **0.2397** | 0.2297 | 0.2391 |

All values are dataset-level mAP@50 on the same 250 images. Class names use no
generated descriptions. The frozen prompt pipelines are:

- `v1`: Gemini sees the clean image and one class at a time.
- `v2 none`: Gemini sees the clean image and all classes together.
- `v2 overlay`: Gemini sees all classes together on a copy annotated with up to
  12 ground-truth boxes per class.

The overlay is used only to generate text. Reference detectors always receive
the original clean image and the generated text prompts. The v2 pipelines are
offline, dataset-specific experiments because prompt generation uses the
ground-truth class list and, for overlay, ground-truth boxes.

### SAM 3 compared with VLMs

![SAM 3 and VLM detection mAP@50 leaderboard](leaderboards/sam3/detection_map50.png)

[mAP@75](leaderboards/sam3/detection_map75.png) |
[mAP@50:95](leaderboards/sam3/detection_map50_95.png) |
[table](leaderboards/sam3/leaderboard.md)

### YOLO-E compared with VLMs

![YOLO-E and VLM detection mAP@50 leaderboard](leaderboards/yoloe/detection_map50.png)

[mAP@75](leaderboards/yoloe/detection_map75.png) |
[mAP@50:95](leaderboards/yoloe/detection_map50_95.png) |
[table](leaderboards/yoloe/leaderboard.md)

## Layout

- `results/`: 12 committed full-dataset runs, one per model and prompt mode
- `leaderboards/`: generated mixed VLM/reference charts and tables
- `prompts/image_conditioned/`: frozen v1, v2 none, and v2 overlay assets
- `scripts/`: reference-only generation and analysis utilities
- `sam3/` and `yoloe/`: isolated model projects, each with its own adapter,
  dependencies, lockfile, and setup guide
- `../src/vlm_exam/reference/`: shared runner, evaluation, reporting, and CLI
  integration without heavyweight model dependencies

## Run a benchmark

Set up the model-specific environment first:

```bash
cd reference/sam3
uv sync
uv run vlm-exam reference-run \
  --model sam3 \
  --dataset-directory ../../data/detection/train \
  --output-directory ../results
```

For YOLO-E, use `reference/yoloe` and select either `yoloe-11l-seg` or
`yoloe-26x-seg`. See each model directory's README for prerequisites and
checkpoint details.

To use a frozen image-conditioned prompt set:

```bash
uv run vlm-exam reference-run \
  --model sam3 \
  --dataset-directory ../../data/detection/train \
  --output-directory ../results \
  --prompt-set ../prompts/image_conditioned/v2-overlay/prompts.jsonl
```

Only full 250-image runs belong in `reference/results/`. Keep smoke and partial
runs in a local ignored directory.

## Validate and regenerate

From the repository root:

```bash
vlm-exam reference-validate \
  --results-file reference/results/<run>.jsonl \
  --dataset-directory data/detection/train

vlm-exam reference-detection-leaderboard \
  --dataset-directory data/detection/train \
  --output-directory reference/leaderboards

python reference/scripts/analyze_reference_detection.py \
  --dataset-directory data/detection/train
```

The mixed leaderboard reads VLM runs from `results/` but only writes under
`reference/leaderboards/`; it never modifies the main VLM leaderboard.
