# Reference detectors

SAM 3 and YOLO-E are local open-vocabulary detector baselines evaluated on the
same 250-image detection dataset as the hosted VLMs. They are comparison
references, not entries in the main VLM benchmark.

Reference model keys never belong in `src/vlm_exam/configs/models.yaml`, raw
reference runs never belong in `results/`, and reference rows are excluded from
the website payload and main leaderboard charts.

## Results

| Model | Class names mAP@50 | Image-conditioned mAP@50 | Best prompt mAP@50 |
| --- | ---: | ---: | ---: |
| SAM 3 | 0.3913 | 0.4878 | 0.5337 |
| YOLOE-11l | 0.1829 | 0.2240 | 0.2462 |
| YOLOE-26x | 0.2020 | 0.2397 | 0.2671 |

Best prompt is a per-image oracle that keeps whichever of the class-name and
image-conditioned runs scores higher on that image. It is a diagnostic
heuristic, not a deployable single-pass setting or a guaranteed upper bound on
dataset-level mAP.

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

### YOLO-E and Gemini 3.5 Flash

![Class-name comparison](leaderboards/yoloe/gemini-focus/detection_map50_class_names.png)

![Image-conditioned comparison](leaderboards/yoloe/gemini-focus/detection_map50_augmented_prompt.png)

## Layout

- `results/`: six committed full-dataset runs, one class-name and one
  image-conditioned run per model
- `leaderboards/`: generated mixed VLM/reference charts and tables
- `prompts/image_conditioned/v1/`: frozen Gemini-generated prompt asset
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

To use the frozen image-conditioned prompts:

```bash
uv run vlm-exam reference-run \
  --model sam3 \
  --dataset-directory ../../data/detection/train \
  --output-directory ../results \
  --prompt-set ../prompts/image_conditioned/v1/prompts.jsonl
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
