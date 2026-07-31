# YOLO-E reference environment

Isolated uv project and adapter package for running YOLO-E reference
benchmarks. Ultralytics, PyTorch, and AGPL model assets stay out of the main
`vlm-exam` environment.

## Setup

```bash
cd reference/yoloe
uv sync
```

First run downloads:

- checkpoint weights (for example `yoloe-11l-seg.pt`)
- MobileCLIP text encoder (`mobileclip_blt.ts`)
- Ultralytics CLIP fork (auto-installed by ultralytics)

The expected checkpoint SHA256 values are pinned in
`src/vlm_exam/reference/configs/reference_models.yaml`; the adapter refuses
weights that do not match.

## Smoke test

```bash
uv run vlm-exam reference-run \
  --model yoloe-11l-seg \
  --dataset-directory ../../data/detection/train \
  --output-directory ../../results-reference-smoke \
  --max-samples 20 \
  --device auto
```

## Full benchmark

```bash
uv run vlm-exam reference-run \
  --model yoloe-11l-seg \
  --dataset-directory ../../data/detection/train \
  --output-directory ../results \
  --device auto

uv run vlm-exam reference-run \
  --model yoloe-26x-seg \
  --dataset-directory ../../data/detection/train \
  --output-directory ../results \
  --device auto
```

## Validate and evaluate

From the main repo root (or this env):

```bash
uv run vlm-exam reference-validate \
  --results-file reference/results/detection_yoloe-11l-seg_reference_*.jsonl \
  --dataset-directory data/detection/train

uv run vlm-exam detection-report \
  --results-directory reference/results \
  --dataset-directory data/detection/train
```

## Notes

- All YOLO-E checkpoints are `-seg` models; reference runs use `boxes` only.
- Default evaluation uses `conf=0.001` to retain low-confidence boxes for mAP ranking.
- On Apple Silicon, MobileCLIP text embeddings run on CPU during `set_classes`
  (MPS does not support the float64 ops in the TorchScript encoder); inference
  still runs on MPS.
- Ultralytics and YOLO-E weights are AGPL-3.0; keep this env isolated from the main package dependencies.
