# SAM 3 reference environment

Isolated uv project and adapter package for running SAM 3 reference
benchmarks. Heavy PyTorch and Transformers dependencies stay out of the main
`vlm-exam` environment.

## Prerequisites

1. Request access to [facebook/sam3](https://huggingface.co/facebook/sam3) on Hugging Face.
2. Authenticate locally:

```bash
hf auth login
```

## Setup

```bash
cd reference/sam3
uv sync
```

First run downloads the `facebook/sam3` checkpoint (~848M parameters). The
revision is pinned in `src/vlm_exam/reference/configs/reference_models.yaml`.

## Smoke test

```bash
uv run vlm-exam reference-run \
  --model sam3 \
  --dataset-directory ../../data/detection/train \
  --output-directory ../../results-reference-smoke \
  --max-samples 5 \
  --device auto
```

## Full benchmark

```bash
uv run vlm-exam reference-run \
  --model sam3 \
  --dataset-directory ../../data/detection/train \
  --output-directory ../results \
  --device auto
```

## Image-conditioned prompts

SAM 3 accepts one text phrase per forward pass; the adapter loops over the
per-image class list. Frozen prompt sets are available under
`reference/prompts/image_conditioned/` for v1, v2 none, and v2 overlay.

```bash
uv run vlm-exam reference-run \
  --model sam3 \
  --dataset-directory ../../data/detection/train \
  --output-directory ../results \
  --prompt-set ../prompts/image_conditioned/v2-overlay/prompts.jsonl
```

## Notes

- Boxes come from `post_process_instance_segmentation` in absolute xyxy pixel
  coordinates of the original image.
- Default `threshold=0.05` is used for COCO-style mAP ranking (manifest records
  this as a deviation from the HF docs default of 0.5).
- Masks are produced by SAM 3 but ignored for benchmark evaluation; only boxes
  are serialized.
- One forward pass is run per prompted class per image; full runs are slower
  than YOLO-E batch vocabulary mode.
