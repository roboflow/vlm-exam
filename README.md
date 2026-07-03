# vlm-exam

Benchmark suite for Vision Language Models. Compare accuracy, cost, and
speed across frontier VLMs on standardized visual tasks.

## Supported tasks

- **VQA / OCR** -- visual question answering and optical character recognition

## Supported providers

- Anthropic (Claude)
- Google (Gemini)
- OpenAI (GPT)

## Installation

```bash
pip install vlm-exam
```

Or install from source:

```bash
git clone https://github.com/roboflow/vlm-exam.git
cd vlm-exam
pip install -e ".[dev]"
```

## Quick start

### CLI

```bash
export ANTHROPIC_API_KEY=...
export GOOGLE_API_KEY=...
export OPENAI_API_KEY=...

vlm-exam run \
    --task vqa \
    --models claude-fable-5,gemini-3.5-flash,gpt-5.5 \
    --effort high \
    --dataset-directory /path/to/vqa/dataset

vlm-exam report --results-directory results/
```

### Python

```python
from vlm_exam import load_config, create_provider, create_task, run_benchmark

config = load_config()
task = create_task("vqa")
samples = task.load_samples("/path/to/vqa/dataset")
provider = create_provider("anthropic", model="claude-fable-5")

results = run_benchmark(task=task, provider=provider, samples=samples, effort="high")
```

## Configuration

Model definitions, pricing, and lab branding live in
`src/vlm_exam/configs/models.yaml`. Add a new model by editing this file --
no code changes required.

## License

Apache 2.0. See [LICENSE](LICENSE).
