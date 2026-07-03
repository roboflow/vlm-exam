# Copyright 2026 Roboflow, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

import click

from vlm_exam.config import load_config
from vlm_exam.providers import create_provider
from vlm_exam.results import RunResult, load_results, save_results
from vlm_exam.runner import run_benchmark
from vlm_exam.tasks import create_task


@click.group()
def main() -> None:
    """vlm-exam: Benchmark suite for Vision Language Models."""


@main.command()
@click.option(
    "--task",
    "task_name",
    required=True,
    help="Task to run (e.g. vqa).",
)
@click.option(
    "--models",
    required=True,
    help="Comma-separated model identifiers.",
)
@click.option(
    "--effort",
    required=True,
    help="Effort level (e.g. low, high).",
)
@click.option(
    "--dataset-directory",
    required=True,
    type=click.Path(exists=True),
    help="Path to the dataset directory.",
)
@click.option(
    "--output-directory",
    default="results",
    type=click.Path(),
    help="Directory to save result files.",
)
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(exists=True),
    help="Path to custom models.yaml config.",
)
def run(
    task_name: str,
    models: str,
    effort: str,
    dataset_directory: str,
    output_directory: str,
    config_path: str | None,
) -> None:
    """Run a benchmark for one or more models."""
    config = load_config(Path(config_path) if config_path else None)
    task = create_task(task_name)
    samples = task.load_samples(dataset_directory)
    model_ids = [model_id.strip() for model_id in models.split(",")]
    output_path = Path(output_directory)

    click.echo(f"Loaded {len(samples)} samples from {dataset_directory}")

    for model_id in model_ids:
        if model_id not in config.models:
            click.echo(f"Warning: model {model_id!r} not found in config, skipping.")
            continue

        model_config = config.models[model_id]
        provider = create_provider(
            model_config.provider, model=model_id
        )

        result = run_benchmark(
            task=task,
            provider=provider,
            samples=samples,
            effort=effort,
            task_name=task_name,
        )

        filename = f"{task_name}_{model_id}_{effort}_{result.timestamp}.jsonl"
        result_path = output_path / filename
        save_results(result, result_path)
        click.echo(f"Results saved to {result_path}")


@main.command()
@click.option(
    "--results-directory",
    default="results",
    type=click.Path(exists=True),
    help="Directory containing result JSONL files.",
)
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(exists=True),
    help="Path to custom models.yaml config.",
)
def report(
    results_directory: str,
    config_path: str | None,
) -> None:
    """Generate summary tables from saved results."""
    config = load_config(Path(config_path) if config_path else None)
    results_path = Path(results_directory)
    result_files = sorted(results_path.glob("*.jsonl"))

    if not result_files:
        click.echo(f"No .jsonl files found in {results_directory}")
        return

    runs: list[RunResult] = []
    for file_path in result_files:
        try:
            runs.append(load_results(file_path))
        except ValueError:
            click.echo(f"Skipping empty file: {file_path}")

    click.echo(
        f"\n{'Model':<25} {'Effort':>6} {'Correct':>8} "
        f"{'Total':>6} {'Accuracy':>9}"
    )
    click.echo("-" * 60)

    for run_result in runs:
        correct = sum(
            sample.correct for sample in run_result.samples
        )
        total = len(run_result.samples)
        accuracy = correct / total * 100 if total > 0 else 0.0

        click.echo(
            f"{run_result.model:<25} {run_result.effort:>6} "
            f"{correct:>8} {total:>6} {accuracy:>8.1f}%"
        )

    click.echo()

    click.echo(
        f"{'Model':<25} {'Effort':>6} {'Input Tok':>10} "
        f"{'Output Tok':>11} {'Cost':>9}"
    )
    click.echo("-" * 67)

    grand_cost = 0.0
    for run_result in runs:
        total_input = sum(
            sample.input_tokens for sample in run_result.samples
        )
        total_output = sum(
            sample.output_tokens for sample in run_result.samples
        )

        pricing = config.models.get(run_result.model)
        if pricing:
            cost = (
                (total_input / 1_000_000)
                * pricing.pricing.input_per_million_tokens
                + (total_output / 1_000_000)
                * pricing.pricing.output_per_million_tokens
            )
        else:
            cost = 0.0
        grand_cost += cost

        click.echo(
            f"{run_result.model:<25} {run_result.effort:>6} "
            f"{total_input:>10,} {total_output:>11,} "
            f"${cost:>8.4f}"
        )

    click.echo(f"\nTotal benchmark cost: ${grand_cost:.4f}")
