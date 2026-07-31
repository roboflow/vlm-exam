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

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from vlm_exam.config import BenchmarkConfig, ModelConfig, PricingConfig, RouteConfig
from vlm_exam.metrics import build_latest_runs_index
from vlm_exam.reference.best_prompt import merge_best_prompt_run
from vlm_exam.reference.config import ReferenceConfig
from vlm_exam.reference.constants import CANONICAL_BEST_PAIRS
from vlm_exam.results import RunResult, load_results, load_results_directory
from vlm_exam.tasks.detection import DetectionSample, compute_dataset_map

MixedDetectionSource = Literal[
    "vlm",
    "reference-baseline",
    "reference-image-conditioned",
    "reference-best",
]
LeaderboardFamily = Literal["sam3", "yoloe"]
ReferenceChartVariant = Literal["baseline", "image-conditioned", "best"]
YoloeGeminiFocusPrompt = Literal["class_names", "augmented_prompt"]

LEADERBOARD_FAMILIES: tuple[LeaderboardFamily, ...] = ("sam3", "yoloe")

REFERENCE_LEADERBOARD_NAMES: dict[str, str] = {
    "sam3": "SAM 3",
    "yoloe-11l-seg": "YOLOE-11l",
    "yoloe-26x-seg": "YOLOE-26x",
}

YOLOE_GEMINI_FOCUS_VLM = "gemini-3.5-flash"

REFERENCE_CHART_VARIANT_SUFFIX: dict[ReferenceChartVariant, str] = {
    "baseline": " (base prompt)",
    "image-conditioned": " (augmented prompt)",
    "best": " (best prompt)",
}

YOLOE_GEMINI_FOCUS_PROMPT_VARIANT: dict[
    YoloeGeminiFocusPrompt, ReferenceChartVariant
] = {
    "class_names": "baseline",
    "augmented_prompt": "image-conditioned",
}

YOLOE_GEMINI_FOCUS_CHART_TITLE: dict[YoloeGeminiFocusPrompt, str] = {
    "class_names": "Object Detection — mAP@50 (class names)",
    "augmented_prompt": "Object Detection — mAP@50 (augmented prompt)",
}

_FAMILY_LEADERBOARD_VARIANTS: dict[
    LeaderboardFamily, tuple[ReferenceChartVariant, ...]
] = {
    "sam3": ("baseline", "best"),
    "yoloe": ("baseline", "best"),
}

_FAMILY_REFERENCE_MODELS: dict[LeaderboardFamily, tuple[str, ...]] = {
    "sam3": ("sam3",),
    "yoloe": ("yoloe-11l-seg", "yoloe-26x-seg"),
}


@dataclass(frozen=True)
class MixedDetectionLeaderboardRow:
    """One ranked entry in the mixed VLM and reference detection leaderboard."""

    key: str
    display_name: str
    source: MixedDetectionSource
    map50: float
    map75: float
    map50_95: float
    image_count: int


@dataclass(frozen=True)
class MixedDetectionLeaderboard:
    """Mixed detection leaderboard spanning VLMs and reference model variants."""

    rows: tuple[MixedDetectionLeaderboardRow, ...]

    @property
    def map50(self) -> dict[str, float]:
        return {row.key: row.map50 for row in self.rows}

    @property
    def map75(self) -> dict[str, float]:
        return {row.key: row.map75 for row in self.rows}

    @property
    def map50_95(self) -> dict[str, float]:
        return {row.key: row.map50_95 for row in self.rows}


def _reference_leaderboard_key(reference_model: str, variant: str) -> str:
    return f"{reference_model}-{variant}"


def reference_leaderboard_display_name(
    reference_model: str,
    variant: ReferenceChartVariant,
) -> str:
    """Return the chart label for a reference model prompt variant."""
    base_name = REFERENCE_LEADERBOARD_NAMES[reference_model]
    return f"{base_name}{REFERENCE_CHART_VARIANT_SUFFIX[variant]}"


def row_chart_label(row: MixedDetectionLeaderboardRow) -> str:
    """Return a chart label with prompt-variant suffixes removed."""
    if row.source == "vlm":
        return row.display_name
    for reference_model, short_name in REFERENCE_LEADERBOARD_NAMES.items():
        if row.key.startswith(f"{reference_model}-"):
            return short_name
    return row.display_name


def chart_config_with_row_labels(
    base_config: BenchmarkConfig,
    rows: MixedDetectionLeaderboard,
) -> BenchmarkConfig:
    """Return a chart config whose row labels omit prompt-variant suffixes."""
    models = dict(base_config.models)
    for row in rows.rows:
        if row.key not in models:
            continue
        models[row.key] = replace(models[row.key], name=row_chart_label(row))
    return BenchmarkConfig(labs=base_config.labs, models=models)


def family_reference_keys(family: LeaderboardFamily) -> frozenset[str]:
    """Return synthetic leaderboard keys for one reference model family."""
    keys: list[str] = []
    for reference_model in _FAMILY_REFERENCE_MODELS[family]:
        for variant in _FAMILY_LEADERBOARD_VARIANTS[family]:
            keys.append(_reference_leaderboard_key(reference_model, variant))
    return frozenset(keys)


def leaderboard_rows_for_family(
    leaderboard: MixedDetectionLeaderboard,
    family: LeaderboardFamily,
) -> MixedDetectionLeaderboard:
    """Filter and re-rank rows for one reference family plus all VLMs."""
    reference_keys = family_reference_keys(family)
    rows = [
        row
        for row in leaderboard.rows
        if row.source == "vlm" or row.key in reference_keys
    ]
    rows.sort(key=lambda row: (-row.map50, row.key))
    return MixedDetectionLeaderboard(rows=tuple(rows))


def leaderboard_rows_for_keys(
    leaderboard: MixedDetectionLeaderboard,
    keys: frozenset[str],
) -> MixedDetectionLeaderboard:
    """Filter and re-rank rows to an explicit key set."""
    rows = [row for row in leaderboard.rows if row.key in keys]
    rows.sort(key=lambda row: (-row.map50, row.key))
    return MixedDetectionLeaderboard(rows=tuple(rows))


def yoloe_gemini_focus_keys(
    prompt: YoloeGeminiFocusPrompt,
    *,
    vlm_model: str = YOLOE_GEMINI_FOCUS_VLM,
) -> frozenset[str]:
    """Return leaderboard keys for one YOLO-E vs Gemini focus chart."""
    variant = YOLOE_GEMINI_FOCUS_PROMPT_VARIANT[prompt]
    keys = {vlm_model}
    for reference_model in _FAMILY_REFERENCE_MODELS["yoloe"]:
        keys.add(_reference_leaderboard_key(reference_model, variant))
    return frozenset(keys)


def leaderboard_rows_for_yoloe_gemini_focus(
    leaderboard: MixedDetectionLeaderboard,
    prompt: YoloeGeminiFocusPrompt,
    *,
    vlm_model: str = YOLOE_GEMINI_FOCUS_VLM,
) -> MixedDetectionLeaderboard:
    """Filter to one VLM plus YOLO-E rows for one prompt setting."""
    keys = yoloe_gemini_focus_keys(prompt, vlm_model=vlm_model)
    return leaderboard_rows_for_keys(leaderboard, keys)


def build_mixed_leaderboard_config(
    vlm_config: BenchmarkConfig,
    reference_config: ReferenceConfig,
) -> BenchmarkConfig:
    """Build a chart config covering VLMs and reference baseline/best variants.

    Args:
        vlm_config: Loaded VLM benchmark configuration.
        reference_config: Loaded reference model registry.

    Returns:
        Combined config with synthetic keys for reference leaderboard rows.
    """
    models = dict(vlm_config.models)
    zero_pricing = PricingConfig(
        input_per_million_tokens=0.0,
        output_per_million_tokens=0.0,
    )
    for reference_model_key in reference_config.models:
        reference_entry = reference_config.models[reference_model_key]
        if reference_model_key not in REFERENCE_LEADERBOARD_NAMES:
            continue
        if reference_entry.lab is None:
            raise ValueError(
                f"Reference model {reference_model_key!r} has no lab; "
                "required for mixed leaderboard rendering."
            )
        for variant in REFERENCE_CHART_VARIANT_SUFFIX:
            key = _reference_leaderboard_key(reference_model_key, variant)
            models[key] = ModelConfig(
                name=reference_leaderboard_display_name(reference_model_key, variant),
                lab=reference_entry.lab,
                routes=(RouteConfig(provider="reference"),),
                pricing=zero_pricing,
                detection_coordinate_format=reference_entry.coordinate_format,
            )
    return BenchmarkConfig(
        labs={**vlm_config.labs, **reference_config.labs},
        models=models,
    )


def _row_from_run(
    run: RunResult,
    sample_index: dict[str, DetectionSample],
    *,
    key: str,
    display_name: str,
    source: MixedDetectionSource,
) -> MixedDetectionLeaderboardRow | None:
    map_result = compute_dataset_map(run, sample_index)
    if map_result is None:
        return None
    return MixedDetectionLeaderboardRow(
        key=key,
        display_name=display_name,
        source=source,
        map50=map_result.map50,
        map75=map_result.map75,
        map50_95=map_result.map50_95,
        image_count=map_result.image_count,
    )


def build_mixed_detection_leaderboard(
    vlm_results_directory: Path,
    sample_index: dict[str, DetectionSample],
    vlm_config: BenchmarkConfig,
    reference_config: ReferenceConfig,
    *,
    repo_root: Path | None = None,
) -> MixedDetectionLeaderboard:
    """Build a mixed detection leaderboard for VLMs and reference variants.

    Args:
        vlm_results_directory: Directory containing VLM detection JSONL runs.
        sample_index: Mapping of image basename to detection sample.
        vlm_config: Loaded VLM benchmark configuration.
        reference_config: Loaded reference model registry.
        repo_root: Repository root used to resolve canonical reference paths.

    Returns:
        Leaderboard rows sorted by mAP@50 descending.
    """
    root = repo_root or Path.cwd()
    chart_config = build_mixed_leaderboard_config(vlm_config, reference_config)
    rows: list[MixedDetectionLeaderboardRow] = []

    vlm_runs = load_results_directory(
        vlm_results_directory,
        pattern="detection_*.jsonl",
    )
    latest_vlm = build_latest_runs_index(vlm_runs, vlm_config)
    for (task_name, effort, model), run in latest_vlm.items():
        if task_name != "detection" or effort != "low":
            continue
        row = _row_from_run(
            run,
            sample_index,
            key=model,
            display_name=vlm_config.models[model].name,
            source="vlm",
        )
        if row is not None:
            rows.append(row)

    for reference_model, baseline_path, image_conditioned_path in CANONICAL_BEST_PAIRS:
        baseline_run = load_results(root / baseline_path)
        image_conditioned_run = load_results(root / image_conditioned_path)
        baseline_key = _reference_leaderboard_key(reference_model, "baseline")
        image_conditioned_key = _reference_leaderboard_key(
            reference_model,
            "image-conditioned",
        )
        best_key = _reference_leaderboard_key(reference_model, "best")
        baseline_row = _row_from_run(
            baseline_run,
            sample_index,
            key=baseline_key,
            display_name=chart_config.models[baseline_key].name,
            source="reference-baseline",
        )
        if baseline_row is not None:
            rows.append(baseline_row)
        image_conditioned_row = _row_from_run(
            image_conditioned_run,
            sample_index,
            key=image_conditioned_key,
            display_name=chart_config.models[image_conditioned_key].name,
            source="reference-image-conditioned",
        )
        if image_conditioned_row is not None:
            rows.append(image_conditioned_row)
        merged_run = merge_best_prompt_run(
            baseline_run,
            image_conditioned_run,
            sample_index,
        ).merged_run
        best_row = _row_from_run(
            merged_run,
            sample_index,
            key=best_key,
            display_name=chart_config.models[best_key].name,
            source="reference-best",
        )
        if best_row is not None:
            rows.append(best_row)

    rows.sort(key=lambda row: (-row.map50, row.key))
    return MixedDetectionLeaderboard(rows=tuple(rows))


def _serialize_rows(
    rows: tuple[MixedDetectionLeaderboardRow, ...],
) -> list[dict[str, object]]:
    return [
        {
            "rank": index,
            "key": row.key,
            "display_name": row.display_name,
            "source": row.source,
            "map50": row.map50,
            "map75": row.map75,
            "map50_95": row.map50_95,
            "image_count": row.image_count,
        }
        for index, row in enumerate(rows, start=1)
    ]


def format_mixed_detection_leaderboard_markdown(
    leaderboard: MixedDetectionLeaderboard,
    *,
    title: str = "Mixed detection leaderboard",
    description: str = (
        "All VLM low-effort runs plus reference base-prompt and best-prompt rows."
    ),
) -> str:
    """Format mixed leaderboard rows as a markdown table."""
    lines = [
        f"# {title}",
        "",
        description,
        "",
        "| Rank | Model | Source | mAP@50 | mAP@75 | mAP@50:95 | Images |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(leaderboard.rows, start=1):
        lines.append(
            f"| {rank} | {row.display_name} | {row.source} | "
            f"{row.map50:.4f} | {row.map75:.4f} | {row.map50_95:.4f} | "
            f"{row.image_count} |"
        )
    return "\n".join(lines) + "\n"


def format_yoloe_gemini_focus_markdown(
    class_names_leaderboard: MixedDetectionLeaderboard,
    augmented_prompt_leaderboard: MixedDetectionLeaderboard,
    *,
    vlm_name: str,
) -> str:
    """Format YOLO-E vs Gemini focus leaderboards as mAP@50 markdown tables."""
    sections = (
        ("class names", class_names_leaderboard),
        ("augmented prompt", augmented_prompt_leaderboard),
    )
    lines = [
        f"# Mixed detection leaderboard — YOLO-E vs {vlm_name}",
        "",
        f"{vlm_name} compared with YOLOE-11l and YOLOE-26x at mAP@50.",
        "",
    ]
    for section_title, section_leaderboard in sections:
        lines.extend(
            [
                f"## {section_title.title()}",
                "",
                "| Rank | Model | mAP@50 |",
                "| ---: | --- | ---: |",
            ]
        )
        for rank, row in enumerate(section_leaderboard.rows, start=1):
            lines.append(f"| {rank} | {row_chart_label(row)} | {row.map50:.4f} |")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def mixed_detection_leaderboard_payload(
    leaderboard: MixedDetectionLeaderboard,
) -> dict[str, object]:
    """Serialize mixed leaderboard rows for JSON export."""
    families = {
        family: _serialize_rows(leaderboard_rows_for_family(leaderboard, family).rows)
        for family in LEADERBOARD_FAMILIES
    }
    for prompt in YOLOE_GEMINI_FOCUS_PROMPT_VARIANT:
        families[f"yoloe_gemini_focus_{prompt}"] = _serialize_rows(
            leaderboard_rows_for_yoloe_gemini_focus(leaderboard, prompt).rows
        )
    return {"families": families}
