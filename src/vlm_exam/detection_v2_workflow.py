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

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import click

from vlm_exam.config import BenchmarkConfig, load_config
from vlm_exam.detection_v2_identity import canonical_model_id, release_model_id


def resolve_detection_profiles(
    models: str,
    arm: str,
    policy: dict[str, Any],
    config: BenchmarkConfig,
    configuration: str | None = None,
    settings: dict[str, Any] | None = None,
) -> list[tuple[str, str | None, dict[str, Any] | None]]:
    """Resolve registry or explicit profiles for all models without side effects."""
    from vlm_exam.detection_v2_inference import current_request_profile
    from vlm_exam.detection_v2_runner import validate_request_profile

    model_ids = [name.strip() for name in models.split(",")]
    if len({canonical_model_id(m) for m in model_ids}) != len(model_ids):
        raise click.UsageError("Duplicate model identities, including aliases")
    if settings is not None and len(model_ids) != 1:
        raise click.UsageError("A single --profile can only be used with one model")
    if settings is not None and configuration is not None:
        raise click.UsageError("Choose either --profile or --configuration")
    selected = []
    for model in model_ids:
        chosen = configuration
        model_config = config.models.get(canonical_model_id(model))
        model = release_model_id(policy, model)
        model_settings = settings
        if settings is None and chosen is None and model_config is not None:
            model_settings = model_config.detection_v2
        if model_settings is None and chosen is None:
            lane = next(
                (
                    r
                    for r in policy["runs"]
                    if (r["model"], r["effort"]) == (model, arm)
                ),
                None,
            )
            if lane is None:
                raise click.UsageError(
                    f"No reviewed profile for {model}/{arm}; "
                    "add detection_v2 settings to models.yaml or supply --profile FILE"
                )
            profiles = lane["configurations"]
            keys = {
                "provider",
                "wire",
                "format",
                "minimum",
                "maximum",
                "max_output_tokens",
                "routing",
                "enabled",
            }
            effective = {
                json.dumps(
                    {k: v for k, v in policy["configurations"][h].items() if k in keys},
                    sort_keys=True,
                )
                for h in profiles
            }
            if len(effective) != 1:
                choices = "\n".join(
                    f"--configuration {h}: "
                    f"{json.dumps(policy['configurations'][h], sort_keys=True)}"
                    for h in profiles
                )
                raise click.UsageError(
                    f"{model}/{arm} has multiple native settings. "
                    f"Repeat this command with one of:\n{choices}"
                )
            chosen = profiles[0]
        if model_settings is None and not any(
            row["model"] == model
            and row["effort"] == arm
            and chosen in row["configurations"]
            for row in policy["runs"]
        ):
            raise click.UsageError(f"Invalid configuration for {model}/{arm}")
        effective_profile = (
            model_settings
            if model_settings is not None
            else current_request_profile(policy["configurations"][chosen])
        )
        if model_settings is not None:
            validate_request_profile(effective_profile, arm)
        else:
            from vlm_exam.providers.openrouter_routing import (
                validate_openrouter_routing,
            )

            validate_openrouter_routing(effective_profile)
        selected.append((model, chosen, model_settings))
    return selected


def run_detection(
    models: str,
    effort: str,
    images: Path,
    output: Path,
    concurrency: int,
    configuration: str | None = None,
    profile: Path | None = None,
    max_samples: int | None = None,
    config_path: Path | None = None,
) -> None:
    """Run the default detection protocol with explicit, auditable native settings."""
    from vlm_exam.detection_v2 import load_release
    from vlm_exam.detection_v2_data import ensure_images
    from vlm_exam.detection_v2_publication import publish_run
    from vlm_exam.detection_v2_runner import run_release

    root = Path.cwd()
    arm = {"low": "minimum", "high": "maximum"}.get(effort, effort)
    if arm not in {"minimum", "maximum", "enabled", "disabled"}:
        raise click.UsageError(
            "Detection v2 effort must be low/minimum, high/maximum, enabled or disabled"
        )
    policy, _ = load_release(root)
    settings = json.loads(profile.read_text()) if profile else None
    selected = resolve_detection_profiles(
        models, arm, policy, load_config(config_path), configuration, settings
    )
    images = ensure_images(root, images)
    for model, chosen, settings in selected:
        target = output / f"{model}__{arm}"
        kwargs: dict[str, Any] = {"profile": settings}
        if max_samples is not None:
            kwargs["max_samples"] = max_samples
        run_release(root, images, model, arm, chosen, target, concurrency, **kwargs)
        if max_samples is None:
            publish_run(root, target)
        click.echo(f"Saved detection v2 {model}/{arm}: {target}")


def with_detection_results(
    summary: Any,
    results: Path,
    config: Any,
    effort: str | None,
    models: set[str] | None,
    all_groups: dict | None = None,
) -> Any:
    """Merge v2 into the established frontend schema, never mixing legacy results."""
    from vlm_exam.summary import (
        OVERVIEW_TASKS,
        CostSummary,
        MetricDefinition,
        MetricValue,
        ModelProtocolSummary,
        ModelSummary,
        ModelTaskResult,
        SpeedSummary,
        TaskSummary,
        TokenSummary,
        _overall,
    )
    from vlm_exam.summary_metadata import detection_configurations

    root = results.resolve().parent
    source = root / "datasets/detection/v2/summary.json"
    if not source.exists():
        return summary
    data = json.loads(source.read_text())
    policy = json.loads((root / "datasets/detection/v2/release.json").read_text())
    if data["dataset_sha256"] != policy["dataset_sha256"]:
        raise ValueError(
            "Detection v2 summary has stale ground truths; run detection-v2 summary"
        )
    labels = json.loads((root / "datasets/detection/v2/model_labels.json").read_text())
    if models is not None:
        models = {canonical_model_id(m) for m in models}
    indexed = {(m.key, m.effort): m for m in summary.models}
    for row in data["models"]:
        model = canonical_model_id(row["model"])
        legacy_effort = {"minimum": "low", "maximum": "high"}.get(
            row["effort"], row["effort"]
        )
        if (models is not None and model not in models) or (
            effort is not None and effort not in {legacy_effort, row["effort"]}
        ):
            continue
        metrics = {
            k: row["metrics"][k] * 100
            for k in ("map50", "f1", "precision", "recall")
            if row["metrics"].get(k) is not None
        }
        n = row["images"]
        ti, to = (
            row["metrics"].get("input_tokens") or 0,
            row["metrics"].get("output_tokens") or 0,
        )
        cfg = config.models.get(model)
        cost = (
            (
                ti * cfg.pricing.input_per_million_tokens
                + to * cfg.pricing.output_per_million_tokens
            )
            / 1e6
            if cfg
            else None
        )
        seconds = row["metrics"].get("seconds_per_image")
        task = ModelTaskResult(
            primary_metric=MetricValue("map50", metrics["map50"])
            if "map50" in metrics
            else None,
            metrics=metrics,
            metric_runs={k: (v,) for k, v in metrics.items()},
            run_count=1,
            sample_count=n,
            failed_sample_count=sum(
                v for k, v in row["response_counts"].items() if k != "returned"
            ),
            tokens=TokenSummary(ti, to, ti + to, (ti + to) / n),
            cost=CostSummary(cost, cost / n if cost is not None else None),
            speed=SpeedSummary(seconds * n if seconds is not None else None, seconds),
            timestamp="",
            timestamps=(),
            evaluated_sample_count=n - row["response_counts"].get("missing", 0),
            provenance={
                "protocol": "detection-v2",
                "dataset_sha256": data["dataset_sha256"],
                "native_effort": row["effort"],
                "configurations": row["configurations"],
                "complete": row["complete"],
                "accepted_failure_count": row.get("accepted_failure_count", 0),
            },
            configuration_details=detection_configurations(
                root, policy, row["model"], row["effort"]
            ),
            subsets=row["subsets"],
        )
        existing = indexed.get((model, legacy_effort))
        tasks = {**(existing.tasks if existing else {}), "detection": task}
        if existing:
            indexed[(model, legacy_effort)] = replace(
                existing,
                tasks=tasks,
                overall=_overall(
                    {k: v for k, v in tasks.items() if k in OVERVIEW_TASKS}
                ),
            )
        else:
            label = labels.get(model, {"name": model, "lab": "unknown"})
            indexed[(model, legacy_effort)] = ModelSummary(
                f"{model}:{legacy_effort}",
                model,
                cfg.name if cfg else label["name"],
                cfg.lab if cfg else label["lab"],
                legacy_effort,
                tasks,
                _overall({k: v for k, v in tasks.items() if k in OVERVIEW_TASKS}),
                ModelProtocolSummary(
                    "detection-v2",
                    "complete" if row["complete"] else "incomplete",
                    1,
                    1,
                ),
            )
    task_defs = [t for t in summary.tasks if t.key != "detection"]
    task_defs.append(
        TaskSummary(
            "detection",
            "Detection v2",
            "map50",
            tuple(
                MetricDefinition(k, label)
                for k, label in (
                    ("map50", "mAP@50"),
                    ("f1", "F1@50"),
                    ("precision", "Precision@50"),
                    ("recall", "Recall@50"),
                )
            ),
        )
    )
    protocol = replace(summary.protocol, detection_repeats=1)
    updated = []
    for model in indexed.values():
        counts = []
        for task, required in protocol.repeats_by_task.items():
            arms = protocol.efforts
            if task == "detection":
                recorded = {
                    r["effort"]
                    for r in data["models"]
                    if canonical_model_id(r["model"]) == model.key
                }
                if recorded == {"enabled"}:
                    arms = ("enabled",)
            for arm in arms:
                if task == "detection":
                    native = {"low": "minimum", "high": "maximum"}.get(arm, arm)
                    lane = next(
                        (
                            r
                            for r in data["models"]
                            if canonical_model_id(r["model"]) == model.key
                            and r["effort"] == native
                        ),
                        None,
                    )
                    count = 1 if lane and lane["complete"] else 0
                else:
                    count = len((all_groups or {}).get((task, arm, model.key), ()))
                counts.append((count, required))
        complete = all(count == required for count, required in counts)
        status = (
            "complete"
            if complete
            else "legacy"
            if model.protocol.status == "legacy"
            else "incomplete"
        )
        updated.append(
            replace(
                model,
                protocol=ModelProtocolSummary(
                    model.protocol.name,
                    status,
                    sum(count for count, _ in counts),
                    sum(required for _, required in counts),
                ),
            )
        )
    return replace(
        summary,
        models=updated,
        tasks=task_defs,
        protocol=protocol,
        efforts=tuple(sorted({m.effort for m in indexed.values()})),
    )


def detection_charts(root: Path, output: Path, models: set[str] | None = None) -> None:
    """Render the default detection charts from the approved v2 summary."""
    import matplotlib.pyplot as plt

    from vlm_exam.config import load_display_config
    from vlm_exam.visualization.charts import (
        plot_accuracy_chart,
        save_leaderboard_chart,
    )

    data = json.loads((root / "datasets/detection/v2/summary.json").read_text())
    display = load_display_config(
        model_labels=root / "datasets/detection/v2/model_labels.json"
    )
    output.mkdir(parents=True, exist_ok=True)
    if models is not None:
        models = {canonical_model_id(m) for m in models}
    for arm, effort in (("minimum", "low"), ("maximum", "high")):
        scores = {
            r["model"]: r["metrics"]["map50"] * 100
            for r in data["models"]
            if r["effort"] == arm
            and r["complete"]
            and (models is None or canonical_model_id(r["model"]) in models)
        }
        if not scores:
            continue
        figure = plot_accuracy_chart(
            scores, display, f"RF-ODv2 mAP@50 · {effort.title()}"
        )
        save_leaderboard_chart(figure, output / f"detection_map50_{effort}.png")
        plt.close(figure)


def detection_report(root: Path) -> None:
    """Print the current detection summary with native effort and coverage."""
    source = root / "datasets/detection/v2/summary.json"
    if not source.exists():
        return
    data = json.loads(source.read_text())
    click.echo("\nRF-ODv2 — subset-macro mAP@50; micro F1@50")
    click.echo(f"{'Model':<30} {'Effort':<9} {'mAP@50':>8} {'F1@50':>8} {'Missing':>8}")
    for row in data["models"]:

        def metric(key: str) -> str:
            value = row["metrics"].get(key)
            return f"{value * 100:.2f}%" if value is not None else "N/A"

        click.echo(
            f"{row['model']:<30} {row['effort']:<9} "
            f"{metric('map50'):>8} {metric('f1'):>8} "
            f"{row['response_counts'].get('missing', 0):>8}"
        )


def visualize_detection(
    root: Path,
    archive: Path,
    images: Path,
    output: Path,
    max_images: int,
    label_mode: str,
    output_format: str,
    image: str | None,
    index: int | None,
) -> None:
    """Render canonical ground truths and frozen predictions using shared cards."""
    import cv2
    import matplotlib.pyplot as plt

    from vlm_exam.config import load_display_config
    from vlm_exam.detection_v2 import (
        _detections,
        load_release,
        read_records,
        score_subset,
    )
    from vlm_exam.detection_v2_data import ensure_images
    from vlm_exam.visualization.detection import (
        plot_detection_card,
        save_annotated_detection,
    )

    _, cases = load_release(root)
    by_id = {c["id"]: c for c in cases}
    records = read_records(archive)
    images = ensure_images(root, images)
    display = load_display_config(
        model_labels=root / "datasets/detection/v2/model_labels.json"
    )
    output.mkdir(parents=True, exist_ok=True)
    count = 0
    for number, row in enumerate(records):
        case = by_id[row["case"]]
        if count >= max_images:
            break
        if index is not None and index != number:
            continue
        if image is not None and image not in {case["id"], case["image"]}:
            continue
        labels = sorted({o["label"] for o in case["objects"] + row["detections"]})
        truth = _detections(case["objects"], labels)
        predicted = _detections(row["detections"], labels)
        pixels = cv2.imread(str(images / case["image"]))
        if pixels is None:
            raise ValueError(f"Cannot decode image {case['id']}")
        path = output / f"{number:03d}_{case['id']}.png"
        pred_labels = [o["label"] for o in row["detections"]]
        if output_format == "plain":
            save_annotated_detection(
                pixels, predicted, pred_labels, path, label_mode=label_mode
            )
        else:
            figure = plot_detection_card(
                image=pixels,
                ground_truth=truth,
                predictions=predicted,
                gt_labels=[o["label"] for o in case["objects"]],
                pred_labels=pred_labels,
                model_id=row["model"],
                config=display,
                map_score=score_subset([case], [row])["map50"]
                if row["status"] != "missing"
                else None,
                label_mode=label_mode,
            )
            figure.savefig(path, dpi=150, facecolor=figure.get_facecolor())
            plt.close(figure)
        count += 1
    click.echo(f"Saved {count} detection v2 images to {output}")
