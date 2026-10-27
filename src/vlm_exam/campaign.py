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

import argparse
import hashlib
import json
import os
import sqlite3
import threading
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from PIL import Image, ImageOps

from vlm_exam.config import ModelConfig, load_config
from vlm_exam.judge import Judge
from vlm_exam.providers import build_model_provider
from vlm_exam.providers.base import EMPTY_RESPONSE_TEXT
from vlm_exam.results import RunResult, SampleResult, save_results
from vlm_exam.tasks import create_task
from vlm_exam.tasks.base import Sample, Task
from vlm_exam.text_benchmark import inference_hash


class CampaignStore:
    """Durable inference and scoring checkpoints for a frozen campaign."""

    def __init__(self, path: Path, manifest: dict[str, Any]) -> None:
        self._lock = threading.Lock()
        self._database = sqlite3.connect(path, check_same_thread=False)
        self._database.execute("PRAGMA journal_mode=WAL")
        self._database.execute("PRAGMA synchronous=FULL")
        self._database.execute(
            "CREATE TABLE IF NOT EXISTS manifest (payload TEXT NOT NULL)"
        )
        self._database.execute(
            "CREATE TABLE IF NOT EXISTS samples "
            "(identity TEXT PRIMARY KEY, prediction TEXT, result TEXT, "
            "attempts INTEGER DEFAULT 0, error TEXT)"
        )
        self._database.execute(
            "CREATE TABLE IF NOT EXISTS charges "
            "(generation TEXT PRIMARY KEY, payload TEXT NOT NULL)"
        )
        encoded = json.dumps(manifest, sort_keys=True)
        existing = self._database.execute("SELECT payload FROM manifest").fetchone()
        if existing and existing[0] != encoded:
            raise ValueError("Frozen campaign changed; refusing incompatible resume")
        if not existing:
            self._database.execute("INSERT INTO manifest VALUES (?)", (encoded,))
        self._database.commit()
        self._spent = sum(
            json.loads(row[0]).get("reported", {}).get("cost", 0) or 0
            for row in self._database.execute("SELECT payload FROM charges")
        )

    def get(self, identity: str) -> dict[str, Any]:
        """Read a sample's saved inference, score and retry state."""
        with self._lock:
            row = self._database.execute(
                "SELECT prediction, result, attempts, error FROM samples "
                "WHERE identity=?",
                (identity,),
            ).fetchone()
        if row is None:
            return {"prediction": None, "result": None, "attempts": 0, "error": None}
        return dict(zip(("prediction", "result", "attempts", "error"), row))

    def save(self, identity: str, field: str, value: Any) -> None:
        """Commit one checkpoint before allowing subsequent paid work."""
        if field not in {"prediction", "result", "error"}:
            raise ValueError(field)
        encoded = json.dumps(value) if field != "error" else value
        with self._lock:
            self._database.execute(
                "INSERT OR IGNORE INTO samples(identity) VALUES (?)", (identity,)
            )
            self._database.execute(
                f"UPDATE samples SET {field}=? WHERE identity=?", (encoded, identity)
            )
            if field == "error":
                self._database.execute(
                    "UPDATE samples SET attempts=attempts+1 WHERE identity=?",
                    (identity,),
                )
            self._database.commit()

    def charge(self, usage: dict[str, Any]) -> None:
        """Record billed usage idempotently, including unsuccessful completions."""
        generation = usage.get("generation_id")
        if not generation:
            return
        with self._lock:
            cursor = self._database.execute(
                "INSERT OR IGNORE INTO charges VALUES (?, ?)",
                (generation, json.dumps(usage)),
            )
            self._database.commit()
            if cursor.rowcount:
                self._spent += usage.get("reported", {}).get("cost", 0) or 0

    def spent(self) -> float:
        """Sum actual reported charges for this campaign."""
        with self._lock:
            return self._spent


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _balance() -> float:
    response = httpx.get(
        "https://openrouter.ai/api/v1/credits",
        headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()["data"]
    return float(data["total_credits"] - data["total_usage"])


def _evaluate(
    identity: str,
    index: int,
    sample: Sample,
    task: Task,
    model: str,
    effort: str,
    configuration: ModelConfig,
    store: CampaignStore,
    judge: Judge,
) -> SampleResult:
    prior = store.get(identity)
    image = ImageOps.exif_transpose(Image.open(sample.image_path)).convert("RGB")
    provider = build_model_provider(model, configuration)
    size = provider.uploaded_image_size(image)
    prompt = task.build_prompt(sample, uploaded_size=size)
    if prior["prediction"]:
        prediction = json.loads(prior["prediction"])
    else:
        started = time.perf_counter()
        answer, usage, retries = provider.predict(image, prompt, effort)
        if usage.metadata.get("api_endpoint") == "https://api.openai.com/v1/responses":
            long_context = usage.input_tokens > 272000
            usage.metadata["reported"]["cost"] = (
                usage.input_tokens
                * configuration.pricing.input_per_million_tokens
                * 1.25
                * (2 if long_context else 1)
                + usage.output_tokens
                * configuration.pricing.output_per_million_tokens
                * (1.5 if long_context else 1)
            ) / 1e6
            usage.metadata["cost_source"] = "conservative_token_estimate"
            usage.metadata["cost_basis"] = (
                "All input at 30-minute cache-write rate; "
                "all output including reasoning; long-context premium when applicable"
            )
        prediction = {
            "answer": answer,
            "usage": asdict(usage),
            "retry": asdict(retries),
            "total_seconds": time.perf_counter() - started,
        }
        store.charge(usage.metadata)
    finish = prediction["usage"]["metadata"].get("finish_reason")
    if finish == "length":
        raise ValueError("Truncated completion; configuration needs review")
    if finish != "stop":
        raise ValueError(f"Incomplete provider response: finish_reason={finish!r}")
    if not prediction["answer"].strip() or prediction["answer"] == EMPTY_RESPONSE_TEXT:
        raise ValueError("Empty completion")
    if not prior["prediction"]:
        store.save(identity, "prediction", prediction)
    try:
        evaluation = task.evaluate(
            sample,
            prediction["answer"],
            judge=judge if task.requires_judge else None,
            uploaded_size=size,
        )
    finally:
        if task.requires_judge:
            store.charge(judge.usage)
    metadata = task.sample_metadata(sample)
    metadata.update(
        resolved_prompt=prompt,
        usage=prediction["usage"]["metadata"],
        attempts=prediction["retry"]["attempts"],
        retries=prediction["retry"]["attempts"] - 1,
        total_seconds=prediction["total_seconds"],
        uploaded_width=(size or image.size)[0],
        uploaded_height=(size or image.size)[1],
    )
    for field in ("match_method", "strict_correct", "judge_correct", "score"):
        value = getattr(evaluation, field)
        if value is not None:
            metadata[field] = value
    metadata.update(evaluation.details or {})
    if task.requires_judge:
        metadata["judge_usage"] = judge.usage
    result = SampleResult(
        index=index,
        image=Path(sample.image_path).name,
        expected=task.expected_text(sample),
        predicted=prediction["answer"],
        correct=evaluation.correct,
        input_tokens=prediction["usage"]["input_tokens"],
        output_tokens=prediction["usage"]["output_tokens"],
        elapsed_seconds=prediction["retry"]["inference_seconds"],
        metadata=metadata,
    )
    store.save(identity, "result", asdict(result))
    return result


def run_campaign(
    plan_path: Path, directory: Path, budget_usd: float, concurrency: int = 32
) -> None:
    """Run a frozen approved plan with one adaptive request pool and safe resume.

    Args:
        plan_path: JSON specifying models, task repeats, datasets and config.
        directory: Durable checkpoints, logs and live status directory.
        budget_usd: Explicitly approved total campaign spend, including judges.
        concurrency: Initial shared worker count, adjustable via control.json.
    """
    if concurrency < 32:
        raise ValueError("Initial shared concurrency must be at least 32")
    plan = json.loads(plan_path.read_text())
    configuration = load_config(Path(plan["config"]))
    if budget_usd <= 0:
        raise ValueError("An approved positive budget is required")
    for model in plan["models"]:
        routes = configuration.models[model].routes
        if len(routes) != 1 or routes[0].provider not in {"openrouter", "openai"}:
            raise ValueError(f"{model}: exactly one paid supported route is required")
        route = routes[0]
        if route.provider == "openai":
            if (
                route.options.get("store") is not False
                or route.options.get("service_tier") != "default"
                or not isinstance(route.options.get("max_output_tokens"), int)
            ):
                raise ValueError(
                    f"{model}: explicit direct OpenAI limits and standard tier required"
                )
            continue
        policy = route.options.get("provider", {})
        if (
            len(policy.get("only", [])) != 1
            or policy.get("allow_fallbacks") is not False
            or policy.get("data_collection") != "deny"
            or policy.get("require_parameters") is not True
            or ":free" in (route.provider_model_id or model)
        ):
            raise ValueError(f"{model}: paid, pinned, no-training route is required")
        if route.options.get("reasoning") != {
            "low": {"effort": "low"},
            "high": {"effort": "high"},
        }:
            raise ValueError(f"{model}: explicit Low and High reasoning is required")
        if not isinstance(route.options.get("max_tokens"), int):
            raise ValueError(f"{model}: an explicit output limit is required")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "dashboard.html").write_text(
        (Path(__file__).parent / "templates" / "campaign.html").read_text()
    )
    tasks: dict[tuple[str, str], Task] = {}
    samples: dict[str, list[Sample]] = {}
    jobs: list[tuple[str, str, str, int, int]] = []
    dataset_signatures = {}
    for name, repeats in plan["tasks"].items():
        dataset = Path(plan["dataset_root"]) / name / "train"
        rows = create_task(name).load_samples(str(dataset))
        samples[name] = rows
        dataset_signatures[name] = _digest(
            {
                "annotations": [
                    (p.name, hashlib.sha256(p.read_bytes()).hexdigest())
                    for p in sorted(dataset.glob("*annotations*"))
                ],
                "images": [
                    (p.name, hashlib.sha256(p.read_bytes()).hexdigest())
                    for p in sorted({Path(s.image_path) for s in rows})
                ],
            }
        )
        for model in plan["models"]:
            args = {}
            if name in {"text", "detection"}:
                args["coordinate_format"] = configuration.models[
                    model
                ].detection_coordinate_format
            if name == "text":
                args["inference_hash"] = inference_hash(configuration.models[model])
            task = create_task(name, **args)
            task.load_samples(str(dataset))
            tasks[(model, name)] = task
        for index in range(len(rows)):
            for effort in ("low", "high"):
                for repeat in range(repeats):
                    for model in plan["models"]:
                        jobs.append((model, name, effort, repeat, index))
    manifest = {
        "plan": plan,
        "datasets": dataset_signatures,
        "models": {m: asdict(configuration.models[m]) for m in plan["models"]},
        "code": {
            str(p.relative_to(Path(__file__).parent)): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in sorted(Path(__file__).parent.rglob("*.py"))
        },
    }
    store = CampaignStore(directory / "checkpoints.sqlite", manifest)
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    judge = Judge(openrouter=True, cache_path=directory / "judge-cache.json")
    pending = [(job, ":".join(map(str, job))) for job in jobs]
    pending = [
        (job, identity)
        for job, identity in pending
        if not store.get(identity)["result"]
    ]
    completed_by_model = Counter(job[0] for job in jobs)
    completed_by_model.subtract(job[0] for job, _ in pending)
    completed = len(jobs) - len(pending)
    limit = concurrency
    maximum = max(128, concurrency)
    status = "running"
    errors: list[dict[str, Any]] = []
    balance = _balance()
    checked = time.monotonic()
    route_limits: dict[str, int] = {}
    route_cooldowns: dict[str, float] = {}
    route_recovery: dict[str, float] = {}
    review_samples: set[str] = set()

    def _groups(job: tuple[str, str, str, int, int]) -> list[str]:
        model, name, *_ = job
        configured = configuration.models[model].routes[0]
        route = (
            "openai-direct"
            if configured.provider == "openai"
            else configured.options["provider"]["only"][0]
        )
        return [route] + (["judge"] if tasks[(model, name)].requires_judge else [])

    reserved = 0.0
    control_path = directory / "control.json"
    if not control_path.exists():
        control_path.write_text(
            json.dumps({"concurrency": concurrency, "pause": False})
        )
    input_counts: dict[str, Any] = {}
    count_path = json.loads(control_path.read_text()).get("input_token_counts_file")
    if count_path:
        counted = json.loads(Path(count_path).read_text())
        if (
            counted["datasets"] != manifest["datasets"]
            or counted["models"] != json.loads(json.dumps(manifest["models"]))
            or {k: v for k, v in counted["code"].items() if k != "campaign.py"}
            != {k: v for k, v in manifest["code"].items() if k != "campaign.py"}
        ):
            raise ValueError("Input-token counts do not match the frozen requests")
        input_counts = counted["counts"]
    with ThreadPoolExecutor(max_workers=maximum) as pool:
        active: dict[Any, tuple[Any, str, float]] = {}
        while pending or active:
            control = json.loads(control_path.read_text())
            deferred = set(control.get("deferred_samples", [])) | review_samples
            for route, initial_limit in control.get("initial_route_limits", {}).items():
                route_limits.setdefault(route, max(1, int(initial_limit)))
                route_recovery.setdefault(route, time.monotonic() + 60)
            if control.get("pause"):
                status = "paused_by_control"
            requested = min(
                maximum, max(1, int(control.get("concurrency", concurrency)))
            )
            limit = requested
            now = time.monotonic()
            for route in route_limits:
                if now >= route_recovery[route]:
                    route_limits[route] = min(requested, route_limits[route] + 1)
                    route_recovery[route] = now + 30
            if time.monotonic() - checked > 30:
                try:
                    balance = _balance()
                    checked = time.monotonic()
                except httpx.HTTPError:
                    checked = time.monotonic() - 26
            while pending and len(active) < limit and status == "running":
                active_routes = Counter(
                    route for value in active.values() for route in _groups(value[0])
                )
                now = time.monotonic()
                position = next(
                    (
                        index
                        for index, (candidate, identity) in enumerate(pending)
                        if identity not in deferred
                        and all(
                            now >= route_cooldowns.get(route, 0)
                            and active_routes[route]
                            < route_limits.get(route, requested)
                            for route in _groups(candidate)
                        )
                    ),
                    None,
                )
                if position is None:
                    break
                job, identity = pending[position]
                model, name, effort, repeat, index = job
                model_config = configuration.models[model]
                pricing = model_config.pricing
                options = model_config.routes[0].options
                tokens = options.get(
                    "max_output_tokens", options.get("max_tokens", 16384)
                )
                reserve = (
                    1048576 * pricing.input_per_million_tokens
                    + tokens * pricing.output_per_million_tokens
                ) / 1e6
                if model_config.routes[0].provider == "openai":
                    count_key = f"{model}:{name}:{effort}:{index}"
                    input_tokens = input_counts.get(count_key, {}).get(
                        "input_tokens", 1048576
                    )
                    if type(input_tokens) is not int or input_tokens <= 0:
                        raise ValueError("Invalid counted input-token allowance")
                    long_context = input_tokens > 272000
                    reserve = (
                        input_tokens
                        * pricing.input_per_million_tokens
                        * 1.25
                        * (2 if long_context else 1)
                        + tokens
                        * pricing.output_per_million_tokens
                        * (1.5 if long_context else 1)
                    ) / 1e6
                if tasks[(model, name)].requires_judge:
                    reserve += (1048576 * 1.5 + 65536 * 9) / 1e6
                if min(budget_usd - store.spent(), balance) - reserved < reserve:
                    if not active:
                        status = "paused_budget"
                    break
                pending.pop(position)
                reserved += reserve
                future = pool.submit(
                    _evaluate,
                    identity,
                    index,
                    samples[name][index],
                    tasks[(model, name)],
                    model,
                    effort,
                    model_config,
                    store,
                    judge,
                )
                active[future] = (job, identity, reserve)
            if not active:
                if status != "running":
                    break
                if pending and all(identity in deferred for _, identity in pending):
                    status = (
                        "paused_configuration"
                        if review_samples
                        else "paused_deferred_samples"
                    )
                    break
                time.sleep(1)
                continue
            done, _ = wait(active, timeout=1, return_when=FIRST_COMPLETED)
            for future in done:
                job, identity, reserve = active.pop(future)
                reserved -= reserve
                try:
                    future.result()
                    completed += 1
                    completed_by_model[job[0]] += 1
                except Exception as error:
                    message = str(error)
                    store.save(identity, "error", message)
                    errors.append({"sample": identity, "error": message})
                    code = getattr(error, "status_code", None)
                    if code == 402 or "insufficient_quota" in message:
                        status = "paused_insufficient_funds"
                    elif "Truncated completion" in message:
                        if store.get(identity)["attempts"] < 2:
                            pending.append((job, identity))
                        else:
                            review_samples.add(identity)
                            pending.append((job, identity))
                    else:
                        route = (
                            "judge"
                            if tasks[(job[0], job[1])].requires_judge
                            and store.get(identity)["prediction"]
                            else _groups(job)[0]
                        )
                        now = time.monotonic()
                        route_cooldowns[route] = now + (30 if code == 429 else 5)
                        if code == 429:
                            in_flight = (
                                sum(
                                    route in _groups(value[0])
                                    for value in active.values()
                                )
                                + 1
                            )
                            route_limits[route] = max(
                                1,
                                min(route_limits.get(route, requested), in_flight // 2),
                            )
                            route_recovery[route] = now + 60
                        if store.get(identity)["attempts"] < 4:
                            pending.append((job, identity))
                print(
                    json.dumps(
                        {
                            "completed": completed,
                            "total": len(jobs),
                            "sample": identity,
                            "concurrency": limit,
                            "spent_usd": store.spent(),
                            "status": status,
                        }
                    ),
                    flush=True,
                )
            payload = {
                "status": status,
                "completed": completed,
                "total": len(jobs),
                "completed_by_model": dict(completed_by_model),
                "active": len(active),
                "deferred_samples": sorted(deferred),
                "concurrency": limit,
                "route_limits": route_limits,
                "cooldown_seconds": {
                    route: max(0, round(until - time.monotonic(), 1))
                    for route, until in route_cooldowns.items()
                },
                "spent_usd": store.spent(),
                "budget_usd": budget_usd,
                "balance_usd": balance,
                "errors": errors[-20:],
                "updated": datetime.now(timezone.utc).isoformat(),
            }
            temporary = directory / "status.tmp"
            temporary.write_text(json.dumps(payload, indent=2))
            temporary.replace(directory / "status.json")
    output = Path(plan["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    for name, repeats in plan["tasks"].items():
        for model in plan["models"]:
            for effort in ("low", "high"):
                for repeat in range(repeats):
                    saved = [
                        store.get(f"{model}:{name}:{effort}:{repeat}:{i}")["result"]
                        for i in range(len(samples[name]))
                    ]
                    if all(saved):
                        timestamp = (
                            datetime.strptime(plan["timestamp"], "%Y%m%d_%H%M%S")
                            + timedelta(seconds=repeat)
                        ).strftime("%Y%m%d_%H%M%S")
                        result = RunResult(
                            model=model,
                            effort=effort,
                            task=name,
                            timestamp=timestamp,
                            samples=[SampleResult(**json.loads(row)) for row in saved],
                        )
                        save_results(
                            result,
                            output / f"{name}_{model}_{effort}_{timestamp}.jsonl",
                        )
    final_status = (
        "complete"
        if completed == len(jobs)
        else "incomplete"
        if status == "running"
        else status
    )
    (directory / "status.json").write_text(
        json.dumps(
            {
                "status": final_status,
                "completed": completed,
                "total": len(jobs),
                "completed_by_model": dict(completed_by_model),
                "updated": datetime.now(timezone.utc).isoformat(),
                "spent_usd": store.spent(),
                "budget_usd": budget_usd,
                "errors": errors[-20:],
            },
            indent=2,
        )
    )
    if completed != len(jobs):
        raise RuntimeError(
            f"Campaign incomplete ({completed}/{len(jobs)}): {final_status}; "
            "checkpoints retained"
        )


def main() -> None:
    """Execute an explicitly budgeted campaign from a frozen JSON plan."""
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--budget-usd", type=float, required=True)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--env-file", type=Path)
    args = parser.parse_args()
    load_dotenv(args.env_file)
    run_campaign(args.plan, args.directory, args.budget_usd, args.concurrency)


if __name__ == "__main__":
    main()
