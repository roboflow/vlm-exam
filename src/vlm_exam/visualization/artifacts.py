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

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from PIL import Image

from vlm_exam.config import DisplayConfig


def json_digest(value: Any) -> str:
    """Fingerprint JSON data independently of indentation and key order."""
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def chart_manifest(
    directory: Path, charts: list[str], summary: dict[str, Any], config: DisplayConfig
) -> dict[str, Any]:
    """Describe chart files and the exact data, identities and renderer used.

    Args:
        directory: Directory holding the rendered PNGs.
        charts: Relative PNG filenames.
        summary: Source score summary, including any rendering filters.
        config: Display configuration used by the renderer.

    Returns:
        Compact deterministic metadata without a duplicated summary.
    """
    root = Path(__file__).resolve().parents[1]
    sources = [
        "text_render.py",
        "visualization/charts.py",
        "visualization/theme.py",
        "visualization/artifacts.py",
    ]
    renderer = hashlib.sha256(
        b"".join((root / source).read_bytes() for source in sources)
    ).hexdigest()
    models = {entry["model"] for entry in summary["configurations"]}
    display = {
        "models": {key: asdict(config.models[key]) for key in sorted(models)},
        "labs": {
            key: asdict(config.labs[key])
            for key in sorted({config.models[m].lab for m in models})
        },
    }
    entries = []
    for filename in sorted(charts):
        path = directory / filename
        with Image.open(path) as image:
            size = list(image.size)
        entries.append(
            {
                "file": filename,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "size": size,
            }
        )
    return {
        "schema_version": 1,
        "summary_sha256": json_digest(summary),
        "display_sha256": json_digest(display),
        "renderer_sha256": renderer,
        "charts": entries,
    }
