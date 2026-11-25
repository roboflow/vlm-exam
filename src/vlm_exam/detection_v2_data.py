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

import fcntl
import hashlib
import io
import json
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from PIL import Image, ImageOps


def _get(url: str, parameters: dict[str, str] | None = None) -> requests.Response:
    for attempt in range(4):
        try:
            response = requests.get(
                url, params=parameters, timeout=(15, 120), allow_redirects=False
            )
            if response.status_code == 200:
                return response
            retryable = response.status_code == 429 or response.status_code >= 500
            if not retryable:
                raise ValueError(
                    f"Roboflow download returned HTTP {response.status_code}"
                )
        except requests.RequestException:
            pass
        if attempt < 3:
            time.sleep(2**attempt)
    raise ValueError("Roboflow download failed after four attempts; retry to resume")


def verify_image(case: dict[str, Any], path: Path, root: Path) -> bool:
    """Verify canonical bytes or an explicitly pinned pixel-identical lossless copy."""
    if not path.is_file():
        return False
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest == case["image_sha256"]:
        return True
    source = root / "datasets/detection/v2/download.json"
    if not source.exists():
        return False
    alternate = (
        json.loads(source.read_text()).get("lossless_copies", {}).get(case["id"])
    )
    if alternate is None or digest != alternate["sha256"]:
        return False
    with Image.open(path) as image:
        upright = ImageOps.exif_transpose(image).convert("RGB")
        return upright.size == (case["width"], case["height"]) and (
            hashlib.sha256(upright.tobytes()).hexdigest() == alternate["pixel_sha256"]
        )


def ensure_images(root: Path, destination: Path) -> Path:
    """Fetch hash-pinned original Roboflow images, preserving canonical task IDs.

    Ground truths are the versioned canonical manifest, not a lossy COCO export.
    Shared photographs are fetched once and reused without changing any pixels.
    A valid existing cache needs no credentials or network connection.
    """
    directory = root / "datasets/detection/v2"
    cases = json.loads((directory / "manifest.json").read_text())["cases"]
    source = json.loads((directory / "download.json").read_text())
    if set(source["images"]) != {case["id"] for case in cases}:
        raise ValueError("Download inventory does not match the canonical manifest")
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / ".download.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        missing = []
        cached = {}
        for case in cases:
            path = destination / case["image"]
            if path.parent.resolve() != destination.resolve():
                raise ValueError("Unsafe image path in manifest")
            if path.exists():
                if not verify_image(case, path, root):
                    raise ValueError(f"Modified cached image: {case['id']}")
                cached[case["image_sha256"]] = path
            else:
                missing.append(case)
        if not missing:
            return destination
        key = os.environ.get("ROBOFLOW_API_KEY")
        if not key and any(case["image_sha256"] not in cached for case in missing):
            raise ValueError(
                "Set ROBOFLOW_API_KEY to download private detection v2 images"
            )
        groups: dict[str, list[dict[str, Any]]] = {}
        for case in missing:
            groups.setdefault(case["image_sha256"], []).append(case)

        def fetch(item: tuple[str, list[dict[str, Any]]]) -> None:
            digest, group = item
            case = group[0]
            if digest in cached:
                data = cached[digest].read_bytes()
            else:
                alternate = source.get("lossless_copies", {}).get(case["id"])
                image_id = (
                    alternate["image_id"] if alternate else source["images"][case["id"]]
                )
                endpoint = (
                    f"https://api.roboflow.com/{source['workspace']}/"
                    f"{source['project']}/images/{image_id}"
                )
                try:
                    details = _get(endpoint, {"api_key": key or ""}).json()
                    url = details["image"]["urls"]["original"]
                    parsed = urlparse(url)
                    if (
                        parsed.scheme != "https"
                        or not parsed.hostname
                        or not (
                            parsed.hostname.endswith(".roboflow.com")
                            or parsed.hostname == "storage.googleapis.com"
                        )
                    ):
                        raise ValueError("Unexpected original image download host")
                    data = _get(url).content
                    if alternate and alternate.get("remove_right_columns"):
                        if (
                            hashlib.sha256(data).hexdigest()
                            != alternate["download_sha256"]
                        ):
                            raise ValueError(
                                f"Modified lossless transport image: {case['id']}"
                            )
                        with Image.open(io.BytesIO(data)) as image:
                            upright = ImageOps.exif_transpose(image).convert("RGB")
                            columns = alternate["remove_right_columns"]
                            if upright.size != (
                                case["width"] + columns,
                                case["height"],
                            ):
                                raise ValueError(
                                    "Unexpected lossless transport dimensions"
                                )
                            upright = upright.crop(
                                (0, 0, case["width"], case["height"])
                            )
                            buffer = io.BytesIO()
                            upright.save(buffer, format="PNG")
                            data = buffer.getvalue()
                except (KeyError, TypeError, requests.exceptions.JSONDecodeError):
                    raise ValueError(
                        f"Invalid Roboflow image metadata: {case['id']}"
                    ) from None
            for entry in group:
                target = destination / entry["image"]
                with tempfile.NamedTemporaryFile(
                    dir=destination, prefix=".download-", suffix=".part", delete=False
                ) as staged:
                    temporary = Path(staged.name)
                    staged.write(data)
                try:
                    if not verify_image(entry, temporary, root):
                        raise ValueError(
                            "Roboflow original differs from canonical image: "
                            f"{entry['id']}"
                        )
                    temporary.replace(target)
                finally:
                    temporary.unlink(missing_ok=True)

        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(fetch, groups.items()))
    return destination
