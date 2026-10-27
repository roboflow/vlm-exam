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
import logging
import os
import tempfile
import threading
from pathlib import Path

from google import genai
from google.genai import types

_logger = logging.getLogger(__name__)

DEFAULT_JUDGE_MODEL = "gemini-3.5-flash"
"""Judge model every committed run is scored with; do not change casually."""

_JUDGE_PROMPT = (
    "You are an evaluation judge. Given a visual question, an expected answer, "
    "and a predicted answer, determine if the predicted answer is correct.\n\n"
    "Consider these as equivalent:\n"
    "- Case differences (e.g., 'RED' vs 'red')\n"
    "- Whitespace/formatting differences (e.g., '205/60 R 16' vs '205/60 R16')\n"
    "- Leading articles (e.g., 'a checkered flag' vs 'checkered flag')\n\n"
    "Consider these as NOT equivalent:\n"
    "- Different numerical values (e.g., '8' vs '18', '230' vs 'G230')\n"
    "- Truncated or partial answers (e.g., '2 0001' vs '2 000111 111112')\n"
    "- Semantically different answers (e.g., 'phone' vs 'laptop')\n"
    "{guidance}\n"
    "Question: {question}\n"
    "Expected answer: {expected}\n"
    "Predicted answer: {predicted}\n\n"
    "Is the predicted answer correct? Reply with exactly YES or NO."
)

_DEFAULT_CACHE_PATH = Path(".vlm_exam_judge_cache.json")


def _cache_key(
    question: str, expected: str, predicted: str, model: str, guidance: str
) -> str:
    payload = json.dumps(
        {
            "question": question,
            "expected": expected,
            "predicted": predicted,
            "model": model,
            "guidance": guidance,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


class Judge:
    """LLM-as-judge for evaluating answer equivalence.

    Args:
        model: Gemini model identifier; defaults to ``DEFAULT_JUDGE_MODEL``.
        api_key: Optional Google API key. Falls back to the
            ``GOOGLE_API_KEY`` environment variable.
        cache_path: Path to the JSON cache file for storing verdicts.
    """

    def __init__(
        self,
        model: str = DEFAULT_JUDGE_MODEL,
        api_key: str | None = None,
        cache_path: Path | None = None,
    ) -> None:
        self._model = model
        self._client = genai.Client(api_key=api_key)
        self._cache_path = cache_path or _DEFAULT_CACHE_PATH
        self._cache_lock = threading.Lock()
        self._cache: dict[str, bool] = self._load_cache()

    @property
    def model(self) -> str:
        """Judge model identifier."""
        return self._model

    def _load_cache(self) -> dict[str, bool]:
        if self._cache_path.exists():
            try:
                with open(self._cache_path) as file:
                    loaded = json.load(file)
            except json.JSONDecodeError:
                loaded = None
            if isinstance(loaded, dict):
                return loaded
            _logger.warning(
                "Judge cache %s is corrupt; starting with an empty cache.",
                self._cache_path,
            )
        return {}

    def _save_cache(self) -> None:
        # Must be called holding _cache_lock. Re-reading and merging the
        # on-disk cache first keeps verdicts written by parallel benchmark
        # processes sharing the cache file; the atomic rename keeps readers
        # from ever observing partial JSON.
        merged = self._load_cache()
        merged.update(self._cache)
        self._cache = merged
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temp_path = tempfile.mkstemp(
            dir=str(self._cache_path.parent),
            prefix=self._cache_path.name,
            suffix=".tmp",
        )
        try:
            with os.fdopen(descriptor, "w") as file:
                json.dump(self._cache, file, indent=2)
            os.replace(temp_path, self._cache_path)
        except BaseException:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise

    def evaluate(
        self, *, question: str, expected: str, predicted: str, guidance: str = ""
    ) -> bool:
        """Judge whether a predicted answer is equivalent to the expected one.

        Args:
            question: The original question for context.
            expected: Ground-truth answer.
            predicted: Model-produced answer.
            guidance: Optional task-specific instructions appended to the
                judge prompt.

        Returns:
            ``True`` if the judge deems the answers equivalent.
        """
        key = _cache_key(question, expected, predicted, self._model, guidance)

        with self._cache_lock:
            if key in self._cache:
                return self._cache[key]

        guidance_block = f"\nTask-specific guidance:\n{guidance}\n" if guidance else ""
        prompt = _JUDGE_PROMPT.format(
            question=question,
            expected=expected,
            predicted=predicted,
            guidance=guidance_block,
        )

        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=[prompt],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                ),
            )
            text = response.text
        except Exception:
            _logger.warning(
                "Judge API call failed for question=%r; defaulting to incorrect.",
                question,
            )
            return False

        if text is None:
            _logger.warning(
                "Judge returned empty response for question=%r; "
                "defaulting to incorrect.",
                question,
            )
            return False

        cleaned = text.strip().upper()
        if cleaned not in ("YES", "NO"):
            _logger.warning(
                "Judge returned unexpected response %r for question=%r; "
                "defaulting to incorrect.",
                text.strip(),
                question,
            )
            return False

        verdict = cleaned == "YES"
        with self._cache_lock:
            self._cache[key] = verdict
            self._save_cache()
        return verdict
