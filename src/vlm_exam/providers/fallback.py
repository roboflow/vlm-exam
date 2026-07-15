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

import logging

import anthropic
import openai
from PIL import Image

from vlm_exam.providers.base import Provider, Usage

_logger = logging.getLogger(__name__)


def is_rate_limit_error(error: Exception) -> bool:
    """Report whether an exception indicates a rate-limit or quota exhaustion.

    Args:
        error: Exception raised by a provider call.

    Returns:
        ``True`` when the error is likely transient quota pressure.
    """
    if isinstance(error, (anthropic.RateLimitError, openai.RateLimitError)):
        return True

    status_code = getattr(error, "status_code", None)
    if status_code == 429:
        return True

    message = str(error).upper()
    if "429" in message and (
        "RATE" in message or "QUOTA" in message or "RESOURCE_EXHAUSTED" in message
    ):
        return True

    return False


class FallbackProvider(Provider):
    """Chains providers and fails over on rate-limit errors.

    Once a route rate-limits, later samples stick to the next route for the
    remainder of the run.
    """

    def __init__(self, model: str, providers: list[Provider]) -> None:
        if not providers:
            raise ValueError("FallbackProvider requires at least one provider.")
        self._model = model
        self._providers = providers
        self._active_index = 0

    @property
    def model(self) -> str:
        return self._model

    @property
    def active_route_index(self) -> int:
        """Index of the provider route currently handling requests."""
        return self._active_index

    def uploaded_image_size(self, image: Image.Image) -> tuple[int, int] | None:
        return self._providers[self._active_index].uploaded_image_size(image)

    def predict(
        self,
        image: Image.Image,
        prompt: str,
        effort: str,
    ) -> tuple[str, Usage]:
        while True:
            provider = self._providers[self._active_index]
            try:
                return provider.predict(image, prompt, effort)
            except Exception as error:
                if not is_rate_limit_error(error):
                    raise
                if self._active_index + 1 >= len(self._providers):
                    raise
                self._active_index += 1
                _logger.warning(
                    "Rate limited on route %d; failing over to route %d.",
                    self._active_index - 1,
                    self._active_index,
                )
