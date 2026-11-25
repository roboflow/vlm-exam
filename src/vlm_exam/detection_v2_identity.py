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

"""Public model aliases without rewriting historical result identities."""

from typing import Any


def canonical_model_id(model: str) -> str:
    """Return the public registry key for a measured model identity."""
    return {"gemini-4-hydrogen-eap": "gemini-4-hydrogen"}.get(model, model)


def release_model_id(policy: dict[str, Any], model: str) -> str:
    """Reuse the existing archive identity; reject competing alias inventories."""
    matches = {
        row["model"]
        for row in policy["runs"]
        if canonical_model_id(row["model"]) == canonical_model_id(model)
    }
    if len(matches) > 1:
        raise ValueError(
            f"Conflicting release identities for {canonical_model_id(model)}"
        )
    return next(iter(matches)) if matches else canonical_model_id(model)
