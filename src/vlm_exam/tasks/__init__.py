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

import importlib
from typing import Any

from vlm_exam.tasks.base import EvaluationResult, Sample, Task

_TASK_REGISTRY: dict[str, str] = {
    "ocr": "vlm_exam.tasks.qa.OCRTask",
    "extraction": "vlm_exam.tasks.qa.ExtractionTask",
    "counting": "vlm_exam.tasks.qa.CountingTask",
    "identification": "vlm_exam.tasks.qa.IdentificationTask",
    "reasoning": "vlm_exam.tasks.qa.ReasoningTask",
    "detection": "vlm_exam.tasks.detection.DetectionTask",
}

QA_TASK_NAMES: tuple[str, ...] = (
    "ocr",
    "extraction",
    "counting",
    "identification",
    "reasoning",
)
"""Names of the question-answering tasks sharing the JSONL format."""

__all__ = [
    "QA_TASK_NAMES",
    "EvaluationResult",
    "Sample",
    "Task",
    "create_task",
]


def create_task(task_name: str, **task_args: Any) -> Task:
    """Create a task instance by name.

    Args:
        task_name: Key in the task registry (e.g. ``"ocr"``).
        **task_args: Keyword arguments forwarded to the task constructor.

    Returns:
        A ready-to-use task instance.

    Raises:
        KeyError: If ``task_name`` is not registered.
    """
    if task_name not in _TASK_REGISTRY:
        available = ", ".join(sorted(_TASK_REGISTRY))
        raise KeyError(f"Unknown task {task_name!r}. Available tasks: {available}")

    qualified_name = _TASK_REGISTRY[task_name]
    module_path, class_name = qualified_name.rsplit(".", 1)
    module = importlib.import_module(module_path)
    task_class = getattr(module, class_name)
    return task_class(**task_args)
