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

from vlm_exam.tasks.base import EvaluationResult, Sample, Task

_TASK_REGISTRY: dict[str, str] = {
    "vqa": "vlm_exam.tasks.vqa.VQATask",
}

__all__ = [
    "EvaluationResult",
    "Sample",
    "Task",
    "create_task",
]


def create_task(task_name: str) -> Task:
    """Create a task instance by name.

    Args:
        task_name: Key in the task registry (e.g. ``"vqa"``).

    Returns:
        A ready-to-use task instance.

    Raises:
        KeyError: If ``task_name`` is not registered.
    """
    if task_name not in _TASK_REGISTRY:
        available = ", ".join(sorted(_TASK_REGISTRY))
        raise KeyError(
            f"Unknown task {task_name!r}. Available tasks: {available}"
        )

    qualified_name = _TASK_REGISTRY[task_name]
    module_path, class_name = qualified_name.rsplit(".", 1)
    module = importlib.import_module(module_path)
    task_class = getattr(module, class_name)
    return task_class()
