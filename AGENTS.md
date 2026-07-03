# Agent Guidelines

Coding standards for AI agents working on this codebase.

## License header

Every `.py` file must begin with this Apache 2.0 header:

```python
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
```

## Type annotations

All function and method signatures must have full type annotations for every
parameter and the return type. No exceptions.

## Documentation

- Use Google-style docstrings on all **public** classes, functions, and
  constants.
- Do NOT add docstrings to private/internal symbols (prefixed with `_`).
- Do NOT add file-level module docstrings. Packages, files, functions,
  variables, and classes should be named clearly enough to be
  self-documenting.

## Comments

Do NOT write code comments unless documenting a non-obvious hack,
workaround, or performance trick (e.g. "exploiting numpy broadcast to
avoid a loop"). Never narrate what the code does.

## Naming

- Names must be short, concise, and easy to understand.
- No abbreviations. Favor clarity over brevity.
- Prefix private symbols with `_` (functions, classes, constants, methods).
- Use `snake_case` for functions, methods, and variables.
- Use `PascalCase` for classes.
- Use `UPPER_SNAKE_CASE` for module-level constants.

## Style

- No emoji in code or documentation.
- Run `ruff check` and `ruff format` before committing.
- Keep imports sorted (enforced by ruff `I` rules).
