---
name: implement-plan
description: MUST be used when implementing code in the codebase based on a report in reports/. Reads the report, plans the physical model, writes tests first, and implements following project architecture and numerical guidelines. Do NOT use for writing reports or generating results figures.
---

# Purpose

Define a disciplined, repeatable process for implementing physics simulations from a written report, ensuring code quality, consistency with project architecture, numerical correctness, and proper testing before completion.

# Rules

1. Always run tests before starting any code change.
2. Read the report in `reports/` thoroughly — understand the Hilbert space, operators, and protocol before coding.
3. Read relevant existing code first — understand existing patterns in `pages/`, `src/`, and the target report's `local.py`.
4. Plan the physical model — document Hilbert space, basis, and operators before coding.
5. Add tests first (TDD) — unit tests co-located in `src/` subdirectories, integration/E2E tests in `tests/`, report-specific tests in the report's `test_local.py`.
6. **New code goes to `local.py`** — All new report-specific simulation code must be added to `reports/YYYY-MM-DD/local.py`. Only promote code to `src/` when it is demonstrably reusable across multiple reports.

# Workflow

## 1. Before starting work

1. **Read the report** — Open the target report in `reports/` and extract: Hilbert space dimensions, basis ordering, operator definitions, circuit protocol, measurement observable, and sensitivity formula.
2. **Run tests**: Ensure nothing is broken before starting changes (`uv run pytest . --testmon --quiet --tb=short`).
3. **Read relevant code**: Understand existing patterns in `pages/`, `src/`, and the target report's `local.py` that match the report's requirements.
4. **Plan the physical model**: Determine the Hilbert space, basis, operators, and any new dataclasses needed.
5. **Clarify ambiguity**: Ask the user to clarify any unclear requirements before making any code changes.

## 2. During implementation

1. Add **tests first** (TDD approach):
   - Unit tests co-located with modules in `src/` subdirectories.
   - Integration/E2E tests in `tests/`.
   - Report-specific tests for `local.py` code in the report's own `test_local.py`.
2. **Add new code to `local.py`** — write all new report-specific simulation functions in `reports/YYYY-MM-DD/local.py`. Do not add them to `src/` modules unless they are needed by multiple reports.

## 3. At the end

Before considering implementation complete, verify:

1. **Tests pass**: Run `uv run pytest . --testmon --quiet --tb=short`
2. **Linter passes**: Run `uv run ruff check . --fix && uv run ruff format .`
3. **Type checks pass**: Run `uv run mypy .`
4. **No existing function signatures changed**
