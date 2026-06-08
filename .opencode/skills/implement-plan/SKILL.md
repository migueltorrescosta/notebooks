---
name: implement-plan
description: MUST be used when implementing code in the codebase based on a report in reports/. Reads the report, plans the physical model, writes tests first, and implements following project architecture and numerical guidelines. Do NOT use for writing reports or generating results figures.
---

# Purpose

Define a disciplined, repeatable process for implementing physics simulations from a written report, ensuring code quality, consistency with project architecture, numerical correctness, and proper testing before completion.

# Rules

1. Always run tests before starting any code change.
2. Read the report in `reports/` thoroughly — understand the Hilbert space, operators, and protocol before coding.
3. Read relevant existing code first — understand existing patterns in `src/`, and the target report's `local.py`.
4. Plan the physical model — document Hilbert space, basis, and operators before coding.
5. Add tests first (TDD) — unit tests co-located in `src/` subdirectories, integration/E2E tests in `tests/`, report-specific tests in the report's `test_local.py`.
6. Check `src/` for available code. Do not reimplement already existing functionality.
6. **New code goes to `local.py`** — New code should be added to `reports/YYYYMMDD/local.py`. Only promote code to `src/` when it is demonstrably reusable across multiple reports.

# Workflow

## 1. Before starting work

1. **Run checks**: Ensure nothing is broken before starting changes — tests (`uv run pytest . --testmon --quiet --tb=short`), linter (`uv run ruff check . --fix && uv run ruff format .`), and type checks (`uv run mypy .` and `uvx pyright src/ pages/`). Pre-existing bugs must be fixed before continuing.
2. **Read the report** — Open the target report in `reports/` and extract: Hilbert space dimensions, basis ordering, operator definitions, circuit protocol, measurement observable, and sensitivity formula.
3. **Clarify ambiguity**: Ask the user to clarify any unclear requirements before making any code changes.
4. **Read relevant code**: Understand existing patterns in `pages/`, `src/`, and the target report's `local.py` that match the report's requirements.
5. **Plan the physical model**: Determine the Hilbert space, basis, operators, and any new dataclasses needed.
6. **Record all relevant metrics**: results are often reused in the future. Record all inputs, outputs and relevant intermediate hidden variables needed to provide detailed information for post-analysis. 

## 2. During implementation

1. Add **tests first** (TDD approach):
   - Unit tests co-located with modules in `src/` subdirectories.
   - Integration/E2E tests in `tests/`.
   - Report-specific tests for `local.py` code in the report's own `test_local.py`.
2. **Add new code to `local.py`** — write all new report-specific simulation functions in `reports/YYYYMMDD/local.py`. Do not add them to `src/` modules unless they are needed by multiple reports.

## 3. At the end

Before considering implementation complete, verify:

0. **Update the CHANGELOG** — If this implementation completes a backlog item or adds significant infrastructure, add an entry under the appropriate weekly section using the format: `- **Title** (#YYYYMMDD) — description`. Remove the completed item from the backlog.
1. **Tests pass**: Run `uv run pytest . --testmon --quiet --tb=short`
2. **Linter passes**: Run `uv run ruff check . --fix && uv run ruff format .`
3. **Type checks pass (mypy)**: Run `uv run mypy .`
4. **Type checks pass (pyright)**: Run `uvx pyright src/ pages/`
5. **No existing function signatures changed**
6. No new code duplicates existing code from `src/` or from other `local.py` files.
7. No unused variables exist in the code, and no unused parameters exist in functions
9. No discrepancies exist between the report and the code implementation