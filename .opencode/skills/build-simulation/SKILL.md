---
name: build-simulation
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

Before considering implementation complete, verify all items in the Workflow Verification section below.

# Workflow Verification

### Before implementation
- [ ] Searched agentmemory for relevant prior implementations and decisions (`project:notebooks`)
- [ ] Read the report (Hilbert space, operators, protocol)
- [ ] Read relevant existing code in `src/` and the report's `local.py` for patterns
- [ ] Consulted CHANGELOG backlog before starting
- [ ] Tests pass before changes
- [ ] Challenged assumptions and ambiguities clarified

### After implementation
- [ ] Followed YAGNI/KISS — no speculative abstractions
- [ ] Tests pass after changes (`uv run pytest . --testmon --quiet --tb=short`)
- [ ] Linting and formatting pass (`uv run ruff check . --fix && uv run ruff format .`)
- [ ] Type checks pass — mypy (`uv run mypy .`) and pyright (`uvx pyright src/ pages/`)
- [ ] No existing function signatures changed
- [ ] No new code duplicates existing code from `src/` or other `local.py` files
- [ ] No unused variables or parameters
- [ ] No discrepancies between the report and the code implementation
- [ ] Trial simulation run completes and validates output format
- [ ] CHANGELOG updated with entry under the appropriate weekly section; backlog entry removed if task came from backlog; any errors that predated the current session added to the backlog
- [ ] Saved key decisions to agentmemory (`project:notebooks`), consolidated if more than 5 saves, and reflected