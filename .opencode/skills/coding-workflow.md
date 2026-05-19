---
name: coding-workflow
description: MUST be used when writing, modifying, or debugging Python code. Covers the full before/during/after lifecycle including testing, linting, type checking, and revision card generation.
---

# Purpose

Define a disciplined, repeatable process for all code changes, ensuring quality, consistency, and proper documentation before completing any task.

# Rules

1. Always run tests before starting any code change.
2. Read relevant code first — understand existing patterns in `pages/` and `src/`.
3. Plan the physical model — document Hilbert space, basis, and operators before coding.
4. Clarify ambiguity — ask the user before making assumptions.
5. Add tests first (TDD) — unit tests in `src/`, integration/E2E tests in `tests/`.
6. Match existing patterns in the codebase.
7. Choose the simpler approach; avoid large refactors or unrelated changes.
8. Use **type hints** for all function signatures.
9. Never make public interface changes — preserve existing function signatures.

# Workflow

## 1. Before starting work

1. **Run tests**: Ensure nothing is broken before starting changes (`uv run pytest . --testmon --quiet --tb=short`)
2. **Read relevant code**: Understand existing patterns in `pages/` and `src/`
3. **Plan the physical model**: Document the Hilbert space, basis, and operators to use
4. **Clarify ambiguity**: Ask the user to clarify any unclear requirements before making any code changes

## 2. During implementation

1. Add **tests first** (TDD approach) — unit tests co-located with modules in `src/`, integration/E2E tests in `tests/`
2. Match existing patterns in the codebase
3. Choose the simpler approach; avoid large refactors or unrelated changes
4. Use **type hints** for all function signatures

## 3. At the end

Before considering a task complete, verify:

1. **Tests pass**: Run `uv run pytest . --testmon --quiet --tb=short`
2. **Linter passes**: Run `uv run ruff check . --fix && uv run ruff format .`
3. **Type checks pass**: Run `uv run mypy .`
4. **Conclusions**: If a new experiment was run based on a report, add a "Conclusions" section to the report detailing the results
5. **Generate revision cards**: If the task produced new physics concepts, key findings, or clarified definitions, create GoCard-format Markdown cards in `revise/`. The ideal moment is right after a report's Conclusions are finalised. Card-worthy content includes new operators, Hilbert space conventions, scaling exponents, sensitivity formulas, noise channel parameters, or experimental verdicts. Follow the GoCard format (YAML frontmatter with `tags`, `created`, `last_reviewed`, `review_interval`, `difficulty`; body with `# Title`, `## Question`, `## Answer`). The **Answer must be at most 5 words** — force yourself to distil each concept to its essence. Place all cards directly in `revise/` (no subdirectories). Do not create cards that merely restate trivial or obvious information — each card must test a specific retrievable concept that aids long-term memory.

# Verification

- [ ] Tests passed before and after changes
- [ ] Linter passes (`ruff check` + `ruff format`)
- [ ] Type checks pass (`mypy`)
- [ ] No existing function signatures changed
- [ ] All new functions have type hints
- [ ] Tests exist for new functionality
- [ ] Revision cards created for new physics concepts (if applicable)
- [ ] Report Conclusions section updated (if experiment was run)

# Anti-patterns

- Starting implementation without running tests first.
- Skipping the "read relevant code" step and duplicating existing patterns.
- Making large refactors unrelated to the task.
- Changing function signatures that other code depends on.
- Forgetting to update the report's Conclusions section after running experiments.
- Creating revision cards for trivial or obvious information.
- Creating revision cards in subdirectories of `revise/`.
- Giving revision card answers longer than 5 words.
