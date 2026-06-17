---
name: tackle-backlog
description: MUST be used when handling infrastructure/non-report tasks from the CHANGELOG backlog (items under `# Backlog > Infrastructure & Tooling` or any non-report work). Reads the backlog item, plans and implements the fix with user approval, and verifies correctness. Do NOT use for report-specific simulation work, which belongs to build-simulation.
---

# Purpose

Handle a single infrastructure or non-report task from the `# Backlog` in `CHANGELOG.md`. These are **do-and-verify** tasks (config changes, test infrastructure, lint rules, shared-module promotions, cyclomatic-complexity refactors, CI/CD setup) that the other skills do not cover.

The skill proceeds in four phases:

1. **Preparation** — Understand the task, its dependencies, and the current state of the codebase.
2. **Planning** — Propose a specific implementation plan and get user approval. Do not write code before approval.
3. **Implementation** — Execute the approved plan, updating code, tests, and configuration as needed.
4. **Wrap-up** — Verify correctness, record lessons, and close the backlog item.

# Rules

1. **User approval required before any code change** — Propose a plan first. Do not modify files until the user explicitly approves.
2. **Non-report only** — This skill is for infrastructure, tooling, config, test-infrastructure, and shared-module-promotion tasks. For report-specific simulation code, use `build-simulation`. For report-specific result generation, use `generate-results`.
3. **No duplication** — Do not re-implement functionality that already exists in `src/` modules or that belongs to another skill's domain.
4. **Fail-fast on pre-existing breakage** — Before implementing, run the test suite and report any pre-existing failures. Do not proceed if the repo is already broken.
5. **Respect the dependency chain** — If the backlog item has explicit dependencies (e.g., "Blocked on:" or a sequence of prerequisite items), flag them before proceeding. Do not implement an item whose dependencies are not yet done.
6. **CHANGELOG conventions** — Follow the same conventions as every other skill:
   - Move the completed item from `# Backlog` to the appropriate weekly `### Infrastructure` section.
   - Use the entry format `- **Title** (#YYYYMMDD) — description` for report-related infra, or `- **Title** — description` for pure-tooling items.
   - If the task was not in the backlog, add an entry under the weekly section only.
7. **Idempotent per-item** — If the work was already completed (item already moved out of backlog), no-op.

# Workflow

## 1. Preparation

1. **Read the specific backlog item** — Open `CHANGELOG.md` and locate the item in the `# Backlog` section. Note:
   - Its theme group (`Infrastructure & Tooling`, `Partially Completed`, or one of the per-theme groups).
   - Its priority emoji (🔴🟠🟡🟢).
   - Any inline `**Verify**:` instructions or expected outputs.
   - Any explicit dependencies or blocking annotations (e.g., "Blocked on:", "See #YYYYMMDD").
2. **Check for dependencies** — If the item mentions a dependency or prerequisite (another backlog item, a specific task, or a `See #YYYYMMDD` reference to an incomplete report), flag it to the user and ask whether they want to defer, tackle the dependency first, or proceed anyway.
3. **Search agentmemory** — Call `memory_recall` or `memory_smart_search` with keywords from the backlog item to find prior context, decisions, or related work.
4. **Read relevant files** — Based on the backlog item, read:
   - Configuration files that may need changes (`pyproject.toml`, `conftest.py`, `opencode.json`, `.streamlit/config.toml`).
   - Source files or test files referenced in the item's description.
   - Any `Verify:` script or test file mentioned inline.
   - Related skill files (`.opencode/skills/*/SKILL.md`) if the task might affect them.
   - The agent definition (`.opencode/agents/research-assistant.md`) if the task adds a new skill or changes the toolchain.
5. **Run current test suite** — Execute `uv run pytest . --testmon --quiet --tb=short` to capture the current state. If any tests fail pre-existing:
   - Report the failures to the user.
   - Add each failure as a new backlog item under `# Backlog > Infrastructure & Tooling` (or appropriate theme group) with 🟢 priority and description referencing the test name and error.
   - Do not proceed until the user confirms they want to proceed despite the pre-existing failures.

## 2. Planning

1. **Synthesize a plan** — Based on the preparation step, write a specific, step-by-step implementation plan. For each step, note:
   - Which files to modify or create.
   - What the change does.
   - How to verify the change (test commands, manual checks).
2. **Flag cross-cutting concerns** — If the change affects:
   - **Other skills** — note which `SKILL.md` files may need updating.
   - **The agent config** — note whether `opencode.json` or `.opencode/agents/research-assistant.md` needs updating.
   - **Backward compatibility** — note whether existing reports, pages, or tests might be affected.
   - **Dependency chain** — if completing this item unblocks or creates new backlog items, note them.
3. **Present the plan to the user** — Use a structured format:
   ```
   ## Plan: {Title of Backlog Item}
   
   ### Steps
   1. {Step 1 description} → {files affected} → {verification command}
   2. ...
   
   ### Cross-Cutting Concerns
   - Skills affected: ...
   - Config affected: ...
   - Backward compatibility: ...
   - Dependencies/blockers: ...
   ```
4. **Wait for approval** — Do not modify any files until the user explicitly approves the plan.

## 3. Implementation

1. **Implement each step** one at a time, verifying as you go:
   - For config changes: run the relevant tool to verify the config is valid (e.g., `uv run ruff check .` after adding a ruff rule).
   - For test changes: run the specific test file to verify the new test passes (`uv run pytest <path> -q --tb=short`).
   - For shared-module promotions: verify no existing imports break.
2. **Update the agent config if needed** — If the task adds a new skill, updates the toolchain, or changes how the agent operates, update `.opencode/agents/research-assistant.md` and/or `opencode.json` as needed.
3. **Update skills if needed** — If the task changes project conventions (new lint rules, new test patterns, new config), update the relevant `SKILL.md` files under `.opencode/skills/` to document the new expectations.
4. **Remove the item from the backlog** — After implementation, move the backlog item to the current weekly `### Infrastructure` section following the established format. If the task was not in the backlog, just add an entry under the current weekly section.

## 4. Wrap-up

1. **Full test suite** — Run `uv run pytest . --testmon --quiet --tb=short` and confirm all tests pass.
2. **Full toolchain** — Run all four checks:
   - `uv run ruff check . --fix && uv run ruff format . --check`
   - `uv run mypy .`
   - `uvx pyright src/ pages/`
3. **Run inline `Verify:` steps** — Execute any verification commands specified in the backlog item (e.g., `**Verify**: uv run radon cc <file> -n B`). Confirm each returns the expected result.
4. **Verify backward compatibility** — Run `uv run pytest reports/*/test_local.py -q --tb=short` to confirm no report-level regressions.
5. **Save lessons to agentmemory** — Call `agentmemory_memory_save()` with:
   - Type: `"workflow"`
   - Tags: `"tackle-backlog", "infrastructure", <backlog-item-keywords>`
   - Content: summary of what was implemented, key decisions made, verification results, and any open items.
6. **Show verification checklist** — Display a structured summary to the user with all verification steps and their results.

# Workflow Verification

### Before implementation
- [ ] Read the specific backlog item (CHANGELOG location, priority, `Verify:` steps, dependencies)
- [ ] Checked for dependencies or blockers (flagged to user if found)
- [ ] Searched agentmemory for relevant prior context (`project:notebooks`)
- [ ] Read all relevant files (config, source, tests, skills, agent definition)
- [ ] Ran full test suite — pre-existing failures reported and added to backlog as new items with 🟢 priority
- [ ] Obtained user approval on the implementation plan before any code changes

### After implementation
- [ ] Full test suite passes (`uv run pytest . --testmon --quiet --tb=short`)
- [ ] Linting and formatting pass (`uv run ruff check . --fix && uv run ruff format . --check`)
- [ ] Static type checking passes (`uv run mypy .`)
- [ ] Live type checking passes (`uvx pyright src/ pages/`)
- [ ] All inline `Verify:` steps from the backlog item executed and passed
- [ ] Backward compatibility verified (no report-level regressions — `uv run pytest reports/*/test_local.py -q --tb=short`)
- [ ] Agent config updated if needed (`.opencode/agents/research-assistant.md`, `opencode.json`)
- [ ] Skills updated if the change introduces new conventions (`.opencode/skills/*/SKILL.md`)
- [ ] CHANGELOG updated: backlog item moved to appropriate weekly `### Infrastructure` section, using the established format
- [ ] Lessons saved to agentmemory (`agentmemory_memory_save()` with type `"workflow"`, tags `"tackle-backlog"`)
- [ ] Verification checklist presented to user
