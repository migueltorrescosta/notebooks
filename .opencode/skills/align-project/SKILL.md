---
name: align-project
description: MUST be used for periodic project-level maintenance — priority alignment (backlog priorities), sanity checks (repo health, stale files, toolchain), and product-focus review (memory consolidation, cross-references). Run every 2–4 weeks or when the project direction feels unclear.
---

# Purpose

Keep the project on track through three pillars:

1. **Priority alignment** — Ensure the backlog 🔴🟠🟡🟢 priorities reflect current reality and the CHANGELOG is structurally sound.
2. **Sanity check** — Detect stale files, shared-infrastructure promotion opportunities, and toolchain regressions.
3. **Product focus** — Verify agentmemory health, skill consistency, and OpenCode configuration integrity.

This skill inspects, reports, and auto-fixes only purely mechanical issues (formatting, colour legend). Substantive decisions (priority reassignment, file deletion, promotion) are surfaced as suggested actions for human approval.

# Rules

1. Read the CHANGELOG.md thoroughly before taking any action.
2. Do **not** modify simulation code in `src/`, `pages/`, `reports/`, or any test file — this is an inspect-and-report skill only.
3. Auto-fix only purely mechanical items (CHANGELOG formatting, colour legend typos, stale whitespace). File deletions and priority changes must be suggested, not applied.
4. Idempotent — running twice with no intervening work should produce no changes.
5. Fail-fast — if the toolchain (ruff, pytest, etc.) is broken, report it immediately and do not proceed to later actions that depend on a healthy repo.
6. Respect skill boundaries — do not duplicate the per-report work of `audit-code`, `compile-report`, or the other skills.
7. Follow the same CHANGELOG update conventions as other skills: entry under the appropriate weekly section, backlog item removed if completed.

# Workflow

## 1. Preparation

1. **Search agentmemory** — Call `memory_recall` or `memory_smart_search` with query "align-project backlog priorities repo health" to find prior runs, decisions, and patterns.
2. **Read the CHANGELOG.md** — Open `CHANGELOG.md` and note:
   - The colour legend at the top of the `# Backlog` section.
   - The full `# Backlog` section — list every item, its priority emoji, and theme group.
   - The current weekly section (date range).
3. **Survey the repo state** — Run `ls reports/` to get a high-level view of active and completed report directories.
4. **Review the findings document** — Read `reports/findings/interferometric_sensitivity_improvements.md` to check whether it accurately reflects the current set of completed experiments. Note any experiments that are missing from the findings document but exist in the CHANGELOG or `reports/` directories. Flag for update if the findings document is out of date.

## 2. Priority Alignment

### 2a. Backlog priority reassignment

1. Read every item in the `# Backlog` section.
2. Reassign 🔴🟠🟡🟢 emojis using this scheme:
   - **🔴 (Critical)** — Exactly 2–3 items that are most urgent and blocking progress. At most one per theme group.
   - **🟠 (High)** — Important but not blocking. Roughly 30% of items.
   - **🟡 (Medium)** — Worth doing when time permits. Roughly 30% of items.
   - **🟢 (Low)** — Nice-to-have, deferred, or awaiting external dependencies. Remaining items.
3. Confirm the colour legend at the top of the `# Backlog` section is present and correct:
   ```
   Priority colours: 🔴🟠🟡🟢
   ```
   If missing or incorrect, fix it.
4. If any item is clearly obsolete (e.g., already completed, superseded by later work, or no longer relevant), flag it for removal.

### 2b. CHANGELOG structural audit

1. Verify each weekly section follows the correct structure:
   - `## Week NN (Mon DD–Sun DD)`
   - Sub-sections: `### New Report`, `### Infrastructure` (both optional but must be in this order if present)
   - Entries use the format: `- **Title** (#YYYYMMDD) — description`
2. Verify no completed report is missing from the CHANGELOG (cross-reference `reports/` directories against CHANGELOG entries).
3. Verify the `# Backlog` section does not contain items that have been completed (check each item against the weekly entries).
5. Flag any structural issues for human fixing.

## 3. Sanity Check

### 3a. Stale file scan

1. **Stale runner scripts** — Glob for `reports/*/run_parallel.py` and `reports/*/sweep_runner.py`. For each file:
   - Check whether its functionality is covered by the experiment module in the same directory.
   - If yes, delete it (the runner's core functionality lives in the experiment module; git history preserves the original).
   - If no, note the gap.
2. **Unreferenced Parquet files** — List all `reports/*/raw_data/*.parquet` files. Cross-reference against their report's `.md` file (grep for the filename stem). Flag any Parquet files not referenced by any report as potentially orphaned.
3. **Orphaned report directories** — Check if any `reports/YYYYMMDD/` directory lacks both a `.md` file and a `.py` experiment module. Flag as possibly stale scaffolding.
4. **Dead code detection** — Run `vulture . --exclude '.venv,.opencode,.git,__pycache__' --sort-by-size` to identify unused functions, methods, and imports. Review findings manually (expect ~75% noise — mock `return_value`, pytest hooks, argparse-dispatch, and public API functions are common false positives).

### 3b. Shared-infrastructure cross-report analysis
1. **Cross-reference for duplicates** — Compare each function/constant name across report directories:
   - **Exact match** — Same name, same signature, same implementation. → Flag for promotion to `src/`.
   - **Near match** — Same logic, different name or minor signature difference. → Flag with suggested unified API.
   - **Superficial match** — Same name, different implementation. → Flag as naming collision.
4. **Check against `src/`** — For each candidate, verify it does not already exist in `src/` modules (avoid re-promoting already-promoted code).
5. Produce a promotion-opportunity table: `| Function/Constant | Reports | Match Type | Suggested `src/` Module |`.

### 3c. Full toolchain health

Run all four checks in order. Stop at the first failure:

1. **Linting** — `uv run ruff check . && uv run ruff format . --check`
2. **Type checking (static)** — `uv run mypy .`
3. **Type checking (live)** — `uvx pyright src/ pages/`
4. **Tests** — `uv run pytest . --testmon --quiet --tb=short`
5. **Coverage** — `uv run coverage run -m pytest -q --tb=short ; uv run coverage report --fail-under=85` 

For any failure, report the full command output and flag as a regression. Do not attempt to fix.

## 4. Product Focus Review

### 4a. Agentmemory health

1. **Diagnose** — Run `agentmemory_memory_diagnose()`. Look for:
   - Stuck or orphaned sessions.
   - Inconsistent memory counts.
2. **Consolidate** — Run `agentmemory_memory_consolidate()` with tier `"semantic"`, then tier `"procedural"`. These are the two tiers flagged as empty in the backlog.
3. **Report** — Note the number of memories before and after consolidation. If both tiers remain empty, flag for investigation.

### 4b. Skill cross-reference check

1. Verify that every SKILL.md file under `.opencode/skills/`:
   - Has a `description` front-matter field.
   - References the CHANGELOG in its workflow or checklist.
   - Uses consistent section naming conventions.
2. Check for any skill that references another skill by name (e.g., `audit-code` mentions `build-simulation`). Ensure the cross-reference is accurate.
3. Flag any inconsistencies.

### 4c. OpenCode configuration consistency

1. Read `opencode.json` to verify:
   - `default_agent` matches an existing agent definition.
   - No stale or missing skill references.
2. List all `.opencode/skills/*/` directories and verify:
   - Each has a `SKILL.md` file.
   - No orphaned skill directories exist (directory with no valid SKILL.md).

## 5. Summary and Recommendations

After all 8 actions, produce a summary structured as:

```
## Align-Project Summary — YYYY-MM-DD

### Changed
- (list of mechanical auto-fixes applied)

### Suggested
- (list of human-review-required items: priority changes, file deletions, promotions)

### Blockers
- (list of regressions or issues found, if any)

### Next review
- (recommended date or trigger for next run)
```

Save this summary to agentmemory via `agentmemory_memory_save()` with type `"pattern"` and tags `"align-project", "maintenance"`.

# Workflow Verification

### Before implementation
- [ ] Searched agentmemory for prior align-project runs and relevant decisions (`project:notebooks`)
- [ ] Read CHANGELOG.md (colour legend, full Backlog, current weekly section)
- [ ] Surveyed reports directory (`ls reports/`)
- [ ] Reviewed findings document (`reports/findings/interferometric_sensitivity_improvements.md`) for currency

### During analysis
- **Priority Alignment (§2a Backlog)**:
  - [ ] Read every item in the `# Backlog` section
  - [ ] Reassigned 🔴🟠🟡🟢 emojis: exactly 2–3 🔴, at most one per theme group, rest roughly evenly split
  - [ ] Confirmed the colour legend at the top of `# Backlog` is present and correct (`Priority colours: 🔴🟠🟡🟢`)
  - [ ] Flagged any obsolete or superseded items for removal (suggested, not applied)
- **CHANGELOG structural audit (§2b)**:
  - [ ] Verified each weekly section follows `## Week NN (Mon DD–Sun DD)` with sub-sections in order (`### New Report` before `### Infrastructure`)
  - [ ] Verified entries use the format `- **Title** (#YYYYMMDD) — description`
  - [ ] Cross-referenced all `reports/YYYYMMDD/` directories against CHANGELOG entries — no completed reports missing
  - [ ] Verified no completed items remain in the Backlog (cross-referenced against weekly entries)
- **Stale file scan (§3a)**:
  - [ ] Scanned for stale runner scripts (`reports/*/run_parallel.py`, `reports/*/sweep_runner.py`) — coverage judged against the experiment module
  - [ ] Scanned for unreferenced Parquet files (`reports/*/raw_data/*.parquet`) — cross-referenced against report `.md` files
  - [ ] Checked for orphaned report directories (any `reports/YYYYMMDD/` lacking both `.md` and an experiment module)
- **Shared-infrastructure analysis (§3b)**:
  - [ ] Cross-referenced each candidate against existing `src/` modules (to avoid re-promoting already-promoted code)
  - [ ] Categorised each duplicate as exact-match, near-match, or superficial-match
  - [ ] Produced a promotion-opportunity table with columns: Function/Constant, Reports, Match Type, Suggested src/ Module
- **Toolchain health (§3c)**:
  - [ ] Ran linting: `uv run ruff check . && uv run ruff format . --check`
  - [ ] Ran static type checking: `uv run mypy .`
  - [ ] Ran live type checking: `uvx pyright src/ pages/`
  - [ ] Ran tests: `uv run pytest . --testmon --quiet --tb=short`
  - [ ] Ran coverage: `uv run coverage run -m pytest -q --tb=short ; uv run coverage report --fail-under=85`
  - [ ] If coverage run was unacceptably slow due to slow tests, add a backlog item to reduce the runtime of the longest running slow test
  - [ ] Reported any toolchain failures as blockers (do not proceed if toolchain is broken)
- **Product focus (§4)**:
  - [ ] Ran `agentmemory_memory_diagnose()` — checked for stuck/orphaned sessions, inconsistent memory counts
  - [ ] Ran `agentmemory_memory_consolidate(tier="semantic")` and `agentmemory_memory_consolidate(tier="procedural")` — recorded results
  - [ ] Verified every SKILL.md under `.opencode/skills/` has a `description` front-matter field, references the CHANGELOG, and uses consistent section naming
  - [ ] Checked all cross-skill references by name (e.g., `audit-code` → `build-simulation`) — verified accuracy
  - [ ] Verified `opencode.json`: `default_agent` matches an existing agent definition; no stale skill references
  - [ ] Verified every `.opencode/skills/*/` directory has a valid `SKILL.md`; no orphaned skill directories

### After implementation
- [ ] Backlog priorities reassigned (2–3 🔴, colour legend confirmed/updated)
- [ ] CHANGELOG structural audit completed (no stale entries, no formatting issues)
- [ ] Stale file scan completed (runner scripts, unreferenced Parquets, orphaned dirs)
- [ ] Shared-infrastructure cross-report analysis completed (promotion table generated)
- [ ] Toolchain health verified (ruff, mypy, pyright, pytest, coverage all pass — or regressions reported)
- [ ] Agentmemory health checked and semantic/procedural tiers consolidated
- [ ] Skill cross-references verified (all SKILL.md files consistent)
- [ ] OpenCode configuration verified (agents, skills, config all consistent)
- [ ] Summary produced with sections: Changed, Suggested, Blockers, Next review
- [ ] Summary saved to agentmemory via `agentmemory_memory_save()` with type `"pattern"` and tags `"align-project", "maintenance"`
- [ ] CHANGELOG updated with entry under the appropriate weekly section using the format `- **Project alignment review** — backlog priorities refreshed, sanity checks completed. [# of backlog items reviewed, # of actions taken/suggested].`; backlog item removed if one was completed by this run
- [ ] No simulation code (`src/`, `pages/`, `reports/*/*.py` experiment modules, tests) was modified — only CHANGELOG formatting, agent definition, and skill files were touched
