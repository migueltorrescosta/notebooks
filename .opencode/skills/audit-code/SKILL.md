---
name: audit-code
description: MUST be used when reviewing code implementation against project standards. Reads the report, then checks the codebase for violations of architecture, testing, numerical, and coding conventions. Suggests fixes but does NOT implement them.
---

# Purpose

Audit the codebase implementation related to a specific report against all project standards. Identify violations of architecture conventions, testing standards, numerical guidelines, and coding workflow rules. Produce a list of specific, actionable fixes without making any code changes.

# Rules

1. Read the report first to understand what was implemented.
2. Inspect all new or modified code related to the report.
3. Flag every violation of the standards with a specific file, line, and suggested fix.
4. Do NOT modify any files — this skill is for auditing and suggesting fixes only.
5. Prioritize violations by severity: correctness > deduplication (shared-infrastructure promotion) > performance > style.
6. Check both the implementation code and the tests.

# Workflow

## 1. Preparation

1. **Read the report** — Open the target report in `reports/` and identify:
   - The Hilbert space dimensions, operators, and protocol implemented
   - The simulation functions and dataclasses used
   - The expected outputs and success criteria
2. **Read the relevant code** — Open all source files related to this report:
   - The report's own experiment module (e.g., `phase_modulated_drive.py`) and its test file (`test_*.py`) in `reports/YYYYMMDD/`
   - Physics modules (`src/physics/`)
   - Evolution modules (`src/evolution/`)
   - Analysis modules (`src/analysis/`)
   - Visualization modules (`src/visualization/`)
   - Streamlit pages (`pages/`)
   - Test files (`tests/` and `src/**/test_*.py`)

## 2. Shared-Infrastructure Analysis

Before listing individual violations, identify opportunities to reduce duplication by promoting shared infrastructure to `src/` modules. This has higher impact than fixing cosmetic or style issues in duplicate code.
1. **Cross-reference for duplicates** — Compare each function/constant name and signature with:
   - Other report experiment modules (same name? same signature? same implementation body?)
   - Existing `src/` modules (does a canonical version already exist?)
4. **Categorise each duplicate**:
   - **Exact match** — Same name, same signature, same implementation. → Promote to `src/`.
   - **Near match** — Same logic, different name or minor signature difference. → Promote with a unifying API, then update all call sites.
   - **Superficial match** — Same name but different implementation. → Keep both local; flag as naming collision (MAJOR).
5. **Flag for promotion** — For each function/constant duplicated across ≥2 reports:
   - Record it as a `MAJOR` violation with the suggested fix: "Create `src/.../...py` containing the shared implementation, then update all N report experiment modules to import from the new module."
   - This must be done **before** any cosmetic-only fixes (E402, line length, etc.) are suggested for the duplicated code, since the cosmetic issues will disappear once the code is promoted to `src/`.

## 3. Reporting violations

For each violation found, produce an entry with:

1. **File** — The absolute or relative path to the file
2. **Line(s)** — The relevant line numbers
3. **Standard violated** — Which standard (e.g., "Code Architecture §Serialization completeness")
4. **Severity** — `CRITICAL` (correctness bug), `MAJOR` (standard violation), `MINOR` (style)
5. **Description** — What is wrong
6. **Suggested fix** — What to change, with enough detail that another agent can implement it

Example:

```
- **File**: `src/physics/mzi_states.py`, line 142
- **Standard**: Numerical Guidelines §Stability
- **Severity**: CRITICAL
- **Description**: Uses `np.linalg.inv` to compute the inverse of the beam-splitter matrix.
- **Suggested fix**: Replace with `np.linalg.solve` or precompute the analytic inverse.
```

## 4. Summary

After the full audit, provide:

- Total number of violations found, broken down by severity
- A brief assessment of overall code quality for this implementation
- Recommended order of fixing (highest severity first)

# Workflow Verification

### Before implementation
- [ ] Searched agentmemory for relevant prior decisions and standards (`project:notebooks`)
- [ ] Read the report (Hilbert space, operators, protocol)
- [ ] Read all relevant source files (report's experiment module and its test file, plus affected modules in `src/`, `pages/`, `tests/`)
- [ ] Consulted CHANGELOG backlog if applicable

### During analysis
- [ ] Cross-referenced each candidate against existing `src/` modules (to avoid re-promoting already-promoted code)
- [ ] Categorised each duplicate as exact-match, near-match, or superficial-match
- [ ] Flagged all ≥2-report duplicates for promotion to `src/` as `MAJOR` violations
- [ ] Ensured cosmetic/lint fixes for duplicated code are deferred until after promotion (the cosmetic issues will disappear when the code moves to `src/`)

### After implementation
- [ ] No code was modified during audit
- [ ] Coverage >= 85% — run `uv run coverage run -m pytest -q --tb=short ; uv run coverage report --fail-under=85`
- [ ] Checked for pyright violations in `src/` and `pages/` (`uvx pyright src/ pages/`)
- [ ] Checked for module-level constants in `src/` (violation of Global Constraint §6)
- [ ] Checked for imports from report experiment modules originating outside the report directory (violation of Global Constraint §7)
- [ ] Checked that new report-specific code was added to the experiment module, not to `src/` (violation of Global Constraint §5)
- [ ] Each violation recorded with file, line, standard violated, severity, description, and suggested fix
- [ ] CHANGELOG updated with entry under the appropriate weekly section if the audit uncovered actionable items; any errors that predated the current session added to the backlog
- [ ] Ran dead code detection — `vulture . --exclude '.venv,.opencode,.git,__pycache__' --sort-by-size` (review findings manually; expect ~75% noise from mock `return_value`, pytest hooks, argparse-dispatch, and public API functions)
- [ ] Saved key findings to agentmemory (`project:notebooks`)
