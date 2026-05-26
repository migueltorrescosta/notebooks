---
name: review-implementation
description: MUST be used when reviewing code implementation against project standards. Reads the report, then checks the codebase for violations of architecture, testing, numerical, and coding conventions. Suggests fixes but does NOT implement them.
---

# Purpose

Audit the codebase implementation related to a specific report against all project standards. Identify violations of architecture conventions, testing standards, numerical guidelines, and coding workflow rules. Produce a list of specific, actionable fixes without making any code changes.

# Rules

1. Read the report first to understand what was implemented.
2. Inspect all new or modified code related to the report.
3. Flag every violation of the standards with a specific file, line, and suggested fix.
4. Do NOT modify any files — this skill is for auditing and suggesting fixes only.
5. Prioritize violations by severity: correctness > performance > style.
6. Check both the implementation code and the tests.

# Workflow

## 1. Preparation

1. **Read the report** — Open the target report in `reports/` and identify:
   - The Hilbert space dimensions, operators, and protocol implemented
   - The simulation functions and dataclasses used
   - The expected outputs and success criteria
2. **Read the relevant code** — Open all source files related to this report:
   - The report's own `local.py` and `test_local.py` in `reports/YYYYMMDD/`
   - Physics modules (`src/physics/`)
   - Evolution modules (`src/evolution/`)
   - Analysis modules (`src/analysis/`)
   - Visualization modules (`src/visualization/`)
   - Streamlit pages (`pages/`)
   - Test files (`tests/` and `src/**/test_*.py`)

## 2. Reporting violations

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

## 3. Summary

After the full audit, provide:

- Total number of violations found, broken down by severity
- A brief assessment of overall code quality for this implementation
- Recommended order of fixing (highest severity first)

# Verification

- [ ] Report read and understood
- [ ] Report's `local.py` and `test_local.py` inspected
- [ ] All modified/added source files in `src/`, `pages/`, `tests/` inspected
- [ ] Checked for **pyright violations** in `src/` and `pages/` (run `pyright src/ pages/`)
- [ ] Checked for **module-level constants** in `src/` code (violation of Global Constraints §6)
- [ ] Checked for **imports from `local.py`** coming from outside the report directory (violation of Global Constraints §7)
- [ ] Checked that new report-specific code was added to `local.py`, not directly to `src/` (violation of Global Constraints §5)
- [ ] Violations recorded with file, line, standard, severity, description, and suggested fix
- [ ] No code was modified during the review
