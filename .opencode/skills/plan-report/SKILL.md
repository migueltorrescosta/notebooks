---
name: plan-report
description: MUST be used when writing a new report in reports/ based on a user request. Surveys existing reports, clarifies the research question, determines the minimum Hilbert space, and writes the formatted markdown file. Do NOT use for editing existing reports or for implementation.
---

# Purpose

Guide the creation of a new simulation report from initial research question through to a complete, formatted markdown document in `reports/`. This skill covers the planning, hypothesis formulation, and writing stages only — never implementation or result generation.

# Rules

1. Always survey existing reports in `reports/` before proposing a new simulation.
2. Distill every physics claim into a specific, testable hypothesis with a clear null and alternative.
3. Determine the minimum Hilbert space needed — never overbuild.
4. Proactively identify likely mistakes in the user's assumptions before drafting a plan.
5. Follow the prescribed report section order, emoji conventions, and formatting rules exactly.
6. Do not generate figures, run simulations, or modify code — this skill is for planning and writing only.

# Workflow

## 1. Before writing

1. **Survey existing reports** — Read `reports/` to identify gaps not yet covered by prior simulations.
2. **Clarify the research question** — Distill the physics claim into a specific, testable hypothesis with a clear null and alternative.
3. **Determine the minimum Hilbert space** — Determine the dimension, basis, and operators needed to answer the question without unnecessary overhead.
4. **Challenge assumptions** — Highlight likely mistakes present in the user's request before drafting the plan.
5. **Iterate with the user** — Ask questions and refine the plan until the user is satisfied. Only then write the report file.

## 2. Writing the report

1. Write the document in `reports/YYYYMMDD/` using the format `YYYY-MM-DD-{title}.md`.
2. Follow the prescribed **section order**, **per-section patterns**, and **formatting rules**.
3. When the report requires a multi-model comparison, include a **Models Survey** section.
4. When conservation-law analysis or analytical bound derivation is central, include a **Physical Invariants / Analytical Bounds** section.

# Verification

- [ ] All existing reports in `reports/` have been surveyed.
- [ ] The hypothesis is falsifiable and maps to success criteria.
- [ ] Assumptions have been surfaced to the user before implementation.
