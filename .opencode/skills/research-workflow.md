---
name: research-workflow
description: MUST be used when planning a new simulation, designing an experiment, or drafting a research plan. Covers survey of existing reports, hypothesis formulation, minimum Hilbert space determination, and assumption validation.
---

# Purpose

Guide the structured execution of quantum simulation research, from initial survey through implementation planning, ensuring rigorous hypothesis testing and minimal computational overhead.

# Rules

1. Always survey existing reports before proposing new simulations.
2. Distill every physics claim into a specific, testable hypothesis with a clear null and alternative.
3. Determine the minimum Hilbert space needed — never overbuild.
4. Proactively identify likely mistakes in the user's assumptions before drafting a plan.
5. Generate figures through the standard pipeline (dataclass to CSV to plot to report embed).
6. Write reports using the prescribed naming convention and format.

# Workflow

## 1. Before writing a plan

1. **Survey existing reports** — Read `reports/` to identify gaps not yet covered by prior simulations.
2. **Clarify the research question** — Distill the physics claim into a specific, testable hypothesis with a clear null and alternative.
3. **Identify the minimum Hilbert space** — Determine the dimension, basis, and operators needed to answer the question without unnecessary overhead.
4. **Challenge assumptions** — Highlight likely mistakes present in the user's request before drafting the plan.

## 2. During implementation

1. Write the document in `reports/` using the format `YYYY-MM-DD-{title}.md`.
2. Follow the **Report Format** as specified in the `report-writing` skill.
3. **Generate figures** — If the report contains parameter-sweep tables suited to line plots:
   - Add `to_dataframe()` / `save_csv()` / `from_csv()` to the relevant result dataclass in `src/`
   - Add a `plot_<description>()` function in `src/visualization/ancilla_plots.py` (or create a new module there) that reads the dataclass or CSV and saves SVG
   - Add a `generate_<name>()` function in `src/visualization/report_figures.py` (runs sim, saves CSV, renders SVG)
   - CSVs go to `reports/raw_data/{date}-{name}.csv`; SVGs go to `reports/figures/{date}-{name}.svg`
   - Embed in the report with `![alt](reports/figures/{date}-{name}.svg)`
4. **Over-cautious serialization**: When adding `to_dataframe()` to a result dataclass, include every input parameter (theta, T_H, bounds, SQL reference) alongside the computed results. The CSV must be fully self-describing — omitted metadata is permanently lost.
5. **Fail-fast deserialization**: When adding `from_csv()`, require all expected columns. Do not silently default metadata fields. Raise a clear `ValueError` listing missing columns and directing the user to regenerate the file. A silent default today is a subtle data corruption tomorrow.

# Verification

- [ ] All existing reports in `reports/` have been surveyed.
- [ ] The hypothesis is falsifiable and maps to success criteria.
- [ ] Hilbert space dimension is explicitly stated and justified.
- [ ] Assumptions have been surfaced to the user before implementation.
- [ ] Figure generation uses the standard pipeline (dataclass → CSV → plot → embed).
- [ ] `to_dataframe()` includes all input parameters, not just computed results.
- [ ] `from_csv()` requires all expected columns — no silent fallback defaults for metadata.

# Anti-patterns

- Proposing simulations without first checking `reports/` for prior work.
- Using larger Hilbert spaces than necessary — wastes compute and obscures results.
- Skipping the assumption-challenge step and implementing a flawed plan.
- Writing reports without following the prescribed section order or emoji conventions.
- Generating figures manually instead of using the pipeline (dataclass → CSV → plot).
- Omitting input parameters from `to_dataframe()` — every CSV should be self-describing.
- Adding silent fallback defaults in `from_csv()` for metadata columns.
