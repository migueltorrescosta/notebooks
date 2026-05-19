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
3. **Identify the minimum Hilbert space** — Load `physics-reference` (§1) to consult available basis conventions and dimension formulas. Determine the dimension, basis, and operators needed to answer the question without unnecessary overhead.
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
4. **Serialization**: Follow `code-architecture.md` (§Code Style) conventions for `to_dataframe()` completeness and `from_csv()` fail-fast deserialization. Every dataclass CSV must be self-describing.

# Verification

- [ ] All existing reports in `reports/` have been surveyed.
- [ ] The hypothesis is falsifiable and maps to success criteria.
- [ ] Hilbert space dimension is explicitly stated and justified.
- [ ] Assumptions have been surfaced to the user before implementation.
- [ ] Figure generation uses the standard pipeline (dataclass → CSV → plot → embed).
- [ ] Serialization follows `code-architecture.md` conventions (completeness + fail-fast).

# Anti-patterns

- Proposing simulations without first checking `reports/` for prior work.
- Using larger Hilbert spaces than necessary — wastes compute and obscures results.
- Skipping the assumption-challenge step and implementing a flawed plan.
- Writing reports without following the prescribed section order or emoji conventions.
- Generating figures manually instead of using the pipeline (dataclass → CSV → plot).
- Violating serialization conventions defined in `code-architecture.md` (§Code Style) — omitting input parameters from `to_dataframe()` or adding silent fallback defaults in `from_csv()`.
