---
name: generate-results
description: MUST be used when running simulations to generate raw data and figures based on a report in reports/. Explores the implemented code, runs parameter sweeps, saves Parquet files and SVGs, and updates code if needed. Does NOT write reports or implement new features.
---

# Purpose

Execute the simulation experiments defined in a report, produce all raw data and figures, and update the implementation if minor adjustments are needed during execution. This skill covers running simulations, generating plots, and saving results — not planning, implementing new code, or writing reports.

# Rules

1. Read the report thoroughly before running any simulations.
2. Explore existing implemented code in `src/` **and** the report's own `local.py` to understand available functions. Functions for the report's simulation may live in either location.
3. Generate all figures through the standard pipeline (dataclass → Parquet → plot → embed).
4. Respect the 100 ms per-simulation performance budget.
5. Save results with the correct naming convention and directory structure.
6. If code needs minor fixes to run correctly, make the minimum changes needed — do not implement new features.
7. If the codebase lacks a required simulation function, flag it for the implement-plan skill rather than building it.

# Workflow

## 1. Preparation

1. **Read the report** — Open the target report in `reports/` and identify:
   - The parameter sweeps (ranges, step sizes, point counts)
   - The expected output quantities (sensitivity vs N, QFI values, etc.)
   - The success criteria and expected physical invariants
2. **Explore existing code** — Read relevant modules in `src/` **and** the report's `local.py` to find:
   - Result dataclasses and their `to_dataframe()` / `from_parquet()` methods
   - Simulation functions that implement the report's protocol
   - Existing figure generation functions in `src/visualization/`
3. **Check for figure generation infrastructure** — Look in `src/visualization/report_figures.py` or create a `generate_<name>()` function there if one does not exist.
4. **Record all relevant metrics**: results are often reused in the future. Record all inputs, outputs and relevant intermediate hidden variables needed to provide detailed information for post-analysis. 

## 2. Running simulations

1. Execute parameter sweeps defined in the report's Numerical Simulation section.
2. Save raw data as Parquet in `reports/{date}/raw_data/{date}-{name}.parquet` (where `{date}` is the YYYYMMDD report directory).
3. Use `to_dataframe()` / `save_parquet()` on result dataclasses (see Code Architecture §Serialization for completeness requirements).
4. If a simulation function is missing or incorrect, fix it minimally. If the fix is substantial, flag it for implement-plan.

## 3. Generating figures

1. Add a `plot_<description>()` function in `src/visualization/ancilla_plots.py` (or create a new module there) that reads the dataclass or Parquet file and saves SVG.
2. Add a `generate_<name>()` function in `src/visualization/report_figures.py` that runs the simulation, saves Parquet, and renders SVG.
3. SVGs go to `reports/{date}/figures/{date}-{name}.svg` (where `{date}` is the YYYYMMDD report directory).
4. Embed in the report with `![alt](figures/{date}-{name}.svg)` (relative path from within the report directory).

## 4. Code updates

- If minor fixes are needed (e.g., missing parameter in `to_dataframe()`, incorrect default), make the minimal change.
- If the code needs substantial changes (new operator, new state type, new solver), flag it for implement-plan rather than building it here.
- After any code change, run `uv run pytest . --testmon --quiet --tb=short` to verify nothing is broken.
- After any code change, run `uv run ruff check . --fix && uv run ruff format .`, `uv run mypy .`, and `uvx pyright src/ pages/`.

# Standard Pipeline Instructions

Every simulation run follows a strict four-step sequence. Execute these in order:

1. **Read the report** – Extract parameter sweeps, output quantities, and success criteria from the report markdown.

2. **Run the simulation** – Call the simulation functions (found in `src/` modules or the report's own `local.py`) for each point in the parameter sweep. Respect the 100 ms per-simulation budget.

3. **Save raw data** – Use the result dataclass's `save_parquet()` method to write to `reports/{date}/raw_data/{date}-{name}.parquet`.

4. **Generate and embed figures** – Create a `plot_<name>()` function in `src/visualization/` that reads the Parquet file and saves an SVG to `reports/{date}/figures/{date}-{name}.svg`. Embed the SVG in the report with `![alt](figures/{date}-{name}.svg)`.

# Figure Aesthetics Guidelines

- Use clear, readable fonts. Label all axes with units where applicable.
- Use colorblind-friendly palettes where possible.
- Save as SVG for scalability and report embedding.
- Include legend when multiple curves are present.
- Plot the SQL and HL as reference lines (dashed) when showing scaling plots.
- Use log-log axes for scaling exponent visualizations.
- Each figure should convey one clear message — avoid overloading a single figure.

# Verification

- [ ] Report's parameter sweeps have been fully executed
- [ ] All raw data saved as Parquet in `reports/{date}/raw_data/`
- [ ] All figures saved as SVG in `reports/{date}/figures/`
