---
name: review-report
description: MUST be used when reviewing and updating a report in reports/ with actual results from raw_data and figures. Reads the report, inspects generated CSVs and SVGs, and updates the report with results. Does NOT change any code.
---

# Purpose

Review a report in `reports/` after simulations have been run and results generated. Inspect the raw data and figures, update the report's Results and Conclusions sections with actual outcomes, and verify the report is internally consistent. This skill exclusively edits the report markdown — never modifies code in `src/`, `pages/`, or tests.

# Rules

1. Read the report first — understand the hypothesis, success criteria, and expected results.
2. Inspect all raw data CSVs in `reports/raw_data/` and figures in `reports/figures/` associated with this report.
3. Update only the report markdown file. Never modify code.
4. Mark each success criterion as PASS or FAIL based on actual data.
5. Update the Results section with quantitative findings and Key Finding paragraphs.
6. Update the Conclusions section with a summary of what was learned.
7. Verify that figure references in the report resolve to actual SVG files.
8. Verify that the report format is correct (section order, emoji placement, etc.).
9. Ensure internal consistency between Results, Success Criteria, and Conclusions.

# Workflow

## 1. Preparation

1. **Read the report** — Open the target report in `reports/` and note:
   - The hypothesis and success criteria
   - The parameter sweeps that were specified
   - The expected physical invariants and bounds
   - Any pre-experiment status tables that need updating
2. **Survey the raw data** — List all files in `reports/raw_data/` matching the report's date prefix.
3. **Survey the figures** — List all files in `reports/figures/` matching the report's date prefix.

## 2. Reviewing results

1. For each CSV in `reports/raw_data/`:
   - Load it using the corresponding dataclass's `from_csv()` method
   - Verify numerical values against expected ranges
   - Check that physical invariants hold (normalisation, unitarity, sensitivity positivity, F_Q ≥ F_C, Δφ_Q ≤ Δφ_C)
   - Note any anomalies or unexpected values
2. For each SVG in `reports/figures/`:
   - Confirm the figure exists and renders correctly
   - Verify the figure filename matches what's referenced in the report
   - Note any missing figures

## 3. Updating the report

1. **Results section** — For each experiment subsection:
   - Replace `PENDING` status with actual `PASS`/`FAIL`
   - Add quantitative results from the CSV data (key numerical values, exponents, comparisons)
   - Add a **Key Finding** paragraph at the end of each subsection
2. **Success Criteria section**:
   - Append `— PASS` or `— FAIL` to each criterion based on actual data
   - Add a summary paragraph describing what passed, what failed, and possible next steps
3. **Conclusions section**:
   - Summarize what was learned and whether the hypothesis was supported
   - Reference specific Results
   - Optionally add **Open items** for future directions

# Verification Checklist

For each report review:

- [ ] Report read and understood before reviewing data
- [ ] All CSVs in `reports/raw_data/{date}-*` inspected
- [ ] All SVGs in `reports/figures/{date}-*` checked for existence
- [ ] Results section updated with actual PASS/FAIL status
- [ ] Key Finding paragraphs added to each experiment subsection
- [ ] Success Criteria updated with PASS/FAIL annotations
- [ ] Conclusions updated with summary of findings
- [ ] Internal consistency: Results match Success Criteria match Conclusions
