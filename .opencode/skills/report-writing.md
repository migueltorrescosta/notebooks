---
name: report-writing
description: MUST be used when creating or editing a report in reports/. Covers section order, emoji conventions, inline callouts, per-section patterns, physics scope checklist, and formatting rules.
---

# Purpose

Define the precise format and structure for all simulation reports in `reports/`, ensuring consistency, completeness, and readability across the project.

# Rules

1. Every report filename MUST follow `YYYY-MM-DD-{title}.md`.
2. Inline annotations use **bold text** instead of emojis. Use only: **Key Finding**, **Open items**, **See**, **Validation**.
3. Status columns in tables use only text: `PASS`, `FAIL`, `PENDING`, `PARTIAL`.
4. Section header emoji goes after `## ` and before the title text: `## 🧪 Hypothesis` (not `## Hypothesis 🧪`).
5. Section-title emojis are the only emojis allowed in reports — never add decorative or ad-hoc emojis outside section headers.
6. Section emojis appear once per document, not on sub-subsections.
7. For reference documents (e.g., `.opencode/skills/physics-reference.md`), apply emojis only to section titles where they fit semantically; all body text and tables follow the same emoji-free rules.
8. Don't include repo-specific filepaths in the Numerical Simulation section (theoretical model should be implementation-agnostic).
9. Don't use breaklines in prose when inline equations suffice.
10. Don't use `|` inside LaTeX math ($...$) or in table cells unless wrapped in LaTeX `\vert `.
11. Don't use `\ket` — it is not recognised by the Markdown engine.
12. Use `\mathbb{1}_n` to signify an $n$ dimensional identity operator.
13. Use existing external implementations (`qutip`, `scipy`, etc). Do not re-implement existing functionality.

# Section Order

All reports MUST include sections in the following order:

| Section | Emoji | Mandatory | When |
|---------|-------|-----------|------|
| Hypothesis | 🧪 | Yes | Always |
| Theoretical Model | ⚛️ | Yes | Always |
| Models Survey | 📊 | No | Multi-model comparison reports |
| Numerical Simulation | 💻 | Yes | Always |
| Expected Failure Conditions | ⚠️ | Yes | Always |
| Results | 🔬 | Yes | Always |
| Success Criteria | ✅ | Yes | Always |
| Physical Invariants / Analytical Bounds | ⚖️ | No | Conservation-law analysis or analytical bound derivation. Use `###` heading level. |
| Conclusions | 🏁 | Yes | Always |

# Per-Section Patterns

## 🧪 Hypothesis
Numbered list of specific, testable claims (or paragraph form for a single claim). Each claim maps to an item in the Success Criteria list.

## ⚛️ Theoretical Model
Continuous prose narrative covering the **Hilbert space** (basis vectors, dimension formula, index ordering convention), **operators** (explicit matrix elements or generator definitions in the chosen basis), the **circuit protocol** as a step-by-step unitary sequence (BS → phase → hold → BS), the **measurement** observable and sensitivity formula used, and any **tables** for states, operators, or noise channels. Avoid subsection headings — the entire section must read as a single flowing exposition. Use **bold font** when introducing and defining each concept for the first time.

The following items MUST appear in this section:
- **State representation** — Fock basis, coherent states, density matrices, etc.
- **Units** — Dimensionless by default unless overridden
- **Conventions** — Phase sign, beam splitter unitary definition
- **Hilbert space** — Dimension and basis explicitly stated

## 📊 Models Survey (survey reports only)
Central table mapping models to expected scaling exponents. Columns: `| Model | Input State | Noise | Expected α | Implementation Status |`. This is the definitive reference. Use text status indicators (PASS/FAIL/PENDING/PARTIAL).

## 💻 Numerical Simulation
Three subsections:

1. **Implementation strategy** — ordered list describing the high-level approach (composable pipeline, key function signatures, dimension management, solvers). Every dataclass must store all input parameters alongside computed results and serialize them in `to_dataframe()` — CSVs must be self-describing.
2. **Parameter sweep** — table with single "Parameter" column (name and symbol combined), single "Range" column (range, step size, and point count combined), and "Purpose" column
3. **Validation** — prose paragraph with inline equations enumerating physical invariants (state normalisation, unitarity, variance positivity, sensitivity positivity, baseline recovery, commutation relations, Hermiticity)

### 🔧 Implementation Status
Bullet-point list with **bold** component names followed by an em-dash and description (e.g., operator construction, state preparation, unitary evolution, sensitivity computation). Include a test count summary line after the list. Only for completed implementations.

## ⚠️ Expected Failure Conditions
Each entry includes: failure condition name and description combined via em-dash, and mitigation strategy. Table format is required: `| Failure | Mitigation |` (failure name and description merged into a single "Failure" column with an em-dash separator).

## 🔬 Results
**Required in every report.** Organise results as subsections, one per experiment. Each subsection ends with a **Key Finding** paragraph. A summary table may conclude the section.

- **Pre-experiment**: a table with `PENDING` status — marks what hasn't been run yet
- **Post-experiment**: a table with actual status (`PASS`/`FAIL`), plus a **Key Finding** paragraph starting with `**Key Finding**`
- Completed reports include a quantitative summary table and cross-references (via `See`) to code modules
- If no simulation was needed (pure analytical), state that clearly

## ✅ Success Criteria
- **Pre-experiment and Post-experiment**: bullet-point list with **bold** criterion name followed by em-dash and expectation (e.g., `- **Decoupled baseline** — $\Delta\theta = 1/T_H$ exactly when $a_{xx}=0$ and $H_A=0$`). Post-experiment lists add `— PASS/FAIL` at the end of each item.
- Follow the criteria list with a short prose paragraph that summarizes what passed and what failed, provides brief reasoning, and suggests possible next steps to test.

## ⚖️ Physical Invariants / Analytical Bounds
Optional section documenting known analytical bounds, conservation laws, and invariants relevant to the simulation. Include explicit mathematical expressions and their domain of validity. This section appears only when conservation-law analysis is central to the report (e.g., when verifying that a noisy channel respects Pauli constraints or that a metrological bound is tight). When used, it serves as a reference for the assertions in the Validation block of the Numerical Simulation section. May be titled either **Physical Invariants** or **Analytical Bounds** depending on the nature of the content.

## 🏁 Conclusions
Final wrap-up section. Summarize what was learned, whether the hypothesis was supported, and the broader implications. Reference specific Results and any Open Questions that remain. This section must always appear last in the document.

Optional unsolved issues and future directions. Start the paragraph with `**Open items**` in bold — do not use a heading.

# Verification

- [ ] Filename follows `YYYY-MM-DD-{title}.md`
- [ ] All mandatory sections present in correct order
- [ ] Section header emojis placed correctly (`## 🧪 Title`, not `## Title 🧪`)
- [ ] No emojis outside section headers
- [ ] Inline callouts use bold text, not emojis
- [ ] Status columns use PASS/FAIL/PENDING/PARTIAL only
- [ ] No `\ket`, no bare pipe in math, breaklines only where necessary
- [ ] Physics scope checklist items covered in Theoretical Model
- [ ] No repo-specific filepaths in Numerical Simulation section

# Anti-patterns

- Writing reports without the emoji system.
- Using subsection headings inside Theoretical Model.
- Using `✅` or other emoji symbols in status tables.
- Adding decorative or ad-hoc emojis outside section headers.
- Skipping the Validation subsection in Numerical Simulation.
- Leaving Results section in `PENDING` status after experiments are complete.
- Using `|` inside LaTeX math without `\vert `.
