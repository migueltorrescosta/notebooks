---
name: numerical-guidelines
description: MUST be used when implementing physics simulations, constructing operators, performing numerical computation, or analyzing simulation data. Covers stability, data structures, invariance checks, performance, and randomness.
---

# Purpose

Ensure numerical correctness, stability, and reproducibility across all physics simulations and data analysis in the project.

# Rules

## Stability

- Use stable methods: `scipy.linalg.expm`, `scipy.linalg.eig`, etc.
- **Never** invert matrices directly (`np.linalg.inv`).
- Prefer `numpy.linalg.solve` for linear systems.

## Data Structures

- Numerical arrays: `numpy.ndarray`.
- Structured data: `@dataclass` (avoid raw dictionaries).
- Tolerance-based comparisons: `np.isclose(a, b, rtol=1e-5, atol=1e-8)`.
- **Serialization completeness**: Every `to_dataframe()` method must include all input parameters alongside computed results, making each CSV self-describing. Omitted metadata is permanently lost.
- **Deserialization strictness**: Every `from_csv()` method must require all expected columns. Do not use `.get()` or conditional fallbacks for required metadata fields — fail fast with a clear error message listing the missing columns.

## Invariance Checks

Include assertions for physical validity:

```python
assert np.isclose(np.sum(probabilities), 1.0), "Probability must be conserved"
assert np.allclose(unitary @ unitary.conj().T, np.eye(n)), "Operator must be unitary"
```

## Error Handling

- Physics/simulation code must raise exceptions or use `assert` — never silently fail.
- **CSV deserialization**: In `from_csv()` methods, require every expected column. No silent fallbacks to defaults for metadata. Raise a `ValueError` listing the missing columns and pointing to regeneration as the fix.
- **Fail-fast principle**: A silent default today is a subtle data corruption tomorrow. When restoring data from a file, defaulting `theta_value` to `1.0` or `sql` to `0.1` masks the actual data loss. Always raise instead.
- UI (`pages/`) code must catch and display exceptions gracefully to the user via `st.error`.

## Performance

- Individual simulations: **< 100 ms**.
- Vectorize operations; use tensor methods for memory constraints.
- Performance regressions **must** fail the test suite.

## Randomness

- All stochastic processes MUST accept a `seed: int | None` parameter.
- Use `numpy.random.default_rng(seed)` for reproducible randomness.
- Document seeds in outputs for reproducibility.
- Default behavior: **deterministic** (no seed → fixed fallback).

# Verification

- [ ] No direct matrix inversions (`np.linalg.inv` used?).
- [ ] All stochastic functions accept a `seed` parameter.
- [ ] Physical invariants checked with assertions (normalisation, unitarity).
- [ ] Error handling separates physics (assert/raise) from UI (try/except/st.error).
- [ ] Simulations run within 100 ms performance budget.
- [ ] `@dataclass` used for structured results, not raw dictionaries.
- [ ] `to_dataframe()` includes all input parameters, not just computed results.
- [ ] `from_csv()` requires all expected columns — no silent fallback defaults for metadata.
- [ ] Legacy CSV handling: fail fast guiding regeneration, not silently serving wrong data.

# Anti-patterns

- Using `np.linalg.inv` instead of `np.linalg.solve`.
- Silently swallowing errors in physics code.
- Using raw dictionaries for simulation results instead of `@dataclass`.
- Hardcoding random seeds instead of accepting parameters.
- Using explicit loops instead of vectorized operations where possible.
- Forgetting to check unitarity of constructed operators.
- Using implicit random state instead of `numpy.random.default_rng(seed)`.
- Adding silent fallback defaults in `from_csv()` for metadata columns (theta_value, sql, T_H, slice_type).
- Omitting input parameters from `to_dataframe()` — the CSV should be self-describing.
- Writing a migration script for legacy CSVs instead of failing fast and guiding regeneration.
