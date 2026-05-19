---
name: testing-standards
description: MUST be used when writing new tests, reviewing test coverage, debugging test failures, or designing test infrastructure. Covers naming conventions, parametrization, error testing, and noise reduction patterns.
---

# Purpose

Establish consistent, maintainable testing conventions across the project, ensuring tests are readable, informative, and easy to debug when they fail.

# Rules

## Test Types

| Test Type | Purpose | Tools |
|-----------|---------|-------|
| Unit | Operators, states, dimension checks | pytest |
| Integration | Full simulation pipelines | pytest |
| Numerical regression | Tolerance-based validation | pytest |
| Statistical | Validate noisy/distributed outputs | pytest |
| UI smoke | Streamlit page loads | pytest |
| Performance | Runtime, scaling | pytest |

## Naming Conventions

- Use domain descriptions, not implementation steps: `test_twin_fock_normalized`, not `test_normalized_twin_fock`.
- Drop `should_`: `test_evolved_state_remains_normalized`, not `test_evolved_state_should_remain_normalized`.
- The `given_`/`then_` delimiters are optional; prefer flat descriptive names.
- Keep variable names consistent across cases (e.g., always `qfi_computed`, `qfi_expected`).

## Helper Conventions

- Prefix fixtures with `_make_`.
- Prefix preconditions with `_given_`.

## Noise Reduction

- Drop docstrings when class/method names are self-documenting.
- Drop redundant assertion messages — `assert X`, not `assert X, "Expected X"` (pytest prints the failing expression).
- Remove `if __name__ == "__main__": pytest.main(...)` guards.
- Remove module-level section comment blocks (test classes are section boundaries in pytest output).

## Metadata Roundtrip Testing

- Every result dataclass with CSV roundtrip (`save_csv` / `from_csv`) must have a test verifying that all metadata fields survive the roundtrip (theta_value, T_H, sql, slice_type, etc.). Do not restrict roundtrip tests to computed arrays only.
- When `from_csv` is changed to fail fast (removing silent defaults), add a test that loading a CSV missing required metadata columns raises the expected error.

## Error Testing

- Use the 2-line `with pytest.raises(ValueError):` form.
- Never use `try`/`except` in tests.

## Parametrization

- Use `@pytest.mark.parametrize` instead of manual `for` loops (enables per-value failure reporting).
- Add `ids=` for non-trivial values.
- Keep variable names consistent across cases (e.g., always `qfi_computed`, `qfi_expected`).

# Verification

- [ ] Test names follow domain-description convention.
- [ ] No `should_` prefixes in test names.
- [ ] No try/except blocks in tests.
- [ ] No `if __name__ == "__main__"` guards in test files.
- [ ] Parametrized tests use `@pytest.mark.parametrize` with `ids=` where appropriate.
- [ ] Assertion messages only present when necessary for clarity.

# Anti-patterns

- Writing tests with implementation-step names like `test_calls_function_x_then_function_y`.
- Using `for` loops instead of `@pytest.mark.parametrize` (hides which value failed).
- Writing `try`/`except` in tests instead of `pytest.raises`.
- Leaving docstrings on tests when the name already explains the purpose.
- Adding `if __name__ == "__main__"` blocks to test files.
- Writing redundant assertion messages like `assert X, "Expected X"`.
