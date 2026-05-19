---
name: code-architecture
description: MUST be used when organizing code, creating new modules, designing Streamlit pages, or understanding the project structure and conventions. Covers functional style, dataclass usage, import patterns, and page conventions.
---

# Purpose

Define the code organization principles, project structure, and Streamlit page conventions that ensure consistency across the codebase.

# Rules

## Code Style

- **Functional style** preferred for physics and computation functions.
- Use **`@dataclass`** for configuration and result structures (e.g., `NoiseConfig`, `SensitivityScalingResult`).
- **Serialization completeness**: Every result dataclass with `to_dataframe()` must serialize all input parameters alongside computed results — theta_value, T_H, SQL, bounds, and any other configuration. The CSV file must be fully self-describing.
- **Fail-fast deserialization**: Every `from_csv()` must require all expected columns. No silent fallback defaults for metadata fields. Raise a `ValueError` with missing columns listed.
- **Minimal abstractions** — extract small reusable functions, but avoid over-engineering.
- **No public interface changes** — preserve existing function signatures.
- **Package management**: `uv` only.

## Project Structure

```
├── .streamlit/             # Streamlit configuration directory
├── jupyter/                # Jupyter notebooks for exploration
├── mathematica/            # Mathematica notebooks
├── pages/                  # Streamlit UI (one page per simulation)
├── reports/                # Markdown files (YYYY-MM-DD-{title}.md)
├── revise/                 # GoCard revision files
├── src/                    # Core modules organized by domain
│   ├── algorithms/         # Algorithm implementations
│   ├── analysis/           # Analysis and estimation methods
│   ├── evolution/          # Time evolution solvers
│   ├── physics/            # Physics models and operators
│   ├── utils/              # Utilities and shared code
│   └── visualization/      # Plotting and visualization
├── tests/                  # Integration & E2E tests
│   ├── test_*.py           # Test files
│   └── ...
├── Home.py                 # Main streamlit entrypoint
├── opencode.json           # OpenCode agent configuration
└── pyproject.toml          # Project configuration (pytest, mypy, ruff)
```

Key notes:
- Unit tests live alongside source modules in `src/` subdirectories.
- Integration and E2E tests live in `tests/`.
- The root `conftest.py` inserts the project root into `sys.path` — this enables absolute imports like `from src.physics.mzi_states import ...` in both `pages/` and `tests/`.
- The `src/__init__.py` package docstring lists simulation capabilities.

## Streamlit Page Conventions

Each page in `pages/` follows:

1. **Colour scheme** — use semantic dividers as documented in README.md (blue=primary, gray=setup, green=results, orange=data, red=summary, violet=special).
2. **No physics logic** — all computation lives in `src/` modules; pages import from `src.*` and only orchestrate UI rendering.
3. **Imports** — use `from src.physics.module import ...` (the project root is on `sys.path` via `conftest.py`).
4. **Error handling** — wrap simulation calls in `try/except` and display errors via `st.error`; never let exceptions propagate unhandled.
5. **State** — use `st.session_state` for persistent UI state (widget values, cached results); use function return values for transient simulation results.
6. **Seeds** — accept an optional user-provided seed; fall back to a deterministic default when none is given.
7. **Page naming** — use `Snake_Case.py`; Streamlit converts underscores to spaces in the sidebar navigation.
8. **Page config** — every page calls `st.set_page_config(page_title=..., page_icon=..., layout="wide")` as its first Streamlit call.
9. **Subpackage docstring** — `pages/__init__.py` contains the subpackage docstring; refer to it for an overview of available simulation pages.

# Verification

- [ ] No physics logic in Streamlit pages (all in `src/`).
- [ ] Imports use absolute `from src.*` pattern.
- [ ] Simulation calls wrapped in try/except with `st.error`.
- [ ] Pages use `Snake_Case.py` naming.
- [ ] Each page calls `st.set_page_config` as its first Streamlit call.
- [ ] `@dataclass` used for structured data, not raw dictionaries.
- [ ] No public interface changes to existing functions.
- [ ] `uv` used for package management, not pip or conda.
- [ ] `to_dataframe()` includes all input parameters, not just computed results.
- [ ] `from_csv()` requires all expected columns — no silent fallback defaults for metadata.

# Anti-patterns

- Putting physics computation directly in Streamlit pages.
- Using relative imports instead of `from src.*`.
- Letting exceptions propagate unhandled in page code.
- Using raw dictionaries for structured results instead of `@dataclass`.
- Adding large refactors unrelated to the task.
- Changing function signatures in existing modules.
- Using pip or conda instead of `uv`.
- Omitting input parameters from `to_dataframe()` — every CSV should be self-describing.
- Adding silent fallback defaults in `from_csv()` for metadata columns (theta_value, sql, T_H, slice_type).
