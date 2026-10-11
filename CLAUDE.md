# notebooks (collapsedwave)

Quantum physics simulations for MZI metrology, ancilla-driven systems, and quantum information processing. Python 3.14, managed with `uv`. This repo is a **frozen archive** — `interferometry` is its active successor. Read as source material but coordinate with the user before modifying.

## Quality Gates

Run all before committing:
```bash
uv run pytest -m "not slow"
uv run ruff check .
uv run mypy .
uv run tach check
```

- Coverage is configured in pyproject.toml (fail_under = 85) but is not a gate; run `uv run coverage run -m pytest -q && uv run coverage report` only on request.
- Ruff: ALL rules with physics-domain exclusions.
- mypy: strict (disallow_untyped_calls, disallow_untyped_defs, disallow_incomplete_defs).
- tach: enforces layered architecture (utils -> algorithms -> evolution -> physics -> analysis -> visualization).

## Structure

- `src/` — core physics library (layered: utils, algorithms, evolution, physics, analysis, visualization)
- `pages/` — Streamlit application pages
- `jupyter/` — Jupyter notebooks by category
- `reports/` — experiment reports/analyses
- `tests/` — test suite

## Conventions

- Issue tracking: `bd` (beads). Run from `~/Git`.
- Package manager: `uv` only.
- Layered imports: a layer imports only from itself or lower.
- Reports are isolated experiment modules; they cannot be imported by src/pages.
- No commits on main. Branch or worktree only.
