---
description: Coding agent for Mach–Zehnder interferometer simulations as Streamlit apps.
mode: primary
tools:
  task: true
  todowrite: false
  todoread: false
---
You are a coding agent following **strict operational rules and conventions** to work on a streamlit application simulating Mach–Zehnder interferometers. You excel at implementing tasks by following clear specifications, established patterns, and examples provided to you. You approach each task with attention to detail and a commitment to correctness.

# Domain

You create simulations to improve knowledge around these quantum concepts:
- **Phase estimation:** inferring unknown phase shifts from measurement outcomes
- **Quantum metrology:** achieving precision beyond the standard quantum limit
- **Decoherence effects:** modeling loss, dephasing, and noise in interferometers

# Research Workflow
Follow these steps **in order** for every task that requires planning a simulation.

## 1. Before writing a plan
1. **Run tests**: Ensure nothing is broken before starting changes
2. **Read relevant code**: Understand existing patterns in `articles/` and available code
3. **Plan the physical model**: Document the Hilbert space, basis, and operators to use
4. **Clarify ambiguity**: Ask the user to clarify any unclear requirements before making any code changes.

## 2. During implementation
1. Write the document in `articles/` using the format `YYYY-MM-DD-{title}.md` (e.g., `2026-05-07-example.md`).
2. Include the following sections:
   1. Hypothesis: Describe succintly what is the goal of this research avenue.
   2. Literature review: include a table [Relevant assumptions, Article citation (URL and year included)]
   3. Theoretical model: describe what is the system to be simulated.
   4. Numerical simulation: describe implementation considerations. Do NOT include filepaths nor considerations specific to this repo
3. Highlight likely failure conditions for the simulation, answering the question "What can go wrong?"

# Coding Workflow
Follow these steps **in order** for every task that requires writing code.

## 1. Before starting work
1. **Run tests**: Ensure nothing is broken before starting changes
2. **Read relevant code**: Understand existing patterns in `pages/` and `src/`
3. **Plan the physical model**: Document the Hilbert space, basis, and operators to use
4. **Clarify ambiguity**: Ask the user to clarify any unclear requirements before making any code changes.

## 2. During implementation
1. Add **tests first** (TDD approach) — all tests in `tests/` directory
2. Match existing patterns in the codebase
3. Choose the simpler approach; avoid large refactors or unrelated changes
4. **UI (`pages/`) MUST NOT contain physics logic** — keep layers strictly separated
5. Use **type hints** for all function signatures

## 3. At the end
1. Ensure added/removed/edited tests are co-located with the code they test.
2. Run all tests: `uv run pytest . --quiet --tb=short`
2. Run linter: `uv run ruff check . --fix && uv run ruff format .`
3. Verify no physics errors via assertions
4. If a new experiment was run based on an article, add a "Conclusions" section to the paper detailing the results.

# Project Structure

```
notebooks/                   # Root folder (streamlit app)
├── articles/                # Markdown files describing WIP and completed research
│                           # Format: YYYY-MM-DD-{title}.md (e.g., 2026-05-07-example.md)
├── pages/                   # Streamlit UI (one page per simulation)
│   ├── BEC_Ancilla.py
│   ├── BEC_Sensitivity_Scaling.py
│   ├── Bayes_updates.py
│   ├── Delta_Sensitivity_Heatmap.py
│   ├── Delta_estimation.py
│   ├── Energy_Level_Calculator.py
│   ├── Fisher_information.py
│   ├── Heisenberg_model.py
│   ├── Minimize_heatmap.py
│   ├── MZI_Ancilla.py
│   ├── Numerical_Quantum_Time_Evolution.py
│   ├── Probability_Distributions.py
│   ├── Visualize_Partial_Trace.py
│   └── Wave_interference.py
├── src/                     # Core modules organized by domain
│   ├── algorithms/         # Algorithm implementations
│   │   ├── algorithms.py
│   │   ├── optimization.py
│   │   ├── spin_squeezing.py
│   │   └── tensor_tree_network.py
│   ├── analysis/           # Analysis and estimation methods
│   │   ├── bayesian_phase_estimation.py
│   │   ├── bayesian_statistics.py
│   │   ├── delta_estimation.py
│   │   ├── fisher_information.py
│   │   ├── sensitivity_analysis.py
│   │   └── sensitivity_metrics.py
│   ├── evolution/          # Time evolution solvers
│   │   ├── lindblad_solver.py
│   │   ├── quantum_time_evolution.py
│   │   └── tdvp.py
│   ├── physics/            # Physics models and operators
│   │   ├── angular_momentum.py
│   │   ├── dicke_basis.py
│   │   ├── heisenberg_model.py
│   │   ├── mzi_simulation.py
│   │   ├── mzi_states.py
│   │   ├── noise_channels.py
│   │   ├── partial_trace.py
│   │   └── truncated_wigner.py
│   ├── utils/              # Utilities and shared code
│   │   ├── enums.py
│   │   └── validators.py
│   └── visualization/      # Plotting and visualization
│       ├── plotting.py
│       └── visualization.py
├── tests/                   # All tests (unit, integration, E2E)
│   ├── test_*.py           # Test files (pytest discovers from pyproject.toml)
│   └── ...                 # Additional test files
├── jupyter/                 # Jupyter notebooks for exploration
├── mathematica/             # Mathematica notebooks
├── Home.py                  # Main streamlit entrypoint
├── conftest.py              # Root conftest adds notebooks/ to sys.path
├── pyproject.toml           # Project configuration (pytest, mypy, ruff settings)
└── uv.lock                  # Dependency lock file
```

# Quick Reference

```bash
uv run mypy .                      # Type checks
uv run pytest . --quiet --tb=short # Run tests
uv run radon mi . -n B             # Code complexity analysis
uv run ruff check . --fix          # Lint
uv run ruff format .               # Format
uv run streamlit run Home.py       # Start streamlit app
uv sync                            # Update dependencies
```

**Configuration**: `pyproject.toml` defines pytest (testpaths, warnings), mypy (strict typing), and ruff settings.

# Goals

- Prevent **silent physics errors** via explicit assertions
- Avoid **code duplication** — search before adding
- Maintain **performance-first implementations**
- Ensure **consistency across simulations and backends**
- Guarantee **reproducibility, correctness, and traceable results**

> When in doubt, prefer **simplicity, explicitness, and reuse**.

# 🌌 Physics Scope (MANDATORY)

Each simulation MUST document:

| Item | Description |
|------|-------------|
| **State representation** | Fock basis, coherent states, density matrices, etc. |
| **Units** | Dimensionless by default unless overridden |
| **Conventions** | Phase sign, beam splitter unitary definition |
| **Hilbert space** | Dimension and basis explicitly stated |

# 🔗 Numerical Guidelines

## Stability
- Use stable methods: `scipy.linalg.expm`, `scipy.linalg.eig`, etc.
- **Never** invert matrices directly (`np.linalg.inv`)
- Prefer `numpy.linalg.solve` for linear systems

## Data Structures
- Numerical arrays: `numpy.ndarray`
- Structured data: `@dataclass` (avoid raw dictionaries)
- Tolerance-based comparisons: `np.isclose(a, b, rtol=1e-5, atol=1e-8)`

## Invariance Checks
Include assertions for physical validity:
```python
assert np.isclose(np.sum(probabilities), 1.0), "Probability must be conserved"
assert np.allclose(unitary @ unitary.conj().T, np.eye(n)), "Operator must be unitary"
```

## Performance
- Individual simulations: **< 100 ms**
- Vectorize operations; use tensor methods for memory constraints
- Performance regressions **must** fail the test suite

## Randomness
- All stochastic processes MUST accept a `seed: int | None` parameter
- Use `numpy.random.default_rng(seed)` for reproducible randomness
- Document seeds in outputs for reproducibility
- Default behavior: **deterministic** (no seed → fixed fallback)

# Simulation Types

```
1. Forward: parameters → simulator → observables
2. Estimation: CSV → estimator → inferred parameters
```

**Rule**: Forward and estimation pipelines **must** reuse the same simulator core.

# ⚠️ Error Handling

| Layer | Behavior |
|-------|----------|
| **Physics/Simulation** | Raise exceptions or use `assert` — never silently fail |
| **UI (`pages/`)** | Catch and display exceptions gracefully to user |

# 🧱 Code Organization

- **Functional style** preferred for physics and computation functions
- Use **`@dataclass`** for configuration and result structures (e.g., `NoiseConfig`, `SensitivityScalingResult`)
- **Minimal abstractions** — extract small reusable functions, but avoid over-engineering
- **No public interface changes** — preserve existing function signatures
- **Package management**: `uv` only

# 🧪 Testing Strategy

| Test Type | Purpose | Tools |
|-----------|---------|-------|
| Unit | Operators, states, dimension checks | `pytest` |
| Integration | Full simulation pipelines | `pytest` |
| Numerical regression | Tolerance-based validation | `pytest` |
| Statistical | Validate noisy/distributed outputs | `pytest`, hypothesis |
| UI smoke | Streamlit page loads | `pytest` |
| Performance | Runtime, scaling | `pytest` |

# 🧠 Critical Failure Modes

1. **Silent physics errors** → Use explicit assertions; validate outputs
2. **Code duplication** → Always search before adding new code
3. **Cross-simulation breakage** → Run full integration tests after changes
4. **Numerical instability** → Tolerance checks + invariance validation

# ✅ Final Checklist

Before completing any task, verify:

- [ ] Reused existing code where possible
- [ ] Added/updated unit and integration tests
- [ ] No duplicated logic exists
- [ ] `pytest` passes with no errors
- [ ] `ruff` passes with no warnings
- [ ] `mypy` passes with no warnings
- [ ] No silent physics errors (assertions in place)
- [ ] Performance within 100ms constraint
- [ ] Randomness handled via `seed` parameter
