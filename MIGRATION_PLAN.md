# Migration Plan: Horizontal Framework → Vertical Research Environment

> **Target:** A repository where a coding agent can implement a new research direction by finding one nearby vertical, reusing stable primitives, and remaining inside a small localized area of the codebase.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current State Analysis](#2-current-state-analysis)
3. [Target Architecture](#3-target-architecture)
4. [Current Horizontal Domains](#4-current-horizontal-domains)
5. [Natural Research Verticals](#5-natural-research-verticals)
6. [Stable Reusable Primitives (Platform Layer)](#6-stable-reusable-primitives-platform-layer)
7. [Accidental Abstractions](#7-accidental-abstractions)
8. [Areas of Poor Locality](#8-areas-of-poor-locality)
9. [Migration Strategy](#9-migration-strategy)
10. [New Extension Pattern](#10-new-extension-pattern)
11. [Appendix: Vertical Fact Sheets](#11-appendix-vertical-fact-sheets)

---

## 1. Executive Summary

### Problem

The repository is organized as a **shared scientific framework** (`src/`). Every research direction imports from the same horizontal layers (`physics/`, `analysis/`, `algorithms/`, `evolution/`, `visualization/`, `utils/`). This means:

- **High traversal radius:** Implementing a single vertical (e.g., ancilla-assisted metrology) requires reading 5+ horizontal modules.
- **Poor locality:** A vertical's simulation code sits in `src/physics/`, its analysis in `src/analysis/`, its visualization in `src/visualization/`, its page in `pages/`, its tests in `tests/` + `src/*/test_*`, and its report in `reports/`. Seven locations for one vertical.
- **Accidental abstractions:** The horizontal layers force shared interfaces that may not exist in the science. `src/analysis/` mixes foundational primitives (Fisher information), experiment-specific orchestrators (scaling survey), and massive single-vertical modules (ancilla_optimization at 2278 lines, weighted_joint_measurement at 2946 lines).
- **Implicit orchestration:** Orchestrators like `scaling_survey.py` reach across physics, analysis, and evolution layers, making the control flow hard to follow.

### Solution

Split the repository into two layers:

1. **`platform/`** — stable, well-tested scientific primitives with no knowledge of specific experiments.
2. **`research/`** — self-contained verticals, each owning its simulation, analysis, orchestration, page, visualization, tests, reports, and data.

Each vertical is a **bounded context**. A vertical may duplicate a calculation that another vertical uses — this is explicitly preferred over shared orchestration infrastructure.

---

## 2. Current State Analysis

### Current Directory Layout

```
src/
├── algorithms/          ← Mixed: spin squeezing (physics) + optimization (numerical) + TTN (specialized)
├── analysis/            ← Largest layer: mixes primitives (Fisher info) with verticals (ancilla_optimization)
│                          Also contains orchestrators (scaling_survey) that reach across 4+ layers
├── evolution/           ← Lindblad solver, Schrodinger solver, TDVP
├── physics/             ← Core MZI models, plus specialized models (hybrid, cavity, distributed)
├── utils/               ← Enums + validators (creates circular dependency with analysis/)
└── visualization/       ← Generic (plot_array) mixed with vertical-specific (ancilla_drive_plots, report_figures)

pages/                   ← 21 thin UI wrappers; each imports from 3-5 src/ subpackages
tests/                   ← Integration tests separated from the code they test
reports/                 ← 12 reports + figures + data; mixed across verticals
```

### Import Graph (Simplified)

```
pages/*.py
  ├── src.analysis.*        (the main consumer)
  ├── src.physics.*         (the models)
  ├── src.algorithms.*      (squeezing, optimization)
  ├── src.evolution.*       (lindblad, time evolution)
  ├── src.visualization.*   (plotting)
  └── src.utils.*           (enums, validators)

src/analysis/*.py
  ├── src.analysis.*        (intra-analysis imports)
  ├── src.physics.*         (dicke_basis, mzi_simulation, mzi_states, hybrid_*)
  ├── src.algorithms.*      (spin_squeezing)
  └── src.evolution.*       (lindblad_solver)

src/physics/*.py
  ├── src.physics.*         (intra-physics imports)
  └── src.utils.*           (validators)

src/utils/*.py
  └── src.analysis.*        (lazy import — circular dependency!)
```

### Key Metrics

| Metric | Value |
|--------|-------|
| Lines in `src/` | ~18,000+ |
| Lines in `src/analysis/` alone | ~9,000+ (~50% of all source) |
| Largest single file | `weighted_joint_measurement.py` (2946 lines) |
| 2nd largest | `ancilla_optimization.py` (2278 lines) |
| 3rd largest | `scaling_survey.py` (1319 lines) |
| Pages importing from 4+ subpackages | ~15 of 21 |
| Circular imports (lazy-loaded) | 1 (validators ↔ sensitivity_analysis) |
| Locations per vertical (avg) | ~7 (physics + analysis + visualization + page + tests/ + reports + data) |

---

## 3. Target Architecture

```
repository-root/
│
├── platform/                          ← STABLE LAYER
│   ├── dicke_basis.py                 # Collective spin operators
│   ├── fisher_information.py          # QFI, CFI, Cramér-Rao bounds
│   ├── lindblad_solver.py             # Lindblad master equation solver
│   ├── mzi_core.py                    # Core MZI operators (beam splitters, phase shifts)
│   ├── noise_channels.py              # Lindblad noise channel definitions
│   ├── partial_trace.py               # Bipartite partial trace
│   ├── scaling_fit.py                 # Log-log regression for scaling exponents
│   ├── spin_squeezing.py              # CSS, OAT, squeezing parameter
│   ├── state_factory.py               # Input state creation (twin_fock, noon, coherent, etc.)
│   ├── states.py                      # Specific state definitions in Dicke basis
│   ├── wigner.py                      # Wigner function computation
│   ├── enums.py                       # Shared enumerations
│   ├── validators.py                  # Core validation utilities
│   ├── plot_array.py                  # Generic heatmap utility (only)
│   └── tests/                         # Unit tests for platform primitives
│
├── research/                          ← RESEARCH LAYER (self-contained verticals)
│   ├── ancilla_basic/                 # Ancilla-assisted metrology (basic)
│   ├── ancilla_drive/                 # Driven-ancilla metrology
│   ├── ancilla_comparison/            # 1+1 vs 2 probe comparison
│   ├── ancilla_phase_modulated/       # Phase-modulated ancilla drive
│   ├── weighted_joint_measurement/    # N,M generalization of joint measurement
│   ├── scaling_survey/                # Unified interferometry scaling survey
│   ├── bayesian_phase_estimation/     # Bayesian inference for phase estimation
│   ├── delta_estimation/              # Delta estimation via system-ancilla
│   ├── bec_ancilla/                   # BEC ancilla-enhanced metrology
│   ├── cavity_mzi/                    # Cavity-enhanced interferometry
│   ├── distributed_mzi/              # Distributed sensor array interferometer
│   ├── hybrid_system/                 # Hybrid oscillator-spin metrology
│   ├── heisenberg_model/              # 1D Heisenberg spin chain
│   ├── single_particle_scaling/       # Single-particle MZI holding-time scaling
│   ├── high_order_squeezing/          # High-order non-Gaussian squeezing
│   ├── kerr_mzi/                      # Kerr nonlinearity MZI
│   ├── dynamical_decoupling/          # DD pulse sequences & filter functions
│   ├── thermal_langevin/              # Thermal Langevin noise model
│   ├── tilt_to_length_noise/          # TTL coupling noise model
│   ├── truncated_wigner/              # Truncated Wigner approximation
│   ├── pseudomode_system/             # Non-Markovian pseudomode model
│   ├── weak_value_mzi/                # Weak-value amplification
│   └── energy_level_calculator/       # 1D Schrödinger energy levels
│
├── Home.py                            ← Streamlit dashboard (discovers research/*/page.py)
│
├── pages/                             ← SHIMS (deprecated after migration)
├── tests/                             ← SHIMS (deprecated after migration)
├── src/                               ← SHIMS (deprecated after migration)
└── reports/                           ← SHIMS (deprecated after migration)
```

### Internal Structure of a Research Vertical

Every research vertical follows this template:

```
research/<vertical_name>/
├── __init__.py              # Exports public API
├── experiment.py            # Experiment specification: parameters, configs, dataclasses
├── simulation.py            # Physics simulations specific to this experiment
├── analysis.py              # Analysis methods, sensitivity computations
├── orchestration.py         # Parameter sweeps, optimization runs, batch execution
├── visualization.py         # Specialized plots for this vertical
├── page.py                  # Streamlit page (if interactive UI needed)
├── tests/
│   ├── __init__.py
│   ├── test_simulation.py
│   └── test_analysis.py
├── reports/
│   ├── 2026-05-12-finding.md
│   └── figures/
│       └── sensitivity_plot.svg
└── data/
    ├── run_20260512.csv
    └── metadata.json
```

> **Note:** Not every file is required. A minimal vertical may have only `simulation.py` and `tests/`. Files are created when the need arises, not pre-emptively.

---

## 4. Current Horizontal Domains

The current `src/` structure imposes six horizontal layers. Each one conflates primitives with vertical-specific code:

### 4.1 `src/physics/` — Physics Models (largest layer)

| File | Classification | Notes |
|------|---------------|-------|
| `dicke_basis.py` | **Platform** | Fundamental operators, used everywhere |
| `mzi_simulation.py` | **Mostly platform** | Core MZI operators (beam_splitter, phase_shift) are platform; noon_state etc. could go either way |
| `mzi_states.py` | **Platform** | State factory used by 5+ verticals |
| `states.py` | **Platform** | Twin-Fock, NOON in Dicke basis |
| `noise_channels.py` | **Platform** | Noise config used by multiple verticals |
| `partial_trace.py` | **Platform** | Utility operation |
| `wigner.py` | **Platform** | Wigner functions (general method) |
| `truncated_wigner.py` | **Platform** | TWA method |
| `bec_ancilla_system.py` | **Vertical** | BEC-specific simulation |
| `cavity_mzi.py` | **Vertical** | Self-contained model |
| `distributed_mzi.py` | **Vertical** | Self-contained model |
| `dynamical_decoupling.py` | **Vertical** | Self-contained protocol |
| `heisenberg_model.py` | **Vertical** | Self-contained model |
| `hybrid_system.py` | **Vertical** | Hybrid oscillator-spin system |
| `hybrid_mzi.py` | **Vertical** | MZI embedding for hybrid system |
| `hybrid_lindblad.py` | **Vertical** | Lindblad for hybrid system |
| `kerr_mzi.py` | **Vertical** | Self-contained model |
| `mzi_lindblad.py` | **Vertical** | Noisy MZI solver (depends on mzi_simulation) |
| `pseudomode_system.py` | **Vertical** | Self-contained model |
| `single_particle_mzi_scaling.py` | **Vertical** | Self-contained model |
| `thermal_langevin.py` | **Vertical** | Self-contained noise model |
| `tilt_to_length_noise.py` | **Vertical** | Self-contained noise model |
| `weak_value_mzi.py` | **Vertical** | Self-contained protocol |

### 4.2 `src/analysis/` — Estimation Theory (most conflated)

| File | Classification | Notes |
|------|---------------|-------|
| `fisher_information.py` | **Platform** | Fundamental estimation theory |
| `scaling_fit.py` | **Platform** | Utility (log-log regression) |
| `bayesian_statistics.py` | **Platform** | General Bayesian utilities |
| `ancilla_optimization.py` | **Vertical** | 2278 lines, single vertical |
| `ancilla_drive_metrology.py` | **Vertical** | 1162 lines, single vertical |
| `ancilla_drive_phase_modulated.py` | **Vertical** | Single vertical |
| `weighted_joint_measurement.py` | **Vertical** | 2946 lines, single vertical |
| `scaling_survey.py` | **Vertical** | 1319 lines, orchestrator |
| `ancilla_comparison.py` | **Vertical** | Single comparison study |
| `delta_estimation.py` | **Vertical** | Single protocol |
| `sensitivity_analysis.py` | **Vertical** | Heatmap / reduced model analysis |
| `sensitivity_metrics.py` | **Vertical** | Three-method comparison |
| `bayesian_phase_estimation.py` | **Vertical** | Bayesian MZI protocol |
| `bec_sensitivity.py` | **Vertical** | BEC-specific scaling |

### 4.3 `src/visualization/` — Plotting

| File | Classification | Notes |
|------|---------------|-------|
| `plotting.py` | **Platform** | `plot_array` — generic heatmap utility |
| `visualization.py` | **Vertical** | General viz (likely unused or vertical-specific) |
| `ancilla_drive_plots.py` | **Vertical** | Specific to ancilla drive |
| `ancilla_plots.py` | **Vertical** | Specific to ancilla system |
| `report_figures.py` | **Vertical** | Report figure generation (spans verticals) |

### 4.4 `src/algorithms/` — Numerical Methods

| File | Classification | Notes |
|------|---------------|-------|
| `spin_squeezing.py` | **Platform** | Used by multiple verticals |
| `optimization.py` | **Platform** | General optimization utilities |
| `tensor_tree_network.py` | **Vertical** | Specialized algorithm |
| `_mcmc_archive.py` | **Archive** | Legacy code, not actively used |

### 4.5 `src/evolution/` — Time Evolution

| File | Classification | Notes |
|------|---------------|-------|
| `lindblad_solver.py` | **Platform** | Core Lindblad solver |
| `quantum_time_evolution.py` | **Vertical** | 1D Schrödinger solver (specialized) |
| `tdvp.py` | **Vertical** | TDVP for spin chains |

### 4.6 `src/utils/` — Shared Infrastructure

| File | Classification | Notes |
|------|---------------|-------|
| `enums.py` | **Platform** | Shared enumerations |
| `validators.py` | **Platform** | Core validators (but creates circular dep) |

---

## 5. Natural Research Verticals

The following self-contained research directions exist in the codebase. Each one should become a directory under `research/`.

| # | Vertical Name | Current Location(s) | Lines | Page(s) | Reports |
|---|--------------|--------------------|-------|---------|---------|
| 1 | **ancilla_basic** | `src/analysis/ancilla_optimization.py` (2278) | 3200+ | `Ancilla_Metrology_Optimization.py` | 2026-05-12, 2026-05-15 |
| 2 | **ancilla_drive** | `src/analysis/ancilla_drive_metrology.py` (1162), `src/visualization/ancilla_drive_plots.py` | 1400+ | `Ancilla_Drive_Metrology.py` | 2026-05-18 |
| 3 | **ancilla_phase_modulated** | `src/analysis/ancilla_drive_phase_modulated.py` | 500+ | (none standalone) | 2026-05-19 |
| 4 | **ancilla_comparison** | `src/analysis/ancilla_comparison.py` | 400+ | `Ancilla_vs_System_Comparison.py` | 2026-05-11 |
| 5 | **weighted_joint_measurement** | `src/analysis/weighted_joint_measurement.py` (2946) | 2946 | (via Ancilla_Metrology_Optimization) | 2026-05-18 |
| 6 | **bayesian_phase_estimation** | `src/analysis/bayesian_phase_estimation.py` + `bayesian_statistics.py` | 800+ | `Bayes_updates.py` | — |
| 7 | **delta_estimation** | `src/analysis/delta_estimation.py` + `sensitivity_analysis.py` + `sensitivity_metrics.py` | 1200+ | `Delta_estimation.py`, `Delta_Sensitivity_Heatmap.py` | — |
| 8 | **bec_ancilla** | `src/physics/bec_ancilla_system.py` + `src/analysis/bec_sensitivity.py` | 700+ | `BEC_Ancilla.py`, `BEC_Sensitivity_Scaling.py` | — |
| 9 | **scaling_survey** | `src/analysis/scaling_survey.py` (1319) | 1319 | `Interferometry_Scaling_Survey.py` | 3 reports |
| 10 | **cavity_mzi** | `src/physics/cavity_mzi.py` | 300+ | (none standalone) | — |
| 11 | **distributed_mzi** | `src/physics/distributed_mzi.py` | 300+ | (none standalone) | — |
| 12 | **hybrid_system** | `src/physics/hybrid_system.py` + `hybrid_mzi.py` + `hybrid_lindblad.py` | 800+ | `High_Order_Squeezing.py` | 2026-05-07 |
| 13 | **heisenberg_model** | `src/physics/heisenberg_model.py` + `src/evolution/tdvp.py` | 400+ | `Heisenberg_model.py` | — |
| 14 | **single_particle_scaling** | `src/physics/single_particle_mzi_scaling.py` | 200+ | `Single_Particle_MZI_Scaling.py` | 2026-05-12 |
| 15 | **kerr_mzi** | `src/physics/kerr_mzi.py` | 200+ | (none standalone) | — |
| 16 | **dynamical_decoupling** | `src/physics/dynamical_decoupling.py` | 300+ | (none standalone) | — |
| 17 | **thermal_langevin** | `src/physics/thermal_langevin.py` | 200+ | (none standalone) | — |
| 18 | **tilt_to_length_noise** | `src/physics/tilt_to_length_noise.py` | 200+ | (none standalone) | — |
| 19 | **truncated_wigner** | `src/physics/truncated_wigner.py` | 200+ | (none standalone) | — |
| 20 | **pseudomode_system** | `src/physics/pseudomode_system.py` | 200+ | (none standalone) | — |
| 21 | **weak_value_mzi** | `src/physics/weak_value_mzi.py` | 200+ | (none standalone) | — |
| 22 | **energy_level_calculator** | `src/evolution/quantum_time_evolution.py` | 400+ | `Energy_Level_Calculator.py`, `Numerical_Quantum_Time_Evolution.py` | — |

---

## 6. Stable Reusable Primitives (Platform Layer)

These modules are truly foundational and should live in `platform/`. They have no knowledge of specific experiments.

| Module | Source | Rationale |
|--------|--------|-----------|
| `dicke_basis.py` | `src/physics/dicke_basis.py` | Collective spin operators (Jx, Jy, Jz). Used by every vertical. |
| `mzi_core.py` | `src/physics/mzi_simulation.py` (subset) | Core operators: `beam_splitter_unitary`, `phase_shift_unitary`, `create_system_operators`. Not the full vertical-specific simulation logic. |
| `state_factory.py` | `src/physics/mzi_states.py` | `input_state_factory`, `two_mode_jz_operator`. Centralized state creation used by 5+ verticals. |
| `states.py` | `src/physics/states.py` | `twin_fock_dicke`, `noon_state_dicke`. State definitions in Dicke basis. |
| `fisher_information.py` | `src/analysis/fisher_information.py` | `quantum_fisher_information_dm`, `classical_fisher_information`. Pure estimation theory. |
| `lindblad_solver.py` | `src/evolution/lindblad_solver.py` | `LindbladConfig`, `evolve_lindblad`. Master equation solver. |
| `noise_channels.py` | `src/physics/noise_channels.py` | Noise channel definitions and Lindblad operators. |
| `partial_trace.py` | `src/physics/partial_trace.py` | Partial trace operations. |
| `spin_squeezing.py` | `src/algorithms/spin_squeezing.py` | `coherent_spin_state`, `generate_squeezed_state`, `optimal_squeezing_time`. |
| `wigner.py` | `src/physics/wigner.py` | Wigner function computation utilities. |
| `scaling_fit.py` | `src/analysis/scaling_fit.py` | `fit_scaling_exponent`. Log-log linear regression. |
| `enums.py` | `src/utils/enums.py` | Shared enumerations (`WavePacket`, `OperatorBasis`, etc.). |
| `validators.py` | `src/utils/validators.py` | Core validation functions (subset that doesn't create circular deps — see §7.3). |
| `plot_array.py` | `src/visualization/plotting.py` | `plot_array` — generic interactive heatmap. The only truly generic visualization utility. |

### Platform Design Principles

1. **Zero knowledge of specific experiments.** A platform module should never import from a vertical.
2. **Pure functions and stable interfaces.** Platform primitives accept numpy arrays / qutip Qobjs and return numpy arrays / qutip Qobjs.
3. **Own tests within `platform/tests/`.** These are the "trusted core" tests.
4. **No orchestration.** Platform modules never contain parameter sweeps, optimization loops, or batch execution.
5. **No Streamlit imports.** Platform modules never import `streamlit`.
6. **No report generation.** Platform modules never write files or produce reports.

---

## 7. Accidental Abstractions

These are abstractions in the current code that don't correspond to natural research boundaries and actively harm navigability.

### 7.1 `src/analysis/` as a Catch-all Layer

**The problem:** `src/analysis/` contains:
- Foundational primitives (Fisher information, 485 lines) — belongs in platform
- Massive single-vertical modules (ancilla_optimization, 2278 lines) — belongs in a vertical
- Experiment-specific orchestrators (scaling_survey, 1319 lines) — belongs in a vertical
- Generic utilities (scaling_fit, bayesian_statistics) — belongs in platform

**The fix:** Decompose into platform primitives + vertical-specific modules. The directory name "analysis" is too generic — everything in the repo is "analysis" in some sense.

### 7.2 `src/visualization/` Mixing Generic and Specific

**The problem:** `plot_array()` is a genuinely reusable heatmap utility. But it sits alongside `ancilla_drive_plots.py` (vertical-specific) and `report_figures.py` (cross-vertical).

**The fix:** Move `plot_array()` to `platform/plot_array.py`. Move vertical-specific visualization modules into their respective verticals.

### 7.3 `src/utils/validators.py` and the Circular Dependency

**The problem:** `validators.py` imports from `src.analysis.sensitivity_analysis` (via a lazy import), creating a circular dependency. This means the "utils" layer depends on the "analysis" layer it's supposed to serve.

```
src/utils/validators.py  ──lazy import──>  src/analysis/sensitivity_analysis.py
     ^                                              │
     └──────────────────────────────────────────────┘  (circular!)
```

**The fix:** Split validators. Those that depend on analysis code move into the consuming verticals. Those that are purely foundational stay in `platform/validators.py`. The circular dependency is a red flag that the layering is wrong.

**Validators that belong in platform:** `validate_orthonormality`, `validate_eigendecomposition`, basic type checking.
**Validators that belong in verticals:** `validate_sensitivity` (depends on sensitivity_analysis), `validate_state_delta_estimation` (delta-specific), `validate_state_mzi` (could be platform or stay in vertical).

### 7.4 `src/algorithms/` as a Mixed Domain

**The problem:** `spin_squeezing.py` is quantum physics (used by physics and analysis modules). `optimization.py` is general numerical methods (used by ancilla optimization). `tensor_tree_network.py` is a specialized method for specific systems. `_mcmc_archive.py` is legacy.

**The fix:** Move `spin_squeezing.py` to `platform/`. Move `optimization.py` to `platform/` (as a general numerical utility). Move `tensor_tree_network.py` to its consuming vertical (likely `heisenberg_model` or standalone). Archive `_mcmc_archive.py`.

### 7.5 `pages/` as a Flat Collection

**The problem:** All 21 pages are in a flat directory with no structure. A vertical's page is disconnected from its simulation/analysis/report code.

**The fix:** Each page moves into `research/<vertical>/page.py`. The dashboard (`Home.py`) discovers pages dynamically from `research/*/page.py`.

### 7.6 `tests/` at Repository Root

**The problem:** Integration tests are separated from the code they test. This makes it harder to assess the test coverage of a single vertical.

**The fix:** Tests move into `platform/tests/` and `research/<vertical>/tests/`. The root `tests/` directory becomes a shim that re-exports or simply redirects.

### 7.7 Backward-Compatible Aliases

**The problem:** Several files define aliases for backward compatibility:
```python
# mzi_simulation.py
validate_state = validate_state_mzi  # from validators

# delta_estimation.py
validate_state = validate_state_delta_estimation  # DIFFERENT function!
```

These aliases suggest modules were moved around and compatibility was retrofitted. They increase cognitive load ("which `validate_state` is this?") and should be resolved during migration.

---

## 8. Areas of Poor Locality

These are specific examples where related code is scattered across the repository, increasing traversal radius.

### 8.1 Ancilla-Assisted Metrology (the worst case)

To understand the ancilla-assisted metrology optimization, an agent must read:

| What | Where | Lines |
|------|-------|-------|
| Experiment specification | `src/analysis/ancilla_optimization.py` (dataclasses) | 50 |
| Simulation code | `src/analysis/ancilla_optimization.py` | 600 |
| Analysis/sensitivity | `src/analysis/ancilla_optimization.py` | 400 |
| Orchestration (scans, search) | `src/analysis/ancilla_optimization.py` | 800 |
| Validation | `src/analysis/ancilla_optimization.py` (validate_*) | 200 |
| Streamlit page | `pages/Ancilla_Metrology_Optimization.py` | 1118 |
| Core operators (depends on) | `src/physics/dicke_basis.py` | 346 |
| Fisher info (depends on) | `src/analysis/fisher_information.py` | 485 |
| Reports | `reports/2026-05-12-*.md`, `reports/2026-05-15-*.md` | 2 files |
| Joint measurement extension | `src/analysis/weighted_joint_measurement.py` | 2946 |
| Tests | `src/analysis/test_ancilla_optimization.py` | varies |
| Integration tests | `tests/test_ancilla_joint_pipeline.py` | varies |
| Data | `reports/raw_data/` (multiple CSV files) | varies |

**Total locations: 9+. Traversal radius: very high.**

### 8.2 Scaling Survey

| What | Where | Lines |
|------|-------|-------|
| Orchestrator | `src/analysis/scaling_survey.py` | 1319 |
| Fisher info (depends on) | `src/analysis/fisher_information.py` | 485 |
| State factory (depends on) | `src/physics/mzi_states.py` | varies |
| Dicke basis (depends on) | `src/physics/dicke_basis.py` | 346 |
| Hybrid system (depends on) | `src/physics/hybrid_system.py`, `hybrid_mzi.py` | 500+ |
| Page | `pages/Interferometry_Scaling_Survey.py` | 817 |
| Scaling fit | `src/analysis/scaling_fit.py` | varies |
| Reports | `reports/2026-05-11-*.md` (3 files) | 3 files |
| Tests | `tests/test_scaling_survey.py` | varies |

**Total locations: 9. Traversal radius: high.**

### 8.3 BEC Ancilla

| What | Where | Lines |
|------|-------|-------|
| Physics simulation | `src/physics/bec_ancilla_system.py` | varies |
| Sensitivity analysis | `src/analysis/bec_sensitivity.py` | varies |
| Lindblad solver (depends on) | `src/evolution/lindblad_solver.py` | 498 |
| Spin squeezing (depends on) | `src/algorithms/spin_squeezing.py` | varies |
| Page 1 | `pages/BEC_Ancilla.py` | varies |
| Page 2 | `pages/BEC_Sensitivity_Scaling.py` | varies |
| Tests | `tests/test_pages_bec.py` | varies |

**Total locations: 7. Traversal radius: medium.**

---

## 9. Migration Strategy

**Key principle: No large rewrites.** Each step preserves existing functionality. All tests pass after each step. Backward-compatible shims are placed so that existing imports continue to work.

### Phase 0: Analysis and Groundwork (1-2 days)

**0.1 Document current dependency graph**
- Run `pytest --testmon` to establish baseline test pass rate.
- Generate a dependency graph of all imports (can use `pydeps` or manual analysis from this document).
- Identify all consumers of each module.

**0.2 Create platform directory skeleton**
```bash
mkdir -p platform/tests
touch platform/__init__.py
```

**0.3 Set up test infrastructure**
- Add `platform/tests/` to pytest testpaths in `pyproject.toml`.
- Root `conftest.py` already adds project root to `sys.path`, so `from platform import ...` will work.

### Phase 1: Extract Platform Primitives (3-4 days)

For each platform module (see §6):

**Pattern: Copy + Shim**

1. Copy the file from its current location to `platform/<module>.py`.
2. Replace the original file with a shim: `from platform.<module> import *  # noqa: F401, F403`
3. Run all tests. They pass because the imports resolve to the same code.
4. Move the module's unit tests to `platform/tests/`.

**Order of extraction (dependencies first):**

| Step | Module | Depends on |
|------|--------|------------|
| 1 | `platform/enums.py` | Nothing |
| 2 | `platform/dicke_basis.py` | enums |
| 3 | `platform/spin_squeezing.py` | dicke_basis |
| 4 | `platform/mzi_core.py` | dicke_basis |
| 5 | `platform/state_factory.py` | dicke_basis, mzi_core |
| 6 | `platform/states.py` | dicke_basis |
| 7 | `platform/partial_trace.py` | dicke_basis |
| 8 | `platform/noise_channels.py` | dicke_basis |
| 9 | `platform/fisher_information.py` | Nothing (pure numpy) |
| 10 | `platform/lindblad_solver.py` | dicke_basis, enums, noise_channels |
| 11 | `platform/wigner.py` | dicke_basis |
| 12 | `platform/scaling_fit.py` | Nothing |
| 13 | `platform/validators.py` | (split — remove circular dep) |
| 14 | `platform/plot_array.py` | Nothing |

**After Phase 1:** The original `src/` modules are still present as shims. All imports resolve. All tests pass.

### Phase 2: Pilot Vertical — `ancilla_drive` (2-3 days)

Pick the most self-contained vertical as the pilot. `ancilla_drive` is a good candidate because:
- It has one main module (`ancilla_drive_metrology.py`, 1162 lines)
- One visualization module (`ancilla_drive_plots.py`)
- One page (`Ancilla_Drive_Metrology.py`)
- One report (2026-05-18)
- It depends on platform primitives (Fisher info, Dicke basis) — not on other verticals

**2.1 Create research directory**
```bash
mkdir -p research/ancilla_drive/{tests,reports/figures,data}
touch research/ancilla_drive/__init__.py
```

**2.2 Extract into vertical**

| Current | → | research/ancilla_drive/ |
|---------|---|------------------------|
| `src/analysis/ancilla_drive_metrology.py` | → | `simulation.py` (core simulation) + `analysis.py` (sensitivity) + `orchestration.py` (parameter sweeps) |
| `src/visualization/ancilla_drive_plots.py` | → | `visualization.py` |
| (page imports) | → | `page.py` |
| `reports/2026-05-18-Ancilla-Drive-Enhanced-Metrology.md` | → | `reports/2026-05-18-finding.md` |

**2.3 Replace originals with shims**
- `src/analysis/ancilla_drive_metrology.py` → `from research.ancilla_drive.simulation import *`
- `src/visualization/ancilla_drive_plots.py` → `from research.ancilla_drive.visualization import *`
- `pages/Ancilla_Drive_Metrology.py` → `from research.ancilla_drive.page import *`

**2.4 Redirect tests**
- Move `src/analysis/test_ancilla_drive_metrology.py` to `research/ancilla_drive/tests/`
- Update imports to use `research.ancilla_drive` instead of `src.analysis`

**2.5 Verify**
- Run all tests. They pass.
- Run `streamlit run Home.py` and verify the Ancilla Drive page works.

**Decision gate:** After the pilot, assess whether the vertical structure improves agent navigation. If yes, proceed with remaining verticals.

### Phase 3: Create Remaining Verticals (5-8 days)

Each vertical follows the same pattern as Phase 2, but without redesign — just relocation and shimming.

**Priority order (most coupled first):**

| Priority | Vertical | Rationale |
|----------|----------|-----------|
| 1 | `ancilla_basic` | Largest single file (2278 lines). Will benefit most from decomposition. Split `ancilla_optimization.py` into `simulation.py`, `analysis.py`, `orchestration.py`. |
| 2 | `weighted_joint_measurement` | 2946 lines in one file. Split into `simulation.py` + `analysis.py` + `orchestration.py`. |
| 3 | `scaling_survey` | 1319 lines, high cross-vertical coupling. Separate the survey orchestrator from the page. |
| 4 | `delta_estimation` | Spans analysis + sensitivity + heatmap. Consolidate. |
| 5 | `ancilla_comparison` | Small, straightforward. |
| 6 | `bayesian_phase_estimation` | Moderate size, clear boundary. |
| 7-21 | Remaining verticals | Each is relatively self-contained. |

**For each vertical, the steps are:**
1. Create `research/<name>/` directory skeleton.
2. Move/copy files from `src/analysis/`, `src/physics/`, `src/visualization/`, `pages/`, `tests/`, `reports/`.
3. Split large files if needed (see §9.1).
4. Replace originals with shims.
5. Redirect tests.
6. Verify.

### Phase 3.1: Splitting Large Files

The three largest files will benefit from decomposition into the standard vertical structure:

**`ancilla_optimization.py` (2278 lines) → `research/ancilla_basic/`:**
- `experiment.py` — dataclasses, constants, Pauli matrices (~100 lines)
- `simulation.py` — state preparation, evolution operators, unitary construction (~400 lines)
- `analysis.py` — sensitivity computation, Fisher information, measurement operators (~400 lines)
- `orchestration.py` — Nelder-Mead optimization, theta scans, alpha scans, random search (~600 lines)
- `validation.py` — unitarity checks, derivative stability, operator validation (~200 lines)
- `page.py` — Streamlit UI (~500 lines)
- `tests/` — test files (~300 lines)

**`weighted_joint_measurement.py` (2946 lines) → `research/weighted_joint_measurement/`:**
- `experiment.py` — config classes
- `simulation.py` — measurement operators, probability computation
- `analysis.py` — sensitivity metrics, Fisher information
- `orchestration.py` — golden-section search, parameter sweeps
- `page.py` — Streamlit UI (when added)
- `tests/`

**`scaling_survey.py` (1319 lines) → `research/scaling_survey/`:**
- `experiment.py` — SurveyConfig, ModelConfig dataclasses
- `simulation.py` — state preparation, sensitivity function generators
- `analysis.py` — survey execution, exponent extraction
- `orchestration.py` — parameter sweeps, batch execution
- `page.py` — Streamlit UI
- `tests/`

### Phase 4: Consolidate (2-3 days)

**4.1 Remove shims**
- After all verticals are migrated and tested, remove the backward-compatible shims from `src/`, `pages/`, `tests/`, and `reports/`.
- Each shim removal is a single commit. If something breaks, the shim is reinstated.

**4.2 Update Home.py**
- `Home.py` currently imports from `pages/`. Update it to discover pages from `research/*/page.py`.
- Pattern: `page_files = sorted(Path("research").glob("*/page.py"))`

**4.3 Clean up deprecated directories**
- Remove `src/` (or archive it).
- Remove `pages/`.
- Remove root `tests/` (tests are now in `platform/tests/` and `research/*/tests/`).
- Move `reports/` to an archive or delete after confirming all reports are in verticals.

**4.4 Final verification**
- Full test suite passes.
- `streamlit run Home.py` renders all pages.
- Each page functions correctly.

### Phase 5: Documentation and Agent Configuration (1 day)

**5.1 Update opencode.json**
- Configure agent context windows to include `platform/` and the relevant `research/<vertical>/` directory.
- Add rules about the vertical structure.

**5.2 Write vertical creation template**
- Create `research/_template/` with the standard skeleton files and comments.
- Add a `create_vertical.sh` script.

**5.3 Document the architecture**
- Update README.md with the new structure.
- Add a short `ARCHITECTURE.md` explaining the platform/vertical split.

---

## 10. New Extension Pattern

After migration, implementing a new research direction follows this pattern:

### 10.1 Standard Workflow

```bash
# 1. Find the closest existing vertical as a reference
ls research/              # Look for similar research
less research/ancilla_drive/  # Inspect a vertical

# 2. Create a new vertical directory
mkdir -p research/my_new_experiment/{tests,reports/figures,data}

# 3. Implement the experiment
#    - Import only from platform/ and standard libraries
#    - Import from nearby verticals if needed (but prefer platform)
cat > research/my_new_experiment/simulation.py << 'EOF'
"""Simulation for my new experiment."""
import numpy as np
from platform.dicke_basis import jz_operator
from platform.state_factory import input_state_factory
from platform.fisher_information import quantum_fisher_information_dm
EOF

# 4. Create the Streamlit page
cat > research/my_new_experiment/page.py << 'EOF'
"""Streamlit page for my new experiment."""
import streamlit as st
from research.my_new_experiment.simulation import my_simulation

st.set_page_config(page_title="My New Experiment")
# ... UI code ...
EOF

# 5. Write tests
cat > research/my_new_experiment/tests/test_simulation.py << 'EOF'
"""Tests for my new experiment."""
from research.my_new_experiment.simulation import my_simulation

def test_my_simulation():
    result = my_simulation()
    assert result is not None
EOF

# 6. Run tests
uv run pytest research/my_new_experiment/tests/

# 7. Verify the page
uv run streamlit run Home.py
# Page is auto-discovered from research/my_new_experiment/page.py
```

### 10.2 What a Coding Agent Needs to Know

To implement a vertical, the agent needs:
1. **The platform primitives** (`platform/`) — ~500 lines of well-tested code.
2. **One nearby vertical** (`research/<similar>/`) — as a structural template.
3. **Standard libraries** (numpy, qutip, scipy, streamlit).

The agent does **not** need to:
- Understand `src/` internal coupling.
- Navigate 7 directories to find related code.
- Understand abstractions designed for shared reuse across all verticals.

### 10.3 When to Add to Platform vs. Keep in Vertical

| Scenario | Action |
|----------|--------|
| A function is used by 3+ verticals | Extract to `platform/` |
| A function is used by 1-2 verticals | Duplicate or keep in vertical |
| A new noise channel model | Start in vertical. Promote to platform if reused by 3+ verticals. |
| A new optimization algorithm | Start in vertical. Promote to platform if it's truly general (e.g., not physics-specific). |
| A new visualization type | Start in vertical. Extract generic utility (e.g., `plot_array`) to platform only when a second vertical needs it. |
| A new state type | Start in vertical. Add to `platform/state_factory.py` if multiple verticals use it. |

**Rule of three:** A primitive moves to platform only when it is used by at least three research verticals. Before that, duplication is preferred.

### 10.4 Explicit Composition over Implicit Systems

- Verticals explicitly import from `platform/`. There is no automatic discovery or plugin system.
- Verticals do not share orchestration code. Each vertical's `orchestration.py` is independent.
- If two verticals need the same orchestration pattern (e.g., "sweep parameter X, compute Y, plot Z"), they each implement it. Shared infrastructure for orchestration is an accidental abstraction.

---

## 11. Appendix: Vertical Fact Sheets

### Vertical 1: ancilla_basic

| Property | Value |
|----------|-------|
| **Source files** | `src/analysis/ancilla_optimization.py` (2278 lines) |
| **Pages** | `pages/Ancilla_Metrology_Optimization.py` (1118 lines) |
| **Reports** | `2026-05-12-Ancilla-Assisted-Metrology-Optimization.md`, `2026-05-15-Ancilla-Assisted-Metrology-Joint-Measurement.md` |
| **Tests** | `src/analysis/test_ancilla_optimization.py`, `tests/test_ancilla_joint_pipeline.py` |
| **Platform deps** | dicke_basis, fisher_information, lindblad_solver, spin_squeezing |
| **Split plan** | experiment(100) + simulation(400) + analysis(400) + orchestration(600) + validation(200) + page(500) |

### Vertical 2: ancilla_drive

| Property | Value |
|----------|-------|
| **Source files** | `src/analysis/ancilla_drive_metrology.py` (1162 lines) |
| **Visualization** | `src/visualization/ancilla_drive_plots.py` |
| **Pages** | `pages/Ancilla_Drive_Metrology.py` |
| **Reports** | `2026-05-18-Ancilla-Drive-Enhanced-Metrology.md` |
| **Tests** | (none standalone) |
| **Platform deps** | dicke_basis, fisher_information, mzi_core |

### Vertical 3: weighted_joint_measurement

| Property | Value |
|----------|-------|
| **Source files** | `src/analysis/weighted_joint_measurement.py` (2946 lines) |
| **Pages** | (via Ancilla_Metrology_Optimization page) |
| **Reports** | `2026-05-18-Weighted-Joint-Measurement-NM-Generalization.md` |
| **Tests** | `tests/test_weighted_joint_measurement.py` |
| **Platform deps** | dicke_basis, fisher_information |
| **Split plan** | experiment(100) + simulation(800) + analysis(800) + orchestration(800) + page(400) |

### Vertical 4: scaling_survey

| Property | Value |
|----------|-------|
| **Source files** | `src/analysis/scaling_survey.py` (1319 lines) |
| **Pages** | `pages/Interferometry_Scaling_Survey.py` (817 lines) |
| **Reports** | `2026-05-11-*.md` (3 reports) |
| **Tests** | `tests/test_scaling_survey.py` |
| **Platform deps** | fisher_information, scaling_fit, dicke_basis, state_factory |
| **Vertical deps** | hybrid_system (needs mzi_core) |

---

## Migration Checklist

- [ ] **Phase 0:** Dependency graph, baseline tests, platform skeleton
- [ ] **Phase 1.1:** platform/enums.py
- [ ] **Phase 1.2:** platform/dicke_basis.py
- [ ] **Phase 1.3:** platform/spin_squeezing.py
- [ ] **Phase 1.4:** platform/mzi_core.py
- [ ] **Phase 1.5:** platform/state_factory.py
- [ ] **Phase 1.6:** platform/states.py
- [ ] **Phase 1.7:** platform/partial_trace.py
- [ ] **Phase 1.8:** platform/noise_channels.py
- [ ] **Phase 1.9:** platform/fisher_information.py
- [ ] **Phase 1.10:** platform/lindblad_solver.py
- [ ] **Phase 1.11:** platform/wigner.py
- [ ] **Phase 1.12:** platform/scaling_fit.py
- [ ] **Phase 1.13:** platform/validators.py (resolved circular dep)
- [ ] **Phase 1.14:** platform/plot_array.py
- [ ] **Phase 2:** Pilot vertical (ancilla_drive)
- [ ] **Phase 3.1:** ancilla_basic vertical
- [ ] **Phase 3.2:** weighted_joint_measurement vertical
- [ ] **Phase 3.3:** scaling_survey vertical
- [ ] **Phase 3.4:** delta_estimation vertical
- [ ] **Phase 3.5:** ancilla_comparison vertical
- [ ] **Phase 3.6:** bayesian_phase_estimation vertical
- [ ] **Phase 3.7:** Remaining verticals (cavity, distributed, hybrid, heisenberg, etc.)
- [ ] **Phase 4.1:** Remove shims
- [ ] **Phase 4.2:** Update Home.py for dynamic page discovery
- [ ] **Phase 4.3:** Clean up deprecated directories
- [ ] **Phase 4.4:** Final verification
- [ ] **Phase 5:** Documentation and agent configuration
