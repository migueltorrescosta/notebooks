"""
Local module for the 2026-05-19 Ancilla-Drive Phase-Modulated Metrology report.

Contains all code exclusive to this report:
- Core physics simulation (ω-modulated Hamiltonian operators, circuit evolution,
  sensitivity computation)
- 4D random search, Nelder–Mead refinement, and ω-scan orchestration
- Exclusive plot functions (combined sensitivity, NM expectation/variance,
  cross-experiment comparison, fraction below SQL)
- Data and figure generation pipeline (``generate_phase_*`` functions)
- CLI entry point for standalone execution

Usage:
    uv run python reports/20260519/phase_modulated_drive.py --force
    uv run python reports/20260519/phase_modulated_drive.py --only phase-decoupled-baseline

This module is importable via ``importlib.import_module("reports.20260519.phase_modulated_drive")``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import multiprocessing as _mp
import os
import sys
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.linalg import expm

# Force non-interactive matplotlib backend before any plotting.
if "MPLBACKEND" not in os.environ:
    os.environ["MPLBACKEND"] = "Agg"
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "1"

# Shared primitives (used by all reports)
from src.analysis.ancilla_drive_metrology import (
    build_iszz_interaction,
    compute_phase_modulated_sensitivity,
    system_only_bs_unitary,
)
from src.analysis.ancilla_drive_results import (
    Drive2DSliceResult,
    DriveDecoupledBaselineResult,
    DriveNelderMeadResult,
    DriveOmegaScanResult,
    DriveRandomSearchResult,
)
from src.analysis.ancilla_optimization import (
    build_two_qubit_operators,
)
from src.analysis.optimisation_pipeline import (
    TwoPhaseConfig,
    build_nm_result,
    build_rs_result,
    make_4d_objective,
    run_omega_scan,
    run_two_phase_pipeline,
)
from src.analysis.slice_scan import sequential_grid_scan
from src.utils.constants import I_4
from src.utils.parallel import parallel_map
from src.utils.paths import report_path_fn
from src.visualization.ancilla_drive_plots import (
    plot_combined_sensitivity,
    plot_drive_nm_expectation_variance,
)

sns.set_theme(style="whitegrid")

# ============================================================================
# Physical constants  (moved from ancilla_drive_phase_modulated.py)
# ============================================================================

DEFAULT_T_BS: float = np.pi / 2.0  # 50/50 beam splitter
DEFAULT_t_hold: float = 10.0  # Holding time (SQL = 0.1)
DEFAULT_PSI0: np.ndarray = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)  # |00⟩
SQL_REFERENCE: float = 1.0 / DEFAULT_t_hold  # Δω_SQL = 0.1
DRIVE_BOUNDS: tuple[float, float] = (-5.0, 5.0)  # Range for all coefficients

# ============================================================================
# Operator Construction  (from ancilla_drive_phase_modulated.py)
# ============================================================================


def build_phase_modulated_drive_hamiltonian(
    omega: float,
    a_x: float,
    a_y: float,
    a_z: float,
    ops: dict[str, np.ndarray],
) -> np.ndarray:
    """Build the ω-modulated ancilla drive Hamiltonian.

    H_A = ω (a_x J_x^A + a_y J_y^A + a_z J_z^A)

    The critical difference from the fixed-drive protocol is the leading ω
    factor: the ancilla drive scales with the unknown phase, creating a
    parametric amplification effect in ∂⟨J_z^S⟩/∂ω.

    Args:
        omega: Unknown phase rate parameter (scales the whole drive).
        a_x: Coefficient for J_x^A.
        a_y: Coefficient for J_y^A.
        a_z: Coefficient for J_z^A.
        ops: Two-qubit operators from build_two_qubit_operators().

    Returns:
        4×4 Hermitian matrix representing the ω-modulated ancilla drive.
    """
    H = np.zeros((4, 4), dtype=complex)
    if a_x != 0.0:
        H += a_x * ops["Jx_A"]
    if a_y != 0.0:
        H += a_y * ops["Jy_A"]
    if a_z != 0.0:
        H += a_z * ops["Jz_A"]
    H = omega * H  # ω-modulation: entire drive scales with the unknown phase
    return 0.5 * (H + H.conj().T)


def build_phase_modulated_hold_hamiltonian(
    omega: float,
    a_x: float,
    a_y: float,
    a_z: float,
    a_zz: float,
    ops: dict[str, np.ndarray],
) -> np.ndarray:
    """Build the total holding Hamiltonian with ω-modulated ancilla drive.

    H = ω J_z^S + H_A + H_int
      = ω J_z^S + ω (a_x J_x^A + a_y J_y^A + a_z J_z^A) + a_zz J_z^S ⊗ J_z^A
      = ω [J_z^S + a_x J_x^A + a_y J_y^A + a_z J_z^A] + a_zz J_z^S ⊗ J_z^A

    The ω factor on the drive terms means ∂H/∂ω = J_z^S + a_x J_x^A + a_y J_y^A
    + a_z J_z^A, which includes ancilla operators. This extra contribution to
    the derivative is the key mechanism for potential SQL violation.

    Args:
        omega: Unknown phase rate parameter.
        a_x: Ancilla J_x drive coefficient.
        a_y: Ancilla J_y drive coefficient.
        a_z: Ancilla J_z drive coefficient.
        a_zz: Ising interaction coefficient.
        ops: Two-qubit operators from build_two_qubit_operators().

    Returns:
        4×4 Hermitian Hamiltonian matrix.
    """
    H = omega * ops["Jz_S"]
    H += build_phase_modulated_drive_hamiltonian(omega, a_x, a_y, a_z, ops)
    H += build_iszz_interaction(a_zz, ops)
    return 0.5 * (H + H.conj().T)


def phase_modulated_hold_unitary(
    t_hold: float,
    omega: float,
    a_x: float,
    a_y: float,
    a_z: float,
    a_zz: float,
    ops: dict[str, np.ndarray],
) -> np.ndarray:
    """Holding-time unitary for the ω-modulated ancilla protocol.

    U_hold(t_hold) = exp(-i t_hold H)
    where H = ω J_z^S + ω(a_x J_x^A + a_y J_y^A + a_z J_z^A) + a_zz J_z^S ⊗ J_z^A.

    Args:
        t_hold: Holding-time strength.
        omega: True phase rate parameter.
        a_x: Ancilla J_x drive coefficient.
        a_y: Ancilla J_y drive coefficient.
        a_z: Ancilla J_z drive coefficient.
        a_zz: Ising interaction coefficient.
        ops: Two-qubit operators from build_two_qubit_operators().

    Returns:
        4×4 unitary matrix.
    """
    H = build_phase_modulated_hold_hamiltonian(omega, a_x, a_y, a_z, a_zz, ops)
    U = expm(-1j * t_hold * H)
    assert np.allclose(U @ U.conj().T, I_4, atol=1e-12), (
        f"Phase-modulated hold unitary not unitary for t_hold={t_hold}, ω={omega}"
    )
    return U


def evolve_phase_modulated_circuit(
    psi0: np.ndarray,
    T_BS: float,
    t_hold: float,
    omega: float,
    a_x: float,
    a_y: float,
    a_z: float,
    a_zz: float,
    ops: dict[str, np.ndarray],
) -> np.ndarray:
    """Run the full ω-modulated ancilla MZI circuit.

    |ψ_final⟩ = U_BS_S · U_hold(t_hold) · U_BS_S · |ψ₀⟩

    The hold unitary uses the ω-modulated H_A = ω (a_x J_x^A + ...).

    Args:
        psi0: Initial 4-vector (must be normalised).
        T_BS: Beam-splitter duration (both BS identical).
        t_hold: Holding-time strength.
        omega: Phase rate parameter.
        a_x: Ancilla J_x drive coefficient.
        a_y: Ancilla J_y drive coefficient.
        a_z: Ancilla J_z drive coefficient.
        a_zz: Ising interaction coefficient.
        ops: Two-qubit operators.

    Returns:
        Final normalised 4-vector state.
    """
    assert np.isclose(np.linalg.norm(psi0), 1.0), "Initial state must be normalised"

    U_bs = system_only_bs_unitary(T_BS)
    psi = U_bs @ psi0
    psi = phase_modulated_hold_unitary(t_hold, omega, a_x, a_y, a_z, a_zz, ops) @ psi
    psi = U_bs @ psi

    assert np.isclose(np.linalg.norm(psi), 1.0), "Final state must be normalised"
    return psi



# ============================================================================
# Decoupled Baseline
# ============================================================================


def compute_phase_modulated_decoupled_baseline(
    t_hold: float = DEFAULT_t_hold,
    omega_true: float = 1.0,
) -> DriveDecoupledBaselineResult:
    """Compute the decoupled baseline sensitivity Δω.

    At (a_x = a_y = a_z = a_zz = 0), the ω-modulated ancilla circuit reduces
    to a standard single-qubit MZI with |1,0⟩ input and 50/50 BS,
    giving Δω = 1/t_hold. The ω factor in H_A is irrelevant when all a_k = 0.

    Args:
        t_hold: Holding-time strength.
        omega_true: True phase rate.

    Returns:
        DriveDecoupledBaselineResult.
    """
    ops = build_two_qubit_operators()
    domega = compute_phase_modulated_sensitivity(
        DEFAULT_PSI0,
        DEFAULT_T_BS,
        t_hold,
        omega_true,
        0.0,
        0.0,
        0.0,
        0.0,
        ops,
    )
    return DriveDecoupledBaselineResult(
        t_hold_value=t_hold,
        delta_omega=domega,
        sql=1.0 / t_hold,
    )


# ============================================================================
# 2D Slice Scan
# ============================================================================


def phase_modulated_2d_slice(
    omega: float,
    drive_range: tuple[float, float] = DRIVE_BOUNDS,
    azz_range: tuple[float, float] = DRIVE_BOUNDS,
    n_drive: int = 201,
    n_azz: int = 201,
    slice_type: str = "ax",
    t_hold: float = DEFAULT_t_hold,
    T_BS: float = DEFAULT_T_BS,
) -> Drive2DSliceResult:
    """Run a 2D slice scan over (a_drive, a_zz) with ω-modulated ancilla drive.

    For slice_type='ax': varies a_x (with a_y = a_z = 0).
    For slice_type='ay': varies a_y (with a_x = a_z = 0).

    Args:
        omega: Phase rate value.
        drive_range: (min, max) for the drive coefficient.
        azz_range: (min, max) for the interaction coefficient.
        n_drive: Number of drive-coefficient points.
        n_azz: Number of a_zz points.
        slice_type: 'ax' or 'ay'.
        t_hold: Holding time (default 10).
        T_BS: Beam-splitter duration (default π/2).

    Returns:
        Drive2DSliceResult with the sensitivity grid.
    """
    if slice_type not in ("ax", "ay"):
        raise ValueError(f"slice_type must be 'ax' or 'ay', got {slice_type}")

    ops = build_two_qubit_operators()
    drive_vals = np.linspace(drive_range[0], drive_range[1], n_drive)
    azz_vals = np.linspace(azz_range[0], azz_range[1], n_azz)

    def _sensitivity(d_val: float, a_val: float) -> float:
        if slice_type == "ax":
            ax, ay, az = d_val, 0.0, 0.0
        else:
            ax, ay, az = 0.0, d_val, 0.0
        return compute_phase_modulated_sensitivity(
            DEFAULT_PSI0,
            T_BS,
            t_hold,
            omega,
            ax,
            ay,
            az,
            a_val,
            ops,
        )

    grid = sequential_grid_scan(drive_vals, azz_vals, _sensitivity)

    return Drive2DSliceResult(
        drive_values=drive_vals,
        azz_values=azz_vals,
        delta_omega_grid=grid,
        omega_value=omega,
        slice_type=slice_type,
        sql=1.0 / t_hold,
    )


# ============================================================================
# Shared Pipeline Helpers
# ============================================================================


def _make_phase_objective(
    omega: float,
    ops: dict[str, np.ndarray],
    psi0: np.ndarray,
    t_hold: float = DEFAULT_t_hold,
    T_BS: float = DEFAULT_T_BS,
) -> Callable[[np.ndarray], float]:
    """Build the raw (unpenalised) Δω objective for a given ω.

    Uses ``make_4d_objective`` from the shared pipeline module.
    """
    return make_4d_objective(
        compute_phase_modulated_sensitivity,
        psi0=psi0,
        T_BS=T_BS,
        t_hold=t_hold,
        omega=omega,
        ops=ops,
    )


# ============================================================================
# 4D Random Search
# ============================================================================


def phase_modulated_random_search(
    omega: float,
    n_samples: int = 500,
    bounds: tuple[float, float] = DRIVE_BOUNDS,
    t_hold: float = DEFAULT_t_hold,
    T_BS: float = DEFAULT_T_BS,
    seed: int | None = 42,
) -> DriveRandomSearchResult:
    """Random search over the 4D parameter space (a_x, a_y, a_z, a_zz).

    Uses the ω-modulated ancilla drive H_A = ω (a_x J_x^A + ...).

    Args:
        omega: Phase rate value.
        n_samples: Number of random points to evaluate.
        bounds: (min, max) for all four coefficients.
        t_hold: Holding time.
        T_BS: Beam-splitter duration.
        seed: Random seed for reproducibility.

    Returns:
        DriveRandomSearchResult with all samples and best found.
    """
    ops = build_two_qubit_operators()
    raw_obj = _make_phase_objective(omega, ops, DEFAULT_PSI0, t_hold=t_hold, T_BS=T_BS)
    return build_rs_result(
        raw_obj,
        n_samples,
        seed or 42,
        omega=omega,
        sql=1.0 / t_hold,
        t_hold=t_hold,
    )



# ============================================================================
# ω Scan with Random Search + Nelder--Mead Refinement
# ============================================================================


def run_phase_modulated_omega_scan(
    omega_values: list[float] | np.ndarray,
    n_random: int = 500,
    n_nm_refine: int = 50,
    seed: int | None = 42,
    maxiter: int = 5000,
    bounds: tuple[float, float] = DRIVE_BOUNDS,
    t_hold: float = DEFAULT_t_hold,
    T_BS: float = DEFAULT_T_BS,
) -> DriveOmegaScanResult:
    """Scan over ω values with 4D random search and Nelder--Mead refinement.

    For each ω:
    1. Run `n_random` random evaluations in the 4D parameter space.
    2. Select the best `n_nm_refine` points.
    3. Run Nelder--Mead refinement from each selected point.
    4. Record the best overall result.

    Args:
        omega_values: ω values to scan.
        n_random: Number of random search points per ω.
        n_nm_refine: Number of Nelder--Mead refinements per ω.
        seed: Base random seed (incremented per ω).
        maxiter: Maximum Nelder--Mead iterations.
        bounds: (min, max) for all parameters.
        t_hold: Holding time.
        T_BS: Beam-splitter duration.

    Returns:
        DriveOmegaScanResult with optimal parameters and sensitivities.
    """
    ops = build_two_qubit_operators()

    config = TwoPhaseConfig(
        n_random=n_random,
        n_nm_refine=n_nm_refine,
        nm_maxiter=maxiter,
        seed=seed,
        bounds=bounds,
    )

    def _rs_fn(n_samples: int, seed: int, **kw: Any) -> DriveRandomSearchResult:
        omega = kw["omega"]
        raw_obj = _make_phase_objective(
            omega, ops, DEFAULT_PSI0, t_hold=t_hold, T_BS=T_BS
        )
        return build_rs_result(
            raw_obj,
            n_samples,
            seed,
            omega=omega,
            sql=1.0 / t_hold,
            t_hold=t_hold,
        )

    def _nm_fn(x0: np.ndarray, seed: int, **kw: Any) -> DriveNelderMeadResult:
        omega = kw["omega_true"]
        raw_obj = _make_phase_objective(
            omega, ops, DEFAULT_PSI0, t_hold=t_hold, T_BS=T_BS
        )
        return build_nm_result(
            raw_obj,
            x0,
            omega=omega,
            ops=ops,
            psi0=DEFAULT_PSI0,
            evolve_fn=lambda psi, ax, ay, az, azz, _ops: evolve_phase_modulated_circuit(
                psi,
                DEFAULT_T_BS,
                t_hold,
                omega,
                ax,
                ay,
                az,
                azz,
                _ops,
            ),
            t_hold=t_hold,
        )

    best_results, all_results = run_omega_scan(
        omega_values=omega_values,
        random_search_fn=_rs_fn,
        nm_fn=_nm_fn,
        config=config,
    )

    omega_arr = np.asarray(omega_values, dtype=float)
    sql = 1.0 / t_hold

    return DriveOmegaScanResult(
        omega_values=omega_arr,
        best_params_per_omega=[
            (
                float(r.params_opt[0]),
                float(r.params_opt[1]),
                float(r.params_opt[2]),
                float(r.params_opt[3]),
            )
            for r in best_results
        ],
        best_delta_omega_per_omega=np.array([r.delta_omega_opt for r in best_results]),
        sql_values=np.full(len(best_results), sql),
        expectation_Jz_per_omega=np.array([r.expectation_Jz for r in best_results]),
        variance_Jz_per_omega=np.array([r.variance_Jz for r in best_results]),
        all_results={
            float(omega): results
            for omega, results in zip(omega_arr, all_results, strict=True)
        },
    )


def plot_drive_cross_experiment_comparison(
    omega_values: np.ndarray,
    best_delta_19: np.ndarray,
    best_delta_18: np.ndarray,
    sql_values: np.ndarray,
    save_path: str | Path,
    figsize: tuple[float, float] = (8, 5),
) -> Path:
    """Compare Δω from the fixed-drive (2026-05-18) and modulated-drive
    (2026-05-19) experiments in a 2×1 vertically stacked figure.

    Upper panel: Overlaid line plots of Δω vs ω for both experiments,
    with the SQL shown as a dashed reference line.

    Lower panel: Ratio Δω_19 / Δω_18 vs ω. A horizontal line at y=1
    separates regimes where the fixed drive (above 1) or modulated drive
    (below 1) performs better.

    Args:
        omega_values: Common ω grid (50 points from the modulated-drive scan).
        best_delta_19: Δω from the modulated-drive scan (2026-05-19).
        best_delta_18: Δω from the fixed-drive scan (2026-05-18),
            interpolated to the same ω grid.
        sql_values: SQL reference values (constant, 0.1) at each ω.
        save_path: Output SVG path.
        figsize: Figure size (width, height).

    Returns:
        Path to saved SVG.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # ── Upper panel: Δω vs ω ──────────────────────────────────────────
    sql_ref = float(sql_values[0]) if len(sql_values) > 0 else 0.1

    ax1.axhline(
        y=sql_ref,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label=rf"SQL = {sql_ref:.4f}",
    )

    ax1.plot(
        omega_values,
        best_delta_18,
        marker="s",
        linestyle="-",
        color="C0",
        markersize=5,
        linewidth=1.8,
        label=r"Fixed drive (2026-05-18)",
    )
    ax1.plot(
        omega_values,
        best_delta_19,
        marker="o",
        linestyle="-",
        color="C3",
        markersize=5,
        linewidth=1.8,
        label=r"Modulated drive (2026-05-19)",
    )

    ax1.set_ylabel(r"$\Delta\omega$")
    ax1.set_title("Cross-experiment comparison: fixed vs modulated drive")
    ax1.legend(fontsize=9)

    # ── Lower panel: ratio Δω_19 / Δω_18 ──────────────────────────────
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(
            np.isfinite(best_delta_18) & (best_delta_18 > 0),
            best_delta_19 / best_delta_18,
            np.nan,
        )

    ax2.plot(
        omega_values,
        ratio,
        marker="o",
        linestyle="-",
        color="C3",
        markersize=4,
        linewidth=1.5,
    )
    ax2.axhline(
        y=1.0, color="gray", linestyle="--", linewidth=1.2, alpha=0.7, label="y = 1"
    )

    # Annotate the minimum ratio
    valid = np.isfinite(ratio)
    if np.any(valid):
        min_idx = np.argmin(ratio[valid])
        min_ratio = float(ratio[valid][min_idx])
        min_omega = float(omega_values[valid][min_idx])
        ax2.annotate(
            f"Best = {min_ratio:.3f}$\\times$ at $\\omega$={min_omega:.1f}",
            xy=(min_omega, min_ratio),
            xytext=(min_omega + 0.6, min_ratio + 0.15),
            arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.2},
            fontsize=10,
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "edgecolor": "gray",
            },
        )

    ax2.set_xlabel(r"$\omega$")
    ax2.set_ylabel(r"$\Delta\omega_{19} \;/\; \Delta\omega_{18}$")
    ax2.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_drive_fraction_below_sql(
    omega_values: np.ndarray,
    fractions_2d_ax: np.ndarray,
    fractions_2d_ay: np.ndarray,
    fractions_random: np.ndarray,
    save_path: str | Path,
    figsize: tuple[float, float] = (8, 5),
) -> Path:
    """Line plot of the fraction of parameter space below SQL as a function of ω.

    Args:
        omega_values: Array of ω values.
        fractions_2d_ax: Fraction below SQL from (a_x, a_zz) slices at each ω.
        fractions_2d_ay: Fraction below SQL from (a_y, a_zz) slices at each ω.
        fractions_random: Fraction below SQL from 4D random search at each ω.
        save_path: Output SVG path.
        figsize: Figure size (width, height).

    Returns:
        Path to saved SVG.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        omega_values,
        fractions_2d_ax,
        "o-",
        color="C0",
        label=r"2D slice $(a_x, a_{zz})$",
        markersize=6,
        linewidth=1.5,
    )
    ax.plot(
        omega_values,
        fractions_2d_ay,
        "s-",
        color="C1",
        label=r"2D slice $(a_y, a_{zz})$",
        markersize=6,
        linewidth=1.5,
    )
    ax.plot(
        omega_values,
        fractions_random,
        "^-",
        color="C2",
        label="4D random search",
        markersize=6,
        linewidth=1.5,
    )

    # Reference lines at y=0 and y=1
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5, linewidth=1)

    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel("Fraction below SQL")
    ax.set_title("Robustness of SQL violation: fraction of parameter space below SQL")
    ax.set_ylim(0, 1)
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


# ============================================================================
# Data / Figure Generation Pipeline
# (moved from src/visualization/report_figures.py)
# ============================================================================

REPORTS_DIR = Path(__file__).resolve().parent.parent
PHASE_DATE = "20260519"
PHASE_OMEGA_VALS = [round(v, 1) for v in np.linspace(0.1, 5.0, 50).tolist()]
PHASE_N_GRID = 201


_parquet_path, _fig_path = report_path_fn(REPORTS_DIR, PHASE_DATE)


# ── Parallel dispatch helper ──────────────────────────────────────────────


# ── Generator functions ───────────────────────────────────────────────────


def generate_phase_decoupled_baseline(force: bool = False) -> None:
    """Phase-modulated decoupled baseline verification."""
    csv_p = _parquet_path("phase-decoupled-baseline")
    fig_p = _fig_path("phase-decoupled-baseline")

    if csv_p.exists() and not force:
        print(f"[skip] {csv_p.name} exists (use --force to overwrite)")
        result = DriveDecoupledBaselineResult.from_parquet(csv_p)
    else:
        print("[run]  Computing phase-modulated decoupled baseline...")
        result = compute_phase_modulated_decoupled_baseline()
        result.save_parquet(csv_p)
        print(f"[save] {csv_p}")

    # Use the shared plot function from the main codebase
    from src.visualization.ancilla_drive_plots import plot_drive_decoupled_baseline

    plot_drive_decoupled_baseline(result, fig_p)
    print(f"[fig]  {fig_p}")


def _run_phase_2d_slice(
    omega: float,
    slice_type: str,
    force: bool,
) -> None:
    """Run a phase-modulated 2D slice scan for a single ω value."""
    tag = f"phase-2d-slice-{slice_type}-azz-omega{omega}"
    csv_p = _parquet_path(tag)
    fig_p = _fig_path(tag)

    if csv_p.exists() and not force:
        print(f"  [skip] {csv_p.name} exists (use --force to overwrite)")
        result = Drive2DSliceResult.from_parquet(csv_p)
    else:
        print(f"  [run]  Computing phase ({slice_type}, a_zz) slice at ω={omega}...")
        result = phase_modulated_2d_slice(
            omega=omega,
            slice_type=slice_type,
            n_drive=PHASE_N_GRID,
            n_azz=PHASE_N_GRID,
        )
        result.save_parquet(csv_p)
        print(f"  [save] {csv_p}")

    from src.visualization.ancilla_drive_plots import plot_drive_2d_slice_heatmap

    plot_drive_2d_slice_heatmap(result, fig_p)
    print(f"  [fig]  {fig_p}")


def generate_phase_2d_slice_ax_azz(force: bool = False) -> None:
    """Phase-modulated 2D slice scans over (a_x, a_zz) at all ω values."""
    n = len(PHASE_OMEGA_VALS)
    print(f"[run]  (a_x, a_zz) phase slice at {n} ω values (parallel)")
    worker = partial(_run_phase_2d_slice, slice_type="ax", force=force)
    parallel_map(worker, PHASE_OMEGA_VALS, desc="(a_x, a_zz) slices")


def generate_phase_2d_slice_ay_azz(force: bool = False) -> None:
    """Phase-modulated 2D slice scans over (a_y, a_zz) at all ω values."""
    n = len(PHASE_OMEGA_VALS)
    print(f"[run]  (a_y, a_zz) phase slice at {n} ω values (parallel)")
    worker = partial(_run_phase_2d_slice, slice_type="ay", force=force)
    parallel_map(worker, PHASE_OMEGA_VALS, desc="(a_y, a_zz) slices")


def _run_phase_random_search(omega: float, force: bool) -> None:
    """Run a phase-modulated 4D random search for a single ω value."""
    tag = f"phase-random-search-omega{omega}"
    csv_p = _parquet_path(tag)
    fig_p = _fig_path(tag)

    if csv_p.exists() and not force:
        print(f"  [skip] {csv_p.name} exists (use --force to overwrite)")
        result = DriveRandomSearchResult.from_parquet(csv_p)
    else:
        print(f"  [run]  Running phase 4D random search at ω={omega} (500 samples)...")
        result = phase_modulated_random_search(
            omega=omega,
            n_samples=500,
            seed=42,
        )
        result.save_parquet(csv_p)
        print(f"  [save] {csv_p}")

    from src.visualization.ancilla_drive_plots import plot_drive_random_search_histogram

    plot_drive_random_search_histogram(result, fig_p)
    print(f"  [fig]  {fig_p}")


def generate_phase_random_search(force: bool = False) -> None:
    """Phase-modulated 4D random search at all ω values (parallel)."""
    n = len(PHASE_OMEGA_VALS)
    print(f"[run]  4D phase random search at {n} ω values (parallel)")
    worker = partial(_run_phase_random_search, force=force)
    parallel_map(worker, PHASE_OMEGA_VALS, desc="random search")


def _run_phase_omega_scan_single(omega: float) -> dict[str, float | np.ndarray]:
    """Run random search + NM refinement for a single ω value.

    Returns a dict with per-ω results that can be aggregated into a
    ``DriveOmegaScanResult``.
    """
    ops = build_two_qubit_operators()
    raw_obj = _make_phase_objective(omega, ops, DEFAULT_PSI0)

    def _rs_fn(n_samples: int, seed: int, **kw: Any) -> DriveRandomSearchResult:
        return build_rs_result(
            raw_obj,
            n_samples,
            seed,
            omega=omega,
            sql=1.0 / DEFAULT_t_hold,
            t_hold=DEFAULT_t_hold,
        )

    def _nm_fn(x0: np.ndarray, seed: int, **kw: Any) -> DriveNelderMeadResult:
        return build_nm_result(
            raw_obj,
            x0,
            omega=omega,
            ops=ops,
            psi0=DEFAULT_PSI0,
            evolve_fn=lambda psi, ax, ay, az, azz, _ops: evolve_phase_modulated_circuit(
                psi,
                DEFAULT_T_BS,
                DEFAULT_t_hold,
                omega,
                ax,
                ay,
                az,
                azz,
                _ops,
            ),
            t_hold=DEFAULT_t_hold,
        )

    config = TwoPhaseConfig(
        n_random=500,
        n_nm_refine=50,
        nm_maxiter=5000,
        seed=42,
        bounds=(-5.0, 5.0),
    )

    best_nm, _ = run_two_phase_pipeline(
        random_search_fn=_rs_fn,
        nm_fn=_nm_fn,
        config=config,
        seed=42 + int(omega * 1000),
    )

    return {
        "omega": omega,
        "best_delta_omega": best_nm.delta_omega_opt,
        "a_x": float(best_nm.params_opt[0]),
        "a_y": float(best_nm.params_opt[1]),
        "a_z": float(best_nm.params_opt[2]),
        "a_zz": float(best_nm.params_opt[3]),
        "expectation_Jz": best_nm.expectation_Jz,
        "variance_Jz": best_nm.variance_Jz,
    }


def _check_omega_scan_cache(
    parquet_path: Path, force: bool
) -> DriveOmegaScanResult | None:
    """Check if cached parquet exists and load it.

    Args:
        parquet_path: Path to cached Parquet file.
        force: If True, ignore cache and return None.

    Returns:
        Loaded result if cached and not forced, else None.
    """
    if not parquet_path.exists() or force:
        return None
    print(f"[skip] {parquet_path.name} exists (use --force to overwrite)")
    return DriveOmegaScanResult.from_parquet(parquet_path)


def _compute_omega_scan_core() -> DriveOmegaScanResult:
    """Run the parallel ω-scan computation (random search + NM refinement)."""
    n = len(PHASE_OMEGA_VALS)
    print(f"[run]  Computing phase ω-scan for {n} ω values (parallel)...")

    max_workers = min(32, os.cpu_count() or 1)
    print(f"  [parallel] Using {max_workers} workers for ω-scan")

    per_omega_results: list[dict[str, Any]] = []
    mp_ctx = _mp.get_context("fork")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=mp_ctx,
    ) as executor:
        fut_to_omega = {
            executor.submit(_run_phase_omega_scan_single, omega): omega
            for omega in PHASE_OMEGA_VALS
        }
        for future in concurrent.futures.as_completed(fut_to_omega):
            omega = fut_to_omega[future]
            try:
                per_omega_results.append(future.result())
                print(f"  [done] ω={omega}")
            except Exception as exc:
                print(f"  [ERROR] ω={omega}: {exc}")
                raise

    per_omega_results.sort(key=lambda r: float(r["omega"]))

    omega_arr = np.array([r["omega"] for r in per_omega_results], dtype=float)
    best_deltas = [float(r["best_delta_omega"]) for r in per_omega_results]
    best_params = [
        (
            float(r["a_x"]),
            float(r["a_y"]),
            float(r["a_z"]),
            float(r["a_zz"]),
        )
        for r in per_omega_results
    ]
    exp_vals = [float(r["expectation_Jz"]) for r in per_omega_results]
    var_vals = [float(r["variance_Jz"]) for r in per_omega_results]

    return DriveOmegaScanResult(
        omega_values=omega_arr,
        best_params_per_omega=best_params,
        best_delta_omega_per_omega=np.array(best_deltas, dtype=float),
        sql_values=np.full(len(omega_arr), 1.0 / DEFAULT_t_hold, dtype=float),
        expectation_Jz_per_omega=np.array(exp_vals, dtype=float),
        variance_Jz_per_omega=np.array(var_vals, dtype=float),
    )


def _save_and_plot_omega_scan(
    result: DriveOmegaScanResult,
    parquet_path: Path,
    fig_path: Path,
) -> None:
    """Save result to Parquet and generate the ω-scan figure."""
    result.save_parquet(parquet_path)
    print(f"[save] {parquet_path}")

    from src.visualization.ancilla_drive_plots import plot_drive_omega_scan

    plot_drive_omega_scan(result, fig_path)
    print(f"[fig]  {fig_path}")


def generate_phase_omega_scan(force: bool = False) -> None:
    """Phase-modulated ω-scan with Nelder-Mead refinement (parallel).

    Check cache → compute (if needed) → save and plot.
    """
    from src.visualization.ancilla_drive_plots import plot_drive_omega_scan

    csv_p = _parquet_path("phase-omega-scan")
    fig_p = _fig_path("phase-omega-scan")

    result = _check_omega_scan_cache(csv_p, force)
    if result is None:
        result = _compute_omega_scan_core()
        _save_and_plot_omega_scan(result, csv_p, fig_p)
    else:
        plot_drive_omega_scan(result, fig_p)
        print(f"[fig]  {fig_p}")


def generate_phase_optimal_params(force: bool = False) -> None:
    """Phase-modulated optimal parameter evolution vs ω."""
    csv_p = _parquet_path("phase-omega-scan")
    fig_p = _fig_path("phase-optimal-params")

    result = DriveOmegaScanResult.from_parquet(csv_p)
    from src.visualization.ancilla_drive_plots import plot_drive_optimal_params

    plot_drive_optimal_params(result, fig_p)
    print(f"[fig]  {fig_p}")


def generate_phase_combined_sensitivity(force: bool = False) -> None:
    """Combined sensitivity plot + NM expectation/variance plot.

    Reads existing per-ω Parquets (2D slices, random search) and the omega-scan
    result to produce two figures:
        - phase-combined-sensitivity.svg : Δω vs ω for all methods + SQL
        - phase-nm-expectation-variance.svg : ⟨J_z^S⟩ and Var(J_z^S) at NM optimum

    This must be run AFTER the data-generation steps
    (phase-2d-slice-*, phase-random-search, phase-omega-scan).
    """
    fig_p1 = _fig_path("phase-combined-sensitivity")
    fig_p2 = _fig_path("phase-nm-expectation-variance")

    # Load NM result
    omega_scan_pq = _parquet_path("phase-omega-scan")
    if not omega_scan_pq.exists():
        print(
            "[skip] phase-omega-scan.parquet does not exist; run 'phase-omega-scan' first"
        )
        return
    nm_result = DriveOmegaScanResult.from_parquet(omega_scan_pq)

    omega_vals = np.array(PHASE_OMEGA_VALS, dtype=float)
    n_omega = len(omega_vals)

    best_nm = np.full(n_omega, np.nan)
    exp_vals = np.full(n_omega, np.nan)
    var_vals = np.full(n_omega, np.nan)

    nm_omega = nm_result.omega_values
    if len(nm_omega) >= n_omega:
        for i in range(n_omega):
            best_nm[i] = float(nm_result.best_delta_omega_per_omega[i])
            exp_vals[i] = float(nm_result.expectation_Jz_per_omega[i])
            var_vals[i] = float(nm_result.variance_Jz_per_omega[i])

    # Collect per-ω minima from 2D slice and random search Parquets
    best_ax = np.full(n_omega, np.nan)
    best_ay = np.full(n_omega, np.nan)
    best_rs = np.full(n_omega, np.nan)

    def _safe_grid_min(grid: np.ndarray) -> float:
        finite_vals = grid[np.isfinite(grid)]
        if finite_vals.size == 0:
            return np.nan
        return float(np.min(finite_vals))

    for i, omega in enumerate(omega_vals):
        for slice_type, best_arr in [("ax", best_ax), ("ay", best_ay)]:
            tag = f"phase-2d-slice-{slice_type}-azz-omega{omega}"
            csv_p = _parquet_path(tag)
            if csv_p.exists():
                result_slice = Drive2DSliceResult.from_parquet(csv_p)
                best_arr[i] = _safe_grid_min(result_slice.delta_omega_grid)

        tag_rs = f"phase-random-search-omega{omega}"
        csv_p_rs = _parquet_path(tag_rs)
        if csv_p_rs.exists():
            result_rs = DriveRandomSearchResult.from_parquet(csv_p_rs)
            best_rs[i] = result_rs.best_delta_omega

    print(f"  [debug] best_ax finite: {np.sum(np.isfinite(best_ax))} / {n_omega}")
    print(f"  [debug] best_ay finite: {np.sum(np.isfinite(best_ay))} / {n_omega}")
    print(f"  [debug] best_rs finite: {np.sum(np.isfinite(best_rs))} / {n_omega}")
    print(f"  [debug] best_nm finite: {np.sum(np.isfinite(best_nm))} / {n_omega}")

    sql_vals = np.full(n_omega, 0.1)

    # Generate combined sensitivity plot (from src.visualization.ancilla_drive_plots)
    plot_combined_sensitivity(
        omega_vals,
        best_ax,
        best_ay,
        best_rs,
        best_nm,
        sql_vals,
        fig_p1,
    )
    print(f"[fig]  {fig_p1}")

    # Generate NM expectation/variance plot (local function)
    plot_drive_nm_expectation_variance(
        omega_vals,
        exp_vals,
        var_vals,
        fig_p2,
    )
    print(f"[fig]  {fig_p2}")


def generate_phase_fraction_below_sql(force: bool = False) -> None:
    """Fraction of parameter space below SQL vs ω for all methods.

    Reads existing per-ω Parquets (2D slices, random search) and computes
    the fraction of points whose Δω falls below the SQL for each ω.
    """
    fig_p = _fig_path("phase-fraction-below-sql")

    omega_vals = np.array(PHASE_OMEGA_VALS, dtype=float)
    n_omega = len(omega_vals)

    fractions_ax = np.full(n_omega, np.nan)
    fractions_ay = np.full(n_omega, np.nan)
    fractions_rs = np.full(n_omega, np.nan)

    for i, omega in enumerate(omega_vals):
        tag_ax = f"phase-2d-slice-ax-azz-omega{omega}"
        csv_ax = _parquet_path(tag_ax)
        if csv_ax.exists():
            result = Drive2DSliceResult.from_parquet(csv_ax)
            fractions_ax[i] = (
                np.sum(result.delta_omega_grid < result.sql)
                / result.delta_omega_grid.size
            )

        tag_ay = f"phase-2d-slice-ay-azz-omega{omega}"
        csv_ay = _parquet_path(tag_ay)
        if csv_ay.exists():
            result = Drive2DSliceResult.from_parquet(csv_ay)
            fractions_ay[i] = (
                np.sum(result.delta_omega_grid < result.sql)
                / result.delta_omega_grid.size
            )

        tag_rs = f"phase-random-search-omega{omega}"
        csv_rs = _parquet_path(tag_rs)
        if csv_rs.exists():
            result = DriveRandomSearchResult.from_parquet(csv_rs)
            fractions_rs[i] = np.sum(result.delta_omega_values < result.sql) / len(
                result.delta_omega_values
            )

    plot_drive_fraction_below_sql(
        omega_vals,
        fractions_ax,
        fractions_ay,
        fractions_rs,
        fig_p,
    )
    print(f"[fig]  {fig_p}")


def generate_phase_cross_experiment_comparison(force: bool = False) -> None:
    """Comparison of fixed-drive (2026-05-18) vs modulated-drive (2026-05-19)
    ω-scan results.

    Loads both Parquets, interpolates the sparse 2026-05-18 data to the fine
    50-point ω grid of the 2026-05-19 scan, and produces a 2×1 figure
    showing Δω vs ω (upper) and the ratio Δω_19/Δω_18 (lower).
    """
    fig_p = _fig_path("phase-cross-experiment-comparison")

    # Load modulated-drive result (2026-05-19, 50 points)
    pq_19 = _parquet_path("phase-omega-scan")
    if not pq_19.exists():
        print(
            "[skip] 20260519-phase-omega-scan.parquet does not exist; "
            "run 'phase-omega-scan' first"
        )
        return
    result_19 = DriveOmegaScanResult.from_parquet(pq_19)

    # Load fixed-drive result (2026-05-18, 5 points)
    csv_18 = REPORTS_DIR / "20260518" / "raw_data" / "20260518-drive-omega-scan.csv"
    if not csv_18.exists():
        print(
            "[skip] 20260518-drive-omega-scan.csv does not exist; "
            "run 'drive-omega-scan' first"
        )
        return
    result_18 = DriveOmegaScanResult.from_parquet(csv_18)

    omega_fine = result_19.omega_values
    omega_coarse = result_18.omega_values
    delta_18_coarse = result_18.best_delta_omega_per_omega
    delta_18_fine = np.interp(omega_fine, omega_coarse, delta_18_coarse)

    sql_fine = result_19.sql_values

    plot_drive_cross_experiment_comparison(
        omega_values=omega_fine,
        best_delta_19=result_19.best_delta_omega_per_omega,
        best_delta_18=delta_18_fine,
        sql_values=sql_fine,
        save_path=fig_p,
    )
    print(f"[fig]  {fig_p}")


# ============================================================================
# CLI Entry Point
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate 2026-05-19 report figures and Parquet data",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run all simulations (overwrite existing Parquets)",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Generate only one dataset, e.g. 'phase-decoupled-baseline'",
    )
    args = parser.parse_args()

    # Ensure directories exist
    (REPORTS_DIR / PHASE_DATE / "raw_data").mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / PHASE_DATE / "figures").mkdir(parents=True, exist_ok=True)

    tasks = {
        "phase-decoupled-baseline": generate_phase_decoupled_baseline,
        "phase-2d-slice-ax-azz": generate_phase_2d_slice_ax_azz,
        "phase-2d-slice-ay-azz": generate_phase_2d_slice_ay_azz,
        "phase-random-search": generate_phase_random_search,
        "phase-omega-scan": generate_phase_omega_scan,
        "phase-optimal-params": generate_phase_optimal_params,
        "phase-combined-sensitivity": generate_phase_combined_sensitivity,
        "phase-fraction-below-sql": generate_phase_fraction_below_sql,
        "phase-cross-experiment-comparison": generate_phase_cross_experiment_comparison,
    }

    if args.only:
        if args.only not in tasks:
            print(f"Unknown dataset '{args.only}'. Options: {list(tasks.keys())}")
            sys.exit(1)
        tasks[args.only](force=args.force)
    else:
        for name, func in tasks.items():
            print(f"\n=== {name} ===")
            func(force=args.force)

    print("\nDone.")


if __name__ == "__main__":
    main()
