r"""
Local module for the 2026-06-01 Heisenberg-Limit MZI: NOON & Twin-Fock report.

Contains all code exclusive to this report:
- Standard Twin-Fock |N/2, N/2⟩ state (unlike the uniform-superposition Twin-Fock in src/)
- Simplified ancilla-free MZI evolution (BS1 → phase → BS2)
- Sensitivity computation via Classical Fisher Information (CFI) from
  the full number-difference distribution P(m|θ), plus QFI bound
- Parquet-serializable dataclass for grid results
- Log-log scaling exponent fitting
- Plot functions for Δθ vs θ and Δθ vs N
- CLI pipeline for generating all data and figures

Usage:
    uv run python reports/20260601/heisenberg_limit_noon_twinfock.py --force
    uv run python reports/20260601/heisenberg_limit_noon_twinfock.py --only noon
    uv run python reports/20260601/heisenberg_limit_noon_twinfock.py --only twin_fock_std

This module is importable via ``importlib.import_module("reports.20260601.heisenberg_limit_noon_twinfock")``.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.analysis.scaling_fit import fit_scaling_exponent
from src.physics.mzi_simulation import (
    compute_mzi_sensitivity_grid,
)
from src.physics.mzi_states import (
    input_state_factory,
    standard_twin_fock_state,
)
from src.utils.paths import report_path_fn
from src.utils.serialization import ParquetSerializable

# Force non-interactive backend before any plotting imports.
if "MPLBACKEND" not in os.environ:
    os.environ["MPLBACKEND"] = "Agg"

sns.set_theme(style="whitegrid")

# ============================================================================
# Constants
# ============================================================================

REPORTS_DIR = Path(__file__).resolve().parent.parent.parent / "reports"
REPORT_DATE = "20260601"
t_hold: float = 10.0  # Holding time (used in dataclass and plots)

# Parameter sweep ranges
NOON_N_RANGE: list[int] = list(range(1, 41))  # 1..40
TF_N_RANGE: list[int] = list(range(2, 41, 2))  # Even 2..40
OMEGA_RANGE: tuple[float, float] = (0.1, 5.0)
OMEGA_STEP: float = 0.1

# Scaling fit
ALPHA_EXPECTED_NOON: float = (
    -1.0
)  # α = scaling exponent for NOON Δω ∝ N^α (Heisenberg → α = -1.0)
ALPHA_TOL: float = 0.02  # Tolerance on fitted α for PASS/FAIL determination


# ============================================================================
# Path Helpers
# ============================================================================


_parquet_path, _fig_path = report_path_fn(REPORTS_DIR, REPORT_DATE)


# ============================================================================
# State Preparation Dispatch
# ============================================================================


def _prepare_state(state_type: str, N: int, max_photons: int) -> np.ndarray:
    """Dispatch state creation by type.

    Args:
        state_type: ``"noon"`` or ``"twin_fock_std"``.
        N: Total photon number.
        max_photons: Hilbert space truncation per mode.

    Returns:
        State vector in the two-mode Fock basis.

    Raises:
        ValueError: If state_type is unknown.
    """
    match state_type:
        case "noon":
            return input_state_factory("noon", N, max_photons)
        case "twin_fock_std":
            return standard_twin_fock_state(N, max_photons)
        case _:
            raise ValueError(f"Unknown state_type: {state_type}")


# ============================================================================
# MziSensitivityData Dataclass
# ============================================================================


@dataclass
class MziSensitivityData(ParquetSerializable):
    r"""All sensitivity data for one state type across :math:`N` and :math:`\omega`.

    Stores a 2D grid indexed by ``(N, omega)``, plus per-:math:`N` QFI bounds.

    The primary sensitivity metric is :math:`\Delta\omega_C` (Classical Fisher
    Information from the full :math:`P(m|\omega)` distribution). The error-
    propagation :math:`\Delta\omega_{\text{EP}}` is retained as a secondary
    diagnostic.

    Attributes:
        state_type: ``"noon"`` or ``"twin_fock_std"``.
        N_values: Array of :math:`N` values, shape ``(n_N,)``.
        omega_values: Array of :math:`\omega` values, shape ``(n_omega,)``.
        expectation_grid: :math:`\langle J_z\rangle` at each ``(N, omega)``.
        variance_grid: :math:`\text{Var}(J_z)` at each ``(N, omega)``.
        derivative_grid: :math:`\partial\langle J_z\rangle/\partial\omega`.
        delta_omega_ep_grid: :math:`\Delta\omega_{\text{EP}}`.
        delta_omega_q_per_N: :math:`\Delta\omega_Q` per :math:`N` (length ``n_N``).
        fisher_classical_grid: :math:`F_C` at each ``(N, omega)``.
        delta_omega_c_grid: :math:`\Delta\omega_C` at each ``(N, omega)``.
        t_hold: Holding time.
    """

    state_type: str
    N_values: np.ndarray
    omega_values: np.ndarray
    expectation_grid: np.ndarray
    variance_grid: np.ndarray
    derivative_grid: np.ndarray
    delta_omega_ep_grid: np.ndarray
    delta_omega_q_per_N: np.ndarray
    fisher_classical_grid: np.ndarray
    delta_omega_c_grid: np.ndarray
    t_hold: float = t_hold

    _PARQUET_COLUMNS: ClassVar[list[str]] = [
        "state_type",
        "N",
        "omega",
        "expectation",
        "variance",
        "derivative",
        "delta_omega_ep",
        "delta_omega_q",
        "fisher_classical",
        "delta_omega_c",
        "t_hold",
    ]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to long-format DataFrame (one row per N, θ combination)."""
        n_N = len(self.N_values)
        n_omega = len(self.omega_values)
        rows: list[dict] = []
        for i in range(n_N):
            for j in range(n_omega):
                dt_ep = float(self.delta_omega_ep_grid[i, j])
                dt_c = float(self.delta_omega_c_grid[i, j])
                rows.append(
                    {
                        "state_type": self.state_type,
                        "N": int(self.N_values[i]),
                        "omega": float(self.omega_values[j]),
                        "expectation": float(self.expectation_grid[i, j]),
                        "variance": float(self.variance_grid[i, j]),
                        "derivative": float(self.derivative_grid[i, j]),
                        "delta_omega_ep": (
                            dt_ep if np.isfinite(dt_ep) else float("inf")
                        ),
                        "delta_omega_q": float(self.delta_omega_q_per_N[i]),
                        "fisher_classical": float(self.fisher_classical_grid[i, j]),
                        "delta_omega_c": (dt_c if np.isfinite(dt_c) else float("inf")),
                        "t_hold": self.t_hold,
                    }
                )
        return pd.DataFrame(rows)

    @classmethod
    def from_parquet(cls, path: str | Path) -> MziSensitivityData:
        """Load from Parquet, reconstructing the 2D grids."""
        df = pd.read_parquet(path)
        cls._validate_columns(df)
        state_type = str(df["state_type"].iloc[0])
        t_hold = float(df["t_hold"].iloc[0])
        N_vals = sorted(df["N"].unique())
        omega_vals = sorted(df["omega"].unique())
        n_N = len(N_vals)
        n_omega = len(omega_vals)

        # Build lookup for reconstruction
        expectation_grid = np.full((n_N, n_omega), np.nan, dtype=float)
        variance_grid = np.full((n_N, n_omega), np.nan, dtype=float)
        derivative_grid = np.full((n_N, n_omega), np.nan, dtype=float)
        delta_omega_ep_grid = np.full((n_N, n_omega), np.nan, dtype=float)
        delta_omega_q_per_N = np.full(n_N, np.nan, dtype=float)
        fisher_classical_grid = np.full((n_N, n_omega), np.nan, dtype=float)
        delta_omega_c_grid = np.full((n_N, n_omega), np.nan, dtype=float)

        for _, row in df.iterrows():
            n_idx = N_vals.index(int(row["N"]))
            t_idx = omega_vals.index(float(row["omega"]))
            expectation_grid[n_idx, t_idx] = row["expectation"]
            variance_grid[n_idx, t_idx] = row["variance"]
            derivative_grid[n_idx, t_idx] = row["derivative"]
            delta_omega_ep_grid[n_idx, t_idx] = row["delta_omega_ep"]
            # All rows for the same N must have the same QFI bound
            dq = float(row["delta_omega_q"])
            if np.isnan(delta_omega_q_per_N[n_idx]):
                delta_omega_q_per_N[n_idx] = dq
            elif not np.isclose(delta_omega_q_per_N[n_idx], dq, rtol=1e-10):
                raise ValueError(
                    f"Inconsistent delta_omega_q for N={row['N']}: "
                    f"expected {delta_omega_q_per_N[n_idx]}, got {dq}. "
                    f"Regenerate the file."
                )
            fisher_classical_grid[n_idx, t_idx] = float(row["fisher_classical"])
            delta_omega_c_grid[n_idx, t_idx] = float(row["delta_omega_c"])

        return cls(
            state_type=state_type,
            N_values=np.array(N_vals, dtype=int),
            omega_values=np.array(omega_vals, dtype=float),
            expectation_grid=expectation_grid,
            variance_grid=variance_grid,
            derivative_grid=derivative_grid,
            delta_omega_ep_grid=delta_omega_ep_grid,
            delta_omega_q_per_N=delta_omega_q_per_N,
            fisher_classical_grid=fisher_classical_grid,
            delta_omega_c_grid=delta_omega_c_grid,
            t_hold=t_hold,
        )


# ============================================================================
# Generate Omega Scan (Single N)
# ============================================================================


def generate_omega_scan(
    state_type: str,
    N: int,
    omega_grid: np.ndarray,
    max_photons: int | None = None,
    t_hold: float = t_hold,
) -> MziSensitivityData:
    r"""Run a :math:`\omega` scan for a single :math:`N` value.

    Args:
        state_type: ``"noon"`` or ``"twin_fock_std"``.
        N: Total photon number.
        omega_grid: :math:`\omega` values to scan.
        max_photons: Hilbert space truncation (defaults to ``N``).
        t_hold: Holding time.

    Returns:
        MziSensitivityData with one N value.
    """
    if max_photons is None:
        max_photons = N
    state = _prepare_state(state_type, N, max_photons)
    skip_bs1 = state_type == "noon"
    result = compute_mzi_sensitivity_grid(
        state,
        omega_grid,
        max_photons,
        t_hold=t_hold,
        skip_bs1=skip_bs1,
    )

    omega_arr = np.asarray(result["omega_values"], dtype=float)
    return MziSensitivityData(
        state_type=state_type,
        N_values=np.array([N], dtype=int),
        omega_values=omega_arr,
        expectation_grid=np.atleast_2d(np.asarray(result["expectation_values"])),
        variance_grid=np.atleast_2d(np.asarray(result["variance_values"])),
        derivative_grid=np.atleast_2d(np.asarray(result["derivative_values"])),
        delta_omega_ep_grid=np.atleast_2d(np.asarray(result["delta_omega_ep"])),
        delta_omega_q_per_N=np.array([float(result["delta_omega_q"])]),
        fisher_classical_grid=np.atleast_2d(np.asarray(result["fisher_classical"])),
        delta_omega_c_grid=np.atleast_2d(np.asarray(result["delta_omega_c"])),
        t_hold=t_hold,
    )


# ============================================================================
# Generate Full Data (All N)
# ============================================================================


def _generate_single_N_data(
    state_type: str,
    N: int,
    omega_grid: np.ndarray,
    t_hold: float = t_hold,
) -> MziSensitivityData | None:
    r"""Run omega scan for a single N, returning None on failure.

    Args:
        state_type: ``"noon"`` or ``"twin_fock_std"``.
        N: Total photon number.
        omega_grid: :math:`\omega` values to scan.
        t_hold: Holding time.

    Returns:
        MziSensitivityData with one N value, or None if the simulation fails.
    """
    try:
        return generate_omega_scan(
            state_type,
            N,
            omega_grid,
            max_photons=N,
            t_hold=t_hold,
        )
    except (ValueError, AssertionError) as exc:
        print(f"Warning: N={N} failed for {state_type}: {exc}")
        return None


def _concatenate_scan_results(
    scan_results: list[MziSensitivityData],
    state_type: str,
    t_hold: float,
) -> MziSensitivityData:
    """Concatenate a list of single-N scan results into a single MziSensitivityData.

    Args:
        scan_results: List of MziSensitivityData, each with one N value.
        state_type: State type label for the combined result.
        t_hold: Holding time.

    Returns:
        Combined MziSensitivityData with all N values.
    """
    return MziSensitivityData(
        state_type=state_type,
        N_values=np.concatenate([r.N_values for r in scan_results]).astype(int),
        omega_values=scan_results[0].omega_values,
        expectation_grid=np.concatenate([r.expectation_grid for r in scan_results]),
        variance_grid=np.concatenate([r.variance_grid for r in scan_results]),
        derivative_grid=np.concatenate([r.derivative_grid for r in scan_results]),
        delta_omega_ep_grid=np.concatenate(
            [r.delta_omega_ep_grid for r in scan_results]
        ),
        delta_omega_q_per_N=np.concatenate(
            [r.delta_omega_q_per_N for r in scan_results]
        ),
        fisher_classical_grid=np.concatenate(
            [r.fisher_classical_grid for r in scan_results]
        ),
        delta_omega_c_grid=np.concatenate([r.delta_omega_c_grid for r in scan_results]),
        t_hold=t_hold,
    )


def generate_full_data(
    state_type: str,
    N_range: list[int],
    omega_grid: np.ndarray,
    t_hold: float = t_hold,
) -> MziSensitivityData:
    r"""Generate sensitivity data for all :math:`N` in a range.

    Args:
        state_type: ``"noon"`` or ``"twin_fock_std"``.
        N_range: List of :math:`N` values.
        omega_grid: :math:`\omega` values to scan.
        t_hold: Holding time.

    Returns:
        MziSensitivityData with all N values.
    """
    scan_results: list[MziSensitivityData] = []
    for idx, N in enumerate(N_range):
        print(
            f"  Sweeping {state_type} N={N} ({idx + 1}/{len(N_range)})...",
            flush=True,
        )
        scan = _generate_single_N_data(state_type, N, omega_grid, t_hold=t_hold)
        if scan is not None:
            scan_results.append(scan)

    if not scan_results:
        raise RuntimeError(f"No valid N values for state_type={state_type}")

    return _concatenate_scan_results(scan_results, state_type, t_hold)


def _maybe_generate_full_data(
    st: str,
    n_range: list[int],
    label: str,
    omega_grid: np.ndarray,
    force: bool,
    only: str | None,
) -> MziSensitivityData | None:
    r"""Load or generate sensitivity data for one state type.

    If ``only`` is set and does not match ``st``, returns ``None``.
    Otherwise loads from Parquet (if exists and not forced) or generates
    fresh data via :func:`generate_full_data`.

    Args:
        st: State type key (e.g. ``"noon"`` or ``"twin_fock_std"``).
        n_range: List of :math:`N` values to simulate.
        label: Human-readable name for logging.
        omega_grid: :math:`\omega` grid.
        force: Re-generate even if Parquet exists.
        only: If set, only load/generate for matching state type.

    Returns:
        MziSensitivityData or None if filtered out by ``only``.
    """
    if only is not None and st != only:
        return None

    pq_path = _parquet_path(f"{st}_sensitivity")
    if pq_path.exists() and not force:
        print(f"Loading existing data for {label} from {pq_path}")
        return MziSensitivityData.from_parquet(pq_path)

    print(f"Generating {label} sensitivity data (N={n_range[0]}..{n_range[-1]})")
    data = generate_full_data(st, n_range, omega_grid, t_hold=t_hold)
    data.save_parquet(pq_path)
    print(f"  Saved to {pq_path}")
    return data


# ============================================================================
# Scaling Exponent Fitting
# ============================================================================


# ============================================================================
# Plot Functions
# ============================================================================
# Analyse Best/Worst Sensitivity from a 2D Grid
# ============================================================================


def analyse_best_worst_sensitivity(
    N_values: np.ndarray,
    omega_values: np.ndarray,
    sensitivity_grid: np.ndarray,
) -> dict:
    """Find best (min) and worst (max) sensitivity at each N.

    Args:
        N_values: Array of N values, shape ``(n_N,)``.
        omega_values: Array of θ values, shape ``(n_omega,)``.
        sensitivity_grid: 2D array of sensitivity values, shape ``(n_N, n_omega)``.

    Returns:
        Dictionary with keys:
        - ``N_values``: Array of N values.
        - ``best_sensitivity``: Minimum sensitivity at each N.
        - ``best_omega``: θ where minimum occurs.
        - ``worst_sensitivity``: Maximum finite sensitivity at each N.
        - ``worst_omega``: θ where maximum occurs.
    """
    n_N = len(N_values)
    best_sens = np.full(n_N, np.inf, dtype=float)
    best_th = np.full(n_N, np.nan, dtype=float)
    worst_sens = np.full(n_N, -np.inf, dtype=float)
    worst_th = np.full(n_N, np.nan, dtype=float)

    for i in range(n_N):
        slice_ = sensitivity_grid[i, :]
        finite_mask = np.isfinite(slice_)
        if np.any(finite_mask):
            full_indices = np.where(finite_mask)[0]

            # Best (minimum)
            b_idx = int(np.argmin(slice_[finite_mask]))
            actual_idx = full_indices[b_idx]
            best_sens[i] = float(slice_[actual_idx])
            best_th[i] = float(omega_values[actual_idx])

            # Worst (maximum finite)
            w_idx = int(np.argmax(slice_[finite_mask]))
            actual_w_idx = full_indices[w_idx]
            worst_sens[i] = float(slice_[actual_w_idx])
            worst_th[i] = float(omega_values[actual_w_idx])

    return {
        "N_values": N_values.copy(),
        "best_sensitivity": best_sens,
        "best_omega": best_th,
        "worst_sensitivity": worst_sens,
        "worst_omega": worst_th,
    }


# ============================================================================
# Plot Functions
# ============================================================================


def plot_delta_omega_overlay(
    data: MziSensitivityData,
    selected_N: list[int] | None = None,
    save_path: str | Path | None = None,
) -> Path:
    """Overlay Δθ_C and Δθ_Q vs θ for multiple N values on a single panel.

    Each N gets a unique colour from the *viridis* colormap.  Solid lines
    show Δθ_C, dashed horizontal lines show the corresponding QFI bound.
    The y-axis uses a log scale so that different N are clearly separated.

    Args:
        data: Sensitivity data containing all N values.
        selected_N: Which N to include (defaults to
            ``[1, 2, 4, 10, 20, 30, 40]`` for NOON,
            ``[2, 4, 10, 20, 30, 40]`` for Twin-Fock).
        save_path: Output SVG path.  Auto-generated if None.

    Returns:
        Path to saved SVG.
    """
    if selected_N is None:
        if data.state_type == "noon":
            selected_N = [1, 2, 4, 10, 20, 30, 40]
        else:
            selected_N = [2, 4, 10, 20, 30, 40]

    if save_path is None:
        save_path = _fig_path(f"{data.state_type}_delta_omega_comparison")
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    state_label = data.state_type.replace("_", " ").title()

    fig, ax = plt.subplots(figsize=(10, 6))

    cmap = plt.colormaps["viridis"]
    colors = cmap(np.linspace(0.15, 0.85, len(selected_N)))

    for idx, N_val in enumerate(selected_N):
        match = np.where(data.N_values == N_val)[0]
        if len(match) == 0:
            continue
        n_idx = match[0]

        omega = data.omega_values
        dt_c = data.delta_omega_c_grid[n_idx, :]
        dt_q = data.delta_omega_q_per_N[n_idx]

        # Δθ_C (solid line)
        c_finite = np.isfinite(dt_c)
        if np.any(c_finite):
            ax.semilogy(
                omega[c_finite],
                dt_c[c_finite],
                color=colors[idx],
                linewidth=1.5,
                label=rf"N={N_val}  $\Delta\omega_{{\mathrm{{C}}}}$",
            )

        # Δθ_Q (dashed horizontal line)
        ax.axhline(
            y=float(dt_q),
            color=colors[idx],
            linestyle="--",
            linewidth=1.0,
            alpha=0.6,
        )

    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$\Delta\omega$")
    ax.set_title(rf"{state_label} — Phase Sensitivity vs $\omega$")
    ax.legend(fontsize=8, loc="best", ncol=1)
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_standard_deviation_comparison(
    data_noon: MziSensitivityData | None,
    data_tf: MziSensitivityData | None,
    save_path: str | Path | None = None,
) -> Path:
    r"""Overlaid line plot of the probe variance :math:`\text{Var}(J_z)` vs N.

    The probe variance is computed from the stored QFI bound:

    .. math::

        \text{Var}(J_z)_{\text{probe}} = \frac{1}{4 t_hold^2 \cdot \Delta\omega_Q^2}

    For NOON this gives :math:`\text{Var}(J_z) = N^2/4` (Heisenberg-limited input).
    For Twin-Fock after BS1 this gives
    :math:`\text{Var}(J_z) = N(N+2)/8` (near-Heisenberg).

    The y-axis shows :math:`\text{Var}(J_z)` on a log-log scale.

    Args:
        data_noon: NOON sensitivity data (or None).
        data_tf: Twin-Fock sensitivity data (or None).
        save_path: Output SVG path.

    Returns:
        Path to saved SVG.
    """
    if save_path is None:
        save_path = _fig_path("variance_histogram")
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    colours = {"noon": "C0", "twin_fock_std": "C1"}
    labels_display = {"noon": "NOON", "twin_fock_std": "Twin-Fock"}
    markers = {"noon": "o", "twin_fock_std": "s"}

    for data, label_st in [
        (data_noon, "noon"),
        (data_tf, "twin_fock_std"),
    ]:
        if data is None:
            continue
        # Probe variance Var(J_z) = 1 / (4 * t_hold² * Δθ_Q²)
        var = 1.0 / (4.0 * t_hold**2 * data.delta_omega_q_per_N**2)
        ax.loglog(
            data.N_values,
            var,
            color=colours[label_st],
            marker=markers[label_st],
            linewidth=1.5,
            label=labels_display[label_st],
        )

    ax.set_xlabel("Total photon number $N$")
    ax.set_ylabel(r"$\mathrm{Var}(J_z)$ (probe variance)")
    ax.set_title(r"Probe Variance $\mathrm{Var}(J_z)$ vs $N$")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_scaling(
    data_noon: MziSensitivityData | None,
    data_tf: MziSensitivityData | None,
    save_path: str | Path | None = None,
    N_min_fit: int = 4,
) -> Path:
    """Log-log plot of best Δθ vs N with analytical QFI bounds and fits.

    Overlays NOON and Twin-Fock scaling data (from Δθ_C) on a single figure.

    Args:
        data_noon: NOON sensitivity data (or None).
        data_tf: Twin-Fock sensitivity data (or None).
        save_path: Output SVG path.
        N_min_fit: Minimum N for exponent fits.

    Returns:
        Path to saved SVG.
    """
    if data_noon is None and data_tf is None:
        raise ValueError("At least one data set must be provided")

    if save_path is None:
        save_path = _fig_path("scaling_comparison")
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Reference lines
    N_ref = np.logspace(0, 1.5, 50)
    ax.plot(
        N_ref,
        1.0 / (t_hold * N_ref),
        "k--",
        alpha=0.4,
        label=r"$\propto 1/N$ (Heisenberg)",
    )
    ax.plot(
        N_ref,
        1.0 / (t_hold * np.sqrt(N_ref)),
        "k:",
        alpha=0.4,
        label=r"$\propto 1/\sqrt{N}$ (SQL)",
    )

    colours = {"noon": "C0", "twin_fock_std": "C1"}
    markers = {"noon": "o", "twin_fock_std": "s"}

    for data, label_name in [(data_noon, "NOON"), (data_tf, "Twin-Fock")]:
        if data is None:
            continue
        colour = colours[data.state_type]
        marker = markers[data.state_type]

        # QFI bound
        ax.loglog(
            data.N_values,
            data.delta_omega_q_per_N,
            f"{colour}--",
            alpha=0.5,
            label=f"{label_name} QFI bound",
        )

        # Best Δθ_C at each N
        analysis = analyse_best_worst_sensitivity(
            data.N_values,
            data.omega_values,
            data.delta_omega_c_grid,
        )
        N_vals = analysis["N_values"]
        best_dt_c = analysis["best_sensitivity"]
        finite = np.isfinite(best_dt_c)
        if np.any(finite):
            ax.loglog(
                N_vals[finite],
                best_dt_c[finite],
                f"{colour}{marker}-",
                label=rf"{label_name} best $\Delta\omega_{{\mathrm{{C}}}}$",
            )

        # Fit exponent
        N_vals_arr = np.array(N_vals, dtype=float)
        best_dt_c_arr = np.array(best_dt_c, dtype=float)
        fit_result = fit_scaling_exponent(
            N_vals_arr, best_dt_c_arr, min_N=int(N_min_fit)
        )
        if fit_result.valid:
            N_fit = fit_result.N_values
            delta_fit = fit_result.C * N_fit**fit_result.alpha
            ax.loglog(
                N_fit,
                delta_fit,
                f"{colour}--",
                alpha=0.7,
                linewidth=1.5,
                label=f"{label_name}: "
                rf"$\alpha = {fit_result.alpha:.3f}$",
            )

    ax.set_xlabel("Total photon number $N$")
    ax.set_ylabel(r"$\Delta\omega$")
    ax.set_title("Phase Sensitivity Scaling in Standard MZI")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_expectation_vs_omega_grid(
    data_noon: MziSensitivityData | None,
    data_tf: MziSensitivityData | None,
    save_path: str | Path | None = None,
) -> Path:
    """Plot ⟨J_z⟩ vs θ for NOON — N=1 (varying) and a representative N>1 (flat).

    This simplification works because for NOON with N ≥ 2 the output
    expectation ⟨J_z⟩ is identically zero at all θ, while N=1 shows the
    familiar single-photon MZI fringe.  The ±σ band is shown as a shaded
    region.

    Shows a single panel with at most two overlaid curves.
    """
    if data_noon is None:
        raise ValueError("NOON data is required for the simplified expectation grid")

    if save_path is None:
        save_path = _fig_path("expectation_grid")
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    omega = data_noon.omega_values

    # N=1: varies sinusoidally
    idx_1 = np.where(data_noon.N_values == 1)[0]
    if len(idx_1) > 0:
        i1 = idx_1[0]
        exp_1 = data_noon.expectation_grid[i1, :]
        var_1 = data_noon.variance_grid[i1, :]
        ax.plot(omega, exp_1, "C0-", linewidth=1.5, label=r"N=1: $\langle J_z \rangle$")
        ax.fill_between(
            omega,
            exp_1 - np.sqrt(var_1),
            exp_1 + np.sqrt(var_1),
            alpha=0.15,
            color="C0",
            label=r"N=1: $\pm\sigma$",
        )

    # Representative N > 1 (use the largest available)
    mask = data_noon.N_values > 1
    if np.any(mask):
        large_idx = np.where(mask)[0][-1]  # largest N
        N_large = int(data_noon.N_values[large_idx])
        exp_large = data_noon.expectation_grid[large_idx, :]
        var_large = data_noon.variance_grid[large_idx, :]
        ax.plot(
            omega,
            exp_large,
            "C1-",
            linewidth=1.5,
            label=rf"N={N_large}: $\langle J_z \rangle$",
        )
        ax.fill_between(
            omega,
            exp_large - np.sqrt(var_large),
            exp_large + np.sqrt(var_large),
            alpha=0.15,
            color="C1",
            label=rf"N={N_large}: $\pm\sigma$",
        )

    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$\langle J_z \rangle$")
    ax.set_title("NOON — Output Expectation (N=1 vs N>1)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ============================================================================
# Plot Orchestration
# ============================================================================


def _maybe_plot_delta_omega_overlays(
    results: dict[str, MziSensitivityData],
    state_configs: list[tuple[str, list[int], str]],
    force: bool,
    only: str | None,
) -> None:
    """Plot Δθ overlay figures for each state type (one per state)."""
    for st, _n_range, _label in state_configs:
        if only is not None and st != only:
            continue
        data = results.get(st)
        if data is None:
            continue
        overlay_path = _fig_path(f"{st}_delta_omega_comparison")
        if not overlay_path.exists() or force:
            sel_N = (
                [1, 2, 4, 10, 20, 30, 40] if st == "noon" else [2, 4, 10, 20, 30, 40]
            )
            plot_delta_omega_overlay(data, selected_N=sel_N, save_path=overlay_path)
            print(f"  Plotted {overlay_path}")


def _maybe_plot_variance_comparison(
    results: dict[str, MziSensitivityData],
    force: bool,
) -> None:
    """Plot probe standard deviation comparison if the file does not exist or force."""
    path = _fig_path("variance_histogram")
    if not path.exists() or force:
        plot_standard_deviation_comparison(
            results.get("noon"),
            results.get("twin_fock_std"),
            save_path=path,
        )
        print(f"  Plotted {path}")


def _maybe_plot_expectation_grid(
    results: dict[str, MziSensitivityData],
    force: bool,
) -> None:
    """Plot simplified expectation grid (NOON-only: N=1 + N>1)."""
    path = _fig_path("expectation_grid")
    if not path.exists() or force:
        plot_expectation_vs_omega_grid(
            results.get("noon"),
            results.get("twin_fock_std"),
            save_path=path,
        )
        print(f"  Plotted {path}")


def _maybe_plot_scaling_comparison(
    results: dict[str, MziSensitivityData],
    force: bool,
) -> None:
    """Plot combined scaling comparison."""
    path = _fig_path("scaling_comparison")
    if not path.exists() or force:
        plot_scaling(
            results.get("noon"),
            results.get("twin_fock_std"),
            save_path=path,
        )
        print(f"  Plotted {path}")


def _generate_plots(
    results: dict[str, MziSensitivityData],
    state_configs: list[tuple[str, list[int], str]],
    force: bool,
    only: str | None,
) -> None:
    """Generate all plots from the computed sensitivity data.

    Args:
        results: Mapping of state type to sensitivity data.
        state_configs: List of ``(state_type, N_range, label)`` tuples.
        force: Re-generate plots even if SVG files exist.
        only: If set, only plot for the specified state type.
    """
    _maybe_plot_delta_omega_overlays(results, state_configs, force, only)
    _maybe_plot_variance_comparison(results, force)
    _maybe_plot_expectation_grid(results, force)
    _maybe_plot_scaling_comparison(results, force)


# ============================================================================
# Main Pipeline
# ============================================================================


def generate_all(
    force: bool = False,
    only: str | None = None,
) -> dict[str, MziSensitivityData]:
    """Generate all data and figures for the report.

    Args:
        force: If True, re-generate even if Parquet files exist.
        only: If set, only generate for this state type ("noon" or "twin_fock_std").

    Returns:
        Dict mapping state_type to MziSensitivityData.
    """
    omega_grid = np.arange(OMEGA_RANGE[0], OMEGA_RANGE[1] + OMEGA_STEP / 2, OMEGA_STEP)

    results: dict[str, MziSensitivityData] = {}

    state_configs: list[tuple[str, list[int], str]] = [
        ("noon", NOON_N_RANGE, "NOON"),
        ("twin_fock_std", TF_N_RANGE, "Twin-Fock"),
    ]

    for st, n_range, label in state_configs:
        data = _maybe_generate_full_data(st, n_range, label, omega_grid, force, only)
        if data is not None:
            results[st] = data

    _generate_plots(results, state_configs, force, only)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Heisenberg-Limit MZI: NOON & Twin-Fock sensitivity simulation",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-generate data and figures even if files exist",
    )
    parser.add_argument(
        "--only",
        type=str,
        choices=["noon", "twin_fock_std"],
        default=None,
        help="Only generate data for this state type",
    )
    args = parser.parse_args()

    generate_all(force=args.force, only=args.only)


if __name__ == "__main__":
    main()
