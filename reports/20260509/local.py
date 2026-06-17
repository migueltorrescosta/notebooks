"""
Local module for the 2026-05-09 ancilla-assisted non-Markovian metrology report.

Implements the three parameter sweeps described in the report (ancilla_sweep,
memory_sweep, time_sweep), a Parquet-serializable result dataclass, and the
CLI pipeline for data and figure generation.

Usage:
    uv run python reports/20260509/local.py --force
    uv run python reports/20260509/local.py --experiment ancilla-nonmarkovian --force
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.physics.pseudomode_system import (  # noqa: F401 — re-exported for tests via importlib
    PseudomodeConfig,
    apply_ancilla_entanglement,
    build_pseudomode_hamiltonian,
    build_pseudomode_lindblad_operators,
    check_pseudomode_occupancy,
    compute_qfi_with_ancilla,
    compute_qfi_without_ancilla,
    create_pseudomode_operators,
    evolve_pseudomode,
    pseudomode_initial_state,
    pseudomode_number_operator,
    qfi_preservation_ratio,
    run_metrology_protocol,
    trace_out_pseudomode,
    trace_out_spin,
    trace_out_spin_and_pseudomode,
    tripartite_operator,
    validate_pseudomode_density,
)
from src.utils.serialization import ParquetSerializable

# Force non-interactive backend before any plotting imports.
if "MPLBACKEND" not in os.environ:
    os.environ["MPLBACKEND"] = "Agg"

sns.set_theme(style="whitegrid")

REPORT_DATE = "20260509"

# Paths
_REPORT_DIR = Path(__file__).resolve().parent

# =============================================================================
# Sweep Defaults (from report Parameter Sweep table)
# =============================================================================

# Default config parameters
DEFAULT_N: int = 20
DEFAULT_K: int = 10
DEFAULT_ALPHA: float = 1.0
DEFAULT_G_SP: float = 0.5
DEFAULT_OMEGA_0: float = 0.0
DEFAULT_LAM: float = 1.0
DEFAULT_T_DECAY: float = 2.0
DEFAULT_TAU: float = 0.1
DEFAULT_DT: float = 0.01

# Ancilla sweep: theta = g_sa * tau  in [0, pi]
ANC_THETA_MIN: float = 0.0
ANC_THETA_MAX: float = np.pi
ANC_N_POINTS: int = 51

# Memory sweep: lambda in [0.05, 10], log-spaced
MEM_LAM_MIN: float = 0.05
MEM_LAM_MAX: float = 10.0
MEM_N_POINTS: int = 40

# Time sweep: T_decay in [0, 10]
TIME_T_MIN: float = 0.0
TIME_T_MAX: float = 10.0
TIME_N_POINTS: int = 51

# Theta values for time sweep overlay
TIME_THETA_VALUES: list[float] = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

# =============================================================================
# Path Helpers
# =============================================================================


def _parquet_path(name: str) -> Path:
    return _REPORT_DIR / "raw_data" / f"{REPORT_DATE}-{name}.parquet"


def _fig_path(name: str) -> Path:
    return _REPORT_DIR / "figures" / f"{REPORT_DATE}-{name}.svg"


# =============================================================================
# Result Dataclass
# =============================================================================


@dataclass
class NonMarkovianSweepData(ParquetSerializable):
    """Result of a single parameter sweep in the non-Markovian ancilla report.

    Each row represents one simulation point (one value of the swept parameter).
    All input parameters are stored alongside computed results so that Parquet
    files are fully self-describing.

    Attributes:
        sweep_type: ``"ancilla"``, ``"memory"``, or ``"time"``.
        sweep_values: Array of primary swept parameter values (theta, lam, or
            T_decay depending on sweep_type).
        ratio_with: QFI preservation ratio with ancilla, R_with = F_Q(T) / F_Q(0).
        ratio_without: QFI preservation ratio without ancilla,
            R_without = F_Q(T) / F_Q(0).
        qfi_with: F_Q(T) with ancilla retained.
        qfi_without: F_Q(T) with ancilla traced out.
        qfi_initial: F_Q(0) before decoherence.
        pm_occupancy: Expectation value <b^dagger b> at final time.
        N: Oscillator Fock truncation.
        K: Pseudomode Fock truncation.
        alpha: Coherent state amplitude.
        g_sp: System-pseudomode coupling strength.
        omega_0: Bath central frequency.
        tau: Ancilla entanglement time.
        dt: RK4 time step.
        lam: Bath correlation rate (pseudomode damping).
        T_decay: Decoherence evolution time.
        theta: Ancilla rotation angle theta = g_sa * tau.
    """

    _PARQUET_COLUMNS: ClassVar[list[str]] = [
        "sweep_type",
        "sweep_value",
        "ratio_with",
        "ratio_without",
        "qfi_with",
        "qfi_without",
        "qfi_initial",
        "pm_occupancy",
        "N",
        "K",
        "alpha",
        "g_sp",
        "omega_0",
        "tau",
        "dt",
        "lam",
        "T_decay",
        "theta",
    ]

    sweep_type: str
    sweep_values: np.ndarray

    ratio_with: np.ndarray
    ratio_without: np.ndarray
    qfi_with: np.ndarray
    qfi_without: np.ndarray
    qfi_initial: np.ndarray
    pm_occupancy: np.ndarray

    N: int
    K: int
    alpha: float
    g_sp: float
    omega_0: float
    tau: float
    dt: float
    lam: float
    T_decay: float
    theta: float

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to long-format DataFrame (one row per sweep point)."""
        n = len(self.sweep_values)
        dtype_float = np.float64
        return pd.DataFrame(
            {
                "sweep_type": [self.sweep_type] * n,
                "sweep_value": np.asarray(self.sweep_values, dtype=dtype_float),
                "ratio_with": np.asarray(self.ratio_with, dtype=dtype_float),
                "ratio_without": np.asarray(self.ratio_without, dtype=dtype_float),
                "qfi_with": np.asarray(self.qfi_with, dtype=dtype_float),
                "qfi_without": np.asarray(self.qfi_without, dtype=dtype_float),
                "qfi_initial": np.asarray(self.qfi_initial, dtype=dtype_float),
                "pm_occupancy": np.asarray(self.pm_occupancy, dtype=dtype_float),
                "N": [self.N] * n,
                "K": [self.K] * n,
                "alpha": [self.alpha] * n,
                "g_sp": [self.g_sp] * n,
                "omega_0": [self.omega_0] * n,
                "tau": [self.tau] * n,
                "dt": [self.dt] * n,
                "lam": [self.lam] * n,
                "T_decay": [self.T_decay] * n,
                "theta": [self.theta] * n,
            }
        )

    @classmethod
    def from_parquet(cls, path: str | Path) -> NonMarkovianSweepData:
        """Load from Parquet, reconstructing arrays from long-format rows."""
        df = pd.read_parquet(path)
        cls._validate_columns(df)

        sweep_type = str(df["sweep_type"].iloc[0])
        sweep_values = df["sweep_value"].to_numpy(dtype=np.float64)
        ratio_with = df["ratio_with"].to_numpy(dtype=np.float64)
        ratio_without = df["ratio_without"].to_numpy(dtype=np.float64)
        qfi_with = df["qfi_with"].to_numpy(dtype=np.float64)
        qfi_without = df["qfi_without"].to_numpy(dtype=np.float64)
        qfi_initial = df["qfi_initial"].to_numpy(dtype=np.float64)
        pm_occupancy = df["pm_occupancy"].to_numpy(dtype=np.float64)

        return cls(
            sweep_type=sweep_type,
            sweep_values=sweep_values,
            ratio_with=ratio_with,
            ratio_without=ratio_without,
            qfi_with=qfi_with,
            qfi_without=qfi_without,
            qfi_initial=qfi_initial,
            pm_occupancy=pm_occupancy,
            N=int(df["N"].iloc[0]),
            K=int(df["K"].iloc[0]),
            alpha=float(df["alpha"].iloc[0]),
            g_sp=float(df["g_sp"].iloc[0]),
            omega_0=float(df["omega_0"].iloc[0]),
            tau=float(df["tau"].iloc[0]),
            dt=float(df["dt"].iloc[0]),
            lam=float(df["lam"].iloc[0]),
            T_decay=float(df["T_decay"].iloc[0]),
            theta=float(df["theta"].iloc[0]),
        )


# =============================================================================
# Sweep Functions
# =============================================================================


def ancilla_sweep(
    theta_grid: np.ndarray,
    base_config: PseudomodeConfig | None = None,
) -> NonMarkovianSweepData:
    """Sweep over ancilla rotation angle theta = g_sa * tau.

    Fixes lambda and T_decay at their default values.  Varies g_sa to
    achieve each theta in the grid (tau is held fixed).

    Args:
        theta_grid: Array of theta values to sweep.
        base_config: Base configuration.  Uses defaults if None.

    Returns:
        NonMarkovianSweepData with sweep_type="ancilla".  The ``theta``
        metadata field is set to 0.0 (sentinel) because the actual theta
        values are stored in ``sweep_values``.
    """
    if base_config is None:
        base_config = PseudomodeConfig(N=DEFAULT_N, K=DEFAULT_K)

    n = len(theta_grid)
    ratio_with = np.zeros(n, dtype=np.float64)
    ratio_without = np.zeros(n, dtype=np.float64)
    qfi_with = np.zeros(n, dtype=np.float64)
    qfi_without = np.zeros(n, dtype=np.float64)
    qfi_initial = np.zeros(n, dtype=np.float64)
    pm_occupancy = np.zeros(n, dtype=np.float64)

    for i, theta in enumerate(theta_grid):
        g_sa = theta / base_config.tau if theta > 0 else 0.0
        config = PseudomodeConfig(
            N=base_config.N,
            K=base_config.K,
            alpha=base_config.alpha,
            g_sa=g_sa,
            tau=base_config.tau,
            g_sp=base_config.g_sp,
            omega_0=base_config.omega_0,
            lam=base_config.lam,
            T_decay=base_config.T_decay,
            dt=base_config.dt,
        )
        result = run_metrology_protocol(config)

        ratio_with[i] = result["ratio_with"]
        ratio_without[i] = result["ratio_without"]
        qfi_with[i] = result["qfi_with"]
        qfi_without[i] = result["qfi_without"]
        qfi_initial[i] = result["qfi_initial"]
        pm_occupancy[i] = result["pm_occupancy"]

        if (i + 1) % 10 == 0 or i == 0 or i == n - 1:
            print(
                f"  theta={theta:.4f}: R_with={ratio_with[i]:.6f}, "
                f"R_without={ratio_without[i]:.6f}, "
                f"pm_occ={pm_occupancy[i]:.4f}",
                flush=True,
            )

    return NonMarkovianSweepData(
        sweep_type="ancilla",
        sweep_values=np.asarray(theta_grid, dtype=np.float64),
        ratio_with=ratio_with,
        ratio_without=ratio_without,
        qfi_with=qfi_with,
        qfi_without=qfi_without,
        qfi_initial=qfi_initial,
        pm_occupancy=pm_occupancy,
        N=base_config.N,
        K=base_config.K,
        alpha=base_config.alpha,
        g_sp=base_config.g_sp,
        omega_0=base_config.omega_0,
        tau=base_config.tau,
        dt=base_config.dt,
        lam=base_config.lam,
        T_decay=base_config.T_decay,
        theta=0.0,  # sentinel: theta varies across sweep values
    )


def memory_sweep(
    lam_grid: np.ndarray,
    theta: float,
    base_config: PseudomodeConfig | None = None,
) -> NonMarkovianSweepData:
    """Sweep over bath correlation rate lambda at fixed theta.

    Args:
        lam_grid: Array of lambda values to sweep (log-spaced recommended).
        theta: Fixed ancilla rotation angle (optimal theta* from ancilla sweep).
        base_config: Base configuration.  Uses defaults if None.

    Returns:
        NonMarkovianSweepData with sweep_type="memory".
    """
    if base_config is None:
        base_config = PseudomodeConfig(N=DEFAULT_N, K=DEFAULT_K)

    n = len(lam_grid)
    ratio_with = np.zeros(n, dtype=np.float64)
    ratio_without = np.zeros(n, dtype=np.float64)
    qfi_with = np.zeros(n, dtype=np.float64)
    qfi_without = np.zeros(n, dtype=np.float64)
    qfi_initial = np.zeros(n, dtype=np.float64)
    pm_occupancy = np.zeros(n, dtype=np.float64)

    g_sa = theta / base_config.tau if theta > 0 else 0.0

    for i, lam in enumerate(lam_grid):
        config = PseudomodeConfig(
            N=base_config.N,
            K=base_config.K,
            alpha=base_config.alpha,
            g_sa=g_sa,
            tau=base_config.tau,
            g_sp=base_config.g_sp,
            omega_0=base_config.omega_0,
            lam=lam,
            T_decay=base_config.T_decay,
            dt=base_config.dt,
        )
        result = run_metrology_protocol(config)

        ratio_with[i] = result["ratio_with"]
        ratio_without[i] = result["ratio_without"]
        qfi_with[i] = result["qfi_with"]
        qfi_without[i] = result["qfi_without"]
        qfi_initial[i] = result["qfi_initial"]
        pm_occupancy[i] = result["pm_occupancy"]

        if (i + 1) % 10 == 0 or i == 0 or i == n - 1:
            print(
                f"  lam={lam:.6f}: R_with={ratio_with[i]:.6f}, "
                f"R_without={ratio_without[i]:.6f}, "
                f"pm_occ={pm_occupancy[i]:.4f}",
                flush=True,
            )

    return NonMarkovianSweepData(
        sweep_type="memory",
        sweep_values=np.asarray(lam_grid, dtype=np.float64),
        ratio_with=ratio_with,
        ratio_without=ratio_without,
        qfi_with=qfi_with,
        qfi_without=qfi_without,
        qfi_initial=qfi_initial,
        pm_occupancy=pm_occupancy,
        N=base_config.N,
        K=base_config.K,
        alpha=base_config.alpha,
        g_sp=base_config.g_sp,
        omega_0=base_config.omega_0,
        tau=base_config.tau,
        dt=base_config.dt,
        lam=float(lam_grid[-1]),
        T_decay=base_config.T_decay,
        theta=theta,
    )


def time_sweep(
    T_grid: np.ndarray,
    theta: float,
    lam: float = DEFAULT_LAM,
    base_config: PseudomodeConfig | None = None,
) -> NonMarkovianSweepData:
    """Sweep over decoherence time T at fixed theta and lambda.

    Args:
        T_grid: Array of T_decay values to sweep.
        theta: Fixed ancilla rotation angle.
        lam: Fixed bath correlation rate.
        base_config: Base configuration.  Uses defaults if None.

    Returns:
        NonMarkovianSweepData with sweep_type="time".
    """
    if base_config is None:
        base_config = PseudomodeConfig(N=DEFAULT_N, K=DEFAULT_K)

    n = len(T_grid)
    ratio_with = np.zeros(n, dtype=np.float64)
    ratio_without = np.zeros(n, dtype=np.float64)
    qfi_with = np.zeros(n, dtype=np.float64)
    qfi_without = np.zeros(n, dtype=np.float64)
    qfi_initial = np.zeros(n, dtype=np.float64)
    pm_occupancy = np.zeros(n, dtype=np.float64)

    g_sa = theta / base_config.tau if theta > 0 else 0.0

    for i, T in enumerate(T_grid):
        config = PseudomodeConfig(
            N=base_config.N,
            K=base_config.K,
            alpha=base_config.alpha,
            g_sa=g_sa,
            tau=base_config.tau,
            g_sp=base_config.g_sp,
            omega_0=base_config.omega_0,
            lam=lam,
            T_decay=T,
            dt=base_config.dt,
        )
        result = run_metrology_protocol(config)

        ratio_with[i] = result["ratio_with"]
        ratio_without[i] = result["ratio_without"]
        qfi_with[i] = result["qfi_with"]
        qfi_without[i] = result["qfi_without"]
        qfi_initial[i] = result["qfi_initial"]
        pm_occupancy[i] = result["pm_occupancy"]

        if (i + 1) % 25 == 0 or i == 0 or i == n - 1:
            print(
                f"  T={T:.4f}: R_with={ratio_with[i]:.6f}, "
                f"R_without={ratio_without[i]:.6f}, "
                f"pm_occ={pm_occupancy[i]:.4f}",
                flush=True,
            )

    return NonMarkovianSweepData(
        sweep_type="time",
        sweep_values=np.asarray(T_grid, dtype=np.float64),
        ratio_with=ratio_with,
        ratio_without=ratio_without,
        qfi_with=qfi_with,
        qfi_without=qfi_without,
        qfi_initial=qfi_initial,
        pm_occupancy=pm_occupancy,
        N=base_config.N,
        K=base_config.K,
        alpha=base_config.alpha,
        g_sp=base_config.g_sp,
        omega_0=base_config.omega_0,
        tau=base_config.tau,
        dt=base_config.dt,
        lam=lam,
        T_decay=float(T_grid[-1]),
        theta=theta,
    )


# =============================================================================
# Helpers
# =============================================================================


def find_optimal_theta(sweep_data: NonMarkovianSweepData) -> float:
    """Find theta* that maximises ratio_with.

    Args:
        sweep_data: Ancilla sweep data.

    Returns:
        Optimal theta value.
    """
    idx = int(np.argmax(sweep_data.ratio_with))
    return float(sweep_data.sweep_values[idx])


# =============================================================================
# Plot Functions
# =============================================================================


def plot_ancilla_sweep(
    data: NonMarkovianSweepData,
    save_path: str | Path | None = None,
) -> Path:
    """Plot R(T) vs theta with and without ancilla.

    Two curves: with ancilla (solid), without ancilla (dashed).
    Highlights the optimal theta*.

    Args:
        data: Ancilla sweep data.
        save_path: Output SVG path.  Auto-generated if None.

    Returns:
        Path to saved SVG.
    """
    if save_path is None:
        save_path = _fig_path("ancilla-sweep")
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    theta = data.sweep_values
    r_with = data.ratio_with
    r_without = data.ratio_without

    theta_opt = float(theta[np.argmax(r_with)])

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        theta,
        r_with,
        "C0-",
        linewidth=1.5,
        label=r"With ancilla $\mathcal{R}_{\text{with}}$",
    )
    ax.plot(
        theta,
        r_without,
        "C1--",
        linewidth=1.5,
        label=r"Without ancilla $\mathcal{R}_{\text{without}}$",
    )
    ax.axhline(
        y=1.0,
        color="gray",
        linestyle=":",
        alpha=0.5,
        label=r"$\mathcal{R}=1$ (perfect)",
    )

    # Highlight optimal theta
    ax.axvline(
        x=theta_opt,
        color="C0",
        linestyle="--",
        alpha=0.4,
        linewidth=1.0,
    )
    ax.plot(
        theta_opt,
        np.max(r_with),
        "C0*",
        markersize=10,
        label=rf"$\theta^* = {theta_opt:.3f}$",
    )

    ax.set_xlabel(r"$\theta = g_{\mathrm{sa}} \tau$")
    ax.set_ylabel(r"QFI preservation ratio $\mathcal{R}(T)$")
    ax.set_title(
        r"Ancilla Protection vs $\theta$ at fixed $\lambda = "
        f"{data.lam}"
        r"$, $T = "
        f"{data.T_decay}"
        r"$"
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0.0)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_memory_sweep(
    data: NonMarkovianSweepData,
    save_path: str | Path | None = None,
) -> Path:
    """Plot Delta R vs lambda showing non-Markovian advantage.

    Delta R = R_with - R_without.  Larger values at small lambda
    indicate stronger ancilla benefit for more non-Markovian baths.

    Args:
        data: Memory sweep data.
        save_path: Output SVG path.  Auto-generated if None.

    Returns:
        Path to saved SVG.
    """
    if save_path is None:
        save_path = _fig_path("memory-sweep")
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    lam = data.sweep_values
    delta_R = data.ratio_with - data.ratio_without
    r_with = data.ratio_with
    r_without = data.ratio_without

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left panel: absolute preservation ratios
    ax1.semilogx(
        lam,
        r_with,
        "C0-",
        linewidth=1.5,
        label=r"With ancilla $\mathcal{R}_{\text{with}}$",
    )
    ax1.semilogx(
        lam,
        r_without,
        "C1--",
        linewidth=1.5,
        label=r"Without ancilla $\mathcal{R}_{\text{without}}$",
    )
    ax1.set_xlabel(r"$\lambda$ (bath correlation rate)")
    ax1.set_ylabel(r"QFI preservation ratio $\mathcal{R}$")
    ax1.set_title(r"QFI Preservation vs $\lambda$")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Right panel: Delta R
    ax2.semilogx(lam, delta_R, "C2-", linewidth=1.5)
    ax2.axhline(y=0.0, color="gray", linestyle=":", alpha=0.5)
    ax2.set_xlabel(r"$\lambda$ (bath correlation rate)")
    ax2.set_ylabel(
        r"$\Delta\mathcal{R} = \mathcal{R}_{\text{with}} - \mathcal{R}_{\text{without}}$"
    )
    ax2.set_title(r"Ancilla Improvement vs $\lambda$")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        rf"Memory sweep at $\theta = {data.theta:.3f}$, $T = {data.T_decay:.1f}$",
        fontsize=12,
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_time_sweep(
    data_by_theta: dict[float, NonMarkovianSweepData],
    lam: float = DEFAULT_LAM,
    save_path: str | Path | None = None,
) -> Path:
    """Plot R(t) trajectories for multiple theta values as an overlay.

    Shows how ancilla protection modifies the decay trajectory over time.
    The theta=0 curve is the no-ancilla baseline.

    Args:
        data_by_theta: Dict mapping theta value to time sweep data.
        lam: Fixed lambda value for the title.
        save_path: Output SVG path.  Auto-generated if None.

    Returns:
        Path to saved SVG.
    """
    if save_path is None:
        save_path = _fig_path("time-sweep")
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 6))

    cmap = plt.colormaps["viridis"]
    thetas = sorted(data_by_theta.keys())
    colors = cmap(np.linspace(0.15, 0.85, len(thetas)))

    for idx, theta_val in enumerate(thetas):
        data = data_by_theta[theta_val]
        T = data.sweep_values
        r_with = data.ratio_with
        label_str = (
            rf"$\theta = {theta_val:.3f}$"
            if theta_val > 0
            else r"$\theta = 0$ (no ancilla)"
        )
        ls = ":" if theta_val == 0 else "-"
        lw = 2.0 if theta_val > 0 else 1.5
        alpha_val = 0.5 if theta_val == 0 else 0.8
        ax.plot(
            T,
            r_with,
            color=colors[idx],
            linestyle=ls,
            linewidth=lw,
            alpha=alpha_val,
            label=label_str,
        )

    ax.set_xlabel(r"Evolution time $T$")
    ax.set_ylabel(r"QFI preservation ratio $\mathcal{R}(T)$")
    ax.set_title(rf"QFI Trajectories at $\lambda = {lam:.3f}$")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0.0)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


# =============================================================================
# Data Generation Pipeline
# =============================================================================


def _generate_ancilla_nonmarkovian_raw_data(force: bool = False) -> dict:
    """Generate all raw data for the Ancilla-Assisted-Non-Markovian report.

    Runs three parameter sweeps:
      1. Ancilla sweep over theta in [0, pi] at fixed lam, T
      2. Memory sweep over lam in [0.05, 10] at optimal theta*
      3. Time sweeps over T in [0, 10] at several theta values

    Args:
        force: If True, re-generate even if Parquet files exist.

    Returns:
        Dict with keys ``ancilla_data``, ``memory_data``, and
        ``time_data_by_theta``.
    """
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)

    base_config = PseudomodeConfig(
        N=DEFAULT_N,
        K=DEFAULT_K,
        alpha=DEFAULT_ALPHA,
        g_sa=0.0,  # will be set per-point in sweeps
        tau=DEFAULT_TAU,
        g_sp=DEFAULT_G_SP,
        omega_0=DEFAULT_OMEGA_0,
        lam=DEFAULT_LAM,
        T_decay=DEFAULT_T_DECAY,
        dt=DEFAULT_DT,
    )

    # ---- 1. Ancilla sweep ----
    anc_path = _parquet_path("ancilla-sweep")
    if anc_path.exists() and not force:
        print("Loading existing ancilla sweep data ...")
        ancilla_data = NonMarkovianSweepData.from_parquet(anc_path)
    else:
        print(f"Running ancilla sweep ({ANC_N_POINTS} points, theta in [0, pi]) ...")
        theta_grid = np.linspace(ANC_THETA_MIN, ANC_THETA_MAX, ANC_N_POINTS)
        ancilla_data = ancilla_sweep(theta_grid, base_config)
        ancilla_data.save_parquet(anc_path)
        print(f"  Saved to {anc_path}")

    # Find optimal theta
    theta_opt = find_optimal_theta(ancilla_data)
    print(f"  Optimal theta* = {theta_opt:.4f}")

    # ---- 2. Memory sweep ----
    mem_path = _parquet_path("memory-sweep")
    if mem_path.exists() and not force:
        print("Loading existing memory sweep data ...")
        memory_data = NonMarkovianSweepData.from_parquet(mem_path)
    else:
        print(f"Running memory sweep ({MEM_N_POINTS} points, lam in [0.05, 10]) ...")
        lam_grid = np.logspace(
            np.log10(MEM_LAM_MIN), np.log10(MEM_LAM_MAX), MEM_N_POINTS
        )
        memory_data = memory_sweep(lam_grid, theta_opt, base_config)
        memory_data.save_parquet(mem_path)
        print(f"  Saved to {mem_path}")

    # ---- 3. Time sweeps (one per theta value) ----
    T_grid = np.linspace(TIME_T_MIN, TIME_T_MAX, TIME_N_POINTS)
    time_data_by_theta: dict[float, NonMarkovianSweepData] = {}

    for theta_val in TIME_THETA_VALUES:
        label = f"time-sweep-theta-{theta_val:.4f}".replace(".", "p")
        t_path = _parquet_path(label)
        if t_path.exists() and not force:
            print(f"Loading existing time sweep at theta={theta_val:.4f} ...")
            time_data_by_theta[theta_val] = NonMarkovianSweepData.from_parquet(t_path)
        else:
            print(
                f"Running time sweep at theta={theta_val:.4f} ({TIME_N_POINTS} points) ..."
            )
            t_data = time_sweep(
                T_grid, theta_val, lam=DEFAULT_LAM, base_config=base_config
            )
            t_data.save_parquet(t_path)
            time_data_by_theta[theta_val] = t_data
            print(f"  Saved to {t_path}")

    return {
        "ancilla_data": ancilla_data,
        "memory_data": memory_data,
        "time_data_by_theta": time_data_by_theta,
    }


def _generate_single_figure(
    parquet_name: str,
    fig_name: str,
    plot_fn: Callable[..., Any],
    force: bool = False,
    **plot_kwargs: Any,
) -> bool:
    """Check cache — load — plot — save for a single-parquet figure.

    Args:
        parquet_name: Stem name of the Parquet file (without date prefix).
        fig_name: Stem name of the SVG file.
        plot_fn: Plot function accepting (data, save_path=...).
        force: If True, overwrite existing SVG.
        **plot_kwargs: Additional keyword arguments passed to plot_fn.

    Returns:
        True if the figure was generated, False if skipped.
    """
    parquet_path = _parquet_path(parquet_name)
    fig_path = _fig_path(fig_name)

    if not parquet_path.exists():
        print(f"  [SKIP] {parquet_name} parquet not found, run raw data first")
        return False

    if fig_path.exists() and not force:
        print(f"    Figure exists ({fig_path}), skipping (use --force to overwrite)")
        return False

    print(f"  Plotting {fig_name} ...")
    data = NonMarkovianSweepData.from_parquet(parquet_path)
    plot_fn(data, save_path=fig_path, **plot_kwargs)
    print(f"    Saved to {fig_path}")
    return True


def _generate_time_sweep_figures(force: bool = False) -> None:
    """Generate the multi-theta time sweep overlay figure."""
    time_data_by_theta: dict[float, NonMarkovianSweepData] = {}
    any_missing = False
    for theta_val in TIME_THETA_VALUES:
        label = f"time-sweep-theta-{theta_val:.4f}".replace(".", "p")
        t_path = _parquet_path(label)
        if t_path.exists():
            time_data_by_theta[theta_val] = NonMarkovianSweepData.from_parquet(t_path)
        else:
            any_missing = True

    if not time_data_by_theta:
        return

    fig_path = _fig_path("time-sweep")
    if fig_path.exists() and not force:
        print(f"    Figure exists ({fig_path}), skipping (use --force to overwrite)")
        return

    print("  Plotting time sweep overlay ...")
    plot_time_sweep(time_data_by_theta, lam=DEFAULT_LAM, save_path=fig_path)
    print(f"    Saved to {fig_path}")

    if any_missing:
        print(
            "  [SKIP] Some time-sweep parquets missing, run raw data generation first"
        )


def _generate_ancilla_nonmarkovian_figures(force: bool = False) -> None:
    """Generate all figures for the Ancilla-Assisted-Non-Markovian report.

    Reads Parquet files from raw_data/, generates SVGs in figures/.
    """
    _generate_single_figure("ancilla-sweep", "ancilla-sweep", plot_ancilla_sweep, force)
    _generate_single_figure("memory-sweep", "memory-sweep", plot_memory_sweep, force)
    _generate_time_sweep_figures(force)


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate report figures and data for 2026-05-09 report",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run all simulations (overwrite existing Parquet/SVG)",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        choices=["ancilla-nonmarkovian"],
        help="Generate only one experiment's data and figures",
    )
    args = parser.parse_args()

    (_REPORT_DIR / "raw_data").mkdir(parents=True, exist_ok=True)
    (_REPORT_DIR / "figures").mkdir(parents=True, exist_ok=True)

    experiments = {
        "ancilla-nonmarkovian": (
            _generate_ancilla_nonmarkovian_raw_data,
            _generate_ancilla_nonmarkovian_figures,
        ),
    }

    if args.experiment:
        raw_fn, fig_fn = experiments[args.experiment]
        print(f"\n=== {args.experiment} ===")
        print("  Raw data ...")
        raw_fn(force=args.force)
        print("  Figures ...")
        fig_fn(force=args.force)
    else:
        for name, (raw_fn, fig_fn) in experiments.items():
            print(f"\n=== {name} ===")
            print("  Raw data ...")
            raw_fn(force=args.force)
            print("  Figures ...")
            fig_fn(force=args.force)

    print("\nDone.")


if __name__ == "__main__":
    main()
