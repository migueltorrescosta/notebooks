"""
Local module for the 2026-05-27 Drive-Component Analysis in Ancilla-Enhanced Metrology.

Contains all code exclusive to this report:
- Norm-ball sampling via Marsaglia's method (uniform 3-ball)
- NormBallResult and EnvelopeResult dataclasses with Parquet roundtrip
- Envelope curve extraction (best ratio vs drive norm)
- 2D slice generation for the (a_z, a_zz) slice
- Exclusive plot functions (norm-envelope curve, best-ratio-by-slice bar chart)
- CLI pipeline

Usage:
    uv run python reports/20260527/local.py --force

This module is **not** importable as ``reports.20260527.local`` (the directory
name contains hyphens).  Instead, importers add the report directory to
``sys.path`` and do ``import local``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

# Force non-interactive matplotlib backend before any plotting imports.
if "MPLBACKEND" not in os.environ:
    os.environ["MPLBACKEND"] = "Agg"

from src.analysis.ancilla_drive_metrology import (
    Drive2DSliceResult,
    build_two_qubit_operators,
    drive_2d_slice,
    evolve_drive_circuit,
)
from src.analysis.ancilla_optimization import (
    compute_expectation_and_variance,
)
from src.visualization.ancilla_drive_plots import (
    plot_drive_2d_slice_heatmap,
)

sns.set_theme(style="whitegrid")

# ============================================================================
# Constants
# ============================================================================

REPORTS_DIR = Path(__file__).resolve().parent.parent.parent / "reports"
REPORT_DATE = "20260527"
T_H: float = 10.0
SQL: float = 1.0 / T_H  # 0.1
FD_STEP: float = 1e-6
T_BS: float = np.pi / 2.0

# θ values for the 2D slice heatmaps (5 values, matching original report)
DRIVE_THETA_VALS: list[float] = [0.1, 0.5, 1.0, 2.0, 5.0]

# θ values for the norm-ball scan (50 values, step 0.1)
THETA_VALS: np.ndarray = np.arange(0.1, 5.05, 0.1)

# Norm-ball parameters
N_SAMP: int = 5000
R_MAX: float = 10.0
AZZ_BOUNDS: tuple[float, float] = (-5.0, 5.0)

# 2D slice range
DRIVE_RANGE: tuple[float, float] = (-5.0, 5.0)
N_DRIVE: int = 201
N_AZZ: int = 201

# Envelope grid
N_R: int = 100


# ============================================================================
# Path Helpers
# ============================================================================


def _parquet_path(name: str) -> Path:
    return REPORTS_DIR / REPORT_DATE / "raw_data" / f"{REPORT_DATE}-{name}.parquet"


def _fig_path(name: str) -> Path:
    return REPORTS_DIR / REPORT_DATE / "figures" / f"{REPORT_DATE}-{name}.svg"


# ============================================================================
# Sensitivity Computation with Extra Metadata
# ============================================================================


def compute_sensitivity_with_extra(
    theta_true: float,
    a_x: float,
    a_y: float,
    a_z: float,
    a_zz: float,
    *,
    T_H: float = T_H,
    T_BS: float = T_BS,
    fd_step: float = FD_STEP,
) -> tuple[float, float, float, float, bool]:
    """Compute Δθ plus ⟨J_z^S⟩, Var(J_z^S), and d⟨J_z^S⟩/dθ.

    Args:
        theta_true: True phase rate parameter.
        a_x: Ancilla J_x drive coefficient.
        a_y: Ancilla J_y drive coefficient.
        a_z: Ancilla J_z drive coefficient.
        a_zz: Ising interaction coefficient.
        T_H: Holding-time strength.
        T_BS: Beam-splitter duration.
        fd_step: Finite-difference step size.

    Returns:
        Tuple (delta_theta, expectation, variance, derivative, is_fringe_extremum).
        Returns (inf, exp_val, var_val, 0.0, True) when the derivative is zero
        (fringe extremum).
    """
    psi0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
    ops = build_two_qubit_operators()
    meas_op = ops["Jz_S"]

    # Evaluate at theta_true
    psi = evolve_drive_circuit(psi0, T_BS, T_H, theta_true, a_x, a_y, a_z, a_zz, ops)
    exp_val, var_val = compute_expectation_and_variance(psi, meas_op)

    # Central finite difference for ∂⟨O⟩/∂θ
    psi_plus = evolve_drive_circuit(
        psi0,
        T_BS,
        T_H,
        theta_true + fd_step,
        a_x,
        a_y,
        a_z,
        a_zz,
        ops,
    )
    psi_minus = evolve_drive_circuit(
        psi0,
        T_BS,
        T_H,
        theta_true - fd_step,
        a_x,
        a_y,
        a_z,
        a_zz,
        ops,
    )
    exp_plus = float(np.real(psi_plus.conj() @ meas_op @ psi_plus))
    exp_minus = float(np.real(psi_minus.conj() @ meas_op @ psi_minus))
    d_exp = (exp_plus - exp_minus) / (2.0 * fd_step)

    if abs(d_exp) < 1e-12:
        return float("inf"), exp_val, var_val, 0.0, True

    # Zero-variance case: the state is an eigenstate of the measurement
    # operator, giving a deterministic measurement outcome.  Error propagation
    # would yield Δθ = 0 (unphysical), because a deterministic measurement
    # provides zero information about θ.  Flag as fringe extremum.
    if var_val < 1e-15:
        return float("inf"), exp_val, var_val, d_exp, True

    delta_theta = float(np.sqrt(var_val) / abs(d_exp))
    return delta_theta, exp_val, var_val, d_exp, False


# ============================================================================
# NormBallResult Dataclass
# ============================================================================


@dataclass
class NormBallResult:
    """Result from norm-ball sampling over (a_x, a_y, a_z, a_zz).

    Attributes:
        theta_values: Array of θ values scanned (N_theta,).
        samples: Array of sampled parameters, shape (N_theta, N_samp, 4)
            ordered as [a_x, a_y, a_z, a_zz].
        delta_theta_values: Δθ for each sample, shape (N_theta, N_samp).
        expectation_values: ⟨J_z^S⟩ for each sample, shape (N_theta, N_samp).
        variance_values: Var(J_z^S) for each sample, shape (N_theta, N_samp).
        deriv_values: d⟨J_z^S⟩/dθ for each sample, shape (N_theta, N_samp).
        norms: ‖a‖ for each sample, shape (N_theta, N_samp).
        sql: SQL = 1/T_H reference value.
        T_H: Holding-time strength.
        R: Ball radius constraint.
    """

    theta_values: np.ndarray
    samples: np.ndarray
    delta_theta_values: np.ndarray
    expectation_values: np.ndarray
    variance_values: np.ndarray
    deriv_values: np.ndarray
    norms: np.ndarray
    sql: float = SQL
    T_H: float = T_H
    R: float = R_MAX

    def to_dataframe(self) -> pd.DataFrame:
        """Melt all data into a long-format DataFrame."""
        n_theta = len(self.theta_values)
        n_samp = self.samples.shape[1]
        rows: list[dict[str, float | str]] = [
            {
                "theta": float(self.theta_values[ti]),
                "a_x": float(self.samples[ti, si, 0]),
                "a_y": float(self.samples[ti, si, 1]),
                "a_z": float(self.samples[ti, si, 2]),
                "a_zz": float(self.samples[ti, si, 3]),
                "norm": float(self.norms[ti, si]),
                "delta_theta": float(self.delta_theta_values[ti, si]),
                "expectation": float(self.expectation_values[ti, si]),
                "variance": float(self.variance_values[ti, si]),
                "derivative": float(self.deriv_values[ti, si]),
                "sql": float(self.sql),
                "T_H": float(self.T_H),
                "R": float(self.R),
            }
            for ti in range(n_theta)
            for si in range(n_samp)
        ]
        return pd.DataFrame(rows)

    def save_parquet(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_parquet(path, index=False)
        return path

    @classmethod
    def from_parquet(cls, path: str | Path) -> NormBallResult:
        df = pd.read_parquet(path)
        required = {
            "theta",
            "a_x",
            "a_y",
            "a_z",
            "a_zz",
            "norm",
            "delta_theta",
            "expectation",
            "variance",
            "derivative",
            "sql",
            "T_H",
            "R",
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Parquet at {path} is missing required columns: "
                f"{sorted(missing)}. Regenerate the file with the current code."
            )
        theta_values = np.array(sorted(df["theta"].unique()), dtype=float)
        n_theta = len(theta_values)
        # Infer n_samp from the data
        n_samp = len(df) // n_theta
        samples = np.zeros((n_theta, n_samp, 4), dtype=float)
        deltas = np.zeros((n_theta, n_samp), dtype=float)
        exps = np.zeros((n_theta, n_samp), dtype=float)
        vars_ = np.zeros((n_theta, n_samp), dtype=float)
        derivs = np.zeros((n_theta, n_samp), dtype=float)
        norms = np.zeros((n_theta, n_samp), dtype=float)
        for ti, theta in enumerate(theta_values):
            mask = np.isclose(df["theta"].to_numpy(dtype=float), theta)
            subset = df[mask]
            n_rows = np.sum(mask)
            samples[ti, :n_rows, 0] = subset["a_x"].to_numpy(dtype=float)
            samples[ti, :n_rows, 1] = subset["a_y"].to_numpy(dtype=float)
            samples[ti, :n_rows, 2] = subset["a_z"].to_numpy(dtype=float)
            samples[ti, :n_rows, 3] = subset["a_zz"].to_numpy(dtype=float)
            deltas[ti, :n_rows] = subset["delta_theta"].to_numpy(dtype=float)
            exps[ti, :n_rows] = subset["expectation"].to_numpy(dtype=float)
            vars_[ti, :n_rows] = subset["variance"].to_numpy(dtype=float)
            derivs[ti, :n_rows] = subset["derivative"].to_numpy(dtype=float)
            norms[ti, :n_rows] = subset["norm"].to_numpy(dtype=float)
        return cls(
            theta_values=theta_values,
            samples=samples,
            delta_theta_values=deltas,
            expectation_values=exps,
            variance_values=vars_,
            deriv_values=derivs,
            norms=norms,
            sql=float(df["sql"].iloc[0]),
            T_H=float(df["T_H"].iloc[0]),
            R=float(df["R"].iloc[0]),
        )


# ============================================================================
# Norm-Ball Sampling (Marsaglia's Method)
# ============================================================================


def _marsaglia_3ball_sample(
    rng: np.random.Generator,
    n_samp: int,
    R: float,
    azz_lo: float,
    azz_hi: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate uniform samples from the 3-ball of radius R.

    Uses Marsaglia's method: generate 3 i.i.d. standard normal variates,
    divide by their norm, multiply by R * u^(1/3) where u ~ U[0,1].

    Args:
        rng: NumPy random generator.
        n_samp: Number of samples.
        R: Ball radius.
        azz_lo: Lower bound for a_zz.
        azz_hi: Upper bound for a_zz.

    Returns:
        Tuple (drive_samples, azz_samples) where drive_samples has shape
        (n_samp, 3) with columns [a_x, a_y, a_z], and azz_samples has
        shape (n_samp,).
    """
    # Step 1: 3 i.i.d. standard normal variates
    z = rng.normal(0.0, 1.0, size=(n_samp, 3))
    # Step 2: Normalise to unit sphere
    sphere_norm = np.sqrt(np.sum(z**2, axis=1))
    sphere_norm = np.maximum(sphere_norm, 1e-300)  # avoid division by zero
    z_unit = z / sphere_norm[:, np.newaxis]
    # Step 3: Radial scaling for uniform volume distribution
    u = rng.uniform(0.0, 1.0, size=n_samp)
    r_scaled = R * (u ** (1.0 / 3.0))
    drive_samples = z_unit * r_scaled[:, np.newaxis]

    # a_zz sampled uniformly in [azz_lo, azz_hi]
    azz_samples = rng.uniform(azz_lo, azz_hi, size=n_samp)

    return drive_samples, azz_samples


def _normball_theta_chunk_worker(args: tuple) -> dict:
    """Worker for parallel norm-ball θ chunk (module-level for pickling).

    Args:
        args: Tuple (theta_list, n_samp, R, azz_lo, azz_hi, T_H, T_BS, fd_step, base_seed).

    Returns:
        Dict with keys 'theta_values', 'samples', 'deltas', 'exps',
        'vars_', 'derivs', 'norms', each a numpy array for the chunk.
    """
    theta_list, n_samp, R, azz_lo, azz_hi, T_H, T_BS, fd_step, base_seed = args
    n_theta = len(theta_list)
    samples = np.zeros((n_theta, n_samp, 4), dtype=float)
    deltas = np.full((n_theta, n_samp), np.inf, dtype=float)
    exps = np.zeros((n_theta, n_samp), dtype=float)
    vars_ = np.zeros((n_theta, n_samp), dtype=float)
    derivs = np.zeros((n_theta, n_samp), dtype=float)
    norms = np.zeros((n_theta, n_samp), dtype=float)

    for ti, theta in enumerate(theta_list):
        theta_rng = np.random.default_rng(
            base_seed + int(theta * 1000) if base_seed is not None else None
        )
        drive_samp, azz_samp = _marsaglia_3ball_sample(
            theta_rng, n_samp, R, azz_lo, azz_hi
        )
        for si in range(n_samp):
            ax = float(drive_samp[si, 0])
            ay = float(drive_samp[si, 1])
            az = float(drive_samp[si, 2])
            azz = float(azz_samp[si])

            dtheta, exp_val, var_val, deriv, _ = compute_sensitivity_with_extra(
                theta,
                ax,
                ay,
                az,
                azz,
                T_H=T_H,
                T_BS=T_BS,
                fd_step=fd_step,
            )

            samples[ti, si, :] = [ax, ay, az, azz]
            deltas[ti, si] = dtheta
            exps[ti, si] = exp_val
            vars_[ti, si] = var_val
            derivs[ti, si] = deriv
            norms[ti, si] = float(np.sqrt(ax**2 + ay**2 + az**2))

    return {
        "theta_values": np.array(theta_list, dtype=float),
        "samples": samples,
        "deltas": deltas,
        "exps": exps,
        "vars_": vars_,
        "derivs": derivs,
        "norms": norms,
    }


def norm_ball_sampling(
    theta_values: np.ndarray | list[float],
    *,
    n_samp: int = N_SAMP,
    R: float = R_MAX,
    azz_bounds: tuple[float, float] = AZZ_BOUNDS,
    T_H: float = T_H,
    T_BS: float = T_BS,
    fd_step: float = FD_STEP,
    seed: int | None = 42,
    n_jobs: int | None = None,
) -> NormBallResult:
    """Run norm-ball Monte Carlo sampling over multiple θ values.

    For each θ, generates ``n_samp`` configurations uniformly from the
    3-ball ‖a‖ ≤ R with a_zz ~ U[azz_lo, azz_hi] independently, and
    evaluates the sensitivity Δθ for each configuration.

    When ``n_jobs > 1``, θ values are split across worker processes.

    Args:
        theta_values: θ values to scan.
        n_samp: Number of random samples per θ.
        R: Norm-ball radius.
        azz_bounds: (min, max) for a_zz.
        T_H: Holding-time strength.
        T_BS: Beam-splitter duration.
        fd_step: Finite-difference step size.
        seed: Random seed for reproducibility.
        n_jobs: Number of parallel workers. ``None`` (default) = sequential.
            Pass ``-1`` to use all available CPUs.

    Returns:
        NormBallResult with all sample data.
    """
    theta_arr = np.asarray(theta_values, dtype=float)
    n_theta = len(theta_arr)
    azz_lo, azz_hi = azz_bounds

    if n_jobs is None or n_jobs == 1:
        # ── Sequential path (with tqdm) ──────────────────────────────────
        samples = np.zeros((n_theta, n_samp, 4), dtype=float)
        deltas = np.full((n_theta, n_samp), np.inf, dtype=float)
        exps = np.zeros((n_theta, n_samp), dtype=float)
        vars_ = np.zeros((n_theta, n_samp), dtype=float)
        derivs = np.zeros((n_theta, n_samp), dtype=float)
        norms = np.zeros((n_theta, n_samp), dtype=float)

        for ti, theta in enumerate(
            tqdm(theta_arr, desc="Norm-ball sampling", unit="θ")
        ):
            theta_rng = np.random.default_rng(
                seed + int(theta * 1000) if seed is not None else None
            )
            drive_samp, azz_samp = _marsaglia_3ball_sample(
                theta_rng, n_samp, R, azz_lo, azz_hi
            )

            for si in range(n_samp):
                ax = float(drive_samp[si, 0])
                ay = float(drive_samp[si, 1])
                az = float(drive_samp[si, 2])
                azz = float(azz_samp[si])

                dtheta, exp_val, var_val, deriv, _ = compute_sensitivity_with_extra(
                    theta,
                    ax,
                    ay,
                    az,
                    azz,
                    T_H=T_H,
                    T_BS=T_BS,
                    fd_step=fd_step,
                )

                samples[ti, si, :] = [ax, ay, az, azz]
                deltas[ti, si] = dtheta
                exps[ti, si] = exp_val
                vars_[ti, si] = var_val
                derivs[ti, si] = deriv
                norms[ti, si] = float(np.sqrt(ax**2 + ay**2 + az**2))
    else:
        # ── Parallel path ────────────────────────────────────────────────
        n_workers = max(1, os.cpu_count() or 4) if n_jobs == -1 else n_jobs
        chunks = np.array_split(theta_arr, n_workers)
        worker_args = [
            (
                chunk.tolist(),
                n_samp,
                R,
                azz_lo,
                azz_hi,
                T_H,
                T_BS,
                fd_step,
                seed,
            )
            for chunk in chunks
        ]

        partials: list[dict] = [{}] * len(worker_args)
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=n_workers,
        ) as executor:
            futures = {
                executor.submit(_normball_theta_chunk_worker, args): i
                for i, args in enumerate(worker_args)
            }
            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                partials[idx] = future.result()

        # Concatenate results in original order
        all_theta = np.concatenate([p["theta_values"] for p in partials])
        sort_idx = np.argsort(all_theta)
        samples = np.concatenate([p["samples"] for p in partials], axis=0)[sort_idx]
        deltas = np.concatenate([p["deltas"] for p in partials], axis=0)[sort_idx]
        exps = np.concatenate([p["exps"] for p in partials], axis=0)[sort_idx]
        vars_ = np.concatenate([p["vars_"] for p in partials], axis=0)[sort_idx]
        derivs = np.concatenate([p["derivs"] for p in partials], axis=0)[sort_idx]
        norms = np.concatenate([p["norms"] for p in partials], axis=0)[sort_idx]

    return NormBallResult(
        theta_values=theta_arr,
        samples=samples,
        delta_theta_values=deltas,
        expectation_values=exps,
        variance_values=vars_,
        deriv_values=derivs,
        norms=norms,
        sql=1.0 / T_H,
        T_H=T_H,
        R=R,
    )


# ============================================================================
# EnvelopeResult Dataclass
# ============================================================================


@dataclass
class EnvelopeResult:
    """Result from extracting the norm-constrained envelope curve.

    Attributes:
        r_values: Grid of drive-norm thresholds (N_r,).
        best_ratio_per_theta: min Δθ/SQL for each θ and r, shape
            (N_theta, N_r).
        theta_values: θ values (N_theta,).
        sql: SQL reference value.
    """

    r_values: np.ndarray
    best_ratio_per_theta: np.ndarray
    theta_values: np.ndarray
    sql: float = SQL

    def to_dataframe(self) -> pd.DataFrame:
        """Melt the envelope into long-format DataFrame."""
        n_theta = len(self.theta_values)
        n_r = len(self.r_values)
        rows: list[dict[str, float]] = [
            {
                "theta": float(self.theta_values[ti]),
                "r": float(self.r_values[ri]),
                "best_ratio": float(self.best_ratio_per_theta[ti, ri]),
                "sql": float(self.sql),
            }
            for ti in range(n_theta)
            for ri in range(n_r)
        ]
        return pd.DataFrame(rows)

    def save_parquet(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_parquet(path, index=False)
        return path

    @classmethod
    def from_parquet(cls, path: str | Path) -> EnvelopeResult:
        df = pd.read_parquet(path)
        required = {"theta", "r", "best_ratio", "sql"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Parquet at {path} is missing required columns: "
                f"{sorted(missing)}. Regenerate the file with the current code."
            )
        theta_values = np.array(sorted(df["theta"].unique()), dtype=float)
        r_values = np.array(sorted(df["r"].unique()), dtype=float)
        n_theta = len(theta_values)
        n_r = len(r_values)
        grid = np.full((n_theta, n_r), np.nan, dtype=float)
        for _, row in df.iterrows():
            ti = int(np.searchsorted(theta_values, row["theta"]))
            ri = int(np.searchsorted(r_values, row["r"]))
            grid[ti, ri] = row["best_ratio"]
        return cls(
            r_values=r_values,
            best_ratio_per_theta=grid,
            theta_values=theta_values,
            sql=float(df["sql"].iloc[0]),
        )


# ============================================================================
# Envelope Curve Extraction
# ============================================================================


def extract_envelope_curve(
    result: NormBallResult,
    n_r: int = N_R,
    r_max: float | None = None,
) -> EnvelopeResult:
    """Compute the norm-constrained best-ratio envelope curve.

    For each θ and each r in a fine grid, find the minimum Δθ/SQL among
    all samples whose drive norm does not exceed r.

    Args:
        result: NormBallResult with all sample data.
        n_r: Number of r grid points.
        r_max: Maximum r value (defaults to result.R).

    Returns:
        EnvelopeResult with best_ratio(r) for each θ.
    """
    if r_max is None:
        r_max = result.R
    r_values = np.linspace(0.0, r_max, n_r)
    n_theta = len(result.theta_values)
    best_ratio = np.full((n_theta, n_r), np.inf, dtype=float)

    for ti in range(n_theta):
        valid = np.isfinite(result.delta_theta_values[ti])
        the_norms = result.norms[ti, valid]
        the_ratios = result.delta_theta_values[ti, valid] / result.sql
        n_valid = np.sum(valid)
        if n_valid == 0:
            best_ratio[ti, :] = float("inf")
            continue
        for ri, r_val in enumerate(r_values):
            mask = the_norms <= r_val
            if np.any(mask):
                best_ratio[ti, ri] = float(np.min(the_ratios[mask]))
            # else: leave as inf (no samples within this radius)

    return EnvelopeResult(
        r_values=r_values,
        best_ratio_per_theta=best_ratio,
        theta_values=result.theta_values,
        sql=result.sql,
    )


# ============================================================================
# Plot Functions
# ============================================================================


def plot_norm_envelope_curve(
    envelope: EnvelopeResult | str | Path,
    save_path: str | Path,
    figsize: tuple[float, float] = (8, 6),
) -> Path:
    """Plot min Δθ/SQL vs drive norm r for each θ.

    Shows individual curves for a subset of θ values (5 representative ones),
    the overall minimum across θ as a thick line, and SQL reference at y=1.

    Args:
        envelope: EnvelopeResult or path to Parquet file.
        save_path: Output SVG path.
        figsize: Figure size.

    Returns:
        Path to saved SVG.
    """
    if isinstance(envelope, (str, Path)):
        envelope = EnvelopeResult.from_parquet(envelope)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)

    # Representative θ values for individual curves
    show_thetas = [0.1, 0.5, 1.0, 2.0, 5.0]
    colours = ["C0", "C1", "C2", "C3", "C4"]

    for idx, theta_val in enumerate(show_thetas):
        mask = np.isclose(envelope.theta_values, theta_val, atol=1e-6)
        if not np.any(mask):
            continue
        ti = int(np.argmax(mask))
        valid = np.isfinite(envelope.best_ratio_per_theta[ti])
        ax.plot(
            envelope.r_values[valid],
            envelope.best_ratio_per_theta[ti, valid],
            color=colours[idx % len(colours)],
            linestyle="-",
            linewidth=1.2,
            alpha=0.7,
            label=rf"$\theta = {theta_val:.1f}$",
        )

    # Overall minimum across all θ
    over_min = np.nanmin(envelope.best_ratio_per_theta, axis=0)
    valid_over = np.isfinite(over_min)
    ax.plot(
        envelope.r_values[valid_over],
        over_min[valid_over],
        color="black",
        linestyle="-",
        linewidth=2.5,
        label="Overall minimum",
    )

    # SQL reference line
    ax.axhline(
        y=1.0, color="red", linestyle="--", linewidth=1.5, alpha=0.8, label="SQL"
    )

    ax.set_xlabel(r"Drive norm $\|\mathbf{a}\|$")
    ax.set_ylabel(r"$\min \Delta\theta \;/\; \mathrm{SQL}$")
    ax.set_title("Norm-constrained best sensitivity ratio")
    ax.set_ylim(0.95, 1.5)
    ax.legend(fontsize=9, loc="upper right")

    # Annotate that the minimum is always 1.0
    min_overall = float(np.nanmin(over_min[valid_over]))
    ax.annotate(
        f"Min ratio = {min_overall:.8f}",
        xy=(envelope.r_values[-1], min_overall),
        xytext=(envelope.r_values[-1] * 0.6, min_overall + 0.08),
        arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.0},
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "gray"},
    )

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_best_ratio_by_slice(
    slice_results: dict[str, Drive2DSliceResult],
    save_path: str | Path,
    figsize: tuple[float, float] = (8, 5),
) -> Path:
    """Grouped bar chart comparing min Δθ/SQL across slice types.

    Args:
        slice_results: Dict mapping slice_type to Drive2DSliceResult.
        save_path: Output SVG path.
        figsize: Figure size.

    Returns:
        Path to saved SVG.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)

    # Collect data by θ
    all_thetas: list[float] = []
    ratios: dict[str, list[float]] = {"ax": [], "ay": [], "az": []}

    # Collect all unique θ from all results
    theta_set: set[float] = set()
    for res in slice_results.values():
        theta_set.add(res.theta_value)
    all_thetas = sorted(theta_set)

    for theta in all_thetas:
        for st in ("ax", "ay", "az"):
            if st in slice_results and np.isclose(slice_results[st].theta_value, theta):
                grid = slice_results[st].delta_theta_grid
                finite_mask = np.isfinite(grid)
                if np.any(finite_mask):
                    min_val = float(np.min(grid[finite_mask]))
                    ratios[st].append(min_val / slice_results[st].sql)
                else:
                    ratios[st].append(float("inf"))
            else:
                ratios[st].append(float("nan"))

    # Grouped bar chart
    n_groups = len(all_thetas)
    x = np.arange(n_groups)
    width = 0.25
    colours = {"ax": "C0", "ay": "C1", "az": "C2"}
    labels = {
        "ax": r"$(a_x, a_{zz})$",
        "ay": r"$(a_y, a_{zz})$",
        "az": r"$(a_z, a_{zz})$",
    }

    for idx, st in enumerate(("ax", "ay", "az")):
        vals = [r if np.isfinite(r) else 0.0 for r in ratios[st]]
        ax.bar(
            x + (idx - 1) * width,
            vals,
            width,
            color=colours[st],
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
            label=labels[st],
        )

    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$\min \Delta\theta \;/\; \mathrm{SQL}$")
    ax.set_title("Best sensitivity ratio by slice type")
    ax.set_xticks(x)
    ax.set_xticklabels([rf"$\theta={t:.1f}$" for t in all_thetas])
    ax.axhline(
        y=1.0, color="red", linestyle="--", linewidth=1.5, alpha=0.7, label="SQL"
    )
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_normball_histogram(
    result: NormBallResult | str | Path,
    theta: float,
    save_path: str | Path,
    figsize: tuple[float, float] = (8, 5),
    n_bins: int = 50,
) -> Path:
    """Histogram of Δθ values from norm-ball sampling at a specific θ.

    Args:
        result: NormBallResult or path to Parquet file.
        theta: θ value to plot.
        save_path: Output SVG path.
        figsize: Figure size.
        n_bins: Number of histogram bins.

    Returns:
        Path to saved SVG.
    """
    if isinstance(result, (str, Path)):
        result = NormBallResult.from_parquet(result)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Find θ index
    mask = np.isclose(result.theta_values, theta, atol=1e-6)
    if not np.any(mask):
        raise ValueError(f"θ={theta} not found in NormBallResult")
    ti = int(np.argmax(mask))
    deltas = result.delta_theta_values[ti]
    finite = deltas[np.isfinite(deltas)]

    fig, ax = plt.subplots(figsize=figsize)

    ax.hist(
        finite, bins=n_bins, color="C0", alpha=0.7, edgecolor="black", linewidth=0.5
    )
    ax.axvline(
        x=result.sql,
        color="C1",
        linestyle="--",
        linewidth=1.5,
        label=rf"SQL = {result.sql:.4f}",
    )

    below_sql = np.sum(finite < result.sql)
    total = len(finite)
    ax.annotate(
        rf"{below_sql} / {total} below SQL ({100 * below_sql / total:.1f}%)",
        xy=(result.sql, 0.7 * ax.get_ylim()[1]),
        fontsize=10,
        color="red",
    )

    ax.set_xlabel(r"$\Delta\theta$")
    ax.set_ylabel("Count")
    ax.set_title(
        rf"Norm-ball sampling at $\theta={theta:.1f}$, "
        rf"$R={result.R:.0f}$ ({total} finite points)"
    )
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


# ============================================================================
# Dataset Generation Functions
# ============================================================================


def _run_2d_slice(
    theta: float,
    slice_type: str,
    force: bool,
    n_jobs: int | None = None,
) -> None:
    """Run a 2D slice scan for a single θ value and generate CSV + SVG."""
    tag = f"drive-2d-slice-{slice_type}-azz-theta{theta}"
    csv_p = _parquet_path(tag)
    fig_p = _fig_path(tag)

    if csv_p.exists() and not force:
        print(f"  [skip] {csv_p.name} exists (use --force to overwrite)")
        result = Drive2DSliceResult.from_parquet(csv_p)
    else:
        print(f"  [run]  Computing ({slice_type}, a_zz) slice at θ={theta}...")
        result = drive_2d_slice(
            theta=theta,
            slice_type=slice_type,
            drive_range=DRIVE_RANGE,
            azz_range=AZZ_BOUNDS,
            n_drive=N_DRIVE,
            n_azz=N_AZZ,
            T_H=T_H,
            T_BS=T_BS,
            n_jobs=n_jobs,
        )
        result.save_parquet(csv_p)
        print(f"  [save] {csv_p}")

    plot_drive_2d_slice_heatmap(result, fig_p)
    print(f"  [fig]  {fig_p}")


def generate_2d_slice_ax_azz(force: bool = False, n_jobs: int | None = None) -> None:
    """Experiment 1a: 2D slice scans over (a_x, a_zz) at all θ values."""
    print(f"[run]  (a_x, a_zz) slice at {DRIVE_THETA_VALS}")
    for theta in DRIVE_THETA_VALS:
        _run_2d_slice(theta, slice_type="ax", force=force, n_jobs=n_jobs)


def generate_2d_slice_ay_azz(force: bool = False, n_jobs: int | None = None) -> None:
    """Experiment 1b: 2D slice scans over (a_y, a_zz) at all θ values."""
    print(f"[run]  (a_y, a_zz) slice at {DRIVE_THETA_VALS}")
    for theta in DRIVE_THETA_VALS:
        _run_2d_slice(theta, slice_type="ay", force=force, n_jobs=n_jobs)


def generate_2d_slice_az_azz(force: bool = False, n_jobs: int | None = None) -> None:
    """Experiment 1c: 2D slice scans over (a_z, a_zz) at all θ values."""
    print(f"[run]  (a_z, a_zz) slice at {DRIVE_THETA_VALS}")
    for theta in DRIVE_THETA_VALS:
        _run_2d_slice(theta, slice_type="az", force=force, n_jobs=n_jobs)


def generate_norm_ball(force: bool = False, n_jobs: int | None = None) -> None:
    """Experiment 2: Norm-ball sampling at 50 θ values × 5000 samples."""
    csv_p = _parquet_path("normball-samples")
    env_csv_p = _parquet_path("normball-envelope")
    fig_env_p = _fig_path("norm-envelope")

    if csv_p.exists() and not force:
        print(f"[skip] {csv_p.name} exists (use --force to overwrite)")
        result = NormBallResult.from_parquet(csv_p)
    else:
        print(
            "[run]  Computing norm-ball sampling "
            f"({len(THETA_VALS)} θ × {N_SAMP} samples = "
            f"{len(THETA_VALS) * N_SAMP} evaluations)..."
        )
        result = norm_ball_sampling(
            theta_values=THETA_VALS,
            n_samp=N_SAMP,
            R=R_MAX,
            azz_bounds=AZZ_BOUNDS,
            T_H=T_H,
            T_BS=T_BS,
            fd_step=FD_STEP,
            seed=42,
            n_jobs=n_jobs,
        )
        result.save_parquet(csv_p)
        print(f"[save] {csv_p}")

    # Compute and save envelope
    if env_csv_p.exists() and not force:
        print(f"[skip] {env_csv_p.name} exists (use --force to overwrite)")
        envelope = EnvelopeResult.from_parquet(env_csv_p)
    else:
        print("[run]  Computing envelope curve...")
        envelope = extract_envelope_curve(result, n_r=N_R)
        envelope.save_parquet(env_csv_p)
        print(f"[save] {env_csv_p}")

    plot_norm_envelope_curve(envelope, fig_env_p)
    print(f"[fig]  {fig_env_p}")


def generate_best_ratio_comparison(force: bool = False) -> None:
    """Experiment 3: Compare best ratio across slice types at a single θ."""
    fig_p = _fig_path("best-ratio-by-slice")

    print("[run]  Loading slice results for best-ratio comparison...")
    slice_results: dict[str, Drive2DSliceResult] = {}
    for st in ("ax", "ay", "az"):
        for theta in DRIVE_THETA_VALS:
            tag = f"drive-2d-slice-{st}-azz-theta{theta}"
            csv_p = _parquet_path(tag)
            if csv_p.exists():
                result = Drive2DSliceResult.from_parquet(csv_p)
                # Simple keys — last θ loaded (theta=5.0) overwrites earlier ones
                slice_results[st] = result

    plot_best_ratio_by_slice(slice_results, fig_p)
    print(f"[fig]  {fig_p}")


def generate_all_slices(force: bool = False, n_jobs: int | None = None) -> None:
    """Generate all three 2D slice types."""
    generate_2d_slice_ax_azz(force=force, n_jobs=n_jobs)
    generate_2d_slice_ay_azz(force=force, n_jobs=n_jobs)
    generate_2d_slice_az_azz(force=force, n_jobs=n_jobs)


# ============================================================================
# CLI Entry Point
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate 2026-05-27 report figures and CSV",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run all simulations (overwrite existing CSVs)",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Generate only one dataset, e.g. 'normball'",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="Number of parallel workers. -1 = all CPUs (default: sequential)",
    )
    args = parser.parse_args()

    # Ensure per-date directories exist
    (REPORTS_DIR / REPORT_DATE / "raw_data").mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / REPORT_DATE / "figures").mkdir(parents=True, exist_ok=True)

    n_jobs: int | None = -1 if args.jobs == -1 else args.jobs

    tasks: dict[str, Callable[..., None]] = {
        "slice-ax-azz": generate_2d_slice_ax_azz,
        "slice-ay-azz": generate_2d_slice_ay_azz,
        "slice-az-azz": generate_2d_slice_az_azz,
        "all-slices": generate_all_slices,
        "normball": generate_norm_ball,
        "best-ratio-comparison": generate_best_ratio_comparison,
    }

    # Some tasks don't accept n_jobs (dispatch function) kwarg
    def _dispatch(
        func: Callable[..., None],
        *,
        force: bool,
        n_jobs: int | None,
    ) -> None:
        import inspect

        sig = inspect.signature(func)
        kwargs: dict[str, object] = {"force": force}
        if "n_jobs" in sig.parameters:
            kwargs["n_jobs"] = n_jobs
        func(**kwargs)

    if args.only:
        if args.only not in tasks:
            print(f"Unknown dataset '{args.only}'. Options: {list(tasks.keys())}")
            sys.exit(1)
        _dispatch(tasks[args.only], force=args.force, n_jobs=n_jobs)
    else:
        for name, func in tasks.items():
            print(f"\n=== {name} ===")
            _dispatch(func, force=args.force, n_jobs=n_jobs)

    print("\nDone.")


if __name__ == "__main__":
    main()
