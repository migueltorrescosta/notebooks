"""
Local module for the 2026-06-10 Free-Ancilla with ω-Modulated Drive report.

Combines the free-ancilla initial state (20260528) with the ω-modulated drive
mechanism (20260519) to test whether freeing the ancilla improves sensitivity
beyond the 4.91× SQL beating already achieved with a fixed |1,0⟩ ancilla.

Three-stage optimisation:
  Stage 1: 2D slice (θ_A, a_zz) at fixed optimal drive (20260519 params).
  Stage 2: 6D random search over (θ_A, φ_A, a_x, a_y, a_z, a_zz).
  Stage 3: Nelder-Mead refinement from best Stage 2 points.

Usage:
    uv run python reports/20260610/free_ancilla_omega_modulated.py --force
    uv run python reports/20260610/free_ancilla_omega_modulated.py --only stage-1-slice
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing as _mp
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import minimize

# Force non-interactive matplotlib backend before any plotting imports.
if "MPLBACKEND" not in os.environ:
    os.environ["MPLBACKEND"] = "Agg"

from src.analysis.ancilla_drive_metrology import (
    evolve_phase_modulated_circuit,
)
from src.analysis.ancilla_optimization import (
    build_two_qubit_operators,
    compute_expectation_and_variance,
    free_ancilla_initial_state,
)
from src.analysis.optimisation_pipeline import (
    TwoPhaseConfig,
    run_omega_scan,
)
from src.analysis.slice_scan import parallel_grid_scan, sequential_grid_scan
from src.utils.monte_carlo import marsaglia_ball_sample
from src.utils.paths import report_path_fn
from src.utils.serialization import ParquetSerializable

sns.set_theme(style="whitegrid")

# ============================================================================
# Constants
# ============================================================================

REPORTS_DIR = Path(__file__).resolve().parent.parent.parent / "reports"
REPORT_DATE = "20260610"
t_hold: float = 10.0
SQL: float = 1.0 / t_hold  # 0.1
FD_STEP: float = 1e-6
T_BS: float = np.pi / 2.0

# ω values for the scan (matching 20260519)
OMEGA_VALS: list[float] = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]

# Random-search sample counts
N_SAMP_STAGE2: int = 3000  # 6D random search
N_NM_REFINE: int = 40  # Nelder-Mead refinements per ω

# Norm-ball parameters
R_MAX: float = 10.0
AZZ_BOUNDS: tuple[float, float] = (-5.0, 5.0)

# 2D slice parameters (Stage 1)
STAGE1_N: int = 101
THETA_A_RANGE: tuple[float, float] = (0.0, np.pi)

# Default 20260519 optimal (a_x, a_y, a_z) drive parameters at each ω
# These were found by Nelder–Mead refinement at each ω in the 20260519 report.
# The best overall Δω = 0.02036 was achieved at ω=0.2 with (a_x, a_y, a_z) = (5.0, -5.0, 4.0).
DEFAULT_FIXED_DRIVE: dict[float, tuple[float, float, float]] = {
    0.1: (5.0, 0.0, 5.0),  # a_zz≈1.5, Δω≈0.0213
    0.2: (5.0, -5.0, 4.0),  # a_zz≈4.0, Δω≈0.02036 (best overall)
    0.5: (4.6, 3.1, 1.1),  # a_zz≈4.1, Δω≈0.0273
    1.0: (5.0, 0.85, 0.0),  # a_zz≈5.0, Δω≈0.0332
    2.0: (4.87, 0.0, 0.0),  # a_zz≈5.0, Δω≈0.0460
    5.0: (5.0, 5.0, 0.0),  # a_zz≈5.0, Δω≈0.0668
}


# ============================================================================
# Path Helpers
# ============================================================================


_parquet_path, _fig_path = report_path_fn(REPORTS_DIR, REPORT_DATE)


# ============================================================================
# Free-Ancilla ω-Modulated Sensitivity
# ============================================================================


def compute_free_ancilla_modulated_sensitivity(
    omega_true: float,
    theta_A: float,
    phi_A: float,
    a_x: float,
    a_y: float,
    a_z: float,
    a_zz: float,
    *,
    t_hold: float = t_hold,
    T_BS: float = T_BS,
    fd_step: float = FD_STEP,
) -> tuple[float, float, float, float, bool]:
    r"""Compute the error-propagation sensitivity :math:`\Delta\omega`.

    Uses the free-ancilla initial state and the :math:`\omega`-modulated drive
    Hamiltonian.

    Delegates to the shared :func:`compute_free_ancilla_sensitivity` in
    ``src.analysis.ancilla_drive_metrology`` with
    ``evolve_fn=evolve_phase_modulated_circuit``.

    Returns:
        Tuple ``(delta_omega, expectation, variance, derivative, is_fringe)``.
    """
    import src.analysis.ancilla_drive_metrology as _adm

    return _adm.compute_free_ancilla_sensitivity(
        evolve_phase_modulated_circuit,
        omega_true,
        theta_A,
        phi_A,
        a_x,
        a_y,
        a_z,
        a_zz,
        t_hold=t_hold,
        T_BS=T_BS,
        fd_step=fd_step,
    )


# ============================================================================
# Dataclasses
# ============================================================================


@dataclass
class FreeAncillaModulatedSearchResult(ParquetSerializable):
    r"""Result from a batch of 6D random-search evaluations.

    Attributes:
        samples: Parameter matrix, shape ``(N, 6)`` with columns
            ``[theta_A, phi_A, a_x, a_y, a_z, a_zz]``.
        delta_omega_values: Sensitivity for each sample, shape ``(N,)``.
        expectation_values: :math:`\langle J_z^S\rangle` for each sample.
        variance_values: :math:`\mathrm{Var}(J_z^S)` for each sample.
        deriv_values: :math:`\partial\langle J_z^S\rangle/\partial\omega`.
        is_fringe: Boolean flag per sample, shape ``(N,)``.
        best_params: The 6-tuple giving the minimum :math:`\Delta\omega`.
        best_delta_omega: The minimum :math:`\Delta\omega` found.
        omega_value: :math:`\omega` at which the search was performed.
        sql: SQL reference value.
        t_hold: Holding-time strength.
        R: Norm-ball radius constraint.
    """

    samples: np.ndarray
    delta_omega_values: np.ndarray
    expectation_values: np.ndarray
    variance_values: np.ndarray
    deriv_values: np.ndarray
    is_fringe: np.ndarray
    best_params: tuple[float, float, float, float, float, float]
    best_delta_omega: float
    omega_value: float = 1.0
    sql: float = SQL
    t_hold: float = t_hold
    R: float = R_MAX

    _PARQUET_COLUMNS: ClassVar[list[str]] = [
        "omega_value",
        "t_hold",
        "sql",
        "R",
        "theta_A",
        "phi_A",
        "a_x",
        "a_y",
        "a_z",
        "a_zz",
        "norm_a",
        "delta_omega",
        "expectation",
        "variance",
        "derivative",
        "is_fringe",
        "ratio",
    ]

    def to_dataframe(self) -> pd.DataFrame:
        n = len(self.samples)
        norms_a = np.sqrt(
            self.samples[:, 2] ** 2 + self.samples[:, 3] ** 2 + self.samples[:, 4] ** 2,
        )
        ratios = np.where(
            np.isfinite(self.delta_omega_values) & (self.sql > 0),
            self.delta_omega_values / self.sql,
            float("inf"),
        )
        return pd.DataFrame(
            {
                "omega_value": [self.omega_value] * n,
                "t_hold": [self.t_hold] * n,
                "sql": [self.sql] * n,
                "R": [self.R] * n,
                "theta_A": self.samples[:, 0],
                "phi_A": self.samples[:, 1],
                "a_x": self.samples[:, 2],
                "a_y": self.samples[:, 3],
                "a_z": self.samples[:, 4],
                "a_zz": self.samples[:, 5],
                "norm_a": norms_a,
                "delta_omega": self.delta_omega_values,
                "expectation": self.expectation_values,
                "variance": self.variance_values,
                "derivative": self.deriv_values,
                "is_fringe": self.is_fringe.astype(int),
                "ratio": ratios,
            },
        )

    @classmethod
    def from_parquet(cls, path: str | Path) -> FreeAncillaModulatedSearchResult:
        df = pd.read_parquet(path)
        cls._validate_columns(df)
        samples = df[["theta_A", "phi_A", "a_x", "a_y", "a_z", "a_zz"]].to_numpy(
            dtype=float,
        )
        deltas = df["delta_omega"].to_numpy(dtype=float)
        exps = df["expectation"].to_numpy(dtype=float)
        vars_ = df["variance"].to_numpy(dtype=float)
        derivs = df["derivative"].to_numpy(dtype=float)

        for name, arr in [
            ("expectation", exps),
            ("variance", vars_),
            ("derivative", derivs),
        ]:
            if np.any(np.isnan(arr)):
                raise ValueError(
                    f"Parquet at {path} contains NaN in '{name}' column",
                )
        fringes = df["is_fringe"].to_numpy(dtype=bool)
        best_idx = int(np.nanargmin(deltas))
        return cls(
            samples=samples,
            delta_omega_values=deltas,
            expectation_values=exps,
            variance_values=vars_,
            deriv_values=derivs,
            is_fringe=fringes,
            best_params=(
                float(samples[best_idx, 0]),
                float(samples[best_idx, 1]),
                float(samples[best_idx, 2]),
                float(samples[best_idx, 3]),
                float(samples[best_idx, 4]),
                float(samples[best_idx, 5]),
            ),
            best_delta_omega=float(deltas[best_idx]),
            omega_value=float(df["omega_value"].iloc[0]),
            sql=float(df["sql"].iloc[0]),
            t_hold=float(df["t_hold"].iloc[0]),
            R=float(df["R"].iloc[0]),
        )


@dataclass
class FreeAncillaModulatedNelderMeadResult(ParquetSerializable):
    r"""Result of a single Nelder-Mead run for the free-ancilla ω-modulated protocol.

    Attributes:
        delta_omega_opt: Best sensitivity :math:`\Delta\omega` found.
        params_opt: Optimal 6-element parameter vector
            ``(theta_A, phi_A, a_x, a_y, a_z, a_zz)``.
        omega_true: True :math:`\omega` used for this optimisation.
        success: Whether the optimiser reported success.
        nfev: Number of function evaluations.
        message: Optimiser message.
        expectation_Jz: :math:`\langle J_z^S\rangle` at the optimal point.
        variance_Jz: :math:`\mathrm{Var}(J_z^S)` at the optimal point.
        t_hold: Holding-time strength.
        sql: SQL = 1 / t_hold.
        T_BS: Beam-splitter duration.
        fd_step: Finite-difference step size.
        history: Objective function values at each iteration.
    """

    delta_omega_opt: float
    params_opt: np.ndarray
    omega_true: float
    success: bool
    nfev: int
    message: str = ""
    expectation_Jz: float = 0.0
    variance_Jz: float = 0.0
    t_hold: float = t_hold
    sql: float = SQL
    T_BS: float = T_BS
    fd_step: float = FD_STEP
    history: list[float] = field(default_factory=list)

    _PARQUET_COLUMNS: ClassVar[list[str]] = [
        "delta_omega",
        "omega_true",
        "success",
        "nfev",
        "message",
        "expectation_Jz",
        "variance_Jz",
        "t_hold",
        "sql",
        "T_BS",
        "fd_step",
        "theta_A",
        "phi_A",
        "a_x",
        "a_y",
        "a_z",
        "a_zz",
        "history_json",
    ]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "delta_omega": [self.delta_omega_opt],
                "omega_true": [self.omega_true],
                "success": [int(self.success)],
                "nfev": [self.nfev],
                "message": [self.message],
                "expectation_Jz": [self.expectation_Jz],
                "variance_Jz": [self.variance_Jz],
                "t_hold": [self.t_hold],
                "sql": [self.sql],
                "T_BS": [self.T_BS],
                "fd_step": [self.fd_step],
                "theta_A": [float(self.params_opt[0])],
                "phi_A": [float(self.params_opt[1])],
                "a_x": [float(self.params_opt[2])],
                "a_y": [float(self.params_opt[3])],
                "a_z": [float(self.params_opt[4])],
                "a_zz": [float(self.params_opt[5])],
                "history_json": [json.dumps(self.history)],
            },
        )

    def _save_sidecars(self, path: Path) -> None:
        history_path = path.with_stem(path.stem + "-history")
        pd.DataFrame({"history": [json.dumps(self.history)]}).to_parquet(
            history_path,
            index=False,
        )

    @classmethod
    def _load_sidecars(cls, path: Path) -> dict:
        history_path = path.with_stem(path.stem + "-history")
        if history_path.exists():
            history_df = pd.read_parquet(history_path)
            return {"history": json.loads(history_df["history"].iloc[0])}
        return {}

    @classmethod
    def from_parquet(
        cls,
        path: str | Path,
    ) -> FreeAncillaModulatedNelderMeadResult:
        path = Path(path)
        df = pd.read_parquet(path)
        cls._validate_columns(df)
        sidecar = cls._load_sidecars(path)
        history = sidecar.get("history", [])
        return cls(
            delta_omega_opt=float(df["delta_omega"].iloc[0]),
            params_opt=np.array(
                [
                    float(df["theta_A"].iloc[0]),
                    float(df["phi_A"].iloc[0]),
                    float(df["a_x"].iloc[0]),
                    float(df["a_y"].iloc[0]),
                    float(df["a_z"].iloc[0]),
                    float(df["a_zz"].iloc[0]),
                ]
            ),
            omega_true=float(df["omega_true"].iloc[0]),
            success=bool(int(df["success"].iloc[0])),
            nfev=int(df["nfev"].iloc[0]),
            message=str(df["message"].iloc[0]),
            expectation_Jz=float(df["expectation_Jz"].iloc[0]),
            variance_Jz=float(df["variance_Jz"].iloc[0]),
            t_hold=float(df["t_hold"].iloc[0]),
            sql=float(df["sql"].iloc[0]),
            T_BS=float(df["T_BS"].iloc[0]),
            fd_step=float(df["fd_step"].iloc[0]),
            history=history,
        )


@dataclass
class FreeAncillaModulatedOmegaScanResult(ParquetSerializable):
    r"""Results of a :math:`\omega` scan for the free-ancilla ω-modulated protocol.

    Attributes:
        omega_values: Array of :math:`\omega` values scanned.
        best_params_per_omega: List of optimal 6-param tuples.
        best_delta_omega_per_omega: Optimal :math:`\Delta\omega` for each
            :math:`\omega`.
        sql_values: SQL = 1/t_hold for each :math:`\omega`.
        expectation_Jz_per_omega: :math:`\langle J_z^S\rangle` at each optimum.
        variance_Jz_per_omega: :math:`\mathrm{Var}(J_z^S)` at each optimum.
        t_hold: Holding-time strength (same for all :math:`\omega`).
    """

    omega_values: np.ndarray = field(default_factory=lambda: np.array([]))
    best_params_per_omega: list[tuple[float, float, float, float, float, float]] = (
        field(default_factory=list)
    )
    best_delta_omega_per_omega: np.ndarray = field(
        default_factory=lambda: np.array([]),
    )
    sql_values: np.ndarray = field(default_factory=lambda: np.array([]))
    expectation_Jz_per_omega: np.ndarray = field(
        default_factory=lambda: np.array([]),
    )
    variance_Jz_per_omega: np.ndarray = field(
        default_factory=lambda: np.array([]),
    )
    t_hold: float = t_hold

    _PARQUET_COLUMNS: ClassVar[list[str]] = [
        "omega",
        "best_delta_omega",
        "sql",
        "t_hold",
        "theta_A",
        "phi_A",
        "a_x",
        "a_y",
        "a_z",
        "a_zz",
        "expectation_Jz",
        "variance_Jz",
        "ratio",
    ]

    def to_dataframe(self) -> pd.DataFrame:
        rows: list[dict[str, float | str]] = []
        for i, omega in enumerate(self.omega_values):
            sql = float(self.sql_values[i]) if i < len(self.sql_values) else SQL
            best = (
                self.best_delta_omega_per_omega[i]
                if i < len(self.best_delta_omega_per_omega)
                else float("inf")
            )
            params = (
                self.best_params_per_omega[i]
                if i < len(self.best_params_per_omega)
                else (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            )
            exp_jz = (
                float(self.expectation_Jz_per_omega[i])
                if i < len(self.expectation_Jz_per_omega)
                else 0.0
            )
            var_jz = (
                float(self.variance_Jz_per_omega[i])
                if i < len(self.variance_Jz_per_omega)
                else 0.0
            )
            rows.append(
                {
                    "omega": float(omega),
                    "best_delta_omega": best,
                    "sql": sql,
                    "t_hold": self.t_hold,
                    "ratio": best / sql
                    if np.isfinite(best) and sql > 0
                    else float("inf"),
                    "theta_A": float(params[0]),
                    "phi_A": float(params[1]),
                    "a_x": float(params[2]),
                    "a_y": float(params[3]),
                    "a_z": float(params[4]),
                    "a_zz": float(params[5]),
                    "expectation_Jz": exp_jz,
                    "variance_Jz": var_jz,
                }
            )
        return pd.DataFrame(rows)

    @classmethod
    def from_parquet(
        cls,
        path: str | Path,
    ) -> FreeAncillaModulatedOmegaScanResult:
        df = pd.read_parquet(path)
        cls._validate_columns(df)
        omegas = df["omega"].to_numpy(dtype=float)
        best = df["best_delta_omega"].to_numpy(dtype=float)
        sql = df["sql"].to_numpy(dtype=float)
        exps = df["expectation_Jz"].to_numpy(dtype=float)
        vars_ = df["variance_Jz"].to_numpy(dtype=float)
        params_list: list[tuple[float, float, float, float, float, float]] = []
        for _, row in df.iterrows():
            params_list.append(
                (
                    float(row["theta_A"]),
                    float(row["phi_A"]),
                    float(row["a_x"]),
                    float(row["a_y"]),
                    float(row["a_z"]),
                    float(row["a_zz"]),
                )
            )
        return cls(
            omega_values=omegas,
            best_params_per_omega=params_list,
            best_delta_omega_per_omega=best,
            sql_values=sql,
            expectation_Jz_per_omega=exps,
            variance_Jz_per_omega=vars_,
            t_hold=float(df["t_hold"].iloc[0]),
        )


@dataclass
class FreeAncillaModulated2DSliceResult(ParquetSerializable):
    r"""Result from a 2D parameter slice over :math:`(\theta_A, a_{zz})`.

    Attributes:
        theta_A_values: Array of :math:`\theta_A` values.
        azz_values: Array of :math:`a_{zz}` values.
        delta_omega_grid: 2D array of :math:`\Delta\omega`, shape
            ``(len(theta_A_values), len(azz_values))``.
        omega_value: The :math:`\omega` value.
        sql: SQL = 1/t_hold reference.
        fixed_drive_params: The fixed (a_x, a_y, a_z) tuple used.
    """

    theta_A_values: np.ndarray
    azz_values: np.ndarray
    delta_omega_grid: np.ndarray
    omega_value: float = 1.0
    sql: float = SQL
    fixed_drive_params: tuple[float, float, float] = (0.0, 0.0, 0.0)

    _PARQUET_COLUMNS: ClassVar[list[str]] = [
        "theta_A",
        "azz",
        "delta_omega",
        "omega_value",
        "sql",
        "phi_A",
        "fixed_ax",
        "fixed_ay",
        "fixed_az",
    ]

    def to_dataframe(self) -> pd.DataFrame:
        n_t = len(self.theta_A_values)
        n_a = len(self.azz_values)
        rows: list[dict[str, float | str]] = [
            {
                "theta_A": float(self.theta_A_values[i]),
                "azz": float(self.azz_values[j]),
                "delta_omega": float(self.delta_omega_grid[i, j]),
                "omega_value": float(self.omega_value),
                "sql": float(self.sql),
                "phi_A": 0.0,
                "fixed_ax": float(self.fixed_drive_params[0]),
                "fixed_ay": float(self.fixed_drive_params[1]),
                "fixed_az": float(self.fixed_drive_params[2]),
            }
            for i in range(n_t)
            for j in range(n_a)
        ]
        return pd.DataFrame(rows)

    @classmethod
    def from_parquet(
        cls,
        path: str | Path,
    ) -> FreeAncillaModulated2DSliceResult:
        df = pd.read_parquet(path)
        cls._validate_columns(df)
        theta_A_unique = sorted(df["theta_A"].unique())
        azz_unique = sorted(df["azz"].unique())
        n_t = len(theta_A_unique)
        n_a = len(azz_unique)
        grid = np.full((n_t, n_a), np.nan, dtype=float)
        for _, row in df.iterrows():
            i = theta_A_unique.index(row["theta_A"])
            j = azz_unique.index(row["azz"])
            grid[i, j] = row["delta_omega"]

        fixed_ax = float(df["fixed_ax"].iloc[0])
        fixed_ay = float(df["fixed_ay"].iloc[0])
        fixed_az = float(df["fixed_az"].iloc[0])

        return cls(
            theta_A_values=np.array(theta_A_unique, dtype=float),
            azz_values=np.array(azz_unique, dtype=float),
            delta_omega_grid=grid,
            omega_value=float(df["omega_value"].iloc[0]),
            sql=float(df["sql"].iloc[0]),
            fixed_drive_params=(fixed_ax, fixed_ay, fixed_az),
        )


# ============================================================================
# Stage 1: 2D Slice (θ_A, a_zz) at Fixed Optimal Drive
# ============================================================================


def _modulated_parallel_worker(args: tuple) -> tuple[int, np.ndarray]:
    """Worker for parallel (θ_A, a_zz) slice (parallel_grid_scan protocol)."""
    theta_A_chunk, azz_vals, start_idx, kw = args
    omega = kw["omega"]
    fixed_drive = kw["fixed_drive"]
    phi_A = kw["phi_A"]
    t_hold = kw["t_hold"]
    T_BS = kw["T_BS"]
    fd_step = kw["fd_step"]
    a_x, a_y, a_z = fixed_drive
    n_t = len(theta_A_chunk)
    n_a = len(azz_vals)
    chunk_grid = np.full((n_t, n_a), np.inf, dtype=float)
    for i, tA in enumerate(theta_A_chunk):
        for j, a_val in enumerate(azz_vals):
            domega, _, _, _, _ = compute_free_ancilla_modulated_sensitivity(
                omega,
                tA,
                phi_A,
                a_x,
                a_y,
                a_z,
                a_val,
                t_hold=t_hold,
                T_BS=T_BS,
                fd_step=fd_step,
            )
            chunk_grid[i, j] = domega
    return start_idx, chunk_grid


def free_ancilla_modulated_2d_slice(
    omega: float,
    theta_A_range: tuple[float, float] = THETA_A_RANGE,
    azz_range: tuple[float, float] = AZZ_BOUNDS,
    n_grid: int = STAGE1_N,
    fixed_drive: tuple[float, float, float] = (0.0, 0.0, 0.0),
    phi_A: float = 0.0,
    t_hold: float = t_hold,
    T_BS: float = T_BS,
    fd_step: float = FD_STEP,
    n_jobs: int | None = None,
) -> FreeAncillaModulated2DSliceResult:
    r"""Run a 2D slice scan over :math:`(\theta_A, a_{zz})`.

    Fixed parameters: :math:`\phi_A = 0` and the provided ``fixed_drive``.

    Args:
        omega: Phase rate value.
        theta_A_range: (min, max) for :math:`\theta_A`.
        azz_range: (min, max) for :math:`a_{zz}`.
        n_grid: Number of points per axis (total grid = n_grid × n_grid).
        fixed_drive: Fixed ``(a_x, a_y, a_z)`` drive coefficients.
        phi_A: Fixed ancilla azimuthal angle (default 0).
        t_hold: Holding time.
        T_BS: Beam-splitter duration.
        fd_step: Finite-difference step size.
        n_jobs: Number of parallel workers. ``None`` = sequential.
            ``-1`` = all CPUs.

    Returns:
        FreeAncillaModulated2DSliceResult.
    """
    theta_A_vals = np.linspace(theta_A_range[0], theta_A_range[1], n_grid)
    azz_vals = np.linspace(azz_range[0], azz_range[1], n_grid)
    a_x, a_y, a_z = fixed_drive

    if n_jobs is None or n_jobs == 1:

        def _sensitivity(theta_A: float, a_zz: float) -> float:
            domega, _, _, _, _ = compute_free_ancilla_modulated_sensitivity(
                omega,
                theta_A,
                phi_A,
                a_x,
                a_y,
                a_z,
                a_zz,
                t_hold=t_hold,
                T_BS=T_BS,
                fd_step=fd_step,
            )
            return domega

        grid = sequential_grid_scan(theta_A_vals, azz_vals, _sensitivity)
    else:
        grid = parallel_grid_scan(
            theta_A_vals,
            azz_vals,
            _modulated_parallel_worker,
            n_jobs=n_jobs,
            omega=omega,
            fixed_drive=fixed_drive,
            phi_A=phi_A,
            t_hold=t_hold,
            T_BS=T_BS,
            fd_step=fd_step,
        )

    return FreeAncillaModulated2DSliceResult(
        theta_A_values=theta_A_vals,
        azz_values=azz_vals,
        delta_omega_grid=grid,
        omega_value=omega,
        sql=1.0 / t_hold,
        fixed_drive_params=fixed_drive,
    )


# ============================================================================
# Stage 2: 6D Random Search
# ============================================================================


def _sample_6d_config(
    rng: np.random.Generator,
    *,
    R: float = R_MAX,
    azz_bounds: tuple[float, float] = AZZ_BOUNDS,
) -> tuple[float, float, float, float, float, float]:
    """Sample a single 6D configuration for the free-ancilla ω-modulated protocol.

    Args:
        rng: NumPy random generator.
        R: Norm-ball radius for the (a_x, a_y, a_z) drive vector.
        azz_bounds: (min, max) for a_zz.

    Returns:
        Tuple ``(theta_A, phi_A, a_x, a_y, a_z, a_zz)``.
    """
    theta_A = rng.uniform(0.0, np.pi)
    phi_A = rng.uniform(0.0, 2.0 * np.pi)
    drive, azz = marsaglia_ball_sample(rng, 1, R, azz_bounds[0], azz_bounds[1])
    return (
        theta_A,
        phi_A,
        float(drive[0, 0]),
        float(drive[0, 1]),
        float(drive[0, 2]),
        float(azz[0]),
    )


def free_ancilla_modulated_random_search(
    omega: float,
    n_samples: int = N_SAMP_STAGE2,
    *,
    R: float = R_MAX,
    azz_bounds: tuple[float, float] = AZZ_BOUNDS,
    t_hold: float = t_hold,
    T_BS: float = T_BS,
    fd_step: float = FD_STEP,
    seed: int | None = 42,
) -> FreeAncillaModulatedSearchResult:
    """6D random search over (θ_A, φ_A, a_x, a_y, a_z, a_zz).

    Args:
        omega: Phase rate value.
        n_samples: Number of random points to evaluate.
        R: Norm-ball radius for (a_x, a_y, a_z).
        azz_bounds: (min, max) for a_zz.
        t_hold: Holding time.
        T_BS: Beam-splitter duration.
        fd_step: Finite-difference step size.
        seed: Random seed for reproducibility.

    Returns:
        FreeAncillaModulatedSearchResult.
    """
    rng = np.random.default_rng(seed)

    samples = np.zeros((n_samples, 6), dtype=float)
    deltas = np.full(n_samples, np.inf, dtype=float)
    exps = np.zeros(n_samples, dtype=float)
    vars_ = np.zeros(n_samples, dtype=float)
    derivs = np.zeros(n_samples, dtype=float)
    fringes = np.zeros(n_samples, dtype=bool)

    for i in range(n_samples):
        theta_A, phi_A, a_x, a_y, a_z, a_zz = _sample_6d_config(
            rng,
            R=R,
            azz_bounds=azz_bounds,
        )
        samples[i, :] = [theta_A, phi_A, a_x, a_y, a_z, a_zz]

        domega, exp_val, var_val, deriv, fringe = (
            compute_free_ancilla_modulated_sensitivity(
                omega,
                theta_A,
                phi_A,
                a_x,
                a_y,
                a_z,
                a_zz,
                t_hold=t_hold,
                T_BS=T_BS,
                fd_step=fd_step,
            )
        )
        deltas[i] = domega
        exps[i] = exp_val
        vars_[i] = var_val
        derivs[i] = deriv
        fringes[i] = fringe

    best_idx = int(np.nanargmin(deltas))
    best_params: tuple[float, float, float, float, float, float] = (
        float(samples[best_idx, 0]),
        float(samples[best_idx, 1]),
        float(samples[best_idx, 2]),
        float(samples[best_idx, 3]),
        float(samples[best_idx, 4]),
        float(samples[best_idx, 5]),
    )

    return FreeAncillaModulatedSearchResult(
        samples=samples,
        delta_omega_values=deltas,
        expectation_values=exps,
        variance_values=vars_,
        deriv_values=derivs,
        is_fringe=fringes,
        best_params=best_params,
        best_delta_omega=float(deltas[best_idx]),
        omega_value=omega,
        sql=1.0 / t_hold,
        t_hold=t_hold,
        R=R,
    )


# ============================================================================
# Stage 3: Nelder-Mead Optimisation
# ============================================================================


def _modulated_6d_objective(
    params: np.ndarray,
    omega_true: float,
    t_hold: float = t_hold,
    T_BS: float = T_BS,
    fd_step: float = FD_STEP,
    a_zz_penalty_lo: float = AZZ_BOUNDS[0],
    a_zz_penalty_hi: float = AZZ_BOUNDS[1],
    norm_ball_R: float = R_MAX,
    penalty_scale: float = 1e6,
) -> float:
    """6D objective function for minimising Δω.

    params = [theta_A, phi_A, a_x, a_y, a_z, a_zz]

    Args:
        params: 6-element parameter vector.
        omega_true: True phase rate.
        t_hold: Holding time.
        T_BS: Beam-splitter duration.
        fd_step: Finite-difference step.
        a_zz_penalty_lo: Lower bound for a_zz.
        a_zz_penalty_hi: Upper bound for a_zz.
        norm_ball_R: Norm-ball radius for drive vector.
        penalty_scale: Scale for bound-violation penalty.

    Returns:
        Δω (plus infinite penalty if bounds violated).
    """
    theta_A = float(params[0])
    phi_A = float(params[1])
    a_x = float(params[2])
    a_y = float(params[3])
    a_z = float(params[4])
    a_zz = float(params[5])

    # Bound enforcement
    penalty = 0.0

    # θ_A ∈ [0, π]
    if theta_A < 0.0:
        penalty += penalty_scale * theta_A**2
    if theta_A > np.pi:
        penalty += penalty_scale * (theta_A - np.pi) ** 2

    # φ_A ∈ [0, 2π)
    if phi_A < 0.0:
        penalty += penalty_scale * phi_A**2
    if phi_A > 2.0 * np.pi:
        penalty += penalty_scale * (phi_A - 2.0 * np.pi) ** 2

    # ||a|| ≤ R
    norm_a = np.sqrt(a_x**2 + a_y**2 + a_z**2)
    if norm_a > norm_ball_R:
        penalty += penalty_scale * (norm_a - norm_ball_R) ** 2

    # a_zz ∈ [lo, hi]
    if a_zz < a_zz_penalty_lo:
        penalty += penalty_scale * (a_zz_penalty_lo - a_zz) ** 2
    if a_zz > a_zz_penalty_hi:
        penalty += penalty_scale * (a_zz - a_zz_penalty_hi) ** 2

    if penalty > 0.0:
        return float(1e10 + penalty)

    domega, _, _, _, _ = compute_free_ancilla_modulated_sensitivity(
        omega_true,
        theta_A,
        phi_A,
        a_x,
        a_y,
        a_z,
        a_zz,
        t_hold=t_hold,
        T_BS=T_BS,
        fd_step=fd_step,
    )
    return domega


def run_modulated_nelder_mead(
    omega_true: float,
    x0: np.ndarray | None = None,
    seed: int | None = None,
    maxiter: int = 5000,
    xatol: float = 1e-8,
    fatol: float = 1e-8,
    adaptive: bool = True,
    t_hold: float = t_hold,
    T_BS: float = T_BS,
    fd_step: float = FD_STEP,
    track_history: bool = False,
) -> FreeAncillaModulatedNelderMeadResult:
    """Run Nelder-Mead optimisation for the free-ancilla ω-modulated protocol.

    Args:
        omega_true: True phase rate parameter.
        x0: Initial 6-parameter vector. Randomly generated if None.
        seed: Random seed (used if x0 is None).
        maxiter: Maximum Nelder-Mead iterations.
        xatol: Absolute parameter tolerance.
        fatol: Absolute function tolerance.
        adaptive: Use adaptive Nelder-Mead parameters.
        t_hold: Holding time.
        T_BS: Beam-splitter duration.
        fd_step: Finite-difference step size.
        track_history: If True, record objective values per iteration.

    Returns:
        FreeAncillaModulatedNelderMeadResult.
    """

    if x0 is None:
        rng = np.random.default_rng(seed)
        theta_A = rng.uniform(0.0, np.pi)
        phi_A = rng.uniform(0.0, 2.0 * np.pi)
        drive, azz = marsaglia_ball_sample(rng, 1, R_MAX, AZZ_BOUNDS[0], AZZ_BOUNDS[1])
        x0 = np.array(
            [
                theta_A,
                phi_A,
                float(drive[0, 0]),
                float(drive[0, 1]),
                float(drive[0, 2]),
                float(azz[0]),
            ]
        )
    else:
        x0 = np.asarray(x0, dtype=float)
        assert x0.shape == (6,), f"x0 must have 6 elements, got {x0.shape}"

    def objective(p: np.ndarray) -> float:
        return _modulated_6d_objective(
            p,
            omega_true,
            t_hold=t_hold,
            T_BS=T_BS,
            fd_step=fd_step,
        )

    history: list[float] = []

    def callback(_x: np.ndarray) -> None:
        if track_history:
            val = objective(_x)
            history.append(val)

    result = minimize(
        objective,
        x0=x0,
        method="Nelder-Mead",
        callback=callback if track_history else None,  # type: ignore[arg-type]
        options={  # type: ignore[call-overload]
            "maxiter": maxiter,
            "xatol": xatol,
            "fatol": fatol,
            "adaptive": adaptive,
        },
    )

    opt_params = result.x.copy()

    # Diagnostics at the optimal point
    ops = build_two_qubit_operators()
    psi_final = evolve_phase_modulated_circuit(
        free_ancilla_initial_state(
            float(opt_params[0]),
            float(opt_params[1]),
        ),
        T_BS,
        t_hold,
        omega_true,
        float(opt_params[2]),
        float(opt_params[3]),
        float(opt_params[4]),
        float(opt_params[5]),
        ops,
    )
    exp_val, var_val = compute_expectation_and_variance(psi_final, ops["Jz_S"])

    return FreeAncillaModulatedNelderMeadResult(
        delta_omega_opt=float(result.fun),
        params_opt=opt_params,
        omega_true=omega_true,
        success=bool(result.success),
        nfev=int(result.nfev),
        message=str(result.message),
        expectation_Jz=exp_val,
        variance_Jz=var_val,
        t_hold=t_hold,
        sql=1.0 / t_hold,
        T_BS=T_BS,
        fd_step=fd_step,
        history=history.copy(),
    )


# ============================================================================
# ω Scan: Stage 2 + Stage 3 Pipeline
# ============================================================================


def run_modulated_omega_scan(
    omega_values: list[float] | np.ndarray,
    n_random: int = N_SAMP_STAGE2,
    n_nm_refine: int = N_NM_REFINE,
    seed: int | None = 42,
    maxiter: int = 5000,
    t_hold: float = t_hold,
    T_BS: float = T_BS,
    R: float = R_MAX,
    azz_bounds: tuple[float, float] = AZZ_BOUNDS,
) -> FreeAncillaModulatedOmegaScanResult:
    """Scan over ω values with 6D random search and Nelder-Mead refinement.

    For each ω:
    1. Run n_random 6D random evaluations.
    2. Select the best n_nm_refine points.
    3. Run Nelder-Mead refinement from each selected point.
    4. Record the best overall result.

    Args:
        omega_values: ω values to scan.
        n_random: Number of random search points per ω.
        n_nm_refine: Number of Nelder-Mead refinements per ω.
        seed: Base random seed (incremented per ω).
        maxiter: Maximum Nelder-Mead iterations.
        t_hold: Holding time.
        T_BS: Beam-splitter duration.
        R: Norm-ball radius.
        azz_bounds: (min, max) for a_zz.

    Returns:
        FreeAncillaModulatedOmegaScanResult.
    """
    config = TwoPhaseConfig(
        n_random=n_random,
        n_nm_refine=n_nm_refine,
        nm_maxiter=maxiter,
        seed=seed,
    )

    def rs_fn(
        n_samples: int, seed: int | None, **kw: Any
    ) -> FreeAncillaModulatedSearchResult:
        return free_ancilla_modulated_random_search(
            omega=kw["omega"],
            n_samples=n_samples,
            R=R,
            azz_bounds=azz_bounds,
            t_hold=t_hold,
            T_BS=T_BS,
            seed=seed,
        )

    def nm_fn(
        x0: np.ndarray, seed: int | None, **kw: Any
    ) -> FreeAncillaModulatedNelderMeadResult:
        return run_modulated_nelder_mead(
            omega_true=kw["omega_true"],
            x0=x0,
            seed=seed,
            maxiter=maxiter,
            t_hold=t_hold,
            T_BS=T_BS,
            track_history=False,
        )

    best_results, _all_results = run_omega_scan(
        omega_values,
        rs_fn,
        nm_fn,
        config,
        rs_kwargs={
            "R": R,
            "azz_bounds": azz_bounds,
            "t_hold": t_hold,
            "T_BS": T_BS,
        },
        nm_kwargs={
            "t_hold": t_hold,
            "T_BS": T_BS,
        },
    )

    omega_arr = np.asarray(omega_values, dtype=float)
    best_params_list: list[tuple[float, float, float, float, float, float]] = []
    best_deltas: list[float] = []
    sql_vals: list[float] = []
    exp_vals: list[float] = []
    var_vals: list[float] = []

    for best_nm in best_results:
        best_params_list.append(
            (
                float(best_nm.params_opt[0]),
                float(best_nm.params_opt[1]),
                float(best_nm.params_opt[2]),
                float(best_nm.params_opt[3]),
                float(best_nm.params_opt[4]),
                float(best_nm.params_opt[5]),
            )
        )
        best_deltas.append(best_nm.delta_omega_opt)
        sql_vals.append(1.0 / t_hold)
        exp_vals.append(best_nm.expectation_Jz)
        var_vals.append(best_nm.variance_Jz)

    return FreeAncillaModulatedOmegaScanResult(
        omega_values=omega_arr,
        best_params_per_omega=best_params_list,
        best_delta_omega_per_omega=np.array(best_deltas, dtype=float),
        sql_values=np.array(sql_vals, dtype=float),
        expectation_Jz_per_omega=np.array(exp_vals, dtype=float),
        variance_Jz_per_omega=np.array(var_vals, dtype=float),
        t_hold=t_hold,
    )


# ============================================================================
# Plot Functions
# ============================================================================


def plot_slice_heatmap(
    result: FreeAncillaModulated2DSliceResult | str | Path,
    save_path: str | Path,
    figsize: tuple[float, float] = (9, 7),
) -> Path:
    r"""Heatmap of :math:`\Delta\omega/\text{SQL}` over :math:`(\theta_A, a_{zz})`.

    Args:
        result: Slice result or path to Parquet.
        save_path: Output SVG path.
        figsize: Figure size.

    Returns:
        Path to saved SVG.
    """
    if isinstance(result, (str, Path)):
        result = FreeAncillaModulated2DSliceResult.from_parquet(result)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    ratio_grid = result.delta_omega_grid / result.sql
    finite_mask = np.isfinite(ratio_grid)
    data_min = np.nanmin(ratio_grid) if np.any(finite_mask) else 0.99
    vmin = min(0.99, data_min * 0.95)
    vmax = max(2.0, np.nanmax(ratio_grid) * 1.05)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.pcolormesh(
        result.azz_values,
        result.theta_A_values,
        np.clip(ratio_grid, vmin, vmax),
        shading="auto",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )
    cbar = fig.colorbar(im, ax=ax, label=r"$\Delta\omega / \mathrm{SQL}$")
    # Dynamic ticks: include 1.0 (SQL) and the actual min for sub-SQL visibility
    tick_values = [1.0]
    if vmin < 0.99:
        tick_values.insert(0, round(vmin, 2))
    tick_values.extend(val for val in [1.25, 1.5, 1.75, 2.0] if val <= vmax)
    cbar.set_ticks(tick_values)

    ax.set_xlabel(r"$a_{zz}$")
    ax.set_ylabel(r"$\theta_A$")
    fx, fy, fz = result.fixed_drive_params
    ax.set_title(
        rf"$(\theta_A, a_{{zz}})$ slice at $\omega={result.omega_value:.1f}$, "
        rf"$a_x={fx:.1f}, a_y={fy:.1f}, a_z={fz:.1f}$, $\phi_A=0$",
    )

    if np.any(finite_mask):
        masked_grid = np.ma.masked_invalid(ratio_grid)
        cs = ax.contour(
            result.azz_values,
            result.theta_A_values,
            masked_grid,
            levels=[1.0],
            colors="red",
            linewidths=1.5,
            linestyles="--",
        )
        ax.clabel(cs, fmt="SQL", fontsize=9, colors="red")

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_best_ratio_by_omega(
    scan_result: FreeAncillaModulatedOmegaScanResult | str | Path,
    save_path: str | Path,
    figsize: tuple[float, float] = (8, 5),
    label: str = "Free ancilla + ω-modulated",
    baseline_ratio: float | None = None,
    baseline_label: str = "Fixed ancilla (20260519)",
) -> Path:
    r"""Line plot of best :math:`\Delta\omega/\text{SQL}` vs :math:`\omega`.

    Args:
        scan_result: OmegaScanResult or path to Parquet.
        save_path: Output SVG path.
        figsize: Figure size.
        label: Label for the data line.
        baseline_ratio: Optional baseline ratio to show as horizontal line.
        baseline_label: Label for the baseline line.

    Returns:
        Path to saved SVG.
    """
    if isinstance(scan_result, (str, Path)):
        scan_result = FreeAncillaModulatedOmegaScanResult.from_parquet(scan_result)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)

    omegas = scan_result.omega_values
    ratios = scan_result.best_delta_omega_per_omega / scan_result.sql_values

    ax.plot(omegas, ratios, "o-", color="C0", linewidth=2, markersize=6, label=label)

    if baseline_ratio is not None:
        ax.axhline(
            y=baseline_ratio,
            color="gray",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            label=baseline_label,
        )

    ax.axhline(
        y=1.0,
        color="red",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label="SQL",
    )
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$\min \Delta\omega \;/\; \mathrm{SQL}$")
    ax.set_title("Best sensitivity ratio vs $\\omega$")
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_optimal_ancilla_state_by_omega(
    scan_result: FreeAncillaModulatedOmegaScanResult | str | Path,
    save_path: str | Path,
    figsize: tuple[float, float] = (8, 6),
) -> Path:
    r"""Plot :math:`\theta_A^*(\omega)` and :math:`\phi_A^*(\omega)` vs :math:`\omega`.

    Args:
        scan_result: OmegaScanResult or path to Parquet.
        save_path: Output SVG path.
        figsize: Figure size.

    Returns:
        Path to saved SVG.
    """
    if isinstance(scan_result, (str, Path)):
        scan_result = FreeAncillaModulatedOmegaScanResult.from_parquet(scan_result)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    omegas = scan_result.omega_values
    theta_opt = np.array([p[0] for p in scan_result.best_params_per_omega])
    phi_opt = np.array([p[1] for p in scan_result.best_params_per_omega])

    ax1.plot(omegas, theta_opt, "o-", color="C0", linewidth=2, markersize=6)
    ax1.set_ylabel(r"$\theta_A^*$")
    ax1.set_title("Optimal ancilla polar angle vs $\\omega$")
    ax1.set_ylim(0, np.pi)

    ax2.plot(omegas, phi_opt, "s-", color="C1", linewidth=2, markersize=6)
    ax2.set_xlabel(r"$\omega$")
    ax2.set_ylabel(r"$\phi_A^*$")
    ax2.set_title("Optimal ancilla azimuthal angle vs $\\omega$")
    ax2.set_ylim(0, 2 * np.pi)

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_cross_experiment_comparison(
    scan_result_free: FreeAncillaModulatedOmegaScanResult | str | Path,
    baseline_delta: np.ndarray | None = None,
    baseline_label: str = "Fixed ancilla (20260519)",
    save_path: str | Path = "",
    figsize: tuple[float, float] = (8, 6),
) -> Path:
    """Comparison chart: free-ancilla vs fixed-ancilla at each ω.

    Args:
        scan_result_free: Free-ancilla OmegaScanResult or path to Parquet.
        baseline_delta: Array of baseline Δω at each ω (same length as scan).
        baseline_label: Label for baseline bars.
        save_path: Output SVG path.
        figsize: Figure size.

    Returns:
        Path to saved SVG.
    """
    if isinstance(scan_result_free, (str, Path)):
        scan_result_free = FreeAncillaModulatedOmegaScanResult.from_parquet(
            scan_result_free,
        )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)

    omegas = scan_result_free.omega_values
    free_ratios = (
        scan_result_free.best_delta_omega_per_omega / scan_result_free.sql_values
    )
    n = len(omegas)
    x = np.arange(n)
    width = 0.35

    ax.bar(
        x - width / 2,
        free_ratios,
        width,
        color="C0",
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
        label=r"Free ancilla + $\omega$-modulated",
    )

    if baseline_delta is not None:
        baseline_ratios = baseline_delta / scan_result_free.sql_values
        ax.bar(
            x + width / 2,
            baseline_ratios,
            width,
            color="gray",
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
            label=baseline_label,
        )

    ax.axhline(
        y=1.0,
        color="red",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label="SQL",
    )
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$\min \Delta\omega \;/\; \mathrm{SQL}$")
    ax.set_title("Cross-experiment comparison")
    ax.set_xticks(x)
    ax.set_xticklabels([rf"$\omega={t:.1f}$" for t in omegas])
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


# ============================================================================
# Data Generation Functions
# ============================================================================


def generate_stage1_slices(force: bool = False, n_jobs: int | None = None) -> None:
    """Stage 1: 2D slices (θ_A, a_zz) at each ω with fixed 20260519 optimal drive."""
    print(f"[run]  Stage 1: (θ_A, a_zz) slices at {OMEGA_VALS}")

    for omega in OMEGA_VALS:
        tag = f"stage1-slice-omega{omega}"
        csv_p = _parquet_path(tag)
        fig_p = _fig_path(f"stage1-slice-omega{omega}")

        if csv_p.exists() and not force:
            print(f"  [skip] {csv_p.name} exists (use --force to overwrite)")
            result = FreeAncillaModulated2DSliceResult.from_parquet(csv_p)
        else:
            if omega not in DEFAULT_FIXED_DRIVE:
                print(f"  [skip] No 20260519 optimal drive for ω={omega}")
                continue
            print(f"  [run]  Computing (θ_A, a_zz) slice at ω={omega}...")
            result = free_ancilla_modulated_2d_slice(
                omega=omega,
                theta_A_range=THETA_A_RANGE,
                azz_range=AZZ_BOUNDS,
                n_grid=STAGE1_N,
                fixed_drive=DEFAULT_FIXED_DRIVE[omega],
                phi_A=0.0,
                n_jobs=n_jobs,
            )
            result.save_parquet(csv_p)
            print(f"  [save] {csv_p}")

        plot_slice_heatmap(result, fig_p)
        print(f"  [fig]  {fig_p}")


def _run_stage2_omega_scan_single(omega: float, force: bool) -> None:
    """Run Stage 2 (random search) for a single ω, with caching."""
    tag = f"stage2-random-omega{omega}"
    csv_p = _parquet_path(tag)

    if csv_p.exists() and not force:
        print(f"    [skip] {csv_p.name} exists (use --force to overwrite)")
        return

    print(
        f"    [run]  Stage 2: 6D random search at ω={omega} ({N_SAMP_STAGE2} samples)..."
    )
    result = free_ancilla_modulated_random_search(
        omega=omega,
        n_samples=N_SAMP_STAGE2,
        seed=42 + int(omega * 1000),
    )
    result.save_parquet(csv_p)
    print(f"    [save] {csv_p}")


def generate_stage2_random_search(force: bool = False) -> None:
    """Stage 2: 6D random search at all ω values (parallel)."""
    print(f"[run]  Stage 2: 6D random search at {len(OMEGA_VALS)} ω values")
    for omega in OMEGA_VALS:
        _run_stage2_omega_scan_single(omega, force)


def _run_stage3_omega_scan_single(omega: float) -> dict[str, float | np.ndarray]:
    """Run Stage 2 + Stage 3 for a single ω.

    Returns a dict with per-ω results for aggregation.
    """
    base_seed: int = 42

    # Stage 2: Random search
    rs_result = free_ancilla_modulated_random_search(
        omega,
        n_samples=N_SAMP_STAGE2,
        seed=base_seed + int(omega * 1000),
    )

    # Sort by Δω, take top N_NM_REFINE
    sorted_indices = np.argsort(rs_result.delta_omega_values)
    top_indices = sorted_indices[:N_NM_REFINE]

    # Stage 3: Nelder-Mead refinement
    nm_results: list[FreeAncillaModulatedNelderMeadResult] = []
    for rank, idx in enumerate(top_indices):
        x0 = rs_result.samples[idx].copy()
        nm = run_modulated_nelder_mead(
            omega_true=omega,
            x0=x0,
            seed=base_seed + int(omega * 1000) + 10000 + rank,
            maxiter=5000,
            track_history=False,
        )
        nm_results.append(nm)

    nm_results.sort(key=lambda r: r.delta_omega_opt)
    best_nm = nm_results[0]

    return {
        "omega": omega,
        "best_delta_omega": best_nm.delta_omega_opt,
        "theta_A": float(best_nm.params_opt[0]),
        "phi_A": float(best_nm.params_opt[1]),
        "a_x": float(best_nm.params_opt[2]),
        "a_y": float(best_nm.params_opt[3]),
        "a_z": float(best_nm.params_opt[4]),
        "a_zz": float(best_nm.params_opt[5]),
        "expectation_Jz": best_nm.expectation_Jz,
        "variance_Jz": best_nm.variance_Jz,
    }


def _compute_omega_scan_core() -> FreeAncillaModulatedOmegaScanResult:
    """Parallel ω scan computation (Stage 2 + Stage 3 for all ω values).

    Runs the 6D random search (Stage 2) and Nelder-Mead refinement (Stage 3)
    for each ω value in OMEGA_VALS via ProcessPoolExecutor, then assembles
    and returns a FreeAncillaModulatedOmegaScanResult.
    """
    print(f"[run]  Computing ω-scan for {len(OMEGA_VALS)} ω values (parallel)...")
    per_omega_results: list[dict] = []

    max_workers = min(32, os.cpu_count() or 1)
    mp_ctx = _mp.get_context("fork")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=mp_ctx,
    ) as executor:
        fut_to_omega = {
            executor.submit(_run_stage3_omega_scan_single, omega): omega
            for omega in OMEGA_VALS
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
            float(r["theta_A"]),
            float(r["phi_A"]),
            float(r["a_x"]),
            float(r["a_y"]),
            float(r["a_z"]),
            float(r["a_zz"]),
        )
        for r in per_omega_results
    ]
    exp_vals = [float(r["expectation_Jz"]) for r in per_omega_results]
    var_vals = [float(r["variance_Jz"]) for r in per_omega_results]
    sql_vals = [1.0 / t_hold] * len(omega_arr)

    return FreeAncillaModulatedOmegaScanResult(
        omega_values=omega_arr,
        best_params_per_omega=best_params,
        best_delta_omega_per_omega=np.array(best_deltas, dtype=float),
        sql_values=np.array(sql_vals, dtype=float),
        expectation_Jz_per_omega=np.array(exp_vals, dtype=float),
        variance_Jz_per_omega=np.array(var_vals, dtype=float),
        t_hold=t_hold,
    )


def generate_omega_scan(force: bool = False) -> None:
    """Full ω scan with Stage 2 + Stage 3 (parallel per ω)."""
    csv_p = _parquet_path("omega-scan")
    fig_p_ratio = _fig_path("omega-scan-ratio")
    fig_p_theta = _fig_path("omega-scan-optimal-theta")
    fig_p_comparison = _fig_path("cross-experiment-comparison")

    if csv_p.exists() and not force:
        print(f"[skip] {csv_p.name} exists (use --force to overwrite)")
        result = FreeAncillaModulatedOmegaScanResult.from_parquet(csv_p)
    else:
        result = _compute_omega_scan_core()
        result.save_parquet(csv_p)
        print(f"[save] {csv_p}")

    plot_best_ratio_by_omega(result, fig_p_ratio)
    print(f"[fig]  {fig_p_ratio}")

    plot_optimal_ancilla_state_by_omega(result, fig_p_theta)
    print(f"[fig]  {fig_p_theta}")

    plot_cross_experiment_comparison(result, save_path=fig_p_comparison)
    print(f"[fig]  {fig_p_comparison}")


# ============================================================================
# CLI Entry Point
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate 2026-06-10 report figures and data",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run all simulations (overwrite existing files)",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Generate only one dataset, e.g. 'omega-scan'",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="Number of parallel workers. -1 = all CPUs (default: sequential)",
    )
    args = parser.parse_args()

    (REPORTS_DIR / REPORT_DATE / "raw_data").mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / REPORT_DATE / "figures").mkdir(parents=True, exist_ok=True)

    n_jobs: int | None = -1 if args.jobs == -1 else args.jobs

    tasks: dict[str, Callable[..., object]] = {
        "stage1-slice": generate_stage1_slices,
        "stage2-random": generate_stage2_random_search,
        "omega-scan": generate_omega_scan,
    }

    def _dispatch(
        func: Callable[..., object],
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
