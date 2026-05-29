"""
Local module for the 2026-05-28 Free-Ancilla-Initial-State in Driven-Ancilla Metrology.

Contains all code exclusive to this report:
- Free-ancilla initial state preparation (theta_A, phi_A parameterisation)
- Scenario dispatchers (A/B/C/D) with random search and Nelder--Mead refinement
- 2D slice over (theta_A, a_zz) with H_A = 0
- Cross-scenario comparison (bar chart, envelope overlay)
- Parquet roundtrip for all new dataclasses

Usage:
    uv run python reports/20260528/local.py --force

This module is **not** importable as ``reports.20260528.local`` (the directory
name contains hyphens).  Instead, importers add the report directory to
``sys.path`` and do ``import local``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

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
    build_two_qubit_operators,
    evolve_drive_circuit,
)
from src.analysis.ancilla_optimization import (
    compute_expectation_and_variance,
    single_qubit_state,
)

sns.set_theme(style="whitegrid")

# ============================================================================
# Constants
# ============================================================================

REPORTS_DIR = Path(__file__).resolve().parent.parent.parent / "reports"
REPORT_DATE = "20260528"
T_H: float = 10.0
SQL: float = 1.0 / T_H  # 0.1
FD_STEP: float = 1e-6
T_BS: float = np.pi / 2.0

# θ values for the scan (5 values, matching the report)
DRIVE_THETA_VALS: list[float] = [0.1, 0.5, 1.0, 2.0, 5.0]

# Random-search sample counts
N_SAMP_A: int = 500  # Scenario A (reproducing 20260518)
N_SAMP_BCD: int = 2000  # Scenarios B, C, D
N_NM_REFINE: int = 30  # Number of Nelder--Mead refinements per θ

# Norm-ball parameters
R_MAX: float = 10.0
AZZ_BOUNDS: tuple[float, float] = (-5.0, 5.0)

# 2D slice parameters
THETA_A_RANGE: tuple[float, float] = (0.0, np.pi)
SLICE_N: int = 201


# ============================================================================
# Path Helpers
# ============================================================================


def _parquet_path(name: str) -> Path:
    return REPORTS_DIR / REPORT_DATE / "raw_data" / f"{REPORT_DATE}-{name}.parquet"


def _fig_path(name: str) -> Path:
    return REPORTS_DIR / REPORT_DATE / "figures" / f"{REPORT_DATE}-{name}.svg"


# ============================================================================
# Free-Ancilla Initial State
# ============================================================================


def free_ancilla_initial_state(theta_A: float, phi_A: float) -> np.ndarray:
    r"""Construct the system--ancilla product initial state.

    :math:`|\Psi_0\rangle = |1,0\rangle_S \otimes |\psi_A(\theta_A, \phi_A)\rangle`

    where :math:`|\psi_A\rangle = \cos(\theta_A/2)\,|1,0\rangle_A
    + e^{i\phi_A}\sin(\theta_A/2)\,|0,1\rangle_A`.

    Args:
        theta_A: Ancilla polar angle :math:`\in [0, \pi]`.
        phi_A: Ancilla azimuthal angle :math:`\in [0, 2\pi)`.

    Returns:
        Normalised 4-vector in the :math:`\{|00\rangle, |01\rangle, |10\rangle,
        |11\rangle\}` basis.
    """
    psi_S = np.array([1.0, 0.0], dtype=complex)  # |1,0⟩_S
    psi_A = single_qubit_state(theta_A, phi_A)
    state = np.kron(psi_S, psi_A)
    assert np.isclose(np.linalg.norm(state), 1.0), (
        "Free-ancilla initial state must be normalised"
    )
    return state


# ============================================================================
# Sensitivity Computation with Free Ancilla
# ============================================================================


def compute_free_ancilla_sensitivity(
    theta_true: float,
    theta_A: float,
    phi_A: float,
    a_x: float,
    a_y: float,
    a_z: float,
    a_zz: float,
    *,
    T_H: float = T_H,
    T_BS: float = T_BS,
    fd_step: float = FD_STEP,
) -> tuple[float, float, float, float, bool]:
    r"""Compute the error-propagation sensitivity :math:`\Delta\theta`.

    Uses the free-ancilla initial state :math:`|\Psi(\theta_A,\phi_A)\rangle`.
    Returns the same 5-tuple as ``compute_sensitivity_with_extra`` in
    ``20260527/local.py``.

    Args:
        theta_true: True phase rate parameter.
        theta_A: Ancilla polar angle.
        phi_A: Ancilla azimuthal angle.
        a_x: Ancilla J_x drive coefficient.
        a_y: Ancilla J_y drive coefficient.
        a_z: Ancilla J_z drive coefficient.
        a_zz: Ising interaction coefficient.
        T_H: Holding-time strength.
        T_BS: Beam-splitter duration.
        fd_step: Finite-difference step size.

    Returns:
        Tuple ``(delta_theta, expectation, variance, derivative, is_fringe)``.
        Returns ``(inf, exp_val, var_val, 0.0, True)`` at a fringe extremum.
    """
    psi0 = free_ancilla_initial_state(theta_A, phi_A)
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

    # Zero-variance case: eigenstate of the measurement operator
    if var_val < 1e-15:
        return float("inf"), exp_val, var_val, d_exp, True

    delta_theta = float(np.sqrt(var_val) / abs(d_exp))
    return delta_theta, exp_val, var_val, d_exp, False


# ============================================================================
# Marsaglia 3-Ball Sampling (copy from 20260527 for independence)
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
        Tuple ``(drive_samples, azz_samples)`` where drive_samples has shape
        ``(n_samp, 3)`` with columns ``[a_x, a_y, a_z]``, and azz_samples has
        shape ``(n_samp,)``.
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


# ============================================================================
# Scenario Sampling Helpers
# ============================================================================


def _sample_scenario_A(
    rng: np.random.Generator,
) -> tuple[float, float, float, float, float, float]:
    """Sample a single random configuration for Scenario A.

    Scenario A: Fixed ancilla :math:`(|1,0\rangle, \theta_A=\\phi_A=0)`,
    free drive :math:`(a_x, a_y, a_z)` from 3-ball, free :math:`a_{zz}`.

    Returns:
        Tuple ``(theta_A, phi_A, a_x, a_y, a_z, a_zz)``.
    """
    drive, azz = _marsaglia_3ball_sample(rng, 1, R_MAX, AZZ_BOUNDS[0], AZZ_BOUNDS[1])
    return (
        0.0,
        0.0,
        float(drive[0, 0]),
        float(drive[0, 1]),
        float(drive[0, 2]),
        float(azz[0]),
    )


def _sample_scenario_B(
    rng: np.random.Generator,
) -> tuple[float, float, float, float, float, float]:
    """Sample a single random configuration for Scenario B.

    Scenario B: Free ancilla :math:`(\theta_A, \\phi_A)`, free drive from 3-ball,
    free :math:`a_{zz}`.

    Returns:
        Tuple ``(theta_A, phi_A, a_x, a_y, a_z, a_zz)``.
    """
    theta_A = rng.uniform(0.0, np.pi)
    phi_A = rng.uniform(0.0, 2.0 * np.pi)
    drive, azz = _marsaglia_3ball_sample(rng, 1, R_MAX, AZZ_BOUNDS[0], AZZ_BOUNDS[1])
    return (
        theta_A,
        phi_A,
        float(drive[0, 0]),
        float(drive[0, 1]),
        float(drive[0, 2]),
        float(azz[0]),
    )


def _sample_scenario_C(
    rng: np.random.Generator,
) -> tuple[float, float, float, float, float, float]:
    """Sample a single random configuration for Scenario C.

    Scenario C: Free ancilla :math:`(\theta_A, \\phi_A)`, no drive
    :math:`(a_x=a_y=a_z=0)`, free :math:`a_{zz}`.

    Returns:
        Tuple ``(theta_A, phi_A, 0.0, 0.0, 0.0, a_zz)``.
    """
    theta_A = rng.uniform(0.0, np.pi)
    phi_A = rng.uniform(0.0, 2.0 * np.pi)
    a_zz = rng.uniform(AZZ_BOUNDS[0], AZZ_BOUNDS[1])
    return (theta_A, phi_A, 0.0, 0.0, 0.0, a_zz)


def _sample_scenario_D(
    rng: np.random.Generator,
) -> tuple[float, float, float, float, float, float]:
    """Sample a single random configuration for Scenario D.

    Scenario D: Free ancilla :math:`(\theta_A, \\phi_A)`, free drive from 3-ball,
    no interaction :math:`(a_{zz}=0)`.

    Returns:
        Tuple ``(theta_A, phi_A, a_x, a_y, a_z, 0.0)``.
    """
    theta_A = rng.uniform(0.0, np.pi)
    phi_A = rng.uniform(0.0, 2.0 * np.pi)
    drive, _ = _marsaglia_3ball_sample(rng, 1, R_MAX, 0.0, 0.0)
    return (
        theta_A,
        phi_A,
        float(drive[0, 0]),
        float(drive[0, 1]),
        float(drive[0, 2]),
        0.0,
    )


SCENARIO_SAMPLERS: dict[
    str,
    Callable[[np.random.Generator], tuple[float, float, float, float, float, float]],
] = {
    "A": _sample_scenario_A,
    "B": _sample_scenario_B,
    "C": _sample_scenario_C,
    "D": _sample_scenario_D,
}

SCENARIO_FREE_PARAMS: dict[str, list[str]] = {
    "A": ["a_x", "a_y", "a_z", "a_zz"],
    "B": ["theta_A", "phi_A", "a_x", "a_y", "a_z", "a_zz"],
    "C": ["theta_A", "phi_A", "a_zz"],
    "D": ["theta_A", "phi_A", "a_x", "a_y", "a_z"],
}


# ============================================================================
# FreeAncillaSearchResult Dataclass
# ============================================================================


@dataclass
class FreeAncillaSearchResult:
    """Result from a batch of random-search evaluations for any scenario.

    Attributes:
        samples: Parameter matrix, shape ``(N, 6)`` with columns
            ``[theta_A, phi_A, a_x, a_y, a_z, a_zz]``.
        delta_theta_values: Sensitivity for each sample, shape ``(N,)``.
        expectation_values: :math:`\\langle J_z^S\rangle` for each sample.
        variance_values: :math:`\text{Var}(J_z^S)` for each sample.
        deriv_values: :math:`\\partial\\langle J_z^S\rangle/\\partial\theta`.
        is_fringe: Boolean flag per sample, shape ``(N,)``.
        best_params: The 6-tuple giving the minimum :math:`\\Delta\theta`.
        best_delta_theta: The minimum :math:`\\Delta\theta` found.
        theta_value: :math:`\theta` at which the search was performed.
        sql: SQL reference value.
        T_H: Holding-time strength.
        R: Norm-ball radius constraint.
        scenario: Scenario label (``'A'``, ``'B'``, ``'C'``, or ``'D'``).
    """

    samples: np.ndarray
    delta_theta_values: np.ndarray
    expectation_values: np.ndarray
    variance_values: np.ndarray
    deriv_values: np.ndarray
    is_fringe: np.ndarray
    best_params: tuple[float, float, float, float, float, float]
    best_delta_theta: float
    theta_value: float = 1.0
    sql: float = SQL
    T_H: float = T_H
    R: float = R_MAX
    scenario: str = "B"

    def __post_init__(self) -> None:
        assert self.scenario in ("A", "B", "C", "D"), (
            f"Unknown scenario: {self.scenario}"
        )

    def to_dataframe(self) -> pd.DataFrame:
        n = len(self.samples)
        norms_a = np.sqrt(
            self.samples[:, 2] ** 2 + self.samples[:, 3] ** 2 + self.samples[:, 4] ** 2
        )
        ratios = np.where(
            np.isfinite(self.delta_theta_values) & (self.sql > 0),
            self.delta_theta_values / self.sql,
            float("inf"),
        )
        return pd.DataFrame(
            {
                "theta_value": [self.theta_value] * n,
                "T_H": [self.T_H] * n,
                "sql": [self.sql] * n,
                "scenario": [self.scenario] * n,
                "R": [self.R] * n,
                "theta_A": self.samples[:, 0],
                "phi_A": self.samples[:, 1],
                "a_x": self.samples[:, 2],
                "a_y": self.samples[:, 3],
                "a_z": self.samples[:, 4],
                "a_zz": self.samples[:, 5],
                "norm_a": norms_a,
                "delta_theta": self.delta_theta_values,
                "expectation": self.expectation_values,
                "variance": self.variance_values,
                "derivative": self.deriv_values,
                "is_fringe": self.is_fringe.astype(int),
                "ratio": ratios,
            },
        )

    def save_parquet(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_parquet(path, index=False)
        return path

    @classmethod
    def from_parquet(cls, path: str | Path) -> FreeAncillaSearchResult:
        df = pd.read_parquet(path)
        required = {
            "theta_value",
            "T_H",
            "sql",
            "scenario",
            "R",
            "theta_A",
            "phi_A",
            "a_x",
            "a_y",
            "a_z",
            "a_zz",
            "delta_theta",
            "expectation",
            "variance",
            "derivative",
            "is_fringe",
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Parquet at {path} is missing required columns: "
                f"{sorted(missing)}. Regenerate the file with the current code."
            )
        samples = df[["theta_A", "phi_A", "a_x", "a_y", "a_z", "a_zz"]].to_numpy(
            dtype=float
        )
        deltas = df["delta_theta"].to_numpy(dtype=float)
        exps = df["expectation"].to_numpy(dtype=float)
        vars_ = df["variance"].to_numpy(dtype=float)
        derivs = df["derivative"].to_numpy(dtype=float)

        for name, arr in [("expectation", exps), ("variance", vars_), ("derivative", derivs)]:
            if np.any(np.isnan(arr)):
                raise ValueError(
                    f"Parquet at {path} contains NaN in '{name}' column"
                )
        fringes = df["is_fringe"].to_numpy(dtype=bool)
        best_idx = int(np.nanargmin(deltas))
        return cls(
            samples=samples,
            delta_theta_values=deltas,
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
            best_delta_theta=float(deltas[best_idx]),
            theta_value=float(df["theta_value"].iloc[0]),
            sql=float(df["sql"].iloc[0]),
            T_H=float(df["T_H"].iloc[0]),
            R=float(df["R"].iloc[0]),
            scenario=str(df["scenario"].iloc[0]),
        )


# ============================================================================
# FreeAncillaNelderMeadResult Dataclass
# ============================================================================


@dataclass
class FreeAncillaNelderMeadResult:
    """Result of a single Nelder--Mead run for the free-ancilla protocol.

    Attributes:
        delta_theta_opt: Best sensitivity :math:`\\Delta\theta` found.
        params_opt: Optimal parameter vector (length varies by scenario:
            4 for A, 6 for B, 3 for C, 5 for D).
        full_params_opt: Full 6-element parameter vector
            ``(theta_A, phi_A, a_x, a_y, a_z, a_zz)``.
        theta_true: True :math:`\theta` used for this optimisation.
        scenario: Scenario label.
        success: Whether the optimiser reported success.
        nfev: Number of function evaluations.
        message: Optimiser message.
        expectation_Jz: :math:`\\langle J_z^S\rangle` at the optimal point.
        variance_Jz: :math:`\text{Var}(J_z^S)` at the optimal point.
        history: Objective function values at each iteration.
    """

    delta_theta_opt: float
    params_opt: np.ndarray
    full_params_opt: tuple[float, float, float, float, float, float]
    theta_true: float
    scenario: str
    success: bool
    nfev: int
    message: str = ""
    expectation_Jz: float = 0.0
    variance_Jz: float = 0.0
    history: list[float] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "delta_theta": [self.delta_theta_opt],
                "theta_true": [self.theta_true],
                "scenario": [self.scenario],
                "success": [int(self.success)],
                "nfev": [self.nfev],
                "message": [self.message],
                "expectation_Jz": [self.expectation_Jz],
                "variance_Jz": [self.variance_Jz],
                "theta_A": [self.full_params_opt[0]],
                "phi_A": [self.full_params_opt[1]],
                "a_x": [self.full_params_opt[2]],
                "a_y": [self.full_params_opt[3]],
                "a_z": [self.full_params_opt[4]],
                "a_zz": [self.full_params_opt[5]],
                "history_json": [json.dumps(self.history)],
            },
        )

    def save_parquet(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_parquet(path, index=False)
        # Sidecar file for history
        history_path = path.with_stem(path.stem + "-history")
        pd.DataFrame({"history": [json.dumps(self.history)]}).to_parquet(
            history_path, index=False
        )
        return path

    @classmethod
    def from_parquet(cls, path: str | Path) -> FreeAncillaNelderMeadResult:
        path = Path(path)
        df = pd.read_parquet(path)
        required = {
            "delta_theta",
            "theta_true",
            "scenario",
            "success",
            "nfev",
            "message",
            "expectation_Jz",
            "variance_Jz",
            "theta_A",
            "phi_A",
            "a_x",
            "a_y",
            "a_z",
            "a_zz",
            "history_json",
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Parquet at {path} is missing required columns: "
                f"{sorted(missing)}. Regenerate the file with the current code."
            )
        history_path = path.with_stem(path.stem + "-history")
        if history_path.exists():
            history_df = pd.read_parquet(history_path)
            history = json.loads(history_df["history"].iloc[0])
        else:
            history = []
        return cls(
            delta_theta_opt=float(df["delta_theta"].iloc[0]),
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
            full_params_opt=(
                float(df["theta_A"].iloc[0]),
                float(df["phi_A"].iloc[0]),
                float(df["a_x"].iloc[0]),
                float(df["a_y"].iloc[0]),
                float(df["a_z"].iloc[0]),
                float(df["a_zz"].iloc[0]),
            ),
            theta_true=float(df["theta_true"].iloc[0]),
            scenario=str(df["scenario"].iloc[0]),
            success=bool(int(df["success"].iloc[0])),
            nfev=int(df["nfev"].iloc[0]),
            message=str(df["message"].iloc[0]),
            expectation_Jz=float(df["expectation_Jz"].iloc[0]),
            variance_Jz=float(df["variance_Jz"].iloc[0]),
            history=history,
        )


# ============================================================================
# FreeAncillaThetaScanResult Dataclass
# ============================================================================


@dataclass
class FreeAncillaThetaScanResult:
    """Results of a :math:`\theta` scan for a free-ancilla scenario.

    Attributes:
        theta_values: Array of :math:`\theta` values scanned.
        best_params_per_theta: List of optimal full 6-param tuples.
        best_delta_theta_per_theta: Optimal :math:`\\Delta\theta` for each
            :math:`\theta`.
        sql_values: SQL = 1/T_H for each :math:`\theta`.
        expectation_Jz_per_theta: :math:`\\langle J_z^S\rangle` at each optimum.
        variance_Jz_per_theta: :math:`\text{Var}(J_z^S)` at each optimum.
        scenario: Scenario label.
    """

    theta_values: np.ndarray = field(default_factory=lambda: np.array([]))
    best_params_per_theta: list[tuple[float, float, float, float, float, float]] = (
        field(
            default_factory=list,
        )
    )
    best_delta_theta_per_theta: np.ndarray = field(default_factory=lambda: np.array([]))
    sql_values: np.ndarray = field(default_factory=lambda: np.array([]))
    expectation_Jz_per_theta: np.ndarray = field(default_factory=lambda: np.array([]))
    variance_Jz_per_theta: np.ndarray = field(default_factory=lambda: np.array([]))
    scenario: str = "B"

    def to_dataframe(self) -> pd.DataFrame:
        rows: list[dict[str, float | str]] = []
        for i, theta in enumerate(self.theta_values):
            sql = float(self.sql_values[i]) if i < len(self.sql_values) else SQL
            best = (
                self.best_delta_theta_per_theta[i]
                if i < len(self.best_delta_theta_per_theta)
                else float("inf")
            )
            params = (
                self.best_params_per_theta[i]
                if i < len(self.best_params_per_theta)
                else (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            )
            exp_jz = (
                float(self.expectation_Jz_per_theta[i])
                if i < len(self.expectation_Jz_per_theta)
                else 0.0
            )
            var_jz = (
                float(self.variance_Jz_per_theta[i])
                if i < len(self.variance_Jz_per_theta)
                else 0.0
            )
            rows.append(
                {
                    "theta": float(theta),
                    "best_delta_theta": best,
                    "sql": sql,
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
                    "scenario": self.scenario,
                },
            )
        return pd.DataFrame(rows)

    def save_parquet(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_parquet(path, index=False)
        return path

    @classmethod
    def from_parquet(cls, path: str | Path) -> FreeAncillaThetaScanResult:
        df = pd.read_parquet(path)
        required = {
            "theta",
            "best_delta_theta",
            "sql",
            "theta_A",
            "phi_A",
            "a_x",
            "a_y",
            "a_z",
            "a_zz",
            "expectation_Jz",
            "variance_Jz",
            "scenario",
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Parquet at {path} is missing required columns: "
                f"{sorted(missing)}. Regenerate the file with the current code."
            )
        thetas = df["theta"].to_numpy(dtype=float)
        best = df["best_delta_theta"].to_numpy(dtype=float)
        sql = df["sql"].to_numpy(dtype=float)
        exps = df["expectation_Jz"].to_numpy(dtype=float)
        vars_ = df["variance_Jz"].to_numpy(dtype=float)
        scenario = str(df["scenario"].iloc[0])
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
            theta_values=thetas,
            best_params_per_theta=params_list,
            best_delta_theta_per_theta=best,
            sql_values=sql,
            expectation_Jz_per_theta=exps,
            variance_Jz_per_theta=vars_,
            scenario=scenario,
        )


# ============================================================================
# FreeAncilla2DSliceResult Dataclass
# ============================================================================


@dataclass
class FreeAncilla2DSliceResult:
    """Result from a 2D parameter slice over :math:`(\theta_A, a_{zz})`.

    Attributes:
        theta_A_values: Array of :math:`\theta_A` values.
        azz_values: Array of :math:`a_{zz}` values.
        delta_theta_grid: 2D array of :math:`\\Delta\theta`, shape
            ``(len(theta_A_values), len(azz_values))``.
        theta_value: The :math:`\theta` value.
        sql: SQL = 1/T_H reference.
    """

    theta_A_values: np.ndarray
    azz_values: np.ndarray
    delta_theta_grid: np.ndarray
    theta_value: float = 1.0
    sql: float = SQL

    def to_dataframe(self) -> pd.DataFrame:
        n_t = len(self.theta_A_values)
        n_a = len(self.azz_values)
        rows: list[dict[str, float | str]] = [
            {
                "theta_A": float(self.theta_A_values[i]),
                "azz": float(self.azz_values[j]),
                "delta_theta": float(self.delta_theta_grid[i, j]),
                "theta_value": float(self.theta_value),
                "sql": float(self.sql),
            }
            for i in range(n_t)
            for j in range(n_a)
        ]
        return pd.DataFrame(rows)

    def save_parquet(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_parquet(path, index=False)
        return path

    @classmethod
    def from_parquet(cls, path: str | Path) -> FreeAncilla2DSliceResult:
        df = pd.read_parquet(path)
        required = {"theta_A", "azz", "delta_theta", "theta_value", "sql"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Parquet at {path} is missing required columns: "
                f"{sorted(missing)}. Regenerate the file with the current code."
            )
        theta_A_unique = sorted(df["theta_A"].unique())
        azz_unique = sorted(df["azz"].unique())
        n_t = len(theta_A_unique)
        n_a = len(azz_unique)
        grid = np.full((n_t, n_a), np.nan, dtype=float)
        for _, row in df.iterrows():
            i = theta_A_unique.index(row["theta_A"])
            j = azz_unique.index(row["azz"])
            grid[i, j] = row["delta_theta"]
        return cls(
            theta_A_values=np.array(theta_A_unique, dtype=float),
            azz_values=np.array(azz_unique, dtype=float),
            delta_theta_grid=grid,
            theta_value=float(df["theta_value"].iloc[0]),
            sql=float(df["sql"].iloc[0]),
        )


# ============================================================================
# Free-Ancilla Random Search
# ============================================================================


def free_ancilla_random_search(
    theta: float,
    scenario: str,
    n_samples: int = N_SAMP_BCD,
    *,
    R: float = R_MAX,
    azz_bounds: tuple[float, float] = AZZ_BOUNDS,
    T_H: float = T_H,
    T_BS: float = T_BS,
    fd_step: float = FD_STEP,
    seed: int | None = 42,
) -> FreeAncillaSearchResult:
    """Random search over the free-ancilla parameter space.

    Args:
        theta: Phase rate value.
        scenario: ``'A'``, ``'B'``, ``'C'``, or ``'D'``.
        n_samples: Number of random points to evaluate.
        R: Norm-ball radius for (a_x, a_y, a_z).
        azz_bounds: (min, max) for a_zz.
        T_H: Holding time.
        T_BS: Beam-splitter duration.
        fd_step: Finite-difference step size.
        seed: Random seed for reproducibility.

    Returns:
        FreeAncillaSearchResult with all samples and best found.
    """
    rng = np.random.default_rng(seed)
    sampler = SCENARIO_SAMPLERS[scenario]

    samples = np.zeros((n_samples, 6), dtype=float)
    deltas = np.full(n_samples, np.inf, dtype=float)
    exps = np.zeros(n_samples, dtype=float)
    vars_ = np.zeros(n_samples, dtype=float)
    derivs = np.zeros(n_samples, dtype=float)
    fringes = np.zeros(n_samples, dtype=bool)

    for i in range(n_samples):
        theta_A, phi_A, a_x, a_y, a_z, a_zz = sampler(rng)

        # Clamp to bounds for the 3-ball
        norm_drive = np.sqrt(a_x**2 + a_y**2 + a_z**2)
        if norm_drive > R:
            scale = R / max(norm_drive, 1e-300)
            a_x *= scale
            a_y *= scale
            a_z *= scale

        samples[i, :] = [theta_A, phi_A, a_x, a_y, a_z, a_zz]

        dtheta, exp_val, var_val, deriv, fringe = compute_free_ancilla_sensitivity(
            theta,
            theta_A,
            phi_A,
            a_x,
            a_y,
            a_z,
            a_zz,
            T_H=T_H,
            T_BS=T_BS,
            fd_step=fd_step,
        )
        deltas[i] = dtheta
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

    return FreeAncillaSearchResult(
        samples=samples,
        delta_theta_values=deltas,
        expectation_values=exps,
        variance_values=vars_,
        deriv_values=derivs,
        is_fringe=fringes,
        best_params=best_params,
        best_delta_theta=float(deltas[best_idx]),
        theta_value=theta,
        sql=1.0 / T_H,
        T_H=T_H,
        R=R,
        scenario=scenario,
    )


# ============================================================================
# Free-Ancilla Nelder--Mead Optimisation
# ============================================================================


def _params_to_full(
    params: np.ndarray, scenario: str
) -> tuple[float, float, float, float, float, float]:
    """Map scenario-specific optimisation parameters to the full 6-param tuple.

    Args:
        params: Parameter vector (length depends on scenario).
        scenario: ``'A'``, ``'B'``, ``'C'``, or ``'D'``.

    Returns:
        Tuple ``(theta_A, phi_A, a_x, a_y, a_z, a_zz)``.
    """
    if scenario == "A":
        # [a_x, a_y, a_z, a_zz]
        return (
            0.0,
            0.0,
            float(params[0]),
            float(params[1]),
            float(params[2]),
            float(params[3]),
        )
    if scenario == "B":
        # [theta_A, phi_A, a_x, a_y, a_z, a_zz]
        return (
            float(params[0]),
            float(params[1]),
            float(params[2]),
            float(params[3]),
            float(params[4]),
            float(params[5]),
        )
    if scenario == "C":
        # [theta_A, phi_A, a_zz]
        return (float(params[0]), float(params[1]), 0.0, 0.0, 0.0, float(params[2]))
    if scenario == "D":
        # [theta_A, phi_A, a_x, a_y, a_z]
        return (
            float(params[0]),
            float(params[1]),
            float(params[2]),
            float(params[3]),
            float(params[4]),
            0.0,
        )
    raise ValueError(f"Unknown scenario: {scenario}")


def _free_ancilla_objective(
    params: np.ndarray,
    theta_true: float,
    scenario: str,
    ops: dict[str, np.ndarray],
    T_H: float = T_H,
    T_BS: float = T_BS,
    fd_step: float = FD_STEP,
    bounds: tuple[float, float] = (-5.0, 5.0),
    penalty_scale: float = 1e6,
) -> float:
    """Objective function for minimising :math:`\\Delta\theta`.

    Args:
        params: Scenario-specific parameter vector.
        theta_true: True phase rate.
        scenario: Scenario label.
        ops: Two-qubit operators.
        T_H: Holding time.
        T_BS: Beam-splitter duration.
        fd_step: Finite-difference step.
        bounds: (min, max) for any scalar parameter.
        penalty_scale: Scale for bound-violation penalty.

    Returns:
        :math:`\\Delta\theta` (plus infinite penalty if bounds violated).
    """
    theta_A, phi_A, a_x, a_y, a_z, a_zz = _params_to_full(params, scenario)

    # Bound enforcement on all scalar params
    lo, hi = bounds
    # θ_A ∈ [0, π], φ_A ∈ [0, 2π)
    penalty = 0.0
    for val in params:
        if val < lo:
            penalty += penalty_scale * (lo - val) ** 2
        if val > hi:
            penalty += penalty_scale * (val - hi) ** 2

    # Check angular bounds separately
    if theta_A < 0.0:
        penalty += penalty_scale * theta_A**2
    if theta_A > np.pi:
        penalty += penalty_scale * (theta_A - np.pi) ** 2
    if phi_A < 0.0:
        penalty += penalty_scale * phi_A**2
    if phi_A > 2.0 * np.pi:
        penalty += penalty_scale * (phi_A - 2.0 * np.pi) ** 2

    if penalty > 0.0:
        return float(1e10 + penalty)

    dtheta, _, _, _, _ = compute_free_ancilla_sensitivity(
        theta_true,
        theta_A,
        phi_A,
        a_x,
        a_y,
        a_z,
        a_zz,
        T_H=T_H,
        T_BS=T_BS,
        fd_step=fd_step,
    )
    return dtheta


def run_free_ancilla_nelder_mead(
    theta_true: float,
    scenario: str,
    x0: np.ndarray | None = None,
    seed: int | None = None,
    maxiter: int = 5000,
    xatol: float = 1e-8,
    fatol: float = 1e-8,
    adaptive: bool = True,
    bounds: tuple[float, float] = (-5.0, 5.0),
    T_H: float = T_H,
    T_BS: float = T_BS,
    track_history: bool = False,
) -> FreeAncillaNelderMeadResult:
    """Run Nelder--Mead optimisation for the free-ancilla protocol.

    Args:
        theta_true: True phase rate parameter.
        scenario: Scenario label.
        x0: Initial parameter vector. Randomly generated if None.
        seed: Random seed (used if x0 is None).
        maxiter: Maximum Nelder--Mead iterations.
        xatol: Absolute parameter tolerance.
        fatol: Absolute function tolerance.
        adaptive: Use adaptive Nelder--Mead parameters.
        bounds: (min, max) for all scalar parameters.
        T_H: Holding time.
        T_BS: Beam-splitter duration.
        track_history: If True, record objective values per iteration.

    Returns:
        FreeAncillaNelderMeadResult.
    """
    ops = build_two_qubit_operators()

    n_params = len(SCENARIO_FREE_PARAMS[scenario])

    if x0 is None:
        rng = np.random.default_rng(seed)
        lo, hi = bounds
        x0 = rng.uniform(lo, hi, size=n_params)
        # For Scenario A, x0 is [ax, ay, az, azz], all in bounds — fine.
        # For B/C/D, theta_A must be in [0, π], phi_A in [0, 2π).
        # We clamp them into the objective but the NM may wander.
        # The penalty handles it.
    else:
        x0 = np.asarray(x0, dtype=float)
        assert x0.shape == (n_params,), (
            f"x0 must have {n_params} elements for scenario {scenario}, got {x0.shape}"
        )

    def objective(p: np.ndarray) -> float:
        return _free_ancilla_objective(
            p,
            theta_true,
            scenario,
            ops,
            T_H=T_H,
            T_BS=T_BS,
            bounds=bounds,
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
    full_params = _params_to_full(opt_params, scenario)

    # Diagnostics at the optimal point
    theta_A, phi_A, a_x, a_y, a_z, a_zz = full_params
    psi_final = evolve_drive_circuit(
        free_ancilla_initial_state(theta_A, phi_A),
        T_BS,
        T_H,
        theta_true,
        a_x,
        a_y,
        a_z,
        a_zz,
        ops,
    )
    exp_val, var_val = compute_expectation_and_variance(psi_final, ops["Jz_S"])

    return FreeAncillaNelderMeadResult(
        delta_theta_opt=float(result.fun),
        params_opt=opt_params,
        full_params_opt=full_params,
        theta_true=theta_true,
        scenario=scenario,
        success=bool(result.success),
        nfev=int(result.nfev),
        message=str(result.message),
        expectation_Jz=exp_val,
        variance_Jz=var_val,
        history=history.copy(),
    )


# ============================================================================
# Theta Scan with Random Search + Nelder--Mead Refinement
# ============================================================================


def run_free_ancilla_theta_scan(
    theta_values: list[float] | np.ndarray,
    scenario: str,
    n_random: int = N_SAMP_BCD,
    n_nm_refine: int = N_NM_REFINE,
    seed: int | None = 42,
    maxiter: int = 5000,
    bounds: tuple[float, float] = (-5.0, 5.0),
    R: float = R_MAX,
    azz_bounds: tuple[float, float] = AZZ_BOUNDS,
    T_H: float = T_H,
    T_BS: float = T_BS,
) -> FreeAncillaThetaScanResult:
    """Scan over :math:`\theta` values with random search + NM refinement.

    For each :math:`\theta`:
    1. Run ``n_random`` random evaluations in the scenario-specific parameter space.
    2. Select the best ``n_nm_refine`` points.
    3. Run Nelder--Mead refinement from each selected point.
    4. Record the best overall result.

    Args:
        theta_values: :math:`\theta` values to scan.
        scenario: Scenario label.
        n_random: Number of random search points per :math:`\theta`.
        n_nm_refine: Number of NM refinements per :math:`\theta`.
        seed: Base random seed (incremented per :math:`\theta`).
        maxiter: Maximum NM iterations.
        bounds: (min, max) for all scalar parameters.
        R: Norm-ball radius.
        azz_bounds: (min, max) for a_zz.
        T_H: Holding time.
        T_BS: Beam-splitter duration.

    Returns:
        FreeAncillaThetaScanResult.
    """
    theta_arr = np.asarray(theta_values, dtype=float)
    base_seed = seed if seed is not None else 42

    best_params_list: list[tuple[float, float, float, float, float, float]] = []
    best_deltas: list[float] = []
    sql_vals: list[float] = []
    exp_vals: list[float] = []
    var_vals: list[float] = []

    for theta in theta_arr:
        # Stage 1: Random search
        rs_result = free_ancilla_random_search(
            theta,
            scenario,
            n_samples=n_random,
            R=R,
            azz_bounds=azz_bounds,
            T_H=T_H,
            T_BS=T_BS,
            seed=base_seed + int(theta * 1000),
        )

        # Sort random-search results by Δθ, take top n_nm_refine
        sorted_indices = np.argsort(rs_result.delta_theta_values)
        top_indices = sorted_indices[:n_nm_refine]

        # Stage 2: Nelder--Mead refinement from each top point
        nm_results: list[FreeAncillaNelderMeadResult] = []
        for rank, idx in enumerate(top_indices):
            # Build x0 from the full 6-param sample, keeping only the free params
            full = rs_result.samples[idx]
            if scenario == "A":
                x0 = full[2:6].copy()  # [a_x, a_y, a_z, a_zz]
            elif scenario == "B":
                x0 = full.copy()  # [theta_A, phi_A, a_x, a_y, a_z, a_zz]
            elif scenario == "C":
                x0 = np.array([full[0], full[1], full[5]])  # [theta_A, phi_A, a_zz]
            elif scenario == "D":
                x0 = np.array(
                    [full[0], full[1], full[2], full[3], full[4]]
                )  # [theta_A, phi_A, a_x, a_y, a_z]
            else:
                raise ValueError(f"Unknown scenario: {scenario}")

            nm = run_free_ancilla_nelder_mead(
                theta_true=theta,
                scenario=scenario,
                x0=x0,
                seed=base_seed + int(theta * 1000) + 10000 + rank,
                maxiter=maxiter,
                bounds=bounds,
                T_H=T_H,
                T_BS=T_BS,
                track_history=False,
            )
            nm_results.append(nm)

        # Sort NM results by Δθ
        nm_results.sort(key=lambda r: r.delta_theta_opt)
        best_nm = nm_results[0]

        best_params_list.append(best_nm.full_params_opt)
        best_deltas.append(best_nm.delta_theta_opt)
        sql_vals.append(1.0 / T_H)
        exp_vals.append(best_nm.expectation_Jz)
        var_vals.append(best_nm.variance_Jz)

    return FreeAncillaThetaScanResult(
        theta_values=theta_arr,
        best_params_per_theta=best_params_list,
        best_delta_theta_per_theta=np.array(best_deltas, dtype=float),
        sql_values=np.array(sql_vals, dtype=float),
        expectation_Jz_per_theta=np.array(exp_vals, dtype=float),
        variance_Jz_per_theta=np.array(var_vals, dtype=float),
        scenario=scenario,
    )


# ============================================================================
# 2D Slice: (theta_A, a_zz)
# ============================================================================


def _free_ancilla_slice_worker(args: tuple) -> tuple[int, np.ndarray]:
    """Worker for parallel (theta_A, a_zz) slice evaluation.

    Args:
        args: Tuple ``(theta, theta_A_chunk, azz_vals, T_H, T_BS, fd_step, start_idx)``.

    Returns:
        Tuple ``(start_idx, chunk_grid)`` where chunk_grid has shape
        ``(len(theta_A_chunk), len(azz_vals))``.
    """
    theta, theta_A_chunk, azz_vals, T_H, T_BS, fd_step, start_idx = args
    n_t = len(theta_A_chunk)
    n_a = len(azz_vals)
    chunk_grid = np.full((n_t, n_a), np.inf, dtype=float)
    phi_A = 0.0  # Fixed azimuth for the slice
    for i, tA in enumerate(theta_A_chunk):
        for j, a_val in enumerate(azz_vals):
            dtheta, _, _, _, _ = compute_free_ancilla_sensitivity(
                theta,
                tA,
                phi_A,
                0.0,
                0.0,
                0.0,
                a_val,
                T_H=T_H,
                T_BS=T_BS,
                fd_step=fd_step,
            )
            chunk_grid[i, j] = dtheta
    return start_idx, chunk_grid


def free_ancilla_2d_slice(
    theta: float,
    theta_A_range: tuple[float, float] = THETA_A_RANGE,
    azz_range: tuple[float, float] = AZZ_BOUNDS,
    n_grid: int = SLICE_N,
    T_H: float = T_H,
    T_BS: float = T_BS,
    fd_step: float = FD_STEP,
    n_jobs: int | None = None,
) -> FreeAncilla2DSliceResult:
    """Run a 2D slice scan over :math:`(\theta_A, a_{zz})`.

    Fixed parameters: :math:`a_x = a_y = a_z = 0`, :math:`\\phi_A = 0`.

    When ``n_jobs > 1``, the grid is split across worker processes.

    Args:
        theta: Phase rate value.
        theta_A_range: (min, max) for :math:`\theta_A`.
        azz_range: (min, max) for :math:`a_{zz}`.
        n_grid: Number of points per axis (total grid = n_grid × n_grid).
        T_H: Holding time.
        T_BS: Beam-splitter duration.
        fd_step: Finite-difference step size.
        n_jobs: Number of parallel workers. ``None`` (default) = sequential.

    Returns:
        FreeAncilla2DSliceResult.
    """
    theta_A_vals = np.linspace(theta_A_range[0], theta_A_range[1], n_grid)
    azz_vals = np.linspace(azz_range[0], azz_range[1], n_grid)

    if n_jobs is None or n_jobs == 1:
        # Sequential path
        grid = np.full((n_grid, n_grid), np.inf, dtype=float)
        phi_A = 0.0
        for i in range(n_grid):
            for j in range(n_grid):
                dtheta, _, _, _, _ = compute_free_ancilla_sensitivity(
                    theta,
                    theta_A_vals[i],
                    phi_A,
                    0.0,
                    0.0,
                    0.0,
                    azz_vals[j],
                    T_H=T_H,
                    T_BS=T_BS,
                    fd_step=fd_step,
                )
                grid[i, j] = dtheta
    else:
        # Parallel path
        n_workers = max(1, os.cpu_count() or 4) if n_jobs == -1 else n_jobs
        indices = np.arange(n_grid)
        chunks = np.array_split(indices, n_workers)
        worker_args = [
            (theta, theta_A_vals[chunk], azz_vals, T_H, T_BS, fd_step, int(chunk[0]))
            for chunk in chunks
        ]

        grid = np.full((n_grid, n_grid), np.inf, dtype=float)
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_free_ancilla_slice_worker, args): args
                for args in worker_args
            }
            for future in concurrent.futures.as_completed(futures):
                start_idx, chunk_grid = future.result()
                n_chunk = chunk_grid.shape[0]
                grid[start_idx : start_idx + n_chunk, :] = chunk_grid

    return FreeAncilla2DSliceResult(
        theta_A_values=theta_A_vals,
        azz_values=azz_vals,
        delta_theta_grid=grid,
        theta_value=theta,
        sql=1.0 / T_H,
    )


# ============================================================================
# Cross-Scenario Comparison Helpers
# ============================================================================


def build_cross_scenario_dataframe(
    scan_results: dict[str, FreeAncillaThetaScanResult],
) -> pd.DataFrame:
    """Build a long-format DataFrame comparing all scenarios.

    Args:
        scan_results: Dict mapping scenario label to ThetaScanResult.

    Returns:
        DataFrame with columns: ``theta``, ``scenario``, ``best_delta_theta``,
        ``ratio``, ``sql``, plus optimal parameters.
    """
    rows: list[dict[str, float | str]] = []
    for scenario, res in scan_results.items():
        for i, theta in enumerate(res.theta_values):
            best = (
                res.best_delta_theta_per_theta[i]
                if i < len(res.best_delta_theta_per_theta)
                else float("inf")
            )
            sql = float(res.sql_values[i]) if i < len(res.sql_values) else SQL
            params = (
                res.best_params_per_theta[i]
                if i < len(res.best_params_per_theta)
                else (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            )
            rows.append(
                {
                    "theta": float(theta),
                    "scenario": scenario,
                    "best_delta_theta": best,
                    "sql": sql,
                    "ratio": best / sql
                    if np.isfinite(best) and sql > 0
                    else float("inf"),
                    "theta_A": float(params[0]),
                    "phi_A": float(params[1]),
                    "a_x": float(params[2]),
                    "a_y": float(params[3]),
                    "a_z": float(params[4]),
                    "a_zz": float(params[5]),
                }
            )
    return pd.DataFrame(rows)


# ============================================================================
# Plot Functions
# ============================================================================


def plot_cross_scenario_comparison(
    comparison_df: pd.DataFrame | str | Path,
    save_path: str | Path,
    figsize: tuple[float, float] = (10, 6),
) -> Path:
    """Grouped bar chart comparing best :math:`\\Delta\theta/\text{SQL}`
    across scenarios for each :math:`\theta`.

    Args:
        comparison_df: DataFrame or path to Parquet file.
        save_path: Output SVG path.
        figsize: Figure size.

    Returns:
        Path to saved SVG.
    """
    if isinstance(comparison_df, (str, Path)):
        comparison_df = pd.read_parquet(comparison_df)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)

    thetas = sorted(comparison_df["theta"].unique())
    scenarios = ["A", "B", "C", "D"]
    colours = {"A": "gray", "B": "C0", "C": "C1", "D": "C2"}

    n_groups = len(thetas)
    x = np.arange(n_groups)
    width = 0.2

    for idx, sc in enumerate(scenarios):
        vals: list[float] = []
        for theta in thetas:
            mask = (comparison_df["theta"] == theta) & (comparison_df["scenario"] == sc)
            subset = comparison_df[mask]
            if len(subset) > 0:
                r = float(subset["ratio"].iloc[0])
                vals.append(r if np.isfinite(r) else 0.0)
            else:
                vals.append(0.0)
        ax.bar(
            x + (idx - 1.5) * width,
            vals,
            width,
            color=colours[sc],
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
            label=f"Scenario {sc}",
        )

    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$\min \Delta\theta \;/\; \mathrm{SQL}$")
    ax.set_title("Cross-scenario comparison of best sensitivity ratio")
    ax.set_xticks(x)
    ax.set_xticklabels([rf"$\theta={t:.1f}$" for t in thetas])
    ax.axhline(
        y=1.0, color="red", linestyle="--", linewidth=1.5, alpha=0.7, label="SQL"
    )
    ax.legend(fontsize=9, loc="upper left")

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_scenario_best_ratio_by_theta(
    scan_result: FreeAncillaThetaScanResult | str | Path,
    save_path: str | Path,
    figsize: tuple[float, float] = (8, 5),
) -> Path:
    """Line plot of best :math:`\\Delta\theta/\text{SQL}` vs :math:`\theta`.

    Args:
        scan_result: ThetaScanResult or path to Parquet.
        save_path: Output SVG path.
        figsize: Figure size.

    Returns:
        Path to saved SVG.
    """
    if isinstance(scan_result, (str, Path)):
        scan_result = FreeAncillaThetaScanResult.from_parquet(scan_result)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)

    thetas = scan_result.theta_values
    ratios = scan_result.best_delta_theta_per_theta / scan_result.sql_values

    ax.plot(thetas, ratios, "o-", color="C0", linewidth=2, markersize=6)
    ax.axhline(
        y=1.0, color="red", linestyle="--", linewidth=1.5, alpha=0.7, label="SQL"
    )
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$\min \Delta\theta \;/\; \mathrm{SQL}$")
    ax.set_title(f"Best sensitivity ratio for Scenario {scan_result.scenario}")
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_theta_A_azz_slice_heatmap(
    result: FreeAncilla2DSliceResult | str | Path,
    save_path: str | Path,
    figsize: tuple[float, float] = (9, 7),
) -> Path:
    """Heatmap of :math:`\\Delta\theta/\text{SQL}` over
    :math:`(\theta_A, a_{zz})`.

    Args:
        result: FreeAncilla2DSliceResult or path to Parquet.
        save_path: Output SVG path.
        figsize: Figure size.

    Returns:
        Path to saved SVG.
    """
    if isinstance(result, (str, Path)):
        result = FreeAncilla2DSliceResult.from_parquet(result)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    ratio_grid = result.delta_theta_grid / result.sql
    finite_mask = np.isfinite(ratio_grid)
    vmin = 0.99
    vmax = 2.0

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
    cbar = fig.colorbar(im, ax=ax, label=r"$\Delta\theta / \mathrm{SQL}$")
    cbar.set_ticks([1.0, 1.25, 1.5, 1.75, 2.0])

    ax.set_xlabel(r"$a_{zz}$")
    ax.set_ylabel(r"$\theta_A$")
    ax.set_title(
        rf"($\theta_A$, $a_{{zz}}$) slice at $\theta={result.theta_value:.1f}$, "
        rf"$a_x=a_y=a_z=0$"
    )

    # Mark SQL contour
    if np.any(finite_mask):
        cs = ax.contour(
            result.azz_values,
            result.theta_A_values,
            ratio_grid,
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


def plot_norm_envelope_comparison(
    scenario_A_result: FreeAncillaThetaScanResult | str | Path,
    scenario_B_result: FreeAncillaThetaScanResult | str | Path,
    save_path: str | Path,
    figsize: tuple[float, float] = (8, 6),
) -> Path:
    """Overlay of best-ratio vs :math:`\\|\\mathbf{a}\\|` for Scenarios A and B.

    For each scenario, plots the optimal :math:`\\Delta\theta/\text{SQL}` at
    each :math:`\theta` against the norm of the optimal drive vector.

    Args:
        scenario_A_result: ThetaScanResult for Scenario A.
        scenario_B_result: ThetaScanResult for Scenario B.
        save_path: Output SVG path.
        figsize: Figure size.

    Returns:
        Path to saved SVG.
    """
    if isinstance(scenario_A_result, (str, Path)):
        scenario_A_result = FreeAncillaThetaScanResult.from_parquet(scenario_A_result)
    if isinstance(scenario_B_result, (str, Path)):
        scenario_B_result = FreeAncillaThetaScanResult.from_parquet(scenario_B_result)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)

    for label, res, marker, colour in [
        ("Scenario A (fixed ancilla)", scenario_A_result, "o", "gray"),
        ("Scenario B (free ancilla)", scenario_B_result, "s", "C0"),
    ]:
        norms = np.array(
            [
                np.sqrt(p[2] ** 2 + p[3] ** 2 + p[4] ** 2)
                for p in res.best_params_per_theta
            ]
        )
        ratios = res.best_delta_theta_per_theta / res.sql_values
        finite = np.isfinite(ratios) & np.isfinite(norms)
        ax.scatter(
            norms[finite],
            ratios[finite],
            marker=marker,
            color=colour,
            s=60,
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
            label=label,
            zorder=3,
        )

    ax.axhline(
        y=1.0, color="red", linestyle="--", linewidth=1.5, alpha=0.7, label="SQL"
    )
    ax.set_xlabel(r"Optimal drive norm $\|\mathbf{a}\|$")
    ax.set_ylabel(r"$\min \Delta\theta \;/\; \mathrm{SQL}$")
    ax.set_title("Norm-envelope comparison: fixed vs free ancilla")
    ax.legend(fontsize=9, loc="upper left")

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


# ============================================================================
# Data Generation Functions
# ============================================================================


def _run_scenario_theta_scan(
    scenario: str,
    force: bool,
    n_random: int | None = None,
    n_nm: int | None = None,
    R: float = R_MAX,
) -> FreeAncillaThetaScanResult:
    """Run full theta scan for a single scenario, with caching."""
    tag = f"theta-scan-{scenario.lower()}"
    csv_p = _parquet_path(tag)
    fig_p = _fig_path(tag)

    # Determine sample counts (Scenario A uses fewer samples)
    if n_random is None:
        n_random = N_SAMP_A if scenario == "A" else N_SAMP_BCD
    if n_nm is None:
        n_nm = 50 if scenario == "A" else N_NM_REFINE

    if csv_p.exists() and not force:
        print(f"  [skip] {csv_p.name} exists (use --force to overwrite)")
        result = FreeAncillaThetaScanResult.from_parquet(csv_p)
    else:
        print(
            f"  [run]  Scenario {scenario} theta scan "
            f"({len(DRIVE_THETA_VALS)} θ × {n_random} random + {n_nm} NM)..."
        )
        result = run_free_ancilla_theta_scan(
            theta_values=DRIVE_THETA_VALS,
            scenario=scenario,
            n_random=n_random,
            n_nm_refine=n_nm,
            R=R,
            T_H=T_H,
            T_BS=T_BS,
        )
        result.save_parquet(csv_p)
        print(f"  [save] {csv_p}")

    plot_scenario_best_ratio_by_theta(result, fig_p)
    print(f"  [fig]  {fig_p}")

    return result


def generate_scenario_A(force: bool = False) -> FreeAncillaThetaScanResult:
    """Experiment 1: Scenario A baseline reproduction."""
    print("[run]  Scenario A (fixed ancilla, free drive + interaction)")
    return _run_scenario_theta_scan("A", force=force, n_random=N_SAMP_A, n_nm=50)


def generate_scenario_B(force: bool = False) -> FreeAncillaThetaScanResult:
    """Experiment 2: Scenario B (primary) — free ancilla, free drive,
    free interaction."""
    print("[run]  Scenario B (free ancilla, free drive, free interaction)")
    return _run_scenario_theta_scan("B", force=force)


def generate_scenario_C(force: bool = False) -> FreeAncillaThetaScanResult:
    """Experiment 3: Scenario C (control) — free ancilla, no drive,
    free interaction."""
    print("[run]  Scenario C (free ancilla, no drive, free interaction)")
    return _run_scenario_theta_scan("C", force=force)


def generate_scenario_D(force: bool = False) -> FreeAncillaThetaScanResult:
    """Experiment 4: Scenario D (control) — free ancilla, free drive,
    no interaction."""
    print("[run]  Scenario D (free ancilla, free drive, no interaction)")
    return _run_scenario_theta_scan("D", force=force)


def generate_2d_slice_theta_A_azz(
    force: bool = False,
    n_jobs: int | None = None,
) -> None:
    """Experiment 5: 2D slice over :math:`(\theta_A, a_{zz})` at all θ values."""
    print(f"[run]  (theta_A, a_zz) slice at {DRIVE_THETA_VALS}")
    for theta in DRIVE_THETA_VALS:
        tag = f"slice-thetaA-azz-theta{theta}"
        csv_p = _parquet_path(tag)
        fig_p = _fig_path(tag)

        if csv_p.exists() and not force:
            print(f"  [skip] {csv_p.name} exists (use --force to overwrite)")
            result = FreeAncilla2DSliceResult.from_parquet(csv_p)
        else:
            print(f"  [run]  Computing (theta_A, a_zz) slice at θ={theta}...")
            result = free_ancilla_2d_slice(
                theta=theta,
                theta_A_range=THETA_A_RANGE,
                azz_range=AZZ_BOUNDS,
                n_grid=SLICE_N,
                T_H=T_H,
                T_BS=T_BS,
                n_jobs=n_jobs,
            )
            result.save_parquet(csv_p)
            print(f"  [save] {csv_p}")

        plot_theta_A_azz_slice_heatmap(result, fig_p)
        print(f"  [fig]  {fig_p}")


def generate_cross_scenario_comparison(
    force: bool = False,
) -> FreeAncillaThetaScanResult | None:
    """Experiment 6: Cross-scenario comparison bar chart.

    Loads existing theta-scan results from all four scenarios and produces
    the comparison plot.
    """
    fig_p = _fig_path("cross-scenario-comparison")
    env_fig_p = _fig_path("norm-envelope-comparison")

    print("[run]  Loading theta-scan results for cross-scenario comparison...")
    scan_results: dict[str, FreeAncillaThetaScanResult] = {}
    for sc in ("A", "B", "C", "D"):
        tag = f"theta-scan-{sc.lower()}"
        csv_p = _parquet_path(tag)
        if csv_p.exists():
            scan_results[sc] = FreeAncillaThetaScanResult.from_parquet(csv_p)
            print(f"  [load] {csv_p.name}")
        else:
            print(f"  [warn] {csv_p.name} not found — skipping scenario {sc}")

    if len(scan_results) < 2:
        print("  [warn] Fewer than 2 scenarios loaded — skipping comparison plots")
        return None

    comparison_df = build_cross_scenario_dataframe(scan_results)
    comparison_csv_p = _parquet_path("cross-scenario-data")
    comparison_df.to_parquet(comparison_csv_p, index=False)
    print(f"  [save] {comparison_csv_p}")

    plot_cross_scenario_comparison(comparison_df, fig_p)
    print(f"  [fig]  {fig_p}")

    # Norm-envelope comparison if both A and B are loaded
    if "A" in scan_results and "B" in scan_results:
        plot_norm_envelope_comparison(
            scan_results["A"],
            scan_results["B"],
            env_fig_p,
        )
        print(f"  [fig]  {env_fig_p}")

    return None


# ============================================================================
# CLI Entry Point
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate 2026-05-28 report figures and data",
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
        help="Generate only one dataset, e.g. 'scenario-B'",
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

    tasks: dict[str, Callable[..., object]] = {
        "scenario-A": generate_scenario_A,
        "scenario-B": generate_scenario_B,
        "scenario-C": generate_scenario_C,
        "scenario-D": generate_scenario_D,
        "slice-thetaA-azz": generate_2d_slice_theta_A_azz,
        "cross-scenario": generate_cross_scenario_comparison,
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
