"""
Local module for the 2026-06-19 Ancilla OAT Pre-Squeezing before omega-Modulated Hold report.

Adds one-axis twisting (OAT) pre-squeezing U_OAT = exp(-i q (J_z^A)^2) to the
multi-particle ancilla omega-modulated hold protocol (#20260612). The ancilla
initial state is changed from the top-Dicke state to a coherent spin state
along -x, enabling non-trivial OAT evolution.

Two phases:
- Phase 1: J_S = 1/2 (N_S=1), J_A = 1 (N_A=2) — asymmetric, dimension 6
- Phase 2: J_A = N/2 (N >= 2) — symmetric, dimension (N+1)^2

Circuit: BS_S -> OAT_A -> Hold -> BS_S -> measure J_z^S.

Usage:
    uv run python reports/20260619/ancilla_oat_pre_squeezing.py --force
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.optimize import minimize

import src.physics.bipartite_operators as _bipartite
from src.algorithms.spin_squeezing import coherent_spin_state
from src.analysis.ancilla_optimization import (
    compute_expectation_and_variance,
)
from src.analysis.checkpoint_recovery import load_checkpoints, run_pending_groups
from src.analysis.decoupled_baseline import (
    generate_decoupled_baseline,
)
from src.analysis.optimisation_pipeline import (
    TwoPhaseConfig,
    run_two_phase_pipeline,
)
from src.analysis.sensitivity_metrics import sql_reference
from src.utils.paths import report_path_fn
from src.utils.serialization import ParquetSerializable
from src.visualization.scaling_plots import (
    plot_n_scaling_optimal_params,
    plot_n_scaling_ratio,
    plot_n_scaling_sensitivity,
)

# ============================================================================
# Constants
# ============================================================================

T_HOLD: float = 10.0  # Holding time (fixed)
T_BS: float = np.pi / 2.0  # 50/50 beam splitter
FD_STEP: float = 1e-6  # Finite-difference step
DRIVE_BOUNDS: tuple[float, float] = (-5.0, 5.0)  # Parameter bounds
Q_BOUNDS: tuple[float, float] = (0.0, 5.0)  # OAT squeezing strength bounds

# omega values for the scan
OMEGA_VALS: list[float] = [0.1, 0.2, 0.5, 1.0, 2.0]

# Phase 1: N_S=1 (J_S=1/2), N_A=2 (J_A=1)
PHASE1_N_SYSTEM: int = 1
PHASE1_N_ANCILLA: int = 2

# Phase 2: N values for symmetric case (J_A = N/2, N >= 2)
# N up to 6 is practical (dim up to 49, ~40ms/eval at N=6).
# N≥7 becomes prohibitively slow (500-900ms/eval) with full optimisation budget.
PHASE2_N_VALS: list[int] = list(range(2, 7))

# Random search parameters
N_RANDOM: int = 1000
N_NM_REFINE: int = 10
NM_MAXITER: int = 2000


# ============================================================================
# Path Helpers
# ============================================================================

REPORTS_DIR = Path(__file__).resolve().parent.parent.parent / "reports"
REPORT_DATE = "20260619"


_parquet_path, _fig_path = report_path_fn(REPORTS_DIR, REPORT_DATE)


# ============================================================================
# Result Dataclasses
# ============================================================================


@dataclass
class OATRandomSearchResult(ParquetSerializable):
    """Result from a 5D random search over (a_x, a_y, a_z, a_zz, q).

    Attributes:
        samples: Array of shape (N, 5) with sampled parameter values.
        delta_omega_values: Array of shape (N,) with Δω for each sample.
        best_params: The (a_x, a_y, a_z, a_zz, q) that gave minimal Δω.
        best_delta_omega: The minimal Δω found.
        omega_value: ω at which the search was performed.
        sql: SQL reference value.
        t_hold: Holding time.
    """

    samples: np.ndarray
    delta_omega_values: np.ndarray
    best_params: tuple[float, float, float, float, float]
    best_delta_omega: float
    omega_value: float = 1.0
    sql: float = 0.1
    t_hold: float = 10.0

    _PARQUET_COLUMNS: ClassVar[list[str]] = [
        "a_x",
        "a_y",
        "a_z",
        "a_zz",
        "q",
        "delta_omega",
        "omega_value",
        "sql",
        "t_hold",
    ]

    def to_dataframe(self) -> pd.DataFrame:
        n = len(self.samples)
        return pd.DataFrame(
            {
                "a_x": self.samples[:, 0],
                "a_y": self.samples[:, 1],
                "a_z": self.samples[:, 2],
                "a_zz": self.samples[:, 3],
                "q": self.samples[:, 4],
                "delta_omega": self.delta_omega_values,
                "omega_value": [self.omega_value] * n,
                "sql": [self.sql] * n,
                "t_hold": [self.t_hold] * n,
            },
        )

    @classmethod
    def from_parquet(cls, path: str | Path) -> OATRandomSearchResult:
        df = pd.read_parquet(path)
        cls._validate_columns(df)
        samples = df[["a_x", "a_y", "a_z", "a_zz", "q"]].to_numpy(dtype=float)
        deltas = df["delta_omega"].to_numpy(dtype=float)
        best_idx = int(np.argmin(deltas))
        return cls(
            samples=samples,
            delta_omega_values=deltas,
            best_params=(
                float(samples[best_idx, 0]),
                float(samples[best_idx, 1]),
                float(samples[best_idx, 2]),
                float(samples[best_idx, 3]),
                float(samples[best_idx, 4]),
            ),
            best_delta_omega=float(deltas[best_idx]),
            omega_value=float(df["omega_value"].iloc[0]),
            sql=float(df["sql"].iloc[0]),
            t_hold=float(df["t_hold"].iloc[0]),
        )


@dataclass
class OATNelderMeadResult(ParquetSerializable):
    """Result of a single Nelder-Mead run for the 5D OAT protocol.

    Attributes:
        delta_omega_opt: Best sensitivity Δω found.
        params_opt: Optimal 5-element parameter vector (a_x, a_y, a_z, a_zz, q).
        omega_true: True ω used for this optimisation.
        success: Whether the optimiser reported success.
        nfev: Number of function evaluations.
        message: Optimiser message.
        expectation_Jz: ⟨J_z^S⟩ at the optimal operating point.
        variance_Jz: Var(J_z^S) at the optimal operating point.
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
    history: list[float] = field(default_factory=list)

    _PARQUET_COLUMNS: ClassVar[list[str]] = [
        "delta_omega",
        "a_x",
        "a_y",
        "a_z",
        "a_zz",
        "q",
        "omega_true",
        "success",
        "nfev",
        "expectation_Jz",
        "variance_Jz",
        "message",
        "history_json",
    ]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "a_x": [float(self.params_opt[0])],
                "a_y": [float(self.params_opt[1])],
                "a_z": [float(self.params_opt[2])],
                "a_zz": [float(self.params_opt[3])],
                "q": [float(self.params_opt[4])],
                "delta_omega": [self.delta_omega_opt],
                "omega_true": [self.omega_true],
                "success": [int(self.success)],
                "nfev": [self.nfev],
                "expectation_Jz": [self.expectation_Jz],
                "variance_Jz": [self.variance_Jz],
                "message": [self.message],
                "history_json": [json.dumps(self.history)],
            },
        )

    def _save_sidecars(self, path: Path) -> None:
        history_path = path.with_stem(path.stem + "-history")
        pd.DataFrame({"history": [self.history]}).to_parquet(
            history_path,
            index=False,
        )

    @classmethod
    def _load_sidecars(cls, path: Path) -> dict:  # type: ignore[override]
        """Load the history sidecar file."""
        history_path = path.with_stem(path.stem + "-history")
        if history_path.exists():
            history = list(pd.read_parquet(history_path)["history"].iloc[0])
        else:
            history = []
        return {"history": history}

    @classmethod
    def from_parquet(cls, path: str | Path) -> OATNelderMeadResult:
        df = pd.read_parquet(path)
        cls._validate_columns(df)
        sidecars = cls._load_sidecars(Path(path))
        return cls(
            delta_omega_opt=float(df["delta_omega"].iloc[0]),
            params_opt=np.array(
                [
                    float(df["a_x"].iloc[0]),
                    float(df["a_y"].iloc[0]),
                    float(df["a_z"].iloc[0]),
                    float(df["a_zz"].iloc[0]),
                    float(df["q"].iloc[0]),
                ],
            ),
            omega_true=float(df["omega_true"].iloc[0]),
            success=bool(int(df["success"].iloc[0])),
            nfev=int(df["nfev"].iloc[0]),
            message=str(df["message"].iloc[0]),
            expectation_Jz=float(df["expectation_Jz"].iloc[0]),
            variance_Jz=float(df["variance_Jz"].iloc[0]),
            history=sidecars["history"],
        )


@dataclass
class OATNScalingResult(ParquetSerializable):
    """Result from 5D optimisation for a single (N, ω) pair with OAT.

    Attributes:
        N: Number of system particles (J_S = N/2).
        n_ancilla: Number of ancilla particles (J_A = n_ancilla/2).
        omega: Phase rate value.
        delta_omega_opt: Best sensitivity Δω found with OAT.
        sql: SQL = 1/(√N × t_hold).
        ratio: SQL / Δω_opt (ratio > 1 means beating SQL).
        a_x_opt: Optimal J_x^A drive coefficient.
        a_y_opt: Optimal J_y^A drive coefficient.
        a_z_opt: Optimal J_z^A drive coefficient.
        a_zz_opt: Optimal Ising interaction coefficient.
        q_opt: Optimal OAT squeezing strength.
        delta_omega_no_oat: Sensitivity with q=0 (same drive params).
        improvement_factor: Δω_no_oat / Δω_opt (>1 means OAT helps).
        expectation_Jz: ⟨J_z^S⟩ at the optimal operating point.
        variance_Jz: Var(J_z^S) at the optimal operating point.
        t_hold: Holding time (fixed at 10.0).
        fd_step: Finite-difference step.
        success: Whether Nelder-Mead reported success.
        nfev: Number of function evaluations.
    """

    N: int
    n_ancilla: int
    omega: float
    delta_omega_opt: float
    sql: float
    ratio: float
    a_x_opt: float
    a_y_opt: float
    a_z_opt: float
    a_zz_opt: float
    q_opt: float
    delta_omega_no_oat: float
    improvement_factor: float
    expectation_Jz: float = 0.0
    variance_Jz: float = 0.0
    t_hold: float = 10.0
    fd_step: float = 1e-6
    success: bool = False
    nfev: int = 0

    _PARQUET_COLUMNS: ClassVar[list[str]] = [
        "N",
        "n_ancilla",
        "omega",
        "delta_omega_opt",
        "sql",
        "ratio",
        "a_x_opt",
        "a_y_opt",
        "a_z_opt",
        "a_zz_opt",
        "q_opt",
        "delta_omega_no_oat",
        "improvement_factor",
        "expectation_Jz",
        "variance_Jz",
        "t_hold",
        "fd_step",
        "success",
        "nfev",
    ]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "N": [self.N],
                "n_ancilla": [self.n_ancilla],
                "omega": [self.omega],
                "delta_omega_opt": [self.delta_omega_opt],
                "sql": [self.sql],
                "ratio": [self.ratio],
                "a_x_opt": [self.a_x_opt],
                "a_y_opt": [self.a_y_opt],
                "a_z_opt": [self.a_z_opt],
                "a_zz_opt": [self.a_zz_opt],
                "q_opt": [self.q_opt],
                "delta_omega_no_oat": [self.delta_omega_no_oat],
                "improvement_factor": [self.improvement_factor],
                "expectation_Jz": [self.expectation_Jz],
                "variance_Jz": [self.variance_Jz],
                "t_hold": [self.t_hold],
                "fd_step": [self.fd_step],
                "success": [int(self.success)],
                "nfev": [self.nfev],
            },
        )

    @classmethod
    def from_parquet(cls, path: str | Path) -> OATNScalingResult:
        path = Path(path)
        df = pd.read_parquet(path)
        cls._validate_columns(df)
        row = df.iloc[0]
        return cls(
            N=int(row["N"]),
            n_ancilla=int(row["n_ancilla"]),
            omega=float(row["omega"]),
            delta_omega_opt=float(row["delta_omega_opt"]),
            sql=float(row["sql"]),
            ratio=float(row["ratio"]),
            a_x_opt=float(row["a_x_opt"]),
            a_y_opt=float(row["a_y_opt"]),
            a_z_opt=float(row["a_z_opt"]),
            a_zz_opt=float(row["a_zz_opt"]),
            q_opt=float(row["q_opt"]),
            delta_omega_no_oat=float(row["delta_omega_no_oat"]),
            improvement_factor=float(row["improvement_factor"]),
            expectation_Jz=float(row["expectation_Jz"]),
            variance_Jz=float(row["variance_Jz"]),
            t_hold=float(row["t_hold"]),
            fd_step=float(row["fd_step"]),
            success=bool(int(row["success"])),
            nfev=int(row["nfev"]),
        )


@dataclass
class OATNScalingScanResult(ParquetSerializable):
    """Collection of OAT N-scaling results for a grid of (N, ω) pairs.

    Attributes:
        results: List of per-(N, ω) OATNScalingResult.
    """

    results: list[OATNScalingResult] = field(default_factory=list)

    _PARQUET_COLUMNS: ClassVar[list[str]] = OATNScalingResult._PARQUET_COLUMNS

    def to_dataframe(self) -> pd.DataFrame:
        if not self.results:
            return pd.DataFrame(columns=self._PARQUET_COLUMNS)
        return pd.concat([r.to_dataframe() for r in self.results], ignore_index=True)

    @classmethod
    def from_parquet(cls, path: str | Path) -> OATNScalingScanResult:
        path = Path(path)
        df = pd.read_parquet(path)
        cls._validate_columns(df)
        results: list[OATNScalingResult] = []
        for _, row in df.iterrows():
            results.append(
                OATNScalingResult(
                    N=int(row["N"]),
                    n_ancilla=int(row["n_ancilla"]),
                    omega=float(row["omega"]),
                    delta_omega_opt=float(row["delta_omega_opt"]),
                    sql=float(row["sql"]),
                    ratio=float(row["ratio"]),
                    a_x_opt=float(row["a_x_opt"]),
                    a_y_opt=float(row["a_y_opt"]),
                    a_z_opt=float(row["a_z_opt"]),
                    a_zz_opt=float(row["a_zz_opt"]),
                    q_opt=float(row["q_opt"]),
                    delta_omega_no_oat=float(row["delta_omega_no_oat"]),
                    improvement_factor=float(row["improvement_factor"]),
                    expectation_Jz=float(row["expectation_Jz"]),
                    variance_Jz=float(row["variance_Jz"]),
                    t_hold=float(row["t_hold"]),
                    fd_step=float(row["fd_step"]),
                    success=bool(int(row["success"])),
                    nfev=int(row["nfev"]),
                ),
            )
        return cls(results=results)

    @property
    def N_values(self) -> np.ndarray:
        return np.array(sorted({r.N for r in self.results}))

    @property
    def omega_values(self) -> np.ndarray:
        return np.array(sorted({r.omega for r in self.results}))


# ============================================================================
# Operator Construction
# ============================================================================


def build_operators(N_S: int, N_A: int) -> dict[str, np.ndarray]:
    """Build operators in the (N_S+1)*(N_A+1)-dimensional total Hilbert space.

    Delegates to :func:`src.physics.bipartite_operators.build_operators`
    with ``N_sys=N_S, N_anc=N_A``.

    Args:
        N_S: Number of system particles (N_S >= 1). System dim = N_S + 1.
        N_A: Number of ancilla particles (N_A >= 1). Ancilla dim = N_A + 1.

    Returns:
        Dict with keys 'Jz_S', 'Jx_S', 'Jy_S', 'Jz_A', 'Jx_A', 'Jy_A',
        each a (N_S+1)*(N_A+1) x (N_S+1)*(N_A+1) complex Hermitian matrix.
        Also includes 'I_S', 'I_A' (identities) and 'I_full'.
    """
    return _bipartite.build_operators(N_sys=N_S, N_anc=N_A)


def build_system_only_bs_unitary(N_S: int, N_A: int, T_bs: float = T_BS) -> np.ndarray:
    """System-only beam-splitter unitary in the total Hilbert space.

    Delegates to :func:`src.physics.bipartite_operators.build_system_only_bs_unitary`
    with ``N_sys=N_S, N_anc=N_A, T_bs=T_bs``.

    Args:
        N_S: Number of system particles.
        N_A: Number of ancilla particles.
        T_bs: Beam-splitter duration (default pi/2 for 50/50).

    Returns:
        (N_S+1)*(N_A+1) x (N_S+1)*(N_A+1) unitary matrix.
    """
    return _bipartite.build_system_only_bs_unitary(
        N_sys=N_S,
        N_anc=N_A,
        T_bs=T_bs,
    )


# ============================================================================
# State Preparation
# ============================================================================


def oat_initial_state(N_S: int, N_A: int) -> np.ndarray:
    """Initial state for the OAT protocol.

    |Psi_0> = |J_S, J_S>_S ⊗ |J_A, -J_A>_x^A

    where the system is in the top Dicke state and the ancilla is a CSS along -x.

    Args:
        N_S: Number of system particles.
        N_A: Number of ancilla particles.

    Returns:
        Normalised complex vector of length (N_S+1)*(N_A+1).
    """
    d_sys = N_S + 1

    # System: top Dicke state |J_S, J_S> = [1, 0, ..., 0]^T
    psi_sys = np.zeros(d_sys, dtype=complex)
    psi_sys[0] = 1.0

    # Ancilla: CSS along -x
    psi_anc = coherent_spin_state(N_A)

    # Product state
    psi = np.kron(psi_sys, psi_anc)
    assert np.isclose(np.linalg.norm(psi), 1.0), (
        f"Initial state not normalised for N_S={N_S}, N_A={N_A}"
    )
    return psi


# ============================================================================
# OAT Unitary
# ============================================================================


def oat_unitary(N_A: int, q: float, d_S: int) -> np.ndarray:
    """OAT unitary acting only on the ancilla.

    U_OAT = I_{d_S} ⊗ diag(exp(-i q m^2))
    where m = N_A/2, N_A/2-1, ..., -N_A/2 (descending Dicke eigenvalues).

    Args:
        N_A: Number of ancilla particles.
        q: Squeezing strength q = chi * T_OAT (dimensionless).
        d_S: System Hilbert space dimension.

    Returns:
        (d_S * (N_A+1)) x (d_S * (N_A+1)) unitary matrix.
    """
    d_A = N_A + 1
    d_tot = d_S * d_A

    # J_z eigenvalues: m = N_A/2, N_A/2-1, ..., -N_A/2
    m_values = np.arange(N_A / 2.0, -N_A / 2.0 - 1, -1)

    # Phase factors: exp(-i q m^2)
    phases = np.exp(-1j * q * m_values**2)

    # Diagonal matrix in Dicke basis
    D = np.diag(phases)

    # Embed into full space
    I_S = np.eye(d_S, dtype=complex)
    U = np.kron(I_S, D).astype(complex)

    # Validate unitarity
    I_full = np.eye(d_tot, dtype=complex)
    assert np.allclose(U @ U.conj().T, I_full, atol=1e-12), (
        f"OAT unitary not unitary for N_A={N_A}, q={q}"
    )

    # Validate diagonal form
    assert np.allclose(U @ U.conj().T, U.conj().T @ U, atol=1e-12), (
        "OAT unitary not normal"
    )

    return U


# ============================================================================
# Hold Hamiltonian
# ============================================================================


def build_modulated_drive_hamiltonian(
    N_A: int,
    omega: float,
    a_x: float,
    a_y: float,
    a_z: float,
    ops: dict[str, np.ndarray],
) -> np.ndarray:
    """Build the omega-modulated ancilla drive Hamiltonian.

    H_A = omega (a_x J_x^A + a_y J_y^A + a_z J_z^A)

    Args:
        N_A: Number of ancilla particles (for dimension check).
        omega: Phase rate parameter (scales the whole drive).
        a_x: J_x^A coefficient.
        a_y: J_y^A coefficient.
        a_z: J_z^A coefficient.
        ops: Operators from build_operators().

    Returns:
        Hermitian matrix in the total Hilbert space.
    """
    d_tot = next(iter(ops.values())).shape[0]

    H = np.zeros((d_tot, d_tot), dtype=complex)
    if a_x != 0.0:
        H += a_x * ops["Jx_A"]
    if a_y != 0.0:
        H += a_y * ops["Jy_A"]
    if a_z != 0.0:
        H += a_z * ops["Jz_A"]
    H = omega * H
    H = 0.5 * (H + H.conj().T)
    assert np.allclose(H, H.conj().T, atol=1e-12), "Drive Hamiltonian not Hermitian"
    return H


def build_iszz_interaction(
    a_zz: float,
    ops: dict[str, np.ndarray],
) -> np.ndarray:
    """Build the Ising interaction Hamiltonian.

    Delegates to :func:`src.physics.bipartite_operators.build_iszz_interaction`.

    Args:
        a_zz: Interaction coupling coefficient.
        ops: Operators from build_operators().

    Returns:
        Hermitian matrix in the total Hilbert space.
    """
    return _bipartite.build_iszz_interaction(a_zz, ops)


def build_hold_hamiltonian(
    N_S: int,
    N_A: int,
    omega: float,
    a_x: float,
    a_y: float,
    a_z: float,
    a_zz: float,
    ops: dict[str, np.ndarray],
) -> np.ndarray:
    """Build the total holding Hamiltonian for the OAT protocol.

    H = omega J_z^S + omega (a_x J_x^A + a_y J_y^A + a_z J_z^A) + a_zz J_z^S J_z^A

    Args:
        N_S: Number of system particles.
        N_A: Number of ancilla particles.
        omega: Phase rate parameter.
        a_x: J_x^A drive coefficient.
        a_y: J_y^A drive coefficient.
        a_z: J_z^A drive coefficient.
        a_zz: Ising interaction coefficient.
        ops: Operators from build_operators().

    Returns:
        Hermitian Hamiltonian matrix.
    """
    H = omega * ops["Jz_S"]
    H += build_modulated_drive_hamiltonian(N_A, omega, a_x, a_y, a_z, ops)
    H += build_iszz_interaction(a_zz, ops)
    H = 0.5 * (H + H.conj().T)
    assert np.allclose(H, H.conj().T, atol=1e-12), (
        f"Total Hamiltonian not Hermitian for N_S={N_S}, N_A={N_A}"
    )
    return H


# ============================================================================
# Circuit Evolution
# ============================================================================


def hold_unitary(
    N_S: int,
    N_A: int,
    t_hold: float,
    omega: float,
    a_x: float,
    a_y: float,
    a_z: float,
    a_zz: float,
    ops: dict[str, np.ndarray],
) -> np.ndarray:
    """Holding-time unitary for the OAT protocol.

    U_hold(t_hold) = exp(-i t_hold H)

    Args:
        N_S: Number of system particles.
        N_A: Number of ancilla particles.
        t_hold: Holding-time strength.
        omega: Phase rate parameter.
        a_x: J_x^A drive coefficient.
        a_y: J_y^A drive coefficient.
        a_z: J_z^A drive coefficient.
        a_zz: Ising interaction coefficient.
        ops: Operators from build_operators().

    Returns:
        Unitary matrix.
    """
    H = build_hold_hamiltonian(N_S, N_A, omega, a_x, a_y, a_z, a_zz, ops)
    U = expm(-1j * t_hold * H)
    d_tot = next(iter(ops.values())).shape[0]
    I_full = np.eye(d_tot, dtype=complex)
    assert np.allclose(U @ U.conj().T, I_full, atol=1e-12), (
        f"Hold unitary not unitary for t_hold={t_hold}, omega={omega}"
    )
    return U


def evolve_oat_circuit(
    N_S: int,
    N_A: int,
    psi0: np.ndarray,
    T_bs: float,
    t_hold: float,
    omega: float,
    a_x: float,
    a_y: float,
    a_z: float,
    a_zz: float,
    q: float,
    ops: dict[str, np.ndarray],
) -> np.ndarray:
    """Run the full OAT protocol circuit.

    |psi_final> = U_BS_S * U_hold(t_hold) * U_OAT * U_BS_S * |psi_0>

    Args:
        N_S: Number of system particles.
        N_A: Number of ancilla particles.
        psi0: Initial state vector (must be normalised).
        T_bs: Beam-splitter duration.
        t_hold: Holding-time strength.
        omega: Phase rate parameter.
        a_x: J_x^A drive coefficient.
        a_y: J_y^A drive coefficient.
        a_z: J_z^A drive coefficient.
        a_zz: Ising interaction coefficient.
        q: OAT squeezing strength.
        ops: Operators from build_operators().

    Returns:
        Final normalised state vector.
    """
    assert np.isclose(np.linalg.norm(psi0), 1.0), (
        f"Initial state not normalised for N_S={N_S}, N_A={N_A}"
    )

    U_bs = build_system_only_bs_unitary(N_S, N_A, T_bs)
    U_oat_val = oat_unitary(N_A, q, N_S + 1)

    psi = U_bs @ psi0
    psi = U_oat_val @ psi
    psi = (
        hold_unitary(
            N_S,
            N_A,
            t_hold,
            omega,
            a_x,
            a_y,
            a_z,
            a_zz,
            ops,
        )
        @ psi
    )
    psi = U_bs @ psi

    assert np.isclose(np.linalg.norm(psi), 1.0), (
        f"Final state not normalised for N_S={N_S}, N_A={N_A}"
    )
    return psi


# ============================================================================
# Sensitivity Computation
# ============================================================================


def compute_oat_sensitivity(
    N_S: int,
    N_A: int,
    psi0: np.ndarray,
    T_bs: float,
    t_hold: float,
    omega_true: float,
    a_x: float,
    a_y: float,
    a_z: float,
    a_zz: float,
    q: float,
    ops: dict[str, np.ndarray],
    fd_step: float = FD_STEP,
    meas_op: np.ndarray | None = None,
) -> float:
    """Compute the error-propagation sensitivity Delta_omega for the OAT protocol.

    Delta_omega = sqrt(Var(O)) / |d<O>/domega|

    where O = J_z^S (default measurement operator).

    Args:
        N_S: Number of system particles.
        N_A: Number of ancilla particles.
        psi0: Initial state vector.
        T_bs: Beam-splitter duration.
        t_hold: Holding-time strength.
        omega_true: True phase rate parameter.
        a_x: J_x^A drive coefficient.
        a_y: J_y^A drive coefficient.
        a_z: J_z^A drive coefficient.
        a_zz: Ising interaction coefficient.
        q: OAT squeezing strength.
        ops: Operators from build_operators().
        fd_step: Finite-difference step size (default 1e-6).
        meas_op: Measurement operator (default ops['Jz_S']).

    Returns:
        Sensitivity Delta_omega. Returns inf if derivative is zero.
    """
    if meas_op is None:
        meas_op = ops["Jz_S"]

    # Evaluate at omega_true
    psi = evolve_oat_circuit(
        N_S,
        N_A,
        psi0,
        T_bs,
        t_hold,
        omega_true,
        a_x,
        a_y,
        a_z,
        a_zz,
        q,
        ops,
    )
    _, var = compute_expectation_and_variance(psi, meas_op)

    # Central finite difference for d<O>/domega
    psi_plus = evolve_oat_circuit(
        N_S,
        N_A,
        psi0,
        T_bs,
        t_hold,
        omega_true + fd_step,
        a_x,
        a_y,
        a_z,
        a_zz,
        q,
        ops,
    )
    psi_minus = evolve_oat_circuit(
        N_S,
        N_A,
        psi0,
        T_bs,
        t_hold,
        omega_true - fd_step,
        a_x,
        a_y,
        a_z,
        a_zz,
        q,
        ops,
    )
    exp_plus = np.real(psi_plus.conj() @ meas_op @ psi_plus)
    exp_minus = np.real(psi_minus.conj() @ meas_op @ psi_minus)
    d_exp = (exp_plus - exp_minus) / (2.0 * fd_step)

    if abs(d_exp) < 1e-12:
        return float("inf")

    if var < 1e-15:
        return float("inf")

    return float(np.sqrt(var) / abs(d_exp))


# ============================================================================
# Decoupled Baseline
# ============================================================================


def compute_oat_decoupled_baseline(
    N_S: int,
    N_A: int,
    omega_true: float = 1.0,
) -> float:
    """Compute the decoupled baseline sensitivity for the OAT protocol.

    At (a_x = a_y = a_z = a_zz = q = 0), the circuit reduces to a standard
    N_S-particle MZI with CSS ancilla, giving Delta_omega = 1/(sqrt(N_S) * T_HOLD).

    Args:
        N_S: Number of system particles.
        N_A: Number of ancilla particles.
        omega_true: Phase rate value.

    Returns:
        Delta_omega at the decoupled configuration.
    """
    ops = build_operators(N_S, N_A)
    psi0 = oat_initial_state(N_S, N_A)
    return compute_oat_sensitivity(
        N_S,
        N_A,
        psi0,
        T_BS,
        T_HOLD,
        omega_true,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        ops,
    )


def verify_oat_decoupled_baseline(
    N_S_values: list[int] | None = None,
    N_A_values: list[int] | None = None,
    omega_values: list[float] | None = None,
    rtol: float = 1e-10,
) -> dict[tuple[int, int, float], bool]:
    """Verify the decoupled baseline for all (N_S, N_A, omega) triples.

    At zero drive and zero interaction, the sensitivity must equal
    Delta_omega = 1/(sqrt(N_S) * T_HOLD) to machine precision.

    Args:
        N_S_values: List of N_S values (default: [1]).
        N_A_values: List of N_A values (default: [2]).
        omega_values: List of omega values (default: all OMEGA_VALS).
        rtol: Relative tolerance for comparison.

    Returns:
        Dict mapping (N_S, N_A, omega) -> PASS/FAIL (True/False).
    """
    if N_S_values is None:
        N_S_values = [1]
    if N_A_values is None:
        N_A_values = [2]
    if omega_values is None:
        omega_values = OMEGA_VALS
    results: dict[tuple[int, int, float], bool] = {}
    for N_S in N_S_values:
        sql_ref = sql_reference(N_S)
        for N_A in N_A_values:
            for omega in omega_values:
                delta = compute_oat_decoupled_baseline(N_S, N_A, omega)
                results[(N_S, N_A, omega)] = bool(
                    np.isclose(delta, sql_ref, rtol=rtol),
                )
    return results


def compute_oat_decoupled_with_oat(
    N_S: int,
    N_A: int,
    q: float,
    omega_true: float = 1.0,
) -> float:
    """Decoupled limit with OAT but no drive/interaction.

    At a_x = a_y = a_z = a_zz = 0 and q > 0, the OAT should not affect
    the J_z^S measurement because H_int = 0 decouples the ancilla.

    Args:
        N_S: Number of system particles.
        N_A: Number of ancilla particles.
        q: OAT squeezing strength.
        omega_true: Phase rate value.

    Returns:
        Delta_omega at the decoupled-with-OAT configuration.
    """
    ops = build_operators(N_S, N_A)
    psi0 = oat_initial_state(N_S, N_A)
    return compute_oat_sensitivity(
        N_S,
        N_A,
        psi0,
        T_BS,
        T_HOLD,
        omega_true,
        0.0,
        0.0,
        0.0,
        0.0,
        q,
        ops,
    )


# ============================================================================
# 5D Random Search
# ============================================================================


def oat_random_search(
    N_S: int,
    N_A: int,
    omega: float,
    n_samples: int = N_RANDOM,
    drive_bounds: tuple[float, float] = DRIVE_BOUNDS,
    q_bounds: tuple[float, float] = Q_BOUNDS,
    seed: int | None = 42,
) -> OATRandomSearchResult:
    """Random search over the 5D parameter space (a_x, a_y, a_z, a_zz, q).

    Args:
        N_S: Number of system particles.
        N_A: Number of ancilla particles.
        omega: Phase rate value.
        n_samples: Number of random points to evaluate.
        drive_bounds: (min, max) for all four drive coefficients.
        q_bounds: (min, max) for the OAT squeezing strength.
        seed: Random seed for reproducibility.

    Returns:
        OATRandomSearchResult with all samples and best found.
    """
    rng = np.random.default_rng(seed)
    ops = build_operators(N_S, N_A)
    psi0 = oat_initial_state(N_S, N_A)
    lo_d, hi_d = drive_bounds
    lo_q, hi_q = q_bounds

    samples = np.zeros((n_samples, 5), dtype=float)
    samples[:, :4] = rng.uniform(lo_d, hi_d, size=(n_samples, 4))
    samples[:, 4] = rng.uniform(lo_q, hi_q, size=n_samples)

    deltas = np.full(n_samples, np.inf, dtype=float)

    for i in range(n_samples):
        ax = float(samples[i, 0])
        ay = float(samples[i, 1])
        az = float(samples[i, 2])
        azz = float(samples[i, 3])
        q_val = float(samples[i, 4])
        domega = compute_oat_sensitivity(
            N_S,
            N_A,
            psi0,
            T_BS,
            T_HOLD,
            omega,
            ax,
            ay,
            az,
            azz,
            q_val,
            ops,
        )
        deltas[i] = domega

    best_idx = int(np.argmin(deltas))
    best_params: tuple[float, float, float, float, float] = (
        float(samples[best_idx, 0]),
        float(samples[best_idx, 1]),
        float(samples[best_idx, 2]),
        float(samples[best_idx, 3]),
        float(samples[best_idx, 4]),
    )

    return OATRandomSearchResult(
        samples=samples,
        delta_omega_values=deltas,
        best_params=best_params,
        best_delta_omega=float(deltas[best_idx]),
        omega_value=omega,
        sql=sql_reference(N_S),
        t_hold=T_HOLD,
    )


# ============================================================================
# 5D Nelder-Mead Optimisation
# ============================================================================


def oat_sensitivity_objective(
    params: np.ndarray,
    N_S: int,
    N_A: int,
    omega_true: float,
    ops: dict[str, np.ndarray],
    psi0: np.ndarray,
    t_hold: float = T_HOLD,
    T_bs: float = T_BS,
    fd_step: float = FD_STEP,
    drive_bounds: tuple[float, float] = DRIVE_BOUNDS,
    q_bounds: tuple[float, float] = Q_BOUNDS,
    penalty_scale: float = 1e6,
) -> float:
    """Objective function for minimising Delta_omega in the OAT protocol.

    params = [a_x, a_y, a_z, a_zz, q] (5 elements).

    Args:
        params: 5-element parameter vector.
        N_S: Number of system particles.
        N_A: Number of ancilla particles.
        omega_true: True phase rate.
        ops: Operators from build_operators().
        psi0: Initial state vector.
        t_hold: Holding time.
        T_bs: Beam-splitter duration.
        fd_step: Finite-difference step.
        drive_bounds: (min, max) for drive coefficients.
        q_bounds: (min, max) for OAT strength.
        penalty_scale: Scale for bound-violation penalty.

    Returns:
        Delta_omega (plus penalty if bounds violated).
    """
    ax = float(params[0])
    ay = float(params[1])
    az = float(params[2])
    azz = float(params[3])
    q_val = float(params[4])

    # Bound enforcement
    lo_d, hi_d = drive_bounds
    lo_q, hi_q = q_bounds
    penalty = 0.0
    for val in (ax, ay, az, azz):
        if val < lo_d:
            penalty += penalty_scale * (lo_d - val) ** 2
        if val > hi_d:
            penalty += penalty_scale * (val - hi_d) ** 2
    if q_val < lo_q:
        penalty += penalty_scale * (lo_q - q_val) ** 2
    if q_val > hi_q:
        penalty += penalty_scale * (q_val - hi_q) ** 2

    if penalty > 0.0:
        return float(1e10 + penalty)

    return compute_oat_sensitivity(
        N_S,
        N_A,
        psi0,
        T_bs,
        t_hold,
        omega_true,
        ax,
        ay,
        az,
        azz,
        q_val,
        ops,
        fd_step,
    )


def run_oat_nelder_mead(
    N_S: int,
    N_A: int,
    omega_true: float,
    ops: dict[str, np.ndarray] | None = None,
    psi0: np.ndarray | None = None,
    x0: np.ndarray | None = None,
    seed: int | None = None,
    maxiter: int = NM_MAXITER,
    xatol: float = 1e-8,
    fatol: float = 1e-8,
    adaptive: bool = True,
    drive_bounds: tuple[float, float] = DRIVE_BOUNDS,
    q_bounds: tuple[float, float] = Q_BOUNDS,
    track_history: bool = False,
) -> OATNelderMeadResult:
    """Run Nelder-Mead optimisation for the 5D OAT protocol.

    Args:
        N_S: Number of system particles.
        N_A: Number of ancilla particles.
        omega_true: True phase rate parameter.
        ops: Operators (built if None).
        psi0: Initial state (built if None).
        x0: Initial 5-parameter vector [ax, ay, az, azz, q]. Random if None.
        seed: Random seed (used if x0 is None).
        maxiter: Maximum Nelder-Mead iterations.
        xatol: Absolute parameter tolerance.
        fatol: Absolute function tolerance.
        adaptive: Use adaptive Nelder-Mead parameters.
        drive_bounds: (min, max) for drive coefficients.
        q_bounds: (min, max) for OAT strength.
        track_history: If True, record objective values per iteration.

    Returns:
        OATNelderMeadResult.
    """
    if ops is None:
        ops = build_operators(N_S, N_A)
    if psi0 is None:
        psi0 = oat_initial_state(N_S, N_A)

    if x0 is None:
        rng = np.random.default_rng(seed)
        lo_d, hi_d = drive_bounds
        lo_q, hi_q = q_bounds
        x0 = np.zeros(5, dtype=float)
        x0[:4] = rng.uniform(lo_d, hi_d, size=4)
        x0[4] = rng.uniform(lo_q, hi_q)
    else:
        x0 = np.asarray(x0, dtype=float)
        assert x0.shape == (5,), f"x0 must have 5 elements, got {x0.shape}"

    def objective(p: np.ndarray) -> float:
        return oat_sensitivity_objective(
            p,
            N_S,
            N_A,
            omega_true,
            ops,
            psi0,
            drive_bounds=drive_bounds,
            q_bounds=q_bounds,
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

    # Compute diagnostics at the optimal point
    psi_final = evolve_oat_circuit(
        N_S,
        N_A,
        psi0,
        T_BS,
        T_HOLD,
        omega_true,
        float(opt_params[0]),
        float(opt_params[1]),
        float(opt_params[2]),
        float(opt_params[3]),
        float(opt_params[4]),
        ops,
    )
    exp_val, var_val = compute_expectation_and_variance(psi_final, ops["Jz_S"])

    return OATNelderMeadResult(
        delta_omega_opt=float(result.fun),
        params_opt=opt_params,
        omega_true=omega_true,
        success=bool(result.success),
        nfev=int(result.nfev),
        message=str(result.message),
        expectation_Jz=exp_val,
        variance_Jz=var_val,
        history=history.copy(),
    )


# ============================================================================
# N-Scaling: Single (N, ω) Worker
# ============================================================================


def _get_oat_optimisation_budget(dim: int) -> tuple[int, int]:
    """Return (n_nm_refine, nm_maxiter) based on Hilbert-space dimension.

    Larger dimensions are slower, so the budget scales down as dim grows
    to keep wall time practical.

    Args:
        dim: Hilbert-space dimension ``(N_S + 1) * (N_A + 1)``.

    Returns:
        Tuple ``(n_nm_refine, nm_maxiter)``.
    """
    if dim <= 16:  # N_S, N_A <= 3
        return 10, 2000
    if dim <= 25:  # N_S, N_A <= 4
        return 8, 1500
    if dim <= 36:  # N_S, N_A <= 5
        return 6, 1000
    # N_S, N_A >= 6
    return 4, 500


def _compute_oat_improvement(delta_no_oat: float, delta_omega_opt: float) -> float:
    """Compute improvement factor of OAT over no-OAT baseline.

    Args:
        delta_no_oat: Sensitivity without OAT (q=0).
        delta_omega_opt: Sensitivity with optimal OAT.

    Returns:
        ``delta_no_oat / delta_omega_opt`` when both are finite and
        positive, otherwise 1.0.
    """
    if (
        np.isfinite(delta_omega_opt)
        and delta_omega_opt > 0
        and np.isfinite(delta_no_oat)
        and delta_no_oat > 0
    ):
        return delta_no_oat / delta_omega_opt
    return 1.0


def _compute_oat_ratio(sql_val: float, delta_omega_opt: float) -> float:
    """Compute SQL ratio ``sql / delta_omega_opt``.

    Args:
        sql_val: Standard Quantum Limit sensitivity.
        delta_omega_opt: Sensitivity with optimal parameters.

    Returns:
        ``sql_val / delta_omega_opt`` when finite and positive,
        otherwise ``NaN``.
    """
    if np.isfinite(delta_omega_opt) and delta_omega_opt > 0:
        return sql_val / delta_omega_opt
    return float("nan")


def run_single_oat_optimisation(
    N_S: int,
    N_A: int,
    omega: float,
    seed: int | None = 42,
) -> OATNScalingResult:
    """Run the full optimisation pipeline for a single (N_S, N_A, ω) triple.

    1. 5D random search (1000 samples).
    2. Nelder-Mead refinement from top points.
    3. No-OAT baseline at the optimal drive parameters.
    4. Return the best result.

    Args:
        N_S: Number of system particles.
        N_A: Number of ancilla particles.
        omega: Phase rate value.
        seed: Base random seed (incremented per call).

    Returns:
        OATNScalingResult with the optimal parameters and sensitivity.
    """
    base_seed = seed if seed is not None else 42
    ops = build_operators(N_S, N_A)
    psi0 = oat_initial_state(N_S, N_A)

    dim = (N_S + 1) * (N_A + 1)
    n_nm_refine, nm_maxiter = _get_oat_optimisation_budget(dim)

    def rs_fn(n_samples: int, seed: int, **kw: object) -> OATRandomSearchResult:
        return oat_random_search(N_S, N_A, omega, n_samples=n_samples, seed=seed)

    def nm_fn(x0: np.ndarray, seed: int, **kw: object) -> OATNelderMeadResult:
        return run_oat_nelder_mead(
            N_S=N_S,
            N_A=N_A,
            omega_true=omega,
            ops=ops,
            psi0=psi0,
            x0=x0,
            seed=seed,
            maxiter=nm_maxiter,
        )

    best_nm, _ = run_two_phase_pipeline(
        rs_fn,
        nm_fn,
        TwoPhaseConfig(n_random=N_RANDOM, n_nm_refine=n_nm_refine, seed=base_seed),
    )

    # No-OAT baseline: recompute with q=0 at optimal drive parameters
    delta_no_oat = compute_oat_sensitivity(
        N_S,
        N_A,
        psi0,
        T_BS,
        T_HOLD,
        omega,
        float(best_nm.params_opt[0]),
        float(best_nm.params_opt[1]),
        float(best_nm.params_opt[2]),
        float(best_nm.params_opt[3]),
        0.0,  # q=0
        ops,
    )

    sql_val = sql_reference(N_S)
    improvement = _compute_oat_improvement(delta_no_oat, best_nm.delta_omega_opt)

    return OATNScalingResult(
        N=N_S,
        n_ancilla=N_A,
        omega=omega,
        delta_omega_opt=best_nm.delta_omega_opt,
        sql=sql_val,
        ratio=_compute_oat_ratio(sql_val, best_nm.delta_omega_opt),
        a_x_opt=float(best_nm.params_opt[0]),
        a_y_opt=float(best_nm.params_opt[1]),
        a_z_opt=float(best_nm.params_opt[2]),
        a_zz_opt=float(best_nm.params_opt[3]),
        q_opt=float(best_nm.params_opt[4]),
        delta_omega_no_oat=delta_no_oat,
        improvement_factor=improvement,
        expectation_Jz=best_nm.expectation_Jz,
        variance_Jz=best_nm.variance_Jz,
        success=best_nm.success,
        nfev=best_nm.nfev,
    )


# ============================================================================
# Data Generation Pipeline
# ============================================================================


def _run_single_oat_for_parallel(
    args: tuple[int, int, float],
) -> dict[str, int | float | str]:
    """Worker for parallel OAT N-scaling scan.

    Args:
        args: Tuple (N_S, N_A, omega).

    Returns:
        Dict of result data.
    """
    N_S, N_A, omega = args
    print(f"  [run] N_S={N_S}, N_A={N_A}, omega={omega}")
    result = run_single_oat_optimisation(N_S, N_A, omega)
    return {
        "N": result.N,
        "n_ancilla": result.n_ancilla,
        "omega": result.omega,
        "delta_omega_opt": result.delta_omega_opt,
        "sql": result.sql,
        "ratio": result.ratio,
        "a_x_opt": result.a_x_opt,
        "a_y_opt": result.a_y_opt,
        "a_z_opt": result.a_z_opt,
        "a_zz_opt": result.a_zz_opt,
        "q_opt": result.q_opt,
        "delta_omega_no_oat": result.delta_omega_no_oat,
        "improvement_factor": result.improvement_factor,
        "expectation_Jz": result.expectation_Jz,
        "variance_Jz": result.variance_Jz,
        "success": int(result.success),
        "nfev": result.nfev,
    }


# ── Checkpoint recovery callbacks ──────────────────────────────────────────


def _build_oat_result_from_row(row: dict) -> OATNScalingResult:
    """Build OATNScalingResult from a Parquet row dict."""
    return OATNScalingResult(
        N=int(row["N"]),
        n_ancilla=int(row["n_ancilla"]),
        omega=float(row["omega"]),
        delta_omega_opt=float(row["delta_omega_opt"]),
        sql=float(row["sql"]),
        ratio=float(row["ratio"]),
        a_x_opt=float(row["a_x_opt"]),
        a_y_opt=float(row["a_y_opt"]),
        a_z_opt=float(row["a_z_opt"]),
        a_zz_opt=float(row["a_zz_opt"]),
        q_opt=float(row["q_opt"]),
        delta_omega_no_oat=float(row["delta_omega_no_oat"]),
        improvement_factor=float(row["improvement_factor"]),
        expectation_Jz=float(row["expectation_Jz"]),
        variance_Jz=float(row["variance_Jz"]),
        success=bool(int(row["success"])),
        nfev=int(row["nfev"]),
    )


def _build_oat_result_from_dict(d: dict) -> OATNScalingResult:
    """Build OATNScalingResult from a worker output dict."""
    return OATNScalingResult(
        N=d["N"],
        n_ancilla=d["n_ancilla"],
        omega=d["omega"],
        delta_omega_opt=d["delta_omega_opt"],
        sql=d["sql"],
        ratio=d["ratio"],
        a_x_opt=d["a_x_opt"],
        a_y_opt=d["a_y_opt"],
        a_z_opt=d["a_z_opt"],
        a_zz_opt=d["a_zz_opt"],
        q_opt=d["q_opt"],
        delta_omega_no_oat=d["delta_omega_no_oat"],
        improvement_factor=d["improvement_factor"],
        expectation_Jz=d["expectation_Jz"],
        variance_Jz=d["variance_Jz"],
        success=bool(d["success"]),
        nfev=d["nfev"],
    )


def _oat_group_key(item: tuple[int, int, float]) -> tuple[int, int]:
    """Group by (N_S, N_A) pair."""
    return (item[0], item[1])


def _make_oat_checkpoint_name(checkpoint_dir: Path) -> Callable:
    """Factory: returns function mapping (N_S, N_A) -> checkpoint Path."""

    def _name_fn(group_key: Any) -> Path:
        return (
            checkpoint_dir
            / f"N_{int(group_key[0]):03d}_NA_{int(group_key[1]):03d}.parquet"
        )

    return _name_fn


def _collect_oat_scan_items(
    include_phase1: bool,
    include_phase2: bool,
    completed: set[tuple[int, int, float]],
) -> list[tuple[int, int, float]]:
    """Collect (N_S, N_A, omega) triples that have not yet been completed.

    Args:
        include_phase1: Whether to include Phase 1 (asymmetric) runs.
        include_phase2: Whether to include Phase 2 (symmetric) runs.
        completed: Set of already-completed (N_S, N_A, omega) triples.

    Returns:
        List of (N_S, N_A, omega) triples still needing execution.
    """
    items: list[tuple[int, int, float]] = []
    if include_phase1:
        items.extend(
            (PHASE1_N_SYSTEM, PHASE1_N_ANCILLA, omega)
            for omega in OMEGA_VALS
            if (PHASE1_N_SYSTEM, PHASE1_N_ANCILLA, omega) not in completed
        )
    if include_phase2:
        items.extend(
            (N, N, omega)
            for N in PHASE2_N_VALS
            for omega in OMEGA_VALS
            if (N, N, omega) not in completed
        )
    return items


def _generate_oat_scan_figures(summary: OATNScalingScanResult) -> None:
    """Generate ratio, sensitivity, and optimal-parameter figures.

    Args:
        summary: The completed OAT N-scaling scan result.
    """
    df = summary.to_dataframe()
    if df.empty:
        print("[skip] No data to plot")
        return
    plot_n_scaling_ratio(df, _fig_path("oat-n-scaling-ratio"))
    print(f"[fig]  {_fig_path('oat-n-scaling-ratio')}")
    plot_n_scaling_sensitivity(
        df, _fig_path("oat-n-scaling-sensitivity"), t_hold=T_HOLD
    )
    print(f"[fig]  {_fig_path('oat-n-scaling-sensitivity')}")
    plot_n_scaling_optimal_params(df, _fig_path("oat-n-scaling-optimal-params"))
    print(f"[fig]  {_fig_path('oat-n-scaling-optimal-params')}")


def generate_oat_n_scaling_scan(
    force: bool = False,
    include_phase1: bool = True,
    include_phase2: bool = True,
) -> None:
    """Full OAT N-scaling scan.

    Phase 1: N_S=1, N_A=2 (asymmetric), 1 x 5 omega values = 5 optimisation runs.
    Phase 2: N_S = 2..10, N_A = N_S (symmetric), 9 x 5 = 45 optimisation runs.

    Each run:
      1. 5D random search with 1000 points.
      2. Nelder-Mead refinement from top 50 points.

    Results are saved to a single Parquet file. Intermediate checkpoints
    are saved per-(N_S, N_A) value.

    Args:
        force: Re-run even if Parquet exists.
        include_phase1: Whether to include Phase 1 (asymmetric) runs.
        include_phase2: Whether to include Phase 2 (symmetric) runs.
    """
    parquet_p = _parquet_path("oat-n-scaling-scan")
    checkpoint_dir = REPORTS_DIR / REPORT_DATE / "raw_data" / "checkpoints"

    if parquet_p.exists() and not force:
        print(f"[skip] {parquet_p.name} exists (use --force to overwrite)")
        summary = OATNScalingScanResult.from_parquet(parquet_p)
    else:
        if force:
            parquet_p.unlink(missing_ok=True)
            if checkpoint_dir.exists():
                import shutil

                shutil.rmtree(checkpoint_dir)

        completed, checkpoint_results = load_checkpoints(
            checkpoint_dir,
            _build_oat_result_from_row,
            ["N", "n_ancilla", "omega"],
        )
        items_to_run = _collect_oat_scan_items(
            include_phase1, include_phase2, completed
        )

        if items_to_run:
            print(
                f"[run] OAT N-scaling scan: {len(items_to_run)} "
                f"remaining (N_S, N_A, omega) triples",
            )
            print(
                f"  (batch by (N_S, N_A) group, {min(32, os.cpu_count() or 1)} workers)"
            )
            new_results = run_pending_groups(
                items_to_run,
                checkpoint_dir,
                _run_single_oat_for_parallel,
                _build_oat_result_from_dict,
                _oat_group_key,
                _make_oat_checkpoint_name(checkpoint_dir),
                OATNScalingScanResult,
            )
            checkpoint_results.extend(new_results)
        else:
            print("  [skip] all triples already completed in checkpoints")

        summary = OATNScalingScanResult(results=checkpoint_results)
        summary.save_parquet(parquet_p)
        print(f"[save] {parquet_p}")

    _generate_oat_scan_figures(summary)


def _build_decoupled_baseline_df() -> pd.DataFrame:
    """Build a DataFrame of decoupled baseline verification results."""
    verifications = verify_oat_decoupled_baseline()
    results_list: list[dict[str, float | int | str]] = []
    for (N_S, N_A, omega), passed in verifications.items():
        sql_ref = sql_reference(N_S)
        delta = compute_oat_decoupled_baseline(N_S, N_A, omega)
        results_list.append(
            {
                "N": N_S,
                "n_ancilla": N_A,
                "omega": omega,
                "delta_omega": delta,
                "sql": sql_ref,
                "ratio": delta / sql_ref if sql_ref > 0 else float("nan"),
                "pass": str(passed),
            },
        )
    return pd.DataFrame(results_list)


# ============================================================================
# CLI Entry Point
# ============================================================================


def main() -> None:
    """CLI entry point for generating all data and figures."""
    parser = argparse.ArgumentParser(
        description="Ancilla OAT Pre-Squeezing before omega-Modulated Hold (20260619)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run all simulations even if Parquet files exist",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Run only a specific generator (e.g., 'decoupled-baseline')",
    )
    parser.add_argument(
        "--skip-phase1",
        action="store_true",
        help="Skip Phase 1 (asymmetric J_S=1/2, J_A=1) runs",
    )
    parser.add_argument(
        "--skip-phase2",
        action="store_true",
        help="Skip Phase 2 (symmetric J_A=N/2) runs",
    )
    args = parser.parse_args()

    force = args.force
    include_phase1 = not args.skip_phase1
    include_phase2 = not args.skip_phase2

    generators: dict[str, tuple[str, str | Callable[..., None], bool]] = {
        "decoupled-baseline": (
            "Decoupled Baseline Verification",
            lambda force=False: generate_decoupled_baseline(
                force=force,
                parquet_path=_parquet_path("decoupled-baseline"),
                compute_fn=_build_decoupled_baseline_df,
                label="OAT decoupled baseline",
            ),
            False,
        ),
        "oat-n-scaling-scan": (
            "OAT N-Scaling Full Scan",
            "generate_oat_n_scaling_scan",
            True,
        ),
    }

    if args.only:
        if args.only not in generators:
            print(f"Unknown generator: {args.only}")
            print(f"Available: {list(generators.keys())}")
            import sys

            sys.exit(1)
        gen_list = [args.only]
    else:
        gen_list = list(generators.keys())

    # Ensure raw_data and figures directories exist
    for d in ["raw_data", "figures"]:
        (REPORTS_DIR / REPORT_DATE / d).mkdir(parents=True, exist_ok=True)

    for key in gen_list:
        name, func_or_name, _ = generators[key]
        print(f"\n{'=' * 60}")
        print(f"  {name}")
        print(f"{'=' * 60}")
        func: Callable[..., None] = (
            globals()[func_or_name] if isinstance(func_or_name, str) else func_or_name
        )
        if key == "oat-n-scaling-scan":
            func(
                force=force,
                include_phase1=include_phase1,
                include_phase2=include_phase2,
            )
        else:
            func(force=force)


if __name__ == "__main__":
    main()
