"""
Local module for the 2026-06-11 N-Scaling of Phase-Modulated Ancilla Drive report.

Extends the ω-modulated ancilla drive mechanism (20260519) to N > 1 system
particles (J_S = N/2) while keeping the ancilla at J_A = 1/2.

Tests whether the 4.91× SQL-violation ratio at N=1 improves with N or saturates.

Operator construction:
- System: (N+1)-dimensional Dicke basis for N particles.
- Ancilla: 2-dimensional single-qubit space.
- Full space: 2(N+1) dimensions with basis ordering:
    {|m_S⟩_S ⊗ |0⟩_A, ... , |m_S⟩_S ⊗ |1⟩_A}  (m_S descending from +J_S to -J_S)

Circuit: BS_S → Hold → BS_S → measure J_z^S.

Usage:
    uv run python reports/20260611/local.py --force
"""

from __future__ import annotations

import argparse
import concurrent.futures
import multiprocessing as _mp
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.linalg import expm
from scipy.optimize import minimize

# Force non-interactive matplotlib backend before any plotting.
if "MPLBACKEND" not in os.environ:
    os.environ["MPLBACKEND"] = "Agg"
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "1"

from src.analysis.ancilla_drive_metrology import (
    DriveNelderMeadResult,
    DriveRandomSearchResult,
)
from src.analysis.ancilla_optimization import (
    compute_expectation_and_variance,
)
from src.physics.beam_splitter import bs_dicke
from src.physics.dicke_basis import jx_operator, jy_operator, jz_operator
from src.utils.constants import I_2, J_X, J_Y, J_Z
from src.utils.enums import OperatorBasis

sns.set_theme(style="whitegrid")

# ============================================================================
# Constants
# ============================================================================

T_HOLD: float = 10.0  # Holding time (fixed)
T_BS: float = np.pi / 2.0  # 50/50 beam splitter
FD_STEP: float = 1e-6  # Finite-difference step
DRIVE_BOUNDS: tuple[float, float] = (-5.0, 5.0)  # Parameter bounds

# ω values for the scan
OMEGA_VALS: list[float] = [0.1, 0.2, 0.5, 1.0, 2.0]

# N values for the scaling scan (1 to 20)
N_VALS: list[int] = list(range(1, 21))

# Random search parameters
N_RANDOM: int = 500
N_NM_REFINE: int = 50
NM_MAXITER: int = 5000


def sql_reference(N: int) -> float:
    """Standard quantum limit for N particles with holding time T_HOLD.

    Δω_SQL = 1 / (√N × T_HOLD)

    Args:
        N: Number of system particles.

    Returns:
        SQL sensitivity value.
    """
    return 1.0 / (np.sqrt(N) * T_HOLD)


# ============================================================================
# Path Helpers
# ============================================================================

REPORTS_DIR = Path(__file__).resolve().parent.parent.parent / "reports"
REPORT_DATE = "20260611"


def _parquet_path(name: str) -> Path:
    return REPORTS_DIR / REPORT_DATE / "raw_data" / f"{REPORT_DATE}-{name}.parquet"


def _fig_path(name: str) -> Path:
    return REPORTS_DIR / REPORT_DATE / "figures" / f"{REPORT_DATE}-{name}.svg"


# ============================================================================
# Operator Construction
# ============================================================================


def build_n_particle_operators(N: int) -> dict[str, np.ndarray]:
    """Build operators in the 2(N+1)-dimensional total Hilbert space.

    Total space: H_S ⊗ H_A with dimension 2(N+1).
    Basis ordering: {|m_S⟩_S ⊗ |0⟩_A, ..., |m_S⟩_S ⊗ |1⟩_A}
    where m_S descends from +J_S to -J_S, and |0⟩_A = |1,0⟩, |1⟩_A = |0,1⟩.

    Args:
        N: Number of system particles (N ≥ 1). System dim = N+1, ancilla dim = 2.

    Returns:
        Dict with keys 'Jz_S', 'Jx_S', 'Jy_S', 'Jz_A', 'Jx_A', 'Jy_A',
        each a 2(N+1) × 2(N+1) complex Hermitian matrix.
        Also includes 'I_S' (N+1)×(N+1) identity and 'I_full'.
    """
    if N < 1:
        raise ValueError(f"N must be ≥ 1, got {N}")

    d_sys = N + 1  # System Hilbert space dimension
    d_tot = 2 * d_sys  # Total Hilbert space dimension

    # System operators in Dicke basis
    Jz_S_dicke = jz_operator(N, basis=OperatorBasis.DICKE)
    Jx_S_dicke = jx_operator(N, basis=OperatorBasis.DICKE)
    Jy_S_dicke = jy_operator(N, basis=OperatorBasis.DICKE)
    I_S = np.eye(d_sys, dtype=complex)

    # Embed into total space via Kronecker products
    ops: dict[str, np.ndarray] = {
        # System: A ⊗ I_2
        "Jz_S": np.kron(Jz_S_dicke, I_2).astype(complex),
        "Jx_S": np.kron(Jx_S_dicke, I_2).astype(complex),
        "Jy_S": np.kron(Jy_S_dicke, I_2).astype(complex),
        # Ancilla: I_{N+1} ⊗ J_k  (J_k = σ_k/2)
        "Jz_A": np.kron(I_S, J_Z).astype(complex),
        "Jx_A": np.kron(I_S, J_X).astype(complex),
        "Jy_A": np.kron(I_S, J_Y).astype(complex),
        # Identities
        "I_S": I_S,
        "I_full": np.eye(d_tot, dtype=complex),
    }

    # Validate dimensions
    for key in ("Jz_S", "Jx_S", "Jy_S", "Jz_A", "Jx_A", "Jy_A"):
        assert ops[key].shape == (d_tot, d_tot), (
            f"{key} has shape {ops[key].shape}, expected ({d_tot}, {d_tot})"
        )

    # Validate Hermiticity
    for key in ("Jz_S", "Jx_S", "Jy_S", "Jz_A", "Jx_A", "Jy_A"):
        assert np.allclose(ops[key], ops[key].conj().T, atol=1e-12), (
            f"{key} is not Hermitian"
        )

    # Validate commutation relations: [J_z^S, J_x^S] = i J_y^S
    comm = ops["Jz_S"] @ ops["Jx_S"] - ops["Jx_S"] @ ops["Jz_S"]
    expected = 1j * ops["Jy_S"]
    assert np.allclose(comm, expected, atol=1e-10), (
        f"[J_z^S, J_x^S] = i J_y^S violated for N={N}"
    )

    return ops


def build_n_particle_system_only_bs_unitary(N: int, T_bs: float = T_BS) -> np.ndarray:
    """System-only beam-splitter unitary in the N-particle space.

    U_BS_S = exp(-i T_bs J_x) ⊗ I_2  (acts on system, identity on ancilla).

    Args:
        N: Number of system particles.
        T_bs: Beam-splitter duration (default π/2 for 50/50).

    Returns:
        2(N+1) × 2(N+1) unitary matrix.
    """
    d_tot = 2 * (N + 1)
    bs_sys = bs_dicke(N, T_bs)
    U = np.kron(bs_sys, I_2).astype(complex)
    I_full = np.eye(d_tot, dtype=complex)
    assert np.allclose(U @ U.conj().T, I_full, atol=1e-12), (
        f"BS unitary not unitary for N={N}, T_bs={T_bs}"
    )
    return U


def build_n_particle_phase_modulated_drive_hamiltonian(
    N: int,
    omega: float,
    a_x: float,
    a_y: float,
    a_z: float,
    ops: dict[str, np.ndarray],
) -> np.ndarray:
    """Build the ω-modulated ancilla drive Hamiltonian in the N-particle space.

    H_A = ω (a_x J_x^A + a_y J_y^A + a_z J_z^A)

    Args:
        N: Number of system particles (for dimension check).
        omega: Phase rate parameter (scales the whole drive).
        a_x: J_x^A coefficient.
        a_y: J_y^A coefficient.
        a_z: J_z^A coefficient.
        ops: Operators from build_n_particle_operators(N).

    Returns:
        2(N+1) × 2(N+1) Hermitian matrix.
    """
    d_tot = 2 * (N + 1)
    H = np.zeros((d_tot, d_tot), dtype=complex)
    if a_x != 0.0:
        H += a_x * ops["Jx_A"]
    if a_y != 0.0:
        H += a_y * ops["Jy_A"]
    if a_z != 0.0:
        H += a_z * ops["Jz_A"]
    H = omega * H
    # Enforce Hermiticity
    H = 0.5 * (H + H.conj().T)
    assert np.allclose(H, H.conj().T, atol=1e-12), (
        f"Drive Hamiltonian not Hermitian for N={N}"
    )
    return H


def build_n_particle_iszz_interaction(
    N: int,
    a_zz: float,
    ops: dict[str, np.ndarray],
) -> np.ndarray:
    """Build the Ising interaction in the N-particle space.

    H_int = a_zz J_z^S ⊗ J_z^A = a_zz (J_z ⊗ I_2) @ (I_{N+1} ⊗ J_z)

    Args:
        N: Number of system particles.
        a_zz: Interaction coupling coefficient.
        ops: Operators from build_n_particle_operators(N).

    Returns:
        2(N+1) × 2(N+1) Hermitian matrix.
    """
    d_tot = 2 * (N + 1)
    H = np.zeros((d_tot, d_tot), dtype=complex)
    if a_zz != 0.0:
        H += a_zz * (ops["Jz_S"] @ ops["Jz_A"])
    return H


def build_n_particle_hold_hamiltonian(
    N: int,
    omega: float,
    a_x: float,
    a_y: float,
    a_z: float,
    a_zz: float,
    ops: dict[str, np.ndarray],
) -> np.ndarray:
    """Build the total holding Hamiltonian for the N-particle system.

    H = ω J_z^S + ω (a_x J_x^A + a_y J_y^A + a_z J_z^A) + a_zz J_z^S ⊗ J_z^A

    Args:
        N: Number of system particles.
        omega: Phase rate parameter.
        a_x: J_x^A drive coefficient.
        a_y: J_y^A drive coefficient.
        a_z: J_z^A drive coefficient.
        a_zz: Ising interaction coefficient.
        ops: Operators from build_n_particle_operators(N).

    Returns:
        2(N+1) × 2(N+1) Hermitian Hamiltonian matrix.
    """
    H = omega * ops["Jz_S"]
    H += build_n_particle_phase_modulated_drive_hamiltonian(
        N,
        omega,
        a_x,
        a_y,
        a_z,
        ops,
    )
    H += build_n_particle_iszz_interaction(N, a_zz, ops)
    # Enforce Hermiticity
    H = 0.5 * (H + H.conj().T)
    assert np.allclose(H, H.conj().T, atol=1e-12), (
        f"Total Hamiltonian not Hermitian for N={N}"
    )
    return H


# ============================================================================
# State Preparation
# ============================================================================


def n_particle_initial_state(N: int) -> np.ndarray:
    """Initial state for the N-particle system.

    |Ψ₀⟩ = |J_S, J_S⟩_S ⊗ |1,0⟩_A

    In the full basis, this is the first basis vector: [1, 0, ..., 0]ᵀ
    of length 2(N+1).

    Args:
        N: Number of system particles.

    Returns:
        Normalised complex vector of length 2(N+1).
    """
    d_tot = 2 * (N + 1)
    psi = np.zeros(d_tot, dtype=complex)
    psi[0] = 1.0
    assert np.isclose(np.linalg.norm(psi), 1.0), (
        f"Initial state not normalised for N={N}"
    )
    return psi


# ============================================================================
# Circuit Evolution
# ============================================================================


def n_particle_hold_unitary(
    N: int,
    T_hold: float,
    omega: float,
    a_x: float,
    a_y: float,
    a_z: float,
    a_zz: float,
    ops: dict[str, np.ndarray],
) -> np.ndarray:
    """Holding-time unitary for the N-particle ω-modulated protocol.

    U_hold(T_hold) = exp(-i T_hold H)
    where H = ω J_z^S + ω(a_x J_x^A + a_y J_y^A + a_z J_z^A) + a_zz J_z^S ⊗ J_z^A.

    Args:
        N: Number of system particles.
        T_hold: Holding-time strength.
        omega: Phase rate parameter.
        a_x: J_x^A drive coefficient.
        a_y: J_y^A drive coefficient.
        a_z: J_z^A drive coefficient.
        a_zz: Ising interaction coefficient.
        ops: Operators from build_n_particle_operators(N).

    Returns:
        2(N+1) × 2(N+1) unitary matrix.
    """
    H = build_n_particle_hold_hamiltonian(N, omega, a_x, a_y, a_z, a_zz, ops)
    U = expm(-1j * T_hold * H)
    d_tot = 2 * (N + 1)
    I_full = np.eye(d_tot, dtype=complex)
    assert np.allclose(U @ U.conj().T, I_full, atol=1e-12), (
        f"Hold unitary not unitary for N={N}, T_hold={T_hold}, ω={omega}"
    )
    return U


def evolve_n_particle_circuit(
    N: int,
    psi0: np.ndarray,
    T_bs: float,
    T_hold: float,
    omega: float,
    a_x: float,
    a_y: float,
    a_z: float,
    a_zz: float,
    ops: dict[str, np.ndarray],
) -> np.ndarray:
    """Run the full N-particle ω-modulated ancilla MZI circuit.

    |ψ_final⟩ = U_BS_S · U_hold(T_hold) · U_BS_S · |ψ₀⟩

    Args:
        N: Number of system particles.
        psi0: Initial state vector (must be normalised).
        T_bs: Beam-splitter duration.
        T_hold: Holding-time strength.
        omega: Phase rate parameter.
        a_x: J_x^A drive coefficient.
        a_y: J_y^A drive coefficient.
        a_z: J_z^A drive coefficient.
        a_zz: Ising interaction coefficient.
        ops: Operators from build_n_particle_operators(N).

    Returns:
        Final normalised state vector.
    """
    assert np.isclose(np.linalg.norm(psi0), 1.0), (
        f"Initial state not normalised for N={N}"
    )
    U_bs = build_n_particle_system_only_bs_unitary(N, T_bs)
    psi = U_bs @ psi0
    psi = n_particle_hold_unitary(N, T_hold, omega, a_x, a_y, a_z, a_zz, ops) @ psi
    psi = U_bs @ psi
    assert np.isclose(np.linalg.norm(psi), 1.0), f"Final state not normalised for N={N}"
    return psi


def compute_n_particle_sensitivity(
    N: int,
    psi0: np.ndarray,
    T_bs: float,
    T_hold: float,
    omega_true: float,
    a_x: float,
    a_y: float,
    a_z: float,
    a_zz: float,
    ops: dict[str, np.ndarray],
    fd_step: float = FD_STEP,
    meas_op: np.ndarray | None = None,
) -> float:
    """Compute the error-propagation sensitivity Δω for the N-particle system.

    Δω = sqrt(Var(O)) / |∂⟨O⟩/∂ω|

    where O = J_z^S (default measurement operator).

    The central finite-difference captures the full ω-dependence
    (both ω J_z^S and ω-modulated ancilla drive channels).

    Args:
        N: Number of system particles.
        psi0: Initial state vector.
        T_bs: Beam-splitter duration.
        T_hold: Holding-time strength.
        omega_true: True phase rate parameter.
        a_x: J_x^A drive coefficient.
        a_y: J_y^A drive coefficient.
        a_z: J_z^A drive coefficient.
        a_zz: Ising interaction coefficient.
        ops: Operators from build_n_particle_operators(N).
        fd_step: Finite-difference step size (default 1e-6).
        meas_op: Measurement operator (default ops['Jz_S']).

    Returns:
        Sensitivity Δω. Returns inf if derivative is zero (fringe extremum).
    """
    if meas_op is None:
        meas_op = ops["Jz_S"]

    # Evaluate at omega_true
    psi = evolve_n_particle_circuit(
        N,
        psi0,
        T_bs,
        T_hold,
        omega_true,
        a_x,
        a_y,
        a_z,
        a_zz,
        ops,
    )
    _, var = compute_expectation_and_variance(psi, meas_op)

    # Central finite difference for ∂⟨O⟩/∂ω
    psi_plus = evolve_n_particle_circuit(
        N,
        psi0,
        T_bs,
        T_hold,
        omega_true + fd_step,
        a_x,
        a_y,
        a_z,
        a_zz,
        ops,
    )
    psi_minus = evolve_n_particle_circuit(
        N,
        psi0,
        T_bs,
        T_hold,
        omega_true - fd_step,
        a_x,
        a_y,
        a_z,
        a_zz,
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
# Dataclasses
# ============================================================================


@dataclass
class NScalingResult:
    """Result from optimising the ω-modulated protocol for a single (N, ω) pair.

    Attributes:
        N: Number of system particles.
        omega: Phase rate value.
        delta_omega_opt: Best sensitivity Δω found.
        sql: SQL = 1/(√N × T_HOLD).
        ratio: SQL / Δω_opt (ratio > 1 means beating SQL).
        a_x_opt: Optimal J_x^A drive coefficient.
        a_y_opt: Optimal J_y^A drive coefficient.
        a_z_opt: Optimal J_z^A drive coefficient.
        a_zz_opt: Optimal Ising interaction coefficient.
        expectation_Jz: ⟨J_z^S⟩ at the optimal operating point.
        variance_Jz: Var(J_z^S) at the optimal operating point.
        T_hold: Holding time (fixed at 10.0).
        fd_step: Finite-difference step.
        success: Whether Nelder-Mead reported success.
        nfev: Number of function evaluations.
    """

    N: int
    omega: float
    delta_omega_opt: float
    sql: float
    ratio: float
    a_x_opt: float
    a_y_opt: float
    a_z_opt: float
    a_zz_opt: float
    expectation_Jz: float = 0.0
    variance_Jz: float = 0.0
    T_hold: float = T_HOLD
    fd_step: float = FD_STEP
    success: bool = False
    nfev: int = 0

    _PARQUET_COLUMNS: ClassVar[list[str]] = [
        "N",
        "omega",
        "delta_omega_opt",
        "sql",
        "ratio",
        "a_x_opt",
        "a_y_opt",
        "a_z_opt",
        "a_zz_opt",
        "expectation_Jz",
        "variance_Jz",
        "T_hold",
        "fd_step",
        "success",
        "nfev",
    ]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "N": [self.N],
                "omega": [self.omega],
                "delta_omega_opt": [self.delta_omega_opt],
                "sql": [self.sql],
                "ratio": [self.ratio],
                "a_x_opt": [self.a_x_opt],
                "a_y_opt": [self.a_y_opt],
                "a_z_opt": [self.a_z_opt],
                "a_zz_opt": [self.a_zz_opt],
                "expectation_Jz": [self.expectation_Jz],
                "variance_Jz": [self.variance_Jz],
                "T_hold": [self.T_hold],
                "fd_step": [self.fd_step],
                "success": [int(self.success)],
                "nfev": [self.nfev],
            },
        )

    def save_parquet(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_parquet(path, index=False)
        return path

    @classmethod
    def from_parquet(cls, path: str | Path) -> NScalingResult:
        path = Path(path)
        df = pd.read_parquet(path)
        missing = {c for c in cls._PARQUET_COLUMNS if c not in df.columns}
        if missing:
            raise ValueError(
                f"Parquet at {path} is missing required columns: "
                f"{sorted(missing)}. Re-run the simulation that generated it."
            )
        row = df.iloc[0]
        return cls(
            N=int(row["N"]),
            omega=float(row["omega"]),
            delta_omega_opt=float(row["delta_omega_opt"]),
            sql=float(row["sql"]),
            ratio=float(row["ratio"]),
            a_x_opt=float(row["a_x_opt"]),
            a_y_opt=float(row["a_y_opt"]),
            a_z_opt=float(row["a_z_opt"]),
            a_zz_opt=float(row["a_zz_opt"]),
            expectation_Jz=float(row["expectation_Jz"]),
            variance_Jz=float(row["variance_Jz"]),
            T_hold=float(row["T_hold"]),
            fd_step=float(row["fd_step"]),
            success=bool(int(row["success"])),
            nfev=int(row["nfev"]),
        )


@dataclass
class NScalingScanResult:
    """Collection of N-scaling results for a grid of (N, ω) pairs.

    Attributes:
        results: List of per-(N, ω) NScalingResult.
        N_values: Array of N values (sorted).
        omega_values: Array of ω values (sorted).
    """

    results: list[NScalingResult] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        if not self.results:
            return pd.DataFrame(columns=NScalingResult._PARQUET_COLUMNS)
        return pd.concat([r.to_dataframe() for r in self.results], ignore_index=True)

    def save_parquet(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_parquet(path, index=False)
        return path

    @classmethod
    def from_parquet(cls, path: str | Path) -> NScalingScanResult:
        path = Path(path)
        df = pd.read_parquet(path)
        required = set(NScalingResult._PARQUET_COLUMNS)
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Parquet at {path} is missing required columns: "
                f"{sorted(missing)}. Re-run the simulation that generated it."
            )
        results: list[NScalingResult] = []
        for _, row in df.iterrows():
            results.append(
                NScalingResult(
                    N=int(row["N"]),
                    omega=float(row["omega"]),
                    delta_omega_opt=float(row["delta_omega_opt"]),
                    sql=float(row["sql"]),
                    ratio=float(row["ratio"]),
                    a_x_opt=float(row["a_x_opt"]),
                    a_y_opt=float(row["a_y_opt"]),
                    a_z_opt=float(row["a_z_opt"]),
                    a_zz_opt=float(row["a_zz_opt"]),
                    expectation_Jz=float(row["expectation_Jz"]),
                    variance_Jz=float(row["variance_Jz"]),
                    T_hold=float(row["T_hold"]),
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
# Decoupled Baseline
# ============================================================================


def compute_n_particle_decoupled_baseline(
    N: int,
    omega_true: float = 1.0,
) -> float:
    """Compute the decoupled baseline sensitivity Δω for N particles.

    At (a_x = a_y = a_z = a_zz = 0), the circuit reduces to a standard
    N-particle MZI with CSS input, giving Δω = 1/(√N × T_HOLD).

    Args:
        N: Number of system particles.
        omega_true: Phase rate value.

    Returns:
        Δω at the decoupled configuration.
    """
    ops = build_n_particle_operators(N)
    psi0 = n_particle_initial_state(N)
    return compute_n_particle_sensitivity(
        N,
        psi0,
        T_BS,
        T_HOLD,
        omega_true,
        0.0,
        0.0,
        0.0,
        0.0,
        ops,
    )


def verify_decoupled_baseline(
    N_values: list[int] | None = None,
    omega_values: list[float] | None = None,
    rtol: float = 1e-10,
) -> dict[tuple[int, float], bool]:
    """Verify the decoupled baseline for all (N, ω) pairs.

    At zero drive and zero interaction, the sensitivity must equal
    Δω = 1/(√N × T_HOLD) to machine precision.

    Args:
        N_values: List of N values (default: 1 to 20).
        omega_values: List of ω values (default: all OMEGA_VALS).
        rtol: Relative tolerance for comparison.

    Returns:
        Dict mapping (N, ω) → PASS/FAIL (True/False).
    """
    if N_values is None:
        N_values = N_VALS
    if omega_values is None:
        omega_values = OMEGA_VALS
    results: dict[tuple[int, float], bool] = {}
    for N in N_values:
        sql_ref = sql_reference(N)
        for omega in omega_values:
            delta = compute_n_particle_decoupled_baseline(N, omega)
            results[(N, omega)] = bool(np.isclose(delta, sql_ref, rtol=rtol))
    return results


# ============================================================================
# 4D Random Search
# ============================================================================


def n_particle_random_search(
    N: int,
    omega: float,
    n_samples: int = N_RANDOM,
    bounds: tuple[float, float] = DRIVE_BOUNDS,
    seed: int | None = 42,
) -> DriveRandomSearchResult:
    """Random search over the 4D parameter space (a_x, a_y, a_z, a_zz).

    Args:
        N: Number of system particles.
        omega: Phase rate value.
        n_samples: Number of random points to evaluate.
        bounds: (min, max) for all four coefficients.
        seed: Random seed for reproducibility.

    Returns:
        DriveRandomSearchResult with all samples and best found.
    """
    rng = np.random.default_rng(seed)
    ops = build_n_particle_operators(N)
    psi0 = n_particle_initial_state(N)
    lo, hi = bounds

    samples = rng.uniform(lo, hi, size=(n_samples, 4))
    deltas = np.full(n_samples, np.inf, dtype=float)

    for i in range(n_samples):
        ax = float(samples[i, 0])
        ay = float(samples[i, 1])
        az = float(samples[i, 2])
        azz = float(samples[i, 3])
        domega = compute_n_particle_sensitivity(
            N,
            psi0,
            T_BS,
            T_HOLD,
            omega,
            ax,
            ay,
            az,
            azz,
            ops,
        )
        deltas[i] = domega

    best_idx = int(np.argmin(deltas))
    best_params: tuple[float, float, float, float] = (
        float(samples[best_idx, 0]),
        float(samples[best_idx, 1]),
        float(samples[best_idx, 2]),
        float(samples[best_idx, 3]),
    )

    return DriveRandomSearchResult(
        samples=samples,
        delta_omega_values=deltas,
        best_params=best_params,
        best_delta_omega=float(deltas[best_idx]),
        omega_value=omega,
        sql=sql_reference(N),
        T_hold=T_HOLD,
    )


# ============================================================================
# Nelder-Mead Optimisation
# ============================================================================


def n_particle_sensitivity_objective(
    params: np.ndarray,
    N: int,
    omega_true: float,
    ops: dict[str, np.ndarray],
    psi0: np.ndarray,
    T_hold: float = T_HOLD,
    T_bs: float = T_BS,
    fd_step: float = FD_STEP,
    bounds: tuple[float, float] = DRIVE_BOUNDS,
    penalty_scale: float = 1e6,
) -> float:
    """Objective function for minimising Δω in the N-particle protocol.

    params = [a_x, a_y, a_z, a_zz] (4 elements).

    Args:
        params: 4-element parameter vector.
        N: Number of system particles.
        omega_true: True phase rate.
        ops: N-particle operators.
        psi0: Initial state vector.
        T_hold: Holding time.
        T_bs: Beam-splitter duration.
        fd_step: Finite-difference step.
        bounds: (min, max) for all parameters.
        penalty_scale: Scale for bound-violation penalty.

    Returns:
        Δω (plus infinite penalty if bounds violated).
    """
    ax = float(params[0])
    ay = float(params[1])
    az = float(params[2])
    azz = float(params[3])

    # Bound enforcement
    lo, hi = bounds
    penalty = 0.0
    for val in (ax, ay, az, azz):
        if val < lo:
            penalty += penalty_scale * (lo - val) ** 2
        if val > hi:
            penalty += penalty_scale * (val - hi) ** 2

    if penalty > 0.0:
        return float(1e10 + penalty)

    return compute_n_particle_sensitivity(
        N,
        psi0,
        T_bs,
        T_hold,
        omega_true,
        ax,
        ay,
        az,
        azz,
        ops,
        fd_step,
    )


def run_n_particle_nelder_mead(
    N: int,
    omega_true: float,
    ops: dict[str, np.ndarray] | None = None,
    psi0: np.ndarray | None = None,
    x0: np.ndarray | None = None,
    seed: int | None = None,
    maxiter: int = NM_MAXITER,
    xatol: float = 1e-8,
    fatol: float = 1e-8,
    adaptive: bool = True,
    bounds: tuple[float, float] = DRIVE_BOUNDS,
    track_history: bool = False,
) -> DriveNelderMeadResult:
    """Run Nelder-Mead optimisation for the N-particle ω-modulated protocol.

    Args:
        N: Number of system particles.
        omega_true: True phase rate parameter.
        ops: N-particle operators (built if None).
        psi0: Initial state (built if None).
        x0: Initial 4-parameter vector [ax, ay, az, azz]. Random if None.
        seed: Random seed (used if x0 is None).
        maxiter: Maximum Nelder-Mead iterations.
        xatol: Absolute parameter tolerance.
        fatol: Absolute function tolerance.
        adaptive: Use adaptive Nelder-Mead parameters.
        bounds: (min, max) for all four parameters.
        track_history: If True, record objective values per iteration.

    Returns:
        DriveNelderMeadResult.
    """
    if ops is None:
        ops = build_n_particle_operators(N)
    if psi0 is None:
        psi0 = n_particle_initial_state(N)

    if x0 is None:
        rng = np.random.default_rng(seed)
        lo, hi = bounds
        x0 = rng.uniform(lo, hi, size=4)
    else:
        x0 = np.asarray(x0, dtype=float)
        assert x0.shape == (4,), f"x0 must have 4 elements, got {x0.shape}"

    def objective(p: np.ndarray) -> float:
        return n_particle_sensitivity_objective(
            p,
            N,
            omega_true,
            ops,
            psi0,
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

    # Compute diagnostics at the optimal point
    psi_final = evolve_n_particle_circuit(
        N,
        psi0,
        T_BS,
        T_HOLD,
        omega_true,
        float(opt_params[0]),
        float(opt_params[1]),
        float(opt_params[2]),
        float(opt_params[3]),
        ops,
    )
    exp_val, var_val = compute_expectation_and_variance(psi_final, ops["Jz_S"])

    return DriveNelderMeadResult(
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
# N-Scaling Scan (Single (N, ω) Worker)
# ============================================================================


def run_single_n_omega(
    N: int,
    omega: float,
    seed: int | None = 42,
) -> NScalingResult:
    """Run the full optimisation pipeline for a single (N, ω) pair.

    1. 4D random search (500 samples).
    2. Nelder-Mead refinement from top 50 points.
    3. Return the best result.

    Args:
        N: Number of system particles.
        omega: Phase rate value.
        seed: Base random seed (incremented per call).

    Returns:
        NScalingResult with the optimal parameters and sensitivity.
    """
    base_seed = seed if seed is not None else 42
    ops = build_n_particle_operators(N)
    psi0 = n_particle_initial_state(N)

    # Stage 1: Random search
    rs_result = n_particle_random_search(
        N,
        omega,
        n_samples=N_RANDOM,
        seed=base_seed,
    )

    # Sort by Δω, take top N_NM_REFINE
    sorted_indices = np.argsort(rs_result.delta_omega_values)
    top_indices = sorted_indices[:N_NM_REFINE]

    # Stage 2: Nelder-Mead refinement from each top point
    nm_results: list[DriveNelderMeadResult] = []
    for rank, idx in enumerate(top_indices):
        x0 = rs_result.samples[idx].copy()
        nm = run_n_particle_nelder_mead(
            N=N,
            omega_true=omega,
            ops=ops,
            psi0=psi0,
            x0=x0,
            seed=base_seed + int(omega * 1000) + 10000 + rank,
        )
        nm_results.append(nm)

    nm_results.sort(key=lambda r: r.delta_omega_opt)
    best_nm = nm_results[0]

    sql_val = sql_reference(N)
    return NScalingResult(
        N=N,
        omega=omega,
        delta_omega_opt=best_nm.delta_omega_opt,
        sql=sql_val,
        ratio=sql_val / best_nm.delta_omega_opt
        if np.isfinite(best_nm.delta_omega_opt) and best_nm.delta_omega_opt > 0
        else float("nan"),
        a_x_opt=float(best_nm.params_opt[0]),
        a_y_opt=float(best_nm.params_opt[1]),
        a_z_opt=float(best_nm.params_opt[2]),
        a_zz_opt=float(best_nm.params_opt[3]),
        expectation_Jz=best_nm.expectation_Jz,
        variance_Jz=best_nm.variance_Jz,
        success=best_nm.success,
        nfev=best_nm.nfev,
    )


# ============================================================================
# Plot Functions
# ============================================================================


def plot_n_scaling_ratio(
    summary: NScalingScanResult,
    save_path: str | Path,
    figsize: tuple[float, float] = (10, 6),
) -> Path:
    """Plot SQL-violation ratio R(N) = SQL/Δω_opt vs N, coloured by ω.

    A horizontal line at R=1 indicates the SQL.
    The 20260519 result at N=1, ω=0.2 (R ≈ 4.91) is shown as a marker.

    Args:
        summary: NScalingScanResult with all (N, ω) results.
        save_path: Output SVG path.
        figsize: Figure size.

    Returns:
        Path to saved SVG.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    df = summary.to_dataframe()
    if df.empty:
        print("[skip] No data to plot.")
        return save_path

    fig, ax = plt.subplots(figsize=figsize)

    omega_values = sorted(df["omega"].unique())
    colours = plt.cm.viridis(np.linspace(0.2, 0.9, len(omega_values)))

    for omega_val, colour in zip(omega_values, colours, strict=False):
        sub = df[np.isclose(df["omega"], omega_val)]
        sub = sub.sort_values("N")
        valid = np.isfinite(sub["ratio"])
        if np.any(valid):
            ax.plot(
                sub["N"][valid],
                sub["ratio"][valid],
                "o-",
                color=colour,
                label=rf"$\omega={omega_val:.1f}$",
                markersize=6,
                linewidth=1.5,
            )

    # SQL reference line
    ax.axhline(
        y=1.0, color="gray", linestyle="--", alpha=0.7, linewidth=1.5, label="SQL (R=1)"
    )

    ax.set_xlabel(r"$N$ (system particles)")
    ax.set_ylabel(r"$R(N) = \Delta\omega_{\mathrm{SQL}} / \Delta\omega_{\mathrm{opt}}$")
    ax.set_title("SQL-violation ratio vs system particle number $N$")
    ax.legend(fontsize=9, title=r"$\omega$")
    ax.set_xlim(left=0.5)

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_n_scaling_sensitivity(
    summary: NScalingScanResult,
    save_path: str | Path,
    figsize: tuple[float, float] = (10, 6),
) -> Path:
    """Plot Δω_opt vs N on log-log axes, coloured by ω.

    SQL lines (Δω_SQL = 1/√N × T_HOLD) are shown for reference.
    The Heisenberg limit (Δω_HL ∝ 1/N) is also shown.

    Args:
        summary: NScalingScanResult with all (N, ω) results.
        save_path: Output SVG path.
        figsize: Figure size.

    Returns:
        Path to saved SVG.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    df = summary.to_dataframe()
    if df.empty:
        print("[skip] No data to plot.")
        return save_path

    fig, ax = plt.subplots(figsize=figsize)

    omega_values = sorted(df["omega"].unique())
    colours = plt.cm.viridis(np.linspace(0.2, 0.9, len(omega_values)))

    for omega_val, colour in zip(omega_values, colours, strict=False):
        sub = df[np.isclose(df["omega"], omega_val)]
        sub = sub.sort_values("N")
        valid = np.isfinite(sub["delta_omega_opt"])
        if np.any(valid):
            ax.loglog(
                sub["N"][valid],
                sub["delta_omega_opt"][valid],
                "o-",
                color=colour,
                label=rf"$\omega={omega_val:.1f}$",
                markersize=6,
                linewidth=1.5,
            )

    # SQL line: Δω_SQL = 1/(√N * T_HOLD)
    N_range = np.linspace(1, 20, 100)
    sql_line = 1.0 / (np.sqrt(N_range) * T_HOLD)
    ax.loglog(
        N_range,
        sql_line,
        "k--",
        alpha=0.7,
        linewidth=1.5,
        label=r"SQL $1/(\sqrt{N}T_H)$",
    )

    # Heisenberg limit: Δω_HL ∝ 1/N
    hl_line = 1.0 / (N_range * T_HOLD)
    ax.loglog(N_range, hl_line, "k:", alpha=0.5, linewidth=1.2, label=r"HL $1/(N T_H)$")

    ax.set_xlabel(r"$N$ (system particles)")
    ax.set_ylabel(r"$\Delta\omega_{\mathrm{opt}}$")
    ax.set_title("Optimal sensitivity vs system particle number $N$")
    ax.legend(fontsize=9, title=r"$\omega$")

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_n_scaling_optimal_params(
    summary: NScalingScanResult,
    save_path: str | Path,
    figsize: tuple[float, float] = (12, 8),
) -> Path:
    """Plot optimal parameters (a_x*, a_y*, a_z*, a_zz*) vs N.

    One panel per parameter, coloured by ω.

    Args:
        summary: NScalingScanResult with all (N, ω) results.
        save_path: Output SVG path.
        figsize: Figure size.

    Returns:
        Path to saved SVG.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    df = summary.to_dataframe()
    if df.empty:
        print("[skip] No data to plot.")
        return save_path

    params = [
        ("a_x_opt", r"$a_x^*$"),
        ("a_y_opt", r"$a_y^*$"),
        ("a_z_opt", r"$a_z^*$"),
        ("a_zz_opt", r"$a_{zz}^*$"),
    ]
    omega_values = sorted(df["omega"].unique())
    colours = plt.cm.viridis(np.linspace(0.2, 0.9, len(omega_values)))

    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True)

    for ax, (param_col, param_label) in zip(axes.flat, params, strict=False):
        for omega_val, colour in zip(omega_values, colours, strict=False):
            sub = df[np.isclose(df["omega"], omega_val)]
            sub = sub.sort_values("N")
            ax.plot(
                sub["N"],
                sub[param_col],
                "o-",
                color=colour,
                label=rf"$\omega={omega_val:.1f}$" if ax == axes.flat[0] else None,
                markersize=5,
                linewidth=1.2,
            )
        ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
        ax.set_ylabel(param_label)

    for ax in axes.flat:
        ax.set_xlabel(r"$N$")
        ax.set_xlim(left=0.5)

    axes.flat[0].legend(fontsize=8, title=r"$\omega$")
    fig.suptitle("Optimal parameters vs system particle number $N$")
    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


# ============================================================================
# Parallel Dispatch Helper
# ============================================================================


def _parallel_map(
    worker_fn,
    items,
    desc: str = "Processing",
    max_workers: int | None = None,
) -> list:
    """Run worker_fn(item) for each item in parallel via process pool.

    Args:
        worker_fn: Callable taking a single item argument.
        items: Iterable of items.
        desc: Short description for progress logging.
        max_workers: Number of subprocess workers (default: CPU count).

    Returns:
        List of results in the same order as items.
    """
    if max_workers is None:
        max_workers = min(32, os.cpu_count() or 1)
    item_list = list(items)
    print(f"  [parallel] {desc}: {len(item_list)} items, {max_workers} workers")

    mp_ctx = _mp.get_context("fork")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=mp_ctx,
    ) as executor:
        fut_to_idx = {
            executor.submit(worker_fn, item): i for i, item in enumerate(item_list)
        }
        results: list = [None] * len(item_list)
        for future in concurrent.futures.as_completed(fut_to_idx):
            idx = fut_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                print(f"  [ERROR] item={item_list[idx]}: {exc}")
                raise
    return results


# ============================================================================
# Data Generation Pipeline
# ============================================================================


def _run_single_n_omega_for_parallel(
    args: tuple[int, float],
) -> dict[str, int | float | str]:
    """Worker for parallel N-scaling scan.

    Args:
        args: Tuple (N, omega).

    Returns:
        Dict of result data.
    """
    N, omega = args
    print(f"  [run] N={N}, ω={omega}")
    result = run_single_n_omega(N, omega)
    return {
        "N": result.N,
        "omega": result.omega,
        "delta_omega_opt": result.delta_omega_opt,
        "sql": result.sql,
        "ratio": result.ratio,
        "a_x_opt": result.a_x_opt,
        "a_y_opt": result.a_y_opt,
        "a_z_opt": result.a_z_opt,
        "a_zz_opt": result.a_zz_opt,
        "expectation_Jz": result.expectation_Jz,
        "variance_Jz": result.variance_Jz,
        "success": int(result.success),
        "nfev": result.nfev,
    }


def generate_n_scaling_scan(force: bool = False) -> None:
    """Full N-scaling scan: 20 N values × 5 ω values = 100 optimisation runs.

    Each (N, ω) pair runs:
      1. 4D random search with 500 points.
      2. Nelder-Mead refinement from top 50 points.

    Results are saved to a single Parquet file. Intermediate checkpoints
    are saved per-N-value to allow resumption if interrupted.

    Args:
        force: Re-run even if Parquet exists.
    """
    csv_p = _parquet_path("n-scaling-scan")
    checkpoint_dir = REPORTS_DIR / REPORT_DATE / "raw_data" / "checkpoints"
    fig_ratio_p = _fig_path("n-scaling-ratio")
    fig_sensitivity_p = _fig_path("n-scaling-sensitivity")
    fig_params_p = _fig_path("n-scaling-optimal-params")

    if csv_p.exists() and not force:
        print(f"[skip] {csv_p.name} exists (use --force to overwrite)")
        summary = NScalingScanResult.from_parquet(csv_p)
    else:
        if force:
            # Remove existing checkpoints and final file
            csv_p.unlink(missing_ok=True)
            if checkpoint_dir.exists():
                import shutil
                shutil.rmtree(checkpoint_dir)

        # Load existing checkpoints if present
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        completed: set[tuple[int, float]] = set()
        checkpoint_results: list[NScalingResult] = []
        for ckpt_file in sorted(checkpoint_dir.glob("N_*.parquet")):
            try:
                df_ckpt = pd.read_parquet(ckpt_file)
                for _, row in df_ckpt.iterrows():
                    n_val = int(row["N"])
                    w_val = float(row["omega"])
                    delta = float(row["delta_omega_opt"])
                    if np.isfinite(delta):
                        completed.add((n_val, w_val))
                        checkpoint_results.append(
                            NScalingResult(
                                N=n_val,
                                omega=w_val,
                                delta_omega_opt=delta,
                                sql=float(row["sql"]),
                                ratio=float(row["ratio"]),
                                a_x_opt=float(row["a_x_opt"]),
                                a_y_opt=float(row["a_y_opt"]),
                                a_z_opt=float(row["a_z_opt"]),
                                a_zz_opt=float(row["a_zz_opt"]),
                                expectation_Jz=float(row.get("expectation_Jz", 0.0)),
                                variance_Jz=float(row.get("variance_Jz", 0.0)),
                                success=bool(int(row.get("success", 0))),
                                nfev=int(row.get("nfev", 0)),
                            ),
                        )
            except Exception as exc:
                print(f"  [warn] Could not load checkpoint {ckpt_file}: {exc}")

        # Process N values sequentially (with ω parallelism within each N)
        items_to_run = [(N, omega) for N in N_VALS for omega in OMEGA_VALS
                        if (N, omega) not in completed]
        if items_to_run:
            print(f"[run] N-scaling scan: {len(items_to_run)} remaining (N, ω) pairs")
            print(f"  (batch by N value, {min(32, os.cpu_count() or 1)} workers)")

            # Group by N and process each group in parallel
            by_N: dict[int, list[tuple[int, float]]] = {}
            for N, omega in items_to_run:
                by_N.setdefault(N, []).append((N, omega))

            for N in sorted(by_N):
                omega_items = by_N[N]
                n_ckpt = checkpoint_dir / f"N_{N:03d}.parquet"
                if n_ckpt.exists():
                    print(f"  [ckpt] N={N} already done, skipping")
                    continue
                print(f"  [batch] N={N}: {len(omega_items)} ω values (parallel)")
                batch_results = _parallel_map(
                    _run_single_n_omega_for_parallel,
                    omega_items,
                    desc=f"N={N} scan",
                )
                # Save checkpoint
                ckpt_list: list[NScalingResult] = []
                for rdict in batch_results:
                    delta = rdict["delta_omega_opt"]
                    if not np.isfinite(delta):
                        print(f"    [skip] N={rdict['N']}, ω={rdict['omega']}: Δω={delta}")
                        continue
                    ckpt_list.append(
                        NScalingResult(
                            N=rdict["N"],
                            omega=rdict["omega"],
                            delta_omega_opt=rdict["delta_omega_opt"],
                            sql=rdict["sql"],
                            ratio=rdict["ratio"],
                            a_x_opt=rdict["a_x_opt"],
                            a_y_opt=rdict["a_y_opt"],
                            a_z_opt=rdict["a_z_opt"],
                            a_zz_opt=rdict["a_zz_opt"],
                            expectation_Jz=rdict["expectation_Jz"],
                            variance_Jz=rdict["variance_Jz"],
                            success=bool(rdict["success"]),
                            nfev=rdict["nfev"],
                        ),
                    )
                if ckpt_list:
                    ckpt_scan = NScalingScanResult(results=ckpt_list)
                    ckpt_scan.save_parquet(n_ckpt)
                    checkpoint_results.extend(ckpt_list)
                    print(f"    [ckpt] saved {n_ckpt.name}")
        else:
            print("  [skip] all pairs already completed in checkpoints")

        # Merge all checkpoint results and save final file
        summary = NScalingScanResult(results=checkpoint_results)
        summary.save_parquet(csv_p)
        print(f"[save] {csv_p}")

    # Generate figures
    plot_n_scaling_ratio(summary, fig_ratio_p)
    print(f"[fig]  {fig_ratio_p}")
    plot_n_scaling_sensitivity(summary, fig_sensitivity_p)
    print(f"[fig]  {fig_sensitivity_p}")
    plot_n_scaling_optimal_params(summary, fig_params_p)
    print(f"[fig]  {fig_params_p}")


def generate_decoupled_baseline(force: bool = False) -> None:
    """Verify the decoupled baseline for all N and ω values."""
    csv_p = _parquet_path("decoupled-baseline")

    if csv_p.exists() and not force:
        print(f"[skip] {csv_p.name} exists (use --force to overwrite)")
        return

    print("[run] Computing decoupled baseline for all (N, ω)...")
    verifications = verify_decoupled_baseline()
    results_list: list[dict[str, float | int | str]] = []
    for (N, omega), passed in verifications.items():
        sql_ref = sql_reference(N)
        delta = compute_n_particle_decoupled_baseline(N, omega)
        results_list.append(
            {
                "N": N,
                "omega": omega,
                "delta_omega": delta,
                "sql": sql_ref,
                "ratio": delta / sql_ref if sql_ref > 0 else float("nan"),
                "pass": str(passed),
            },
        )
    df = pd.DataFrame(results_list)
    csv_p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(csv_p, index=False)
    print(f"[save] {csv_p}")


def generate_n1_consistency(force: bool = False) -> None:
    """Verify N=1 consistency with the 20260519 report.

    At N=1, ω=0.2, the pipeline should find Δω ≈ 0.02036 (R ≈ 4.91).
    """
    csv_p = _parquet_path("n1-consistency")

    if csv_p.exists() and not force:
        print(f"[skip] {csv_p.name} exists (use --force to overwrite)")
        df = pd.read_parquet(csv_p)
    else:
        print("[run] N=1 consistency check at ω=0.2...")
        result = run_single_n_omega(N=1, omega=0.2)
        result.save_parquet(csv_p)
        print(f"[save] {csv_p}")
        df = result.to_dataframe()

    delta = float(df["delta_omega_opt"].iloc[0])
    ratio = float(df["ratio"].iloc[0])
    print(f"  N=1, ω=0.2: Δω = {delta:.6f}, R = {ratio:.3f}")
    print("  Expected:   Δω ≈ 0.02036, R ≈ 4.91")


# ============================================================================
# CLI Entry Point
# ============================================================================


def main() -> None:
    """CLI entry point for generating all data and figures."""
    parser = argparse.ArgumentParser(
        description="N-Scaling of Phase-Modulated Ancilla Drive (20260611)",
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
    args = parser.parse_args()

    force = args.force

    generators: dict[str, tuple[str, str, bool]] = {
        "decoupled-baseline": (
            "Decoupled Baseline Verification",
            "generate_decoupled_baseline",
            False,
        ),
        "n1-consistency": (
            "N=1 Consistency Check",
            "generate_n1_consistency",
            False,
        ),
        "n-scaling-scan": (
            "N-Scaling Full Scan",
            "generate_n_scaling_scan",
            True,
        ),
    }

    if args.only:
        if args.only not in generators:
            print(f"Unknown generator: {args.only}")
            print(f"Available: {list(generators.keys())}")
            sys.exit(1)
        gen_list = [args.only]
    else:
        gen_list = list(generators.keys())

    # Ensure raw_data and figures directories exist
    for d in ["raw_data", "figures"]:
        (REPORTS_DIR / REPORT_DATE / d).mkdir(parents=True, exist_ok=True)

    for key in gen_list:
        name, func_name, _ = generators[key]
        print(f"\n{'=' * 60}")
        print(f"  {name}")
        print(f"{'=' * 60}")
        func = globals()[func_name]
        func(force=force)


if __name__ == "__main__":
    import sys

    main()
