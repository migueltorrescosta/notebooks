"""
Local module for the 2026-06-16 General 4-Parameter Interaction with
omega-Modulated Drive report.

Combines the ω-modulated ancilla drive mechanism (20260519) with the
general 4-parameter bilinear interaction (20260521) in a single protocol.

Three Hilbert space configurations:
  Step 1 (N=1): Both S and A are spin-1/2, dim = 4.
  Step 2 (N>1, J_A=1/2): N-particle system, single-qubit ancilla, dim = 2(N+1).
  Step 3 (N>1, J_A=N/2): N-particle system, N-particle ancilla, dim = (N+1)^2.

Hamiltonian:
  H = ω J_z^S + ω(a_x J_x^A + a_y J_y^A + a_z J_z^A) + H_int
  H_int = α_xx J_x^S J_x^A + α_xz J_x^S J_z^A + α_zx J_z^S J_x^A + α_zz J_z^S J_z^A

Circuit: BS_S → Hold → BS_S → measure J_z^S.

Usage:
    uv run python reports/20260616/local.py --force
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.optimize import minimize

from src.analysis.ancilla_optimization import (
    compute_expectation_and_variance,
)
from src.analysis.decoupled_baseline import (
    generate_decoupled_baseline,
    verify_decoupled_baseline,
)
from src.analysis.sensitivity_metrics import sql_reference
from src.physics.beam_splitter import bs_dicke
from src.physics.dicke_basis import jx_operator, jy_operator, jz_operator
from src.utils.constants import I_2, J_X, J_Y, J_Z
from src.utils.enums import OperatorBasis
from src.utils.parallel import parallel_map
from src.utils.paths import fig_path, parquet_path
from src.utils.serialization import ParquetSerializable

# ── BLAS threading ──────────────────────────────────────────────────────────
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    if _var not in os.environ:
        os.environ[_var] = "1"
if "MPLBACKEND" not in os.environ:
    os.environ["MPLBACKEND"] = "Agg"

# ── Physical constants ──────────────────────────────────────────────────────
T_HOLD: float = 10.0  # Holding time (fixed, SQL = 0.1/√N)
T_BS: float = np.pi / 2.0  # 50/50 beam splitter
FD_STEP: float = 1e-6  # Finite-difference step for ∂⟨J_z^S⟩/∂ω
DRIVE_BOUNDS: tuple[float, float] = (-5.0, 5.0)  # a_x, a_y, a_z bounds
ALPHA_BOUNDS: tuple[float, float] = (-20.0, 20.0)  # α_{ij} bounds
N_RANDOM: int = 5000  # Random search samples per (N, ω)
N_BFGS_REFINE: int = 200  # L-BFGS-B refinements per (N, ω)
BFGS_MAXITER: int = 1000  # Max L-BFGS-B iterations
BFGS_GTOL: float = 1e-6  # L-BFGS-B gradient tolerance


@dataclass
class CombinedProtocolConfig:
    """Configuration for the combined ω-modulated drive + 4-parameter interaction.

    Bundles the six parameters that control bounds enforcement, timing, and
    finite-difference numerics into a single argument, reducing cognitive
    overhead and caller verbosity.

    Attributes:
        drive_bounds: (min, max) for a_x, a_y, a_z drive parameters.
        alpha_bounds: (min, max) for α_{ij} interaction parameters.
        t_hold: Holding time for sensitivity evolution.
        T_bs: Beam-splitter duration.
        fd_step: Finite-difference step for ∂⟨J_z^S⟩/∂ω.
        penalty_scale: Scale for bound-violation quadratic penalty.
    """

    drive_bounds: tuple[float, float] = DRIVE_BOUNDS
    alpha_bounds: tuple[float, float] = ALPHA_BOUNDS
    t_hold: float = T_HOLD
    T_bs: float = T_BS
    fd_step: float = FD_STEP
    penalty_scale: float = 1e6


# ω values for N-scaling scans
OMEGA_VALS_N_SCALING: list[float] = [0.1, 0.2, 0.5, 1.0, 2.0]

# N values for scaling scans
N_VALS_FIXED_ANCILLA: list[int] = list(range(1, 14))  # Step 2
N_VALS_FULL_ANCILLA: list[int] = list(range(1, 14))  # Step 3


# ============================================================================
# Operator Construction
# ============================================================================


def build_fixed_ancilla_combined_operators(N: int) -> dict[str, np.ndarray]:
    """Build operators for N system particles with a single-qubit ancilla.

    System: Dicke basis (dim = N+1), Ancilla: single qubit (dim = 2).
    Total Hilbert space dim = 2(N+1).
    Basis ordering: {|m_S⟩_S ⊗ |0⟩_A, ..., |m_S⟩_S ⊗ |1⟩_A}
    where m_S descends from +J_S to -J_S.

    Returns:
        Dict with keys 'Jz_S', 'Jx_S', 'Jy_S', 'Jz_A', 'Jx_A', 'Jy_A',
        'I_full'. All are 2(N+1) × 2(N+1) Hermitian matrices.
    """
    if N < 1:
        raise ValueError(f"N must be ≥ 1, got {N}")

    d_sys = N + 1
    d_tot = 2 * d_sys

    Jz_S_dicke = jz_operator(N, basis=OperatorBasis.DICKE)
    Jx_S_dicke = jx_operator(N, basis=OperatorBasis.DICKE)
    Jy_S_dicke = jy_operator(N, basis=OperatorBasis.DICKE)
    I_S = np.eye(d_sys, dtype=complex)

    ops: dict[str, np.ndarray] = {
        "Jz_S": np.kron(Jz_S_dicke, I_2).astype(complex),
        "Jx_S": np.kron(Jx_S_dicke, I_2).astype(complex),
        "Jy_S": np.kron(Jy_S_dicke, I_2).astype(complex),
        "Jz_A": np.kron(I_S, J_Z).astype(complex),
        "Jx_A": np.kron(I_S, J_X).astype(complex),
        "Jy_A": np.kron(I_S, J_Y).astype(complex),
        "I_full": np.eye(d_tot, dtype=complex),
    }

    for key in ("Jz_S", "Jx_S", "Jy_S", "Jz_A", "Jx_A", "Jy_A"):
        assert ops[key].shape == (d_tot, d_tot), (
            f"{key} has shape {ops[key].shape}, expected ({d_tot}, {d_tot})"
        )
        assert np.allclose(ops[key], ops[key].conj().T, atol=1e-12), (
            f"{key} is not Hermitian for N={N}"
        )

    # Verify [J_z^S, J_x^S] = i J_y^S
    comm = ops["Jz_S"] @ ops["Jx_S"] - ops["Jx_S"] @ ops["Jz_S"]
    assert np.allclose(comm, 1j * ops["Jy_S"], atol=1e-10), (
        f"[J_z^S, J_x^S] = i J_y^S violated for N={N}"
    )

    return ops


def build_full_ancilla_combined_operators(N: int) -> dict[str, np.ndarray]:
    """Build operators for N system + N ancilla particles (J_S = J_A = N/2).

    Both S and A use Dicke basis (dim = N+1 each).
    Total Hilbert space dim = (N+1)^2.
    Basis ordering: {|m_S⟩_S ⊗ |m_A⟩_A} with m descending from +J to -J.

    For N=1, this reproduces the standard 2-qubit operators (dim=4).

    Returns:
        Dict with keys 'Jz_S', 'Jx_S', 'Jy_S', 'Jz_A', 'Jx_A', 'Jy_A',
        'I_full'. All are (N+1)^2 × (N+1)^2 Hermitian matrices.
    """
    if N < 1:
        raise ValueError(f"N must be ≥ 1, got {N}")

    d = N + 1
    d_tot = d * d

    Jz_dicke = jz_operator(N, basis=OperatorBasis.DICKE)
    Jx_dicke = jx_operator(N, basis=OperatorBasis.DICKE)
    Jy_dicke = jy_operator(N, basis=OperatorBasis.DICKE)
    I_d = np.eye(d, dtype=complex)

    ops: dict[str, np.ndarray] = {
        "Jz_S": np.kron(Jz_dicke, I_d).astype(complex),
        "Jx_S": np.kron(Jx_dicke, I_d).astype(complex),
        "Jy_S": np.kron(Jy_dicke, I_d).astype(complex),
        "Jz_A": np.kron(I_d, Jz_dicke).astype(complex),
        "Jx_A": np.kron(I_d, Jx_dicke).astype(complex),
        "Jy_A": np.kron(I_d, Jy_dicke).astype(complex),
        "I_full": np.eye(d_tot, dtype=complex),
    }

    for key in ("Jz_S", "Jx_S", "Jy_S", "Jz_A", "Jx_A", "Jy_A"):
        assert ops[key].shape == (d_tot, d_tot), (
            f"{key} has shape {ops[key].shape}, expected ({d_tot}, {d_tot})"
        )
        assert np.allclose(ops[key], ops[key].conj().T, atol=1e-12), (
            f"{key} is not Hermitian for N={N}"
        )

    comm = ops["Jz_S"] @ ops["Jx_S"] - ops["Jx_S"] @ ops["Jz_S"]
    assert np.allclose(comm, 1j * ops["Jy_S"], atol=1e-10), (
        f"[J_z^S, J_x^S] = i J_y^S violated for N={N}"
    )

    return ops


# ============================================================================
# Hamiltonian Construction
# ============================================================================


def build_combined_hold_hamiltonian(
    N: int,
    omega: float,
    a_x: float,
    a_y: float,
    a_z: float,
    alpha_xx: float,
    alpha_xz: float,
    alpha_zx: float,
    alpha_zz: float,
    ops: dict[str, np.ndarray],
) -> np.ndarray:
    r"""Build the total holding Hamiltonian for the combined protocol.

    H = \omega J_z^S
      + \omega (a_x J_x^A + a_y J_y^A + a_z J_z^A)
      + \alpha_{xx} J_x^S J_x^A + \alpha_{xz} J_x^S J_z^A
      + \alpha_{zx} J_z^S J_x^A + \alpha_{zz} J_z^S J_z^A

    Args:
        N: Number of system particles (for dimension check).
        omega: Unknown phase rate parameter.
        a_x, a_y, a_z: Ancilla drive coefficients.
        alpha_xx, alpha_xz, alpha_zx, alpha_zz: Interaction coefficients.
        ops: Operators from build_*_combined_operators().

    Returns:
        Hermitian Hamiltonian matrix.
    """
    d_tot = ops["I_full"].shape[0]

    # ω J_z^S
    H = omega * ops["Jz_S"]

    # ω-modulated ancilla drive
    H_drive = np.zeros((d_tot, d_tot), dtype=complex)
    if a_x != 0.0:
        H_drive += a_x * ops["Jx_A"]
    if a_y != 0.0:
        H_drive += a_y * ops["Jy_A"]
    if a_z != 0.0:
        H_drive += a_z * ops["Jz_A"]
    H += omega * H_drive

    # 4-parameter interaction
    H_int = np.zeros((d_tot, d_tot), dtype=complex)
    if alpha_xx != 0.0:
        H_int += alpha_xx * (ops["Jx_S"] @ ops["Jx_A"])
    if alpha_xz != 0.0:
        H_int += alpha_xz * (ops["Jx_S"] @ ops["Jz_A"])
    if alpha_zx != 0.0:
        H_int += alpha_zx * (ops["Jz_S"] @ ops["Jx_A"])
    if alpha_zz != 0.0:
        H_int += alpha_zz * (ops["Jz_S"] @ ops["Jz_A"])
    H += H_int

    # Enforce Hermiticity
    H = 0.5 * (H + H.conj().T)
    assert np.allclose(H, H.conj().T, atol=1e-12), (
        f"Combined Hamiltonian not Hermitian for N={N}"
    )
    return H


# ============================================================================
# Beam-Splitter and Initial State
# ============================================================================


def build_combined_system_bs_unitary(
    N: int,
    ancilla_dim: int,
    T_bs: float = T_BS,
) -> np.ndarray:
    """System-only beam-splitter unitary.

    U_BS_S = exp(-i T_bs J_x^S) ⊗ I_A

    Args:
        N: Number of system particles.
        ancilla_dim: Dimension of ancilla space (2 or N+1).
        T_bs: Beam-splitter duration (default π/2 for 50/50).

    Returns:
        d_tot × d_tot unitary matrix.
    """
    bs_sys = bs_dicke(N, T_bs)
    I_A = np.eye(ancilla_dim, dtype=complex)
    U = np.kron(bs_sys, I_A).astype(complex)
    d_tot = (N + 1) * ancilla_dim
    I_full = np.eye(d_tot, dtype=complex)
    assert np.allclose(U @ U.conj().T, I_full, atol=1e-12), (
        f"BS unitary not unitary for N={N}, ancilla_dim={ancilla_dim}"
    )
    return U


def combined_initial_state(d_tot: int) -> np.ndarray:
    """Initial state |Ψ₀⟩ = |J_S, J_S⟩_S ⊗ |J_A, J_A⟩_A.

    This is the first computational basis vector: [1, 0, ..., 0]ᵀ.

    Args:
        d_tot: Total Hilbert space dimension.

    Returns:
        Normalised complex vector of length d_tot.
    """
    psi = np.zeros(d_tot, dtype=complex)
    psi[0] = 1.0
    assert np.isclose(np.linalg.norm(psi), 1.0), "Initial state not normalised"
    return psi


# ============================================================================
# Circuit Evolution
# ============================================================================


def combined_hold_unitary(
    N: int,
    t_hold: float,
    omega: float,
    a_x: float,
    a_y: float,
    a_z: float,
    alpha_xx: float,
    alpha_xz: float,
    alpha_zx: float,
    alpha_zz: float,
    ops: dict[str, np.ndarray],
) -> np.ndarray:
    """Holding-time unitary for the combined protocol.

    U_hold(t_hold) = exp(-i t_hold H)

    Args:
        N: Number of system particles.
        t_hold: Holding-time strength.
        omega: Phase rate parameter.
        a_x, a_y, a_z: Ancilla drive coefficients.
        alpha_xx, alpha_xz, alpha_zx, alpha_zz: Interaction coefficients.
        ops: Operators from build_*_combined_operators().

    Returns:
        d_tot × d_tot unitary matrix.
    """
    H = build_combined_hold_hamiltonian(
        N,
        omega,
        a_x,
        a_y,
        a_z,
        alpha_xx,
        alpha_xz,
        alpha_zx,
        alpha_zz,
        ops,
    )
    U = expm(-1j * t_hold * H)
    # Relax tolerance for large N where expm accumulates numerical errors
    _tol = 1e-10 if N <= 8 else 1e-8
    assert np.allclose(U @ U.conj().T, ops["I_full"], atol=_tol), (
        f"Hold unitary not unitary for N={N}, t_hold={t_hold}"
    )
    return U


def evolve_combined_circuit(
    N: int,
    psi0: np.ndarray,
    T_bs: float,
    t_hold: float,
    omega: float,
    a_x: float,
    a_y: float,
    a_z: float,
    alpha_xx: float,
    alpha_xz: float,
    alpha_zx: float,
    alpha_zz: float,
    ops: dict[str, np.ndarray],
    ancilla_dim: int,
) -> np.ndarray:
    r"""Run the full combined-protocol MZI circuit.

    |\psi_{\text{final}}\rangle = U_{\text{BS}}^{(S)} \,
        U_{\text{hold}}(t_{\text{hold}}) \,
        U_{\text{BS}}^{(S)} \, |\psi_0\rangle

    Args:
        N: Number of system particles.
        psi0: Initial state vector.
        T_bs: Beam-splitter duration.
        t_hold: Holding-time strength.
        omega: Phase rate parameter.
        a_x, a_y, a_z: Ancilla drive coefficients.
        alpha_xx, alpha_xz, alpha_zx, alpha_zz: Interaction coefficients.
        ops: Operators from build_*_combined_operators().
        ancilla_dim: Dimension of ancilla space.

    Returns:
        Final normalised state vector.
    """
    assert np.isclose(np.linalg.norm(psi0), 1.0), "Initial state must be normalised"

    U_bs = build_combined_system_bs_unitary(N, ancilla_dim, T_bs)
    psi = U_bs @ psi0
    psi = (
        combined_hold_unitary(
            N,
            t_hold,
            omega,
            a_x,
            a_y,
            a_z,
            alpha_xx,
            alpha_xz,
            alpha_zx,
            alpha_zz,
            ops,
        )
        @ psi
    )
    psi = U_bs @ psi

    assert np.isclose(np.linalg.norm(psi), 1.0), "Final state not normalised"
    return psi


# ============================================================================
# Sensitivity Computation
# ============================================================================


def compute_combined_sensitivity(
    N: int,
    psi0: np.ndarray,
    T_bs: float,
    t_hold: float,
    omega_true: float,
    a_x: float,
    a_y: float,
    a_z: float,
    alpha_xx: float,
    alpha_xz: float,
    alpha_zx: float,
    alpha_zz: float,
    ops: dict[str, np.ndarray],
    ancilla_dim: int,
    fd_step: float = FD_STEP,
    meas_op: np.ndarray | None = None,
) -> float:
    r"""Compute the error-propagation sensitivity \Delta\omega.

    \Delta\omega = \sqrt{\mathrm{Var}(O)} / |\partial\langle O\rangle/\partial\omega|

    The central finite-difference captures the full \omega-dependence
    from both H_S and the \omega-modulated ancilla drive.

    Args:
        N: Number of system particles.
        psi0: Initial state vector.
        T_bs: Beam-splitter duration.
        t_hold: Holding-time strength.
        omega_true: True phase rate parameter.
        a_x, a_y, a_z: Ancilla drive coefficients.
        alpha_xx, alpha_xz, alpha_zx, alpha_zz: Interaction coefficients.
        ops: Operators from build_*_combined_operators().
        ancilla_dim: Dimension of ancilla space.
        fd_step: Finite-difference step size.
        meas_op: Measurement operator (default ops['Jz_S']).

    Returns:
        Sensitivity \Delta\omega (positive float). Returns inf if
        derivative is zero (fringe extremum).
    """
    if meas_op is None:
        meas_op = ops["Jz_S"]

    # Evaluate at omega_true
    psi = evolve_combined_circuit(
        N,
        psi0,
        T_bs,
        t_hold,
        omega_true,
        a_x,
        a_y,
        a_z,
        alpha_xx,
        alpha_xz,
        alpha_zx,
        alpha_zz,
        ops,
        ancilla_dim,
    )
    _, var = compute_expectation_and_variance(psi, meas_op)

    # Central finite difference for ∂⟨O⟩/∂ω
    psi_plus = evolve_combined_circuit(
        N,
        psi0,
        T_bs,
        t_hold,
        omega_true + fd_step,
        a_x,
        a_y,
        a_z,
        alpha_xx,
        alpha_xz,
        alpha_zx,
        alpha_zz,
        ops,
        ancilla_dim,
    )
    psi_minus = evolve_combined_circuit(
        N,
        psi0,
        T_bs,
        t_hold,
        omega_true - fd_step,
        a_x,
        a_y,
        a_z,
        alpha_xx,
        alpha_xz,
        alpha_zx,
        alpha_zz,
        ops,
        ancilla_dim,
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
class CombinedOptimizationResult(ParquetSerializable):
    """Result from optimising the combined protocol for a single (N, ω) pair.

    The 7D parameter space: (a_x, a_y, a_z, α_xx, α_xz, α_zx, α_zz).

    Attributes:
        N: Number of system particles.
        omega: Phase rate value.
        delta_omega_opt: Best sensitivity Δω found.
        sql: SQL = 1/(√N × t_hold).
        ratio: SQL / Δω_opt.
        a_x_opt, a_y_opt, a_z_opt: Optimal drive parameters.
        alpha_xx_opt, alpha_xz_opt, alpha_zx_opt, alpha_zz_opt: Optimal α.
        expectation_Jz: ⟨J_z^S⟩ at optimal point.
        variance_Jz: Var(J_z^S) at optimal point.
        t_hold: Holding time.
        fd_step: Finite-difference step.
        success: Whether the best L-BFGS-B run converged.
        nfev: Number of function evaluations in best run.
        n_starts: Number of random starts for L-BFGS-B.
        n_converged: Number of starts that converged.
    """

    N: int
    omega: float
    delta_omega_opt: float
    sql: float
    ratio: float
    a_x_opt: float
    a_y_opt: float
    a_z_opt: float
    alpha_xx_opt: float
    alpha_xz_opt: float
    alpha_zx_opt: float
    alpha_zz_opt: float
    expectation_Jz: float = 0.0
    variance_Jz: float = 0.0
    t_hold: float = T_HOLD
    fd_step: float = FD_STEP
    success: bool = False
    nfev: int = 0
    n_starts: int = 0
    n_converged: int = 0

    _PARQUET_COLUMNS: ClassVar[list[str]] = [
        "N",
        "omega",
        "delta_omega_opt",
        "sql",
        "ratio",
        "a_x_opt",
        "a_y_opt",
        "a_z_opt",
        "alpha_xx_opt",
        "alpha_xz_opt",
        "alpha_zx_opt",
        "alpha_zz_opt",
        "expectation_Jz",
        "variance_Jz",
        "t_hold",
        "fd_step",
        "success",
        "nfev",
        "n_starts",
        "n_converged",
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
                "alpha_xx_opt": [self.alpha_xx_opt],
                "alpha_xz_opt": [self.alpha_xz_opt],
                "alpha_zx_opt": [self.alpha_zx_opt],
                "alpha_zz_opt": [self.alpha_zz_opt],
                "expectation_Jz": [self.expectation_Jz],
                "variance_Jz": [self.variance_Jz],
                "t_hold": [self.t_hold],
                "fd_step": [self.fd_step],
                "success": [int(self.success)],
                "nfev": [self.nfev],
                "n_starts": [self.n_starts],
                "n_converged": [self.n_converged],
            },
        )

    @classmethod
    def from_parquet(cls, path: str | Path) -> CombinedOptimizationResult:
        path = Path(path)
        df = pd.read_parquet(path)
        cls._validate_columns(df)
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
            alpha_xx_opt=float(row["alpha_xx_opt"]),
            alpha_xz_opt=float(row["alpha_xz_opt"]),
            alpha_zx_opt=float(row["alpha_zx_opt"]),
            alpha_zz_opt=float(row["alpha_zz_opt"]),
            expectation_Jz=float(row["expectation_Jz"]),
            variance_Jz=float(row["variance_Jz"]),
            t_hold=float(row["t_hold"]),
            fd_step=float(row["fd_step"]),
            success=bool(int(row["success"])),
            nfev=int(row["nfev"]),
            n_starts=int(row["n_starts"]),
            n_converged=int(row["n_converged"]),
        )


@dataclass
class CombinedNScalingScanResult(ParquetSerializable):
    """Collection of N-scaling results for the combined protocol.

    Attributes:
        results: List of per-(N, ω) CombinedOptimizationResult.
    """

    results: list[CombinedOptimizationResult] = field(default_factory=list)

    _PARQUET_COLUMNS: ClassVar[list[str]] = CombinedOptimizationResult._PARQUET_COLUMNS

    def to_dataframe(self) -> pd.DataFrame:
        if not self.results:
            return pd.DataFrame(columns=self._PARQUET_COLUMNS)
        return pd.concat([r.to_dataframe() for r in self.results], ignore_index=True)

    @classmethod
    def from_parquet(cls, path: str | Path) -> CombinedNScalingScanResult:
        path = Path(path)
        df = pd.read_parquet(path)
        cls._validate_columns(df)
        results: list[CombinedOptimizationResult] = []
        for _, row in df.iterrows():
            results.append(
                CombinedOptimizationResult(
                    N=int(row["N"]),
                    omega=float(row["omega"]),
                    delta_omega_opt=float(row["delta_omega_opt"]),
                    sql=float(row["sql"]),
                    ratio=float(row["ratio"]),
                    a_x_opt=float(row["a_x_opt"]),
                    a_y_opt=float(row["a_y_opt"]),
                    a_z_opt=float(row["a_z_opt"]),
                    alpha_xx_opt=float(row["alpha_xx_opt"]),
                    alpha_xz_opt=float(row["alpha_xz_opt"]),
                    alpha_zx_opt=float(row["alpha_zx_opt"]),
                    alpha_zz_opt=float(row["alpha_zz_opt"]),
                    expectation_Jz=float(row["expectation_Jz"]),
                    variance_Jz=float(row["variance_Jz"]),
                    t_hold=float(row["t_hold"]),
                    fd_step=float(row["fd_step"]),
                    success=bool(int(row["success"])),
                    nfev=int(row["nfev"]),
                    n_starts=int(row["n_starts"]),
                    n_converged=int(row["n_converged"]),
                ),
            )
        return cls(results=results)

    @property
    def N_values(self) -> np.ndarray:
        return np.array(sorted({r.N for r in self.results}))

    @property
    def omega_values(self) -> np.ndarray:
        return np.array(sorted({r.omega for r in self.results}))


@dataclass
class Combined2DSliceResult(ParquetSerializable):
    """Result from a 2D parameter slice scan over (α_xx, α_zz).

    Attributes:
        N: Number of system particles (always 1 for this scan).
        alpha_xx_values: Array of α_xx values.
        alpha_zz_values: Array of α_zz values.
        delta_omega_grid: 2D array of Δω values, shape
            (len(alpha_xx_values), len(alpha_zz_values)).
        omega_value: The ω value at which the scan was performed.
        a_x_fixed: Fixed drive a_x value.
        a_y_fixed: Fixed drive a_y value.
        a_z_fixed: Fixed drive a_z value.
        alpha_xz_fixed: Fixed α_xz value (always 0 for this scan).
        alpha_zx_fixed: Fixed α_zx value (always 0 for this scan).
        sql: SQL reference value.
        t_hold: Holding time.
    """

    alpha_xx_values: np.ndarray
    alpha_zz_values: np.ndarray
    delta_omega_grid: np.ndarray
    omega_value: float
    a_x_fixed: float
    a_y_fixed: float
    a_z_fixed: float
    N: int = 1
    alpha_xz_fixed: float = 0.0
    alpha_zx_fixed: float = 0.0
    sql: float = 0.1
    t_hold: float = T_HOLD

    _PARQUET_COLUMNS: ClassVar[list[str]] = [
        "N",
        "alpha_xx",
        "alpha_zz",
        "delta_omega",
        "omega_value",
        "a_x_fixed",
        "a_y_fixed",
        "a_z_fixed",
        "alpha_xz_fixed",
        "alpha_zx_fixed",
        "sql",
        "t_hold",
    ]

    def to_dataframe(self) -> pd.DataFrame:
        n_xx = len(self.alpha_xx_values)
        n_zz = len(self.alpha_zz_values)
        rows = [
            {
                "N": self.N,
                "alpha_xx": float(self.alpha_xx_values[i]),
                "alpha_zz": float(self.alpha_zz_values[j]),
                "delta_omega": float(self.delta_omega_grid[i, j]),
                "omega_value": float(self.omega_value),
                "a_x_fixed": float(self.a_x_fixed),
                "a_y_fixed": float(self.a_y_fixed),
                "a_z_fixed": float(self.a_z_fixed),
                "alpha_xz_fixed": float(self.alpha_xz_fixed),
                "alpha_zx_fixed": float(self.alpha_zx_fixed),
                "sql": float(self.sql),
                "t_hold": float(self.t_hold),
            }
            for i in range(n_xx)
            for j in range(n_zz)
        ]
        return pd.DataFrame(rows)

    @classmethod
    def from_parquet(cls, path: str | Path) -> Combined2DSliceResult:
        df = pd.read_parquet(path)
        cls._validate_columns(df)
        alpha_xx_unique = sorted(df["alpha_xx"].unique())
        alpha_zz_unique = sorted(df["alpha_zz"].unique())
        n_xx = len(alpha_xx_unique)
        n_zz = len(alpha_zz_unique)
        grid = np.full((n_xx, n_zz), np.nan, dtype=float)
        for _, row in df.iterrows():
            i = alpha_xx_unique.index(row["alpha_xx"])
            j = alpha_zz_unique.index(row["alpha_zz"])
            grid[i, j] = row["delta_omega"]
        return cls(
            N=int(df["N"].iloc[0]),
            alpha_xx_values=np.array(alpha_xx_unique, dtype=float),
            alpha_zz_values=np.array(alpha_zz_unique, dtype=float),
            delta_omega_grid=grid,
            omega_value=float(df["omega_value"].iloc[0]),
            a_x_fixed=float(df["a_x_fixed"].iloc[0]),
            a_y_fixed=float(df["a_y_fixed"].iloc[0]),
            a_z_fixed=float(df["a_z_fixed"].iloc[0]),
            alpha_xz_fixed=float(df["alpha_xz_fixed"].iloc[0]),
            alpha_zx_fixed=float(df["alpha_zx_fixed"].iloc[0]),
            sql=float(df["sql"].iloc[0]),
            t_hold=float(df["t_hold"].iloc[0]),
        )


@dataclass
class CombinedRandomSearchResult(ParquetSerializable):
    """Result from a 7D random search over (a, α) parameters.

    Attributes:
        N: Number of system particles.
        samples: Array of shape (N_samp, 7) with sampled parameter values.
        delta_omega_values: Array of shape (N_samp,) with Δω for each sample.
        best_params: The 7-element tuple that gave minimal Δω.
        best_delta_omega: The minimal Δω found.
        omega_value: ω at which the search was performed.
        sql: SQL reference value.
    """

    N: int
    samples: np.ndarray
    delta_omega_values: np.ndarray
    best_params: tuple[float, float, float, float, float, float, float]
    best_delta_omega: float
    omega_value: float = 1.0
    sql: float = 0.1
    t_hold: float = T_HOLD

    _PARQUET_COLUMNS: ClassVar[list[str]] = [
        "N",
        "a_x",
        "a_y",
        "a_z",
        "alpha_xx",
        "alpha_xz",
        "alpha_zx",
        "alpha_zz",
        "delta_omega",
        "omega_value",
        "sql",
        "t_hold",
    ]

    def to_dataframe(self) -> pd.DataFrame:
        n = len(self.samples)
        return pd.DataFrame(
            {
                "N": [self.N] * n,
                "a_x": self.samples[:, 0],
                "a_y": self.samples[:, 1],
                "a_z": self.samples[:, 2],
                "alpha_xx": self.samples[:, 3],
                "alpha_xz": self.samples[:, 4],
                "alpha_zx": self.samples[:, 5],
                "alpha_zz": self.samples[:, 6],
                "delta_omega": self.delta_omega_values,
                "omega_value": [self.omega_value] * n,
                "sql": [self.sql] * n,
                "t_hold": [self.t_hold] * n,
            },
        )

    @classmethod
    def from_parquet(cls, path: str | Path) -> CombinedRandomSearchResult:
        df = pd.read_parquet(path)
        cls._validate_columns(df)
        samples = df[
            ["a_x", "a_y", "a_z", "alpha_xx", "alpha_xz", "alpha_zx", "alpha_zz"]
        ].to_numpy(dtype=float)
        deltas = df["delta_omega"].to_numpy(dtype=float)
        best_idx = int(np.argmin(deltas))
        return cls(
            N=int(df["N"].iloc[0]),
            samples=samples,
            delta_omega_values=deltas,
            best_params=(
                float(samples[best_idx, 0]),
                float(samples[best_idx, 1]),
                float(samples[best_idx, 2]),
                float(samples[best_idx, 3]),
                float(samples[best_idx, 4]),
                float(samples[best_idx, 5]),
                float(samples[best_idx, 6]),
            ),
            best_delta_omega=float(deltas[best_idx]),
            omega_value=float(df["omega_value"].iloc[0]),
            sql=float(df["sql"].iloc[0]),
            t_hold=float(df["t_hold"].iloc[0]),
        )


# ============================================================================
# 7D Random Search
# ============================================================================


def _params_to_args(
    params: np.ndarray,
) -> tuple[float, float, float, float, float, float, float]:
    """Extract named parameters from a 7-element array.

    Order: a_x, a_y, a_z, α_xx, α_xz, α_zx, α_zz.
    """
    return (
        float(params[0]),
        float(params[1]),
        float(params[2]),
        float(params[3]),
        float(params[4]),
        float(params[5]),
        float(params[6]),
    )


def combined_objective(
    params: np.ndarray,
    N: int,
    omega_true: float,
    ops: dict[str, np.ndarray],
    psi0: np.ndarray,
    ancilla_dim: int,
    cfg: CombinedProtocolConfig | None = None,
) -> float:
    """Objective function for minimising Δω in the combined protocol.

    params = [a_x, a_y, a_z, α_xx, α_xz, α_zx, α_zz] (7 elements).

    Bounds enforcement: a parameters use cfg.drive_bounds, α parameters use
    cfg.alpha_bounds. Out-of-bounds values receive a quadratic penalty.

    Returns:
        Δω (plus penalty if bounds violated).
    """
    if cfg is None:
        cfg = CombinedProtocolConfig()
    a_x, a_y, a_z, alpha_xx, alpha_xz, alpha_zx, alpha_zz = _params_to_args(params)

    # Bound enforcement (different bounds for drive vs α)
    penalty = 0.0
    lo_d, hi_d = cfg.drive_bounds
    for val in (a_x, a_y, a_z):
        if val < lo_d:
            penalty += cfg.penalty_scale * (lo_d - val) ** 2
        if val > hi_d:
            penalty += cfg.penalty_scale * (val - hi_d) ** 2
    lo_a, hi_a = cfg.alpha_bounds
    for val in (alpha_xx, alpha_xz, alpha_zx, alpha_zz):
        if val < lo_a:
            penalty += cfg.penalty_scale * (lo_a - val) ** 2
        if val > hi_a:
            penalty += cfg.penalty_scale * (val - hi_a) ** 2

    if penalty > 0.0:
        return float(1e10 + penalty)

    return compute_combined_sensitivity(
        N,
        psi0,
        cfg.T_bs,
        cfg.t_hold,
        omega_true,
        a_x,
        a_y,
        a_z,
        alpha_xx,
        alpha_xz,
        alpha_zx,
        alpha_zz,
        ops,
        ancilla_dim,
        cfg.fd_step,
    )


def combined_random_search(
    N: int,
    omega: float,
    ops: dict[str, np.ndarray],
    psi0: np.ndarray,
    ancilla_dim: int,
    n_samples: int = N_RANDOM,
    cfg: CombinedProtocolConfig | None = None,
    seed: int | None = 42,
) -> CombinedRandomSearchResult:
    """7D random search over (a_x, a_y, a_z, α_xx, α_xz, α_zx, α_zz).

    Drive parameters sampled uniformly in cfg.drive_bounds.
    α parameters sampled uniformly in cfg.alpha_bounds.

    Args:
        N: Number of system particles.
        omega: Phase rate value.
        ops: Operators from build_*_combined_operators().
        psi0: Initial state vector.
        ancilla_dim: Dimension of ancilla space.
        n_samples: Number of random points.
        cfg: Protocol configuration (bounds, timing, fd_step).
        seed: Random seed for reproducibility.

    Returns:
        CombinedRandomSearchResult.
    """
    if cfg is None:
        cfg = CombinedProtocolConfig()
    rng = np.random.default_rng(seed)
    lo_d, hi_d = cfg.drive_bounds
    lo_a, hi_a = cfg.alpha_bounds

    # Generate samples: first 3 columns = drive params, last 4 = α params
    samples = np.zeros((n_samples, 7), dtype=float)
    samples[:, :3] = rng.uniform(lo_d, hi_d, size=(n_samples, 3))
    samples[:, 3:] = rng.uniform(lo_a, hi_a, size=(n_samples, 4))

    deltas = np.full(n_samples, np.inf, dtype=float)

    for i in range(n_samples):
        a_x, a_y, a_z, alpha_xx, alpha_xz, alpha_zx, alpha_zz = (
            float(samples[i, j]) for j in range(7)
        )
        domega = compute_combined_sensitivity(
            N,
            psi0,
            cfg.T_bs,
            cfg.t_hold,
            omega,
            a_x,
            a_y,
            a_z,
            alpha_xx,
            alpha_xz,
            alpha_zx,
            alpha_zz,
            ops,
            ancilla_dim,
        )
        deltas[i] = domega

    best_idx = int(np.argmin(deltas))
    best_params = (
        float(samples[best_idx, 0]),
        float(samples[best_idx, 1]),
        float(samples[best_idx, 2]),
        float(samples[best_idx, 3]),
        float(samples[best_idx, 4]),
        float(samples[best_idx, 5]),
        float(samples[best_idx, 6]),
    )

    return CombinedRandomSearchResult(
        N=N,
        samples=samples,
        delta_omega_values=deltas,
        best_params=best_params,
        best_delta_omega=float(deltas[best_idx]),
        omega_value=omega,
        sql=sql_reference(N),
        t_hold=cfg.t_hold,
    )


# ============================================================================
# L-BFGS-B Optimisation
# ============================================================================


def run_combined_bfgs_optimization(
    N: int,
    omega_true: float,
    ops: dict[str, np.ndarray],
    psi0: np.ndarray,
    ancilla_dim: int,
    n_starts: int = N_BFGS_REFINE,
    cfg: CombinedProtocolConfig | None = None,
    seed: int | None = 42,
    maxiter: int = BFGS_MAXITER,
    gtol: float = BFGS_GTOL,
) -> CombinedOptimizationResult:
    """Run multi-start L-BFGS-B optimisation for the combined protocol.

    For each start:
    1. Generate random initial point in the 7D parameter space.
    2. Run L-BFGS-B with bounded optimisation.
    3. Select the run with lowest Δω.

    Args:
        N: Number of system particles.
        omega_true: True phase rate parameter.
        ops: Operators from build_*_combined_operators().
        psi0: Initial state vector.
        ancilla_dim: Dimension of ancilla space.
        n_starts: Number of random starts.
        cfg: Protocol configuration (bounds, timing, fd_step).
        seed: Base random seed (incremented per start).
        maxiter: Maximum L-BFGS-B iterations.
        gtol: L-BFGS-B gradient convergence tolerance.

    Returns:
        CombinedOptimizationResult with best parameters found.
    """
    if cfg is None:
        cfg = CombinedProtocolConfig()
    lo_d, hi_d = cfg.drive_bounds
    lo_a, hi_a = cfg.alpha_bounds
    base_seed = seed if seed is not None else 42

    best_delta = float("inf")
    best_params_7 = np.zeros(7, dtype=float)
    n_converged = 0
    best_nfev = 0
    best_success = False

    for start in range(n_starts):
        rng = np.random.default_rng(base_seed + int(omega_true * 1000) + start)
        x0 = np.zeros(7, dtype=float)
        x0[:3] = rng.uniform(lo_d, hi_d, size=3)
        x0[3:] = rng.uniform(lo_a, hi_a, size=4)

        single_result = _run_single_bfgs_from_x0(
            N,
            omega_true,
            ops,
            psi0,
            ancilla_dim,
            x0,
            cfg=cfg,
            maxiter=maxiter,
            gtol=gtol,
        )

        if single_result.success:
            n_converged += 1

        delta_val = single_result.delta_omega_opt
        if np.isfinite(delta_val) and delta_val < best_delta:
            best_delta = delta_val
            best_params_7 = np.array(
                [
                    single_result.a_x_opt,
                    single_result.a_y_opt,
                    single_result.a_z_opt,
                    single_result.alpha_xx_opt,
                    single_result.alpha_xz_opt,
                    single_result.alpha_zx_opt,
                    single_result.alpha_zz_opt,
                ],
                dtype=float,
            )
            best_nfev = single_result.nfev
            best_success = single_result.success

    # Recompute diagnostics at the optimal point (combined result metadata)
    psi_final = evolve_combined_circuit(
        N,
        psi0,
        cfg.T_bs,
        cfg.t_hold,
        omega_true,
        float(best_params_7[0]),
        float(best_params_7[1]),
        float(best_params_7[2]),
        float(best_params_7[3]),
        float(best_params_7[4]),
        float(best_params_7[5]),
        float(best_params_7[6]),
        ops,
        ancilla_dim,
    )
    exp_val, var_val = compute_expectation_and_variance(psi_final, ops["Jz_S"])

    sql_val = sql_reference(N)
    ratio_val = (
        sql_val / best_delta
        if np.isfinite(best_delta) and best_delta > 0
        else float("nan")
    )

    return CombinedOptimizationResult(
        N=N,
        omega=omega_true,
        delta_omega_opt=best_delta,
        sql=sql_val,
        ratio=ratio_val,
        a_x_opt=float(best_params_7[0]),
        a_y_opt=float(best_params_7[1]),
        a_z_opt=float(best_params_7[2]),
        alpha_xx_opt=float(best_params_7[3]),
        alpha_xz_opt=float(best_params_7[4]),
        alpha_zx_opt=float(best_params_7[5]),
        alpha_zz_opt=float(best_params_7[6]),
        expectation_Jz=exp_val,
        variance_Jz=var_val,
        t_hold=cfg.t_hold,
        fd_step=cfg.fd_step,
        success=best_success,
        nfev=best_nfev,
        n_starts=n_starts,
        n_converged=n_converged,
    )


def run_combined_single_n_omega(
    N: int,
    omega: float,
    ancilla_dim: int,
    n_random: int = N_RANDOM,
    n_bfgs_refine: int = N_BFGS_REFINE,
    seed: int | None = 42,
) -> CombinedOptimizationResult:
    """Full optimisation pipeline for a single (N, ω) pair.

    1. Build operators (fixed or full ancilla based on ancilla_dim).
    2. 7D random search.
    3. L-BFGS-B refinement from top random-search points.
    4. Return the best CombinedOptimizationResult.

    Args:
        N: Number of system particles.
        omega: Phase rate value.
        ancilla_dim: Dimension of ancilla space.
        n_random: Random search samples.
        n_bfgs_refine: Number of L-BFGS-B refinements.
        seed: Base random seed.

    Returns:
        CombinedOptimizationResult.
    """
    base_seed = seed if seed is not None else 42

    # Build operators — full-ancilla for ancilla_dim=N+1, fixed for ancilla_dim=2
    if ancilla_dim == N + 1:
        ops = build_full_ancilla_combined_operators(N)
    elif ancilla_dim == 2:
        ops = build_fixed_ancilla_combined_operators(N)
    else:
        raise ValueError(
            f"Unsupported ancilla_dim={ancilla_dim}; expected 2 or {N + 1}"
        )

    d_tot = (N + 1) * ancilla_dim
    psi0 = combined_initial_state(d_tot)

    # Stage 1: Random search
    rs_result = combined_random_search(
        N,
        omega,
        ops,
        psi0,
        ancilla_dim,
        n_samples=n_random,
        seed=base_seed,
    )

    # Sort by Δω, take top n_bfgs_refine
    sorted_indices = np.argsort(rs_result.delta_omega_values)
    top_indices = sorted_indices[:n_bfgs_refine]

    # Stage 2: L-BFGS-B refinement from each top point
    bfgs_results: list[CombinedOptimizationResult] = []
    for rank, idx in enumerate(top_indices):
        x0 = rs_result.samples[idx].copy()
        # Use a single-start BFS call for this refinement
        bfgs = _run_single_bfgs_from_x0(
            N,
            omega,
            ops,
            psi0,
            ancilla_dim,
            x0,
            seed=base_seed + int(omega * 1000) + 10000 + rank,
        )
        bfgs_results.append(bfgs)

    # Sort by Δω and take best
    bfgs_results.sort(key=lambda r: r.delta_omega_opt)
    return bfgs_results[0]


def _run_single_bfgs_from_x0(
    N: int,
    omega_true: float,
    ops: dict[str, np.ndarray],
    psi0: np.ndarray,
    ancilla_dim: int,
    x0: np.ndarray,
    cfg: CombinedProtocolConfig | None = None,
    seed: int | None = 42,
    maxiter: int = BFGS_MAXITER,
    gtol: float = BFGS_GTOL,
) -> CombinedOptimizationResult:
    """Run a single L-BFGS-B refinement from a given starting point.

    Args:
        N: Number of system particles.
        omega_true: True phase rate parameter.
        ops: Operators from build_*_combined_operators().
        psi0: Initial state vector.
        ancilla_dim: Dimension of ancilla space.
        x0: Starting point (7-element array).
        cfg: Protocol configuration (bounds, timing, fd_step, penalty_scale).
        seed: Random seed (unused here, kept for API consistency).
        maxiter: Maximum L-BFGS-B iterations.
        gtol: L-BFGS-B gradient convergence tolerance.

    Returns:
        CombinedOptimizationResult for this single run.
    """
    if cfg is None:
        cfg = CombinedProtocolConfig()
    lo_d, hi_d = cfg.drive_bounds
    lo_a, hi_a = cfg.alpha_bounds
    bounds_ls = [(lo_d, hi_d)] * 3 + [(lo_a, hi_a)] * 4

    result = minimize(
        combined_objective,
        x0,
        args=(
            N,
            omega_true,
            ops,
            psi0,
            ancilla_dim,
            cfg,
        ),
        method="L-BFGS-B",
        bounds=bounds_ls,
        options={
            "maxiter": maxiter,
            "gtol": gtol,
            "ftol": 1e-12,
        },
    )

    opt_params = result.x.copy()
    delta_val = float(result.fun)

    # Diagnostics
    psi_final = evolve_combined_circuit(
        N,
        psi0,
        cfg.T_bs,
        cfg.t_hold,
        omega_true,
        float(opt_params[0]),
        float(opt_params[1]),
        float(opt_params[2]),
        float(opt_params[3]),
        float(opt_params[4]),
        float(opt_params[5]),
        float(opt_params[6]),
        ops,
        ancilla_dim,
    )
    exp_val, var_val = compute_expectation_and_variance(psi_final, ops["Jz_S"])

    sql_val = sql_reference(N)
    ratio_val = (
        sql_val / delta_val
        if np.isfinite(delta_val) and delta_val > 0
        else float("nan")
    )

    return CombinedOptimizationResult(
        N=N,
        omega=omega_true,
        delta_omega_opt=delta_val,
        sql=sql_val,
        ratio=ratio_val,
        a_x_opt=float(opt_params[0]),
        a_y_opt=float(opt_params[1]),
        a_z_opt=float(opt_params[2]),
        alpha_xx_opt=float(opt_params[3]),
        alpha_xz_opt=float(opt_params[4]),
        alpha_zx_opt=float(opt_params[5]),
        alpha_zz_opt=float(opt_params[6]),
        expectation_Jz=exp_val,
        variance_Jz=var_val,
        t_hold=cfg.t_hold,
        fd_step=cfg.fd_step,
        success=bool(result.success),
        nfev=int(result.nfev),
        n_starts=1,
        n_converged=int(result.success),
    )


# ============================================================================
# Decoupled Baseline
# ============================================================================


def compute_combined_decoupled_baseline(
    N: int,
    omega_true: float = 1.0,
    ancilla_dim: int = 2,
) -> float:
    """Compute the decoupled baseline sensitivity Δω.

    At all parameters zero, the circuit reduces to a standard MZI:
    Δω = 1/(√N × T_HOLD).

    Args:
        N: Number of system particles.
        omega_true: Phase rate value.
        ancilla_dim: Dimension of ancilla space.

    Returns:
        Δω at the decoupled configuration.
    """
    if ancilla_dim == N + 1:
        ops = build_full_ancilla_combined_operators(N)
    else:
        ops = build_fixed_ancilla_combined_operators(N)
    d_tot = (N + 1) * ancilla_dim
    psi0 = combined_initial_state(d_tot)

    return compute_combined_sensitivity(
        N,
        psi0,
        T_BS,
        T_HOLD,
        omega_true,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        ops,
        ancilla_dim,
    )


# ============================================================================
# 2D Slice Scan (Step 1 Landscape)
# ============================================================================


def _combined_slice_worker(args: tuple) -> tuple[int, np.ndarray]:
    """Worker for parallel 2D slice evaluation (module-level for pickling).

    Args:
        args: Tuple (omega, a_x, a_y, a_z, alpha_xx_chunk, alpha_zz_vals,
                     ancilla_dim, start_idx).

    Returns:
        Tuple (start_idx, chunk_grid).
    """
    (omega, a_x, a_y, a_z, alpha_xx_chunk, alpha_zz_vals, ancilla_dim, start_idx) = args
    local_ops = build_full_ancilla_combined_operators(1)
    d_tot = 2 * ancilla_dim  # N=1
    psi0 = combined_initial_state(d_tot)
    n_xx = len(alpha_xx_chunk)
    n_zz = len(alpha_zz_vals)
    chunk_grid = np.full((n_xx, n_zz), np.inf, dtype=float)
    for i, a_xx in enumerate(alpha_xx_chunk):
        for j, a_zz in enumerate(alpha_zz_vals):
            chunk_grid[i, j] = compute_combined_sensitivity(
                1,
                psi0,
                T_BS,
                T_HOLD,
                omega,
                a_x,
                a_y,
                a_z,
                float(a_xx),
                0.0,
                0.0,
                float(a_zz),
                local_ops,
                ancilla_dim,
            )
    return start_idx, chunk_grid


def _run_parallel_combined_slice(
    omega: float,
    a_x: float,
    a_y: float,
    a_z: float,
    alpha_xx_vals: np.ndarray,
    alpha_zz_vals: np.ndarray,
    ancilla_dim: int,
    n_jobs: int,
) -> np.ndarray:
    """Run parallel 2D slice evaluation across multiple workers.

    Args:
        omega: Phase rate value.
        a_x, a_y, a_z: Fixed drive coefficients.
        alpha_xx_vals: Array of alpha_xx values.
        alpha_zz_vals: Array of alpha_zz values.
        ancilla_dim: Ancilla dimension.
        n_jobs: Number of parallel workers (-1 for all cores).

    Returns:
        2D grid array of Delta_omega values,
        shape (len(alpha_xx_vals), len(alpha_zz_vals)).
    """
    n_workers = max(1, os.cpu_count() or 4) if n_jobs == -1 else n_jobs
    indices = np.arange(len(alpha_xx_vals))
    chunks = np.array_split(indices, n_workers)
    worker_args = [
        (
            omega,
            a_x,
            a_y,
            a_z,
            alpha_xx_vals[chunk],
            alpha_zz_vals,
            ancilla_dim,
            int(chunk[0]),
        )
        for chunk in chunks
    ]
    grid = np.full((len(alpha_xx_vals), len(alpha_zz_vals)), np.inf, dtype=float)
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=n_workers,
    ) as executor:
        futures = {
            executor.submit(_combined_slice_worker, args): args for args in worker_args
        }
        for future in concurrent.futures.as_completed(futures):
            start_idx, chunk_grid = future.result()
            n_chunk = chunk_grid.shape[0]
            grid[start_idx : start_idx + n_chunk, :] = chunk_grid
    return grid


def combined_2d_slice(
    omega: float,
    a_x: float = 0.0,
    a_y: float = 0.0,
    a_z: float = 0.0,
    alpha_xx_range: tuple[float, float] = (-20.0, 20.0),
    alpha_zz_range: tuple[float, float] = (-20.0, 20.0),
    n_alpha_xx: int = 201,
    n_alpha_zz: int = 201,
    ancilla_dim: int = 2,
    n_jobs: int | None = None,
) -> Combined2DSliceResult:
    """Run a 2D slice scan over (α_xx, α_zz) at N=1.

    Args:
        omega: Phase rate value.
        a_x, a_y, a_z: Fixed drive coefficients.
        alpha_xx_range: (min, max) for α_xx.
        alpha_zz_range: (min, max) for α_zz.
        n_alpha_xx: Number of α_xx points.
        n_alpha_zz: Number of α_zz points.
        ancilla_dim: Ancilla dimension (2 for J_A=1/2 or N+1=2 for J_A=N/2 at N=1).
        n_jobs: Number of parallel workers. None = sequential.

    Returns:
        Combined2DSliceResult.
    """
    alpha_xx_vals = np.linspace(alpha_xx_range[0], alpha_xx_range[1], n_alpha_xx)
    alpha_zz_vals = np.linspace(alpha_zz_range[0], alpha_zz_range[1], n_alpha_zz)

    if n_jobs is None or n_jobs == 1:
        _, grid = _combined_slice_worker(
            (
                omega,
                a_x,
                a_y,
                a_z,
                alpha_xx_vals,
                alpha_zz_vals,
                ancilla_dim,
                0,
            )
        )
    else:
        grid = _run_parallel_combined_slice(
            omega,
            a_x,
            a_y,
            a_z,
            alpha_xx_vals,
            alpha_zz_vals,
            ancilla_dim,
            n_jobs,
        )

    return Combined2DSliceResult(
        alpha_xx_values=alpha_xx_vals,
        alpha_zz_values=alpha_zz_vals,
        delta_omega_grid=grid,
        omega_value=omega,
        a_x_fixed=a_x,
        a_y_fixed=a_y,
        a_z_fixed=a_z,
        sql=sql_reference(1),
    )


# ============================================================================
# N-Scaling Scans
# ============================================================================


def _run_combined_single_n_omega_worker(
    args: tuple[int, float, int],
) -> dict[str, int | float | str]:
    """Worker for parallel N-scaling scan.

    Args:
        args: Tuple (N, omega, ancilla_dim).

    Returns:
        Dict of result data.
    """
    N, omega, ancilla_dim = args
    print(f"  [run] N={N}, ω={omega}, ancilla_dim={ancilla_dim}")
    result = run_combined_single_n_omega(N, omega, ancilla_dim)
    return {
        "N": result.N,
        "omega": result.omega,
        "delta_omega_opt": result.delta_omega_opt,
        "sql": result.sql,
        "ratio": result.ratio,
        "a_x_opt": result.a_x_opt,
        "a_y_opt": result.a_y_opt,
        "a_z_opt": result.a_z_opt,
        "alpha_xx_opt": result.alpha_xx_opt,
        "alpha_xz_opt": result.alpha_xz_opt,
        "alpha_zx_opt": result.alpha_zx_opt,
        "alpha_zz_opt": result.alpha_zz_opt,
        "expectation_Jz": result.expectation_Jz,
        "variance_Jz": result.variance_Jz,
        "success": int(result.success),
        "nfev": result.nfev,
        "n_starts": result.n_starts,
        "n_converged": result.n_converged,
    }


def _make_result_from_dict(rdict: dict) -> CombinedOptimizationResult:
    """Convert a worker result dict to CombinedOptimizationResult."""
    delta = rdict["delta_omega_opt"]
    return CombinedOptimizationResult(
        N=rdict["N"],
        omega=rdict["omega"],
        delta_omega_opt=delta,
        sql=rdict["sql"],
        ratio=rdict["ratio"],
        a_x_opt=rdict["a_x_opt"],
        a_y_opt=rdict["a_y_opt"],
        a_z_opt=rdict["a_z_opt"],
        alpha_xx_opt=rdict["alpha_xx_opt"],
        alpha_xz_opt=rdict["alpha_xz_opt"],
        alpha_zx_opt=rdict["alpha_zx_opt"],
        alpha_zz_opt=rdict["alpha_zz_opt"],
        expectation_Jz=rdict["expectation_Jz"],
        variance_Jz=rdict["variance_Jz"],
        success=bool(rdict["success"]),
        nfev=rdict["nfev"],
        n_starts=rdict["n_starts"],
        n_converged=rdict["n_converged"],
    )


# ============================================================================
# Path Helpers
# ============================================================================

_REPORTS_DIR = Path(__file__).resolve().parent.parent.parent / "reports"
_REPORT_DATE = "20260616"


def _parquet_path(name: str) -> Path:
    return parquet_path(_REPORTS_DIR, _REPORT_DATE, name)


def _fig_path(name: str) -> Path:
    return fig_path(_REPORTS_DIR, _REPORT_DATE, name)


# ============================================================================
# Data Generation Functions
# ============================================================================


def _build_decoupled_baseline_df(
    ancilla_dim: int,
) -> pd.DataFrame:
    """Build a DataFrame of decoupled baseline verification results."""
    N_vals = N_VALS_FIXED_ANCILLA if ancilla_dim == 2 else N_VALS_FULL_ANCILLA
    verifications = verify_decoupled_baseline(
        N_values=N_vals,
        compute_fn=compute_combined_decoupled_baseline,
        ancilla_dim=ancilla_dim,
    )
    results_list: list[dict[str, float | int | str]] = []
    for (N, omega), passed in verifications.items():
        sql_ref = sql_reference(N)
        delta = compute_combined_decoupled_baseline(N, omega, ancilla_dim)
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
    return pd.DataFrame(results_list)


def run_decoupled_baseline(
    force: bool = False,
    ancilla_dim: int = 2,
) -> None:
    """Verify decoupled baseline for all (N, ω) pairs."""
    generate_decoupled_baseline(
        force=force,
        parquet_path=_parquet_path(f"decoupled-baseline-dim{ancilla_dim}"),
        compute_fn=_build_decoupled_baseline_df,
        compute_kwargs={"ancilla_dim": ancilla_dim},
        label=f"decoupled baseline (ancilla_dim={ancilla_dim})",
    )


def generate_n1_consistency(force: bool = False) -> None:
    """Verify N=1 consistency and test combined protocol.

    Runs at ω=0.2 (best ω for #20260519) and ω=3.8 (best ω for #20260521).
    """
    csv_p = _parquet_path("n1-consistency")

    if csv_p.exists() and not force:
        print(f"[skip] {csv_p.name} exists (use --force to overwrite)")
        return

    print("[run] N=1 consistency checks...")
    omegas = [0.2, 3.8]
    results_list: list[CombinedOptimizationResult] = []
    for omega in omegas:
        print(f"  N=1, ω={omega} (ancilla_dim=2, fixed-ancilla path)...")
        result = run_combined_single_n_omega(1, omega, ancilla_dim=2)
        results_list.append(result)
        print(f"    Δω = {result.delta_omega_opt:.6f}, R = {result.ratio:.3f}")

    summary = CombinedNScalingScanResult(results=results_list)
    summary.save_parquet(csv_p)
    print(f"[save] {csv_p}")


def _recover_checkpoints(
    checkpoint_dir: Path,
) -> tuple[set[tuple[int, float]], list[CombinedOptimizationResult]]:
    """Load existing checkpoints from a directory.

    Args:
        checkpoint_dir: Directory containing N_*.parquet checkpoint files.

    Returns:
        Tuple of (completed_set, results_list).
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    completed: set[tuple[int, float]] = set()
    checkpoint_results: list[CombinedOptimizationResult] = []
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
                        CombinedOptimizationResult(
                            N=n_val,
                            omega=w_val,
                            delta_omega_opt=delta,
                            sql=float(row["sql"]),
                            ratio=float(row["ratio"]),
                            a_x_opt=float(row["a_x_opt"]),
                            a_y_opt=float(row["a_y_opt"]),
                            a_z_opt=float(row["a_z_opt"]),
                            alpha_xx_opt=float(row["alpha_xx_opt"]),
                            alpha_xz_opt=float(row["alpha_xz_opt"]),
                            alpha_zx_opt=float(row["alpha_zx_opt"]),
                            alpha_zz_opt=float(row["alpha_zz_opt"]),
                            expectation_Jz=float(row["expectation_Jz"]),
                            variance_Jz=float(row["variance_Jz"]),
                            t_hold=float(row["t_hold"]),
                            fd_step=float(row["fd_step"]),
                            success=bool(int(row["success"])),
                            nfev=int(row["nfev"]),
                            n_starts=int(row["n_starts"]),
                            n_converged=int(row["n_converged"]),
                        ),
                    )
        except Exception as exc:
            print(f"  [warn] Could not load checkpoint {ckpt_file}: {exc}")
    return completed, checkpoint_results


def _run_single_optimisation(
    items_to_run: list[tuple[int, float, int]],
    checkpoint_dir: Path,
    checkpoint_results: list[CombinedOptimizationResult],
    desc_prefix: str,
) -> None:
    """Run optimisation for (N, ω) pairs, grouped by N with checkpointing.

    Groups remaining (N, ω) items by N, dispatches each group via
    parallel_map, and saves per-N checkpoint files.

    Args:
        items_to_run: List of (N, omega, ancilla_dim) tuples.
        checkpoint_dir: Directory for per-N checkpoint files.
        checkpoint_results: Mutable list to extend with results.
        desc_prefix: Prefix for progress messages (e.g., "Step2", "Step3").
    """
    by_N: dict[int, list[tuple[int, float, int]]] = {}
    for N, omega, ad in items_to_run:
        by_N.setdefault(N, []).append((N, omega, ad))

    for N in sorted(by_N):
        omega_items = by_N[N]
        n_ckpt = checkpoint_dir / f"N_{N:03d}.parquet"
        if n_ckpt.exists():
            print(f"  [ckpt] N={N} already done, skipping")
            continue
        print(f"  [batch] N={N}: {len(omega_items)} ω values (parallel)")
        batch_results = parallel_map(
            _run_combined_single_n_omega_worker,
            omega_items,
            desc=f"{desc_prefix} N={N}",
        )
        ckpt_list: list[CombinedOptimizationResult] = []
        for rdict in batch_results:
            if not np.isfinite(rdict["delta_omega_opt"]):
                continue
            ckpt_list.append(_make_result_from_dict(rdict))
        if ckpt_list:
            ckpt_scan = CombinedNScalingScanResult(results=ckpt_list)
            ckpt_scan.save_parquet(n_ckpt)
            checkpoint_results.extend(ckpt_list)
            print(f"    [ckpt] saved {n_ckpt.name}")


def generate_step2_n_scaling(force: bool = False) -> None:
    """Step 2: N-scaling scan with J_A = 1/2 (ancilla_dim=2).

    20 N values × 5 ω values = 100 optimisation runs.
    """
    csv_p = _parquet_path("step2-n-scaling")
    checkpoint_dir = _REPORTS_DIR / _REPORT_DATE / "raw_data" / "checkpoints_step2"

    if csv_p.exists() and not force:
        print(f"[skip] {csv_p.name} exists (use --force to overwrite)")
        return

    if force:
        csv_p.unlink(missing_ok=True)
        if checkpoint_dir.exists():
            import shutil

            shutil.rmtree(checkpoint_dir)

    completed, checkpoint_results = _recover_checkpoints(checkpoint_dir)

    items_to_run = [
        (N, omega, 2)
        for N in N_VALS_FIXED_ANCILLA
        for omega in OMEGA_VALS_N_SCALING
        if (N, omega) not in completed
    ]

    if items_to_run:
        print(f"[run] Step 2 N-scaling: {len(items_to_run)} remaining (N, ω) pairs")
        _run_single_optimisation(
            items_to_run,
            checkpoint_dir,
            checkpoint_results,
            "Step2",
        )
    else:
        print("  [skip] all pairs already completed in checkpoints")

    summary = CombinedNScalingScanResult(results=checkpoint_results)
    summary.save_parquet(csv_p)
    print(f"[save] {csv_p}")


def generate_step3_n_scaling(force: bool = False) -> None:
    """Step 3: N-scaling scan with J_A = N/2 (ancilla_dim = N+1).

    10 N values × 5 ω values = 50 optimisation runs.
    """
    csv_p = _parquet_path("step3-n-scaling")
    checkpoint_dir = _REPORTS_DIR / _REPORT_DATE / "raw_data" / "checkpoints_step3"

    if csv_p.exists() and not force:
        print(f"[skip] {csv_p.name} exists (use --force to overwrite)")
        return

    if force:
        csv_p.unlink(missing_ok=True)
        if checkpoint_dir.exists():
            import shutil

            shutil.rmtree(checkpoint_dir)

    completed, checkpoint_results = _recover_checkpoints(checkpoint_dir)

    items_to_run = [
        (N, omega, N + 1)
        for N in N_VALS_FULL_ANCILLA
        for omega in OMEGA_VALS_N_SCALING
        if (N, omega) not in completed
    ]

    if items_to_run:
        print(f"[run] Step 3 N-scaling: {len(items_to_run)} remaining (N, ω) pairs")
        _run_single_optimisation(
            items_to_run,
            checkpoint_dir,
            checkpoint_results,
            "Step3",
        )
    else:
        print("  [skip] all pairs already completed in checkpoints")

    summary = CombinedNScalingScanResult(results=checkpoint_results)
    summary.save_parquet(csv_p)
    print(f"[save] {csv_p}")


def generate_n1_full_omega_scan(force: bool = False) -> None:
    """Step 1: Full 7D optimisation at N=1 over 50 ω values (0.1 to 5.0).

    Uses ancilla_dim=2 (the N=1 fixed-ancilla path gives the same result).
    """
    csv_p = _parquet_path("n1-omega-scan")

    if csv_p.exists() and not force:
        print(f"[skip] {csv_p.name} exists (use --force to overwrite)")
        return

    print("[run] N=1 ω scan (50 ω values)...")
    omega_vals = [round(v, 1) for v in np.linspace(0.1, 5.0, 50)]
    results_list: list[CombinedOptimizationResult] = []
    for i, omega in enumerate(omega_vals):
        print(f"  [{i + 1}/{len(omega_vals)}] ω={omega:.1f}")
        result = run_combined_single_n_omega(1, omega, ancilla_dim=2)
        results_list.append(result)
        print(f"    Δω = {result.delta_omega_opt:.6f}, R = {result.ratio:.3f}")

    summary = CombinedNScalingScanResult(results=results_list)
    summary.save_parquet(csv_p)
    print(f"[save] {csv_p}")


def generate_n1_2d_slice(force: bool = False) -> None:
    """Step 1: 2D slice scan over (α_xx, α_zz) at optimal drive for selected ω.

    Uses the optimal a_x, a_y, a_z from the N=1 full ω scan, and scans
    α_xx, α_zz ∈ [-20, 20] with α_xz = α_zx = 0.
    """
    csv_p = _parquet_path("n1-2d-slice")

    if csv_p.exists() and not force:
        print(f"[skip] {csv_p.name} exists (use --force to overwrite)")
        return

    print("[run] N=1 2D slice scan (α_xx × α_zz) at ω=0.2...")
    # Use optimal drive from #20260519 at ω=0.2 as fixed
    result = combined_2d_slice(
        omega=0.2,
        a_x=0.0,
        a_y=0.0,
        a_z=0.0,
        ancilla_dim=2,
    )
    result.save_parquet(csv_p)
    print(f"[save] {csv_p}")


def _plot_scaling_analysis() -> pd.DataFrame:
    """Compute scaling exponents from Step 2 and Step 3 results.

    Loads step2-n-scaling and step3-n-scaling Parquet files, fits log-log
    scaling exponent alpha = d(log Delta_omega) / d(log N) for each omega
    value, and returns a DataFrame with columns [scan, omega, alpha,
    n_points].

    Returns:
        DataFrame of scaling exponents, or empty DataFrame if no data.
    """
    rows: list[dict] = []

    # Try to load Step 2 results
    step2_p = _parquet_path("step2-n-scaling")
    if step2_p.exists():
        step2 = CombinedNScalingScanResult.from_parquet(step2_p)
        df2 = step2.to_dataframe()
        for omega in sorted(df2["omega"].unique()):
            sub = df2[df2["omega"] == omega].sort_values("N")
            if len(sub) >= 3:
                log_N = np.log(sub["N"].values)
                log_delta = np.log(sub["delta_omega_opt"].values)
                alpha = np.polyfit(log_N, log_delta, 1)[0]
                rows.append(
                    {
                        "scan": "step2_JA_half",
                        "omega": omega,
                        "alpha": alpha,
                        "n_points": len(sub),
                    }
                )

    # Try to load Step 3 results
    step3_p = _parquet_path("step3-n-scaling")
    if step3_p.exists():
        step3 = CombinedNScalingScanResult.from_parquet(step3_p)
        df3 = step3.to_dataframe()
        for omega in sorted(df3["omega"].unique()):
            sub = df3[df3["omega"] == omega].sort_values("N")
            if len(sub) >= 3:
                log_N = np.log(sub["N"].values)
                log_delta = np.log(sub["delta_omega_opt"].values)
                alpha = np.polyfit(log_N, log_delta, 1)[0]
                rows.append(
                    {
                        "scan": "step3_JA_halfN",
                        "omega": omega,
                        "alpha": alpha,
                        "n_points": len(sub),
                    }
                )

    return pd.DataFrame(rows)


def generate_scaling_analysis(force: bool = False) -> None:
    """Compute scaling exponents from Step 2 and Step 3 results."""
    csv_p = _parquet_path("scaling-analysis")

    if csv_p.exists() and not force:
        print(f"[skip] {csv_p.name} exists (use --force to overwrite)")
        return

    print("[run] Scaling analysis...")
    df = _plot_scaling_analysis()

    csv_p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(csv_p, index=False)
    print(f"[save] {csv_p}")
    if not df.empty:
        for _, row in df.iterrows():
            print(f"  {row['scan']}, ω={row['omega']}: α={row['alpha']:.4f}")


# ============================================================================
# CLI Entry Point
# ============================================================================


def main() -> None:
    """CLI entry point for generating all data and figures."""
    parser = argparse.ArgumentParser(
        description="General 4-Parameter Interaction with ω-Modulated Drive (20260616)",
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

    generators: dict[str, tuple[str, str]] = {
        "decoupled-baseline": (
            "Decoupled Baseline (J_A=1/2)",
            "run_decoupled_baseline",
        ),
        "n1-consistency": (
            "N=1 Consistency Checks",
            "generate_n1_consistency",
        ),
        "n1-omega-scan": (
            "N=1 Full ω Scan",
            "generate_n1_full_omega_scan",
        ),
        "n1-2d-slice": (
            "N=1 2D Slice (α_xx × α_zz)",
            "generate_n1_2d_slice",
        ),
        "step2-n-scaling": (
            "Step 2: N-Scaling (J_A=1/2)",
            "generate_step2_n_scaling",
        ),
        "step3-n-scaling": (
            "Step 3: N-Scaling (J_A=N/2)",
            "generate_step3_n_scaling",
        ),
        "scaling-analysis": (
            "Scaling Exponent Analysis",
            "generate_scaling_analysis",
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
        (_REPORTS_DIR / _REPORT_DATE / d).mkdir(parents=True, exist_ok=True)

    for key in gen_list:
        name, func_name = generators[key]
        print(f"\n{'=' * 60}")
        print(f"  {name}")
        print(f"{'=' * 60}")
        func = globals()[func_name]
        func(force=force)


if __name__ == "__main__":
    main()
