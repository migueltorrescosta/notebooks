"""
Local module for the 2026-07-01 Mixed ω-Modulated Drive with EP/CFI/QFI Comparison.

Contains all code exclusive to this report:
- Core physics simulation (partial ω-modulated Hamiltonian operators, circuit
  evolution, three-way sensitivity computation: EP, CFI, QFI)
- 4D random search, Nelder–Mead refinement, and ω-scan orchestration
- Exclusive plot functions (EP-vs-CFI-vs-QFI trio plot, combined ω-scan)
- Data and figure generation pipeline (``generate_*`` functions)
- CLI entry point for standalone execution

Key difference from #20260519 (fully-modulated drive):
  #20260519: H_A = ω (a_x J_x^A + a_y J_y^A + a_z J_z^A)
  #20260701: H_A = a_x J_x^A + a_y J_y^A + ω a_z J_z^A
  Only the a_z component carries the ω-modulation; a_x, a_y are static.

Usage:
    uv run python reports/r20260701/partial_omega_modulated_drive.py --force
    uv run python reports/r20260701/partial_omega_modulated_drive.py --only decoupled-baseline
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from scipy.linalg import expm

from src.analysis.ancilla_drive_metrology import (
    build_iszz_interaction,
    system_only_bs_unitary,
)
from src.analysis.ancilla_drive_results import (
    Drive2DSliceResult,
    DriveNelderMeadResult,
    DriveOmegaScanResult,
    DriveRandomSearchResult,
)
from src.analysis.ancilla_optimization import (
    build_two_qubit_operators,
    compute_expectation_and_variance,
)
from src.analysis.optimisation_pipeline import (
    TwoPhaseConfig,
    build_nm_result,
    build_rs_result,
    make_4d_objective,
    run_nelder_mead,
    run_omega_scan,
)
from src.analysis.slice_scan import sequential_grid_scan
from src.utils.constants import I_4
from src.utils.serialization import ParquetSerializable

# ============================================================================
# Physical Constants  (local copies; not promoted to src/)
# ============================================================================

DEFAULT_T_BS: float = np.pi / 2.0  # 50/50 beam splitter
DEFAULT_T_HOLD: float = 10.0  # Holding time (SQL = 0.1)
DEFAULT_PSI0: np.ndarray = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)  # |00⟩
SQL_REFERENCE: float = 1.0 / DEFAULT_T_HOLD  # Δω_SQL = 0.1
DRIVE_BOUNDS: tuple[float, float] = (-5.0, 5.0)  # Range for all coefficients
FD_STEP: float = 1e-6  # Finite-difference step for derivatives
PROB_FLOOR: float = 1e-12  # Floor for probability in CFI denominator


def _configure_environment() -> None:
    """Set non-interactive matplotlib backend and OMP thread limit.

    Must be called before any plotting or numerical routines that spawn
    threads.  Safe to call multiple times (guard checks existing env vars).
    """
    if "MPLBACKEND" not in os.environ:
        os.environ["MPLBACKEND"] = "Agg"
    if "OMP_NUM_THREADS" not in os.environ:
        os.environ["OMP_NUM_THREADS"] = "1"
    sns.set_theme(style="whitegrid")


# ============================================================================
# Hamiltonian Construction  (partial ω-modulation)
# ============================================================================


def build_partial_drive_hamiltonian(
    omega: float,
    a_x: float,
    a_y: float,
    a_z: float,
    ops: dict[str, np.ndarray],
) -> np.ndarray:
    """Build the partially ω-modulated ancilla drive Hamiltonian.

    H_A = a_x J_x^A + a_y J_y^A + ω a_z J_z^A

    Only the a_z component is modulated by ω; a_x and a_y are static.
    This is the key difference from #20260519 where H_A = ω (a_x J_x^A + ...).

    Args:
        omega: Unknown phase rate parameter (only enters a_z term).
        a_x: Static coefficient for J_x^A (no ω-modulation).
        a_y: Static coefficient for J_y^A (no ω-modulation).
        a_z: ω-modulated coefficient for J_z^A.
        ops: Two-qubit operators from build_two_qubit_operators().

    Returns:
        4×4 Hermitian matrix representing the partially ω-modulated ancilla drive.
    """
    H = np.zeros((4, 4), dtype=complex)
    if a_x != 0.0:
        H += a_x * ops["Jx_A"]
    if a_y != 0.0:
        H += a_y * ops["Jy_A"]
    if a_z != 0.0:
        H += omega * a_z * ops["Jz_A"]  # Only a_z has ω-modulation
    return 0.5 * (H + H.conj().T)


def build_partial_hold_hamiltonian(
    omega: float,
    a_x: float,
    a_y: float,
    a_z: float,
    a_zz: float,
    ops: dict[str, np.ndarray],
) -> np.ndarray:
    """Build the total holding Hamiltonian with partial ω-modulation.

    H = ω J_z^S + H_A + H_int
      = ω J_z^S + (a_x J_x^A + a_y J_y^A + ω a_z J_z^A) + a_zz J_z^S ⊗ J_z^A

    ∂H/∂ω = J_z^S + a_z J_z^A  (no a_x, a_y contributions to the derivative).

    Args:
        omega: Unknown phase rate parameter.
        a_x: Static ancilla J_x drive coefficient.
        a_y: Static ancilla J_y drive coefficient.
        a_z: ω-modulated ancilla J_z drive coefficient.
        a_zz: Ising interaction coefficient.
        ops: Two-qubit operators from build_two_qubit_operators().

    Returns:
        4×4 Hermitian Hamiltonian matrix.
    """
    H = omega * ops["Jz_S"]
    H += build_partial_drive_hamiltonian(omega, a_x, a_y, a_z, ops)
    H += build_iszz_interaction(a_zz, ops)
    return 0.5 * (H + H.conj().T)


def partial_hold_unitary(
    t_hold: float,
    omega: float,
    a_x: float,
    a_y: float,
    a_z: float,
    a_zz: float,
    ops: dict[str, np.ndarray],
) -> np.ndarray:
    """Holding-time unitary for the partially ω-modulated ancilla protocol.

    U_hold(t_hold) = exp(-i t_hold H)

    Args:
        t_hold: Holding-time strength.
        omega: True phase rate parameter.
        a_x: Static ancilla J_x drive coefficient.
        a_y: Static ancilla J_y drive coefficient.
        a_z: ω-modulated ancilla J_z drive coefficient.
        a_zz: Ising interaction coefficient.
        ops: Two-qubit operators from build_two_qubit_operators().

    Returns:
        4×4 unitary matrix.
    """
    H = build_partial_hold_hamiltonian(omega, a_x, a_y, a_z, a_zz, ops)
    U = expm(-1j * t_hold * H)
    assert np.allclose(U @ U.conj().T, I_4, atol=1e-12), (
        f"Partial hold unitary not unitary for t_hold={t_hold}, ω={omega}"
    )
    return U


# ============================================================================
# Circuit Evolution
# ============================================================================


def evolve_partial_circuit(
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
    """Run the full partially ω-modulated ancilla MZI circuit.

    |ψ_final⟩ = U_BS_S · U_hold(t_hold) · U_BS_S · |ψ₀⟩

    Args:
        psi0: Initial 4-vector (must be normalised).
        T_BS: Beam-splitter duration (both BS identical).
        t_hold: Holding-time strength.
        omega: Phase rate parameter.
        a_x: Static ancilla J_x drive coefficient.
        a_y: Static ancilla J_y drive coefficient.
        a_z: ω-modulated ancilla J_z drive coefficient.
        a_zz: Ising interaction coefficient.
        ops: Two-qubit operators.

    Returns:
        Final normalised 4-vector state.
    """
    assert np.isclose(np.linalg.norm(psi0), 1.0), "Initial state must be normalised"

    U_bs = system_only_bs_unitary(T_BS)
    psi = U_bs @ psi0
    psi = partial_hold_unitary(t_hold, omega, a_x, a_y, a_z, a_zz, ops) @ psi
    psi = U_bs @ psi

    assert np.isclose(np.linalg.norm(psi), 1.0), "Final state must be normalised"
    return psi


# ============================================================================
# Sensitivity Computation  (three metrics)
# ============================================================================


def compute_partial_sensitivity_ep(
    psi0: np.ndarray,
    T_BS: float,
    t_hold: float,
    omega_true: float,
    a_x: float,
    a_y: float,
    a_z: float,
    a_zz: float,
    ops: dict[str, np.ndarray],
    fd_step: float = FD_STEP,
    meas_op: np.ndarray | None = None,
) -> float:
    """Compute the error-propagation sensitivity Δω_EP.

    Δω_EP = sqrt(Var(J_z^S)) / |∂⟨J_z^S⟩/∂ω|

    The central finite-difference step captures the ω-dependence from both
    H_S (= ω J_z^S) and the a_z component of H_A (= ω a_z J_z^A).

    Args:
        psi0: Initial 4-vector (product state).
        T_BS: Beam-splitter duration.
        t_hold: Holding-time strength.
        omega_true: True phase rate parameter.
        a_x: Static ancilla J_x drive coefficient.
        a_y: Static ancilla J_y drive coefficient.
        a_z: ω-modulated ancilla J_z drive coefficient.
        a_zz: Ising interaction coefficient.
        ops: Two-qubit operators (must contain 'Jz_S').
        fd_step: Finite-difference step size (default 1e-6).
        meas_op: Measurement operator. Defaults to ops['Jz_S'] (S-only).

    Returns:
        Sensitivity Δω_EP (positive float). Returns inf if derivative is zero.
    """
    if meas_op is None:
        meas_op = ops["Jz_S"]

    # Evaluate at omega_true
    psi = evolve_partial_circuit(
        psi0,
        T_BS,
        t_hold,
        omega_true,
        a_x,
        a_y,
        a_z,
        a_zz,
        ops,
    )
    _, var = compute_expectation_and_variance(psi, meas_op)

    # Central finite difference for ∂⟨O⟩/∂ω
    psi_plus = evolve_partial_circuit(
        psi0,
        T_BS,
        t_hold,
        omega_true + fd_step,
        a_x,
        a_y,
        a_z,
        a_zz,
        ops,
    )
    psi_minus = evolve_partial_circuit(
        psi0,
        T_BS,
        t_hold,
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

    if abs(d_exp) < 1e-12 or var < 1e-15:
        return float("inf")

    return float(np.sqrt(var) / abs(d_exp))


def _compute_marginal_probability_plus(
    psi: np.ndarray,
) -> float:
    """Compute P(+1/2 | ω) = probability that J_z^S = +1/2.

    For the 4-vector |ψ⟩ in the {|00⟩, |01⟩, |10⟩, |11⟩} basis:
      P(+) = |⟨00|ψ⟩|² + |⟨01|ψ⟩|²   (system m=+1/2, ancilla any state)

    Args:
        psi: Normalised 4-vector final state.

    Returns:
        Probability P(+1/2 | ω) ∈ [0, 1].
    """
    return float(abs(psi[0]) ** 2 + abs(psi[1]) ** 2)


def compute_partial_sensitivity_cfi(
    psi0: np.ndarray,
    T_BS: float,
    t_hold: float,
    omega_true: float,
    a_x: float,
    a_y: float,
    a_z: float,
    a_zz: float,
    ops: dict[str, np.ndarray],
    fd_step: float = FD_STEP,
) -> float:
    """Compute the Classical Fisher Information sensitivity Δω_CFI.

    For a binary measurement (J_z^S eigenvalues ±1/2):
      P(+) = probability of +1/2 outcome
      F_C = (∂P/∂ω)² / (P(1-P))
      Δω_CFI = 1 / √F_C

    Due to the binary nature of J_z^S, this is EXACTLY equal to Δω_EP
    (the error-propagation based sensitivity). This function is provided
    as an independent computation to verify the equivalence numerically.

    Args:
        psi0: Initial 4-vector (product state).
        T_BS: Beam-splitter duration.
        t_hold: Holding-time strength.
        omega_true: True phase rate parameter.
        a_x: Static ancilla J_x drive coefficient.
        a_y: Static ancilla J_y drive coefficient.
        a_z: ω-modulated ancilla J_z drive coefficient.
        a_zz: Ising interaction coefficient.
        ops: Two-qubit operators (unused, kept for signature consistency).
        fd_step: Finite-difference step size (default 1e-6).

    Returns:
        Sensitivity Δω_CFI (positive float). Returns inf if derivative is zero
        or probability is 0 or 1.
    """
    # Compute P(+) at omega_true
    psi = evolve_partial_circuit(
        psi0,
        T_BS,
        t_hold,
        omega_true,
        a_x,
        a_y,
        a_z,
        a_zz,
        ops,
    )
    p_plus = _compute_marginal_probability_plus(psi)

    # Central finite difference for ∂P/∂ω
    psi_plus = evolve_partial_circuit(
        psi0,
        T_BS,
        t_hold,
        omega_true + fd_step,
        a_x,
        a_y,
        a_z,
        a_zz,
        ops,
    )
    psi_minus = evolve_partial_circuit(
        psi0,
        T_BS,
        t_hold,
        omega_true - fd_step,
        a_x,
        a_y,
        a_z,
        a_zz,
        ops,
    )
    p_plus_plus = _compute_marginal_probability_plus(psi_plus)
    p_plus_minus = _compute_marginal_probability_plus(psi_minus)
    dP = (p_plus_plus - p_plus_minus) / (2.0 * fd_step)

    # P(1-P) product
    p_product = p_plus * (1.0 - p_plus)
    if p_product < PROB_FLOOR or abs(dP) < 1e-12:
        return float("inf")

    f_c = (dP * dP) / p_product
    return 1.0 / np.sqrt(f_c)


def compute_partial_sensitivity_qfi(
    psi0: np.ndarray,
    T_BS: float,
    t_hold: float,
    omega_true: float,
    a_x: float,
    a_y: float,
    a_z: float,
    a_zz: float,
    ops: dict[str, np.ndarray],
    fd_step: float = FD_STEP,
) -> float:
    """Compute the Quantum Fisher Information sensitivity Δω_QFI.

    For a pure state |ψ(ω)⟩:
      F_Q = 4 (⟨ψ'|ψ'⟩ - |⟨ψ'|ψ⟩|²)
    where |ψ'⟩ = ∂|ψ⟩/∂ω is computed via central finite differences.

    Args:
        psi0: Initial 4-vector (product state).
        T_BS: Beam-splitter duration.
        t_hold: Holding-time strength.
        omega_true: True phase rate parameter.
        a_x: Static ancilla J_x drive coefficient.
        a_y: Static ancilla J_y drive coefficient.
        a_z: ω-modulated ancilla J_z drive coefficient.
        a_zz: Ising interaction coefficient.
        ops: Two-qubit operators (unused, kept for signature consistency).
        fd_step: Finite-difference step size (default 1e-6).

    Returns:
        Sensitivity Δω_QFI (positive float). Returns inf if F_Q is zero.
    """
    # Compute |ψ⟩ at omega_true
    psi = evolve_partial_circuit(
        psi0,
        T_BS,
        t_hold,
        omega_true,
        a_x,
        a_y,
        a_z,
        a_zz,
        ops,
    )

    # Compute |ψ'⟩ via central finite difference
    psi_plus = evolve_partial_circuit(
        psi0,
        T_BS,
        t_hold,
        omega_true + fd_step,
        a_x,
        a_y,
        a_z,
        a_zz,
        ops,
    )
    psi_minus = evolve_partial_circuit(
        psi0,
        T_BS,
        t_hold,
        omega_true - fd_step,
        a_x,
        a_y,
        a_z,
        a_zz,
        ops,
    )
    psi_prime = (psi_plus - psi_minus) / (2.0 * fd_step)

    # F_Q = 4 (⟨ψ'|ψ'⟩ - |⟨ψ'|ψ⟩|²)
    overlap = np.real(np.vdot(psi_prime, psi_prime))
    projection = abs(np.vdot(psi_prime, psi)) ** 2
    f_q = 4.0 * (overlap - projection)

    if f_q < PROB_FLOOR:
        return float("inf")

    return 1.0 / np.sqrt(f_q)


def compute_all_sensitivities(
    psi0: np.ndarray,
    T_BS: float,
    t_hold: float,
    omega_true: float,
    a_x: float,
    a_y: float,
    a_z: float,
    a_zz: float,
    ops: dict[str, np.ndarray],
    fd_step: float = FD_STEP,
) -> dict[str, float]:
    """Compute all three sensitivity metrics for a single configuration.

    Returns a dictionary with keys:
      - 'delta_omega_ep': Error-propagation sensitivity
      - 'delta_omega_cfi': Classical Fisher Information sensitivity
      - 'delta_omega_qfi': Quantum Fisher Information sensitivity
      - 'expectation_Jz': ⟨J_z^S⟩
      - 'variance_Jz': Var(J_z^S)
      - 'fisher_classical': F_C
      - 'fisher_quantum': F_Q

    Args:
        psi0: Initial 4-vector (product state).
        T_BS: Beam-splitter duration.
        t_hold: Holding-time strength.
        omega_true: True phase rate parameter.
        a_x: Static ancilla J_x drive coefficient.
        a_y: Static ancilla J_y drive coefficient.
        a_z: ω-modulated ancilla J_z drive coefficient.
        a_zz: Ising interaction coefficient.
        ops: Two-qubit operators (must contain 'Jz_S').
        fd_step: Finite-difference step size (default 1e-6).

    Returns:
        Dictionary with all sensitivity metrics.
    """
    # Final state at omega_true
    psi = evolve_partial_circuit(
        psi0,
        T_BS,
        t_hold,
        omega_true,
        a_x,
        a_y,
        a_z,
        a_zz,
        ops,
    )
    exp_val, var_val = compute_expectation_and_variance(psi, ops["Jz_S"])

    # Derivatives at ω ± δ
    psi_plus = evolve_partial_circuit(
        psi0,
        T_BS,
        t_hold,
        omega_true + fd_step,
        a_x,
        a_y,
        a_z,
        a_zz,
        ops,
    )
    psi_minus = evolve_partial_circuit(
        psi0,
        T_BS,
        t_hold,
        omega_true - fd_step,
        a_x,
        a_y,
        a_z,
        a_zz,
        ops,
    )

    # ⟨J_z^S⟩ derivative
    exp_plus = np.real(psi_plus.conj() @ ops["Jz_S"] @ psi_plus)
    exp_minus = np.real(psi_minus.conj() @ ops["Jz_S"] @ psi_minus)
    d_exp = (exp_plus - exp_minus) / (2.0 * fd_step)

    # Error-propagation sensitivity
    if abs(d_exp) < 1e-12 or var_val < 1e-15:
        delta_ep = float("inf")
    else:
        delta_ep = float(np.sqrt(var_val) / abs(d_exp))

    # CFI from binary measurement: F_C = (∂P/∂ω)² / (P(1-P))
    p_plus = _compute_marginal_probability_plus(psi)
    p_plus_plus = _compute_marginal_probability_plus(psi_plus)
    p_plus_minus = _compute_marginal_probability_plus(psi_minus)
    dP = (p_plus_plus - p_plus_minus) / (2.0 * fd_step)

    p_product = p_plus * (1.0 - p_plus)
    if p_product < PROB_FLOOR or abs(dP) < 1e-12:
        delta_cfi = float("inf")
        f_c = 0.0
    else:
        f_c = (dP * dP) / p_product
        delta_cfi = 1.0 / np.sqrt(f_c)

    # QFI via pure-state formula: F_Q = 4(⟨ψ'|ψ'⟩ - |⟨ψ'|ψ⟩|²)
    psi_prime = (psi_plus - psi_minus) / (2.0 * fd_step)
    overlap = np.real(np.vdot(psi_prime, psi_prime))
    projection = abs(np.vdot(psi_prime, psi)) ** 2
    f_q = 4.0 * (overlap - projection)

    if f_q < PROB_FLOOR:
        delta_qfi = float("inf")
    else:
        delta_qfi = 1.0 / np.sqrt(f_q)

    return {
        "delta_omega_ep": delta_ep,
        "delta_omega_cfi": delta_cfi,
        "delta_omega_qfi": delta_qfi,
        "expectation_Jz": float(exp_val),
        "variance_Jz": float(var_val),
        "fisher_classical": float(f_c),
        "fisher_quantum": float(f_q),
    }


# ============================================================================
# Result Dataclass
# ============================================================================


@dataclass
class PartialOmegaDriveResult(ParquetSerializable):
    """Sensitivity results for a single configuration of the partial ω-modulated drive.

    Stores all input parameters alongside the three sensitivity metrics
    and related quantities. Fully self-describing for Parquet serialisation.

    Attributes:
        omega: Phase rate parameter.
        a_x: Static ancilla J_x drive coefficient.
        a_y: Static ancilla J_y drive coefficient.
        a_z: ω-modulated ancilla J_z drive coefficient.
        a_zz: Ising interaction coefficient.
        t_hold: Holding-time strength.
        delta_omega_ep: Error-propagation sensitivity Δω_EP.
        delta_omega_cfi: Classical Fisher Information sensitivity Δω_CFI.
        delta_omega_qfi: Quantum Fisher Information sensitivity Δω_QFI.
        expectation_Jz: ⟨J_z^S⟩ at the operating point.
        variance_Jz: Var(J_z^S) at the operating point.
        fisher_classical: Classical Fisher Information F_C.
        fisher_quantum: Quantum Fisher Information F_Q.
        sql: SQL reference = 1/t_hold.
    """

    omega: float
    a_x: float
    a_y: float
    a_z: float
    a_zz: float
    t_hold: float
    delta_omega_ep: float
    delta_omega_cfi: float
    delta_omega_qfi: float
    expectation_Jz: float
    variance_Jz: float
    fisher_classical: float
    fisher_quantum: float
    sql: float = SQL_REFERENCE

    _PARQUET_COLUMNS: ClassVar[list[str]] = [
        "omega",
        "a_x",
        "a_y",
        "a_z",
        "a_zz",
        "t_hold",
        "delta_omega_ep",
        "delta_omega_cfi",
        "delta_omega_qfi",
        "expectation_Jz",
        "variance_Jz",
        "fisher_classical",
        "fisher_quantum",
        "sql",
    ]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "omega": [self.omega],
                "a_x": [self.a_x],
                "a_y": [self.a_y],
                "a_z": [self.a_z],
                "a_zz": [self.a_zz],
                "t_hold": [self.t_hold],
                "delta_omega_ep": [self.delta_omega_ep],
                "delta_omega_cfi": [self.delta_omega_cfi],
                "delta_omega_qfi": [self.delta_omega_qfi],
                "expectation_Jz": [self.expectation_Jz],
                "variance_Jz": [self.variance_Jz],
                "fisher_classical": [self.fisher_classical],
                "fisher_quantum": [self.fisher_quantum],
                "sql": [self.sql],
            },
        )

    @classmethod
    def from_parquet(cls, path: str | Path) -> PartialOmegaDriveResult:
        df = pd.read_parquet(path)
        cls._validate_columns(df)
        return cls(
            omega=float(df["omega"].iloc[0]),
            a_x=float(df["a_x"].iloc[0]),
            a_y=float(df["a_y"].iloc[0]),
            a_z=float(df["a_z"].iloc[0]),
            a_zz=float(df["a_zz"].iloc[0]),
            t_hold=float(df["t_hold"].iloc[0]),
            delta_omega_ep=float(df["delta_omega_ep"].iloc[0]),
            delta_omega_cfi=float(df["delta_omega_cfi"].iloc[0]),
            delta_omega_qfi=float(df["delta_omega_qfi"].iloc[0]),
            expectation_Jz=float(df["expectation_Jz"].iloc[0]),
            variance_Jz=float(df["variance_Jz"].iloc[0]),
            fisher_classical=float(df["fisher_classical"].iloc[0]),
            fisher_quantum=float(df["fisher_quantum"].iloc[0]),
            sql=float(df["sql"].iloc[0]),
        )


# ============================================================================
# Decoupled Baseline
# ============================================================================

# compute_decoupled_baseline imported from src/analysis/ancilla_drive_scans
# at the top of the file. For the decoupled case (a_x = a_y = a_z = a_zz = 0),
# the partial ω-modulation reduces to the same standard single-qubit MZI as
# the full ω-modulation, so the shared function is exact.


# ============================================================================
# 2D Slice Scan
# ============================================================================


def partial_2d_slice(
    omega: float,
    drive_range: tuple[float, float] = DRIVE_BOUNDS,
    azz_range: tuple[float, float] = DRIVE_BOUNDS,
    n_drive: int = 201,
    n_azz: int = 201,
    slice_type: str = "ax",
    t_hold: float = DEFAULT_T_HOLD,
    T_BS: float = DEFAULT_T_BS,
) -> Drive2DSliceResult:
    """Run a 2D slice scan over (a_drive, a_zz) with partial ω-modulation.

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
        return compute_partial_sensitivity_ep(
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


def _make_rs_nm_fns(
    omega: float,
    ops: dict[str, np.ndarray],
    psi0: np.ndarray = DEFAULT_PSI0,
    t_hold: float = DEFAULT_T_HOLD,
    T_BS: float = DEFAULT_T_BS,
) -> tuple[Callable, Callable]:
    """Build RS and NM worker functions for the two-phase pipeline.

    Returns (rs_fn, nm_fn) where each accepts (n_samples, seed, **kw) /
    (x0, seed, **kw) respectively, consistent with the callback-based
    ``run_omega_scan`` / ``run_two_phase_pipeline`` interface.

    Args:
        omega: Phase rate parameter.
        ops: Two-qubit operators.
        psi0: Initial state.
        t_hold: Holding time.
        T_BS: Beam-splitter duration.

    Returns:
        Tuple of (random_search_fn, nelder_mead_fn).
    """
    raw_obj = make_4d_objective(
        compute_partial_sensitivity_ep,
        psi0=psi0,
        T_BS=T_BS,
        t_hold=t_hold,
        omega=omega,
        ops=ops,
    )

    def _rs_fn(n_samples: int, seed: int, **kw: Any) -> DriveRandomSearchResult:
        return build_rs_result(
            raw_obj,
            n_samples,
            seed,
            omega=omega,
            sql=1.0 / t_hold,
            t_hold=t_hold,
        )

    def _nm_fn(x0: np.ndarray, seed: int, **kw: Any) -> DriveNelderMeadResult:
        return build_nm_result(
            raw_obj,
            x0,
            omega=omega,
            ops=ops,
            psi0=psi0,
            evolve_fn=lambda psi, ax, ay, az, azz, _ops: evolve_partial_circuit(
                psi,
                T_BS,
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

    return _rs_fn, _nm_fn


# ============================================================================
# 4D Random Search
# ============================================================================


def partial_random_search(
    omega: float,
    n_samples: int = 500,
    bounds: tuple[float, float] = DRIVE_BOUNDS,
    t_hold: float = DEFAULT_T_HOLD,
    T_BS: float = DEFAULT_T_BS,
    seed: int | None = 42,
) -> DriveRandomSearchResult:
    """Random search over the 4D parameter space (a_x, a_y, a_z, a_zz).

    Uses the partially ω-modulated drive H_A = a_x J_x^A + a_y J_y^A + ω a_z J_z^A.

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
    raw_obj = make_4d_objective(
        compute_partial_sensitivity_ep,
        psi0=DEFAULT_PSI0,
        T_BS=T_BS,
        t_hold=t_hold,
        omega=omega,
        ops=ops,
    )
    return build_rs_result(
        raw_obj,
        n_samples,
        seed if seed is not None else 42,
        omega=omega,
        sql=1.0 / t_hold,
        t_hold=t_hold,
    )


# ============================================================================
# Nelder-Mead Optimisation
# ============================================================================


def run_partial_nelder_mead(
    omega_true: float,
    x0: np.ndarray | None = None,
    seed: int | None = None,
    maxiter: int = 5000,
    xatol: float = 1e-8,
    fatol: float = 1e-8,
    adaptive: bool = True,
    bounds: tuple[float, float] = DRIVE_BOUNDS,
    t_hold: float = DEFAULT_T_HOLD,
    T_BS: float = DEFAULT_T_BS,
    track_history: bool = False,
) -> DriveNelderMeadResult:
    """Run Nelder-Mead optimisation for the partially ω-modulated protocol.

    Args:
        omega_true: True phase rate parameter.
        x0: Initial 4-parameter vector [ax, ay, az, azz]. Random if None.
        seed: Random seed (used if x0 is None).
        maxiter: Maximum Nelder-Mead iterations.
        xatol: Absolute parameter tolerance.
        fatol: Absolute function tolerance.
        adaptive: Use adaptive Nelder-Mead parameters.
        bounds: (min, max) for all four parameters.
        t_hold: Holding time.
        T_BS: Beam-splitter duration.
        track_history: If True, record objective values per iteration.

    Returns:
        DriveNelderMeadResult.
    """
    ops = build_two_qubit_operators()

    if x0 is None:
        rng = np.random.default_rng(seed)
        lo, hi = bounds
        x0 = rng.uniform(lo, hi, size=4)
    else:
        x0 = np.asarray(x0, dtype=float)
        assert x0.shape == (4,), f"x0 must have 4 elements, got {x0.shape}"

    raw_obj = make_4d_objective(
        compute_partial_sensitivity_ep,
        psi0=DEFAULT_PSI0,
        T_BS=T_BS,
        t_hold=t_hold,
        omega=omega_true,
        ops=ops,
    )
    nm = run_nelder_mead(
        raw_obj,
        x0=x0,
        bounds=bounds,
        maxiter=maxiter,
        xatol=xatol,
        fatol=fatol,
        adaptive=adaptive,
        track_history=track_history,
    )

    opt_p = nm["x_opt"]
    psi_final = evolve_partial_circuit(
        DEFAULT_PSI0,
        T_BS,
        t_hold,
        omega_true,
        float(opt_p[0]),
        float(opt_p[1]),
        float(opt_p[2]),
        float(opt_p[3]),
        ops,
    )
    exp_val, var_val = compute_expectation_and_variance(psi_final, ops["Jz_S"])

    return DriveNelderMeadResult(
        delta_omega_opt=nm["fun_opt"],
        params_opt=opt_p,
        omega_true=omega_true,
        success=nm["success"],
        nfev=nm["nfev"],
        message=nm["message"],
        expectation_Jz=exp_val,
        variance_Jz=var_val,
        history=nm["history"],
    )


# ============================================================================
# ω Scan with Random Search + Nelder-Mead Refinement
# ============================================================================


def run_partial_omega_scan(
    omega_values: list[float] | np.ndarray,
    n_random: int = 500,
    n_nm_refine: int = 50,
    seed: int | None = 42,
    maxiter: int = 5000,
    bounds: tuple[float, float] = DRIVE_BOUNDS,
    t_hold: float = DEFAULT_T_HOLD,
    T_BS: float = DEFAULT_T_BS,
) -> DriveOmegaScanResult:
    """Scan over ω values with 4D random search and Nelder-Mead refinement.

    For each ω:
    1. Run `n_random` random evaluations in the 4D parameter space.
    2. Select the best `n_nm_refine` points.
    3. Run Nelder-Mead refinement from each selected point.
    4. Record the best overall result.

    Args:
        omega_values: ω values to scan.
        n_random: Number of random search points per ω.
        n_nm_refine: Number of Nelder-Mead refinements per ω.
        seed: Base random seed (incremented per ω).
        maxiter: Maximum Nelder-Mead iterations.
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
        rs_fn, _ = _make_rs_nm_fns(kw["omega"], ops, t_hold=t_hold, T_BS=T_BS)
        return rs_fn(n_samples, seed)

    def _nm_fn(x0: np.ndarray, seed: int, **kw: Any) -> DriveNelderMeadResult:
        _, nm_fn = _make_rs_nm_fns(kw["omega_true"], ops, t_hold=t_hold, T_BS=T_BS)
        return nm_fn(x0, seed)

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


# ============================================================================
# CLI Entry Point
# ============================================================================


def main() -> None:
    """CLI entry point that delegates to the generation module.

    Import is deferred to avoid circular-import issues: the generation module
    imports symbols from this module at module-load time, and this function
    is only called at script invocation time, when both modules are fully
    loaded.
    """
    import importlib

    gen = importlib.import_module(
        "reports.r20260701._pomd_generation",
    )
    gen.run_cli()


if __name__ == "__main__":
    main()
