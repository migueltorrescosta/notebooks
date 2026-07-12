"""
Experiment module for the 2026-06-28 Free-Ancilla ω-Modulated Drive with
Weighted Joint Measurement report.

Combines two independent SQL-violation mechanisms:
  - Free-ancilla initial state (Bloch angles θ_A, φ_A)
  - Weighted joint measurement M(ψ) = cosψ·J_z^S + sinψ·J_z^A

Three experiments:
  Exp. 1: N=1 compounding (qubit ancilla J_A=1/2, 7D optimisation)
  Exp. 2: N>1 scaling with qubit ancilla (J_A=1/2)
  Exp. 3: N>1 scaling with multi-particle ancilla (J_A=N/2, free CSS state)

Circuit: BS_S → Hold → BS_S → measure M(ψ).

Usage:
    uv run python reports/r20260628/free_ancilla_joint_measurement.py --force
    uv run python reports/r20260628/free_ancilla_joint_measurement.py --only exp1
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from collections.abc import Callable

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.optimize import minimize

from src.algorithms.coherent_spin_state import coherent_spin_state
from src.analysis.ancilla_optimization import (
    compute_expectation_and_variance,
    single_qubit_state,
)
from src.analysis.optimisation_pipeline import (
    TwoPhaseConfig,
    run_two_phase_pipeline,
)
from src.analysis.sensitivity_metrics import sql_reference
from src.physics.bipartite_operators import (
    build_operators,
    build_system_only_bs_unitary,
)
from src.physics.joint_measurement import (
    build_bipartite_joint_measurement_operator,
    build_joint_measurement_operator,
)
from src.physics.n_particle_drive import (
    build_n_particle_operators,
    compute_n_particle_sensitivity,
    evolve_n_particle_circuit,
)
from src.utils.monte_carlo import marsaglia_ball_sample
from src.utils.paths import report_path_fn
from src.utils.serialization import ParquetSerializable

# ============================================================================
# Constants
# ============================================================================

T_HOLD: float = 10.0  # Holding time (fixed)
T_BS: float = np.pi / 2.0  # 50/50 beam splitter
FD_STEP: float = 1e-6  # Finite-difference step
DRIVE_BOUNDS: tuple[float, float] = (-5.0, 5.0)  # Drive parameter bounds
R_MAX: float = 10.0  # 3-ball radius for drive vector (Exp 1&2)
R_MAX_BIPARTITE: float = 5.0  # Reduced 3-ball radius for Exp 3 (stability)
PSI_BOUNDS: tuple[float, float] = (-np.pi, np.pi)  # Measurement angle bounds
THETA_A_BOUNDS: tuple[float, float] = (0.0, np.pi)  # Ancilla polar angle
PHI_A_BOUNDS: tuple[float, float] = (0.0, 2.0 * np.pi)  # Ancilla azimuth
AZZ_BOUNDS: tuple[float, float] = (-5.0, 5.0)  # Ising coupling bounds

# ω values
OMEGA_VALS: list[float] = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
OMEGA_VALS_QUICK: list[float] = [0.1, 0.2, 0.5]
OMEGA_VALS_SCALING: list[float] = [0.1, 0.2, 0.5, 1.0, 2.0]

# N values
N_VALS: list[int] = list(range(1, 21))

# Random search parameters
N_RANDOM_EXP1: int = 5000  # Exp 1 (N=1, 6 ω values)
N_RANDOM_EXP2: int = 3000  # Exp 2 (N>1, per (N, ω))
N_RANDOM_EXP3: int = 2000  # Exp 3 (multi-particle, reduced)
N_NM_REFINE_EXP1: int = 60  # Nelder-Mead refinements per ω (Exp 1)
N_NM_REFINE_EXP2: int = 40  # Nelder-Mead refinements per ω (Exp 2)
N_NM_REFINE_EXP3: int = 30  # Nelder-Mead refinements per ω (Exp 3)
NM_MAXITER: int = 5000

# ============================================================================
# Path Helpers
# ============================================================================

REPORTS_DIR = Path(__file__).resolve().parent.parent.parent / "reports"
REPORT_DATE = "20260628"

_parquet_path, _fig_path = report_path_fn(REPORTS_DIR, REPORT_DATE)


# ============================================================================
# State Preparation
# ============================================================================


def n_particle_free_ancilla_initial_state(
    N: int,
    theta_A: float,
    phi_A: float,
) -> np.ndarray:
    """Free-ancilla initial state in the N-particle space (J_A=1/2).

    |Ψ₀⟩ = |J_S, J_S⟩_S ⊗ (cos(θ_A/2)|1,0⟩_A + e^{iφ_A} sin(θ_A/2)|0,1⟩_A)

    Dimension: 2(N+1). The system is in the top Dicke state.

    Args:
        N: Number of system particles.
        theta_A: Ancilla polar angle ∈ [0, π].
        phi_A: Ancilla azimuthal angle ∈ [0, 2π).

    Returns:
        Normalised complex vector of length 2(N+1).
    """
    d_sys = N + 1
    d_tot = 2 * d_sys

    # System: |J_S, J_S⟩ (first Dicke basis vector)
    psi_sys = np.zeros(d_sys, dtype=complex)
    psi_sys[0] = 1.0

    # Ancilla: free qubit state
    psi_anc = single_qubit_state(theta_A, phi_A)

    state = np.kron(psi_sys, psi_anc).astype(complex)
    assert state.shape == (d_tot,), f"Expected ({d_tot},), got {state.shape}"
    assert np.isclose(np.linalg.norm(state), 1.0), (
        "Free-ancilla N-particle initial state must be normalised"
    )
    return state


def multi_particle_free_css_initial_state(
    N: int,
    theta_A: float,
    phi_A: float,
) -> np.ndarray:
    """Free-CSS initial state for the multi-particle ancilla (J_A=N/2).

    |Ψ₀⟩ = |J_S, J_S⟩_S ⊗ |θ_A, φ_A⟩_A

    Dimension: (N+1)^2. The system is in the top Dicke state, the ancilla
    is a coherent spin state with polar angle θ_A and azimuth φ_A.

    Args:
        N: Number of particles per subsystem.
        theta_A: Ancilla polar angle ∈ [0, π].
        phi_A: Ancilla azimuthal angle ∈ [0, 2π).

    Returns:
        Normalised complex vector of length (N+1)^2.
    """
    d_sys = N + 1
    d_tot = d_sys * d_sys

    # System: |J_S, J_S⟩ (first Dicke basis vector)
    psi_sys = np.zeros(d_sys, dtype=complex)
    psi_sys[0] = 1.0

    # Ancilla: CSS state
    J_A = N / 2.0
    psi_anc = coherent_spin_state(J_A, theta_A, phi_A)

    state = np.kron(psi_sys, psi_anc).astype(complex)
    assert state.shape == (d_tot,), f"Expected ({d_tot},), got {state.shape}"
    assert np.isclose(np.linalg.norm(state), 1.0), (
        "Free-CSS initial state must be normalised"
    )
    return state


# ============================================================================
# Sensitivity — J_A=1/2 (Experiments 1 & 2)
# ============================================================================


def compute_qubit_free_ancilla_joint_sensitivity(
    N: int,
    omega_true: float,
    theta_A: float,
    phi_A: float,
    a_x: float,
    a_y: float,
    a_z: float,
    a_zz: float,
    psi: float,
    *,
    t_hold: float = T_HOLD,
    T_bs: float = T_BS,
    fd_step: float = FD_STEP,
) -> float:
    """Sensitivity Δω for N-particle system + qubit ancilla + joint meas.

    Builds the free-ancilla initial state, the joint measurement operator
    M(ψ), and delegates to compute_n_particle_sensitivity.

    Args:
        N: Number of system particles.
        omega_true: True phase rate parameter.
        theta_A: Ancilla polar angle.
        phi_A: Ancilla azimuthal angle.
        a_x: J_x^A drive coefficient.
        a_y: J_y^A drive coefficient.
        a_z: J_z^A drive coefficient.
        a_zz: Ising interaction coefficient.
        psi: Measurement weight angle.
        t_hold: Holding time.
        T_bs: Beam-splitter duration.
        fd_step: Finite-difference step size.

    Returns:
        Sensitivity Δω (positive float). Returns inf at fringe extremum.
    """
    ops = build_n_particle_operators(N)
    psi0 = n_particle_free_ancilla_initial_state(N, theta_A, phi_A)
    M = build_joint_measurement_operator(N, psi, ops)

    return compute_n_particle_sensitivity(
        N,
        psi0,
        T_bs,
        t_hold,
        omega_true,
        a_x,
        a_y,
        a_z,
        a_zz,
        ops,
        fd_step=fd_step,
        meas_op=M,
    )


def compute_qubit_free_ancilla_joint_sensitivity_with_details(
    N: int,
    omega_true: float,
    theta_A: float,
    phi_A: float,
    a_x: float,
    a_y: float,
    a_z: float,
    a_zz: float,
    psi: float,
    *,
    t_hold: float = T_HOLD,
    T_bs: float = T_BS,
    fd_step: float = FD_STEP,
) -> tuple[float, float, float, float, bool]:
    """Sensitivity Δω with diagnostic details (J_A=1/2).

    Returns:
        Tuple (delta_omega, expectation_M, variance_M, d_exp, is_fringe).
    """
    ops = build_n_particle_operators(N)
    psi0 = n_particle_free_ancilla_initial_state(N, theta_A, phi_A)
    M = build_joint_measurement_operator(N, psi, ops)

    # Evaluate at omega_true
    psi_final = evolve_n_particle_circuit(
        N,
        psi0,
        T_bs,
        t_hold,
        omega_true,
        a_x,
        a_y,
        a_z,
        a_zz,
        ops,
    )
    exp_val, var_val = compute_expectation_and_variance(psi_final, M)

    # Central finite difference for ∂⟨M⟩/∂ω
    psi_plus = evolve_n_particle_circuit(
        N,
        psi0,
        T_bs,
        t_hold,
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
        t_hold,
        omega_true - fd_step,
        a_x,
        a_y,
        a_z,
        a_zz,
        ops,
    )
    exp_plus = float(np.real(psi_plus.conj() @ M @ psi_plus))
    exp_minus = float(np.real(psi_minus.conj() @ M @ psi_minus))
    d_exp = (exp_plus - exp_minus) / (2.0 * fd_step)

    is_fringe = abs(d_exp) < 1e-12 or var_val < 1e-15
    if is_fringe:
        return float("inf"), exp_val, var_val, d_exp, True

    delta = float(np.sqrt(var_val) / abs(d_exp))
    return delta, exp_val, var_val, d_exp, False


# ============================================================================
# Sensitivity — J_A=N/2 (Experiment 3)
# ============================================================================


def build_bipartite_hold_hamiltonian(
    N: int,
    omega: float,
    a_x: float,
    a_y: float,
    a_z: float,
    a_zz: float,
    ops: dict[str, np.ndarray],
) -> np.ndarray:
    """Build the total holding Hamiltonian for the bipartite (J_A=N/2) space.

    H = ω J_z^S + ω (a_x J_x^A + a_y J_y^A + a_z J_z^A) + a_{zz} J_z^S ⊗ J_z^A

    Args:
        N: Number of particles per subsystem.
        omega: Phase rate parameter.
        a_x: J_x^A drive coefficient.
        a_y: J_y^A drive coefficient.
        a_z: J_z^A drive coefficient.
        a_zz: Ising interaction coefficient.
        ops: Operators from build_operators(N, N).

    Returns:
        (N+1)^2 × (N+1)^2 Hermitian matrix.
    """
    H = omega * ops["Jz_S"]

    if a_x != 0.0 or a_y != 0.0 or a_z != 0.0:
        H_drive = np.zeros_like(ops["Jz_A"], dtype=complex)
        if a_x != 0.0:
            H_drive += a_x * ops["Jx_A"]
        if a_y != 0.0:
            H_drive += a_y * ops["Jy_A"]
        if a_z != 0.0:
            H_drive += a_z * ops["Jz_A"]
        H += omega * H_drive

    if a_zz != 0.0:
        H += a_zz * (ops["Jz_S"] @ ops["Jz_A"])

    H = 0.5 * (H + H.conj().T)
    d_tot = (N + 1) ** 2
    assert H.shape == (d_tot, d_tot), (
        f"H has shape {H.shape}, expected ({d_tot}, {d_tot})"
    )
    assert np.allclose(H, H.conj().T, atol=1e-12), (
        "Bipartite hold Hamiltonian not Hermitian"
    )
    return H


def bipartite_hold_unitary(
    N: int,
    t_hold: float,
    omega: float,
    a_x: float,
    a_y: float,
    a_z: float,
    a_zz: float,
    ops: dict[str, np.ndarray],
) -> np.ndarray:
    """Holding-time unitary for the bipartite (J_A=N/2) space.

    U_hold(t_hold) = exp(-i t_hold H)

    Args:
        N: Number of particles per subsystem.
        t_hold: Holding-time strength.
        omega: Phase rate parameter.
        a_x: J_x^A drive coefficient.
        a_y: J_y^A drive coefficient.
        a_z: J_z^A drive coefficient.
        a_zz: Ising interaction coefficient.
        ops: Operators from build_operators(N, N).

    Returns:
        (N+1)^2 × (N+1)^2 unitary matrix.
    """
    H = build_bipartite_hold_hamiltonian(N, omega, a_x, a_y, a_z, a_zz, ops)
    U = expm(-1j * t_hold * H)
    d_tot = (N + 1) ** 2
    I_full = np.eye(d_tot, dtype=complex)
    assert np.allclose(U @ U.conj().T, I_full, atol=1e-12), (
        f"Bipartite hold unitary not unitary for N={N}, ω={omega}"
    )
    return U


def evolve_bipartite_circuit(
    N: int,
    psi0: np.ndarray,
    T_bs: float,
    t_hold: float,
    omega: float,
    a_x: float,
    a_y: float,
    a_z: float,
    a_zz: float,
    ops: dict[str, np.ndarray],
) -> np.ndarray:
    """Run the full bipartite MZI circuit (J_A=N/2).

    |ψ_final⟩ = U_BS_S · U_hold · U_BS_S · |ψ₀⟩

    Args:
        N: Number of particles per subsystem.
        psi0: Initial state vector.
        T_bs: Beam-splitter duration.
        t_hold: Holding-time strength.
        omega: Phase rate parameter.
        a_x: J_x^A drive coefficient.
        a_y: J_y^A drive coefficient.
        a_z: J_z^A drive coefficient.
        a_zz: Ising interaction coefficient.
        ops: Operators from build_operators(N, N).

    Returns:
        Final normalised state vector.
    """
    assert np.isclose(np.linalg.norm(psi0), 1.0), "Initial state must be normalised"
    U_bs = build_system_only_bs_unitary(N, N, T_bs)
    psi = U_bs @ psi0
    psi = bipartite_hold_unitary(N, t_hold, omega, a_x, a_y, a_z, a_zz, ops) @ psi
    psi = U_bs @ psi
    assert np.isclose(np.linalg.norm(psi), 1.0), "Final state not normalised"
    return psi


def compute_bipartite_sensitivity(
    N: int,
    psi0: np.ndarray,
    T_bs: float,
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
    """Error-propagation sensitivity Δω in the bipartite (J_A=N/2) space.

    Delegates to evolve_bipartite_circuit with central finite differences.

    Args:
        N: Number of particles per subsystem.
        psi0: Initial state vector.
        T_bs: Beam-splitter duration.
        t_hold: Holding-time strength.
        omega_true: True phase rate parameter.
        a_x: J_x^A drive coefficient.
        a_y: J_y^A drive coefficient.
        a_z: J_z^A drive coefficient.
        a_zz: Ising interaction coefficient.
        ops: Operators from build_operators(N, N).
        fd_step: Finite-difference step size.
        meas_op: Measurement operator (default ops['Jz_S']).

    Returns:
        Sensitivity Δω (positive float). Returns inf at fringe extremum.
    """
    if meas_op is None:
        meas_op = ops["Jz_S"]

    # Evaluate at omega_true
    psi = evolve_bipartite_circuit(
        N,
        psi0,
        T_bs,
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
    psi_plus = evolve_bipartite_circuit(
        N,
        psi0,
        T_bs,
        t_hold,
        omega_true + fd_step,
        a_x,
        a_y,
        a_z,
        a_zz,
        ops,
    )
    psi_minus = evolve_bipartite_circuit(
        N,
        psi0,
        T_bs,
        t_hold,
        omega_true - fd_step,
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
        return float("inf")
    if var < 1e-15:
        return float("inf")

    return float(np.sqrt(var) / abs(d_exp))


def compute_bipartite_free_css_joint_sensitivity(
    N: int,
    omega_true: float,
    theta_A: float,
    phi_A: float,
    a_x: float,
    a_y: float,
    a_z: float,
    a_zz: float,
    psi: float,
    *,
    t_hold: float = T_HOLD,
    T_bs: float = T_BS,
    fd_step: float = FD_STEP,
) -> float:
    """Sensitivity Δω for bipartite (J_A=N/2) system + free CSS + joint meas.

    Args:
        N: Number of particles per subsystem.
        omega_true: True phase rate parameter.
        theta_A: Ancilla CSS polar angle.
        phi_A: Ancilla CSS azimuthal angle.
        a_x: J_x^A drive coefficient.
        a_y: J_y^A drive coefficient.
        a_z: J_z^A drive coefficient.
        a_zz: Ising interaction coefficient.
        psi: Measurement weight angle.
        t_hold: Holding time.
        T_bs: Beam-splitter duration.
        fd_step: Finite-difference step size.

    Returns:
        Sensitivity Δω (positive float). Returns inf at fringe extremum.
    """
    ops = build_operators(N, N)
    psi0 = multi_particle_free_css_initial_state(N, theta_A, phi_A)
    M = build_bipartite_joint_measurement_operator(N, psi, ops)

    return compute_bipartite_sensitivity(
        N,
        psi0,
        T_bs,
        t_hold,
        omega_true,
        a_x,
        a_y,
        a_z,
        a_zz,
        ops,
        fd_step=fd_step,
        meas_op=M,
    )


# ============================================================================
# Decoupled Baseline
# ============================================================================


def decoupled_baseline(N: int, t_hold: float = T_HOLD) -> float:
    """Decoupled baseline sensitivity (all params zero, J_z^S measurement).

    Δω = 1/(√N × t_hold)

    Args:
        N: Number of system particles.
        t_hold: Holding time.

    Returns:
        SQL sensitivity value.
    """
    return 1.0 / (np.sqrt(N) * t_hold)


# ============================================================================
# 7D Random Search — J_A=1/2 (Experiments 1 & 2)
# ============================================================================


@dataclass
class QubitFreeAncillaJointRandomSearchResult:
    """Result of a 7D random search (J_A=1/2).

    Parameters: [θ_A, φ_A, a_x, a_y, a_z, a_zz, ψ].

    Attributes:
        samples: Array of shape (n_samples, 7).
        delta_omega_values: Δω for each sample.
        best_params: Best 7-element parameter vector.
        best_delta_omega: Best Δω found.
        omega_value: Phase rate value.
        sql: SQL reference value.
        t_hold: Holding time.
    """

    samples: np.ndarray
    delta_omega_values: np.ndarray
    best_params: tuple[float, ...]
    best_delta_omega: float
    omega_value: float
    sql: float
    t_hold: float


def qubit_free_ancilla_joint_random_search(
    N: int,
    omega: float,
    n_samples: int = N_RANDOM_EXP2,
    R: float = R_MAX,
    t_hold: float = T_HOLD,
    T_bs: float = T_BS,
    seed: int | None = 42,
) -> QubitFreeAncillaJointRandomSearchResult:
    """7D random search for J_A=1/2 over (θ_A, φ_A, a, a_zz, ψ).

    Drive vector a = (a_x, a_y, a_z) sampled from the 3-ball of radius R
    via Marsaglia's method.

    Args:
        N: Number of system particles.
        omega: Phase rate value.
        n_samples: Number of random points to evaluate.
        R: 3-ball radius for drive vector.
        t_hold: Holding time.
        T_bs: Beam-splitter duration.
        seed: Random seed for reproducibility.

    Returns:
        QubitFreeAncillaJointRandomSearchResult.
    """
    rng = np.random.default_rng(seed)

    # Pre-allocate samples: [θ_A, φ_A, a_x, a_y, a_z, a_zz, ψ]
    samples = np.empty((n_samples, 7), dtype=float)

    # θ_A ∈ [0, π]
    samples[:, 0] = rng.uniform(THETA_A_BOUNDS[0], THETA_A_BOUNDS[1], size=n_samples)
    # φ_A ∈ [0, 2π)
    samples[:, 1] = rng.uniform(PHI_A_BOUNDS[0], PHI_A_BOUNDS[1], size=n_samples)
    # a from 3-ball
    drive_samples, azz_samples = marsaglia_ball_sample(
        rng,
        n_samples,
        R,
        AZZ_BOUNDS[0],
        AZZ_BOUNDS[1],
    )
    samples[:, 2:5] = drive_samples  # a_x, a_y, a_z
    samples[:, 5] = azz_samples  # a_zz
    # ψ ∈ [-π, π]
    samples[:, 6] = rng.uniform(PSI_BOUNDS[0], PSI_BOUNDS[1], size=n_samples)

    deltas = np.full(n_samples, np.inf, dtype=float)

    for i in range(n_samples):
        domega = compute_qubit_free_ancilla_joint_sensitivity(
            N,
            omega,
            float(samples[i, 0]),  # theta_A
            float(samples[i, 1]),  # phi_A
            float(samples[i, 2]),  # a_x
            float(samples[i, 3]),  # a_y
            float(samples[i, 4]),  # a_z
            float(samples[i, 5]),  # a_zz
            float(samples[i, 6]),  # psi
            t_hold=t_hold,
            T_bs=T_bs,
        )
        deltas[i] = domega

    best_idx = int(np.argmin(deltas))
    best_params = tuple(float(samples[best_idx, j]) for j in range(7))

    return QubitFreeAncillaJointRandomSearchResult(
        samples=samples,
        delta_omega_values=deltas,
        best_params=best_params,
        best_delta_omega=float(deltas[best_idx]),
        omega_value=omega,
        sql=sql_reference(N, t_hold),
        t_hold=t_hold,
    )


# ============================================================================
# 7D Random Search — J_A=N/2 (Experiment 3)
# ============================================================================


@dataclass
class BipartiteFreeAncillaJointRandomSearchResult:
    """Result of a 7D random search for J_A=N/2.

    Parameters: [θ_A, φ_A, a_x, a_y, a_z, a_zz, ψ].
    """

    samples: np.ndarray
    delta_omega_values: np.ndarray
    best_params: tuple[float, ...]
    best_delta_omega: float
    omega_value: float
    sql: float
    t_hold: float


def bipartite_free_ancilla_joint_random_search(
    N: int,
    omega: float,
    n_samples: int = N_RANDOM_EXP3,
    R: float = R_MAX_BIPARTITE,
    t_hold: float = T_HOLD,
    T_bs: float = T_BS,
    seed: int | None = 42,
) -> BipartiteFreeAncillaJointRandomSearchResult:
    """7D random search for J_A=N/2 over (θ_A, φ_A, a, a_zz, ψ).

    Drive vector a sampled from the 3-ball of radius R.

    Args:
        N: Number of particles per subsystem.
        omega: Phase rate value.
        n_samples: Number of random points to evaluate.
        R: 3-ball radius for drive vector.
        t_hold: Holding time.
        T_bs: Beam-splitter duration.
        seed: Random seed for reproducibility.

    Returns:
        BipartiteFreeAncillaJointRandomSearchResult.
    """
    rng = np.random.default_rng(seed)

    # Pre-allocate samples
    samples = np.empty((n_samples, 7), dtype=float)

    samples[:, 0] = rng.uniform(THETA_A_BOUNDS[0], THETA_A_BOUNDS[1], size=n_samples)
    samples[:, 1] = rng.uniform(PHI_A_BOUNDS[0], PHI_A_BOUNDS[1], size=n_samples)
    drive_samples, azz_samples = marsaglia_ball_sample(
        rng,
        n_samples,
        R,
        AZZ_BOUNDS[0],
        AZZ_BOUNDS[1],
    )
    samples[:, 2:5] = drive_samples
    samples[:, 5] = azz_samples
    samples[:, 6] = rng.uniform(PSI_BOUNDS[0], PSI_BOUNDS[1], size=n_samples)

    deltas = np.full(n_samples, np.inf, dtype=float)

    for i in range(n_samples):
        domega = compute_bipartite_free_css_joint_sensitivity(
            N,
            omega,
            float(samples[i, 0]),
            float(samples[i, 1]),
            float(samples[i, 2]),
            float(samples[i, 3]),
            float(samples[i, 4]),
            float(samples[i, 5]),
            float(samples[i, 6]),
            t_hold=t_hold,
            T_bs=T_bs,
        )
        deltas[i] = domega

    best_idx = int(np.argmin(deltas))
    best_params = tuple(float(samples[best_idx, j]) for j in range(7))

    return BipartiteFreeAncillaJointRandomSearchResult(
        samples=samples,
        delta_omega_values=deltas,
        best_params=best_params,
        best_delta_omega=float(deltas[best_idx]),
        omega_value=omega,
        sql=sql_reference(N, t_hold),
        t_hold=t_hold,
    )


# ============================================================================
# Objective Functions for Nelder-Mead
# ============================================================================


def _add_bound_penalty(
    val: float,
    bounds: tuple[float, float],
    scale: float,
) -> float:
    """Quadratic penalty when *val* lies outside *bounds*."""
    lo, hi = bounds
    if val < lo:
        return scale * (lo - val) ** 2
    if val > hi:
        return scale * (val - hi) ** 2
    return 0.0


def qubit_free_ancilla_joint_objective(
    params: np.ndarray,
    N: int,
    omega_true: float,
    t_hold: float = T_HOLD,
    T_bs: float = T_BS,
    fd_step: float = FD_STEP,
    theta_bounds: tuple[float, float] = THETA_A_BOUNDS,
    phi_bounds: tuple[float, float] = PHI_A_BOUNDS,
    drive_bounds: tuple[float, float] = DRIVE_BOUNDS,
    azz_bounds: tuple[float, float] = AZZ_BOUNDS,
    psi_bounds: tuple[float, float] = PSI_BOUNDS,
    penalty_scale: float = 1e6,
) -> float:
    """Objective function for minimising Δω (J_A=1/2, 7D).

    params = [θ_A, φ_A, a_x, a_y, a_z, a_zz, ψ].

    Args:
        params: 7-element parameter vector.
        N: Number of system particles.
        omega_true: True phase rate.
        t_hold: Holding time.
        T_bs: Beam-splitter duration.
        fd_step: Finite-difference step.
        theta_bounds: (min, max) for θ_A.
        phi_bounds: (min, max) for φ_A.
        drive_bounds: (min, max) for a_x, a_y, a_z.
        azz_bounds: (min, max) for a_zz.
        psi_bounds: (min, max) for ψ.
        penalty_scale: Scale for bound-violation penalty.

    Returns:
        Δω (plus infinite penalty if bounds violated).
    """
    theta_A, phi_A = float(params[0]), float(params[1])
    a_x, a_y, a_z = float(params[2]), float(params[3]), float(params[4])
    a_zz_val = float(params[5])
    psi_val = float(params[6])

    penalty = 0.0
    penalty += _add_bound_penalty(theta_A, theta_bounds, penalty_scale)
    penalty += _add_bound_penalty(phi_A, phi_bounds, penalty_scale)
    lo, hi = drive_bounds
    for val in (a_x, a_y, a_z):
        penalty += _add_bound_penalty(val, (lo, hi), penalty_scale)
    penalty += _add_bound_penalty(a_zz_val, azz_bounds, penalty_scale)
    penalty += _add_bound_penalty(psi_val, psi_bounds, penalty_scale)

    if penalty > 0.0:
        return float(1e10 + penalty)

    return compute_qubit_free_ancilla_joint_sensitivity(
        N,
        omega_true,
        theta_A,
        phi_A,
        a_x,
        a_y,
        a_z,
        a_zz_val,
        psi_val,
        t_hold=t_hold,
        T_bs=T_bs,
        fd_step=fd_step,
    )


def bipartite_free_ancilla_joint_objective(
    params: np.ndarray,
    N: int,
    omega_true: float,
    t_hold: float = T_HOLD,
    T_bs: float = T_BS,
    fd_step: float = FD_STEP,
    theta_bounds: tuple[float, float] = THETA_A_BOUNDS,
    phi_bounds: tuple[float, float] = PHI_A_BOUNDS,
    drive_bounds: tuple[float, float] = DRIVE_BOUNDS,
    azz_bounds: tuple[float, float] = AZZ_BOUNDS,
    psi_bounds: tuple[float, float] = PSI_BOUNDS,
    penalty_scale: float = 1e6,
) -> float:
    """Objective function for minimising Δω (J_A=N/2, 7D).

    params = [θ_A, φ_A, a_x, a_y, a_z, a_zz, ψ].
    """
    theta_A, phi_A = float(params[0]), float(params[1])
    a_x, a_y, a_z = float(params[2]), float(params[3]), float(params[4])
    a_zz_val = float(params[5])
    psi_val = float(params[6])

    penalty = 0.0
    penalty += _add_bound_penalty(theta_A, theta_bounds, penalty_scale)
    penalty += _add_bound_penalty(phi_A, phi_bounds, penalty_scale)
    lo, hi = drive_bounds
    for val in (a_x, a_y, a_z):
        penalty += _add_bound_penalty(val, (lo, hi), penalty_scale)
    penalty += _add_bound_penalty(a_zz_val, azz_bounds, penalty_scale)
    penalty += _add_bound_penalty(psi_val, psi_bounds, penalty_scale)

    if penalty > 0.0:
        return float(1e10 + penalty)

    return compute_bipartite_free_css_joint_sensitivity(
        N,
        omega_true,
        theta_A,
        phi_A,
        a_x,
        a_y,
        a_z,
        a_zz_val,
        psi_val,
        t_hold=t_hold,
        T_bs=T_bs,
        fd_step=fd_step,
    )


# ============================================================================
# Nelder-Mead — J_A=1/2 (Experiments 1 & 2)
# ============================================================================


@dataclass
class QubitFreeAncillaJointNMSensitivityResult:
    """Result of a single Nelder-Mead run (J_A=1/2, 7D).

    Attributes:
        delta_omega_opt: Best sensitivity Δω found.
        params_opt: Optimal 7-element parameter vector.
        omega_true: True ω used for this optimisation.
        success: Whether the optimiser reported success.
        nfev: Number of function evaluations.
        message: Optimiser message.
        expectation_M: ⟨M⟩ at optimum.
        variance_M: Var(M) at optimum.
        d_expectation: ∂⟨M⟩/∂ω at optimum.
        theta_A_opt: Optimal θ_A.
        phi_A_opt: Optimal φ_A.
        a_x_opt, a_y_opt, a_z_opt, a_zz_opt: Optimal drive params.
        psi_opt: Optimal ψ.
        history: Objective values per iteration (if tracked).
        t_hold: Holding time.
        sql: SQL reference.
    """

    delta_omega_opt: float
    params_opt: np.ndarray
    omega_true: float
    success: bool
    nfev: int
    message: str = ""
    expectation_M: float = 0.0
    variance_M: float = 0.0
    d_expectation: float = 0.0
    theta_A_opt: float = 0.0
    phi_A_opt: float = 0.0
    a_x_opt: float = 0.0
    a_y_opt: float = 0.0
    a_z_opt: float = 0.0
    a_zz_opt: float = 0.0
    psi_opt: float = 0.0
    history: list[float] = field(default_factory=list)
    t_hold: float = T_HOLD
    sql: float = 0.0


def run_qubit_free_ancilla_joint_nelder_mead(
    N: int,
    omega_true: float,
    x0: np.ndarray | None = None,
    seed: int | None = None,
    maxiter: int = NM_MAXITER,
    xatol: float = 1e-8,
    fatol: float = 1e-8,
    adaptive: bool = True,
    t_hold: float = T_HOLD,
    T_bs: float = T_BS,
    track_history: bool = False,
) -> QubitFreeAncillaJointNMSensitivityResult:
    """Nelder-Mead optimisation for J_A=1/2, 7D parameter space.

    Args:
        N: Number of system particles.
        omega_true: True phase rate parameter.
        x0: Initial 7-parameter vector. Random if None.
        seed: Random seed (used if x0 is None).
        maxiter: Maximum iterations.
        xatol: Absolute parameter tolerance.
        fatol: Absolute function tolerance.
        adaptive: Use adaptive Nelder-Mead parameters.
        t_hold: Holding time.
        T_bs: Beam-splitter duration.
        track_history: If True, record objective values.

    Returns:
        QubitFreeAncillaJointNMSensitivityResult.
    """
    if x0 is None:
        rng = np.random.default_rng(seed)
        x0 = np.array(
            [
                rng.uniform(*THETA_A_BOUNDS),
                rng.uniform(*PHI_A_BOUNDS),
                rng.uniform(*DRIVE_BOUNDS),
                rng.uniform(*DRIVE_BOUNDS),
                rng.uniform(*DRIVE_BOUNDS),
                rng.uniform(*AZZ_BOUNDS),
                rng.uniform(*PSI_BOUNDS),
            ]
        )
    else:
        x0 = np.asarray(x0, dtype=float)
        assert x0.shape == (7,), f"x0 must have 7 elements, got {x0.shape}"

    def objective(p: np.ndarray) -> float:
        return qubit_free_ancilla_joint_objective(
            p,
            N,
            omega_true,
            t_hold=t_hold,
            T_bs=T_bs,
        )

    history: list[float] = []

    def callback(xk: np.ndarray) -> None:
        if track_history:
            val = objective(xk)
            history.append(val)

    result = minimize(
        objective,
        x0=x0,
        method="Nelder-Mead",
        callback=callback if track_history else None,
        options={
            "maxiter": maxiter,
            "xatol": xatol,
            "fatol": fatol,
            "adaptive": adaptive,
        },
    )

    opt_params = result.x.copy()
    theta_A_opt = float(opt_params[0])
    phi_A_opt = float(opt_params[1])
    a_x_opt = float(opt_params[2])
    a_y_opt = float(opt_params[3])
    a_z_opt = float(opt_params[4])
    a_zz_opt = float(opt_params[5])
    psi_opt = float(opt_params[6])

    # Diagnostics at the optimal point
    details_fn = compute_qubit_free_ancilla_joint_sensitivity_with_details
    _, exp_m, var_m, d_exp, _ = details_fn(
        N,
        omega_true,
        theta_A_opt,
        phi_A_opt,
        a_x_opt,
        a_y_opt,
        a_z_opt,
        a_zz_opt,
        psi_opt,
        t_hold=t_hold,
        T_bs=T_bs,
    )

    return QubitFreeAncillaJointNMSensitivityResult(
        delta_omega_opt=float(result.fun),
        params_opt=opt_params,
        omega_true=omega_true,
        success=bool(result.success),
        nfev=int(result.nfev),
        message=str(result.message),
        expectation_M=exp_m,
        variance_M=var_m,
        d_expectation=d_exp,
        theta_A_opt=theta_A_opt,
        phi_A_opt=phi_A_opt,
        a_x_opt=a_x_opt,
        a_y_opt=a_y_opt,
        a_z_opt=a_z_opt,
        a_zz_opt=a_zz_opt,
        psi_opt=psi_opt,
        history=history.copy(),
        t_hold=t_hold,
        sql=sql_reference(N, t_hold),
    )


# ============================================================================
# Nelder-Mead — J_A=N/2 (Experiment 3)
# ============================================================================


@dataclass
class BipartiteFreeAncillaJointNMSensitivityResult:
    """Result of a single Nelder-Mead run (J_A=N/2, 7D)."""

    delta_omega_opt: float
    params_opt: np.ndarray
    omega_true: float
    success: bool
    nfev: int
    message: str = ""
    expectation_M: float = 0.0
    variance_M: float = 0.0
    d_expectation: float = 0.0
    theta_A_opt: float = 0.0
    phi_A_opt: float = 0.0
    a_x_opt: float = 0.0
    a_y_opt: float = 0.0
    a_z_opt: float = 0.0
    a_zz_opt: float = 0.0
    psi_opt: float = 0.0
    history: list[float] = field(default_factory=list)
    t_hold: float = T_HOLD
    sql: float = 0.0


def run_bipartite_free_ancilla_joint_nelder_mead(
    N: int,
    omega_true: float,
    x0: np.ndarray | None = None,
    seed: int | None = None,
    maxiter: int = NM_MAXITER,
    xatol: float = 1e-8,
    fatol: float = 1e-8,
    adaptive: bool = True,
    t_hold: float = T_HOLD,
    T_bs: float = T_BS,
    track_history: bool = False,
) -> BipartiteFreeAncillaJointNMSensitivityResult:
    """Nelder-Mead optimisation for J_A=N/2, 7D parameter space.

    Args:
        N: Number of particles per subsystem.
        omega_true: True phase rate parameter.
        x0: Initial 7-parameter vector. Random if None.
        seed: Random seed (used if x0 is None).
        maxiter: Maximum iterations.
        xatol: Absolute parameter tolerance.
        fatol: Absolute function tolerance.
        adaptive: Use adaptive Nelder-Mead parameters.
        t_hold: Holding time.
        T_bs: Beam-splitter duration.
        track_history: If True, record objective values.

    Returns:
        BipartiteFreeAncillaJointNMSensitivityResult.
    """
    if x0 is None:
        rng = np.random.default_rng(seed)
        x0 = np.array(
            [
                rng.uniform(*THETA_A_BOUNDS),
                rng.uniform(*PHI_A_BOUNDS),
                rng.uniform(*DRIVE_BOUNDS),
                rng.uniform(*DRIVE_BOUNDS),
                rng.uniform(*DRIVE_BOUNDS),
                rng.uniform(*AZZ_BOUNDS),
                rng.uniform(*PSI_BOUNDS),
            ]
        )
    else:
        x0 = np.asarray(x0, dtype=float)
        assert x0.shape == (7,), f"x0 must have 7 elements, got {x0.shape}"

    def objective(p: np.ndarray) -> float:
        return bipartite_free_ancilla_joint_objective(
            p,
            N,
            omega_true,
            t_hold=t_hold,
            T_bs=T_bs,
        )

    history: list[float] = []

    def callback(xk: np.ndarray) -> None:
        if track_history:
            val = objective(xk)
            history.append(val)

    result = minimize(
        objective,
        x0=x0,
        method="Nelder-Mead",
        callback=callback if track_history else None,
        options={
            "maxiter": maxiter,
            "xatol": xatol,
            "fatol": fatol,
            "adaptive": adaptive,
        },
    )

    opt_params = result.x.copy()
    theta_A_opt = float(opt_params[0])
    phi_A_opt = float(opt_params[1])
    a_x_opt = float(opt_params[2])
    a_y_opt = float(opt_params[3])
    a_z_opt = float(opt_params[4])
    a_zz_opt = float(opt_params[5])
    psi_opt = float(opt_params[6])

    # Diagnostics at the optimal point
    ops = build_operators(N, N)
    psi0 = multi_particle_free_css_initial_state(N, theta_A_opt, phi_A_opt)
    M = build_bipartite_joint_measurement_operator(N, psi_opt, ops)

    psi_final = evolve_bipartite_circuit(
        N,
        psi0,
        T_bs,
        t_hold,
        omega_true,
        a_x_opt,
        a_y_opt,
        a_z_opt,
        a_zz_opt,
        ops,
    )
    exp_m, var_m = compute_expectation_and_variance(psi_final, M)

    # Derivative
    psi_plus = evolve_bipartite_circuit(
        N,
        psi0,
        T_bs,
        t_hold,
        omega_true + FD_STEP,
        a_x_opt,
        a_y_opt,
        a_z_opt,
        a_zz_opt,
        ops,
    )
    psi_minus = evolve_bipartite_circuit(
        N,
        psi0,
        T_bs,
        t_hold,
        omega_true - FD_STEP,
        a_x_opt,
        a_y_opt,
        a_z_opt,
        a_zz_opt,
        ops,
    )
    d_exp = (
        np.real(psi_plus.conj() @ M @ psi_plus)
        - np.real(psi_minus.conj() @ M @ psi_minus)
    ) / (2.0 * FD_STEP)

    return BipartiteFreeAncillaJointNMSensitivityResult(
        delta_omega_opt=float(result.fun),
        params_opt=opt_params,
        omega_true=omega_true,
        success=bool(result.success),
        nfev=int(result.nfev),
        message=str(result.message),
        expectation_M=float(exp_m),
        variance_M=float(var_m),
        d_expectation=float(d_exp),
        theta_A_opt=theta_A_opt,
        phi_A_opt=phi_A_opt,
        a_x_opt=a_x_opt,
        a_y_opt=a_y_opt,
        a_z_opt=a_z_opt,
        a_zz_opt=a_zz_opt,
        psi_opt=psi_opt,
        history=history.copy(),
        t_hold=t_hold,
        sql=sql_reference(N, t_hold),
    )


# ============================================================================
# Scaling Result Dataclass — J_A=1/2 (Experiments 1 & 2)
# ============================================================================


@dataclass
class QubitFreeAncillaJointNScalingResult(ParquetSerializable):
    """Result for a single (N, ω) pair with J_A=1/2.

    All input parameters and computed results are stored for self-describing
    Parquet serialisation.
    """

    N: int
    omega: float
    delta_omega_opt: float
    sql: float
    ratio: float
    theta_A_opt: float
    phi_A_opt: float
    a_x_opt: float
    a_y_opt: float
    a_z_opt: float
    a_zz_opt: float
    psi_opt: float
    expectation_M: float = 0.0
    variance_M: float = 0.0
    d_expectation: float = 0.0
    t_hold: float = T_HOLD
    success: bool = False
    nfev: int = 0

    _PARQUET_COLUMNS: ClassVar[list[str]] = [
        "N",
        "omega",
        "delta_omega_opt",
        "sql",
        "ratio",
        "theta_A_opt",
        "phi_A_opt",
        "a_x_opt",
        "a_y_opt",
        "a_z_opt",
        "a_zz_opt",
        "psi_opt",
        "expectation_M",
        "variance_M",
        "d_expectation",
        "t_hold",
        "success",
        "nfev",
    ]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "N": self.N,
                    "omega": self.omega,
                    "delta_omega_opt": self.delta_omega_opt,
                    "sql": self.sql,
                    "ratio": self.ratio,
                    "theta_A_opt": self.theta_A_opt,
                    "phi_A_opt": self.phi_A_opt,
                    "a_x_opt": self.a_x_opt,
                    "a_y_opt": self.a_y_opt,
                    "a_z_opt": self.a_z_opt,
                    "a_zz_opt": self.a_zz_opt,
                    "psi_opt": self.psi_opt,
                    "expectation_M": self.expectation_M,
                    "variance_M": self.variance_M,
                    "d_expectation": self.d_expectation,
                    "t_hold": self.t_hold,
                    "success": int(self.success),
                    "nfev": self.nfev,
                }
            ]
        )

    @classmethod
    def from_parquet(cls, path: str | Path) -> QubitFreeAncillaJointNScalingResult:
        df = pd.read_parquet(path)
        cls._validate_columns(df)
        row = df.iloc[0]
        return cls(
            N=int(row["N"]),
            omega=float(row["omega"]),
            delta_omega_opt=float(row["delta_omega_opt"]),
            sql=float(row["sql"]),
            ratio=float(row["ratio"]),
            theta_A_opt=float(row["theta_A_opt"]),
            phi_A_opt=float(row["phi_A_opt"]),
            a_x_opt=float(row["a_x_opt"]),
            a_y_opt=float(row["a_y_opt"]),
            a_z_opt=float(row["a_z_opt"]),
            a_zz_opt=float(row["a_zz_opt"]),
            psi_opt=float(row["psi_opt"]),
            expectation_M=float(row["expectation_M"]),
            variance_M=float(row["variance_M"]),
            d_expectation=float(row["d_expectation"]),
            t_hold=float(row["t_hold"]),
            success=bool(row["success"]),
            nfev=int(row["nfev"]),
        )


@dataclass
class QubitFreeAncillaJointNScalingScanResult(ParquetSerializable):
    """Collection of QubitFreeAncillaJointNScalingResult for multiple (N, ω)."""

    results: list[QubitFreeAncillaJointNScalingResult]

    _PARQUET_COLUMNS: ClassVar[list[str]] = [
        "N",
        "omega",
        "delta_omega_opt",
        "sql",
        "ratio",
        "theta_A_opt",
        "phi_A_opt",
        "a_x_opt",
        "a_y_opt",
        "a_z_opt",
        "a_zz_opt",
        "psi_opt",
        "expectation_M",
        "variance_M",
        "d_expectation",
        "t_hold",
        "success",
        "nfev",
    ]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                col: [getattr(r, col) for r in self.results]
                for col in self._PARQUET_COLUMNS
            }
        )

    @classmethod
    def from_parquet(cls, path: str | Path) -> QubitFreeAncillaJointNScalingScanResult:
        df = pd.read_parquet(path)
        cls._validate_columns(df)
        results = []
        for _, row in df.iterrows():
            results.append(
                QubitFreeAncillaJointNScalingResult(
                    N=int(row["N"]),
                    omega=float(row["omega"]),
                    delta_omega_opt=float(row["delta_omega_opt"]),
                    sql=float(row["sql"]),
                    ratio=float(row["ratio"]),
                    theta_A_opt=float(row["theta_A_opt"]),
                    phi_A_opt=float(row["phi_A_opt"]),
                    a_x_opt=float(row["a_x_opt"]),
                    a_y_opt=float(row["a_y_opt"]),
                    a_z_opt=float(row["a_z_opt"]),
                    a_zz_opt=float(row["a_zz_opt"]),
                    psi_opt=float(row["psi_opt"]),
                    expectation_M=float(row["expectation_M"]),
                    variance_M=float(row["variance_M"]),
                    d_expectation=float(row["d_expectation"]),
                    t_hold=float(row["t_hold"]),
                    success=bool(row["success"]),
                    nfev=int(row["nfev"]),
                )
            )
        return cls(results=results)

    def __len__(self) -> int:
        return len(self.results)

    def __getitem__(self, idx: int) -> QubitFreeAncillaJointNScalingResult:
        return self.results[idx]

    def to_numpy_arrays(self) -> dict[str, np.ndarray]:
        d: dict[str, list] = {col: [] for col in self._PARQUET_COLUMNS}
        for r in self.results:
            for col in self._PARQUET_COLUMNS:
                d[col].append(getattr(r, col))
        return {k: np.array(v) for k, v in d.items()}


# ============================================================================
# Scaling Result Dataclass — J_A=N/2 (Experiment 3)
# ============================================================================


@dataclass
class BipartiteFreeAncillaJointNScalingResult(ParquetSerializable):
    """Result for a single (N, ω) pair with J_A=N/2."""

    N: int
    omega: float
    delta_omega_opt: float
    sql: float
    ratio: float
    theta_A_opt: float
    phi_A_opt: float
    a_x_opt: float
    a_y_opt: float
    a_z_opt: float
    a_zz_opt: float
    psi_opt: float
    expectation_M: float = 0.0
    variance_M: float = 0.0
    d_expectation: float = 0.0
    t_hold: float = T_HOLD
    success: bool = False
    nfev: int = 0

    _PARQUET_COLUMNS: ClassVar[list[str]] = [
        "N",
        "omega",
        "delta_omega_opt",
        "sql",
        "ratio",
        "theta_A_opt",
        "phi_A_opt",
        "a_x_opt",
        "a_y_opt",
        "a_z_opt",
        "a_zz_opt",
        "psi_opt",
        "expectation_M",
        "variance_M",
        "d_expectation",
        "t_hold",
        "success",
        "nfev",
    ]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "N": self.N,
                    "omega": self.omega,
                    "delta_omega_opt": self.delta_omega_opt,
                    "sql": self.sql,
                    "ratio": self.ratio,
                    "theta_A_opt": self.theta_A_opt,
                    "phi_A_opt": self.phi_A_opt,
                    "a_x_opt": self.a_x_opt,
                    "a_y_opt": self.a_y_opt,
                    "a_z_opt": self.a_z_opt,
                    "a_zz_opt": self.a_zz_opt,
                    "psi_opt": self.psi_opt,
                    "expectation_M": self.expectation_M,
                    "variance_M": self.variance_M,
                    "d_expectation": self.d_expectation,
                    "t_hold": self.t_hold,
                    "success": int(self.success),
                    "nfev": self.nfev,
                }
            ]
        )

    @classmethod
    def from_parquet(cls, path: str | Path) -> BipartiteFreeAncillaJointNScalingResult:
        df = pd.read_parquet(path)
        cls._validate_columns(df)
        row = df.iloc[0]
        return cls(
            N=int(row["N"]),
            omega=float(row["omega"]),
            delta_omega_opt=float(row["delta_omega_opt"]),
            sql=float(row["sql"]),
            ratio=float(row["ratio"]),
            theta_A_opt=float(row["theta_A_opt"]),
            phi_A_opt=float(row["phi_A_opt"]),
            a_x_opt=float(row["a_x_opt"]),
            a_y_opt=float(row["a_y_opt"]),
            a_z_opt=float(row["a_z_opt"]),
            a_zz_opt=float(row["a_zz_opt"]),
            psi_opt=float(row["psi_opt"]),
            expectation_M=float(row["expectation_M"]),
            variance_M=float(row["variance_M"]),
            d_expectation=float(row["d_expectation"]),
            t_hold=float(row["t_hold"]),
            success=bool(row["success"]),
            nfev=int(row["nfev"]),
        )


@dataclass
class BipartiteFreeAncillaJointNScalingScanResult(ParquetSerializable):
    """Collection of BipartiteFreeAncillaJointNScalingResult for multiple (N, ω)."""

    results: list[BipartiteFreeAncillaJointNScalingResult]

    _PARQUET_COLUMNS: ClassVar[list[str]] = (
        BipartiteFreeAncillaJointNScalingResult._PARQUET_COLUMNS
    )

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                col: [getattr(r, col) for r in self.results]
                for col in self._PARQUET_COLUMNS
            }
        )

    @classmethod
    def from_parquet(
        cls, path: str | Path
    ) -> BipartiteFreeAncillaJointNScalingScanResult:
        df = pd.read_parquet(path)
        cls._validate_columns(df)
        results = []
        for _, row in df.iterrows():
            results.append(
                BipartiteFreeAncillaJointNScalingResult(
                    N=int(row["N"]),
                    omega=float(row["omega"]),
                    delta_omega_opt=float(row["delta_omega_opt"]),
                    sql=float(row["sql"]),
                    ratio=float(row["ratio"]),
                    theta_A_opt=float(row["theta_A_opt"]),
                    phi_A_opt=float(row["phi_A_opt"]),
                    a_x_opt=float(row["a_x_opt"]),
                    a_y_opt=float(row["a_y_opt"]),
                    a_z_opt=float(row["a_z_opt"]),
                    a_zz_opt=float(row["a_zz_opt"]),
                    psi_opt=float(row["psi_opt"]),
                    expectation_M=float(row["expectation_M"]),
                    variance_M=float(row["variance_M"]),
                    d_expectation=float(row["d_expectation"]),
                    t_hold=float(row["t_hold"]),
                    success=bool(row["success"]),
                    nfev=int(row["nfev"]),
                )
            )
        return cls(results=results)

    def __len__(self) -> int:
        return len(self.results)

    def __getitem__(self, idx: int) -> BipartiteFreeAncillaJointNScalingResult:
        return self.results[idx]

    def to_numpy_arrays(self) -> dict[str, np.ndarray]:
        d: dict[str, list] = {col: [] for col in self._PARQUET_COLUMNS}
        for r in self.results:
            for col in self._PARQUET_COLUMNS:
                d[col].append(getattr(r, col))
        return {k: np.array(v) for k, v in d.items()}


# ============================================================================
# Single (N, ω) Run — J_A=1/2 (Exp 1 & 2)
# ============================================================================


def run_single_qubit_n_omega(
    N: int,
    omega: float,
    n_random: int = N_RANDOM_EXP2,
    n_nm_refine: int = N_NM_REFINE_EXP2,
    seed: int | None = None,
    maxiter: int = NM_MAXITER,
) -> QubitFreeAncillaJointNScalingResult:
    """Two-phase optimisation for a single (N, ω) pair with J_A=1/2.

    Args:
        N: Number of system particles.
        omega: True phase rate.
        n_random: Number of random search samples.
        n_nm_refine: Number of Nelder-Mead refinements.
        seed: Base RNG seed.
        maxiter: Max Nelder-Mead iterations.

    Returns:
        QubitFreeAncillaJointNScalingResult.
    """
    base_seed = seed if seed is not None else 42

    def rs_fn(
        n_samples: int,
        seed: int,
        **kw: object,
    ) -> QubitFreeAncillaJointRandomSearchResult:
        return qubit_free_ancilla_joint_random_search(
            N,
            omega,
            n_samples=n_samples,
            seed=seed,
        )

    def nm_fn(
        x0: np.ndarray,
        seed: int,
        **kw: object,
    ) -> QubitFreeAncillaJointNMSensitivityResult:
        return run_qubit_free_ancilla_joint_nelder_mead(
            N,
            omega_true=omega,
            x0=x0,
            seed=seed,
            maxiter=maxiter,
        )

    best_nm, _ = run_two_phase_pipeline(
        rs_fn,
        nm_fn,
        config=TwoPhaseConfig(
            n_random=n_random,
            n_nm_refine=n_nm_refine,
            seed=base_seed,
        ),
    )

    sql = sql_reference(N, T_HOLD)
    ratio = sql / best_nm.delta_omega_opt if best_nm.delta_omega_opt > 0 else 0.0

    return QubitFreeAncillaJointNScalingResult(
        N=N,
        omega=omega,
        delta_omega_opt=best_nm.delta_omega_opt,
        sql=sql,
        ratio=ratio,
        theta_A_opt=best_nm.theta_A_opt,
        phi_A_opt=best_nm.phi_A_opt,
        a_x_opt=best_nm.a_x_opt,
        a_y_opt=best_nm.a_y_opt,
        a_z_opt=best_nm.a_z_opt,
        a_zz_opt=best_nm.a_zz_opt,
        psi_opt=best_nm.psi_opt,
        expectation_M=best_nm.expectation_M,
        variance_M=best_nm.variance_M,
        d_expectation=best_nm.d_expectation,
        t_hold=T_HOLD,
        success=best_nm.success,
        nfev=best_nm.nfev,
    )


# ============================================================================
# Single (N, ω) Run — J_A=N/2 (Exp 3)
# ============================================================================


def run_single_bipartite_n_omega(
    N: int,
    omega: float,
    n_random: int = N_RANDOM_EXP3,
    n_nm_refine: int = N_NM_REFINE_EXP3,
    seed: int | None = None,
    maxiter: int = NM_MAXITER,
) -> BipartiteFreeAncillaJointNScalingResult:
    """Two-phase optimisation for a single (N, ω) pair with J_A=N/2.

    Args:
        N: Number of particles per subsystem.
        omega: True phase rate.
        n_random: Number of random search samples.
        n_nm_refine: Number of Nelder-Mead refinements.
        seed: Base RNG seed.
        maxiter: Max Nelder-Mead iterations.

    Returns:
        BipartiteFreeAncillaJointNScalingResult.
    """
    base_seed = seed if seed is not None else 42

    def rs_fn(
        n_samples: int, seed: int, **kw: object
    ) -> BipartiteFreeAncillaJointRandomSearchResult:
        return bipartite_free_ancilla_joint_random_search(
            N,
            omega,
            n_samples=n_samples,
            seed=seed,
        )

    def nm_fn(
        x0: np.ndarray, seed: int, **kw: object
    ) -> BipartiteFreeAncillaJointNMSensitivityResult:
        return run_bipartite_free_ancilla_joint_nelder_mead(
            N,
            omega_true=omega,
            x0=x0,
            seed=seed,
            maxiter=maxiter,
        )

    best_nm, _ = run_two_phase_pipeline(
        rs_fn,
        nm_fn,
        config=TwoPhaseConfig(
            n_random=n_random,
            n_nm_refine=n_nm_refine,
            seed=base_seed,
        ),
    )

    sql = sql_reference(N, T_HOLD)
    ratio = sql / best_nm.delta_omega_opt if best_nm.delta_omega_opt > 0 else 0.0

    return BipartiteFreeAncillaJointNScalingResult(
        N=N,
        omega=omega,
        delta_omega_opt=best_nm.delta_omega_opt,
        sql=sql,
        ratio=ratio,
        theta_A_opt=best_nm.theta_A_opt,
        phi_A_opt=best_nm.phi_A_opt,
        a_x_opt=best_nm.a_x_opt,
        a_y_opt=best_nm.a_y_opt,
        a_z_opt=best_nm.a_z_opt,
        a_zz_opt=best_nm.a_zz_opt,
        psi_opt=best_nm.psi_opt,
        expectation_M=best_nm.expectation_M,
        variance_M=best_nm.variance_M,
        d_expectation=best_nm.d_expectation,
        t_hold=T_HOLD,
        success=best_nm.success,
        nfev=best_nm.nfev,
    )


# ============================================================================
# Decoupled / Baseline Runs
# ============================================================================


def run_decoupled_baseline(
    N: int,
    omega: float = 0.2,
) -> QubitFreeAncillaJointNScalingResult:
    """Decoupled baseline: all params zero, J_z^S measurement.

    Used to verify Δω = SQL exactly.

    Args:
        N: Number of system particles.
        omega: Phase rate (does not affect SQL).

    Returns:
        QubitFreeAncillaJointNScalingResult with decoupled parameters.
    """
    sql = sql_reference(N, T_HOLD)
    delta = decoupled_baseline(N, T_HOLD)
    return QubitFreeAncillaJointNScalingResult(
        N=N,
        omega=omega,
        delta_omega_opt=delta,
        sql=sql,
        ratio=sql / delta if delta > 0 else 0.0,
        theta_A_opt=0.0,
        phi_A_opt=0.0,
        a_x_opt=0.0,
        a_y_opt=0.0,
        a_z_opt=0.0,
        a_zz_opt=0.0,
        psi_opt=0.0,
        expectation_M=0.0,
        variance_M=0.0,
        d_expectation=0.0,
        t_hold=T_HOLD,
        success=True,
        nfev=0,
    )


# ============================================================================
# Data Generation — Experiment 1: N=1 compounding
# ============================================================================


def generate_exp1_compounding(
    omega_values: list[float] | None = None,
    seed: int = 42,
    maxiter: int = NM_MAXITER,
    progress_callback: Callable[[int, int], None] | None = None,
) -> QubitFreeAncillaJointNScalingScanResult:
    """Experiment 1: N=1 compounding across ω values.

    Args:
        omega_values: List of ω values (default OMEGA_VALS).
        seed: Base RNG seed.
        maxiter: Max Nelder-Mead iterations.
        progress_callback: Optional fn(current, total) for progress.

    Returns:
        QubitFreeAncillaJointNScalingScanResult.
    """
    if omega_values is None:
        omega_values = OMEGA_VALS

    results: list[QubitFreeAncillaJointNScalingResult] = []
    total = len(omega_values)

    for idx, omega in enumerate(omega_values):
        if progress_callback:
            progress_callback(idx + 1, total)

        res = run_single_qubit_n_omega(
            N=1,
            omega=omega,
            n_random=N_RANDOM_EXP1,
            n_nm_refine=N_NM_REFINE_EXP1,
            seed=seed + int(omega * 1000),
            maxiter=maxiter,
        )
        results.append(res)

    return QubitFreeAncillaJointNScalingScanResult(results=results)


# ============================================================================
# Data Generation — Experiment 2: N>1 scaling, J_A=1/2
# ============================================================================


def generate_exp2_qubit_scaling(
    omega_value: float = 0.2,
    N_range: list[int] | None = None,
    seed: int = 42,
    maxiter: int = NM_MAXITER,
    progress_callback: Callable[[int, int], None] | None = None,
) -> QubitFreeAncillaJointNScalingScanResult:
    """Experiment 2: N>1 scaling with qubit ancilla (J_A=1/2).

    Args:
        omega_value: True ω for all runs.
        N_range: List of N values (default [1, 2, ..., 20]).
        seed: Base RNG seed.
        maxiter: Max Nelder-Mead iterations per run.
        progress_callback: Optional fn(current, total) for progress.

    Returns:
        QubitFreeAncillaJointNScalingScanResult.
    """
    if N_range is None:
        N_range = N_VALS

    results: list[QubitFreeAncillaJointNScalingResult] = []
    total = len(N_range)

    for idx, N in enumerate(N_range):
        if progress_callback:
            progress_callback(idx + 1, total)

        res = run_single_qubit_n_omega(
            N=N,
            omega=omega_value,
            seed=seed + N,
            maxiter=maxiter,
        )
        results.append(res)

    return QubitFreeAncillaJointNScalingScanResult(results=results)


def generate_exp2_sonly_control(
    omega_value: float = 0.2,
    N_range: list[int] | None = None,
    seed: int = 42,
    maxiter: int = NM_MAXITER,
    progress_callback: Callable[[int, int], None] | None = None,
) -> QubitFreeAncillaJointNScalingScanResult:
    """S-Only control for Experiment 2: fixes ψ=0 and θ_A=0.

    Reproduces #20260613 results with fixed ancilla. Reuses the shared 7D
    objective ``qubit_free_ancilla_joint_objective`` with ``theta_A=0, psi=0``
    and only optimises the 4D drive parameters ``(a_x, a_y, a_z, a_zz)``.

    Args:
        omega_value: True ω for all runs.
        N_range: List of N values.
        seed: Base RNG seed.
        maxiter: Max Nelder-Mead iterations.
        progress_callback: Optional progress callback.

    Returns:
        QubitFreeAncillaJointNScalingScanResult.
    """
    if N_range is None:
        N_range = N_VALS

    results: list[QubitFreeAncillaJointNScalingResult] = []
    total = len(N_range)

    for idx, N in enumerate(N_range):
        if progress_callback:
            progress_callback(idx + 1, total)

        # Fixed ancilla (θ_A=0), S-only (ψ=0): optimise 4D drive params.
        # Wrap the shared 7D objective by fixing theta_A=0, phi_A=0, psi=0.
        def obj_4d(
            params_4d: np.ndarray,
            N_curry: int = N,
            omega_val: float = omega_value,
        ) -> float:
            p_7d = np.array(
                [
                    0.0,
                    0.0,  # theta_A=0, phi_A=0
                    float(params_4d[0]),
                    float(params_4d[1]),  # a_x, a_y
                    float(params_4d[2]),
                    float(params_4d[3]),  # a_z, a_zz
                    0.0,  # psi=0
                ]
            )
            return qubit_free_ancilla_joint_objective(
                p_7d,
                N_curry,
                omega_val,
            )

        rng = np.random.default_rng(seed + N + 10000)
        x0 = rng.uniform(*DRIVE_BOUNDS, size=4)

        nm_result = minimize(
            obj_4d,
            x0=x0,
            method="Nelder-Mead",
            options={
                "maxiter": maxiter,
                "xatol": 1e-8,
                "fatol": 1e-8,
                "adaptive": True,
            },
        )

        opt_p = nm_result.x.copy()
        sql = sql_reference(N, T_HOLD)
        ratio = sql / nm_result.fun if nm_result.fun > 0 else 0.0

        results.append(
            QubitFreeAncillaJointNScalingResult(
                N=N,
                omega=omega_value,
                delta_omega_opt=float(nm_result.fun),
                sql=sql,
                ratio=ratio,
                theta_A_opt=0.0,
                phi_A_opt=0.0,
                a_x_opt=float(opt_p[0]),
                a_y_opt=float(opt_p[1]),
                a_z_opt=float(opt_p[2]),
                a_zz_opt=float(opt_p[3]),
                psi_opt=0.0,
                success=bool(nm_result.success),
                nfev=int(nm_result.nfev),
            )
        )

    return QubitFreeAncillaJointNScalingScanResult(results=results)


# ============================================================================
# Data Generation — Experiment 3: N>1 scaling, J_A=N/2
# ============================================================================


def generate_exp3_bipartite_scaling(
    omega_value: float = 0.2,
    N_range: list[int] | None = None,
    seed: int = 42,
    maxiter: int = NM_MAXITER,
    progress_callback: Callable[[int, int], None] | None = None,
) -> BipartiteFreeAncillaJointNScalingScanResult:
    """Experiment 3: N>1 scaling with multi-particle ancilla (J_A=N/2).

    Args:
        omega_value: True ω for all runs.
        N_range: List of N values (default [1, 2, ..., 20]).
        seed: Base RNG seed.
        maxiter: Max Nelder-Mead iterations per run.
        progress_callback: Optional fn(current, total) for progress.

    Returns:
        BipartiteFreeAncillaJointNScalingScanResult.
    """
    if N_range is None:
        N_range = N_VALS

    results: list[BipartiteFreeAncillaJointNScalingResult] = []
    total = len(N_range)

    for idx, N in enumerate(N_range):
        if progress_callback:
            progress_callback(idx + 1, total)

        res = run_single_bipartite_n_omega(
            N=N,
            omega=omega_value,
            seed=seed + N,
            maxiter=maxiter,
        )
        results.append(res)

    return BipartiteFreeAncillaJointNScalingScanResult(results=results)


# ============================================================================
# Plotting
# ============================================================================


def _arrays_from_qubit_scan(
    scan: QubitFreeAncillaJointNScalingScanResult,
) -> dict[str, np.ndarray]:
    return scan.to_numpy_arrays()


def _arrays_from_bipartite_scan(
    scan: BipartiteFreeAncillaJointNScalingScanResult,
) -> dict[str, np.ndarray]:
    return scan.to_numpy_arrays()


def plot_ratio_vs_omega(
    result: QubitFreeAncillaJointNScalingScanResult,
    baseline_ratios: dict[float, float] | None = None,
    save_path: str | None = None,
) -> matplotlib.figure.Figure:
    """Ratio Δω/Δω_SQL vs ω for Experiment 1 (N=1 compounding).

    Args:
        result: Scaling scan result.
        baseline_ratios: Optional dict mapping ω to baseline ratio for overlay.
        save_path: Optional SVG path.

    Returns:
        Matplotlib figure.
    """
    arr = _arrays_from_qubit_scan(result)
    omegas = arr["omega"]
    # Ratio = Δω_SQL / Δω (SQL line at 1.0, values > 1 beat SQL)
    ratios = arr["sql"] / arr["delta_omega_opt"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(
        omegas,
        ratios,
        "o-",
        color="C0",
        markersize=8,
        linewidth=2,
        label="Joint + Free Ancilla",
    )

    if baseline_ratios:
        base_omegas = sorted(baseline_ratios.keys())
        base_ratios = [baseline_ratios[o] for o in base_omegas]
        ax.semilogy(
            base_omegas,
            base_ratios,
            "s--",
            color="gray",
            markersize=6,
            label="Baseline (fixed ancilla, S-only)",
        )

    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5, label="SQL")
    ax.set_xlabel("ω", fontsize=13)
    ax.set_ylabel("Δω_SQL / Δω", fontsize=13)
    ax.set_title("N=1 Compounding: 7D Optimisation", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, format="svg", bbox_inches="tight")
    return fig


def plot_ratio_vs_n(
    joint_result: QubitFreeAncillaJointNScalingScanResult,
    sonly_result: QubitFreeAncillaJointNScalingScanResult | None = None,
    save_path: str | None = None,
) -> matplotlib.figure.Figure:
    """Ratio Δω/Δω_SQL vs N for qubit ancilla (Experiments 1&2).

    Args:
        joint_result: Combined protocol scaling results.
        sonly_result: Optional S-only control results.
        save_path: Optional SVG path.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    j = _arrays_from_qubit_scan(joint_result)
    ratio_joint = j["sql"] / j["delta_omega_opt"]

    ax.plot(
        j["N"],
        ratio_joint,
        "o-",
        color="C0",
        label="Combined (free A + joint M)",
        markersize=6,
        linewidth=2,
    )

    if sonly_result is not None:
        s = _arrays_from_qubit_scan(sonly_result)
        ratio_sonly = s["sql"] / s["delta_omega_opt"]
        ax.plot(
            s["N"],
            ratio_sonly,
            "s--",
            color="C1",
            label="S-only J$_z^S$ (fixed A)",
            markersize=6,
        )

    ax.axhline(1.0, color="gray", linestyle=":", label="SQL", alpha=0.5)
    ax.set_xlabel("N", fontsize=13)
    ax.set_ylabel("Δω_SQL / Δω", fontsize=13)
    ax.set_title(f"Phase Sensitivity vs N (ω = {j['omega'][0]:g})", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, format="svg", bbox_inches="tight")
    return fig


def plot_bipartite_ratio_vs_n(
    result: BipartiteFreeAncillaJointNScalingScanResult,
    save_path: str | None = None,
) -> matplotlib.figure.Figure:
    """Ratio Δω/Δω_SQL vs N for multi-particle ancilla (Experiment 3).

    Args:
        result: Bipartite scaling results.
        save_path: Optional SVG path.

    Returns:
        Matplotlib figure.
    """
    arr = _arrays_from_bipartite_scan(result)
    Ns = arr["N"]
    ratios = arr["sql"] / arr["delta_omega_opt"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        Ns,
        ratios,
        "o-",
        color="C2",
        markersize=6,
        linewidth=2,
        label="J$_A$=N/2, free CSS + joint M",
    )
    ax.axhline(1.0, color="gray", linestyle=":", label="SQL", alpha=0.5)
    ax.set_xlabel("N", fontsize=13)
    ax.set_ylabel("Δω_SQL / Δω", fontsize=13)
    ax.set_title(
        f"Multi-Particle Ancilla Sensitivity (ω = {arr['omega'][0]:g})", fontsize=14
    )
    ax.legend(fontsize=11)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, format="svg", bbox_inches="tight")
    return fig


def plot_optimal_params_vs_n(
    result: QubitFreeAncillaJointNScalingScanResult,
    save_path: str | None = None,
) -> matplotlib.figure.Figure:
    """Optimal parameters vs N for qubit ancilla case.

    Args:
        result: Scaling scan result.
        save_path: Optional SVG path.

    Returns:
        Matplotlib figure.
    """
    arr = _arrays_from_qubit_scan(result)
    Ns = arr["N"]

    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

    params = [
        (arr["theta_A_opt"], "$\\theta_A$"),
        (arr["phi_A_opt"], "$\\phi_A$"),
        (arr["a_zz_opt"], "$a_{zz}$"),
        (arr["psi_opt"], "$\\psi$"),
    ]
    for ax, (vals, label) in zip(axes, params, strict=False):
        ax.plot(Ns, vals, "o-", markersize=5)
        ax.set_ylabel(label, fontsize=12)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("N", fontsize=13)
    fig.suptitle(
        f"Optimal Parameters vs N (ω = {arr['omega'][0]:g})",
        fontsize=14,
    )
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, format="svg", bbox_inches="tight")
    return fig


# ============================================================================
# CLI Entry Point
# ============================================================================


def main() -> None:
    """CLI entry point for generating experiment data."""

    parser = argparse.ArgumentParser(
        description="Generate free-ancilla joint-measurement data."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-generation (overwrite existing files).",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Run only one dataset: exp1, exp2, exp3, decoupled, sonly.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed (default: 42).",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=NM_MAXITER,
        help=f"Nelder-Mead max iterations (default: {NM_MAXITER}).",
    )

    args = parser.parse_args()

    base_seed = args.seed
    maxiter = args.maxiter

    # Ensure output directories exist
    raw_dir = REPORTS_DIR / f"r{REPORT_DATE}" / "raw_data"
    fig_dir = REPORTS_DIR / f"r{REPORT_DATE}" / "figures"
    raw_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    generators: dict[str, tuple[str, Callable[[], Any]]] = {}

    # --- Experiment 1: N=1 compounding ---
    def _run_exp1() -> None:
        print(f"=== Experiment 1: N=1 compounding (seed={base_seed}) ===")
        result = generate_exp1_compounding(seed=base_seed, maxiter=maxiter)
        path = raw_dir / f"{REPORT_DATE}-exp1-compounding.parquet"
        if args.force or not path.exists():
            result.save_parquet(path)
            print(f"  Saved {path}")
        else:
            print(f"  Exists (use --force to overwrite): {path}")

        # Plot
        fig_path_svg = fig_dir / f"{REPORT_DATE}-exp1-ratio-vs-omega.svg"
        fig = plot_ratio_vs_omega(result, save_path=str(fig_path_svg))
        print(f"  Saved {fig_path_svg}")
        plt.close(fig)

    generators["exp1"] = ("Experiment 1: N=1 compounding", _run_exp1)

    # --- Experiment 2: N>1 scaling, J_A=1/2 ---
    def _run_exp2() -> None:
        omega_vals = OMEGA_VALS_SCALING
        for ow in omega_vals:
            print(f"=== Experiment 2: J_A=1/2 scaling, ω={ow} (seed={base_seed}) ===")
            result = generate_exp2_qubit_scaling(
                omega_value=ow,
                seed=base_seed,
                maxiter=maxiter,
            )
            path = raw_dir / f"{REPORT_DATE}-exp2-qubit-scaling-omega{ow}.parquet"
            if args.force or not path.exists():
                result.save_parquet(path)
                print(f"  Saved {path}")
            else:
                print(f"  Exists (use --force to overwrite): {path}")

            fig_path_svg = fig_dir / f"{REPORT_DATE}-exp2-ratio-vs-n-omega{ow}.svg"
            fig = plot_ratio_vs_n(result, save_path=str(fig_path_svg))
            print(f"  Saved {fig_path_svg}")
            plt.close(fig)

            params_path_svg = fig_dir / f"{REPORT_DATE}-exp2-params-vs-n-omega{ow}.svg"
            fig2 = plot_optimal_params_vs_n(result, save_path=str(params_path_svg))
            print(f"  Saved {params_path_svg}")
            plt.close(fig2)

    generators["exp2"] = ("Experiment 2: J_A=1/2 scaling", _run_exp2)

    # --- Experiment 2 S-only control ---
    def _run_sonly() -> None:
        print(f"=== Experiment 2: S-only control (seed={base_seed}) ===")
        result = generate_exp2_sonly_control(
            omega_value=0.2,
            seed=base_seed,
            maxiter=maxiter,
        )
        path = raw_dir / f"{REPORT_DATE}-exp2-sonly-control.parquet"
        if args.force or not path.exists():
            result.save_parquet(path)
            print(f"  Saved {path}")
        else:
            print(f"  Exists (use --force to overwrite): {path}")

    generators["sonly"] = ("Experiment 2: S-only control", _run_sonly)

    # --- Experiment 3: J_A=N/2 scaling ---
    def _run_exp3() -> None:
        omega_vals = OMEGA_VALS_QUICK
        for ow in omega_vals:
            print(f"=== Experiment 3: J_A=N/2 scaling, ω={ow} (seed={base_seed}) ===")
            result = generate_exp3_bipartite_scaling(
                omega_value=ow,
                seed=base_seed,
                maxiter=maxiter,
            )
            path = raw_dir / f"{REPORT_DATE}-exp3-bipartite-scaling-omega{ow}.parquet"
            if args.force or not path.exists():
                result.save_parquet(path)
                print(f"  Saved {path}")
            else:
                print(f"  Exists (use --force to overwrite): {path}")

            fig_path_svg = fig_dir / f"{REPORT_DATE}-exp3-ratio-vs-n-omega{ow}.svg"
            fig = plot_bipartite_ratio_vs_n(result, save_path=str(fig_path_svg))
            print(f"  Saved {fig_path_svg}")
            plt.close(fig)

    generators["exp3"] = ("Experiment 3: J_A=N/2 scaling", _run_exp3)

    # --- Decoupled baseline ---
    def _run_decoupled() -> None:
        print("=== Decoupled baseline ===")
        results = [run_decoupled_baseline(N) for N in [1, 2, 3, 4, 5, 8, 10, 15, 20]]
        scan = QubitFreeAncillaJointNScalingScanResult(results=results)
        path = raw_dir / f"{REPORT_DATE}-decoupled-baseline.parquet"
        if args.force or not path.exists():
            scan.save_parquet(path)
            print(f"  Saved {path}")
        else:
            print(f"  Exists (use --force to overwrite): {path}")

    generators["decoupled"] = ("Decoupled baseline", _run_decoupled)

    # Dispatch
    if args.only is not None:
        if args.only in generators:
            name, fn = generators[args.only]
            print(f"Running: {name}")
            fn()
        else:
            print(f"Unknown dataset '{args.only}'. Options: {list(generators.keys())}")
            sys.exit(1)
    else:
        for name, fn in generators.values():
            print(f"\n--- {name} ---")
            fn()

    print("\nDone.")


if __name__ == "__main__":
    main()
