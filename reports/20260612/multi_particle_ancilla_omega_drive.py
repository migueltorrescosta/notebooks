"""
Local module for the 2026-06-12 Multi-Particle Ancilla ω-Modulated Drive report.

Extends the ω-modulated ancilla drive mechanism (20260611) from a single-particle
ancilla (J_A = 1/2) to an N-particle ancilla (J_A = N/2), matching the system size.

Tests whether the SQL-violation ratio R(N) improves when both system and ancilla
have O(N) contributions to ∂H/∂ω, potentially unlocking Heisenberg-limit scaling.

Operator construction:
- System: (N+1)-dimensional Dicke basis for N particles (J_S = N/2).
- Ancilla: (N+1)-dimensional Dicke basis for N particles (J_A = N/2).
- Full space: (N+1)^2 dimensions with basis ordering:
    {|m_S⟩_S ⊗ |m_A⟩_A}  (m_S, m_A both descending from +J to -J)
    Index: i = m_S_idx * (N+1) + m_A_idx

Circuit: BS_S → Hold → BS_S → measure J_z^S.

Usage:
    uv run python reports/20260612/multi_particle_ancilla_omega_drive.py --force
"""

from __future__ import annotations

__all__ = [
    "NScalingScanResult",
    "plot_n_scaling_optimal_params",
    "plot_n_scaling_ratio",
    "plot_n_scaling_sensitivity",
]

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.linalg import expm

from src.analysis.ancilla_optimization import compute_expectation_and_variance
from src.analysis.decoupled_baseline import (
    build_decoupled_baseline_df,
    generate_decoupled_baseline,
)
from src.analysis.n_scaling_result import (
    NScalingResult,
    NScalingScanResult,
)
from src.analysis.n_scaling_sweep import run_n_scaling_scan
from src.analysis.optimisation_pipeline import (
    TwoPhaseConfig,
    build_nm_result,
    build_rs_result,
    make_4d_objective,
    run_two_phase_pipeline,
)
from src.analysis.sensitivity_metrics import sql_reference
from src.physics.beam_splitter import bs_dicke
from src.physics.dicke_basis import jx_operator, jy_operator, jz_operator
from src.utils.enums import OperatorBasis
from src.utils.paths import report_path_fn
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

# omega values for the scan
OMEGA_VALS: list[float] = [0.1, 0.2, 0.5, 1.0, 2.0]

# N values for the scaling scan (1 to 20)
N_VALS: list[int] = list(range(1, 21))

# Random search parameters
N_RANDOM: int = 500
N_NM_REFINE: int = 50
NM_MAXITER: int = 5000


# ============================================================================
# Path Helpers
# ============================================================================

REPORTS_DIR = Path(__file__).resolve().parent.parent.parent / "reports"
REPORT_DATE = "20260612"


_parquet_path, _fig_path = report_path_fn(REPORTS_DIR, REPORT_DATE)


# ============================================================================
# Operator Construction
# ============================================================================


def build_multi_particle_operators(N: int) -> dict[str, np.ndarray]:
    """Build operators in the (N+1)^2-dimensional total Hilbert space.

    Total space: H_S ⊗ H_A with dimension (N+1)^2.
    Basis ordering: {|m_S⟩_S ⊗ |m_A⟩_A}
    where m_S and m_A each descend from +N/2 to -N/2.
    Basis index: i = m_S_idx * (N+1) + m_A_idx.

    Both S and A use Dicke operators of dimension (N+1) (J_S = J_A = N/2).

    Args:
        N: Number of particles per subsystem (N >= 1).
           System dim = N+1, ancilla dim = N+1, total dim = (N+1)^2.

    Returns:
        Dict with keys 'Jz_S', 'Jx_S', 'Jy_S', 'Jz_A', 'Jx_A', 'Jy_A',
        each an (N+1)^2 x (N+1)^2 complex Hermitian matrix.
        Also includes 'I_S' and 'I_A' ((N+1)x(N+1) identities) and 'I_full'.
    """
    if N < 1:
        raise ValueError(f"N must be >= 1, got {N}")

    d_sys = N + 1  # System Hilbert space dimension
    d_anc = N + 1  # Ancilla Hilbert space dimension
    d_tot = d_sys * d_anc  # Total Hilbert space dimension

    # System and ancilla operators in Dicke basis (same N for both)
    Jz_dicke = jz_operator(N, basis=OperatorBasis.DICKE)
    Jx_dicke = jx_operator(N, basis=OperatorBasis.DICKE)
    Jy_dicke = jy_operator(N, basis=OperatorBasis.DICKE)
    I_S = np.eye(d_sys, dtype=complex)

    # Embed into total space via Kronecker products
    ops: dict[str, np.ndarray] = {
        # System: J_k(N) ⊗ I_{N+1}
        "Jz_S": np.kron(Jz_dicke, I_S).astype(complex),
        "Jx_S": np.kron(Jx_dicke, I_S).astype(complex),
        "Jy_S": np.kron(Jy_dicke, I_S).astype(complex),
        # Ancilla: I_{N+1} ⊗ J_k(N)
        "Jz_A": np.kron(I_S, Jz_dicke).astype(complex),
        "Jx_A": np.kron(I_S, Jx_dicke).astype(complex),
        "Jy_A": np.kron(I_S, Jy_dicke).astype(complex),
        # Identities
        "I_S": I_S,
        "I_A": np.eye(d_anc, dtype=complex),
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
            f"{key} is not Hermitian for N={N}"
        )

    # Validate commutation relations: [J_z^S, J_x^S] = i J_y^S
    comm = ops["Jz_S"] @ ops["Jx_S"] - ops["Jx_S"] @ ops["Jz_S"]
    expected = 1j * ops["Jy_S"]
    assert np.allclose(comm, expected, atol=1e-10), (
        f"[J_z^S, J_x^S] = i J_y^S violated for N={N}"
    )

    # Validate commutation relations for ancilla: [J_z^A, J_x^A] = i J_y^A
    comm_a = ops["Jz_A"] @ ops["Jx_A"] - ops["Jx_A"] @ ops["Jz_A"]
    expected_a = 1j * ops["Jy_A"]
    assert np.allclose(comm_a, expected_a, atol=1e-10), (
        f"[J_z^A, J_x^A] = i J_y^A violated for N={N}"
    )

    return ops


def build_multi_particle_system_only_bs_unitary(
    N: int,
    T_bs: float = T_BS,
) -> np.ndarray:
    """System-only beam-splitter unitary in the multi-particle space.

    U_BS_S = exp(-i T_bs J_x(N)) ⊗ I_{N+1}  (acts on system, identity on ancilla).

    Args:
        N: Number of particles per subsystem.
        T_bs: Beam-splitter duration (default pi/2 for 50/50).

    Returns:
        (N+1)^2 x (N+1)^2 unitary matrix.
    """
    d_tot = (N + 1) ** 2
    bs_sys = bs_dicke(N, T_bs)
    I_A = np.eye(N + 1, dtype=complex)
    U = np.kron(bs_sys, I_A).astype(complex)
    I_full = np.eye(d_tot, dtype=complex)
    assert np.allclose(U @ U.conj().T, I_full, atol=1e-12), (
        f"BS unitary not unitary for N={N}, T_bs={T_bs}"
    )
    return U


def build_multi_particle_phase_modulated_drive_hamiltonian(
    N: int,
    omega: float,
    a_x: float,
    a_y: float,
    a_z: float,
    ops: dict[str, np.ndarray],
) -> np.ndarray:
    """Build the omega-modulated ancilla drive Hamiltonian.

    H_A = omega (a_x J_x^A + a_y J_y^A + a_z J_z^A)

    Args:
        N: Number of particles per subsystem (for dimension check).
        omega: Phase rate parameter (scales the whole drive).
        a_x: J_x^A coefficient.
        a_y: J_y^A coefficent.
        a_z: J_z^A coefficient.
        ops: Operators from build_multi_particle_operators(N).

    Returns:
        (N+1)^2 x (N+1)^2 Hermitian matrix.
    """
    d_tot = (N + 1) ** 2
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


def build_multi_particle_iszz_interaction(
    N: int,
    a_zz: float,
    ops: dict[str, np.ndarray],
) -> np.ndarray:
    """Build the Ising interaction in the multi-particle space.

    H_int = a_zz J_z^S J_z^A = a_zz (J_z(N) ⊗ I_{N+1}) @ (I_{N+1} ⊗ J_z(N))

    Args:
        N: Number of particles per subsystem.
        a_zz: Interaction coupling coefficient.
        ops: Operators from build_multi_particle_operators(N).

    Returns:
        (N+1)^2 x (N+1)^2 Hermitian matrix.
    """
    H = np.zeros(((N + 1) ** 2, (N + 1) ** 2), dtype=complex)
    if a_zz != 0.0:
        H += a_zz * (ops["Jz_S"] @ ops["Jz_A"])
    # Enforce Hermiticity
    H = 0.5 * (H + H.conj().T)
    assert np.allclose(H, H.conj().T, atol=1e-12), (
        f"Ising interaction not Hermitian for N={N}"
    )
    return H


def build_multi_particle_hold_hamiltonian(
    N: int,
    omega: float,
    a_x: float,
    a_y: float,
    a_z: float,
    a_zz: float,
    ops: dict[str, np.ndarray],
) -> np.ndarray:
    """Build the total holding Hamiltonian for the multi-particle system.

    H = omega J_z^S + omega (a_x J_x^A + a_y J_y^A + a_z J_z^A) + a_zz J_z^S J_z^A

    Args:
        N: Number of particles per subsystem.
        omega: Phase rate parameter.
        a_x: J_x^A drive coefficient.
        a_y: J_y^A drive coefficient.
        a_z: J_z^A drive coefficient.
        a_zz: Ising interaction coefficient.
        ops: Operators from build_multi_particle_operators(N).

    Returns:
        (N+1)^2 x (N+1)^2 Hermitian Hamiltonian matrix.
    """
    H = omega * ops["Jz_S"]
    H += build_multi_particle_phase_modulated_drive_hamiltonian(
        N,
        omega,
        a_x,
        a_y,
        a_z,
        ops,
    )
    H += build_multi_particle_iszz_interaction(N, a_zz, ops)
    # Enforce Hermiticity
    H = 0.5 * (H + H.conj().T)
    assert np.allclose(H, H.conj().T, atol=1e-12), (
        f"Total Hamiltonian not Hermitian for N={N}"
    )
    return H


# ============================================================================
# State Preparation
# ============================================================================


def multi_particle_initial_state(N: int) -> np.ndarray:
    """Initial state for the multi-particle system.

    |Psi_0> = |J_S, J_S>_S ⊗ |J_A, J_A>_A

    Both subsystems are in the top Dicke state |+N/2>.
    In the full basis, this is the first basis vector: [1, 0, ..., 0]^T
    of length (N+1)^2.

    Args:
        N: Number of particles per subsystem.

    Returns:
        Normalised complex vector of length (N+1)^2.
    """
    d_tot = (N + 1) ** 2
    psi = np.zeros(d_tot, dtype=complex)
    psi[0] = 1.0
    assert np.isclose(np.linalg.norm(psi), 1.0), (
        f"Initial state not normalised for N={N}"
    )
    return psi


# ============================================================================
# Circuit Evolution
# ============================================================================


def multi_particle_hold_unitary(
    N: int,
    t_hold: float,
    omega: float,
    a_x: float,
    a_y: float,
    a_z: float,
    a_zz: float,
    ops: dict[str, np.ndarray],
) -> np.ndarray:
    """Holding-time unitary for the multi-particle omega-modulated protocol.

    U_hold(t_hold) = exp(-i t_hold H)
    where H = omega J_z^S + omega(a_x J_x^A + a_y J_y^A + a_z J_z^A) + a_zz J_z^S J_z^A.

    Args:
        N: Number of particles per subsystem.
        t_hold: Holding-time strength.
        omega: Phase rate parameter.
        a_x: J_x^A drive coefficient.
        a_y: J_y^A drive coefficient.
        a_z: J_z^A drive coefficient.
        a_zz: Ising interaction coefficient.
        ops: Operators from build_multi_particle_operators(N).

    Returns:
        (N+1)^2 x (N+1)^2 unitary matrix.
    """
    H = build_multi_particle_hold_hamiltonian(N, omega, a_x, a_y, a_z, a_zz, ops)
    U = expm(-1j * t_hold * H)
    d_tot = (N + 1) ** 2
    I_full = np.eye(d_tot, dtype=complex)
    assert np.allclose(U @ U.conj().T, I_full, atol=1e-12), (
        f"Hold unitary not unitary for N={N}, t_hold={t_hold}, omega={omega}"
    )
    return U


def evolve_multi_particle_circuit(
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
    """Run the full multi-particle omega-modulated ancilla MZI circuit.

    |psi_final> = U_BS_S * U_hold(t_hold) * U_BS_S * |psi_0>

    Args:
        N: Number of particles per subsystem.
        psi0: Initial state vector (must be normalised).
        T_bs: Beam-splitter duration.
        t_hold: Holding-time strength.
        omega: Phase rate parameter.
        a_x: J_x^A drive coefficient.
        a_y: J_y^A drive coefficient.
        a_z: J_z^A drive coefficient.
        a_zz: Ising interaction coefficient.
        ops: Operators from build_multi_particle_operators(N).

    Returns:
        Final normalised state vector.
    """
    assert np.isclose(np.linalg.norm(psi0), 1.0), (
        f"Initial state not normalised for N={N}"
    )
    U_bs = build_multi_particle_system_only_bs_unitary(N, T_bs)
    psi = U_bs @ psi0
    psi = (
        multi_particle_hold_unitary(
            N,
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
    assert np.isclose(np.linalg.norm(psi), 1.0), f"Final state not normalised for N={N}"
    return psi


def compute_multi_particle_sensitivity(
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
    """Compute the error-propagation sensitivity Delta_omega.

    Delta_omega = sqrt(Var(O)) / |d<O>/domega|

    where O = J_z^S (default measurement operator).

    The central finite-difference captures the full omega-dependence
    (both omega J_z^S and omega-modulated ancilla drive channels).

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
        ops: Operators from build_multi_particle_operators(N).
        fd_step: Finite-difference step size (default 1e-6).
        meas_op: Measurement operator (default ops['Jz_S']).

    Returns:
        Sensitivity Delta_omega. Returns inf if derivative is zero (fringe extremum).
    """
    if meas_op is None:
        meas_op = ops["Jz_S"]

    # Evaluate at omega_true
    psi = evolve_multi_particle_circuit(
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

    # Central finite difference for d<O>/domega
    psi_plus = evolve_multi_particle_circuit(
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
    psi_minus = evolve_multi_particle_circuit(
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


def compute_multi_particle_decoupled_baseline(
    N: int,
    omega_true: float = 1.0,
) -> float:
    """Compute the decoupled baseline sensitivity Delta_omega for N particles.

    At (a_x = a_y = a_z = a_zz = 0), the circuit reduces to a standard
    N-particle MZI with CSS input, giving Delta_omega = 1/(sqrt(N) * T_HOLD).

    Args:
        N: Number of particles per subsystem.
        omega_true: Phase rate value.

    Returns:
        Delta_omega at the decoupled configuration.
    """
    ops = build_multi_particle_operators(N)
    psi0 = multi_particle_initial_state(N)
    return compute_multi_particle_sensitivity(
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


def verify_multi_particle_decoupled_baseline(
    N_values: list[int] | None = None,
    omega_values: list[float] | None = None,
    rtol: float = 1e-10,
) -> dict[tuple[int, float], bool]:
    """Verify the decoupled baseline for all (N, omega) pairs.

    At zero drive and zero interaction, the sensitivity must equal
    Delta_omega = 1/(sqrt(N) * T_HOLD) to machine precision.

    Args:
        N_values: List of N values (default: 1 to 20).
        omega_values: List of omega values (default: all OMEGA_VALS).
        rtol: Relative tolerance for comparison.

    Returns:
        Dict mapping (N, omega) -> PASS/FAIL (True/False).
    """
    if N_values is None:
        N_values = N_VALS
    if omega_values is None:
        omega_values = OMEGA_VALS
    results: dict[tuple[int, float], bool] = {}
    for N in N_values:
        sql_ref = sql_reference(N)
        for omega in omega_values:
            delta = compute_multi_particle_decoupled_baseline(N, omega)
            results[(N, omega)] = bool(np.isclose(delta, sql_ref, rtol=rtol))
    return results


# ============================================================================
# N-Scaling Scan (Single (N, omega) Worker)
# ============================================================================


def _make_multi_objective(
    N: int,
    omega: float,
    ops: dict[str, np.ndarray],
    psi0: np.ndarray,
) -> Callable[[np.ndarray], float]:
    """Build the raw (unpenalised) Δω objective for a given (N, ω) pair.

    Uses ``make_4d_objective`` from the shared pipeline module.
    """
    from functools import partial

    return make_4d_objective(
        partial(compute_multi_particle_sensitivity, N),
        psi0=psi0,
        T_BS=T_BS,
        t_hold=T_HOLD,
        omega=omega,
        ops=ops,
        fd_step=FD_STEP,
    )


def run_single_n_omega(
    N: int,
    omega: float,
    n_random: int = N_RANDOM,
    n_nm_refine: int = N_NM_REFINE,
    seed: int | None = 42,
) -> NScalingResult:
    """Run the full optimisation pipeline for a single (N, omega) pair.

    1. 4D random search (n_random samples).
    2. Nelder-Mead refinement from top n_nm_refine points.
    3. Return the best result.

    Args:
        N: Number of particles per subsystem.
        omega: Phase rate value.
        n_random: Number of random search samples.
        n_nm_refine: Number of Nelder-Mead refinements.
        seed: Base random seed (incremented per call).

    Returns:
        NScalingResult with the optimal parameters and sensitivity.
    """
    base_seed = seed if seed is not None else 42
    ops = build_multi_particle_operators(N)
    psi0 = multi_particle_initial_state(N)
    raw_obj = _make_multi_objective(N, omega, ops, psi0)

    best_nm, _ = run_two_phase_pipeline(
        random_search_fn=lambda n_samples, seed, **kw: build_rs_result(
            raw_obj,
            n_samples,
            seed,
            omega=omega,
            sql=sql_reference(N),
            t_hold=T_HOLD,
        ),
        nm_fn=lambda x0, seed, **kw: build_nm_result(
            raw_obj,
            x0,
            omega=omega,
            ops=ops,
            psi0=psi0,
            evolve_fn=lambda psi, ax, ay, az, azz, _ops: evolve_multi_particle_circuit(
                N,
                psi,
                T_BS,
                T_HOLD,
                omega,
                ax,
                ay,
                az,
                azz,
                _ops,
            ),
            t_hold=T_HOLD,
            maxiter=NM_MAXITER,
        ),
        config=TwoPhaseConfig(
            n_random=n_random, n_nm_refine=n_nm_refine, seed=base_seed
        ),
    )

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
# Data Generation Pipeline
# ============================================================================


def generate_n_scaling_scan(force: bool = False) -> None:
    """Full N-scaling scan: 20 N values x 5 omega values = 100 optimisation runs.

    Each (N, omega) pair runs:
      1. 4D random search with 500 points.
      2. Nelder-Mead refinement from top 50 points.

    Results are saved to a single Parquet file. Intermediate checkpoints
    are saved per-N-value to allow resumption if interrupted.

    Delegates to :func:`src.analysis.n_scaling_sweep.run_n_scaling_scan`.

    Args:
        force: Re-run even if Parquet exists.
    """
    run_n_scaling_scan(
        force=force,
        run_single_n_omega=run_single_n_omega,
        n_values=N_VALS,
        omega_values=OMEGA_VALS,
        parquet_path=_parquet_path("n-scaling-scan"),
        checkpoint_dir=REPORTS_DIR / REPORT_DATE / "raw_data" / "checkpoints",
        fig_ratio_path=_fig_path("n-scaling-ratio"),
        fig_sensitivity_path=_fig_path("n-scaling-sensitivity"),
        fig_params_path=_fig_path("n-scaling-optimal-params"),
        t_hold=T_HOLD,
    )


def generate_n1_consistency(force: bool = False) -> None:
    """Verify N=1 consistency with the 20260519 report.

    At N=1, omega=0.2, the pipeline should find Delta_omega ~ 0.02036 (R ~ 4.91).
    """
    parquet_p = _parquet_path("n1-consistency")

    if parquet_p.exists() and not force:
        print(f"[skip] {parquet_p.name} exists (use --force to overwrite)")
        df = pd.read_parquet(parquet_p)
    else:
        print("[run] N=1 consistency check at omega=0.2...")
        result = run_single_n_omega(N=1, omega=0.2)
        result.save_parquet(parquet_p)
        print(f"[save] {parquet_p}")
        df = result.to_dataframe()

    delta = float(df["delta_omega_opt"].iloc[0])
    ratio = float(df["ratio"].iloc[0])
    print(f"  N=1, omega=0.2: Delta_omega = {delta:.6f}, R = {ratio:.3f}")
    print("  Expected:   Delta_omega approx 0.02036, R approx 4.91")


# ============================================================================
# CLI Entry Point
# ============================================================================


def main() -> None:
    """CLI entry point for generating all data and figures."""
    parser = argparse.ArgumentParser(
        description="Multi-Particle Ancilla omega-Modulated Drive (20260612)",
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

    generators: dict[str, tuple[str, str | Callable[..., None], bool]] = {
        "decoupled-baseline": (
            "Decoupled Baseline Verification",
            lambda force=False: generate_decoupled_baseline(
                force=force,
                parquet_path=_parquet_path("decoupled-baseline"),
                compute_fn=build_decoupled_baseline_df,
                compute_kwargs={
                    "compute_fn": compute_multi_particle_decoupled_baseline
                },
                label="decoupled baseline",
            ),
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
        name, func_or_name, _ = generators[key]
        print(f"\n{'=' * 60}")
        print(f"  {name}")
        print(f"{'=' * 60}")
        func = (
            globals()[func_or_name] if isinstance(func_or_name, str) else func_or_name
        )
        func(force=force)


if __name__ == "__main__":
    import sys

    main()
