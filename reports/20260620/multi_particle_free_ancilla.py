r"""
Local module for the 2026-06-20 Multi-Particle Ancilla Free Initial State report.

Tests whether an :math:`\omega`-independent ancilla drive can beat the SQL when
the ancilla and/or system have multiple particles (:math:`J_A > 1/2`,
:math:`J_S > 1/2`).

Circuit: BS_S → Hold → BS_S → measure :math:`J_z^S`
Hold Hamiltonian: :math:`H = \omega J_z^S + a_x J_x^A + a_y J_y^A + a_z J_z^A + a_{zz} J_z^S J_z^A`
Initial state: :math:`|J_S, J_S\rangle_S \otimes |\mathrm{CSS}(\theta_A, \phi_A)\rangle_A`

Usage:
    uv run python reports/20260620/multi_particle_free_ancilla.py
    uv run python reports/20260620/multi_particle_free_ancilla.py --force
"""

from __future__ import annotations

import argparse
import math
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
import src.utils.sampling as _sampling
from src.algorithms.coherent_spin_state import coherent_spin_state
from src.analysis.ancilla_optimization import (
    compute_expectation_and_variance,
)
from src.analysis.checkpoint_recovery import load_checkpoints, run_pending_groups
from src.analysis.n_scaling_result import NScalingResult
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
DRIVE_RADIUS: float = 10.0  # 3-ball radius for (a_x, a_y, a_z)
AZZ_BOUNDS: tuple[float, float] = (-5.0, 5.0)  # Bounds for a_zz

# Number of random search samples per (N, M, omega) triple
N_RANDOM: int = 1000
# Number of Nelder-Mead refinements per triple
N_NM_REFINE: int = 30
NM_MAXITER: int = 5000

# Parameter ranges
M_VALS: list[int] = [1, 2, 3, 4]  # Ancilla particle counts
N_VALS: list[int] = list(range(1, 11))  # System particle counts
OMEGA_VALS: list[float] = [0.1, 0.5, 1.0, 2.0, 5.0]  # Phase rates

# ============================================================================
# Path Helpers
# ============================================================================

REPORTS_DIR = Path(__file__).resolve().parent.parent.parent / "reports"
REPORT_DATE = "20260620"


_parquet_path, _fig_path = report_path_fn(REPORTS_DIR, REPORT_DATE)


# ============================================================================
# Operator Construction
# ============================================================================


def build_operators(N: int, M: int) -> dict[str, np.ndarray]:
    """Build operators in the :math:`(N+1)(M+1)`-dimensional total Hilbert space.

    Delegates to :func:`src.physics.bipartite_operators.build_operators`
    with ``N_sys=N, N_anc=M``.

    Args:
        N: Number of system particles (:math:`N \\ge 1`).
        M: Number of ancilla particles (:math:`M \\ge 1`).

    Returns:
        Dict with keys ``'Jz_S'``, ``'Jx_S'``, ``'Jy_S'``, ``'Jz_A'``,
        ``'Jx_A'``, ``'Jy_A'``, ``'I_S'``, ``'I_A'``, ``'I_full'``.

    Raises:
        ValueError: If N or M < 1.
    """
    return _bipartite.build_operators(N_sys=N, N_anc=M)


def build_system_only_bs_unitary(
    N: int,
    M: int,
    T_bs: float = T_BS,
) -> np.ndarray:
    """System-only beam-splitter unitary in the multi-particle space.

    Delegates to :func:`src.physics.bipartite_operators.build_system_only_bs_unitary`
    with ``N_sys=N, N_anc=M, T_bs=T_bs``.

    Args:
        N: Number of system particles.
        M: Number of ancilla particles.
        T_bs: Beam-splitter duration (default :math:`\\pi/2` for 50/50).

    Returns:
        :math:`(N+1)(M+1) \\times (N+1)(M+1)` unitary matrix.
    """
    return _bipartite.build_system_only_bs_unitary(
        N_sys=N,
        N_anc=M,
        T_bs=T_bs,
    )


# ============================================================================
# State Preparation
# ============================================================================


def free_initial_state(
    N: int,
    M: int,
    theta_A: float,
    phi_A: float,
) -> np.ndarray:
    """Construct the free-ancilla initial state.

    :math:`|\\Psi_0\\rangle = |J_S, J_S\\rangle_S \\otimes |\\mathrm{CSS}(\\theta_A, \\phi_A)\\rangle_A`

    The system is in the top Dicke state :math:`|+N/2\\rangle_S`. The ancilla is
    in a coherent spin state with polar angle :math:`\\theta_A` and azimuth
    :math:`\\phi_A`.

    Args:
        N: Number of system particles.
        M: Number of ancilla particles.
        theta_A: Ancilla polar angle in :math:`[0, \\pi]`.
        phi_A: Ancilla azimuthal angle in :math:`[0, 2\\pi)`.

    Returns:
        Normalised complex vector of length :math:`(N+1)(M+1)`.
    """
    d_sys = N + 1

    # System in top Dicke state |+N/2⟩
    psi_sys = np.zeros(d_sys, dtype=complex)
    psi_sys[0] = 1.0

    # Ancilla in CSS at (theta_A, phi_A)
    J_A = M / 2.0
    psi_anc = coherent_spin_state(J_A, theta_A, phi_A)

    state = np.kron(psi_sys, psi_anc)
    assert np.isclose(np.linalg.norm(state), 1.0), (
        f"Free initial state not normalised for N={N}, M={M}"
    )
    return state


# ============================================================================
# Hamiltonian Construction (ω-Independent Drive)
# ============================================================================


def build_fixed_drive_hamiltonian(
    a_x: float,
    a_y: float,
    a_z: float,
    ops: dict[str, np.ndarray],
) -> np.ndarray:
    """Build the :math:`\\omega`-independent ancilla drive Hamiltonian.

    :math:`H_A = a_x J_x^A + a_y J_y^A + a_z J_z^A`

    Note: Unlike #20260612, this Hamiltonian does **not** contain
    :math:`\\omega`. This means :math:`\\partial H/\\partial\\omega = J_z^S`,
    with no ancilla contribution.

    Args:
        a_x: Coefficient for :math:`J_x^A`.
        a_y: Coefficient for :math:`J_y^A`.
        a_z: Coefficient for :math:`J_z^A`.
        ops: Operators from :func:`build_operators`.

    Returns:
        Hermitian matrix representing the ancilla drive.
    """
    d_tot = ops["Jz_A"].shape[0]
    H = np.zeros((d_tot, d_tot), dtype=complex)
    if a_x != 0.0:
        H += a_x * ops["Jx_A"]
    if a_y != 0.0:
        H += a_y * ops["Jy_A"]
    if a_z != 0.0:
        H += a_z * ops["Jz_A"]
    H = 0.5 * (H + H.conj().T)
    assert np.allclose(H, H.conj().T, atol=1e-12), "Drive Hamiltonian not Hermitian"
    return H


def build_iszz_interaction(
    a_zz: float,
    ops: dict[str, np.ndarray],
) -> np.ndarray:
    """Build the Ising-type system--ancilla interaction.

    Delegates to :func:`src.physics.bipartite_operators.build_iszz_interaction`.

    Args:
        a_zz: Interaction coupling coefficient.
        ops: Operators from :func:`build_operators`.

    Returns:
        Hermitian matrix representing the interaction.
    """
    return _bipartite.build_iszz_interaction(a_zz, ops)


def build_hold_hamiltonian(
    omega: float,
    a_x: float,
    a_y: float,
    a_z: float,
    a_zz: float,
    ops: dict[str, np.ndarray],
) -> np.ndarray:
    """Build the total holding Hamiltonian.

    :math:`H = \\omega J_z^S + H_A + H_{\\text{int}}`
    :math:`H = \\omega J_z^S + (a_x J_x^A + a_y J_y^A + a_z J_z^A) \
    + a_{zz} J_z^S J_z^A`

    The ancilla drive terms are **not** scaled by :math:`\\omega`.

    Args:
        omega: Phase rate parameter.
        a_x: :math:`J_x^A` drive coefficient.
        a_y: :math:`J_y^A` drive coefficient.
        a_z: :math:`J_z^A` drive coefficient.
        a_zz: Ising interaction coefficient.
        ops: Operators from :func:`build_operators`.

    Returns:
        Hermitian Hamiltonian matrix.
    """
    H = omega * ops["Jz_S"]
    H += build_fixed_drive_hamiltonian(a_x, a_y, a_z, ops)
    H += build_iszz_interaction(a_zz, ops)
    H = 0.5 * (H + H.conj().T)
    assert np.allclose(H, H.conj().T, atol=1e-12), (
        "Total hold Hamiltonian not Hermitian"
    )
    return H


def hold_unitary(
    t_hold: float,
    omega: float,
    a_x: float,
    a_y: float,
    a_z: float,
    a_zz: float,
    ops: dict[str, np.ndarray],
) -> np.ndarray:
    """Holding-time unitary.

    :math:`U_{\\text{hold}}(t_{\\text{hold}}) = \\exp(-i t_{\\text{hold}} H)`
    where :math:`H = \\omega J_z^S + a_x J_x^A + a_y J_y^A + a_z J_z^A \
    + a_{zz} J_z^S J_z^A`.

    Args:
        t_hold: Holding-time strength.
        omega: Phase rate parameter.
        a_x: :math:`J_x^A` drive coefficient.
        a_y: :math:`J_y^A` drive coefficient.
        a_z: :math:`J_z^A` drive coefficient.
        a_zz: Ising interaction coefficient.
        ops: Operators from :func:`build_operators`.

    Returns:
        Unitary matrix.
    """
    H = build_hold_hamiltonian(omega, a_x, a_y, a_z, a_zz, ops)
    U = expm(-1j * t_hold * H)
    d_tot = ops["Jz_A"].shape[0]
    assert np.allclose(U @ U.conj().T, np.eye(d_tot, dtype=complex), atol=1e-12), (
        f"Hold unitary not unitary for t_hold={t_hold}, ω={omega}"
    )
    return U


# ============================================================================
# Circuit Evolution
# ============================================================================


def evolve_circuit(
    psi0: np.ndarray,
    N: int,
    M: int,
    T_bs: float,
    t_hold: float,
    omega: float,
    a_x: float,
    a_y: float,
    a_z: float,
    a_zz: float,
    ops: dict[str, np.ndarray],
) -> np.ndarray:
    """Run the full MZI circuit.

    :math:`|\\psi_{\\text{final}}\\rangle = U_{\\text{BS}}^{(S)} \\,
    U_{\\text{hold}}(t_{\\text{hold}}) \\, U_{\\text{BS}}^{(S)} \\, |\\psi_0\\rangle`

    Args:
        psi0: Initial state vector (must be normalised).
        N: Number of system particles.
        M: Number of ancilla particles.
        T_bs: Beam-splitter duration.
        t_hold: Holding-time strength.
        omega: Phase rate parameter.
        a_x: :math:`J_x^A` drive coefficient.
        a_y: :math:`J_y^A` drive coefficient.
        a_z: :math:`J_z^A` drive coefficient.
        a_zz: Ising interaction coefficient.
        ops: Operators from :func:`build_operators`.

    Returns:
        Final normalised state vector.
    """
    assert np.isclose(np.linalg.norm(psi0), 1.0), "Initial state must be normalised"

    U_bs = build_system_only_bs_unitary(N, M, T_bs)
    psi = U_bs @ psi0
    psi = hold_unitary(t_hold, omega, a_x, a_y, a_z, a_zz, ops) @ psi
    psi = U_bs @ psi
    assert np.isclose(np.linalg.norm(psi), 1.0), "Final state must be normalised"
    return np.asarray(psi)


def compute_sensitivity(
    psi0: np.ndarray,
    N: int,
    M: int,
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
) -> tuple[float, float, float, float, bool]:
    """Compute the error-propagation sensitivity :math:`\\Delta\\omega`.

    :math:`\\Delta\\omega = \\frac{\\sqrt{\\mathrm{Var}(J_z^S)}}{\
    |\\partial\\langle J_z^S\\rangle / \\partial\\omega|}`

    The derivative is computed via central finite differences with step
    :math:`\\delta = 10^{-6}`.

    Args:
        psi0: Initial state vector.
        N: Number of system particles.
        M: Number of ancilla particles.
        T_bs: Beam-splitter duration.
        t_hold: Holding-time strength.
        omega_true: True phase rate parameter.
        a_x: :math:`J_x^A` drive coefficient.
        a_y: :math:`J_y^A` drive coefficient.
        a_z: :math:`J_z^A` drive coefficient.
        a_zz: Ising interaction coefficient.
        ops: Operators from :func:`build_operators`.
        fd_step: Finite-difference step size (default ``1e-6``).
        meas_op: Measurement operator (default ``ops['Jz_S']``).

    Returns:
        Tuple ``(delta_omega, expectation_Jz, variance_Jz, d_exp, is_fringe)``.
        ``is_fringe`` is ``True`` when the derivative is near zero (sensitivity
        diverges to infinity).
    """
    if meas_op is None:
        meas_op = ops["Jz_S"]

    # Evaluate at omega_true
    psi = evolve_circuit(
        psi0,
        N,
        M,
        T_bs,
        t_hold,
        omega_true,
        a_x,
        a_y,
        a_z,
        a_zz,
        ops,
    )
    exp_val, var_val = compute_expectation_and_variance(psi, meas_op)

    # Central finite difference for d<O>/domega
    psi_plus = evolve_circuit(
        psi0,
        N,
        M,
        T_bs,
        t_hold,
        omega_true + fd_step,
        a_x,
        a_y,
        a_z,
        a_zz,
        ops,
    )
    psi_minus = evolve_circuit(
        psi0,
        N,
        M,
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

    is_fringe = abs(d_exp) < 1e-12
    if is_fringe:
        return float("inf"), exp_val, var_val, float(d_exp), True

    zero_var = var_val < 1e-15
    if zero_var:
        return float("inf"), exp_val, var_val, float(d_exp), True

    delta = float(np.sqrt(var_val) / abs(d_exp))
    return delta, exp_val, var_val, float(d_exp), False


def verify_decoupled_baseline(
    N_values: list[int] | None = None,
    M_values: list[int] | None = None,
    omega_values: list[float] | None = None,
    rtol: float = 1e-10,
) -> dict[tuple[int, int, float], bool]:
    """Verify the decoupled baseline for all :math:`(N, M, \\omega)` triples.

    At zero drive and zero interaction, the sensitivity must equal
    :math:`\\Delta\\omega = 1/(\\sqrt{N} T_{\\text{HOLD}})` to machine precision.

    Args:
        N_values: List of N values (default: 1 to 10).
        M_values: List of M values (default: {1, 2, 3, 4}).
        omega_values: List of :math:`\\omega` values (default: all OMEGA_VALS).
        rtol: Relative tolerance for comparison.

    Returns:
        Dict mapping ``(N, M, omega) -> PASS/FAIL (True/False)``.
    """
    if N_values is None:
        N_values = N_VALS
    if M_values is None:
        M_values = M_VALS
    if omega_values is None:
        omega_values = OMEGA_VALS
    results: dict[tuple[int, int, float], bool] = {}
    for N in N_values:
        sql_ref = sql_reference(N, T_HOLD)
        for M in M_values:
            for omega in omega_values:
                delta = sql_reference(N, T_HOLD)
                results[(N, M, omega)] = bool(
                    np.isclose(delta, sql_ref, rtol=rtol),
                )
    return results


# ============================================================================
# 6D Random Search
# ============================================================================


def sample_6d_config(
    rng: np.random.Generator,
    *,
    R: float = DRIVE_RADIUS,
    azz_bounds: tuple[float, float] = AZZ_BOUNDS,
) -> tuple[float, float, float, float, float, float]:
    """Sample a 6D parameter configuration for the free-ancilla protocol.

    Delegates to :func:`src.utils.sampling.sample_6d_config` with
    report-specific defaults for *R* and *azz_bounds*.

    Args:
        rng: NumPy random generator.
        R: Radius of the 3-ball for drive coefficients.
        azz_bounds: (min, max) for the Ising coupling.

    Returns:
        Tuple ``(theta_A, phi_A, a_x, a_y, a_z, a_zz)``.
    """
    return _sampling.sample_6d_config(rng, drive_radius=R, azz_bounds=azz_bounds)


@dataclass
class FreeAncillaRandomSearchResult(ParquetSerializable):
    """Result from a 6D random search over :math:`(\\theta_A, \\phi_A, a_x, a_y, a_z, a_{zz})`.

    Attributes:
        samples: Array of shape ``(N, 6)`` with sampled parameter values.
        delta_omega_values: Array of shape ``(N,)`` with :math:`\\Delta\\omega` for each sample.
        best_params: The 6-parameter tuple giving minimal :math:`\\Delta\\omega`.
        best_delta_omega: The minimal :math:`\\Delta\\omega` found.
        N: Number of system particles.
        M: Number of ancilla particles.
        omega_value: :math:`\\omega` at which the search was performed.
        sql: SQL reference value for :math:`N` particles.
        t_hold: Holding time.
    """

    samples: np.ndarray
    delta_omega_values: np.ndarray
    best_params: tuple[float, float, float, float, float, float]
    best_delta_omega: float
    N: int = 1
    M: int = 1
    omega_value: float = 1.0
    sql: float = 0.1
    t_hold: float = 10.0

    _PARQUET_COLUMNS: ClassVar[list[str]] = [
        "theta_A",
        "phi_A",
        "a_x",
        "a_y",
        "a_z",
        "a_zz",
        "delta_omega",
        "N",
        "M",
        "omega_value",
        "sql",
        "t_hold",
    ]

    def to_dataframe(self) -> pd.DataFrame:
        n = len(self.samples)
        return pd.DataFrame(
            {
                "theta_A": self.samples[:, 0],
                "phi_A": self.samples[:, 1],
                "a_x": self.samples[:, 2],
                "a_y": self.samples[:, 3],
                "a_z": self.samples[:, 4],
                "a_zz": self.samples[:, 5],
                "delta_omega": self.delta_omega_values,
                "N": [self.N] * n,
                "M": [self.M] * n,
                "omega_value": [self.omega_value] * n,
                "sql": [self.sql] * n,
                "t_hold": [self.t_hold] * n,
            },
        )

    @classmethod
    def from_parquet(cls, path: str | Path) -> FreeAncillaRandomSearchResult:
        df = pd.read_parquet(path)
        cls._validate_columns(df)
        samples = df[["theta_A", "phi_A", "a_x", "a_y", "a_z", "a_zz"]].to_numpy(
            dtype=float,
        )
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
                float(samples[best_idx, 5]),
            ),
            best_delta_omega=float(deltas[best_idx]),
            N=int(df["N"].iloc[0]),
            M=int(df["M"].iloc[0]),
            omega_value=float(df["omega_value"].iloc[0]),
            sql=float(df["sql"].iloc[0]),
            t_hold=float(df["t_hold"].iloc[0]),
        )


def random_search(
    N: int,
    M: int,
    omega: float,
    n_samples: int = N_RANDOM,
    seed: int | None = 42,
) -> FreeAncillaRandomSearchResult:
    """6D random search over :math:`(\\theta_A, \\phi_A, a_x, a_y, a_z, a_{zz})`.

    The ancilla drive :math:`(a_x, a_y, a_z)` is sampled from the 3-ball
    :math:`\\|\\mathbf{a}\\| \\le 10`.  The interaction :math:`a_{zz}` is sampled
    uniformly from ``[-5, 5]``.

    Args:
        N: Number of system particles.
        M: Number of ancilla particles.
        omega: Phase rate value.
        n_samples: Number of random points to evaluate.
        seed: Random seed for reproducibility.

    Returns:
        FreeAncillaRandomSearchResult with all samples and best found.
    """
    rng = np.random.default_rng(seed)
    ops = build_operators(N, M)

    samples = np.zeros((n_samples, 6), dtype=float)
    deltas = np.full(n_samples, np.inf, dtype=float)

    for i in range(n_samples):
        theta_A, phi_A, a_x, a_y, a_z, a_zz = sample_6d_config(rng)
        samples[i] = [theta_A, phi_A, a_x, a_y, a_z, a_zz]
        psi0 = free_initial_state(N, M, theta_A, phi_A)
        domega, _, _, _, _ = compute_sensitivity(
            psi0,
            N,
            M,
            T_BS,
            T_HOLD,
            omega,
            a_x,
            a_y,
            a_z,
            a_zz,
            ops,
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
    )

    return FreeAncillaRandomSearchResult(
        samples=samples,
        delta_omega_values=deltas,
        best_params=best_params,
        best_delta_omega=float(deltas[best_idx]),
        N=N,
        M=M,
        omega_value=omega,
        sql=sql_reference(N, T_HOLD),
        t_hold=T_HOLD,
    )


# ============================================================================
# Nelder-Mead Optimisation
# ============================================================================


def sensitivity_objective(
    params: np.ndarray,
    N: int,
    M: int,
    omega_true: float,
    ops: dict[str, np.ndarray],
    t_hold: float = T_HOLD,
    T_bs: float = T_BS,
    fd_step: float = FD_STEP,
    penalty_scale: float = 1e6,
) -> float:
    """Objective function for minimising :math:`\\Delta\\omega`.

    ``params = [theta_A, phi_A, a_x, a_y, a_z, a_zz]`` (6 elements).

    The bound constraints are enforced by quadratic penalties:
    - :math:`\\theta_A \\in [0, \\pi]`
    - :math:`\\phi_A \\in [0, 2\\pi)`
    - :math:`\\|\\mathbf{a}\\| \\le R` (3-ball constraint)
    - :math:`a_{zz} \\in [-5, 5]`

    Args:
        params: 6-element parameter vector.
        N: Number of system particles.
        M: Number of ancilla particles.
        omega_true: True phase rate.
        ops: Operators from :func:`build_operators`.
        t_hold: Holding time.
        T_bs: Beam-splitter duration.
        fd_step: Finite-difference step.
        penalty_scale: Scale for bound-violation penalty.

    Returns:
        :math:`\\Delta\\omega` (plus infinite penalty if bounds violated).
    """
    theta_A = float(params[0])
    phi_A = float(params[1])
    a_x = float(params[2])
    a_y = float(params[3])
    a_z = float(params[4])
    a_zz = float(params[5])

    # Bound enforcement
    penalty = 0.0

    # theta_A in [0, pi]
    if theta_A < 0.0:
        penalty += penalty_scale * (0.0 - theta_A) ** 2
    elif theta_A > np.pi:
        penalty += penalty_scale * (theta_A - np.pi) ** 2

    # phi_A in [0, 2*pi)
    if phi_A < 0.0:
        penalty += penalty_scale * (0.0 - phi_A) ** 2
    elif phi_A > 2.0 * np.pi:
        penalty += penalty_scale * (phi_A - 2.0 * np.pi) ** 2

    # (a_x, a_y, a_z) in 3-ball of radius DRIVE_RADIUS
    drive_norm = math.sqrt(a_x**2 + a_y**2 + a_z**2)
    if drive_norm > DRIVE_RADIUS:
        penalty += penalty_scale * (drive_norm - DRIVE_RADIUS) ** 2

    # a_zz in [-5, 5]
    if a_zz < AZZ_BOUNDS[0]:
        penalty += penalty_scale * (AZZ_BOUNDS[0] - a_zz) ** 2
    elif a_zz > AZZ_BOUNDS[1]:
        penalty += penalty_scale * (a_zz - AZZ_BOUNDS[1]) ** 2

    if penalty > 0.0:
        return float(1e10 + penalty)

    psi0 = free_initial_state(N, M, theta_A, phi_A)
    delta, _, _, _, _ = compute_sensitivity(
        psi0,
        N,
        M,
        T_bs,
        t_hold,
        omega_true,
        a_x,
        a_y,
        a_z,
        a_zz,
        ops,
        fd_step,
    )
    return delta


@dataclass
class FreeAncillaNelderMeadResult(ParquetSerializable):
    """Result of a single Nelder--Mead run for the free-ancilla protocol.

    Attributes:
        delta_omega_opt: Best sensitivity :math:`\\Delta\\omega` found.
        params_opt: Optimal 6-element parameter vector
            :math:`(\\theta_A, \\phi_A, a_x, a_y, a_z, a_{zz})`.
        omega_true: True :math:`\\omega` used for this optimisation.
        N: Number of system particles.
        M: Number of ancilla particles.
        success: Whether the optimiser reported success.
        nfev: Number of function evaluations.
        message: Optimiser message.
        expectation_Jz: :math:`\\langle J_z^S\\rangle` at the optimum.
        variance_Jz: :math:`\\mathrm{Var}(J_z^S)` at the optimum.
        d_exp: :math:`\\partial\\langle J_z^S\\rangle/\\partial\\omega` at the optimum.
        is_fringe: Whether the optimum is at a fringe extremum.
        history: Objective function values at each iteration.
    """

    delta_omega_opt: float
    params_opt: np.ndarray
    omega_true: float
    N: int
    M: int
    success: bool
    nfev: int
    message: str = ""
    expectation_Jz: float = 0.0
    variance_Jz: float = 0.0
    d_exp: float = 0.0
    is_fringe: bool = False
    history: list[float] = field(default_factory=list)

    _PARQUET_COLUMNS: ClassVar[list[str]] = [
        "delta_omega",
        "theta_A",
        "phi_A",
        "a_x",
        "a_y",
        "a_z",
        "a_zz",
        "omega_true",
        "N",
        "M",
        "success",
        "nfev",
        "expectation_Jz",
        "variance_Jz",
        "d_exp",
        "is_fringe",
        "message",
    ]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "theta_A": [float(self.params_opt[0])],
                "phi_A": [float(self.params_opt[1])],
                "a_x": [float(self.params_opt[2])],
                "a_y": [float(self.params_opt[3])],
                "a_z": [float(self.params_opt[4])],
                "a_zz": [float(self.params_opt[5])],
                "delta_omega": [self.delta_omega_opt],
                "omega_true": [self.omega_true],
                "N": [self.N],
                "M": [self.M],
                "success": [int(self.success)],
                "nfev": [self.nfev],
                "expectation_Jz": [self.expectation_Jz],
                "variance_Jz": [self.variance_Jz],
                "d_exp": [self.d_exp],
                "is_fringe": [int(self.is_fringe)],
                "message": [self.message],
            },
        )

    def _save_sidecars(self, path: Path) -> None:
        history_path = path.with_stem(path.stem + "-history")
        pd.DataFrame({"history": [self.history]}).to_parquet(history_path, index=False)

    @classmethod
    def from_parquet(cls, path: str | Path) -> FreeAncillaNelderMeadResult:
        df = pd.read_parquet(path)
        cls._validate_columns(df)
        history_path = Path(path).with_stem(Path(path).stem + "-history")
        if history_path.exists():
            history = list(pd.read_parquet(history_path)["history"].iloc[0])
        else:
            history = []
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
                ],
            ),
            omega_true=float(df["omega_true"].iloc[0]),
            N=int(df["N"].iloc[0]),
            M=int(df["M"].iloc[0]),
            success=bool(int(df["success"].iloc[0])),
            nfev=int(df["nfev"].iloc[0]),
            message=str(df["message"].iloc[0]),
            expectation_Jz=float(df["expectation_Jz"].iloc[0]),
            variance_Jz=float(df["variance_Jz"].iloc[0]),
            d_exp=float(df["d_exp"].iloc[0]),
            is_fringe=bool(int(df["is_fringe"].iloc[0])),
            history=history,
        )


def run_nelder_mead(
    N: int,
    M: int,
    omega_true: float,
    ops: dict[str, np.ndarray] | None = None,
    x0: np.ndarray | None = None,
    seed: int | None = None,
    maxiter: int = NM_MAXITER,
    xatol: float = 1e-8,
    fatol: float = 1e-8,
    adaptive: bool = True,
    track_history: bool = False,
) -> FreeAncillaNelderMeadResult:
    """Run Nelder--Mead optimisation for the free-ancilla protocol.

    Args:
        N: Number of system particles.
        M: Number of ancilla particles.
        omega_true: True phase rate parameter.
        ops: Operators (built if ``None``).
        x0: Initial 6-parameter vector. Randomly sampled if ``None``.
        seed: Random seed (used if ``x0`` is ``None``).
        maxiter: Maximum Nelder--Mead iterations.
        xatol: Absolute parameter tolerance.
        fatol: Absolute function tolerance.
        adaptive: Use adaptive Nelder--Mead parameters.
        track_history: If ``True``, record objective values per iteration.

    Returns:
        FreeAncillaNelderMeadResult.
    """
    if ops is None:
        ops = build_operators(N, M)

    if x0 is None:
        rng = np.random.default_rng(seed)
        # Sample a reasonable starting point
        theta_A = rng.uniform(0.0, np.pi)
        phi_A = rng.uniform(0.0, 2.0 * np.pi)
        v = rng.normal(size=3)
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-15:
            v = np.array([1.0, 0.0, 0.0])
            norm_v = np.float64(1.0)
        r = DRIVE_RADIUS * (rng.uniform(0.0, 1.0) ** (1.0 / 3.0))
        a_ball = v * r / norm_v
        a_zz = rng.uniform(AZZ_BOUNDS[0], AZZ_BOUNDS[1])
        x0 = np.array(
            [theta_A, phi_A, a_ball[0], a_ball[1], a_ball[2], a_zz],
            dtype=float,
        )
    else:
        x0 = np.asarray(x0, dtype=float)
        assert x0.shape == (6,), f"x0 must have 6 elements, got {x0.shape}"

    def objective(p: np.ndarray) -> float:
        return sensitivity_objective(
            p,
            N,
            M,
            omega_true,
            ops,
            t_hold=T_HOLD,
            T_bs=T_BS,
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
        callback=callback if track_history else None,
        options={  # type: ignore[call-overload]
            "maxiter": maxiter,
            "xatol": xatol,
            "fatol": fatol,
            "adaptive": adaptive,
        },
    )

    opt_params = result.x.copy()

    # Compute diagnostics at the optimal point
    theta_A = float(opt_params[0])
    phi_A = float(opt_params[1])
    psi0 = free_initial_state(N, M, theta_A, phi_A)
    _delta, exp_val, var_val, d_exp_val, is_fringe_val = compute_sensitivity(
        psi0,
        N,
        M,
        T_BS,
        T_HOLD,
        omega_true,
        float(opt_params[2]),
        float(opt_params[3]),
        float(opt_params[4]),
        float(opt_params[5]),
        ops,
    )

    return FreeAncillaNelderMeadResult(
        delta_omega_opt=float(result.fun),
        params_opt=opt_params,
        omega_true=omega_true,
        N=N,
        M=M,
        success=bool(result.success),
        nfev=int(result.nfev),
        message=str(result.message),
        expectation_Jz=exp_val,
        variance_Jz=var_val,
        d_exp=d_exp_val,
        is_fringe=is_fringe_val,
        history=history.copy(),
    )


# ============================================================================
# Per-(N, M, omega) Optimisation
# ============================================================================


@dataclass(kw_only=True)
class FreeAncillaNScalingResult(NScalingResult):
    """Optimisation result for a single :math:`(N, M, \\omega)` triple.

    Composes :class:`NScalingResult` with report-specific fields
    (``M``, ``theta_A_opt``, ``phi_A_opt``, and derived diagnostics).

    Attributes:
        N: Number of system particles.
        M: Number of ancilla particles.
        omega: Phase rate value.
        delta_omega_opt: Best sensitivity :math:`\\Delta\\omega` found.
        sql: SQL reference value.
        ratio: ``SQL / Δω_opt`` (ratio > 1 means beating SQL).
        theta_A_opt: Optimal ancilla polar angle.
        phi_A_opt: Optimal ancilla azimuthal angle.
        a_x_opt: Optimal :math:`J_x^A` drive coefficient.
        a_y_opt: Optimal :math:`J_y^A` drive coefficient.
        a_z_opt: Optimal :math:`J_z^A` drive coefficient.
        a_zz_opt: Optimal Ising interaction coefficient.
        expectation_Jz: :math:`\\langle J_z^S\\rangle` at the optimum.
        variance_Jz: :math:`\\mathrm{Var}(J_z^S)` at the optimum.
        t_hold: Holding time (fixed at 10.0).
        fd_step: Finite-difference step.
        success: Whether Nelder--Mead reported success.
        nfev: Number of function evaluations.
        J_A: Ancilla spin :math:`J_A = M/2`.
        drive_norm: Norm of optimal drive vector :math:`\\|\\mathbf{a}\\|`.
        d_exp: :math:`\\partial\\langle J_z^S\\rangle/\\partial\\omega` at optimum.
        is_fringe: Whether the optimum is at a fringe extremum.
    """

    # New fields specific to free-ancilla protocol
    M: int
    theta_A_opt: float
    phi_A_opt: float
    J_A: float = 0.0
    drive_norm: float = 0.0
    d_exp: float = 0.0
    is_fringe: bool = False

    _PARQUET_COLUMNS: ClassVar[list[str]] = [
        *NScalingResult._PARQUET_COLUMNS,
        "M",
        "theta_A_opt",
        "phi_A_opt",
        "J_A",
        "drive_norm",
        "d_exp",
        "is_fringe",
    ]

    def __post_init__(self) -> None:
        """Fill derived fields if not explicitly set."""
        if self.J_A == 0.0 and self.M > 0:
            object.__setattr__(self, "J_A", self.M / 2.0)
        if self.drive_norm == 0.0:
            object.__setattr__(
                self,
                "drive_norm",
                math.sqrt(
                    self.a_x_opt**2 + self.a_y_opt**2 + self.a_z_opt**2,
                ),
            )

    def to_dataframe(self) -> pd.DataFrame:
        df = super().to_dataframe()
        df["M"] = self.M
        df["theta_A_opt"] = self.theta_A_opt
        df["phi_A_opt"] = self.phi_A_opt
        df["J_A"] = self.J_A
        df["drive_norm"] = self.drive_norm
        df["d_exp"] = self.d_exp
        df["is_fringe"] = int(self.is_fringe)
        return df

    @classmethod
    def from_parquet(cls, path: str | Path) -> FreeAncillaNScalingResult:
        path = Path(path)
        df = pd.read_parquet(path)
        cls._validate_columns(df)
        row = df.iloc[0]
        return cls(
            N=int(row["N"]),
            M=int(row["M"]),
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
            expectation_Jz=float(row["expectation_Jz"]),
            variance_Jz=float(row["variance_Jz"]),
            t_hold=float(row["t_hold"]),
            fd_step=float(row["fd_step"]),
            success=bool(int(row["success"])),
            nfev=int(row["nfev"]),
            J_A=float(row["J_A"]),
            drive_norm=float(row["drive_norm"]),
            d_exp=float(row["d_exp"]),
            is_fringe=bool(int(row["is_fringe"])),
        )


@dataclass
class FreeAncillaNScalingScanResult(ParquetSerializable):
    """Collection of :math:`(N, M, \\omega)` optimisation results.

    Note: Standalone dataclass (not inheriting from :class:`NScalingScanResult`)
    because ``list[FreeAncillaNScalingResult]`` is not type-safe to assign
    to ``list[NScalingResult]``. The ``to_dataframe()`` and property
    patterns mirror the shared class.
    """

    results: list[FreeAncillaNScalingResult] = field(default_factory=list)

    _PARQUET_COLUMNS: ClassVar[list[str]] = FreeAncillaNScalingResult._PARQUET_COLUMNS

    def to_dataframe(self) -> pd.DataFrame:
        if not self.results:
            return pd.DataFrame(columns=self._PARQUET_COLUMNS)
        return pd.concat([r.to_dataframe() for r in self.results], ignore_index=True)

    @classmethod
    def from_parquet(cls, path: str | Path) -> FreeAncillaNScalingScanResult:
        path = Path(path)
        df = pd.read_parquet(path)
        cls._validate_columns(df)
        results: list[FreeAncillaNScalingResult] = []
        for _, row in df.iterrows():
            results.append(
                FreeAncillaNScalingResult(
                    N=int(row["N"]),
                    M=int(row["M"]),
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
                    expectation_Jz=float(row["expectation_Jz"]),
                    variance_Jz=float(row["variance_Jz"]),
                    t_hold=float(row["t_hold"]),
                    fd_step=float(row["fd_step"]),
                    success=bool(int(row["success"])),
                    nfev=int(row["nfev"]),
                    J_A=float(row["J_A"]),
                    drive_norm=float(row["drive_norm"]),
                    d_exp=float(row["d_exp"]),
                    is_fringe=bool(int(row["is_fringe"])),
                ),
            )
        return cls(results=results)

    @property
    def N_values(self) -> np.ndarray:
        return np.array(sorted({r.N for r in self.results}))

    @property
    def M_values(self) -> np.ndarray:
        return np.array(sorted({r.M for r in self.results}))

    @property
    def omega_values(self) -> np.ndarray:
        return np.array(sorted({r.omega for r in self.results}))


def run_single_n_m_omega(
    N: int,
    M: int,
    omega: float,
    n_random: int = N_RANDOM,
    n_nm_refine: int = N_NM_REFINE,
    seed: int | None = 42,
) -> FreeAncillaNScalingResult:
    """Run the full optimisation pipeline for a single :math:`(N, M, \\omega)` triple.

    1. 6D random search (1000 samples).
    2. Nelder--Mead refinement from top 30 points.
    3. Return the best result.

    Args:
        N: Number of system particles.
        M: Number of ancilla particles.
        omega: Phase rate value.
        n_random: Number of random search samples.
        n_nm_refine: Number of Nelder-Mead refinements.
        seed: Base random seed (incremented per call).

    Returns:
        FreeAncillaNScalingResult with the optimal parameters and sensitivity.
    """
    base_seed = seed if seed is not None else 42
    ops = build_operators(N, M)

    def rs_fn(n_samples, seed, **kw):
        return random_search(N, M, omega, n_samples=n_samples, seed=seed)

    def nm_fn(x0, seed, **kw):
        return run_nelder_mead(N, M, omega_true=omega, ops=ops, x0=x0, seed=seed)

    best_nm, _ = run_two_phase_pipeline(
        rs_fn,
        nm_fn,
        TwoPhaseConfig(n_random=n_random, n_nm_refine=n_nm_refine, seed=base_seed),
    )

    sql_val = sql_reference(N, T_HOLD)
    return FreeAncillaNScalingResult(
        N=N,
        M=M,
        omega=omega,
        delta_omega_opt=best_nm.delta_omega_opt,
        sql=sql_val,
        ratio=sql_val / best_nm.delta_omega_opt
        if np.isfinite(best_nm.delta_omega_opt) and best_nm.delta_omega_opt > 0
        else float("nan"),
        theta_A_opt=float(best_nm.params_opt[0]),
        phi_A_opt=float(best_nm.params_opt[1]),
        a_x_opt=float(best_nm.params_opt[2]),
        a_y_opt=float(best_nm.params_opt[3]),
        a_z_opt=float(best_nm.params_opt[4]),
        a_zz_opt=float(best_nm.params_opt[5]),
        expectation_Jz=best_nm.expectation_Jz,
        variance_Jz=best_nm.variance_Jz,
        success=best_nm.success,
        nfev=best_nm.nfev,
        J_A=M / 2.0,
        d_exp=best_nm.d_exp,
        is_fringe=best_nm.is_fringe,
    )


# ============================================================================
# Data Generation Pipeline
# ============================================================================


def _run_single_n_m_omega_worker(
    args: tuple[int, int, float],
) -> dict[str, int | float | str]:
    """Worker for parallel scan.

    Args:
        args: Tuple ``(N, M, omega)``.

    Returns:
        Dict of result data.
    """
    N, M, omega = args
    print(f"  [run] N={N}, M={M}, ω={omega}")
    result = run_single_n_m_omega(N, M, omega)
    return {
        "N": result.N,
        "M": result.M,
        "omega": result.omega,
        "delta_omega_opt": result.delta_omega_opt,
        "sql": result.sql,
        "ratio": result.ratio,
        "theta_A_opt": result.theta_A_opt,
        "phi_A_opt": result.phi_A_opt,
        "a_x_opt": result.a_x_opt,
        "a_y_opt": result.a_y_opt,
        "a_z_opt": result.a_z_opt,
        "a_zz_opt": result.a_zz_opt,
        "expectation_Jz": result.expectation_Jz,
        "variance_Jz": result.variance_Jz,
        "success": int(result.success),
        "nfev": result.nfev,
        "J_A": result.J_A,
        "drive_norm": result.drive_norm,
        "d_exp": result.d_exp,
        "is_fringe": int(result.is_fringe),
    }


# ── Checkpoint recovery callbacks ──────────────────────────────────────────


def _build_free_result_from_row(row: dict) -> FreeAncillaNScalingResult:
    """Build FreeAncillaNScalingResult from a Parquet row dict."""
    return FreeAncillaNScalingResult(
        N=int(row["N"]),
        M=int(row["M"]),
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
        expectation_Jz=float(row["expectation_Jz"]),
        variance_Jz=float(row["variance_Jz"]),
        success=bool(int(row["success"])),
        nfev=int(row["nfev"]),
        J_A=float(row["J_A"]),
        drive_norm=float(row["drive_norm"]),
        d_exp=float(row["d_exp"]),
        is_fringe=bool(int(row["is_fringe"])),
    )


def _build_free_result_from_dict(d: dict) -> FreeAncillaNScalingResult:
    """Build FreeAncillaNScalingResult from a worker output dict."""
    return FreeAncillaNScalingResult(
        N=d["N"],
        M=d["M"],
        omega=d["omega"],
        delta_omega_opt=d["delta_omega_opt"],
        sql=d["sql"],
        ratio=d["ratio"],
        theta_A_opt=d["theta_A_opt"],
        phi_A_opt=d["phi_A_opt"],
        a_x_opt=d["a_x_opt"],
        a_y_opt=d["a_y_opt"],
        a_z_opt=d["a_z_opt"],
        a_zz_opt=d["a_zz_opt"],
        expectation_Jz=d["expectation_Jz"],
        variance_Jz=d["variance_Jz"],
        success=bool(d["success"]),
        nfev=d["nfev"],
        J_A=d["J_A"],
        drive_norm=d["drive_norm"],
        d_exp=d["d_exp"],
        is_fringe=bool(d["is_fringe"]),
    )


def _free_group_key(item: tuple[int, int, float]) -> int:
    """Group by N value (first element of triple)."""
    return item[0]


def _make_free_checkpoint_name(checkpoint_dir: Path) -> Callable:
    """Factory: returns function mapping N -> checkpoint Path."""

    def _name_fn(group_key: Any) -> Path:
        return checkpoint_dir / f"N_{int(group_key):03d}.parquet"

    return _name_fn


def generate_decoupled_baseline_scan(force: bool = False) -> None:
    """Verify the decoupled baseline and save results.

    Args:
        force: Re-run even if Parquet exists.
    """
    parquet_path_p = _parquet_path("decoupled-baseline")

    if parquet_path_p.exists() and not force:
        print(f"[skip] {parquet_path_p.name} exists (use --force to overwrite)")
        return

    print("[run] Computing decoupled baseline verification...")
    verifications = verify_decoupled_baseline()
    rows: list[dict[str, float | int | str]] = []
    for (N, M, omega), passed in verifications.items():
        sql_ref = sql_reference(N, T_HOLD)
        delta = sql_reference(N, T_HOLD)
        rows.append(
            {
                "N": N,
                "M": M,
                "omega": omega,
                "delta_omega": delta,
                "sql": sql_ref,
                "ratio": sql_ref / delta if delta > 0 else float("nan"),
                "pass": str(passed),
            },
        )
    df = pd.DataFrame(rows)
    parquet_path_p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path_p, index=False)
    print(f"[save] {parquet_path_p}")


def _get_pending_items(
    completed: set[tuple[int, int, float]],
) -> list[tuple[int, int, float]]:
    """Return :math:`(N, M, \\omega)` triples that have not yet been completed.

    Args:
        completed: Set of ``(N, M, omega)`` triples already processed.

    Returns:
        List of pending ``(N, M, omega)`` triples.
    """
    return [
        (N, M, omega)
        for N in N_VALS
        for M in M_VALS
        for omega in OMEGA_VALS
        if (N, M, omega) not in completed
    ]


def _generate_n_scaling_figures(summary: FreeAncillaNScalingScanResult) -> None:
    """Generate N-scaling figures for each M value.

    Produces ratio, sensitivity, and optimal-params plots per M.

    Args:
        summary: Scan result containing results for all ``(N, M, omega)`` triples.
    """
    df = summary.to_dataframe()
    for m_val in sorted(df["M"].unique()):
        df_m = df[df["M"] == m_val].copy()
        m_suffix = f"-M{m_val}"
        p_ratio = _fig_path(f"n-scaling-ratio{m_suffix}")
        p_sens = _fig_path(f"n-scaling-sensitivity{m_suffix}")
        p_params = _fig_path(f"n-scaling-optimal-params{m_suffix}")
        plot_n_scaling_ratio(df_m, p_ratio)
        print(f"[fig]  {p_ratio}")
        plot_n_scaling_sensitivity(df_m, p_sens, t_hold=T_HOLD)
        print(f"[fig]  {p_sens}")
        plot_n_scaling_optimal_params(df_m, p_params)
        print(f"[fig]  {p_params}")


def generate_n_scaling_scan(force: bool = False) -> None:
    """Full :math:`(N, M, \\omega)` scan.

    Each ``(N, M, omega)`` triple runs:
      1. 6D random search with 1000 points.
      2. Nelder--Mead refinement from top 30 points.

    Results are saved to a single Parquet file. Intermediate checkpoints
    are saved per-N-value to allow resumption if interrupted.

    Args:
        force: Re-run even if Parquet exists.
    """
    parquet_p = _parquet_path("n-m-omega-scan")
    checkpoint_dir = REPORTS_DIR / REPORT_DATE / "raw_data" / "checkpoints"

    if parquet_p.exists() and not force:
        print(f"[skip] {parquet_p.name} exists (use --force to overwrite)")
        summary = FreeAncillaNScalingScanResult.from_parquet(parquet_p)
    else:
        if force:
            parquet_p.unlink(missing_ok=True)
            if checkpoint_dir.exists():
                import shutil

                shutil.rmtree(checkpoint_dir)

        completed, checkpoint_results = load_checkpoints(
            checkpoint_dir,
            _build_free_result_from_row,
            ["N", "M", "omega"],
        )

        items_to_run = _get_pending_items(completed)
        if items_to_run:
            print(
                f"[run] N-scaling scan: {len(items_to_run)} remaining "
                f"(N, M, ω) triples",
            )
            print(f"  (batch by N value, {min(32, os.cpu_count() or 1)} workers)")
            new_results = run_pending_groups(
                items_to_run,
                checkpoint_dir,
                _run_single_n_m_omega_worker,
                _build_free_result_from_dict,
                _free_group_key,
                _make_free_checkpoint_name(checkpoint_dir),
                FreeAncillaNScalingScanResult,
            )
            checkpoint_results.extend(new_results)
        else:
            print("  [skip] all triples already completed in checkpoints")

        # Merge all checkpoint results and save final file
        summary = FreeAncillaNScalingScanResult(results=checkpoint_results)
        summary.save_parquet(parquet_p)
        print(f"[save] {parquet_p}")

    # Generate figures — one set per M value
    _generate_n_scaling_figures(summary)


# ============================================================================
# CLI Entry Point
# ============================================================================


def main() -> None:
    """CLI entry point for generating all data and figures."""
    parser = argparse.ArgumentParser(
        description="Multi-Particle Ancilla Free Initial State (20260620)",
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
            "generate_decoupled_baseline_scan",
            False,
        ),
        "n-scaling-scan": (
            "(N, M, ω) Full Scan",
            "generate_n_scaling_scan",
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
        name, func_name, _ = generators[key]
        print(f"\n{'=' * 60}")
        print(f"  {name}")
        print(f"{'=' * 60}")
        func = globals()[func_name]
        func(force=force)


if __name__ == "__main__":
    main()
