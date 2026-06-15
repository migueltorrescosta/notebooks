"""
Local module for the 2026-05-18 Weighted Joint Measurement N,M-Generalization report.

Contains all code exclusive to this report:
- Core physics simulation (collective spin operators, CSS states, beam-splitter
  unitaries, circuit evolution, weighted measurement and sensitivity)
- Golden-section sub-optimisation for the weight angle psi
- L-BFGS-B circuit parameter optimisation with FD or AD gradients
- N-scaling and M-scaling analysis with bootstrap confidence intervals
- Alpha-scan with weight re-optimisation
- Validation helpers
- Exclusive plot functions (N-scaling, M-scaling, alpha-scan)

Usage:
    uv run python reports/20260518/local.py --force

This module is **not** importable as ``reports.20260518.local`` (the directory
name contains hyphens).  Instead, importers add the report directory to
``sys.path`` and do ``import local``.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast, overload

# Force non-interactive matplotlib backend before any plotting imports.
if "MPLBACKEND" not in os.environ:
    os.environ["MPLBACKEND"] = "Agg"
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "1"

if TYPE_CHECKING:
    from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.optimize import minimize

from src.analysis.ancilla_drive_metrology import (
    Drive2DSliceResult,
    DriveDecoupledBaselineResult,
    DriveOmegaScanResult,
    DriveRandomSearchResult,
    compute_drive_decoupled_baseline,
    drive_2d_slice,
    drive_random_search,
    run_drive_omega_scan,
)
from src.physics.dicke_basis import jx_operator, jy_operator, jz_operator
from src.physics.multi_mzi import single_bs_unitary
from src.utils.enums import OperatorBasis
from src.visualization.ancilla_drive_plots import (
    plot_drive_2d_slice_heatmap,
    plot_drive_decoupled_baseline,
    plot_drive_omega_scan,
    plot_drive_optimal_params,
    plot_drive_random_search_histogram,
)

sns.set_theme(style="whitegrid")
# ============================================================================
# Utility: Golden-Section Search for Weight Angle psi
# ============================================================================

GOLDEN_RATIO: float = (np.sqrt(5.0) - 1.0) / 2.0  # ~0.618


def golden_section_minimize(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-8,
    max_iter: int = 200,
) -> tuple[float, float]:
    """Minimize a unimodal 1D function via golden-section search.

    Args:
        f: Function to minimize (takes a scalar float, returns a scalar float).
        a: Left bound of the search interval.
        b: Right bound of the search interval.
        tol: Absolute tolerance on the parameter (default 1e-8).
        max_iter: Maximum iterations (default 200).

    Returns:
        Tuple (x_min, f_min) where x_min is the argmin and f_min = f(x_min).

    Raises:
        RuntimeError: If max_iter is exceeded without convergence.
    """
    # Ensure a < b
    if a > b:
        a, b = b, a

    c = b - GOLDEN_RATIO * (b - a)
    d = a + GOLDEN_RATIO * (b - a)
    fc = f(c)
    fd = f(d)

    for _ in range(max_iter):
        if abs(b - a) < tol:
            x_min = (a + b) / 2.0
            return x_min, f(x_min)

        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - GOLDEN_RATIO * (b - a)
            fc = f(c)
        else:
            a = c
            c = d
            fc = fd
            d = a + GOLDEN_RATIO * (b - a)
            fd = f(d)

    raise RuntimeError(
        f"Golden-section search did not converge in {max_iter} iterations"
    )


# ============================================================================
# Operator Construction (NumPy → Torch)
# ============================================================================


def build_collective_operators(N: int, M: int) -> dict[str, np.ndarray]:
    """Build collective spin operators for the N+M system in numpy.

    Constructs J_z, J_x, J_y for system (spin J_S = N/2) and ancilla
    (spin J_A = M/2), embedded into the full (N+1)(M+1)-dimensional space
    via Kronecker products.

    Args:
        N: Number of system particles.
        M: Number of ancilla particles.

    Returns:
        Dict with keys:
            'Jz_S', 'Jx_S', 'Jy_S', 'Jz_A', 'Jx_A', 'Jy_A'
        each an (N+1)(M+1) × (N+1)(M+1) numpy array (complex).
    """
    d_S = N + 1
    d_A = M + 1

    Jz_S_np = jz_operator(N, basis=OperatorBasis.DICKE)
    Jx_S_np = jx_operator(N, basis=OperatorBasis.DICKE)
    Jy_S_np = jy_operator(N, basis=OperatorBasis.DICKE)
    Jz_A_np = jz_operator(M, basis=OperatorBasis.DICKE)
    Jx_A_np = jx_operator(M, basis=OperatorBasis.DICKE)
    Jy_A_np = jy_operator(M, basis=OperatorBasis.DICKE)

    I_S = np.eye(d_S, dtype=complex)
    I_A = np.eye(d_A, dtype=complex)

    return {
        "Jz_S": np.kron(Jz_S_np, I_A).astype(complex),
        "Jx_S": np.kron(Jx_S_np, I_A).astype(complex),
        "Jy_S": np.kron(Jy_S_np, I_A).astype(complex),
        "Jz_A": np.kron(I_S, Jz_A_np).astype(complex),
        "Jx_A": np.kron(I_S, Jx_A_np).astype(complex),
        "Jy_A": np.kron(I_S, Jy_A_np).astype(complex),
    }


def operators_to_torch(
    ops_np: dict[str, np.ndarray],
) -> dict[str, torch.Tensor]:
    """Convert numpy operators to torch tensors (complex128).

    Args:
        ops_np: Dictionary of numpy operators.

    Returns:
        Dictionary of torch tensors (complex128, device='cpu').
    """
    return {
        key: torch.tensor(val, dtype=torch.complex128) for key, val in ops_np.items()
    }


def build_interaction_hamiltonian_np(
    alpha: tuple[float, float, float, float],
    ops_np: dict[str, np.ndarray],
) -> np.ndarray:
    """Build H_int = sum_{ij} alpha_{ij} J_i^S ⊗ J_j^A in numpy.

    Args:
        alpha: (alpha_xx, alpha_xz, alpha_zx, alpha_zz).
        ops_np: Operators from build_collective_operators().

    Returns:
        Full-space Hermitian interaction matrix (numpy, complex).
    """
    a_xx, a_xz, a_zx, a_zz = alpha
    d = ops_np["Jz_S"].shape[0]
    H = np.zeros((d, d), dtype=complex)
    if a_xx != 0.0:
        H += a_xx * (ops_np["Jx_S"] @ ops_np["Jx_A"])
    if a_xz != 0.0:
        H += a_xz * (ops_np["Jx_S"] @ ops_np["Jz_A"])
    if a_zx != 0.0:
        H += a_zx * (ops_np["Jz_S"] @ ops_np["Jx_A"])
    if a_zz != 0.0:
        H += a_zz * (ops_np["Jz_S"] @ ops_np["Jz_A"])
    return 0.5 * (H + H.conj().T)


# ============================================================================
# CSS State Preparation (NumPy)
# ============================================================================


def css_state_np(J: float, theta: float) -> np.ndarray:
    """Create a coherent spin state |theta, 0> = exp(-i theta J_y) |J, -J>.

    The azimuthal angle phi is fixed to 0 (gauge freedom). The ground state
    |J, -J> is the last basis vector in the Dicke ordering m = J, J-1, ..., -J.

    Args:
        J: Total spin (N/2 or M/2).
        theta: Polar angle in [0, pi].

    Returns:
        Normalised (2J+1)-dimensional complex vector.
    """
    dim = int(2 * J + 1)
    if dim == 1:
        return np.array([1.0], dtype=complex)

    # |J, -J> is the last basis vector (index dim-1)
    ground = np.zeros(dim, dtype=complex)
    ground[-1] = 1.0

    if abs(theta) < 1e-15:
        return ground

    # Build J_y operator
    N = int(2 * J)
    Jy = jy_operator(N, basis=OperatorBasis.DICKE)

    # Rotate via matrix exponential
    from scipy.linalg import expm

    U = expm(-1j * float(theta) * Jy)
    state = U @ ground

    # Normalise
    norm = np.linalg.norm(state)
    assert abs(norm - 1.0) < 1e-12, f"CSS normalisation failed: norm={norm}"
    return state


def product_css_state_np(
    theta_S: float,
    theta_A: float,
    N: int,
    M: int,
) -> np.ndarray:
    """Create a product CSS state |theta_S, 0> ⊗ |theta_A, 0>.

    Args:
        theta_S: System polar angle.
        theta_A: Ancilla polar angle.
        N: System particle number.
        M: Ancilla particle number.

    Returns:
        Normalised (N+1)(M+1)-dimensional complex vector.
    """
    psi_S = css_state_np(N / 2.0, theta_S)
    psi_A = css_state_np(M / 2.0, theta_A)
    state = np.kron(psi_S, psi_A)
    assert abs(np.linalg.norm(state) - 1.0) < 1e-12, (
        "Product state normalisation failed"
    )
    return state


def product_css_state_torch(
    theta_S: torch.Tensor,
    theta_A: torch.Tensor,
    N: int,
    M: int,
    ops_torch: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Product CSS state in torch with AD support.

    Builds |psi> = exp(-i*theta_S*Jy_S) * exp(-i*theta_A*Jy_A) * |J_S,-J_S> ⊗ |J_A,-J_A>
    using torch.linalg.matrix_exp for differentiability.

    Args:
        theta_S: System polar angle (torch scalar, requires_grad).
        theta_A: Ancilla polar angle (torch scalar, requires_grad).
        N: System particle number.
        M: Ancilla particle number.
        ops_torch: Torch operators.

    Returns:
        Normalised (N+1)(M+1)-dim torch tensor (complex128, differentiable).
    """
    d = (N + 1) * (M + 1)
    ground = torch.zeros(d, dtype=torch.complex128)
    ground[-1] = 1.0  # |J_S,-J_S> ⊗ |J_A,-J_A> is the last basis vector

    psi = torch.linalg.matrix_exp(-1j * theta_S * ops_torch["Jy_S"]) @ ground
    return torch.linalg.matrix_exp(-1j * theta_A * ops_torch["Jy_A"]) @ psi


# ============================================================================
# Beam-Splitter Unitaries (NumPy)
# ============================================================================


# bs_unitary_np replaced by single_bs_unitary from src.physics.multi_mzi


def full_bs_unitary_np(N: int, M: int, T_BS: float) -> np.ndarray:
    """Full beam-splitter unitary: U_BS(T_BS; N) ⊗ U_BS(T_BS; M).

    Args:
        N: System particle number.
        M: Ancilla particle number.
        T_BS: Beam-splitter duration.

    Returns:
        (N+1)(M+1) × (N+1)(M+1) unitary matrix.
    """
    U_S = single_bs_unitary(N, T_BS)
    U_A = single_bs_unitary(M, T_BS)
    return np.kron(U_S, U_A)


# ============================================================================
# Full Circuit (NumPy Forward Pass)
# ============================================================================


def evolve_full_np(
    psi0: np.ndarray,
    T_BS1: float,
    T_BS2: float,
    t_hold: float,
    omega_true: float,
    alpha: tuple[float, float, float, float],
    ops_np: dict[str, np.ndarray],
    N: int,
    M: int,
) -> np.ndarray:
    """Run the full MZI circuit in numpy.

    |psi_final> = U_BS(T_BS2) · U_hold(t_hold, omega_true, alpha) · U_BS(T_BS1) · |psi0>

    Args:
        psi0: Initial state vector of dimension (N+1)(M+1).
        T_BS1: First beam-splitter duration.
        T_BS2: Second beam-splitter duration.
        t_hold: Holding time.
        omega_true: True phase rate parameter.
        alpha: (alpha_xx, alpha_xz, alpha_zx, alpha_zz).
        ops_np: Operators from build_collective_operators().
        N: System particle number.
        M: Ancilla particle number.

    Returns:
        Final state vector.
    """
    assert np.isclose(np.linalg.norm(psi0), 1.0), "Initial state must be normalised"

    # BS1
    U_BS1 = full_bs_unitary_np(N, M, T_BS1)
    psi = U_BS1 @ psi0

    # Hold
    H_int = build_interaction_hamiltonian_np(alpha, ops_np)
    H_hold = omega_true * ops_np["Jz_S"] + H_int
    H_hold = 0.5 * (H_hold + H_hold.conj().T)

    from scipy.linalg import expm as scipy_expm

    U_hold = scipy_expm(-1j * t_hold * H_hold)
    d = ops_np["Jz_S"].shape[0]
    assert np.allclose(U_hold @ U_hold.conj().T, np.eye(d), atol=1e-12), (
        "Hold unitary not unitary"
    )
    psi = U_hold @ psi

    # BS2
    U_BS2 = full_bs_unitary_np(N, M, T_BS2)
    psi = U_BS2 @ psi

    assert np.isclose(np.linalg.norm(psi), 1.0), "Final state must be normalised"
    return psi


# ============================================================================
# Torch Circuit (AD-Enabled)
# ============================================================================


def _state_to_torch(state_np: np.ndarray) -> torch.Tensor:
    """Convert a numpy state vector to a torch tensor.

    Args:
        state_np: Numpy array (complex).

    Returns:
        Torch tensor (complex128).
    """
    return torch.tensor(state_np, dtype=torch.complex128)


def build_hold_hamiltonian_torch(
    omega_true: float | torch.Tensor,
    alpha: tuple[
        float | torch.Tensor,
        float | torch.Tensor,
        float | torch.Tensor,
        float | torch.Tensor,
    ],
    ops_torch: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Build H_hold = omega * J_z^S + H_int as a torch tensor.

    Supports both float and torch.Tensor alpha for AD through
    interaction coefficients.

    Args:
        omega_true: True phase rate parameter.
        alpha: (alpha_xx, alpha_xz, alpha_zx, alpha_zz).
        ops_torch: Torch operators.

    Returns:
        Torch tensor (complex128) of the hold Hamiltonian.
    """
    a_xx, a_xz, a_zx, a_zz = alpha

    # Detect AD path: alpha elements are torch tensors
    use_ad = any(isinstance(a, torch.Tensor) for a in (a_xx, a_xz, a_zx, a_zz))

    if use_ad:
        # AD path: keep all terms in the graph (no conditional branches)
        H_int = (
            a_xx * (ops_torch["Jx_S"] @ ops_torch["Jx_A"])
            + a_xz * (ops_torch["Jx_S"] @ ops_torch["Jz_A"])
            + a_zx * (ops_torch["Jz_S"] @ ops_torch["Jx_A"])
            + a_zz * (ops_torch["Jz_S"] @ ops_torch["Jz_A"])
        )
    else:
        # Original path: skip zero terms for efficiency
        H_int = torch.zeros_like(ops_torch["Jz_S"])
        if a_xx != 0.0:
            H_int += a_xx * (ops_torch["Jx_S"] @ ops_torch["Jx_A"])
        if a_xz != 0.0:
            H_int += a_xz * (ops_torch["Jx_S"] @ ops_torch["Jz_A"])
        if a_zx != 0.0:
            H_int += a_zx * (ops_torch["Jz_S"] @ ops_torch["Jx_A"])
        if a_zz != 0.0:
            H_int += a_zz * (ops_torch["Jz_S"] @ ops_torch["Jz_A"])

    if isinstance(omega_true, torch.Tensor):
        H = omega_true * ops_torch["Jz_S"] + H_int
    else:
        H = float(omega_true) * ops_torch["Jz_S"] + H_int
    return 0.5 * (H + H.conj().T)


def evolve_full_torch(
    psi0: torch.Tensor,
    T_BS1: torch.Tensor,
    T_BS2: torch.Tensor,
    t_hold: torch.Tensor,
    omega_true: torch.Tensor,
    alpha: tuple[
        float | torch.Tensor,
        float | torch.Tensor,
        float | torch.Tensor,
        float | torch.Tensor,
    ],
    ops_torch: dict[str, torch.Tensor],
    N: int,
    M: int,
) -> torch.Tensor:
    """Run the full MZI circuit in torch with AD support.

    When beam-splitter durations require grad, uses torch-based
    matrix exponentials for the BS unitaries. When they don't,
    falls back to the efficient numpy path. The hold unitary is
    always computed via torch.linalg.matrix_exp. Alpha elements
    may be floats or torch tensors for AD through interaction
    coefficients.

    Args:
        psi0: Initial state vector (torch tensor, complex128).
        T_BS1: First beam-splitter duration (torch scalar).
        T_BS2: Second beam-splitter duration (torch scalar).
        t_hold: Holding time (torch scalar).
        omega_true: True phase rate parameter (torch scalar).
        alpha: (alpha_xx, alpha_xz, alpha_zx, alpha_zz).
        ops_torch: Torch operators.
        N: System particle number.
        M: Ancilla particle number.

    Returns:
        Final state vector (torch tensor, complex128).
    """
    # BS1: use torch-based expm if T_BS1 requires grad
    if T_BS1.requires_grad:
        Jx_total = ops_torch["Jx_S"] + ops_torch["Jx_A"]
        U_BS1 = torch.linalg.matrix_exp(-1j * T_BS1 * Jx_total)
    else:
        U_BS1_np = full_bs_unitary_np(N, M, float(T_BS1.detach()))
        U_BS1 = torch.tensor(U_BS1_np, dtype=torch.complex128)
    psi = U_BS1 @ psi0

    # Hold unitary via torch.linalg.matrix_exp
    omega_scalar = (
        float(omega_true.detach())
        if isinstance(omega_true, torch.Tensor)
        else omega_true
    )
    H_hold = build_hold_hamiltonian_torch(omega_scalar, alpha, ops_torch)
    U_hold = torch.linalg.matrix_exp(-1j * t_hold * H_hold)
    psi = U_hold @ psi

    # BS2: use torch-based expm if T_BS2 requires grad
    if T_BS2.requires_grad:
        Jx_total = ops_torch["Jx_S"] + ops_torch["Jx_A"]
        U_BS2 = torch.linalg.matrix_exp(-1j * T_BS2 * Jx_total)
    else:
        U_BS2_np = full_bs_unitary_np(N, M, float(T_BS2.detach()))
        U_BS2 = torch.tensor(U_BS2_np, dtype=torch.complex128)
    return U_BS2 @ psi


# ============================================================================
# Weighted Measurement and Sensitivity
# ============================================================================


def build_weighted_operator_np(
    a: float,
    b: float,
    ops_np: dict[str, np.ndarray],
) -> np.ndarray:
    """Build M(a,b) = a*J_z^S + b*J_z^A with a^2 + b^2 = 1.

    Args:
        a: System weight coefficient.
        b: Ancilla weight coefficient (a^2 + b^2 = 1).
        ops_np: Operators from build_collective_operators().

    Returns:
        Weighted measurement operator matrix.
    """
    assert abs(a**2 + b**2 - 1.0) < 1e-14, (
        f"Weight normalisation failed: a^2+b^2={a**2 + b**2}"
    )
    return a * ops_np["Jz_S"] + b * ops_np["Jz_A"]


def expectation_and_variance(
    psi: np.ndarray,
    operator: np.ndarray,
) -> tuple[float, float]:
    """Compute ⟨psi|O|psi⟩ and Var(O) for a pure state.

    Args:
        psi: Normalised state vector.
        operator: Hermitian operator matrix.

    Returns:
        Tuple (expectation, variance).
    """
    exp_val = float(np.real(psi.conj() @ operator @ psi))
    op_sq = operator @ operator
    exp_sq = float(np.real(psi.conj() @ op_sq @ psi))
    raw_var = exp_sq - exp_val**2
    assert raw_var >= -1e-12, f"Unphysical negative variance: Var = {raw_var:.2e}"
    var_val = max(0.0, raw_var)
    return exp_val, var_val


def compute_covariance_sa(
    psi: np.ndarray,
    ops_np: dict[str, np.ndarray],
) -> float:
    """Compute Cov(J_z^S, J_z^A) = ⟨J_z^S J_z^A⟩ - ⟨J_z^S⟩⟨J_z^A⟩.

    Args:
        psi: Normalised state vector.
        ops_np: Operators from build_collective_operators().

    Returns:
        Covariance value.
    """
    Jz_S = ops_np["Jz_S"]
    Jz_A = ops_np["Jz_A"]
    exp_S = float(np.real(psi.conj() @ Jz_S @ psi))
    exp_A = float(np.real(psi.conj() @ Jz_A @ psi))
    exp_SA = float(np.real(psi.conj() @ (Jz_S @ Jz_A) @ psi))
    cov = exp_SA - exp_S * exp_A
    assert np.isfinite(cov), f"Covariance is not finite: {cov}"
    return float(cov)


def compute_six_moments(
    psi: np.ndarray,
    ops_np: dict[str, np.ndarray],
) -> tuple[float, float, float, float, float, float]:
    """Compute the six moments needed for weighted sensitivity.

    Returns:
        Tuple (exp_S, exp_A, var_S, var_A, cov_SA, norm_check).
        norm_check = ⟨psi|psi⟩ (should be 1.0).
    """
    Jz_S = ops_np["Jz_S"]
    Jz_A = ops_np["Jz_A"]

    exp_S, var_S = expectation_and_variance(psi, Jz_S)
    exp_A, var_A = expectation_and_variance(psi, Jz_A)
    cov_SA = compute_covariance_sa(psi, ops_np)
    norm = float(np.real(psi.conj() @ psi))

    return exp_S, exp_A, var_S, var_A, cov_SA, norm


def _compute_six_moments_torch(
    psi: torch.Tensor,
    ops_torch: dict[str, torch.Tensor],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Compute six moments from a torch state (differentiable).

    Uses torch operations throughout so that gradients propagate
    through the state vector to the circuit parameters.

    Args:
        psi: State vector (torch tensor, complex128).
        ops_torch: Torch operators.

    Returns:
        Tuple (exp_S, exp_A, var_S, var_A, cov_SA, norm) as torch scalars.
    """
    Jz_S = ops_torch["Jz_S"]
    Jz_A = ops_torch["Jz_A"]

    exp_S = torch.real(psi.conj() @ Jz_S @ psi)
    Jz_S_sq = Jz_S @ Jz_S
    exp_S_sq = torch.real(psi.conj() @ Jz_S_sq @ psi)
    var_S = exp_S_sq - exp_S**2

    exp_A = torch.real(psi.conj() @ Jz_A @ psi)
    Jz_A_sq = Jz_A @ Jz_A
    exp_A_sq = torch.real(psi.conj() @ Jz_A_sq @ psi)
    var_A = exp_A_sq - exp_A**2

    Jz_S_Jz_A = Jz_S @ Jz_A
    exp_SA = torch.real(psi.conj() @ Jz_S_Jz_A @ psi)
    cov_SA = exp_SA - exp_S * exp_A

    norm = torch.real(psi.conj() @ psi)

    return exp_S, exp_A, var_S, var_A, cov_SA, norm


def compute_weighted_delta_omega(
    a: float,
    b: float,
    moments: tuple[float, float, float, float, float, float],
    d_moments: tuple[float, float, float, float, float, float] | None = None,
    fd_step: float = 1e-6,
    N: int = 1,
    M: int = 1,
) -> float:
    """Compute Δθ = sqrt(Var(M)) / |∂⟨M⟩/∂θ| for weighted M = a*J_z^S + b*J_z^A.

    If d_moments is provided, uses it for the derivative. Otherwise, the
    caller must handle derivative computation separately.

    Args:
        a: System weight (a^2 + b^2 = 1).
        b: Ancilla weight.
        moments: Six moments (exp_S, exp_A, var_S, var_A, cov_SA, norm)
            at the evaluation point.
        d_moments: Six derivatives of moments with respect to omega.
            If None, derivative is computed separately.
        fd_step: Finite-difference step size (default 1e-6).
        N: System particle number (used for threshold normalisation).
        M: Ancilla particle number (used for threshold normalisation).

    Returns:
        Δθ (positive float), or inf if derivative is zero.
    """
    _exp_S, _exp_A, var_S, var_A, cov_SA, _norm = moments

    # Mean of M (computed for clarity; not used in Δθ formula)
    #   E[M] = a * E[Jz_S] + b * E[Jz_A]

    # Variance of M: a^2 Var(Jz_S) + b^2 Var(Jz_A) + 2ab Cov(Jz_S, Jz_A)
    var_M = a**2 * var_S + b**2 * var_A + 2.0 * a * b * cov_SA
    assert var_M >= -1e-12, f"Negative Var(M): {var_M}"
    var_M = max(0.0, var_M)

    # Derivative of expectation
    if d_moments is not None:
        d_exp_S, d_exp_A, _, _, _, _ = d_moments
        d_exp_M = a * d_exp_S + b * d_exp_A
    else:
        raise ValueError("d_moments must be provided")

    # --- Fringe-extremum / numerical-noise detection ---
    # The derivative can vanish at fringe extrema, or be dominated by
    # numerical noise (~1e-10 for finite differences through expm).
    # A hard threshold of 1e-8 ensures we don't treat numerical noise
    # as a signal, while remaining well below any physical derivative
    # (which scales as ~N * t_hold, typically >= 0.1 for our test cases).
    if abs(d_exp_M) < 1e-8:
        return float("inf")

    # Additionally, if the variance is numerically zero (< machine
    # precision for the operator's scale), the measurement has no
    # information content at this operating point. This catches
    # pathological 0/0 cases (e.g., a=0 with ancilla in a J_z eigenstate).
    if var_M < 1e-14:
        return float("inf")

    return float(np.sqrt(var_M) / abs(d_exp_M))


def delta_omega_from_psi(
    psi: float,
    moments: tuple[float, float, float, float, float, float],
    d_moments: tuple[float, float, float, float, float, float],
) -> float:
    """Compute Δθ for a given weight angle ψ = arctan(b/a).

    Args:
        psi: Weight angle such that (a,b) = (cos ψ, sin ψ).
        moments: Six moments at the evaluation point.
        d_moments: Six derivative moments.

    Returns:
        Δθ value (positive float).
    """
    a = np.cos(psi)
    b = np.sin(psi)
    return compute_weighted_delta_omega(a, b, moments, d_moments)


def optimize_weight_psi(
    moments: tuple[float, float, float, float, float, float],
    d_moments: tuple[float, float, float, float, float, float],
) -> tuple[float, float, float, float]:
    """Optimise the weight angle ψ via golden-section search on safe intervals.

    The objective Δθ(ψ) has singularities at ψ = π/2 and 3π/2 (where a = 0,
    so the derivative |d⟨M⟩/dθ| vanishes for the decoupled ancilla). To avoid
    these, we split [0, 2π) into three intervals that exclude neighbourhoods
    around the singularities, run golden-section search on each, and pick the
    best result. This gives a continuous ψ^* for the envelope theorem.

    Args:
        moments: Six moments at the evaluation point.
        d_moments: Six derivative moments.

    Returns:
        Tuple (psi_opt, delta_omega_opt, a_opt, b_opt) where
        (a_opt, b_opt) = (cos(psi_opt), sin(psi_opt)) are the optimal weights.
    """
    f = partial(delta_omega_from_psi, moments=moments, d_moments=d_moments)

    # Singularity locations
    sing1 = np.pi / 2.0
    sing2 = 3.0 * np.pi / 2.0
    margin = 0.05  # rad — safe margin around singularities

    # Build safe intervals
    safe_intervals: list[tuple[float, float]] = []
    for lo, hi in [
        (0.0, sing1 - margin),
        (sing1 + margin, sing2 - margin),
        (sing2 + margin, 2.0 * np.pi),
    ]:
        if hi - lo > 1e-3:
            safe_intervals.append((lo, hi))

    # Run golden-section on each safe interval
    psi_opt = 0.0
    delta_omega_opt = float("inf")
    found_finite = False

    for lo, hi in safe_intervals:
        # Check if there's any finite value in this interval
        test_psi = (lo + hi) / 2.0
        test_val = f(test_psi)
        if not np.isfinite(test_val):
            # Quick scan to check if any point in interval is finite
            scan = np.linspace(lo, hi, 21)
            scan_vals = [f(p) for p in scan]
            if not any(np.isfinite(v) for v in scan_vals):
                continue  # entire interval is singular — skip

        try:
            psi_gs, dt_gs = golden_section_minimize(f, lo, hi, tol=1e-12, max_iter=200)
            if np.isfinite(dt_gs) and dt_gs < delta_omega_opt:
                psi_opt = float(psi_gs)
                delta_omega_opt = float(dt_gs)
                found_finite = True
        except (RuntimeError, ValueError):
            continue

    if not found_finite:
        # Fallback: coarse grid (should rarely happen)
        grid = np.linspace(0.0, 2.0 * np.pi, 1001)
        for p in grid:
            v = f(p)
            if np.isfinite(v) and v < delta_omega_opt:
                psi_opt = float(p)
                delta_omega_opt = float(v)
                found_finite = True

    if not found_finite:
        return 0.0, float("inf"), 1.0, 0.0

    psi_opt = psi_opt % (2.0 * np.pi)
    a_opt = float(np.cos(psi_opt))
    b_opt = float(np.sin(psi_opt))

    return psi_opt, delta_omega_opt, a_opt, b_opt


# ============================================================================
# Sensitivity Computation (Full Forward Pass)
# ============================================================================


def compute_moments_and_derivatives(
    psi0: np.ndarray,
    T_BS1: float,
    T_BS2: float,
    t_hold: float,
    omega_true: float,
    alpha: tuple[float, float, float, float],
    ops_np: dict[str, np.ndarray],
    N: int,
    M: int,
    fd_step: float = 1e-6,
) -> tuple[
    tuple[float, float, float, float, float, float],
    tuple[float, float, float, float, float, float],
]:
    """Compute the six moments and their omega-derivatives.

    Args:
        psi0: Initial state vector.
        T_BS1: First beam-splitter duration.
        T_BS2: Second beam-splitter duration.
        t_hold: Holding time.
        omega_true: True phase rate parameter.
        alpha: (alpha_xx, alpha_xz, alpha_zx, alpha_zz).
        ops_np: Operators from build_collective_operators().
        N: System particle number.
        M: Ancilla particle number.
        fd_step: Finite-difference step for derivative (default 1e-6).

    Returns:
        Tuple (moments, d_moments) each as (exp_S, exp_A, var_S, var_A, cov, norm).
    """
    # Moments at omega_true
    psi = evolve_full_np(psi0, T_BS1, T_BS2, t_hold, omega_true, alpha, ops_np, N, M)
    moments = compute_six_moments(psi, ops_np)

    # Moments at omega_true + fd_step
    psi_plus = evolve_full_np(
        psi0, T_BS1, T_BS2, t_hold, omega_true + fd_step, alpha, ops_np, N, M
    )
    moments_plus = compute_six_moments(psi_plus, ops_np)

    # Moments at omega_true - fd_step
    psi_minus = evolve_full_np(
        psi0, T_BS1, T_BS2, t_hold, omega_true - fd_step, alpha, ops_np, N, M
    )
    moments_minus = compute_six_moments(psi_minus, ops_np)

    # Central finite differences
    d_moments_raw = tuple(
        (p - m) / (2.0 * fd_step)
        for p, m in zip(moments_plus, moments_minus, strict=True)
    )
    d_moments = cast("tuple[float, float, float, float, float, float]", d_moments_raw)

    return moments, d_moments


@overload
def compute_sensitivity_weighted(
    psi0: np.ndarray,
    T_BS1: float,
    T_BS2: float,
    t_hold: float,
    omega_true: float,
    alpha: tuple[float, float, float, float],
    ops_np: dict[str, np.ndarray],
    N: int,
    M: int,
    fd_step: float = 1e-6,
    return_optimal_weights: Literal[False] = False,
) -> float: ...


@overload
def compute_sensitivity_weighted(
    psi0: np.ndarray,
    T_BS1: float,
    T_BS2: float,
    t_hold: float,
    omega_true: float,
    alpha: tuple[float, float, float, float],
    ops_np: dict[str, np.ndarray],
    N: int,
    M: int,
    fd_step: float = 1e-6,
    return_optimal_weights: Literal[True] = True,
) -> tuple[float, float, float, float]: ...


def compute_sensitivity_weighted(
    psi0: np.ndarray,
    T_BS1: float,
    T_BS2: float,
    t_hold: float,
    omega_true: float,
    alpha: tuple[float, float, float, float],
    ops_np: dict[str, np.ndarray],
    N: int,
    M: int,
    fd_step: float = 1e-6,
    return_optimal_weights: bool = False,
) -> float | tuple[float, float, float, float]:
    """Compute weighted-optimal Δθ via golden-section sub-optimisation.

    Args:
        psi0: Initial state vector.
        T_BS1: First beam-splitter duration.
        T_BS2: Second beam-splitter duration.
        t_hold: Holding time.
        omega_true: True phase rate parameter.
        alpha: (alpha_xx, alpha_xz, alpha_zx, alpha_zz).
        ops_np: Operators from build_collective_operators().
        N: System particle number.
        M: Ancilla particle number.
        fd_step: Finite-difference step (default 1e-6).
        return_optimal_weights: If True, also return (psi*, a*, b*, delta_omega*).

    Returns:
        If return_optimal_weights is False: Δθ (positive float).
        If True: tuple (psi_opt, a_opt, b_opt, delta_omega_opt).
    """
    moments, d_moments = compute_moments_and_derivatives(
        psi0, T_BS1, T_BS2, t_hold, omega_true, alpha, ops_np, N, M, fd_step
    )

    psi_opt, delta_omega_opt, a_opt, b_opt = optimize_weight_psi(moments, d_moments)

    if return_optimal_weights:
        return psi_opt, a_opt, b_opt, delta_omega_opt
    return delta_omega_opt


def compute_sensitivity_sonly(
    psi0: np.ndarray,
    T_BS1: float,
    T_BS2: float,
    t_hold: float,
    omega_true: float,
    alpha: tuple[float, float, float, float],
    ops_np: dict[str, np.ndarray],
    N: int,
    M: int,
    fd_step: float = 1e-6,
) -> float:
    """Compute Δθ using S-only measurement (J_z^S).

    Args:
        Same as compute_sensitivity_weighted.

    Returns:
        Δθ (positive float).
    """
    moments, d_moments = compute_moments_and_derivatives(
        psi0, T_BS1, T_BS2, t_hold, omega_true, alpha, ops_np, N, M, fd_step
    )
    # S-only: a=1, b=0 → psi=0
    return compute_weighted_delta_omega(1.0, 0.0, moments, d_moments)


# ============================================================================
# Exact Closed-Form Expressions for Analytical Benchmarks
# ============================================================================


def _so3_rotate_z(point: np.ndarray, angle: float) -> np.ndarray:
    """Rotate a 3D point about the z-axis by angle (SO(3) matrix)."""
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return R @ point


def _so3_rotate_y(point: np.ndarray, angle: float) -> np.ndarray:
    """Rotate a 3D point about the y-axis by angle (SO(3) matrix)."""
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return R @ point


def _so3_rotate_x(point: np.ndarray, angle: float) -> np.ndarray:
    """Rotate a 3D point about the x-axis by angle (SO(3) matrix)."""
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    return R @ point


def _exact_zero_interaction_moments(
    N: int,
    M: int,
    t_hold: float,
    omega_true: float = 1.0,
    theta_S: float = 0.0,
    theta_A: float = 0.0,
    T_BS1: float = np.pi / 2.0,
    T_BS2: float = np.pi / 2.0,
) -> tuple[
    tuple[float, float, float, float, float, float],
    tuple[float, float, float, float, float, float],
]:
    """Exact closed-form moments and omega-derivatives for α=0 (zero interaction).

    Uses SO(3) rotation formulas: the system evolves as
    R_x(T_BS2) R_z(t_hold θ) R_x(T_BS1) R_y(Θ_S) |J_S, -J_S⟩,
    which is a CSS at a known Bloch-sphere point. The ancilla evolves
    independently as R_x(T_BS1+T_BS2) R_y(Θ_A) |J_A, -J_A⟩.

    Since S and A are in a product state, Cov = 0 always.

    Returns:
        Tuple (moments, d_moments) each as
        (exp_S, exp_A, var_S, var_A, cov_SA, norm).
        d_moments are computed via analytical θ-derivative.
    """
    J_S = N / 2.0
    J_A = M / 2.0
    phi = t_hold * omega_true

    # --- System: SO(3) rotation sequence ---
    # Start from south pole (0, 0, -1)
    r = np.array([0.0, 0.0, -1.0])

    # Apply R_y(Θ_S)
    r = _so3_rotate_y(r, -theta_S)  # CSS uses exp(-i Θ J_y), which in SO(3) is R_y(-Θ)
    # Apply R_x(T_BS1)
    r = _so3_rotate_x(r, -T_BS1)  # exp(-i T J_x) in SO(3) is R_x(-T)
    # Apply R_z(φ) where φ = t_hold θ
    r = _so3_rotate_z(r, -phi)  # exp(-i φ J_z) in SO(3) is R_z(-φ)
    # Apply R_x(T_BS2)
    r = _so3_rotate_x(r, -T_BS2)  # exp(-i T J_x) in SO(3) is R_x(-T)

    r_z_S = r[2]

    # System moments
    exp_S = float(J_S * r_z_S)
    var_S = float(J_S / 2.0 * (1.0 - r_z_S**2))

    # Analytical θ-derivative
    # Recompute r_z symbolically from the SO(3) chain
    c_theta_S = np.cos(theta_S)
    s_theta_S = np.sin(theta_S)
    s_BS1 = np.sin(T_BS1)
    s_BS2 = np.sin(T_BS2)
    c_phi = np.cos(phi)
    s_phi = np.sin(phi)

    # Verified derivation: r_z matches the SO(3) rotation chain result
    dr_z_domega = t_hold * (
        s_theta_S * s_BS2 * c_phi + s_BS1 * c_theta_S * s_BS2 * s_phi
    )
    d_exp_S = float(J_S * dr_z_domega)

    # --- Ancilla: no θ dependence ---
    r_A = np.array([0.0, 0.0, -1.0])
    r_A = _so3_rotate_y(r_A, -theta_A)
    r_A = _so3_rotate_x(r_A, -(T_BS1 + T_BS2))
    r_z_A = r_A[2]

    exp_A = float(J_A * r_z_A)
    var_A = float(J_A / 2.0 * (1.0 - r_z_A**2))

    # Covariance is zero (product state)
    cov_SA = 0.0
    norm = 1.0

    moments = (exp_S, exp_A, var_S, var_A, cov_SA, norm)
    d_moments = (d_exp_S, 0.0, 0.0, 0.0, 0.0, 0.0)

    return moments, d_moments


def _exact_alpha_zz_moments(
    N: int,
    M: int,
    alpha_zz: float,
    t_hold: float,
    omega_true: float = 1.0,
    theta_S: float = 0.0,
    theta_A: float = 0.0,
    T_BS1: float = np.pi / 2.0,
    T_BS2: float = np.pi / 2.0,
    fd_step: float = 1e-6,
) -> tuple[
    tuple[float, float, float, float, float, float],
    tuple[float, float, float, float, float, float],
]:
    """Exact moments and omega-derivatives for α_zz-only interaction.

    Uses the diagonal nature of the hold Hamiltonian:
        H_hold = θ J_z^S + α_zz J_z^S ⊗ J_z^A
    which is diagonal in the product Dicke basis |m_S, m_A⟩.

    The hold evolution is applied as elementwise phase multiplication
    (no full-space matrix exponentiation). BS rotations use subsystem-level
    expm on J_x (identical to the primary path), providing an independent
    validation of the hold evolution.

    Returns:
        Tuple (moments, d_moments) each as
        (exp_S, exp_A, var_S, var_A, cov_SA, norm).
    """
    from scipy.linalg import expm

    d_S = N + 1
    d_A = M + 1

    # CSS states (uses subsystem expm for J_y rotation, same as numerical path)
    psi_S = css_state_np(N / 2.0, theta_S)
    psi_A = css_state_np(M / 2.0, theta_A)

    # BS1 rotations (subsystem expm only, not full-space)
    U_BS1_S = expm(-1j * T_BS1 * jx_operator(N, basis=OperatorBasis.DICKE))
    U_BS1_A = expm(-1j * T_BS1 * jx_operator(M, basis=OperatorBasis.DICKE))
    psi_S_bs1 = U_BS1_S @ psi_S
    psi_A_bs1 = U_BS1_A @ psi_A

    # Build full state after BS1 as product state in matrix form
    # psi_hold[i, j] = |m_S=i, m_A=j⟩ amplitude
    psi_hold = np.outer(psi_S_bs1, psi_A_bs1)  # d_S × d_A matrix

    # m-values for each index (Dicke ordering: m = N/2, ..., -N/2)
    m_S_vals = np.arange(N / 2.0, -N / 2.0 - 1, -1)
    m_A_vals = np.arange(M / 2.0, -M / 2.0 - 1, -1)

    # Precompute m_S and m_A value matrices for elementwise phase
    m_S_mat = m_S_vals[:, np.newaxis]  # d_S × 1
    m_A_mat = m_A_vals[np.newaxis, :]  # 1 × d_A

    # J_z operators for final moment computation
    Jz_S_np = jz_operator(N, basis=OperatorBasis.DICKE)
    Jz_A_np = jz_operator(M, basis=OperatorBasis.DICKE)
    Jz_S_full = np.kron(Jz_S_np, np.eye(d_A, dtype=complex))
    Jz_A_full = np.kron(np.eye(d_S, dtype=complex), Jz_A_np)
    Jz_S_sq = Jz_S_full @ Jz_S_full
    Jz_A_sq = Jz_A_full @ Jz_A_full
    Jz_S_Jz_A = Jz_S_full @ Jz_A_full

    # BS2 rotation matrices (subsystem only)
    U_BS2_S = expm(-1j * T_BS2 * jx_operator(N, basis=OperatorBasis.DICKE))
    U_BS2_A = expm(-1j * T_BS2 * jx_operator(M, basis=OperatorBasis.DICKE))

    # Helper to compute moments at a given ω
    def _compute_moments_at_omega(
        omega_val: float,
    ) -> tuple[float, float, float, float, float, float]:
        phi = t_hold * omega_val

        # Apply diagonal hold phase elementwise
        # phase = exp(-i t_hold (θ m_S + α_zz m_S m_A))
        phase = np.exp(-1j * (phi * m_S_mat + t_hold * alpha_zz * m_S_mat * m_A_mat))
        psi_hold_phased = psi_hold * phase

        # Apply BS2 via Kronecker product: R_S ⊗ R_A
        # Efficient: reshape, multiply, reshape back
        psi_final_mat = U_BS2_S @ psi_hold_phased @ U_BS2_A.T
        psi_final = psi_final_mat.reshape(-1)

        # Compute moments
        exp_S = float(np.real(psi_final.conj() @ Jz_S_full @ psi_final))
        exp_A = float(np.real(psi_final.conj() @ Jz_A_full @ psi_final))
        exp_S_sq = float(np.real(psi_final.conj() @ Jz_S_sq @ psi_final))
        exp_A_sq = float(np.real(psi_final.conj() @ Jz_A_sq @ psi_final))
        exp_SA = float(np.real(psi_final.conj() @ Jz_S_Jz_A @ psi_final))
        norm = float(np.real(psi_final.conj() @ psi_final))

        var_S = exp_S_sq - exp_S**2
        var_A = exp_A_sq - exp_A**2
        cov_SA = exp_SA - exp_S * exp_A

        return exp_S, exp_A, var_S, var_A, cov_SA, norm

    # Moments at omega_true
    moments = _compute_moments_at_omega(omega_true)

    # Central finite differences for θ-derivatives
    moments_plus = _compute_moments_at_omega(omega_true + fd_step)
    moments_minus = _compute_moments_at_omega(omega_true - fd_step)

    d_moments = cast(
        "tuple[float, float, float, float, float, float]",
        tuple(
            (p - m) / (2.0 * fd_step)
            for p, m in zip(moments_plus, moments_minus, strict=True)
        ),
    )

    return moments, d_moments


# ============================================================================
# Analytical Benchmarks
# ============================================================================


def analytical_benchmark_zero_interaction(
    N: int,
    M: int,
    t_hold: float,
    omega_true: float = 1.0,
    theta_S: float = 0.0,
    theta_A: float = 0.0,
    T_BS1: float = np.pi / 2.0,
    T_BS2: float | None = None,
    fd_step: float = 1e-5,
) -> dict[str, Any]:
    """Benchmark: zero interaction (alpha = 0).

    At alpha=0, the optimal weight is a* = 1, b* = 0 (S-only measurement),
    and the sensitivity should match the exact closed-form expression
    derived from SO(3) rotation formulas to within 10^{-10} relative error.

    Uses S-only (a=1,b=0) directly to avoid optimizer bias from the
    degenerate weight landscape (Var_A can be zero for certain parameters).

    Args:
        N: System particle number.
        M: Ancilla particle number.
        t_hold: Holding time.
        omega_true: True phase rate (default 1.0).
        theta_S: System CSS angle (default 0.0).
        theta_A: Ancilla CSS angle (default 0.0).
        T_BS1: First beam-splitter duration (default pi/2).
        T_BS2: Second beam-splitter duration (default same as T_BS1).
        fd_step: Finite-difference step (tiny for high precision).

    Returns:
        Dict with keys: 'delta_omega_numerical', 'delta_omega_exact',
        'relative_error_to_exact', 'optimal_weight_a',
        'optimal_weight_b', 'psi_opt', 'expected_sql'.
    """
    if T_BS2 is None:
        T_BS2 = T_BS1

    alpha = (0.0, 0.0, 0.0, 0.0)
    ops_np = build_collective_operators(N, M)
    psi0 = product_css_state_np(theta_S, theta_A, N, M)

    # Use S-only (a=1, b=0) directly — optimal at α=0, avoids optimizer bias
    delta_omega_opt = compute_sensitivity_sonly(
        psi0,
        T_BS1,
        T_BS2,
        t_hold,
        omega_true,
        alpha,
        ops_np,
        N,
        M,
        fd_step,
    )
    a_opt, b_opt, psi_opt = 1.0, 0.0, 0.0

    # Exact closed-form moments (SO(3) rotation formulas)
    exact_moments, exact_d_moments = _exact_zero_interaction_moments(
        N,
        M,
        t_hold,
        omega_true,
        theta_S,
        theta_A,
        T_BS1,
        T_BS2,
    )

    # Exact S-only (a=1, b=0) sensitivity
    delta_omega_exact = compute_weighted_delta_omega(
        1.0,
        0.0,
        exact_moments,
        exact_d_moments,
    )

    relative_error_to_exact = (
        abs(delta_omega_opt - delta_omega_exact) / delta_omega_exact
        if delta_omega_exact > 0 and np.isfinite(delta_omega_exact)
        else float("inf")
    )

    # Expected SQL: for CSS at optimal, Δθ_SQL = 1/(sqrt(N) t_hold)
    expected_sql = 1.0 / (np.sqrt(N) * t_hold)

    return {
        "N": N,
        "M": M,
        "t_hold": t_hold,
        "delta_omega_numerical": delta_omega_opt,
        "delta_omega_exact": delta_omega_exact,
        "relative_error_to_exact": relative_error_to_exact,
        "expected_sql": expected_sql,
        "optimal_weight_a": a_opt,
        "optimal_weight_b": b_opt,
        "psi_opt": psi_opt,
    }


def analytical_benchmark_alpha_zz_only(
    N: int,
    M: int,
    alpha_zz: float,
    t_hold: float,
    omega_true: float = 1.0,
    theta_S: float = 0.0,
    theta_A: float = 0.0,
    T_BS1: float = np.pi / 2.0,
    T_BS2: float | None = None,
    fd_step: float = 1e-5,
) -> dict[str, Any]:
    """Benchmark: only alpha_zz is non-zero (diagonal evolution).

    When only alpha_zz != 0, the hold Hamiltonian is
    H_hold = omega J_z^S + alpha_zz J_z^S ⊗ J_z^A,
    which is diagonal in the product Dicke basis |m_S, m_A⟩.

    The numerical result is compared against an exact computation that
    applies the hold via elementwise phase multiplication (no full-space
    matrix exponentiation), achieving 10^{-10} relative error for all
    six raw moments. The sensitivity Δθ is also compared, but when the
    derivative d⟨M⟩/dθ is very small the relative error may be larger.

    Args:
        N: System particle number.
        M: Ancilla particle number.
        alpha_zz: ZZ interaction strength.
        t_hold: Holding time.
        omega_true: True phase rate (default 1.0).
        theta_S: System CSS angle (default 0.0).
        theta_A: Ancilla CSS angle (default 0.0).
        T_BS1: First beam-splitter duration (default pi/2).
        T_BS2: Second beam-splitter duration (default same as T_BS1).
        fd_step: Finite-difference step.

    Returns:
        Dict with results including 'exact_moments', 'exact_d_moments',
        'moments_max_rel_error', 'delta_omega_exact', 'relative_error_to_exact'.
    """
    if T_BS2 is None:
        T_BS2 = T_BS1

    alpha = (0.0, 0.0, 0.0, alpha_zz)
    ops_np = build_collective_operators(N, M)
    psi0 = product_css_state_np(theta_S, theta_A, N, M)

    # Numerical moments and weighted measurement
    num_moments_raw, num_d_moments_raw = compute_moments_and_derivatives(
        psi0,
        T_BS1,
        T_BS2,
        t_hold,
        omega_true,
        alpha,
        ops_np,
        N,
        M,
        fd_step,
    )
    num_moments = np.array(num_moments_raw)

    # Weighted Δθ (optimal and S-only)
    psi_opt, dt_weighted, a_opt, b_opt = optimize_weight_psi(
        num_moments_raw,
        num_d_moments_raw,
    )
    dt_sonly = compute_weighted_delta_omega(
        1.0,
        0.0,
        num_moments_raw,
        num_d_moments_raw,
    )

    # Exact moments (elementwise hold phase, no full-space expm)
    exact_moments_raw, exact_d_moments_raw = _exact_alpha_zz_moments(
        N,
        M,
        alpha_zz,
        t_hold,
        omega_true,
        theta_S,
        theta_A,
        T_BS1,
        T_BS2,
        fd_step,
    )
    exact_moments = np.array(exact_moments_raw)

    # Compare moments elementwise
    # For small values, use absolute tolerance; for large values, use relative
    moment_max_rel = 0.0
    moment_max_abs = 0.0
    for i in range(6):
        denom = max(abs(num_moments[i]), abs(exact_moments[i]), 1e-15)
        rel_err = abs(num_moments[i] - exact_moments[i]) / denom
        abs_err = abs(num_moments[i] - exact_moments[i])
        moment_max_rel = max(moment_max_rel, rel_err)
        moment_max_abs = max(moment_max_abs, abs_err)

    # S-only Δθ from exact moments
    dt_sonly_exact = compute_weighted_delta_omega(
        1.0,
        0.0,
        exact_moments_raw,
        exact_d_moments_raw,
    )
    rel_error_sonly = (
        abs(dt_sonly - dt_sonly_exact) / dt_sonly_exact
        if dt_sonly_exact > 0 and np.isfinite(dt_sonly_exact)
        else float("inf")
    )

    # Weighted Δθ from exact moments (using numerical optimal φ)
    dt_weighted_exact = compute_weighted_delta_omega(
        a_opt,
        b_opt,
        exact_moments_raw,
        exact_d_moments_raw,
    )
    rel_error_weighted = (
        abs(dt_weighted - dt_weighted_exact) / dt_weighted_exact
        if dt_weighted_exact > 0 and np.isfinite(dt_weighted_exact)
        else float("inf")
    )

    return {
        "N": N,
        "M": M,
        "alpha_zz": alpha_zz,
        "t_hold": t_hold,
        # S-only comparison (best-conditioned)
        "delta_omega_numerical": dt_sonly,
        "delta_omega_exact": dt_sonly_exact,
        "relative_error_to_exact": rel_error_sonly,
        # Weighted comparison (diagnostic)
        "delta_omega_weighted": dt_weighted,
        "delta_omega_weighted_exact": dt_weighted_exact,
        "relative_error_weighted": rel_error_weighted,
        # Moment comparison
        "moments_max_rel_error": moment_max_rel,
        "moments_max_abs_error": moment_max_abs,
        "num_moments": num_moments_raw,
        "exact_moments": exact_moments_raw,
        "optimal_weight_a": a_opt,
        "optimal_weight_b": b_opt,
        "psi_opt": psi_opt,
    }


# ============================================================================
# Gradient via Finite Differences (for L-BFGS-B)
# ============================================================================


def _objective_and_gradient_fd(
    params: np.ndarray,
    N: int,
    M: int,
    omega_true: float,
    ops_np: dict[str, np.ndarray],
    fd_step: float = 1e-6,
    grad_step: float = 1e-5,
) -> tuple[float, np.ndarray]:
    """Objective and gradient via finite differences for L-BFGS-B.

    Parameter vector (9 elements):
        [theta_S, theta_A, T_BS1, T_BS2, t_hold, alpha_xx, alpha_xz, alpha_zx, alpha_zz]

    Args:
        params: 9-element parameter vector.
        N: System particle number.
        M: Ancilla particle number.
        omega_true: True phase rate.
        ops_np: Operators from build_collective_operators().
        fd_step: FD step for omega derivative in sensitivity (default 1e-6).
        grad_step: FD step for gradient computation (default 1e-5).

    Returns:
        Tuple (objective_value, gradient_vector).
    """
    theta_S, theta_A, T_BS1, T_BS2, t_hold = params[:5]
    alpha = (float(params[5]), float(params[6]), float(params[7]), float(params[8]))

    # Objective at the current point
    psi0 = product_css_state_np(float(theta_S), float(theta_A), N, M)
    f0 = float(
        compute_sensitivity_weighted(
            psi0,
            float(T_BS1),
            float(T_BS2),
            float(t_hold),
            omega_true,
            alpha,
            ops_np,
            N,
            M,
            fd_step=fd_step,
            return_optimal_weights=False,
        )
    )

    # Gradient via central finite differences
    grad = np.zeros(9, dtype=float)
    for i in range(9):
        params_plus = params.copy()
        params_plus[i] += grad_step
        theta_S_p, theta_A_p, T_BS1_p, T_BS2_p, t_hold_p = params_plus[:5]
        alpha_p = (
            float(params_plus[5]),
            float(params_plus[6]),
            float(params_plus[7]),
            float(params_plus[8]),
        )
        psi0_p = product_css_state_np(float(theta_S_p), float(theta_A_p), N, M)
        fp = float(
            compute_sensitivity_weighted(
                psi0_p,
                float(T_BS1_p),
                float(T_BS2_p),
                float(t_hold_p),
                omega_true,
                alpha_p,
                ops_np,
                N,
                M,
                fd_step=fd_step,
                return_optimal_weights=False,
            )
        )

        params_minus = params.copy()
        params_minus[i] -= grad_step
        theta_S_m, theta_A_m, T_BS1_m, T_BS2_m, t_hold_m = params_minus[:5]
        alpha_m = (
            float(params_minus[5]),
            float(params_minus[6]),
            float(params_minus[7]),
            float(params_minus[8]),
        )
        psi0_m = product_css_state_np(float(theta_S_m), float(theta_A_m), N, M)
        fm = float(
            compute_sensitivity_weighted(
                psi0_m,
                float(T_BS1_m),
                float(T_BS2_m),
                float(t_hold_m),
                omega_true,
                alpha_m,
                ops_np,
                N,
                M,
                fd_step=fd_step,
                return_optimal_weights=False,
            )
        )

        grad[i] = (fp - fm) / (2.0 * grad_step)

    return f0, grad


def _objective_and_gradient_ad(
    params: np.ndarray,
    N: int,
    M: int,
    omega_true: float,
    ops_np: dict[str, np.ndarray],
    fd_step: float = 1e-6,
) -> tuple[float, np.ndarray]:
    """Objective and gradient via automatic differentiation for L-BFGS-B.

    Uses torch.autograd.grad to differentiate through the circuit at the
    optimal weight angle psi^*, applying the envelope theorem (Danskin's
    theorem): since psi^* is an interior minimiser, the gradient of the
    objective equals the partial derivative of Delta_omega with respect
    to the circuit parameters, evaluated at psi^*.

    The omega-derivative (d_moments) is computed via finite differences
    through the SAME torch circuit (with grad), so the gradient correctly
    captures the dependence of d_moments on the circuit parameters.
    The weight angle psi^* is treated as constant (envelope theorem),
    so the psi-subproblem is evaluated via numpy on detached values.

    Parameter vector (9 elements):
        [theta_S, theta_A, T_BS1, T_BS2, t_hold, alpha_xx, alpha_xz, alpha_zx, alpha_zz]

    Args:
        params: 9-element parameter vector.
        N: System particle number.
        M: Ancilla particle number.
        omega_true: True phase rate.
        ops_np: Operators from build_collective_operators().
        fd_step: FD step for omega derivative in sensitivity (default 1e-6).

    Returns:
        Tuple (objective_value, gradient_vector).
    """
    ops_torch = operators_to_torch(ops_np)

    # Convert all 9 params to a single torch tensor with requires_grad
    p_torch = torch.tensor(params, dtype=torch.float64, requires_grad=True)
    theta_S_t = p_torch[0]
    theta_A_t = p_torch[1]
    T_BS1_t = p_torch[2]
    T_BS2_t = p_torch[3]
    t_hold_t = p_torch[4]
    alpha_t: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] = (
        p_torch[5],
        p_torch[6],
        p_torch[7],
        p_torch[8],
    )

    # --- Step 1: Build CSS state (shared across omega evaluations) ---
    psi0 = product_css_state_torch(theta_S_t, theta_A_t, N, M, ops_torch)

    # --- Step 2: Run circuit at omega, omega+δ, omega-δ (all with grad) ---
    omega_t = torch.tensor(omega_true, dtype=torch.float64)
    omega_plus_t = torch.tensor(omega_true + fd_step, dtype=torch.float64)
    omega_minus_t = torch.tensor(omega_true - fd_step, dtype=torch.float64)

    psi = evolve_full_torch(
        psi0,
        T_BS1_t,
        T_BS2_t,
        t_hold_t,
        omega_t,
        alpha_t,
        ops_torch,
        N,
        M,
    )
    psi_plus = evolve_full_torch(
        psi0,
        T_BS1_t,
        T_BS2_t,
        t_hold_t,
        omega_plus_t,
        alpha_t,
        ops_torch,
        N,
        M,
    )
    psi_minus = evolve_full_torch(
        psi0,
        T_BS1_t,
        T_BS2_t,
        t_hold_t,
        omega_minus_t,
        alpha_t,
        ops_torch,
        N,
        M,
    )

    # --- Step 3: Compute moments (torch, in graph) ---
    exp_S, exp_A, var_S, var_A, cov_SA, norm = _compute_six_moments_torch(
        psi, ops_torch
    )
    exp_S_p, exp_A_p, var_S_p, var_A_p, cov_SA_p, norm_p = _compute_six_moments_torch(
        psi_plus, ops_torch
    )
    exp_S_m, exp_A_m, var_S_m, var_A_m, cov_SA_m, norm_m = _compute_six_moments_torch(
        psi_minus, ops_torch
    )

    # --- Step 4: d_moments via FD (in the graph) ---
    inv_2fd = 1.0 / (2.0 * fd_step)
    d_exp_S = (exp_S_p - exp_S_m) * inv_2fd
    d_exp_A = (exp_A_p - exp_A_m) * inv_2fd
    d_var_S = (var_S_p - var_S_m) * inv_2fd
    d_var_A = (var_A_p - var_A_m) * inv_2fd
    d_cov_SA = (cov_SA_p - cov_SA_m) * inv_2fd
    d_norm = (norm_p - norm_m) * inv_2fd

    # --- Step 5: Find psi^* via numpy on detached tensors ---
    moments_np = cast(
        "tuple[float, float, float, float, float, float]",
        tuple(
            float(t.detach().numpy())
            for t in (exp_S, exp_A, var_S, var_A, cov_SA, norm)
        ),
    )
    d_moments_np = cast(
        "tuple[float, float, float, float, float, float]",
        tuple(
            float(t.detach().numpy())
            for t in (d_exp_S, d_exp_A, d_var_S, d_var_A, d_cov_SA, d_norm)
        ),
    )
    _, _delta_omega_opt, a_opt, b_opt = optimize_weight_psi(
        moments_np,
        d_moments_np,
    )

    # --- Step 6: Re-evaluate Δθ at psi^* as a torch scalar ---
    a_t = torch.tensor(a_opt, dtype=torch.float64)
    b_t = torch.tensor(b_opt, dtype=torch.float64)

    d_exp_M = a_t * d_exp_S + b_t * d_exp_A

    var_M = a_t**2 * var_S + b_t**2 * var_A + 2.0 * a_t * b_t * cov_SA
    var_M = torch.where(var_M < 0.0, torch.zeros_like(var_M), var_M)

    # --- Fringe-extremum detection ---
    if torch.abs(d_exp_M) < 1e-8 or var_M < 1e-14:
        grad = np.zeros(9, dtype=float)
        return float("inf"), grad

    delta_omega = torch.sqrt(var_M) / torch.abs(d_exp_M)

    # --- Step 7: Compute gradient via AD ---
    (grad_torch,) = torch.autograd.grad(delta_omega, p_torch)
    grad = grad_torch.detach().numpy().copy()

    return float(delta_omega.detach().numpy()), grad


# ============================================================================
# L-BFGS-B Optimisation
# ============================================================================


def get_bounds_nm(N: int, M: int) -> dict[str, tuple[float, float]]:
    """Get default bounds for the 9-parameter optimisation.

    Args:
        N: System particle number.
        M: Ancilla particle number (unused, for interface consistency).

    Returns:
        Dict of bound tuples.
    """
    return {
        "bloch_theta": (0.0, np.pi),  # theta_S, theta_A
        "T_BS": (0.0, np.pi),  # T_BS1, T_BS2
        "t_hold": (0.1, 20.0),  # t_hold
        "alpha": (-2.0, 2.0),  # alpha_xx, _xz, _zx, _zz
    }


def random_params_nm(
    rng: np.random.Generator,
    N: int,
    M: int,
    bounds: dict[str, tuple[float, float]] | None = None,
) -> np.ndarray:
    """Generate random 9-element parameter vector within bounds.

    Args:
        rng: Random number generator.
        N: System particle number.
        M: Ancilla particle number.
        bounds: Optional custom bounds (uses defaults if None).

    Returns:
        9-element array: [theta_S, theta_A, T_BS1, T_BS2, t_hold,
                         alpha_xx, alpha_xz, alpha_zx, alpha_zz].
    """
    if bounds is None:
        bounds = get_bounds_nm(N, M)

    theta_lo, theta_hi = bounds["bloch_theta"]
    tbs_lo, tbs_hi = bounds["T_BS"]
    th_lo, th_hi = bounds["t_hold"]
    alpha_lo, alpha_hi = bounds["alpha"]

    theta_S = rng.uniform(theta_lo, theta_hi)
    theta_A = rng.uniform(theta_lo, theta_hi)
    T_BS1 = rng.uniform(tbs_lo, tbs_hi)
    T_BS2 = rng.uniform(tbs_lo, tbs_hi)
    t_hold = rng.uniform(th_lo, th_hi)
    alpha = rng.uniform(alpha_lo, alpha_hi, size=4)

    return np.array(
        [theta_S, theta_A, T_BS1, T_BS2, t_hold, *alpha],
        dtype=float,
    )


def run_lbfgsb_optimisation(
    N: int,
    M: int,
    omega_true: float = 1.0,
    x0: np.ndarray | None = None,
    seed: int | None = 42,
    maxiter: int = 500,
    fd_step: float = 1e-6,
    grad_step: float = 1e-5,
    bounds_dict: dict[str, tuple[float, float]] | None = None,
    method: str = "fd",
) -> dict[str, Any]:
    """Run L-BFGS-B optimisation for the N,M weighted joint measurement.

    Args:
        N: System particle number.
        M: Ancilla particle number.
        omega_true: True phase rate.
        x0: Initial 9-element parameter vector (random if None).
        seed: Random seed (used if x0 is None).
        maxiter: Maximum L-BFGS-B iterations.
        fd_step: FD step for omega derivative.
        grad_step: FD step for gradient computation (only used for
            method='fd').
        bounds_dict: Custom bounds (uses defaults if None).
        method: Gradient computation method. 'fd' for finite differences
            (default, 18 circuit evaluations per gradient call), 'ad' for
            automatic differentiation via torch.autograd.grad
            (3 circuit evaluations per gradient call). Use 'ad' for
            faster and more accurate gradients; use 'fd' as reference.

    Returns:
        Dict with keys: 'delta_omega_opt', 'params_opt', 'success',
        'message', 'nfev', 'njev', 'N', 'M', 'psi_opt', 'a_opt', 'b_opt',
        'exp_S', 'exp_A', 'var_S', 'var_A', 'cov_SA'.
    """
    if method not in ("fd", "ad"):
        raise ValueError(f"method must be 'fd' or 'ad', got {method!r}")

    if bounds_dict is None:
        bounds_dict = get_bounds_nm(N, M)

    ops_np = build_collective_operators(N, M)

    if x0 is None:
        rng = np.random.default_rng(seed)
        x0 = random_params_nm(rng, N, M, bounds_dict)

    # Build bound list for L-BFGS-B
    bound_list = [
        bounds_dict["bloch_theta"],  # theta_S
        bounds_dict["bloch_theta"],  # theta_A
        bounds_dict["T_BS"],  # T_BS1
        bounds_dict["T_BS"],  # T_BS2
        bounds_dict["t_hold"],  # t_hold
        bounds_dict["alpha"],  # alpha_xx
        bounds_dict["alpha"],  # alpha_xz
        bounds_dict["alpha"],  # alpha_zx
        bounds_dict["alpha"],  # alpha_zz
    ]

    # Select gradient function based on method
    if method == "fd":
        _obj_grad_fn: Callable[..., tuple[float, np.ndarray]] = (
            _objective_and_gradient_fd
        )
        obj_grad_kwargs: dict[str, Any] = {
            "fd_step": fd_step,
            "grad_step": grad_step,
        }
    else:
        _obj_grad_fn = _objective_and_gradient_ad
        obj_grad_kwargs = {"fd_step": fd_step}

    def _objective_grad_wrapper(params: np.ndarray) -> tuple[float, np.ndarray]:
        return _obj_grad_fn(params, N, M, omega_true, ops_np, **obj_grad_kwargs)

    # Objective function for scipy
    def objective(params: np.ndarray) -> float:
        val, _ = _objective_grad_wrapper(params)
        return val

    # Gradient function for scipy
    def gradient(params: np.ndarray) -> np.ndarray:
        _, grad = _objective_grad_wrapper(params)
        return grad

    result = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        jac=gradient,
        bounds=bound_list,
        options={
            "maxiter": maxiter,
            "ftol": 1e-12,
            "gtol": 1e-8,
            "maxls": 50,
        },
    )

    opt_params = result.x
    theta_S_opt, theta_A_opt, T_BS1_opt, T_BS2_opt, t_hold_opt = opt_params[:5]
    alpha_opt = (
        float(opt_params[5]),
        float(opt_params[6]),
        float(opt_params[7]),
        float(opt_params[8]),
    )

    # Compute optimal weights and diagnostics at the optimal point
    psi0_opt = product_css_state_np(float(theta_S_opt), float(theta_A_opt), N, M)

    try:
        weighted_result = compute_sensitivity_weighted(
            psi0_opt,
            float(T_BS1_opt),
            float(T_BS2_opt),
            float(t_hold_opt),
            omega_true,
            alpha_opt,
            ops_np,
            N,
            M,
            fd_step=fd_step,
            return_optimal_weights=True,
        )
        psi_opt, a_opt, b_opt, delta_omega_opt = weighted_result
    except (ValueError, np.linalg.LinAlgError, TypeError):
        psi_opt, a_opt, b_opt = 0.0, 1.0, 0.0
        delta_omega_opt = float(result.fun)

    # Diagnostics
    moments, _d_moments = compute_moments_and_derivatives(
        psi0_opt,
        float(T_BS1_opt),
        float(T_BS2_opt),
        float(t_hold_opt),
        omega_true,
        alpha_opt,
        ops_np,
        N,
        M,
        fd_step,
    )
    exp_S, exp_A, var_S, var_A, cov_SA, _ = moments

    return {
        "N": N,
        "M": M,
        "delta_omega_opt": delta_omega_opt,
        "params_opt": opt_params,
        "success": bool(result.success),
        "message": str(result.message),
        "nfev": int(result.nfev),
        "njev": int(getattr(result, "njev", 0)),
        "psi_opt": psi_opt,
        "a_opt": a_opt,
        "b_opt": b_opt,
        "exp_S": exp_S,
        "exp_A": exp_A,
        "var_S": var_S,
        "var_A": var_A,
        "cov_SA": cov_SA,
    }


# ============================================================================
# N-Scaling and M-Scaling Analysis
# ============================================================================


@dataclass
class NScalingResult:
    """Result from N-scaling analysis.

    Attributes:
        N_values: Array of N values scanned.
        M_value: Fixed ancilla size used.
        delta_omega_values: Mean Δθ per N across seeds (after optimisation).
        delta_omega_std: Standard deviation of Δθ across seeds per N.
            Used for weighted regression weights w_N = 1/σ_N².
        delta_omega_seeds: Full per-seed Δθ array, shape (len(N_values), num_seeds).
            None when not collected (legacy/backward-compatible).
        psi_opt_values: Optimal weight angle for each N (from best seed).
        a_opt_values: Optimal weight a for each N (from best seed).
        b_opt_values: Optimal weight b for each N (from best seed).
        scaling_exponent: Fitted exponent nu in Δθ ∝ N^{-nu}. Median of bootstrap
            distribution; the point estimate.
        scaling_exponent_err: Standard error of nu from the weighted fit on the
            full dataset (pre-bootstrap). Retained for backward compatibility.
        scaling_exponent_ci: 95% bootstrap confidence interval for nu as
            (lower_2.5, upper_97.5).
        curvature: Quadratic coefficient beta in weighted log-log quadratic fit.
            Median of bootstrap distribution.
        curvature_err: Standard error of beta from the full-dataset quadratic fit.
            Retained for backward compatibility.
        curvature_ci: 95% bootstrap confidence interval for beta as
            (lower_2.5, upper_97.5).
        R_squared: Goodness of fit (weighted R² from the linear model on full data).
        num_seeds: Number of random seeds used per N.
        n_bootstrap: Number of bootstrap resamples used for CIs. Default 10000.
        sql_scaling: Expected SQL exponent (-0.5).
        hl_scaling: Expected Heisenberg exponent (-1.0).
    """

    N_values: np.ndarray
    M_value: int
    delta_omega_values: np.ndarray
    psi_opt_values: np.ndarray
    a_opt_values: np.ndarray
    b_opt_values: np.ndarray
    scaling_exponent: float
    scaling_exponent_err: float
    curvature: float
    curvature_err: float
    R_squared: float
    num_seeds: int
    delta_omega_std: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=float)
    )
    delta_omega_seeds: np.ndarray | None = None
    scaling_exponent_ci: tuple[float, float] = (float("nan"), float("nan"))
    curvature_ci: tuple[float, float] = (float("nan"), float("nan"))
    n_bootstrap: int = 10000
    sql_scaling: float = -0.5
    hl_scaling: float = -1.0

    def to_dataframe(self) -> pd.DataFrame:
        """Flatten into a DataFrame (mean per N + std + scalar metadata)."""
        n = len(self.N_values)
        data: dict[str, np.ndarray] = {
            "N": self.N_values,
            "delta_omega": self.delta_omega_values,
            "delta_omega_std": self.delta_omega_std
            if len(self.delta_omega_std) == n
            else np.full(n, float("nan")),
            "psi_opt": self.psi_opt_values,
            "a_opt": self.a_opt_values,
            "b_opt": self.b_opt_values,
            "M_value": np.full(n, self.M_value, dtype=int),
            "scaling_exponent": np.full(n, self.scaling_exponent),
            "scaling_exponent_err": np.full(n, self.scaling_exponent_err),
            "scaling_exponent_ci_lower": np.full(n, self.scaling_exponent_ci[0]),
            "scaling_exponent_ci_upper": np.full(n, self.scaling_exponent_ci[1]),
            "curvature": np.full(n, self.curvature),
            "curvature_err": np.full(n, self.curvature_err),
            "curvature_ci_lower": np.full(n, self.curvature_ci[0]),
            "curvature_ci_upper": np.full(n, self.curvature_ci[1]),
            "R_squared": np.full(n, self.R_squared),
            "num_seeds": np.full(n, self.num_seeds, dtype=int),
            "n_bootstrap": np.full(n, self.n_bootstrap, dtype=int),
            "sql_scaling": np.full(n, self.sql_scaling),
            "hl_scaling": np.full(n, self.hl_scaling),
        }
        return pd.DataFrame(data)

    def save_parquet(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_parquet(path, index=False)
        return path

    @classmethod
    def from_parquet(cls, path: str | Path) -> NScalingResult:
        """Load from a Parquet file saved by save_parquet()."""
        path = Path(path)
        df = pd.read_parquet(path)

        # Reconstruct arrays from the DataFrame
        N_values = df["N"].to_numpy(dtype=int)
        delta_omega_values = df["delta_omega"].to_numpy(dtype=float)
        delta_omega_std = df["delta_omega_std"].to_numpy(dtype=float)
        psi_opt_values = df["psi_opt"].to_numpy(dtype=float)
        a_opt_values = df["a_opt"].to_numpy(dtype=float)
        b_opt_values = df["b_opt"].to_numpy(dtype=float)

        # Read scalar metadata from the first row
        required_scalar_cols = [
            "M_value",
            "scaling_exponent",
            "scaling_exponent_err",
            "scaling_exponent_ci_lower",
            "scaling_exponent_ci_upper",
            "curvature",
            "curvature_err",
            "curvature_ci_lower",
            "curvature_ci_upper",
            "R_squared",
            "num_seeds",
            "n_bootstrap",
            "sql_scaling",
            "hl_scaling",
        ]
        missing = [c for c in required_scalar_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing required scalar columns in {path}: {missing}. "
                f"Re-run the simulation that generated this file."
            )

        if "delta_omega_std" not in df.columns:
            raise ValueError(
                f"Missing required column 'delta_omega_std' in {path}. "
                f"Re-run the simulation that generated this file."
            )

        return cls(
            N_values=N_values,
            M_value=int(df["M_value"].iloc[0]),
            delta_omega_values=delta_omega_values,
            psi_opt_values=psi_opt_values,
            a_opt_values=a_opt_values,
            b_opt_values=b_opt_values,
            scaling_exponent=float(df["scaling_exponent"].iloc[0]),
            scaling_exponent_err=float(df["scaling_exponent_err"].iloc[0]),
            curvature=float(df["curvature"].iloc[0]),
            curvature_err=float(df["curvature_err"].iloc[0]),
            R_squared=float(df["R_squared"].iloc[0]),
            num_seeds=int(df["num_seeds"].iloc[0]),
            delta_omega_std=delta_omega_std,
            scaling_exponent_ci=(
                float(df["scaling_exponent_ci_lower"].iloc[0]),
                float(df["scaling_exponent_ci_upper"].iloc[0]),
            ),
            curvature_ci=(
                float(df["curvature_ci_lower"].iloc[0]),
                float(df["curvature_ci_upper"].iloc[0]),
            ),
            n_bootstrap=int(df["n_bootstrap"].iloc[0]),
            sql_scaling=float(df["sql_scaling"].iloc[0]),
            hl_scaling=float(df["hl_scaling"].iloc[0]),
        )


@dataclass
class MScalingResult:
    """Result from M-scaling analysis (diminishing returns from ancilla).

    Attributes:
        M_values: Array of M values scanned.
        N_value: Fixed system size used.
        delta_omega_values: Best Δθ for each M.
        psi_opt_values: Optimal weight angle for each M.
        a_opt_values: Optimal weight a for each M.
        b_opt_values: Optimal weight b for each M.
        improvement_01: Fractional improvement from M=0 to M=1.
        diminishing_threshold: Threshold for diminishing returns.
    """

    M_values: np.ndarray
    N_value: int
    delta_omega_values: np.ndarray
    psi_opt_values: np.ndarray
    a_opt_values: np.ndarray
    b_opt_values: np.ndarray
    improvement_01: float = 0.0
    diminishing_threshold: float = 0.5

    def to_dataframe(self) -> pd.DataFrame:
        n = len(self.M_values)
        data = {
            "M": self.M_values,
            "delta_omega": self.delta_omega_values,
            "psi_opt": self.psi_opt_values,
            "a_opt": self.a_opt_values,
            "b_opt": self.b_opt_values,
            "N_value": np.full(n, self.N_value, dtype=int),
            "improvement_01": np.full(n, self.improvement_01),
            "diminishing_threshold": np.full(n, self.diminishing_threshold),
        }
        return pd.DataFrame(data)

    def save_parquet(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_parquet(path, index=False)
        return path

    @classmethod
    def from_parquet(cls, path: str | Path) -> MScalingResult:
        """Load from a Parquet file saved by save_parquet()."""
        path = Path(path)
        df = pd.read_parquet(path)

        required_scalar_cols = ["N_value", "improvement_01", "diminishing_threshold"]
        missing = [c for c in required_scalar_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing required scalar columns in {path}: {missing}. "
                f"Re-run the simulation that generated this file."
            )

        return cls(
            M_values=df["M"].to_numpy(dtype=int),
            N_value=int(df["N_value"].iloc[0]),
            delta_omega_values=df["delta_omega"].to_numpy(dtype=float),
            psi_opt_values=df["psi_opt"].to_numpy(dtype=float),
            a_opt_values=df["a_opt"].to_numpy(dtype=float),
            b_opt_values=df["b_opt"].to_numpy(dtype=float),
            improvement_01=float(df["improvement_01"].iloc[0]),
            diminishing_threshold=float(df["diminishing_threshold"].iloc[0]),
        )


# ============================================================================
# Weighted Log-Log Regression Helpers
# ============================================================================


def _weighted_loglog_linear(
    log_N: np.ndarray,
    log_dt: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float, float, np.ndarray]:
    """Weighted log-log linear regression: log(Δθ) = -nu·log(N) + c.

    Solves the weighted least-squares problem with design matrix
    A = [-log_N, 1] and weight matrix W = diag(weights).

    Args:
        log_N: log(N) values.
        log_dt: log(Δθ) values.
        weights: Weights w_N = 1/σ_N² (positive).

    Returns:
        (nu, c, R_squared, residuals) where nu is the scaling exponent,
        c is the intercept, R_sq is the weighted R², and residuals are
        the weighted residuals.
    """
    A = np.column_stack([-log_N, np.ones_like(log_N)])
    sqrt_w = np.sqrt(weights)
    Aw = A * sqrt_w[:, np.newaxis]
    yw = log_dt * sqrt_w
    coeffs, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
    nu = float(coeffs[0])
    c = float(coeffs[1])

    # Weighted residuals and R²
    residuals = log_dt - (A @ coeffs)
    ss_res = np.sum(weights * residuals**2)
    y_bar = np.average(log_dt, weights=weights)
    ss_tot = np.sum(weights * (log_dt - y_bar) ** 2)
    R_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return nu, c, float(R_sq), residuals


def _weighted_loglog_quadratic(
    log_N: np.ndarray,
    log_dt: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float, float, float]:
    """Weighted log-log quadratic fit: log(Δθ) = -nu·log(N) + beta·(log N)² + c.

    Returns:
        (nu, beta, c, R_sq) where beta is the curvature diagnostic.
    """
    A = np.column_stack([-log_N, log_N**2, np.ones_like(log_N)])
    sqrt_w = np.sqrt(weights)
    Aw = A * sqrt_w[:, np.newaxis]
    yw = log_dt * sqrt_w
    coeffs, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
    nu = float(coeffs[0])
    beta = float(coeffs[1])
    c = float(coeffs[2])

    # Weighted R²
    residuals = log_dt - (A @ coeffs)
    ss_res = np.sum(weights * residuals**2)
    y_bar = np.average(log_dt, weights=weights)
    ss_tot = np.sum(weights * (log_dt - y_bar) ** 2)
    R_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return nu, beta, c, float(R_sq)


def _bootstrap_scaling(
    log_N: np.ndarray,
    log_dt: np.ndarray,
    weights: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap resampling for scaling exponent and curvature.

    Resamples (N, Δθ, weight) tuples with replacement, recomputes the
    weighted linear and quadratic fits, and returns the bootstrap
    distributions of ν and β.

    Args:
        log_N: log(N) values.
        log_dt: log(Δθ) values.
        weights: Weights w_N = 1/σ_N².
        n_bootstrap: Number of bootstrap resamples (e.g. 10⁴).
        rng: NumPy random generator.

    Returns:
        (nu_bootstrap, beta_bootstrap) arrays of shape (n_bootstrap,),
        containing the bootstrapped ν and curvature β values.
    """
    n_pts = len(log_N)
    nu_b = np.zeros(n_bootstrap)
    beta_b = np.zeros(n_bootstrap)

    for b in range(n_bootstrap):
        # Resample with replacement
        idx = rng.integers(0, n_pts, size=n_pts)
        log_N_b = log_N[idx]
        log_dt_b = log_dt[idx]
        w_b = weights[idx]

        nu_b[b], _, _, _ = _weighted_loglog_linear(log_N_b, log_dt_b, w_b)
        _, beta_b[b], _, _ = _weighted_loglog_quadratic(log_N_b, log_dt_b, w_b)

    return nu_b, beta_b


def run_n_scaling(
    N_values: list[int],
    M: int = -1,
    omega_true: float = 1.0,
    num_seeds: int = 20,
    maxiter: int = 200,
    seed: int = 42,
    fd_step: float = 1e-6,
    grad_step: float = 1e-5,
    n_bootstrap: int = 10000,
) -> NScalingResult:
    """Run N-scaling analysis for the weighted joint measurement.

    For each N in N_values, runs L-BFGS-B optimisation with ``num_seeds``
    random restarts and collects all per-seed Δθ values. The per-N mean
    and standard deviation are used in a weighted log-log regression:
        log(Δθ) = -ν·log(N) + c,
    with weights w_N = 1 / σ_N².

    A bootstrap (with replacement, 10⁴ resamples) estimates the 95%
    confidence interval for ν. The curvature diagnostic β from a quadratic
    fit log(Δθ) = -ν·log(N) + β·(log N)² + c is also bootstrapped.

    Args:
        N_values: List of system particle numbers to scan.
        M: Ancilla particle number. If -1, uses M = N for each N.
        omega_true: True phase rate.
        num_seeds: Number of random seeds per N. Default 20; raise
            further for production runs.
        maxiter: Maximum L-BFGS-B iterations per run.
        seed: Base random seed.
        fd_step: FD step for omega derivative.
        grad_step: FD step for gradient.
        n_bootstrap: Number of bootstrap resamples for CI. Default 10000.

    Returns:
        NScalingResult with fitted scaling exponent, bootstrap CI,
        curvature diagnostic, and per-seed statistics.
    """
    N_arr = np.array(N_values, dtype=int)
    n_N = len(N_arr)
    # Use seed as base for the outer optimisation rng
    outer_rng = np.random.default_rng(seed)

    # Per-seed storage: (n_N, num_seeds)
    all_delta_omega = np.full((n_N, num_seeds), np.inf, dtype=float)
    all_psi = np.full((n_N, num_seeds), np.nan, dtype=float)
    all_a = np.full((n_N, num_seeds), np.nan, dtype=float)
    all_b = np.full((n_N, num_seeds), np.nan, dtype=float)

    for i, N_val in enumerate(N_arr):
        M_val: int = N_val if M == -1 else M
        for restart in range(num_seeds):
            rng = np.random.default_rng(seed + N_val * 1000 + restart)
            x0 = random_params_nm(rng, N_val, M_val)
            result = run_lbfgsb_optimisation(
                N_val,
                M_val,
                omega_true=omega_true,
                x0=x0,
                maxiter=maxiter,
                fd_step=fd_step,
                grad_step=grad_step,
            )
            dt = result["delta_omega_opt"]
            all_delta_omega[i, restart] = dt
            all_psi[i, restart] = result["psi_opt"]
            all_a[i, restart] = result["a_opt"]
            all_b[i, restart] = result["b_opt"]

    # --- Per-N statistics (mean, std, best) ---
    dt_valid = np.where(np.isfinite(all_delta_omega), all_delta_omega, np.nan)
    dt_mean = np.nanmean(dt_valid, axis=1)
    dt_std = np.nanstd(dt_valid, axis=1)
    # Fallback: if std is zero (e.g. single seed), use a small positive value
    dt_std = np.maximum(dt_std, 1e-30)

    # Best per-N corresponds to minimum Δθ across seeds
    best_idx = np.nanargmin(all_delta_omega, axis=1)
    psi_opt = all_psi[np.arange(n_N), best_idx]
    a_opt = all_a[np.arange(n_N), best_idx]
    b_opt = all_b[np.arange(n_N), best_idx]

    # --- Weighted log-log regression ---
    finite_mask = np.isfinite(dt_mean)
    # Require at least 3 points for a meaningful fit
    if np.sum(finite_mask) < 3:
        return NScalingResult(
            N_values=N_arr,
            M_value=M,
            delta_omega_values=dt_mean,
            psi_opt_values=psi_opt,
            a_opt_values=a_opt,
            b_opt_values=b_opt,
            scaling_exponent=float("nan"),
            scaling_exponent_err=float("nan"),
            curvature=float("nan"),
            curvature_err=float("nan"),
            R_squared=float("nan"),
            num_seeds=num_seeds,
            delta_omega_std=dt_std,
            delta_omega_seeds=all_delta_omega,
            scaling_exponent_ci=(float("nan"), float("nan")),
            curvature_ci=(float("nan"), float("nan")),
            n_bootstrap=n_bootstrap,
        )

    N_fit = N_arr[finite_mask]
    dt_mean_fit = dt_mean[finite_mask]
    dt_std_fit = dt_std[finite_mask]
    log_N = np.log(N_fit.astype(float))
    log_dt = np.log(dt_mean_fit)
    weights = 1.0 / (dt_std_fit**2)  # w_N = 1/σ_N²

    # --- Weighted linear fit on full data ---
    nu_lin, c_lin, R_sq, _ = _weighted_loglog_linear(log_N, log_dt, weights)

    # --- Weighted quadratic fit on full data (curvature) ---
    nu_quad, beta_quad, c_quad, _ = _weighted_loglog_quadratic(log_N, log_dt, weights)

    # --- Standard errors from weighted linear fit ---
    A_lin = np.column_stack([-log_N, np.ones_like(log_N)])
    W = np.diag(weights)
    n_pts = len(log_N)
    residuals_lin = log_dt - (A_lin @ np.array([nu_lin, c_lin]))
    mse = (
        np.sum(weights * residuals_lin**2) / (n_pts - 2) if n_pts > 2 else float("nan")
    )
    try:
        cov_nu_c = mse * np.linalg.inv(A_lin.T @ W @ A_lin)
        nu_err = float(np.sqrt(cov_nu_c[0, 0])) if np.isfinite(mse) else float("nan")
    except np.linalg.LinAlgError:
        nu_err = float("nan")

    # --- Standard error for beta from weighted quadratic fit ---
    A_quad = np.column_stack([-log_N, log_N**2, np.ones_like(log_N)])
    quad_coeffs = np.array([nu_quad, beta_quad, c_quad])
    residuals_quad = log_dt - (A_quad @ quad_coeffs)
    mse_q = (
        np.sum(weights * residuals_quad**2) / (n_pts - 3) if n_pts > 3 else float("nan")
    )
    try:
        cov_quad = mse_q * np.linalg.inv(A_quad.T @ W @ A_quad)
        beta_err = (
            float(np.sqrt(cov_quad[1, 1])) if np.isfinite(mse_q) else float("nan")
        )
    except np.linalg.LinAlgError:
        beta_err = float("nan")

    # --- Bootstrap for CI on ν and β ---
    nu_bootstrap, beta_bootstrap = _bootstrap_scaling(
        log_N,
        log_dt,
        weights,
        n_bootstrap,
        outer_rng,
    )

    scaling_exponent = float(np.median(nu_bootstrap))
    scaling_exponent_ci = (
        float(np.percentile(nu_bootstrap, 2.5)),
        float(np.percentile(nu_bootstrap, 97.5)),
    )
    curvature = float(np.median(beta_bootstrap))
    curvature_ci = (
        float(np.percentile(beta_bootstrap, 2.5)),
        float(np.percentile(beta_bootstrap, 97.5)),
    )

    return NScalingResult(
        N_values=N_arr,
        M_value=M,
        delta_omega_values=dt_mean,
        psi_opt_values=psi_opt,
        a_opt_values=a_opt,
        b_opt_values=b_opt,
        scaling_exponent=scaling_exponent,
        scaling_exponent_err=nu_err,
        curvature=curvature,
        curvature_err=beta_err,
        R_squared=R_sq,
        num_seeds=num_seeds,
        delta_omega_std=dt_std,
        delta_omega_seeds=all_delta_omega,
        scaling_exponent_ci=scaling_exponent_ci,
        curvature_ci=curvature_ci,
        n_bootstrap=n_bootstrap,
    )


def run_m_scaling(
    M_values: list[int],
    N: int = 4,
    omega_true: float = 1.0,
    num_seeds: int = 20,
    maxiter: int = 200,
    seed: int = 42,
    fd_step: float = 1e-6,
    grad_step: float = 1e-5,
) -> MScalingResult:
    """Run M-scaling analysis (diminishing returns from ancilla size).

    Args:
        M_values: List of ancilla particle numbers to scan.
        N: Fixed system particle number (default 4).
        omega_true: True phase rate.
        num_seeds: Number of random seeds per M.
        maxiter: Maximum L-BFGS-B iterations per run.
        seed: Base random seed.
        fd_step: FD step for omega derivative.
        grad_step: FD step for gradient.

    Returns:
        MScalingResult.
    """
    M_arr = np.array(M_values, dtype=int)
    delta_omega_arr = np.full(len(M_arr), np.inf, dtype=float)
    psi_arr = np.full(len(M_arr), np.nan, dtype=float)
    a_arr = np.full(len(M_arr), np.nan, dtype=float)
    b_arr = np.full(len(M_arr), np.nan, dtype=float)

    for i, M_val in enumerate(M_arr):
        best_dt = float("inf")
        best_psi = float("nan")
        best_a = float("nan")
        best_b = float("nan")

        for restart in range(num_seeds):
            rng = np.random.default_rng(seed + M_val * 1000 + restart)
            x0 = random_params_nm(rng, N, M_val)
            result = run_lbfgsb_optimisation(
                N,
                M_val,
                omega_true=omega_true,
                x0=x0,
                maxiter=maxiter,
                fd_step=fd_step,
                grad_step=grad_step,
            )
            dt = result["delta_omega_opt"]
            if np.isfinite(dt) and dt < best_dt:
                best_dt = dt
                best_psi = result["psi_opt"]
                best_a = result["a_opt"]
                best_b = result["b_opt"]

        delta_omega_arr[i] = best_dt
        psi_arr[i] = best_psi
        a_arr[i] = best_a
        b_arr[i] = best_b

    # Compute improvement from M=0 to M=1
    improvement_01 = 0.0
    if (
        len(M_arr) >= 2
        and M_arr[0] == 0
        and M_arr[1] == 1
        and np.isfinite(delta_omega_arr[0])
        and np.isfinite(delta_omega_arr[1])
        and delta_omega_arr[0] > 0
    ):
        improvement_01 = (delta_omega_arr[0] - delta_omega_arr[1]) / delta_omega_arr[0]

    return MScalingResult(
        M_values=M_arr,
        N_value=N,
        delta_omega_values=delta_omega_arr,
        psi_opt_values=psi_arr,
        a_opt_values=a_arr,
        b_opt_values=b_arr,
        improvement_01=improvement_01,
    )


# ============================================================================
# Alpha Scan with Weight Re-Optimisation
# ============================================================================


@dataclass
class AlphaReoptResultNM:
    """Result from scanning alpha with weight re-optimisation.

    Attributes:
        alpha_name: Which coefficient was scanned.
        alpha_values: Array of alpha values.
        delta_omega_weighted: Best Δθ with weighted joint measurement.
        delta_omega_sonly: Δθ with S-only measurement at same params.
        a_opt_values: Optimal weight a for each alpha.
        b_opt_values: Optimal weight b for each alpha.
        psi_opt_values: Optimal psi for each alpha.
        N: System particle number.
        M: Ancilla particle number.
    """

    alpha_name: str
    alpha_values: np.ndarray
    delta_omega_weighted: np.ndarray
    delta_omega_sonly: np.ndarray
    a_opt_values: np.ndarray
    b_opt_values: np.ndarray
    psi_opt_values: np.ndarray
    N: int
    M: int

    def to_dataframe(self) -> pd.DataFrame:
        n = len(self.alpha_values)
        data = {
            "alpha": self.alpha_values,
            "delta_omega_weighted": self.delta_omega_weighted,
            "delta_omega_sonly": self.delta_omega_sonly,
            "a_opt": self.a_opt_values,
            "b_opt": self.b_opt_values,
            "psi_opt": self.psi_opt_values,
            "alpha_name": np.full(n, self.alpha_name, dtype=object),
            "N": np.full(n, self.N, dtype=int),
            "M": np.full(n, self.M, dtype=int),
        }
        return pd.DataFrame(data)

    def save_parquet(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_parquet(path, index=False)
        return path

    @classmethod
    def from_parquet(cls, path: str | Path) -> AlphaReoptResultNM:
        """Load from a Parquet file saved by save_parquet()."""
        path = Path(path)
        df = pd.read_parquet(path)

        required_scalar_cols = ["alpha_name", "N", "M"]
        missing = [c for c in required_scalar_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing required scalar columns in {path}: {missing}. "
                f"Re-run the simulation that generated this file."
            )

        return cls(
            alpha_name=str(df["alpha_name"].iloc[0]),
            alpha_values=df["alpha"].to_numpy(dtype=float),
            delta_omega_weighted=df["delta_omega_weighted"].to_numpy(dtype=float),
            delta_omega_sonly=df["delta_omega_sonly"].to_numpy(dtype=float),
            a_opt_values=df["a_opt"].to_numpy(dtype=float),
            b_opt_values=df["b_opt"].to_numpy(dtype=float),
            psi_opt_values=df["psi_opt"].to_numpy(dtype=float),
            N=int(df["N"].iloc[0]),
            M=int(df["M"].iloc[0]),
        )


def run_alpha_scan_with_reoptimisation(
    alpha_name: str,
    N: int,
    M: int,
    alpha_values: np.ndarray | None = None,
    omega_true: float = 1.0,
    num_seeds: int = 3,
    maxiter: int = 100,
    fd_step: float = 1e-6,
    grad_step: float = 1e-5,
) -> AlphaReoptResultNM:
    """Scan a single alpha coefficient with state re-optimisation.

    Args:
        alpha_name: 'xx', 'xz', 'zx', or 'zz'.
        N: System particle number.
        M: Ancilla particle number.
        alpha_values: Array of alpha values to scan.
        omega_true: True phase rate.
        num_seeds: Number of random seeds per alpha value.
        maxiter: Maximum L-BFGS-B iterations per run.
        fd_step: FD step for omega derivative.
        grad_step: FD step for gradient.

    Returns:
        AlphaReoptResultNM.
    """
    alpha_idx_map = {"xx": 0, "xz": 1, "zx": 2, "zz": 3}
    if alpha_name not in alpha_idx_map:
        raise ValueError(
            f"alpha_name must be one of {list(alpha_idx_map.keys())}, got {alpha_name}"
        )
    scan_idx = alpha_idx_map[alpha_name]

    if alpha_values is None:
        alpha_values = np.linspace(-2.0, 2.0, 21)

    alpha_arr = np.asarray(alpha_values, dtype=float)
    n_pts = len(alpha_arr)

    dt_weighted = np.full(n_pts, np.inf, dtype=float)
    dt_sonly = np.full(n_pts, np.inf, dtype=float)
    a_opt_arr = np.full(n_pts, np.nan, dtype=float)
    b_opt_arr = np.full(n_pts, np.nan, dtype=float)
    psi_opt_arr = np.full(n_pts, np.nan, dtype=float)

    ops_np = build_collective_operators(N, M)

    for i, a_val in enumerate(alpha_arr):
        alpha_list: list[float] = [0.0, 0.0, 0.0, 0.0]
        alpha_list[scan_idx] = a_val
        fixed_alpha = cast("tuple[float, float, float, float]", tuple(alpha_list))

        best_dt_w = float("inf")
        best_dt_s = float("inf")
        best_psi = float("nan")
        best_a_v = float("nan")
        best_b_v = float("nan")

        for restart in range(num_seeds):
            rng = np.random.default_rng(42 + i * 1000 + restart)
            x0 = random_params_nm(rng, N, M)

            # Build objective: weighted with fixed alpha
            def make_obj(
                alpha_fixed: tuple[float, float, float, float],
            ) -> Callable[[np.ndarray], float]:
                def obj(params: np.ndarray) -> float:
                    theta_S, theta_A, Tb1, Tb2, Th = params[:5]
                    psi0 = product_css_state_np(float(theta_S), float(theta_A), N, M)
                    dt = compute_sensitivity_weighted(
                        psi0,
                        float(Tb1),
                        float(Tb2),
                        float(Th),
                        omega_true,
                        alpha_fixed,
                        ops_np,
                        N,
                        M,
                        fd_step=fd_step,
                        return_optimal_weights=False,
                    )
                    return dt if np.isfinite(dt) else 1e10

                return obj

            # Weighted optimisation (over state params only, fixed alpha)
            state_bounds = [
                (0.0, np.pi),  # theta_S
                (0.0, np.pi),  # theta_A
                (0.0, np.pi),  # T_BS1
                (0.0, np.pi),  # T_BS2
                (0.1, 20.0),  # t_hold
            ]
            obj_w = make_obj(fixed_alpha)
            res_w = minimize(
                obj_w,
                x0[:5],
                method="L-BFGS-B",
                bounds=state_bounds,
                options={"maxiter": maxiter, "ftol": 1e-10},
            )
            opt_params_5 = res_w.x
            theta_S_opt, theta_A_opt, Tb1_opt, Tb2_opt, Th_opt = opt_params_5
            psi0_opt = product_css_state_np(
                float(theta_S_opt), float(theta_A_opt), N, M
            )
            w_result = compute_sensitivity_weighted(
                psi0_opt,
                float(Tb1_opt),
                float(Tb2_opt),
                float(Th_opt),
                omega_true,
                fixed_alpha,
                ops_np,
                N,
                M,
                fd_step=fd_step,
                return_optimal_weights=True,
            )
            psi_w, a_w, b_w, dt_w_val = w_result
            if np.isfinite(dt_w_val) and dt_w_val < best_dt_w:
                best_dt_w = dt_w_val
                best_psi = psi_w
                best_a_v = a_w
                best_b_v = b_w

            # S-only at the same parameters
            dt_s_val = compute_sensitivity_sonly(
                psi0_opt,
                float(Tb1_opt),
                float(Tb2_opt),
                float(Th_opt),
                omega_true,
                fixed_alpha,
                ops_np,
                N,
                M,
                fd_step=fd_step,
            )
            if np.isfinite(dt_s_val) and dt_s_val < best_dt_s:
                best_dt_s = dt_s_val

        dt_weighted[i] = best_dt_w
        dt_sonly[i] = best_dt_s
        a_opt_arr[i] = best_a_v
        b_opt_arr[i] = best_b_v
        psi_opt_arr[i] = best_psi

    return AlphaReoptResultNM(
        alpha_name=alpha_name,
        alpha_values=alpha_arr,
        delta_omega_weighted=dt_weighted,
        delta_omega_sonly=dt_sonly,
        a_opt_values=a_opt_arr,
        b_opt_values=b_opt_arr,
        psi_opt_values=psi_opt_arr,
        N=N,
        M=M,
    )


# ============================================================================
# Validation Helpers
# ============================================================================


def validate_operators_nm(ops_np: dict[str, np.ndarray], N: int, M: int) -> bool:
    """Validate all operators for the N+M system.

    Args:
        ops_np: Operators from build_collective_operators().
        N: System particle number.
        M: Ancilla particle number.

    Returns:
        True if all checks pass.

    Raises:
        AssertionError: If any check fails.
    """
    d_S = N + 1
    d_A = M + 1
    d = d_S * d_A

    for name, op in ops_np.items():
        assert op.shape == (d, d), f"{name} must be {d}×{d}, got {op.shape}"
        assert np.allclose(op, op.conj().T, atol=1e-12), f"{name} must be Hermitian"

    # Commutation: [Jz_S, Jx_S] = i Jy_S
    comm_zx_S = ops_np["Jz_S"] @ ops_np["Jx_S"] - ops_np["Jx_S"] @ ops_np["Jz_S"]
    assert np.allclose(comm_zx_S, 1j * ops_np["Jy_S"], atol=1e-12), (
        "[Jz_S, Jx_S] = i Jy_S failed"
    )

    # Commutation: [Jz_A, Jx_A] = i Jy_A
    comm_zx_A = ops_np["Jz_A"] @ ops_np["Jx_A"] - ops_np["Jx_A"] @ ops_np["Jz_A"]
    assert np.allclose(comm_zx_A, 1j * ops_np["Jy_A"], atol=1e-12), (
        "[Jz_A, Jx_A] = i Jy_A failed"
    )

    return True


def validate_css_state(N: int, theta: float) -> bool:
    """Validate a CSS state.

    Args:
        N: Particle number.
        theta: Polar angle.

    Returns:
        True if normalised.
    """
    J = N / 2.0
    state = css_state_np(J, theta)
    norm = np.linalg.norm(state)
    assert abs(norm - 1.0) < 1e-12, f"CSS state norm = {norm}"
    assert state.shape == (N + 1,), (
        f"CSS state shape mismatch: {state.shape} vs {(N + 1,)}"
    )
    return True


def validate_hl_bound(delta_omega: float, N: int, t_hold: float) -> bool:
    """Validate Heisenberg limit: Δθ ≥ 1/(N * t_hold).

    Args:
        delta_omega: Sensitivity value.
        N: System particle number.
        t_hold: Holding time.

    Returns:
        True if bound holds (with 1e-6 tolerance).
    """
    hl = 1.0 / (N * t_hold)
    assert delta_omega >= hl - 1e-6, (
        f"HL bound violated: Δθ={delta_omega:.6e} < 1/(N t_hold)={hl:.6e}"
    )
    return True


def plot_n_scaling(
    result: NScalingResult | str | Path,
    save_path: str | Path,
    figsize: tuple[float, float] = (7, 5),
) -> Path:
    """Log-log plot of Δθ vs N with weighted fit, SQL, and Heisenberg limits.

    Shows:
    - Mean Δθ per N with error bars (std across seeds)
    - Weighted log-log linear fit line with shaded 95% bootstrap CI
    - SQL reference: Δθ_SQL = 1/(√N · t_hold)
    - Heisenberg limit reference: Δθ_HL = 1/(N · t_hold)
    """
    if isinstance(result, (str, Path)):
        result = NScalingResult.from_parquet(result)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)

    # Use an effective t_hold for reference lines. Since t_hold is optimised
    # per N and varies, show the SQL/HL lines using the maximum t_hold
    # from the bounds (20.0), giving the most optimistic reference.
    t_hold_ref = 20.0

    # Data points with error bars
    ax.errorbar(
        result.N_values,
        result.delta_omega_values,
        yerr=result.delta_omega_std,
        fmt="o",
        color="C0",
        capsize=4,
        markersize=6,
        label=r"$\Delta\omega$ (mean $\pm$ std)",
    )

    # Best Δθ per N (minimum across seeds)
    best_per_N = (
        np.min(result.delta_omega_seeds, axis=1)
        if result.delta_omega_seeds is not None
        else result.delta_omega_values
    )
    ax.scatter(
        result.N_values,
        best_per_N,
        marker="*",
        color="C3",
        s=80,
        zorder=5,
        label="Best per N",
    )

    # Weighted fit line (over the N range)
    N_fit = np.logspace(
        np.log10(max(result.N_values.min(), 1)),
        np.log10(result.N_values.max()),
        100,
    )
    log_N_fit = np.log(N_fit)
    nu = result.scaling_exponent
    c_fit = np.mean(
        np.log(result.delta_omega_values) + nu * np.log(result.N_values.astype(float))
    )
    dt_fit = np.exp(-nu * log_N_fit + c_fit)
    ax.loglog(
        N_fit,
        dt_fit,
        "--",
        color="C0",
        alpha=0.7,
        label=rf"Fit: $\nu = {nu:.3f}$",
    )

    # Shaded 95% CI for the fit (from bootstrap)
    if not np.isnan(result.scaling_exponent_ci[0]):
        nu_lo, nu_hi = result.scaling_exponent_ci
        dt_lo = np.exp(-nu_lo * log_N_fit + c_fit)
        dt_hi = np.exp(-nu_hi * log_N_fit + c_fit)
        ax.fill_between(
            N_fit,
            dt_lo,
            dt_hi,
            alpha=0.15,
            color="C0",
            label=rf"95% CI: $[{nu_lo:.3f}, {nu_hi:.3f}]$",
        )

    # SQL and HL reference lines (using t_hold_ref)
    N_range = np.logspace(
        np.log10(max(result.N_values.min(), 1)),
        np.log10(result.N_values.max()),
        100,
    )
    ax.loglog(
        N_range,
        1.0 / (np.sqrt(N_range) * t_hold_ref),
        ":",
        color="C2",
        alpha=0.6,
        label=r"SQL: $1/(\sqrt{N}\,t_hold)$",
    )
    ax.loglog(
        N_range,
        1.0 / (N_range * t_hold_ref),
        ":",
        color="C4",
        alpha=0.6,
        label=r"HL: $1/(N\,t_hold)$",
    )

    ax.set_xlabel(r"$N$ (system particles)")
    ax.set_ylabel(r"$\Delta\omega$")
    ax.set_title(rf"N-scaling: $\Delta\omega$ vs $N$ (M={result.M_value})")
    ax.legend(fontsize="small")
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


# ──────────────────────────────────────────────
# 7. M-scaling (diminishing returns)
# ──────────────────────────────────────────────


def plot_m_scaling(
    result: MScalingResult | str | Path,
    save_path: str | Path,
    figsize: tuple[float, float] = (7, 5),
) -> Path:
    """Plot Δθ vs M showing diminishing returns from ancilla size.

    Shows:
    - Best Δθ per M
    - Optimal weight a* and b* per M (twin axis)
    - SQL reference for N=4
    """
    if isinstance(result, (str, Path)):
        result = MScalingResult.from_parquet(result)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=figsize)
    fig.subplots_adjust(right=0.85)

    # Δθ vs M
    ax1.semilogy(
        result.M_values,
        result.delta_omega_values,
        marker="o",
        linestyle="-",
        color="C0",
        linewidth=2,
        markersize=6,
        label=r"$\Delta\omega$ (best)",
    )

    # SQL reference for N=4
    t_hold_ref = 20.0
    sql = 1.0 / (np.sqrt(result.N_value) * t_hold_ref)
    ax1.axhline(
        y=sql,
        color="C2",
        linestyle=":",
        alpha=0.6,
        label=rf"SQL: $1/(\sqrt{{{result.N_value}}}\,t_hold)$",
    )

    # Ensure all data points are within the plotted y-range
    y_min = 0.99 * min(result.delta_omega_values)
    y_max = 1.01 * max(result.delta_omega_values)
    ax1.set_ylim(y_min, y_max)

    ax1.set_xlabel(r"$M$ (ancilla particles)")
    ax1.set_ylabel(r"$\Delta\omega$")
    ax1.set_title(
        rf"M-scaling: $\Delta\omega$ vs ancilla size $M$ (N={result.N_value})"
    )

    # Annotate improvement from M=0 to M=1
    if result.improvement_01 > 0:
        improvement_pct = result.improvement_01 * 100
        text_y = y_min / 1.015
        ax1.annotate(
            rf"$\Delta\omega$ improvement: {improvement_pct:.1f}\%",
            xy=(1, result.delta_omega_values[1]),
            xytext=(4, text_y),
            arrowprops={"arrowstyle": "->", "color": "gray"},
            fontsize=10,
        )

    ax1.legend(fontsize="small")

    # Twin axis for optimal weights
    ax2 = ax1.twinx()
    ax2.plot(
        result.M_values,
        result.a_opt_values,
        marker="s",
        linestyle="--",
        color="C3",
        alpha=0.6,
        markersize=4,
        label=r"$a^*$",
    )
    ax2.plot(
        result.M_values,
        result.b_opt_values,
        marker="^",
        linestyle="--",
        color="C4",
        alpha=0.6,
        markersize=4,
        label=r"$b^*$",
    )
    ax2.set_ylabel(r"Optimal weight $a^*, b^*$")
    ax2.legend(fontsize="small", loc="lower right")

    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


# ──────────────────────────────────────────────
# 8. α-scan with weight re-optimisation (N,M-general)
# ──────────────────────────────────────────────


def plot_weighted_alpha_scan(
    result: AlphaReoptResultNM | str | Path,
    save_path: str | Path,
    figsize: tuple[float, float] = (7, 5),
) -> Path:
    """Plot Δθ vs α for weighted joint and S-only measurements.

    Shows:
    - Weighted joint measurement Δθ vs α
    - S-only measurement Δθ vs α at same optimised parameters
    - Optimal weight angle (secondary axis)
    """
    if isinstance(result, (str, Path)):
        result = AlphaReoptResultNM.from_parquet(result)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=figsize)

    # Weighted joint measurement
    ax1.plot(
        result.alpha_values,
        result.delta_omega_weighted,
        marker="o",
        linestyle="-",
        color="C0",
        linewidth=2,
        markersize=6,
        label=r"Weighted joint: $\Delta\omega$",
    )

    # S-only measurement
    ax1.plot(
        result.alpha_values,
        result.delta_omega_sonly,
        marker="s",
        linestyle="--",
        color="C1",
        linewidth=2,
        markersize=6,
        label=r"S-only: $\Delta\omega$",
    )

    ax1.set_xlabel(rf"$\alpha_{{{result.alpha_name}}}$")
    ax1.set_ylabel(r"$\Delta\omega$")
    ax1.set_title(rf"$\alpha_{{{result.alpha_name}}}$ scan: N={result.N}, M={result.M}")
    ax1.legend(fontsize="small")

    # Twin axis for weight angle
    ax2 = ax1.twinx()
    ax2.plot(
        result.alpha_values,
        result.psi_opt_values,
        marker=".",
        linestyle=":",
        color="gray",
        alpha=0.5,
        markersize=3,
        label=r"$\psi^*$",
    )
    ax2.set_ylabel(r"Optimal $\psi$ (rad)")
    ax2.legend(fontsize="small", loc="lower right")

    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


# ============================================================================
# (C) Report Generation Pipeline
# ============================================================================

REPORTS_DIR = Path(__file__).resolve().parent.parent
REPORT_DATE = "20260518"
DRIVE_OMEGA_VALS = [0.1, 0.5, 1.0, 2.0, 5.0]


def _parquet_path(name: str) -> Path:
    return REPORTS_DIR / REPORT_DATE / "raw_data" / f"{REPORT_DATE}-{name}.parquet"


def _fig_path(name: str) -> Path:
    return REPORTS_DIR / REPORT_DATE / "figures" / f"{REPORT_DATE}-{name}.svg"


def generate_n_scaling(force: bool = False) -> None:
    """N-scaling with optimal weights, M=N."""
    csv_p = _parquet_path("n-scaling")
    fig_p = _fig_path("n-scaling")

    if csv_p.exists() and not force:
        print(f"[skip] {csv_p.name} exists (use --force to overwrite)")
        result = NScalingResult.from_parquet(csv_p)
    else:
        print("[run]  Computing N-scaling (may be slow)...")
        N_vals = [1, 2, 3, 4, 6, 8, 12, 16]
        result = run_n_scaling(
            N_values=N_vals,
            M=-1,
            omega_true=1.0,
            num_seeds=5,
            maxiter=200,
            seed=42,
            n_bootstrap=10000,
        )
        result.save_parquet(csv_p)
        print(f"[save] {csv_p}")

    result = NScalingResult.from_parquet(csv_p)
    plot_n_scaling(result, fig_p)
    print(f"[fig]  {fig_p}")


def generate_m_scaling(force: bool = False) -> None:
    """M-scaling at fixed N=4."""
    csv_p = _parquet_path("m-scaling")
    fig_p = _fig_path("m-scaling")

    if csv_p.exists() and not force:
        print(f"[skip] {csv_p.name} exists (use --force to overwrite)")
        result = MScalingResult.from_parquet(csv_p)
    else:
        print("[run]  Computing M-scaling (may be slow)...")
        M_vals = [0, 1, 2, 3, 4, 6, 8, 12]
        result = run_m_scaling(
            M_values=M_vals,
            N=4,
            omega_true=1.0,
            num_seeds=5,
            maxiter=200,
            seed=42,
        )
        result.save_parquet(csv_p)
        print(f"[save] {csv_p}")

    result = MScalingResult.from_parquet(csv_p)
    plot_m_scaling(result, fig_p)
    print(f"[fig]  {fig_p}")


def generate_weighted_alpha_scan(force: bool = False) -> None:
    r"""\alpha_{xx} scan with weight re-optimisation, N=M=4."""
    csv_p = _parquet_path("alpha-scan-nm")
    fig_p = _fig_path("alpha-scan-nm")

    if csv_p.exists() and not force:
        print(f"[skip] {csv_p.name} exists (use --force to overwrite)")
        result = AlphaReoptResultNM.from_parquet(csv_p)
    else:
        print("[run]  Computing \u03b1_{xx} scan with re-optimisation...")
        alpha_vals = np.linspace(-2.0, 2.0, 21)
        result = run_alpha_scan_with_reoptimisation(
            alpha_name="xx",
            N=4,
            M=4,
            alpha_values=alpha_vals,
            omega_true=1.0,
            num_seeds=3,
            maxiter=100,
        )
        result.save_parquet(csv_p)
        print(f"[save] {csv_p}")

    result = AlphaReoptResultNM.from_parquet(csv_p)
    plot_weighted_alpha_scan(result, fig_p)
    print(f"[fig]  {fig_p}")


def generate_drive_decoupled_baseline(force: bool = False) -> None:
    """Experiment 1: Decoupled baseline verification."""
    csv_p = _parquet_path("drive-decoupled-baseline")
    fig_p = _fig_path("drive-decoupled-baseline")

    if csv_p.exists() and not force:
        print(f"[skip] {csv_p.name} exists (use --force to overwrite)")
        result = DriveDecoupledBaselineResult.from_parquet(csv_p)
    else:
        print("[run]  Computing drive decoupled baseline...")
        result = compute_drive_decoupled_baseline()
        result.save_parquet(csv_p)
        print(f"[save] {csv_p}")

    plot_drive_decoupled_baseline(result, fig_p)
    print(f"[fig]  {fig_p}")


def _run_drive_2d_slice(
    omega: float,
    slice_type: str,
    force: bool,
) -> None:
    """Run a 2D slice scan for a single \u03c9 value and generate CSV + SVG."""
    tag = f"drive-2d-slice-{slice_type}-azz-omega{omega}"
    csv_p = _parquet_path(tag)
    fig_p = _fig_path(tag)

    if csv_p.exists() and not force:
        print(f"  [skip] {csv_p.name} exists (use --force to overwrite)")
        result = Drive2DSliceResult.from_parquet(csv_p)
    else:
        print(f"  [run]  Computing ({slice_type}, a_zz) slice at \u03c9={omega}...")
        result = drive_2d_slice(
            omega=omega,
            slice_type=slice_type,
            n_drive=201,
            n_azz=201,
        )
        result.save_parquet(csv_p)
        print(f"  [save] {csv_p}")

    plot_drive_2d_slice_heatmap(result, fig_p)
    print(f"  [fig]  {fig_p}")


def generate_drive_2d_slice_ax_azz(force: bool = False) -> None:
    """Experiment 2a: 2D slice scans over (a_x, a_zz) at all \u03c9 values."""
    print(f"[run]  (a_x, a_zz) slice at {DRIVE_OMEGA_VALS}")
    for omega in DRIVE_OMEGA_VALS:
        _run_drive_2d_slice(omega, slice_type="ax", force=force)


def generate_drive_2d_slice_ay_azz(force: bool = False) -> None:
    """Experiment 2b: 2D slice scans over (a_y, a_zz) at all \u03c9 values."""
    print(f"[run]  (a_y, a_zz) slice at {DRIVE_OMEGA_VALS}")
    for omega in DRIVE_OMEGA_VALS:
        _run_drive_2d_slice(omega, slice_type="ay", force=force)


def generate_drive_random_search(force: bool = False) -> None:
    """Experiment 3: 4D random search at all \u03c9 values."""
    print(f"[run]  4D random search at {DRIVE_OMEGA_VALS}")
    for omega in DRIVE_OMEGA_VALS:
        tag = f"drive-random-search-omega{omega}"
        csv_p = _parquet_path(tag)
        fig_p = _fig_path(tag)

        if csv_p.exists() and not force:
            print(f"  [skip] {csv_p.name} exists (use --force to overwrite)")
            result = DriveRandomSearchResult.from_parquet(csv_p)
        else:
            print(
                f"  [run]  Running 4D random search at \u03c9={omega} (500 samples)..."
            )
            result = drive_random_search(
                omega=omega,
                n_samples=500,
                seed=42,
            )
            result.save_parquet(csv_p)
            print(f"  [save] {csv_p}")

        plot_drive_random_search_histogram(result, fig_p)
        print(f"  [fig]  {fig_p}")


def generate_drive_omega_scan(force: bool = False) -> None:
    """Experiments 4 & 5: \u03c9-scan with Nelder-Mead refinement."""
    csv_p = _parquet_path("drive-omega-scan")
    fig_p = _fig_path("drive-omega-scan")

    if csv_p.exists() and not force:
        print(f"[skip] {csv_p.name} exists (use --force to overwrite)")
        result = DriveOmegaScanResult.from_parquet(csv_p)
    else:
        print("[run]  Computing drive \u03c9-scan (may be slow)...")
        result = run_drive_omega_scan(
            omega_values=DRIVE_OMEGA_VALS,
            n_random=500,
            n_nm_refine=50,
            seed=42,
            maxiter=5000,
        )
        result.save_parquet(csv_p)
        print(f"[save] {csv_p}")

    plot_drive_omega_scan(result, fig_p)
    print(f"[fig]  {fig_p}")


def generate_drive_optimal_params(force: bool = False) -> None:
    """Optimal parameter evolution vs \u03c9."""
    csv_p = _parquet_path("drive-omega-scan")
    fig_p = _fig_path("drive-optimal-params")

    result = DriveOmegaScanResult.from_parquet(csv_p)
    plot_drive_optimal_params(result, fig_p)
    print(f"[fig]  {fig_p}")


# ============================================================================
# (D) CLI Entry Point
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate 2026-05-18 report figures and CSVs",
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
        help="Generate only one dataset, e.g. 'n-scaling'",
    )
    args = parser.parse_args()

    # Ensure per-date directories exist.
    (REPORTS_DIR / REPORT_DATE / "raw_data").mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / REPORT_DATE / "figures").mkdir(parents=True, exist_ok=True)

    tasks: dict[str, Callable[[bool], None]] = {
        "n-scaling": generate_n_scaling,
        "m-scaling": generate_m_scaling,
        "alpha-scan-nm": generate_weighted_alpha_scan,
        "drive-decoupled-baseline": generate_drive_decoupled_baseline,
        "drive-2d-slice-ax-azz": generate_drive_2d_slice_ax_azz,
        "drive-2d-slice-ay-azz": generate_drive_2d_slice_ay_azz,
        "drive-random-search": generate_drive_random_search,
        "drive-omega-scan": generate_drive_omega_scan,
        "drive-optimal-params": generate_drive_optimal_params,
    }

    if args.only:
        if args.only not in tasks:
            print(f"Unknown dataset '{args.only}'. Options: {list(tasks.keys())}")
            sys.exit(1)
        tasks[args.only](args.force)
    else:
        for name, func in tasks.items():
            print(f"\n=== {name} ===")
            func(args.force)

    print("\nDone.")


if __name__ == "__main__":
    main()
