"""
Multi-particle MZI operator construction in the Dicke basis.

Provides building blocks for constructing operators in the combined S⊗A
(system–ancilla) space used in multi-MZI protocols:

- ``dicke_single_operators(N)``:  returns the single-subsystem
  (N+1)×(N+1) Dicke-basis matrices J_z, J_x, J_y.
- ``embed_combined_operators(N)``: returns the full (N+1)²×(N+1)²
  Kronecker-product operators J_k^S = J_k ⊗ I and J_k^A = I ⊗ J_k
  for k ∈ {x, y, z}.
- ``single_bs_unitary(N, T)``: cached (N+1)×(N+1) beam-splitter unitary
  U = exp(-i T J_x).
- ``dual_bs_unitary(N, T)``: (N+1)²×(N+1)² dual beam-splitter unitary
  U = U_single ⊗ U_single.

Usage::

    from src.physics.multi_mzi import (
        build_hold_hamiltonian,
        compute_reduced_expectation_and_variance,
        dicke_single_operators,
        dual_bs_unitary,
        embed_combined_operators,
        evolve_circuit,
        hold_unitary,
        single_bs_unitary,
    )

    single = dicke_single_operators(4)          # keys: Jz, Jx, Jy
    combined = embed_combined_operators(4)      # keys: Jz_S, Jz_A, Jx_S, Jx_A, Jy_S, Jy_A, I
    U_bs = single_bs_unitary(4)                 # cached 5×5 unitary
    U_dual = dual_bs_unitary(4)                 # 25×25 dual unitary
"""

from __future__ import annotations

from functools import cache

import numpy as np
from scipy.linalg import expm

from src.physics.dicke_basis import jx_operator, jy_operator, jz_operator


def dicke_single_operators(N: int) -> dict[str, np.ndarray]:
    """Build single-subsystem Dicke-basis operators for *N* particles.

    Returns (N+1)×(N+1) matrices for J_z, J_x, J_y in the Dicke basis
    with *m* descending from +N/2 to -N/2.

    Args:
        N: Number of particles (dim = N+1).

    Returns:
        Dict with keys ``'Jz'``, ``'Jx'``, ``'Jy'``.
    """
    return {
        "Jz": jz_operator(N),
        "Jx": jx_operator(N),
        "Jy": jy_operator(N),
    }


def embed_combined_operators(N: int) -> dict[str, np.ndarray]:
    """Build (N+1)²×(N+1)² operators in the combined S⊗A space.

    J_k^S = J_k ⊗ I_{N+1}
    J_k^A = I_{N+1} ⊗ J_k

    The basis ordering is |m_S, m_A⟩ with both *m* descending from +J to -J.
    System index rows, ancilla index columns.

    Args:
        N: Particle number per subsystem (dim = N+1 per subsystem).

    Returns:
        Dict with keys ``'Jz_S'``, ``'Jz_A'``, ``'Jx_S'``, ``'Jx_A'``,
        ``'Jy_S'``, ``'Jy_A'``, and ``'I'`` (full identity).
    """
    single = dicke_single_operators(N)
    eye = np.eye(N + 1, dtype=float)
    return {
        "Jz_S": np.kron(single["Jz"], eye),
        "Jz_A": np.kron(eye, single["Jz"]),
        "Jx_S": np.kron(single["Jx"], eye),
        "Jx_A": np.kron(eye, single["Jx"]),
        "Jy_S": np.kron(single["Jy"], eye),
        "Jy_A": np.kron(eye, single["Jy"]),
        "I": np.kron(eye, eye),
    }


def build_hold_hamiltonian(
    N: int,
    theta: float,
    alpha_xx: float,
    ops: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Build the total holding Hamiltonian in the combined S⊗A space.

    H = θ (J_z^S + J_z^A) + α_xx J_x^S J_x^A

    Args:
        N: Particle number per subsystem.
        theta: Unknown phase rate.
        alpha_xx: XX coupling strength.
        ops: Pre-computed embedded operators. If None, built fresh.

    Returns:
        (N+1)² × (N+1)² Hermitian Hamiltonian matrix.
    """
    if ops is None:
        ops = embed_combined_operators(N)
    dim = (N + 1) ** 2
    H = np.zeros((dim, dim), dtype=complex)
    H += theta * (ops["Jz_S"] + ops["Jz_A"])
    if alpha_xx != 0.0:
        # J_x^S J_x^A = (J_x ⊗ I)(I ⊗ J_x) = J_x ⊗ J_x
        H += alpha_xx * (ops["Jx_S"] @ ops["Jx_A"])
    return 0.5 * (H + H.conj().T)


def hold_unitary(
    N: int,
    T_H: float,
    theta: float,
    alpha_xx: float,
    ops: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Holding-time unitary in the combined S⊗A space.

    U_hold(T_H) = exp(-i T_H H)

    Args:
        N: Particle number per subsystem.
        T_H: Holding time.
        theta: Unknown phase rate.
        alpha_xx: XX coupling strength.
        ops: Pre-computed embedded operators.

    Returns:
        (N+1)² × (N+1)² unitary matrix.
    """
    H = build_hold_hamiltonian(N, theta, alpha_xx, ops)
    U = expm(-1j * T_H * H)
    dim = (N + 1) ** 2
    assert np.allclose(U @ U.conj().T, np.eye(dim), atol=1e-12)
    return U


@cache
def single_bs_unitary(N: int, T: float = np.pi / 2.0) -> np.ndarray:
    """Single-subsystem 50/50 beam-splitter unitary (cached by N and T).

    U_BS = exp(-i T J_x)

    Results are cached by (N, T) to avoid repeated matrix exponentiation.

    Args:
        N: Particle number (dim = N+1).
        T: Beam-splitter angle (default π/2 for 50/50).

    Returns:
        (N+1) × (N+1) unitary matrix.
    """
    Jx = jx_operator(N)
    U = expm(-1j * T * Jx)
    assert np.allclose(U @ U.conj().T, np.eye(N + 1), atol=1e-12)
    return U


def dual_bs_unitary(N: int, T: float = np.pi / 2.0) -> np.ndarray:
    """Dual beam-splitter unitary: BS on both S and A.

    U_BS = exp(-i T J_x) ⊗ exp(-i T J_x)

    Args:
        N: Particle number per subsystem.
        T: Beam-splitter angle (default π/2 for 50/50).

    Returns:
        (N+1)² × (N+1)² unitary matrix.
    """
    U_single = single_bs_unitary(N, T)
    return np.kron(U_single, U_single)


def compute_reduced_expectation_and_variance(
    psi: np.ndarray,
    N: int,
    meas_op: np.ndarray,
) -> tuple[float, float]:
    """Compute ⟨J_z^S⟩ and Var(J_z^S) after tracing out the ancilla.

    The final pure state |ψ⟩ of length (N+1)² is reshaped into an
    (N+1) × (N+1) matrix (rows = system index, columns = ancilla index).
    The reduced density matrix is ρ_S = Ψ Ψ^†.

    Args:
        psi: Final pure state vector (length (N+1)²).
        N: Particle number per subsystem.
        meas_op: (N+1) × (N+1) measurement operator (e.g., J_z).

    Returns:
        Tuple (expectation, variance).
    """
    # Reshape: rows = system, columns = ancilla
    psi_mat = psi.reshape(N + 1, N + 1)  # (N+1, N+1)
    rho_S = psi_mat @ psi_mat.conj().T  # (N+1, N+1)

    # Verify trace preservation
    trace = float(np.real(np.trace(rho_S)))
    assert np.isclose(trace, 1.0, atol=1e-12), f"Reduced trace = {trace} != 1"

    # Expectation and variance
    exp_val = float(np.real(np.trace(rho_S @ meas_op)))
    op_sq = meas_op @ meas_op
    exp_sq = float(np.real(np.trace(rho_S @ op_sq)))
    raw_var = exp_sq - exp_val**2

    # Clamp negative variance near zero (numerical round-off)
    if raw_var < 0 and raw_var > -1e-12:
        raw_var = 0.0
    assert raw_var >= -1e-12, f"Unphysical negative variance: {raw_var:.2e}"

    return float(exp_val), float(max(0.0, raw_var))


def evolve_circuit(
    N: int,
    psi0: np.ndarray,
    theta: float,
    alpha_xx: float,
    ops: dict[str, np.ndarray],
    T_BS: float = np.pi / 2.0,
    T_H: float = 10.0,
) -> np.ndarray:
    """Run the full dual-MZI circuit.

    |ψ_final⟩ = U_BS · U_hold(T_H) · U_BS · |ψ₀⟩

    where U_BS acts on both S and A (dual MZI).

    Args:
        N: Particle number per subsystem.
        psi0: Initial state vector (length (N+1)²).
        theta: Unknown phase rate.
        alpha_xx: XX coupling strength.
        ops: Embedded operators.
        T_BS: Beam-splitter angle (default π/2).
        T_H: Holding time (default 10).

    Returns:
        Final state vector (length (N+1)²).
    """
    assert np.isclose(np.linalg.norm(psi0), 1.0), "Initial state must be normalised"
    U_bs = dual_bs_unitary(N, T_BS)
    psi = U_bs @ psi0
    psi = hold_unitary(N, T_H, theta, alpha_xx, ops) @ psi
    psi = U_bs @ psi
    assert np.isclose(np.linalg.norm(psi), 1.0), "Final state must be normalised"
    return psi
