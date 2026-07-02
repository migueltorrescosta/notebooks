"""
Multi-particle MZI operator construction in the Dicke basis.

Provides building blocks for constructing operators in the combined S⊗A
(system–ancilla) space used in multi-MZI protocols:

- ``dicke_single_operators(N)``:  returns the single-subsystem
  (N+1)×(N+1) Dicke-basis matrices J_z, J_x, J_y.
- ``embed_combined_operators(N)``: returns the full (N+1)²×(N+1)²
  Kronecker-product operators J_k^S = J_k ⊗ I and J_k^A = I ⊗ J_k
  for k ∈ {x, y, z}.
- ``single_bs_unitary(N, T_BS)``: cached (N+1)×(N+1) beam-splitter unitary
  U = exp(-i T_BS J_x).
- ``dual_bs_unitary(N, T_BS)``: (N+1)²×(N+1)² dual beam-splitter unitary
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

import numpy as np
from scipy.linalg import expm

from src.physics.dicke_basis import jx_operator, jy_operator, jz_operator
from src.utils.enums import OperatorBasis


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
        "Jz": jz_operator(N, basis=OperatorBasis.DICKE),
        "Jx": jx_operator(N, basis=OperatorBasis.DICKE),
        "Jy": jy_operator(N, basis=OperatorBasis.DICKE),
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
    omega: float,
    alpha_xx: float,
    ops: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Build the total holding Hamiltonian in the combined S⊗A space.

    H = ω (J_z^S + J_z^A) + α_xx J_x^S J_x^A

    Args:
        N: Particle number per subsystem.
        omega: Unknown phase rate.
        alpha_xx: XX coupling strength.
        ops: Pre-computed embedded operators. If None, built fresh.

    Returns:
        (N+1)² × (N+1)² Hermitian Hamiltonian matrix.
    """
    if ops is None:
        ops = embed_combined_operators(N)
    dim = (N + 1) ** 2
    H = np.zeros((dim, dim), dtype=complex)
    H += omega * (ops["Jz_S"] + ops["Jz_A"])
    if alpha_xx != 0.0:
        # J_x^S J_x^A = (J_x ⊗ I)(I ⊗ J_x) = J_x ⊗ J_x
        H += alpha_xx * (ops["Jx_S"] @ ops["Jx_A"])
    return 0.5 * (H + H.conj().T)


def hold_unitary_dicke(
    N: int,
    t_hold: float,
    omega: float,
    alpha_xx: float,
    ops: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Holding-time unitary in the combined S⊗A space (Dicke basis).

    U_hold(t_hold) = exp(-i t_hold H)

    Args:
        N: Particle number per subsystem.
        t_hold: Holding time.
        omega: Unknown phase rate.
        alpha_xx: XX coupling strength.
        ops: Pre-computed embedded operators.

    Returns:
        (N+1)² × (N+1)² unitary matrix.
    """
    H = build_hold_hamiltonian(N, omega, alpha_xx, ops)
    U = expm(-1j * t_hold * H)
    dim = (N + 1) ** 2
    assert np.allclose(U @ U.conj().T, np.eye(dim), atol=1e-12)
    return U


def single_bs_unitary(N: int, T_BS: float = np.pi / 2.0) -> np.ndarray:
    """Single-subsystem 50/50 beam-splitter unitary (cached by N and T_BS).

    U_BS = exp(-i T_BS J_x)

    Results are cached by (N, T_BS) to avoid repeated matrix exponentiation.

    Args:
        N: Particle number (dim = N+1).
        T_BS: Beam-splitter angle (default π/2 for 50/50).

    Returns:
        (N+1) × (N+1) unitary matrix.
    """
    from src.physics.beam_splitter import bs_dicke  # fmt: skip

    return bs_dicke(N, T_BS)


def dual_bs_unitary(N: int, T_BS: float = np.pi / 2.0) -> np.ndarray:
    """Dual beam-splitter unitary: BS on both S and A.

    U_BS = exp(-i T_BS J_x) ⊗ exp(-i T_BS J_x)

    Args:
        N: Particle number per subsystem.
        T_BS: Beam-splitter angle (default π/2 for 50/50).

    Returns:
        (N+1)² × (N+1)² unitary matrix.
    """
    return np.kron(single_bs_unitary(N, T_BS), single_bs_unitary(N, T_BS))


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
    omega: float,
    alpha_xx: float,
    ops: dict[str, np.ndarray],
    T_BS: float = np.pi / 2.0,
    t_hold: float = 10.0,
) -> np.ndarray:
    """Run the full dual-MZI circuit.

    |ψ_final⟩ = U_BS · U_hold(t_hold) · U_BS · |ψ₀⟩

    where U_BS acts on both S and A (dual MZI).

    Args:
        N: Particle number per subsystem.
        psi0: Initial state vector (length (N+1)²).
        omega: Unknown phase rate.
        alpha_xx: XX coupling strength.
        ops: Embedded operators.
        T_BS: Beam-splitter angle (default π/2).
        t_hold: Holding time (default 10).

    Returns:
        Final state vector (length (N+1)²).
    """
    assert np.isclose(np.linalg.norm(psi0), 1.0), "Initial state must be normalised"
    U_bs = dual_bs_unitary(N, T_BS)
    psi = U_bs @ psi0
    psi = hold_unitary_dicke(N, t_hold, omega, alpha_xx, ops) @ psi
    psi = U_bs @ psi
    assert np.isclose(np.linalg.norm(psi), 1.0), "Final state must be normalised"
    return psi


def compute_multi_particle_sensitivity(
    N: int,
    psi0: np.ndarray,
    omega_true: float,
    alpha_xx: float,
    ops: dict[str, np.ndarray],
    meas_op: np.ndarray | None = None,
    fd_step: float = 1e-6,
    T_BS: float = np.pi / 2.0,
    t_hold: float = 10.0,
) -> tuple[float, float, float, float]:
    """Compute the error-propagation sensitivity Δω for multi-particle MZI.

    Δω = √Var(J_z^S) / |∂⟨J_z^S⟩/∂ω|

    Also returns ⟨J_z^S⟩, Var(J_z^S), and ∂⟨J_z^S⟩/∂ω at omega_true.

    Args:
        N: Particle number per subsystem.
        psi0: Initial state vector.
        omega_true: True phase rate.
        alpha_xx: XX coupling strength.
        ops: Embedded operators.
        meas_op: (N+1)×(N+1) measurement operator (default = J_z single).
        fd_step: Central finite-difference step size.
        T_BS: Beam-splitter angle.
        t_hold: Holding time.

    Returns:
        Tuple (delta_omega, expectation, variance, derivative).
        Returns (inf, exp, var, 0.0) if derivative is zero.
    """
    if meas_op is None:
        meas_op = jz_operator(N, basis=OperatorBasis.DICKE)

    # Evaluate at omega_true
    psi = evolve_circuit(N, psi0, omega_true, alpha_xx, ops, T_BS, t_hold)
    exp_val, var_val = compute_reduced_expectation_and_variance(psi, N, meas_op)

    # Central finite difference for ∂⟨J_z^S⟩/∂ω
    psi_plus = evolve_circuit(
        N, psi0, omega_true + fd_step, alpha_xx, ops, T_BS, t_hold
    )
    psi_minus = evolve_circuit(
        N, psi0, omega_true - fd_step, alpha_xx, ops, T_BS, t_hold
    )

    exp_plus, _ = compute_reduced_expectation_and_variance(psi_plus, N, meas_op)
    exp_minus, _ = compute_reduced_expectation_and_variance(psi_minus, N, meas_op)
    d_exp = (exp_plus - exp_minus) / (2.0 * fd_step)

    if abs(d_exp) < 1e-12:
        return float("inf"), exp_val, var_val, 0.0

    delta_omega = float(np.sqrt(var_val) / abs(d_exp))
    return delta_omega, exp_val, var_val, d_exp
