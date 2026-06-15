"""
Shared N-particle ω-modulated drive infrastructure.

Provides operator construction, circuit evolution, and sensitivity computation
for an N-particle system (J_S = N/2) with a single-qubit ancilla (J_A = 1/2),
under an ω-modulated ancilla drive with Ising interaction.

Total Hilbert space: H_S ⊗ H_A with dimension 2(N+1).
Basis ordering: {|m_S⟩_S ⊗ |0⟩_A, ..., |m_S⟩_S ⊗ |1⟩_A}
where m_S descends from +J_S to -J_S, and |0⟩_A = |1,0⟩, |1⟩_A = |0,1⟩.

Circuit: BS_S → Hold → BS_S → measure.

Used by reports #20260611 and #20260613.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from src.analysis.ancilla_optimization import compute_expectation_and_variance
from src.physics.beam_splitter import bs_dicke
from src.physics.dicke_basis import jx_operator, jy_operator, jz_operator
from src.utils.constants import I_2, J_X, J_Y, J_Z
from src.utils.enums import OperatorBasis

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
        Also includes 'I_S' (N+1)×(N+1) identity and 'I_full' (full-space identity).
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
        "I_S": I_S,
        "I_full": np.eye(d_tot, dtype=complex),
    }

    for key in ("Jz_S", "Jx_S", "Jy_S", "Jz_A", "Jx_A", "Jy_A"):
        assert ops[key].shape == (d_tot, d_tot), (
            f"{key} has shape {ops[key].shape}, expected ({d_tot}, {d_tot})"
        )
        assert np.allclose(ops[key], ops[key].conj().T, atol=1e-12), (
            f"{key} is not Hermitian"
        )

    comm = ops["Jz_S"] @ ops["Jx_S"] - ops["Jx_S"] @ ops["Jz_S"]
    expected = 1j * ops["Jy_S"]
    assert np.allclose(comm, expected, atol=1e-10), (
        f"[J_z^S, J_x^S] = i J_y^S violated for N={N}"
    )

    return ops


def build_n_particle_system_only_bs_unitary(
    N: int, T_bs: float = np.pi / 2.0
) -> np.ndarray:
    """System-only beam-splitter unitary in the N-particle space.

    U_BS_S = exp(-i T_bs J_x) ⊗ I_2

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

    H_int = a_zz J_z^S ⊗ J_z^A

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

    This is the first basis vector: [1, 0, ..., 0]ᵀ of length 2(N+1).

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
    t_hold: float,
    omega: float,
    a_x: float,
    a_y: float,
    a_z: float,
    a_zz: float,
    ops: dict[str, np.ndarray],
) -> np.ndarray:
    """Holding-time unitary for the N-particle ω-modulated protocol.

    U_hold(t_hold) = exp(-i t_hold H)
    where H = ω J_z^S + ω(a_x J_x^A + a_y J_y^A + a_z J_z^A) + a_zz J_z^S ⊗ J_z^A.

    Args:
        N: Number of system particles.
        t_hold: Holding-time strength.
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
    U = expm(-1j * t_hold * H)
    d_tot = 2 * (N + 1)
    I_full = np.eye(d_tot, dtype=complex)
    assert np.allclose(U @ U.conj().T, I_full, atol=1e-12), (
        f"Hold unitary not unitary for N={N}, t_hold={t_hold}, ω={omega}"
    )
    return U


def evolve_n_particle_circuit(
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
    """Run the full N-particle ω-modulated ancilla MZI circuit.

    |ψ_final⟩ = U_BS_S · U_hold(t_hold) · U_BS_S · |ψ₀⟩

    Args:
        N: Number of system particles.
        psi0: Initial state vector (must be normalised).
        T_bs: Beam-splitter duration.
        t_hold: Holding-time strength.
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
    psi = n_particle_hold_unitary(N, t_hold, omega, a_x, a_y, a_z, a_zz, ops) @ psi
    psi = U_bs @ psi
    assert np.isclose(np.linalg.norm(psi), 1.0), f"Final state not normalised for N={N}"
    return psi


def compute_n_particle_sensitivity(
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
    fd_step: float = 1e-6,
    meas_op: np.ndarray | None = None,
) -> float:
    """Compute the error-propagation sensitivity Δω for the N-particle system.

    Δω = sqrt(Var(O)) / |∂⟨O⟩/∂ω|

    where O is the measurement operator (default: J_z^S).

    The central finite-difference captures the full ω-dependence
    (both ω J_z^S and ω-modulated ancilla drive channels).

    Args:
        N: Number of system particles.
        psi0: Initial state vector.
        T_bs: Beam-splitter duration.
        t_hold: Holding-time strength.
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

    psi = evolve_n_particle_circuit(
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


def compute_n_particle_decoupled_baseline(
    N: int,
    omega_true: float = 1.0,
) -> float:
    """Compute the decoupled baseline sensitivity Δω for N particles.

    At (a_x = a_y = a_z = a_zz = 0), the circuit reduces to a standard
    N-particle MZI with CSS input, giving Δω = 1/(√N × t_hold).

    Args:
        N: Number of system particles.
        omega_true: Phase rate value (does not affect SQL).

    Returns:
        Δω at the decoupled configuration.
    """
    ops = build_n_particle_operators(N)
    psi0 = n_particle_initial_state(N)
    return compute_n_particle_sensitivity(
        N,
        psi0,
        np.pi / 2.0,
        10.0,
        omega_true,
        0.0,
        0.0,
        0.0,
        0.0,
        ops,
    )
