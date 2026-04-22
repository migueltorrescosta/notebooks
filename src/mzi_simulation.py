"""
Mach-Zehnder Interferometer with Ancilla Simulation

Physical Model:
- Two-mode bosonic system (interferometer arms)
- Ancilla system (entangled probe)
- Variable beam splitter
- Phase shift in one arm
- System-ancilla coupling for entanglement

Hilbert Space:
- System: Fock basis for 2 modes with max N photons (dimension: (N+1)²)
- Ancilla: Spin-J system (dimension: 2J+1)
"""

from typing import Tuple

import numpy as np
import scipy


# =============================================================================
# State Preparation
# =============================================================================


def fock_state(n1: int, n2: int, max_photons: int) -> np.ndarray:
    """Create a Fock state vector |n1, n2> in the two-mode Fock basis."""
    dim = (max_photons + 1) ** 2
    state = np.zeros(dim, dtype=complex)
    idx = n1 * (max_photons + 1) + n2
    state[idx] = 1.0
    return state


def vacuum_state(max_photons: int) -> np.ndarray:
    """Create vacuum state |0,0>."""
    return fock_state(0, 0, max_photons)


def single_photon_state(mode: int, max_photons: int) -> np.ndarray:
    """Create single photon state |1,0> or |0,1>."""
    if mode == 0:
        return fock_state(1, 0, max_photons)
    else:
        return fock_state(0, 1, max_photons)


def noon_state(N: int, max_photons: int) -> np.ndarray:
    """Create NOON state (|N,0> + |0,N>)/sqrt(2)."""
    # Need max_photons >= N for NOON state to fit in basis
    effective_max = max(N, max_photons)
    state = fock_state(N, 0, effective_max) + fock_state(0, N, effective_max)
    return state / np.sqrt(2)


def coherent_state(alpha: complex, max_photons: int) -> np.ndarray:
    """Create coherent state |alpha> in one mode (other mode vacuum)."""
    state = np.zeros((max_photons + 1) ** 2, dtype=complex)
    for n in range(max_photons + 1):
        amplitude = (
            alpha**n
            / np.sqrt(scipy.special.factorial(n))
            * np.exp(-(abs(alpha) ** 2) / 2)
        )
        state[n * (max_photons + 1)] = amplitude
    return state


def fock_state_n(n: int, max_photons: int) -> np.ndarray:
    """Create Fock state |n> in mode 0 (mode 1 vacuum)."""
    return fock_state(n, 0, max_photons)


# =============================================================================
# Operator Construction
# =============================================================================


def create_system_operators(
    max_photons: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create annihilation and creation operators for two-mode system."""
    dim = (max_photons + 1) ** 2

    a0 = np.zeros((dim, dim), dtype=complex)
    a0_dag = np.zeros((dim, dim), dtype=complex)
    a1 = np.zeros((dim, dim), dtype=complex)
    a1_dag = np.zeros((dim, dim), dtype=complex)

    for n1 in range(max_photons + 1):
        for n2 in range(max_photons + 1):
            idx = n1 * (max_photons + 1) + n2

            # a0 |n1, n2> = sqrt(n1) |n1-1, n2>
            if n1 > 0:
                new_idx = (n1 - 1) * (max_photons + 1) + n2
                a0[idx, new_idx] = np.sqrt(n1)
                a0_dag[new_idx, idx] = np.sqrt(n1)

            # a1 |n1, n2> = sqrt(n2) |n1, n2-1>
            if n2 > 0:
                new_idx = n1 * (max_photons + 1) + (n2 - 1)
                a1[idx, new_idx] = np.sqrt(n2)
                a1_dag[new_idx, idx] = np.sqrt(n2)

    return a0, a1, a0_dag, a1_dag


def create_ancilla_operators(ancilla_dim: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create spin operators for ancilla (dimension = 2J+1)."""
    j = (ancilla_dim - 1) / 2

    # J_z operator (diagonal in computational basis)
    jz = np.zeros((ancilla_dim, ancilla_dim), dtype=complex)
    for m in range(ancilla_dim):
        magnetic_number = j - m
        jz[m, m] = magnetic_number

    # J_+ and J_- operators
    j_plus = np.zeros((ancilla_dim, ancilla_dim), dtype=complex)
    j_minus = np.zeros((ancilla_dim, ancilla_dim), dtype=complex)
    for m in range(ancilla_dim - 1):
        # J_+ |j, m> = sqrt(j(j+1) - m(m+1)) |j, m+1>
        val = np.sqrt(j * (j + 1) - (j - 1 - m) * (j - m))
        j_plus[m + 1, m] = val
        j_minus[m, m + 1] = val

    # J_x = (J_+ + J_-) / 2
    jx = (j_plus + j_minus) / 2

    return jx, jz


def beam_splitter_unitary(theta: float, phi: float, max_photons: int) -> np.ndarray:
    r"""
    Beam splitter unitary in Fock space.

    Transforms mode operators:
    a -> cos(θ)*a + i*sin(θ)*e^{i*φ}*b
    b -> cos(θ)*b - i*sin(θ)*e^{-i*φ}*a

    This creates a symmetric 50/50 beam splitter for θ=π/4.
    """
    dim = (max_photons + 1) ** 2
    bs = np.zeros((dim, dim), dtype=complex)

    for n1 in range(max_photons + 1):
        for n2 in range(max_photons + 1):
            idx_in = n1 * (max_photons + 1) + n2

            # m = photons moving from mode 0 -> 1
            # k = photons moving from mode 1 -> 0
            for m in range(n1 + 1):
                for k in range(n2 + 1):
                    n1_out = n1 - m + k
                    n2_out = n2 - k + m

                    if n1_out <= max_photons and n2_out <= max_photons:
                        count = scipy.special.comb(n1, m) * scipy.special.comb(n2, k)
                        # Symmetric form: i^{m+k} accounts for the symmetric transformation
                        amplitude = (
                            count
                            * (np.cos(theta) ** (n1 + n2 - m - k))
                            * (np.sin(theta) ** (m + k))
                            * (1j ** (m + k))
                            * np.exp(1j * k * phi)
                        )
                        idx_out = n1_out * (max_photons + 1) + n2_out
                        bs[idx_out, idx_in] = amplitude

    return bs


def phase_shift_unitary(phi: float, max_photons: int) -> np.ndarray:
    """
    Phase shift unitary on mode 1.
    P(phi) = exp(i * phi * n1)
    """
    dim = (max_photons + 1) ** 2
    phase_op = np.zeros((dim, dim), dtype=complex)

    for n1 in range(max_photons + 1):
        for n2 in range(max_photons + 1):
            idx = n1 * (max_photons + 1) + n2
            phase_op[idx, idx] = np.exp(1j * phi * n2)

    return phase_op


def system_ancilla_interaction_unitary(
    g: float,
    interaction_time: float,
    coupling_type: str,
    max_photons: int,
    ancilla_dim: int,
) -> np.ndarray:
    """
    System-ancilla interaction unitary: U = exp(-i * H_int * t)

    Types:
    - "phase_coupling": H = g * n_photon * J_z (phase-dependent)
    - "flip_flop": H = g * (a + a^dagger) * J_x (exchange)
    """
    a0, a1, a0_dag, a1_dag = create_system_operators(max_photons)
    jx, jz = create_ancilla_operators(ancilla_dim)

    n_photon = a0_dag @ a0  # Number operator in mode 0

    match coupling_type:
        case "phase_coupling":
            H_int = g * np.kron(n_photon, jz)
        case "flip_flop":
            # H = g * (a0 + a0^\dagger) \otimes Jx
            H_int = g * (np.kron(a0 + a0_dag, jx))
        case _:
            H_int = g * np.kron(n_photon, jz)

    return scipy.linalg.expm(-1j * interaction_time * H_int)


# =============================================================================
# State Evolution
# =============================================================================


def evolve_mzi(
    initial_system_state: np.ndarray,
    theta: float,
    phi_bs: float,
    phi_phase: float,
    g: float,
    interaction_time: float,
    coupling_type: str,
    max_photons: int,
    ancilla_dim: int,
) -> np.ndarray:
    """
    Evolve initial state through MZI with ancilla.

    Circuit:
    1. BS1: Beam splitter 1
    2. Phase shift in arm 1
    3. System-ancilla interaction
    4. BS2: Beam splitter 2
    """
    ancilla_dim_val = ancilla_dim

    # Initial ancilla state (ground state |0>)
    ancilla_0 = np.zeros(ancilla_dim_val, dtype=complex)
    ancilla_0[0] = 1.0

    # Full initial state |psi> ⊗ |0>
    full_state = np.kron(initial_system_state, ancilla_0)

    # BS1 (on system, identity on ancilla)
    bs = beam_splitter_unitary(theta, phi_bs, max_photons)
    bs_full = np.kron(bs, np.eye(ancilla_dim_val, dtype=complex))
    state = bs_full @ full_state

    # Phase shift (on system, identity on ancilla)
    phase = phase_shift_unitary(phi_phase, max_photons)
    phase_full = np.kron(phase, np.eye(ancilla_dim_val, dtype=complex))
    state = phase_full @ state

    # System-ancilla interaction
    U_int = system_ancilla_interaction_unitary(
        g, interaction_time, coupling_type, max_photons, ancilla_dim_val
    )
    state = U_int @ state

    # BS2
    state = bs_full @ state

    return state


def get_reduced_density_matrix(
    full_state: np.ndarray,
    max_photons: int,
    ancilla_dim: int,
    trace_out_ancilla: bool = True,
) -> np.ndarray:
    """
    Get density matrix.

    If trace_out_ancilla=True: returns reduced system density matrix
    If trace_out_ancilla=False: returns full system+ancilla density matrix
    """
    sys_dim = (max_photons + 1) ** 2

    # Density matrix
    rho = np.outer(full_state, full_state.conj())

    if trace_out_ancilla:
        # Reshape and trace
        rho_reshaped = rho.reshape(sys_dim, ancilla_dim, sys_dim, ancilla_dim)
        rho_sys = np.trace(rho_reshaped, axis1=1, axis2=3)
        return rho_sys
    else:
        return rho


def compute_output_probabilities(
    full_state: np.ndarray, max_photons: int, ancilla_dim: int
) -> Tuple[float, float]:
    """
    Compute probability of finding photon in output mode 0 vs mode 1.
    """
    rho_sys = get_reduced_density_matrix(
        full_state, max_photons, ancilla_dim, trace_out_ancilla=True
    )

    # Probability = sum over diagonal elements * photon number in that mode
    P0 = 0.0
    P1 = 0.0

    for n1 in range(max_photons + 1):
        for n2 in range(max_photons + 1):
            idx = n1 * (max_photons + 1) + n2
            prob = np.real(rho_sys[idx, idx])
            P0 += prob * n1
            P1 += prob * n2

    total = P0 + P1
    if total > 0:
        return P0 / total, P1 / total
    else:
        return 0.5, 0.5


def compute_interference_fringe(
    phase_range: np.ndarray,
    initial_system_state: np.ndarray,
    theta: float,
    phi_bs: float,
    g: float,
    interaction_time: float,
    coupling_type: str,
    max_photons: int,
    ancilla_dim: int,
) -> np.ndarray:
    """Compute interference pattern: P(out0) vs phase."""
    probs = []
    for phi in phase_range:
        state = evolve_mzi(
            initial_system_state,
            theta,
            phi_bs,
            phi,
            g,
            interaction_time,
            coupling_type,
            max_photons,
            ancilla_dim,
        )
        p0, _ = compute_output_probabilities(state, max_photons, ancilla_dim)
        probs.append(p0)
    return np.array(probs)


def compute_all_stage_states(
    initial_system_state: np.ndarray,
    theta: float,
    phi_bs: float,
    phi_phase: float,
    g: float,
    interaction_time: float,
    coupling_type: str,
    max_photons: int,
    ancilla_dim: int,
) -> dict:
    """Compute states at each stage of MZI."""
    anc = ancilla_dim

    # Initial ancilla
    ancilla_0 = np.zeros(anc, dtype=complex)
    ancilla_0[0] = 1.0
    full_initial = np.kron(initial_system_state, ancilla_0)

    # After BS1
    bs = beam_splitter_unitary(theta, phi_bs, max_photons)
    bs_full = np.kron(bs, np.eye(anc, dtype=complex))
    state_bs1 = bs_full @ full_initial

    # After phase
    phase = phase_shift_unitary(phi_phase, max_photons)
    phase_full = np.kron(phase, np.eye(anc, dtype=complex))
    state_phase = phase_full @ state_bs1

    # After interaction
    U_int = system_ancilla_interaction_unitary(
        g, interaction_time, coupling_type, max_photons, anc
    )
    state_int = U_int @ state_phase

    # After BS2 (final)
    state_final = bs_full @ state_int

    return {
        "initial": full_initial,
        "after_bs1": state_bs1,
        "after_phase": state_phase,
        "after_interaction": state_int,
        "final": state_final,
    }


# =============================================================================
# Input State Preparation
# =============================================================================


def prepare_input_state(
    state_type: str,
    max_photons: int,
    n_particles: int = 1,
    alpha: complex = 1.0 + 0j,
    mode: int = 0,
) -> np.ndarray:
    """Prepare input state based on type."""
    match state_type:
        case "vacuum":
            return vacuum_state(max_photons)
        case "single_photon":
            return single_photon_state(mode, max_photons)
        case "coherent":
            return coherent_state(alpha, max_photons)
        case "fock":
            return fock_state_n(n_particles, max_photons)
        case "noon":
            # Ensure max_photons >= n_particles
            effective_max = max(n_particles, max_photons)
            return noon_state(n_particles, effective_max)
        case _:
            return vacuum_state(max_photons)


# =============================================================================
# Validation
# =============================================================================


def validate_state(state: np.ndarray) -> bool:
    """Validate that state is normalized."""
    norm = np.sqrt(np.sum(np.abs(state) ** 2))
    return np.isclose(norm, 1.0, atol=1e-10)


def validate_unitary(U: np.ndarray, tol: float = 1e-8) -> bool:
    """Validate that matrix is unitary."""
    return np.allclose(U @ U.conj().T, np.eye(U.shape[0]), atol=tol)
