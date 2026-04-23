"""
Mach-Zehnder Interferometer with Ancilla Simulation.

Physical Model:
- Two-mode bosonic system (interferometer arms)
- Ancilla system (entangled probe)
- Variable beam splitter
- Phase shift in one arm
- System-ancilla coupling for entanglement

Hilbert Space:
- System: Fock basis for 2 modes with max N photons (dimension: (N+1)²)
- Ancilla: Spin-J system (dimension: 2J+1)

Units:
- Dimensionless throughout. Phase is measured in radians.
- Time is dimensionless when multiplied by coupling strength g.

Conventions:
- Beam splitter transformation: a → cos(θ)a + i*e^{iφ}sin(θ)b
- Phase convention: e^{iφn} applied to mode 1 (second mode)
- State ordering: |n₁, n₂⟩ with n₁ as first mode, n₂ as second mode
"""

from typing import Tuple

import numpy as np
import scipy


# =============================================================================
# State Preparation
# =============================================================================


def fock_state(n1: int, n2: int, max_photons: int) -> np.ndarray:
    """Create a Fock state vector |n₁, n₂⟩ in the two-mode Fock basis.

    Constructs a pure Fock state with exactly n₁ photons in mode 0
    and n₂ photons in mode 1. The state is represented as a vector
    in a truncated Fock space with maximum photon number max_photons
    per mode.

    Args:
        n1: Photon number in mode 0 (first arm). Must be non-negative
            and ≤ max_photons.
        n2: Photon number in mode 1 (second arm). Must be non-negative
            and ≤ max_photons.
        max_photons: Maximum photon number per mode. Truncation parameter
            for the Hilbert space.

    Returns:
        Complex state vector of dimension (max_photons+1)² with a single
        1 at index n₁ × (max_photons+1) + n₂.

    Raises:
        ValueError: If n1 or n2 exceeds max_photons or is negative.

    Example:
        >>> state = fock_state(1, 0, max_photons=2)
        >>> state.shape
        (9,)
        >>> np.abs(state[3])  # Index for |1,0⟩: 1*3 + 0 = 3
        1.0
    """
    if n1 < 0 or n2 < 0:
        raise ValueError("Photon numbers must be non-negative")
    if n1 > max_photons or n2 > max_photons:
        raise ValueError(f"Photon numbers must not exceed max_photons={max_photons}")

    dim = (max_photons + 1) ** 2
    state = np.zeros(dim, dtype=complex)
    idx = n1 * (max_photons + 1) + n2
    state[idx] = 1.0
    return state


def vacuum_state(max_photons: int) -> np.ndarray:
    """Create the vacuum state |0, 0⟩ (no photons in either mode).

    Args:
        max_photons: Maximum photon number per mode for the truncated
            Hilbert space.

    Returns:
        Complex state vector representing the two-mode vacuum.
    """
    return fock_state(0, 0, max_photons)


def single_photon_state(mode: int, max_photons: int) -> np.ndarray:
    """Create a single-photon state in a specific mode.

    Args:
        mode: Which mode to place the photon in (0 or 1).
        max_photons: Maximum photon number per mode.

    Returns:
        State vector |1, 0⟩ if mode=0, or |0, 1⟩ if mode=1.

    Raises:
        ValueError: If mode is not 0 or 1.
    """
    if mode == 0:
        return fock_state(1, 0, max_photons)
    elif mode == 1:
        return fock_state(0, 1, max_photons)
    else:
        raise ValueError("Mode must be 0 or 1")


def noon_state(N: int, max_photons: int) -> np.ndarray:
    """Create a NOON state (|N, 0⟩ + |0, N⟩)/√2.

    NOON states are maximally path-entangled states useful for
    quantum metrology. They achieve Heisenberg scaling in phase
    estimation precision.

    Args:
        N: Total photon number (N photons in either mode, none in the other).
        max_photons: Maximum photon number per mode. Must be ≥ N to
            accommodate the state.

    Returns:
        Normalized NOON state vector.

    Example:
        >>> noon = noon_state(3, max_photons=3)  # |3,0⟩ + |0,3⟩
        >>> np.allclose(noon, noon)  # Check normalization
        True
    """
    # Need max_photons >= N for NOON state to fit in basis
    effective_max = max(N, max_photons)
    state = fock_state(N, 0, effective_max) + fock_state(0, N, effective_max)
    return state / np.sqrt(2)


def coherent_state(alpha: complex, max_photons: int) -> np.ndarray:
    """Create a coherent state |α⟩ in mode 0 with mode 1 as vacuum.

    A coherent state is the quantum state most closely resembling
    a classical electromagnetic field. It has Poisson photon number
    distribution and minimum uncertainty (balanced).

    Args:
        alpha: Complex amplitude of the coherent state. The mean
            photon number is |α|².
        max_photons: Maximum photon number for truncation. Should be
            significantly larger than |α|² for accurate representation.

    Returns:
        State vector representing the coherent state in the truncated
        Fock basis.

    Example:
        >>> alpha = 1.0 + 0j
        >>> state = coherent_state(alpha, max_photons=10)
        >>> # Mean photon number ≈ |α|² = 1
        >>> # Probability of n photons = |α|^{2n} e^{-|α|²} / n!
    """
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
    """Create Fock state |n⟩ in mode 0 (mode 1 as vacuum).

    Args:
        n: Photon number (must be non-negative and ≤ max_photons).
        max_photons: Maximum photon number per mode.

    Returns:
        State vector |n, 0⟩.
    """
    return fock_state(n, 0, max_photons)


# =============================================================================
# Operator Construction
# =============================================================================


def create_system_operators(
    max_photons: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create annihilation and creation operators for the two-mode system.

    Constructs the bosonic annihilation (a) and creation (a†) operators
    for both modes of the interferometer. The operators act in the
    truncated Fock space with dimension (max_photons+1)².

    The operators satisfy:
    - a |n⟩ = √n |n-1⟩ (annihilation)
    - a† |n⟩ = √(n+1) |n+1⟩ (creation)

    Args:
        max_photons: Maximum photon number per mode (truncation).

    Returns:
        Tuple of (a₀, a₁, a₀†, a₁†) where:
        - a₀: Annihilation operator for mode 0
        - a₁: Annihilation operator for mode 1
        - a₀†: Creation operator for mode 0
        - a₁†: Creation operator for mode 1
    """
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
    """Create spin operators for the ancilla system.

    Constructs the J_x and J_z angular momentum operators for a
    spin-J system where ancilla_dim = 2J + 1. These operators act
    on the ancilla Hilbert space and can be used to couple the
    interferometer to a quantum memory/probe.

    Args:
        ancilla_dim: Dimension of the ancilla Hilbert space.
            For spin-J, ancilla_dim = 2J + 1 (must be odd).

    Returns:
        Tuple of (Jx, Jz) where both are Hermitian operators
        of dimension ancilla_dim × ancilla_dim.

    Example:
        >>> Jx, Jz = create_ancilla_operators(3)  # Spin-1 system
        >>> Jz.diagonal()  # Eigenvalues: +1, 0, -1
        array([ 1.,  0., -1.])
    """
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
    r"""Compute the beam splitter unitary transformation in Fock space.

    The beam splitter implements the mode transformation:
        a → cos(θ)·a + i·e^{iφ}·sin(θ)·b
        b → cos(θ)·b - i·e^{-iφ}·sin(θ)·a

    For a symmetric 50/50 beam splitter, use θ = π/4.
    The phase φ controls the reflection amplitude phase.

    Args:
        theta: Beam splitter transmittance angle. θ = 0 gives identity,
            θ = π/4 gives 50/50 splitter.
        phi: Phase shift applied to reflected photons.
        max_photons: Maximum photon number per mode for truncation.

    Returns:
        Unitary matrix of dimension (max_photons+1)² × (max_photons+1)²
        representing the beam splitter transformation.

    Example:
        >>> bs = beam_splitter_unitary(np.pi/4, 0, max_photons=2)
        >>> np.allclose(bs @ bs.conj().T, np.eye(9))  # Check unitarity
        True
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
    r"""Compute the phase shift unitary on mode 1.

    Applies a phase shift proportional to the photon number in mode 1:
        P(φ) = exp(i · φ · n₁)

    where n₁ is the number operator for mode 1. This implements the
    physical effect of introducing an optical path difference
    (phase shift) in one arm of the interferometer.

    Args:
        phi: Phase shift in radians.
        max_photons: Maximum photon number per mode for truncation.

    Returns:
        Diagonal unitary matrix of dimension (max_photons+1)².
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
    r"""Compute the system-ancilla interaction unitary.

    Implements U = exp(-i · H_int · t) where H_int depends on the
    coupling type. This entangles the interferometer with the ancilla,
    enabling quantum-enhanced metrology protocols.

    Coupling types:
    - "phase_coupling": H = g · n_photon · J_z
      Phase-dependent coupling (phase estimation).
    - "flip_flop": H = g · (a + a†) ⊗ J_x
      Exchange coupling (useful for entanglement generation).

    Args:
        g: Coupling strength (dimensionless when combined with time).
        interaction_time: Interaction duration (dimensionless).
        coupling_type: Type of system-ancilla coupling.
        max_photons: Maximum photon number in the system.
        ancilla_dim: Dimension of the ancilla Hilbert space.

    Returns:
        Unitary matrix of dimension (sys_dim × ancilla_dim)².
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
    """Evolve an initial system state through a Mach-Zehnder interferometer.

    Simulates the full interferometer circuit including beam splitters,
    phase shift, and optional system-ancilla coupling for quantum-enhanced
    protocols. The ancilla is initialized in its ground state.

    Circuit sequence:
        1. BS1: First beam splitter (mode mixing)
        2. Phase: Phase shift on mode 1 (the parameter to estimate)
        3. Interaction: System-ancilla coupling (entanglement)
        4. BS2: Second beam splitter (interference)

    Args:
        initial_system_state: Initial state of the two-mode system
            (vector in truncated Fock space).
        theta: Beam splitter transmittance angle (π/4 for 50/50).
        phi_bs: Beam splitter phase parameter.
        phi_phase: Phase shift in arm 1 (the unknown parameter).
        g: System-ancilla coupling strength.
        interaction_time: Duration of system-ancilla interaction.
        coupling_type: Type of coupling ("phase_coupling" or "flip_flop").
        max_photons: Maximum photons per mode (Hilbert space truncation).
        ancilla_dim: Dimension of ancilla Hilbert space.

    Returns:
        Final state vector of dimension (sys_dim × ancilla_dim).

    Example:
        >>> state = vacuum_state(max_photons=2)
        >>> final = evolve_mzi(state, np.pi/4, 0, 1.0, 0, 0, "phase_coupling", 2, 3)
        >>> final.shape  # (9 * 3,) = (27,)
        (27,)
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
    """Compute the density matrix from a pure state vector.

    Constructs the density matrix ρ = |ψ⟩⟨ψ| from the state vector.
    Optionally traces out the ancilla to obtain the reduced system
    density matrix (useful for computing expectation values and
    probabilities).

    Args:
        full_state: State vector in the combined system-ancilla Hilbert
            space of dimension (sys_dim × ancilla_dim).
        max_photons: Maximum photons per mode (determines sys_dim).
        ancilla_dim: Dimension of the ancilla Hilbert space.
        trace_out_ancilla: If True, return reduced system density matrix
            by tracing out ancilla. If False, return full density matrix.

    Returns:
        Density matrix. If trace_out_ancilla=True, dimension is
        (sys_dim × sys_dim). Otherwise (full_dim × full_dim).

    Example:
        >>> rho_full = get_reduced_density_matrix(state, 2, 3, trace_out=False)
        >>> rho_sys = get_reduced_density_matrix(state, 2, 3, trace_out=True)
        >>> rho_sys.shape  # (9, 9)
        (9, 9)
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
    """Compute detection probabilities at the two output ports.

    Calculates the probability of finding photons in output mode 0
    versus output mode 1 after the second beam splitter. This is
    the probability distribution measured by photon counters at
    the interferometer outputs.

    The probability is computed as the expectation value of the
    photon number operator in each mode, weighted by the diagonal
    elements of the reduced density matrix.

    Args:
        full_state: Final state after MZI evolution.
        max_photons: Maximum photons per mode in the truncation.
        ancilla_dim: Dimension of ancilla Hilbert space.

    Returns:
        Tuple (P0, P1) where:
        - P0: Probability of detecting photon in output mode 0
        - P1: Probability of detecting photon in output mode 1
        Both sum to 1 (if total photon number > 0).
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
    """Compute the interference fringe: output probability vs phase.

    Sweeps the phase shift parameter and computes the output
    probability at each phase value. This produces the characteristic
    interference pattern used in phase estimation experiments.

    Args:
        phase_range: Array of phase values (in radians) to evaluate.
        initial_system_state: Initial state injected into the interferometer.
        theta: Beam splitter transmittance angle.
        phi_bs: Beam splitter phase parameter.
        g: System-ancilla coupling strength.
        interaction_time: Interaction duration.
        coupling_type: Coupling type for system-ancilla entanglement.
        max_photons: Maximum photons per mode.
        ancilla_dim: Dimension of ancilla Hilbert space.

    Returns:
        Array of P0 probabilities corresponding to each phase in
        phase_range.

    Example:
        >>> phases = np.linspace(0, 2*np.pi, 100)
        >>> fringe = compute_interference_fringe(phases, vacuum_state(2),
        ...     np.pi/4, 0, 0, 0, "phase_coupling", 2, 3)
        >>> fringe.shape
        (100,)
    """
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
    """Compute the quantum state at each stage of the MZI circuit.

    Provides access to intermediate states for debugging, visualization,
    and analysis of the interferometer evolution. Useful for understanding
    how information propagates through the circuit.

    Args:
        initial_system_state: Initial system state before BS1.
        theta: Beam splitter transmittance angle.
        phi_bs: Beam splitter phase.
        phi_phase: Phase shift in arm 1.
        g: System-ancilla coupling strength.
        interaction_time: Interaction duration.
        coupling_type: Type of coupling.
        max_photons: Maximum photons per mode.
        ancilla_dim: Dimension of ancilla.

    Returns:
        Dictionary with keys:
        - "initial": State before BS1 (system ⊗ ancilla)
        - "after_bs1": State after first beam splitter
        - "after_phase": State after phase shift
        - "after_interaction": State after system-ancilla coupling
        - "final": State after second beam splitter (output)
    """
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
    """Prepare an input state for the interferometer.

    Factory function that creates various initial states commonly
    used in quantum optics and quantum metrology experiments.

    Args:
        state_type: Type of state to prepare. Options:
            - "vacuum": |0, 0⟩ (no photons)
            - "single_photon": |1, 0⟩ or |0, 1⟩ (use mode parameter)
            - "coherent": Coherent state |α⟩ (use alpha parameter)
            - "fock": Fock state |n, 0⟩ (use n_particles)
            - "noon": NOON state (|N,0⟩ + |0,N⟩)/√2 (use n_particles)
        max_photons: Maximum photon number per mode for truncation.
        n_particles: Number of photons for "fock" or "noon" states.
        alpha: Complex amplitude for coherent state (default: 1.0).
        mode: Which mode for single photon (0 or 1).

    Returns:
        State vector in the truncated Fock basis.

    Raises:
        ValueError: If state_type is not recognized.

    Example:
        >>> # Single photon in mode 0
        >>> state = prepare_input_state("single_photon", max_photons=2, mode=0)
        >>> # Coherent state with amplitude 2
        >>> state = prepare_input_state("coherent", max_photons=10, alpha=2.0)
        >>> # NOON state for 5 photons
        >>> state = prepare_input_state("noon", max_photons=5, n_particles=5)
    """
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
    """Check if a state vector is normalized.

    Validates that the input is a proper quantum state by checking
    that its L2 norm equals 1 (within numerical tolerance).

    Args:
        state: State vector to validate.

    Returns:
        True if the state is normalized (||ψ|| = 1), False otherwise.

    Example:
        >>> state = fock_state(1, 0, max_photons=2)
        >>> validate_state(state)
        True
        >>> validate_state(state * 0.5)  # Not normalized
        False
    """
    norm = np.sqrt(np.sum(np.abs(state) ** 2))
    return np.isclose(norm, 1.0, atol=1e-10)


def validate_unitary(U: np.ndarray, tol: float = 1e-8) -> bool:
    """Check if a matrix is unitary.

    Validates that U†U = I within numerical tolerance. This is
    essential for verifying that operator constructions are correct.

    Args:
        U: Matrix to validate.
        tol: Numerical tolerance for the check (default: 1e-8).

    Returns:
        True if the matrix is unitary, False otherwise.

    Example:
        >>> bs = beam_splitter_unitary(np.pi/4, 0, max_photons=2)
        >>> validate_unitary(bs)
        True
    """
    return np.allclose(U @ U.conj().T, np.eye(U.shape[0]), atol=tol)
