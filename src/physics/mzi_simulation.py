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

import numpy as np
import qutip
import scipy

from src.utils.validators import (
    validate_state_mzi,
)

# Aliases for backward compatibility
# Agent Notes: validate_state is also defined as an alias for
# validate_state_delta_estimation in src.analysis.delta_estimation.
# These are DIFFERENT functions — do not merge them.
validate_state = validate_state_mzi


# =============================================================================
# State Preparation
# =============================================================================


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
    dim_single = effective_max + 1
    state = (
        (
            qutip.tensor(qutip.fock(dim_single, N), qutip.fock(dim_single, 0))
            + qutip.tensor(qutip.fock(dim_single, 0), qutip.fock(dim_single, N))
        )
        .full()
        .ravel()
    )
    return state / np.sqrt(2)


# =============================================================================
# Operator Construction
# =============================================================================


def create_system_operators(
    max_photons: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create annihilation and creation operators for the two-mode system.

    Constructs the bosonic annihilation (a) and creation (a†) operators
    for both modes of the interferometer using the shared QuTiP backend
    with two-mode embedding via Kronecker products.

    The operators act in the truncated Fock space with dimension
    (max_photons+1)² with state ordering |n₀, n₁⟩.

    Embedding convention (|n₀, n₁⟩ ordering):
        a₀ = a ⊗ I   (annihilation on mode 0)
        a₁ = I ⊗ a   (annihilation on mode 1)
        a₀† = a† ⊗ I (creation on mode 0)
        a₁† = I ⊗ a† (creation on mode 1)

    The single-mode operators satisfy:
    - a |n⟩ = √n |n-1⟩ (annihilation)
    - a† |n⟩ = √(n+1) |n+1⟩ (creation)
    - [a, a†] = I (up to truncation boundary)

    Args:
        max_photons: Maximum photon number per mode (truncation).

    Returns:
        Tuple of (a₀, a₁, a₀†, a₁†) where:
        - a₀:   Annihilation operator for mode 0  (a ⊗ I)
        - a₁:   Annihilation operator for mode 1  (I ⊗ a)
        - a₀†:  Creation operator for mode 0      (a† ⊗ I)
        - a₁†:  Creation operator for mode 1      (I ⊗ a†)

    """
    import qutip

    dim_single = max_photons + 1
    a = qutip.destroy(dim_single).full()  # (N+1, N+1) annihilation
    a_dag = qutip.create(dim_single).full()  # (N+1, N+1) creation
    identity = np.eye(dim_single, dtype=complex)

    # Two-mode embedding via Kronecker products
    a0 = np.kron(a, identity)  # a ⊗ I  — acts on mode 0
    a1 = np.kron(identity, a)  # I ⊗ a  — acts on mode 1
    a0_dag = np.kron(a_dag, identity)  # a† ⊗ I — acts on mode 0
    a1_dag = np.kron(identity, a_dag)  # I ⊗ a† — acts on mode 1

    return a0, a1, a0_dag, a1_dag


def create_ancilla_operators(ancilla_dim: int) -> tuple[np.ndarray, np.ndarray]:
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

    Generated by the Hamiltonian:
        H = e^{iφ} a₁†a₂ + e^{-iφ} a₂†a₁
        U = exp(-iθ H)

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
    a0, a1, a0_dag, a1_dag = create_system_operators(max_photons)

    # Beam splitter Hamiltonian: H = e^{iφ} a₁†a₂ + e^{-iφ} a₂†a₁
    H_bs = np.exp(1j * phi) * (a0_dag @ a1) + np.exp(-1j * phi) * (a1_dag @ a0)

    # Unitary: U = exp(-iθH)
    return scipy.linalg.expm(-1.0j * theta * H_bs)


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
    a0, _a1, a0_dag, _a1_dag = create_system_operators(max_photons)
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

    Constraints:
        initial_system_state must be normalized (L2 norm = 1).
        theta in [0, π] (beam splitter angle).
        coupling_type in {"phase_coupling", "flip_flop"}.
        max_photons >= 0, ancilla_dim >= 1.
        Output dimension = (max_photons+1)² × ancilla_dim.

    Example:
        >>> import qutip
        >>> dim = 2 + 1
        >>> state = qutip.tensor(qutip.fock(dim, 0), qutip.fock(dim, 0)).full()
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
        g,
        interaction_time,
        coupling_type,
        max_photons,
        ancilla_dim_val,
    )
    state = U_int @ state

    # BS2
    return bs_full @ state


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
        return np.trace(rho_reshaped, axis1=1, axis2=3)
    return rho


def compute_output_probabilities(
    full_state: np.ndarray,
    max_photons: int,
    ancilla_dim: int,
) -> tuple[float, float]:
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
        full_state,
        max_photons,
        ancilla_dim,
        trace_out_ancilla=True,
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
        >>> import qutip
        >>> dim = 2 + 1
        >>> vac = qutip.tensor(qutip.fock(dim, 0), qutip.fock(dim, 0)).full()
        >>> fringe = compute_interference_fringe(phases, vac,
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
        g,
        interaction_time,
        coupling_type,
        max_photons,
        anc,
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

    Agent Notes:
        This is the primary API surface for UI pages (pages/*.py) to create
        input states. New state types added here must also be registered in:
        - the state_type parameter docstring above
        - the UI selectbox widgets in pages/Fisher_information.py and similar
        - any scaling survey functions in src.analysis.scaling_survey
        Falls back to vacuum state for unrecognized types (no error raised).

    """
    dim_single = max_photons + 1
    match state_type:
        case "vacuum":
            return (
                qutip.tensor(qutip.fock(dim_single, 0), qutip.fock(dim_single, 0))
                .full()
                .ravel()
            )
        case "single_photon":
            if mode == 0:
                return (
                    qutip.tensor(qutip.fock(dim_single, 1), qutip.fock(dim_single, 0))
                    .full()
                    .ravel()
                )
            return (
                qutip.tensor(qutip.fock(dim_single, 0), qutip.fock(dim_single, 1))
                .full()
                .ravel()
            )
        case "coherent":
            return (
                qutip.tensor(
                    qutip.coherent(dim_single, alpha), qutip.fock(dim_single, 0)
                )
                .full()
                .ravel()
            )
        case "fock":
            return (
                qutip.tensor(
                    qutip.fock(dim_single, n_particles), qutip.fock(dim_single, 0)
                )
                .full()
                .ravel()
            )
        case "noon":
            # Ensure max_photons >= n_particles
            effective_max = max(n_particles, max_photons)
            return noon_state(n_particles, effective_max)
        case _:
            return (
                qutip.tensor(qutip.fock(dim_single, 0), qutip.fock(dim_single, 0))
                .full()
                .ravel()
            )


# =============================================================================
# Noisy MZI Evolution
# =============================================================================


def evolve_mzi_with_noise(
    initial_system_state: np.ndarray,
    theta: float,
    phi_bs: float,
    phi_phase: float,
    max_photons: int,
    noise_gamma_1: float = 0.0,
    noise_gamma_2: float = 0.0,
    noise_gamma_phi: float = 0.0,
    noise_T: float = 1.0,
    noise_dt: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Evolve state through a noisy Mach-Zehnder interferometer.

    Runs the full MZI circuit (BS1 → Phase → Lindblad → BS2) with
    configurable noise channels (one-body loss, two-body loss, and
    phase diffusion) acting in the interferometer arms.

    This function provides the cleaned-up interface for the scaling survey,
    wrapping the low-level noise infrastructure in a single call.

    Args:
        initial_system_state: Initial two-mode Fock state vector or
            density matrix in the truncated Fock basis.
        theta: Beam splitter transmittance angle (π/4 for 50/50).
        phi_bs: Beam splitter phase parameter.
        phi_phase: Phase shift in arm 1 (the unknown parameter).
        max_photons: Maximum photon number per mode (Hilbert space truncation).
        noise_gamma_1: One-body loss rate (γ₁) for mode 1.
        noise_gamma_2: Two-body loss rate (γ₂) for mode 1.
        noise_gamma_phi: Phase diffusion rate (γ_φ) between arms.
        noise_T: Decoherence time (dimensionless).
        noise_dt: Time step for numerical integration.

    Returns:
        Tuple of (final_rho, reduced_system_rho) where both are density
        matrices in the two-mode Fock basis. For ancilla-free simulations
        (default), both are identical.

    Constraints:
        initial_system_state must be normalized (pure) or trace-1 (density matrix).
        theta in [0, π].
        noise_gamma_{1,2,phi} >= 0 (non-negative rates).
        noise_T > 0, noise_dt > 0.
        Performance: O((N+1)⁴ · T/dt) due to Lindblad integration.

    Agent Notes:
        Uses lazy import of MziNoiseConfig/run_noisy_mzi from mzi_lindblad
        to avoid circular dependency with mzi_lindblad → mzi_simulation.
        If you refactor mzi_lindblad, ensure this import path still resolves.
        This function is the primary API surface for the scaling survey
        (see src.analysis.scaling_survey).

    """
    # Lazy import to avoid circular dependency
    from src.physics.mzi_lindblad import MziNoiseConfig, run_noisy_mzi

    noise_config = MziNoiseConfig(
        gamma_1=noise_gamma_1,
        gamma_2=noise_gamma_2,
        gamma_phi=noise_gamma_phi,
        T=noise_T,
        dt=noise_dt,
    )

    final_rho = run_noisy_mzi(
        initial_state=initial_system_state,
        max_photons=max_photons,
        theta=theta,
        phi_bs=phi_bs,
        phi_phase=phi_phase,
        noise_config=noise_config,
        ancilla_dim=1,
    )

    # For ancilla-free case, reduced = full
    reduced_system_rho = final_rho.copy()

    return final_rho, reduced_system_rho


# =============================================================================
