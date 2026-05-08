"""
MZI Embedding for Hybrid Oscillator-Spin System.

Implements the MZI readout protocol for the hybrid system:
1. Two-mode embedding: ρ_hybrid ⊗ |0⟩⟨0| (vacuum in second mode)
2. MZI evolution: BS → phase shift φ → BS
3. QFI computation with G = n₁ ⊗ I_spin

Hilbert Space:
- Original hybrid: dim = 2(N+1)
- After embedding: dim = 2(N+1) × (N+1) = 2(N+1)²
  (hybrid ⊗ vacuum mode)
- Phase generator acts on mode 1 only: G = n₁ ⊗ I_osc2 ⊗ I_spin

Units:
- Dimensionless throughout (ℏ = 1)
- Phase φ in radians.
"""

from typing import Tuple
import numpy as np
import scipy

from .wigner import wigner_function_single


# =============================================================================
# Two-Mode Embedding
# =============================================================================

def embed_hybrid_in_mzi(
    hybrid_state: np.ndarray,
    N: int,
) -> np.ndarray:
    """Embed hybrid state into two-mode MZI space.

    Creates: ρ_2mode = |ψ⟩_hybrid ⊗ |0⟩_vacuum
    where |0⟩_vacuum is the vacuum state in mode 2.

    State ordering:
    - Mode 1: hybrid oscillator (N+1 Fock states)
    - Mode 2: vacuum mode (N+1 Fock states)
    - Spin: 2 states

    Total dimension: 2 × (N+1)²
    Index = (n1*(N+1) + n2) * 2 + s

    Args:
        hybrid_state: State vector of shape (2(N+1),).
        N: Maximum photon number (truncation).

    Returns:
        Embedded state vector of shape (2(N+1)²,).
    """
    dim_osc = N + 1
    dim_hybrid = 2 * dim_osc
    dim_mzi = 2 * (dim_osc ** 2)  # hybrid ⊗ mode2

    if hybrid_state.shape != (dim_hybrid,):
        raise ValueError(f"hybrid_state must have shape ({dim_hybrid},), got {hybrid_state.shape}")

    embedded = np.zeros(dim_mzi, dtype=complex)

    # Embed as: |n1⟩_mode1 ⊗ |0⟩_mode2 ⊗ |σ⟩_spin
    # Index in embedded space: (n1*(N+1) + 0) * 2 + s = n1*(N+1)*2 + s
    for n1 in range(dim_osc):
        for s in range(2):  # spin state
            hybrid_idx = n1 * 2 + s
            mzi_idx = n1 * dim_osc * 2 + s  # n2=0, so (n1*(N+1)+0)*2+s
            embedded[mzi_idx] = hybrid_state[hybrid_idx]

    return embedded


# =============================================================================
# MZI Operators
# =============================================================================

def mzi_beam_splitter(N: int, theta: float = np.pi / 4) -> np.ndarray:
    """Construct beam splitter unitary for modes 1 and 2.

    Uses the generator-based approach: U = exp(-iθ G) where
    G = i(a1†a2 - a1a2†) is the beam splitter generator.

    This approach guarantees unitarity.

    Args:
        N: Maximum photon number.
        theta: Beam splitter angle (π/4 = 50/50).

    Returns:
        Unitary of shape (2(N+1)², 2(N+1)²).
    """
    dim_osc = N + 1
    dim_modes = dim_osc ** 2

    # Build annihilation operators for mode 1 and mode 2
    # Mode 1: a1 ⊗ I_2
    a1 = np.zeros((dim_modes, dim_modes), dtype=complex)
    for n1 in range(dim_osc):
        for n2 in range(dim_osc):
            idx = n1 * dim_osc + n2
            if n1 >0:
                a1[idx - dim_osc, idx] = np.sqrt(n1)  # a1|n1,n2⟩ = √n1|n1-1,n2⟩

    # Mode 2: I_1 ⊗ a2
    a2 = np.zeros((dim_modes, dim_modes), dtype=complex)
    for n1 in range(dim_osc):
        for n2 in range(dim_osc):
            idx = n1 * dim_osc + n2
            if n2 >0:
                a2[idx - 1, idx] = np.sqrt(n2)  # a2|n1,n2⟩ = √n2|n1,n2-1⟩

    # Beam splitter generator: G = i(a1†a2 - a1a2†)
    a1_dag = a1.conj().T
    a2_dag = a2.conj().T

    # Compute unitary: U = exp(-iθ G) = exp(θ * (a1†a2 - a1a2†))
    G = 1j * (a1_dag @ a2 - a1 @ a2_dag)
    bs_modes = scipy.linalg.expm(-1j * theta * G)

    # Embed with spin identity
    bs_full = np.kron(bs_modes, np.eye(2, dtype=complex))

    return bs_full


def mzi_phase_shift(N: int, phi: float) -> np.ndarray:
    """Construct phase shift unitary on mode 1.

    U_phase = exp(i φ n₁) ⊗ I_mode2 ⊗ I_spin

    Args:
        N: Maximum photon number.
        phi: Phase shift in radians.

    Returns:
        Unitary of shape (2(N+1)², 2(N+1)²).
    """
    dim_osc = N + 1
    dim_modes = dim_osc ** 2

    phase_op = np.zeros((dim_modes, dim_modes), dtype=complex)

    for n1 in range(dim_osc):
        for n2 in range(dim_osc):
            idx = n1 * dim_osc + n2
            phase_op[idx, idx] = np.exp(1j * phi * n1)

    # Embed with spin identity
    return np.kron(phase_op, np.eye(2, dtype=complex))


def mzi_phase_generator(N: int) -> np.ndarray:
    """Construct phase generator G = n₁ ⊗ I_mode2 ⊗ I_spin.

    Used for QFI computation.

    Args:
        N: Maximum photon number.

    Returns:
        Generator matrix of shape (2(N+1)², 2(N+1)²).
    """
    dim_osc = N + 1
    dim_modes = dim_osc ** 2

    # n₁ in mode space: diagonal with value n1
    n1_op = np.zeros((dim_modes, dim_modes), dtype=complex)
    for n1 in range(dim_osc):
        for n2 in range(dim_osc):
            idx = n1 * dim_osc + n2
            n1_op[idx, idx] = n1

    # Embed with spin identity
    return np.kron(n1_op, np.eye(2, dtype=complex))


# =============================================================================
# MZI Evolution
# =============================================================================

def evolve_hybrid_mzi(
    hybrid_state: np.ndarray,
    N: int,
    phi: float,
    theta: float = np.pi / 4,
) -> np.ndarray:
    """Evolve hybrid state through MZI.

    Sequence: embed → BS1 → phase shift → BS2

    Args:
        hybrid_state: Input hybrid state of shape (2(N+1),).
        N: Maximum photon number.
        phi: Phase shift in mode 1 (unknown parameter).
        theta: Beam splitter angle (default π/4 = 50/50).

    Returns:
        Output state vector of shape (2(N+1)²,).
    """
    # Embed into MZI space
    state = embed_hybrid_in_mzi(hybrid_state, N)

    # BS1
    bs = mzi_beam_splitter(N, theta)
    state = bs @ state

    # Phase shift
    ps = mzi_phase_shift(N, phi)
    state = ps @ state

    # BS2
    state = bs @ state

    return state


# =============================================================================
# Output Probabilities
# =============================================================================

def mzi_output_probabilities(
    final_state: np.ndarray,
    N: int,
) -> np.ndarray:
    """Compute output probabilities P(n1, n2, s) from MZI output.

    Args:
        final_state: Output state vector of shape (2(N+1)²,).
        N: Maximum photon number.

    Returns:
        Array of probabilities for each (n1, n2, s) configuration.
        Sum should be 1.
    """
    probs = np.abs(final_state) ** 2
    return probs


def mzi_marginal_photon_probs(
    final_state: np.ndarray,
    N: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute marginal photon number probabilities P(n1), P(n2).

    Args:
        final_state: Output state vector.
        N: Maximum photon number.

    Returns:
        Tuple (P1, P2) where P1[n1] = P(n1) summed over n2 and spin.
    """
    dim_osc = N + 1

    probs = np.abs(final_state) ** 2

    P1 = np.zeros(dim_osc, dtype=float)
    P2 = np.zeros(dim_osc, dtype=float)

    for n1 in range(dim_osc):
        for n2 in range(dim_osc):
            idx_base = (n1 * dim_osc + n2) * 2
            # Sum over spin (2 components)
            total = np.sum(probs[idx_base : idx_base + 2])
            P1[n1] += total
            P2[n2] += total

    return P1, P2


# =============================================================================
# QFI Computation
# =============================================================================

def qfi_hybrid_mzi(
    hybrid_state: np.ndarray,
    N: int,
) -> float:
    """Compute QFI for MZI phase estimation with hybrid input.

    For pure state |ψ(φ)⟩, the QFI is:
        F_Q = 4 Var(G) = 4(⟨G²⟩ - ⟨G⟩²)
    where G = n₁ ⊗ I_mode2 ⊗ I_spin.

    Args:
        hybrid_state: Input hybrid state (before MZI).
        N: Maximum photon number.

    Returns:
        Quantum Fisher Information F_Q.
    """
    # Embed state into MZI space
    embedded = embed_hybrid_in_mzi(hybrid_state, N)

    # Phase generator
    G = mzi_phase_generator(N)

    # Compute expectation values
    G_psi = G @ embedded
    G2_psi = G @ G_psi

    exp_G = np.vdot(embedded, G_psi).real
    exp_G2 = np.vdot(embedded, G2_psi).real

    var_G = exp_G2 - exp_G ** 2
    var_G = max(0.0, var_G)  # Numerical safety

    return 4.0 * var_G


# =============================================================================
# Wigner Function for Hybrid State
# =============================================================================

def extract_oscillator_density(hybrid_state: np.ndarray, N: int) -> np.ndarray:
    """Extract oscillator density matrix from hybrid state (trace out spin).

    Args:
        hybrid_state: State vector of shape (2(N+1),).
        N: Maximum photon number.

    Returns:
        Density matrix of shape (N+1, N+1).
    """
    dim_osc = N + 1

    # Convert to density matrix
    rho_hybrid = np.outer(hybrid_state, hybrid_state.conj())

    # Reshape to (dim_osc, 2, dim_osc, 2) and trace over spin
    rho_reshaped = rho_hybrid.reshape(dim_osc, 2, dim_osc, 2)
    rho_osc = np.trace(rho_reshaped, axis1=1, axis2=3)

    return rho_osc


def compute_wigner_for_state(
    hybrid_state: np.ndarray,
    N: int,
    x_max: float = 5.0,
    n_points: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Wigner function for oscillator part of hybrid state.

    Args:
        hybrid_state: Hybrid state vector.
        N: Maximum photon number.
        x_max: Range for x and p axes.
        n_points: Number of grid points per axis.

    Returns:
        Tuple (X, P, W) where X and P are 1D arrays, W is 2D array.
    """
    # Extract oscillator density matrix
    rho_osc = extract_oscillator_density(hybrid_state, N)

    # Create quadrature grid
    x = np.linspace(-x_max, x_max, n_points)
    p = np.linspace(-x_max, x_max, n_points)

    # Compute Wigner
    W = wigner_function_single(rho_osc, x, p)

    return x, p, W
