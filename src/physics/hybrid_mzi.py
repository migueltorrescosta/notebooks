"""
MZI Embedding for Hybrid Oscillator-Spin System.

Implements the MZI readout protocol for the hybrid system:
1. Two-mode embedding: ρ_hybrid ⊗ |0⟩⟨0| (vacuum in second mode)
2. MZI evolution: BS → phase shift φ → BS
3. QFI computation with G = n₁ ⊗ I_spin

Physical Model:
- The hybrid system (oscillator + spin) is embedded as the first mode
  of a two-mode interferometer
- A vacuum state is appended as the second mode to enable MZI readout
- MZI protocol: BS → phase encoding on mode 1 → BS → measurement
- QFI computed with phase generator G = n₁ (number in mode 1)

Hilbert Space:
- Original hybrid: dim = 2(N+1)
- After embedding: dim = 2(N+1) × (N+1) = 2(N+1)²
  (hybrid ⊗ vacuum mode)
- Phase generator acts on mode 1 only: G = n₁ ⊗ I_osc2 ⊗ I_spin

Units:
- Dimensionless throughout (ℏ = 1)
- Phase φ in radians

Conventions:
- Beam splitter: 50:50 symmetric (θ = π/4, φ = 0)
- Phase shift on mode 1: exp(i · φ · n₁)
- State ordering after embedding: |n_hyb⟩_osc ⊗ |σ⟩_spin ⊗ |n⟩_vac
- QFI convention: F_Q = 4 · Var(G) for pure states
  (see src.analysis.fisher_information)

Functions:
- ``qfi_hybrid_mzi`` — compute QFI for MZI phase estimation
- ``extract_oscillator_density`` — trace out spin from hybrid state
- ``embed_hybrid_in_mzi`` — embed hybrid state into two-mode MZI space
- ``mzi_beam_splitter`` — beam splitter in the embedded two-mode space
- ``mzi_phase_shift`` — phase shift unitary on mode 1
- ``mzi_phase_generator`` — generator n₁ ⊗ I for QFI computation
- ``evolve_hybrid_mzi`` — full MZI circuit evolution
- ``mzi_output_probabilities`` — P(n1, n2, s) from MZI output
- ``mzi_marginal_photon_probs`` — marginal P(n1), P(n2)
- ``wigner_function_single`` — Wigner function for single-mode density matrix
- ``wigner_from_hybrid_state`` — Wigner from hybrid state via spin trace
- ``compute_wigner_for_state`` — Wigner function for oscillator part of hybrid state
- ``wigner_is_negative`` — check if Wigner function has negative values
"""

import numpy as np
import qutip
import scipy

# =============================================================================
# QFI Computation
# =============================================================================


def qfi_hybrid_mzi(
    hybrid_state: np.ndarray,
    N: int,
) -> float:
    """Compute QFI for MZI phase estimation with hybrid input.

    For the MZI, the phase shift exp(-iφ n₁) is applied between two
    beam splitters.  The correct generator is n₁ acting on the state
    *after* the first beam splitter.  For a 50:50 beam splitter and
    a pure input |ψ⟩|0⟩, this reduces to an analytical expression
    in terms of the input state's photon number statistics:

        F_Q = ⟨n²⟩ - ⟨n⟩² + ⟨n⟩ = Var(n) + ⟨n⟩

    where ⟨n⟩, ⟨n²⟩ are computed on the oscillator after tracing out
    the spin.  This formula is exact for the |ψ⟩|0⟩ ⊗ |spin⟩ setup
    and avoids constructing the BS matrix explicitly.

    Args:
        hybrid_state: Input hybrid state vector of shape (2(N+1),).
        N: Maximum photon number.

    Returns:
        Quantum Fisher Information F_Q.

    """
    # Extract oscillator density matrix (trace out spin)
    rho_osc = extract_oscillator_density(hybrid_state, N)

    # Photon number operator in oscillator space (diagonal)
    n_op = np.diag(np.arange(N + 1, dtype=complex))

    # Compute first and second moments of n
    mean_n = np.real(np.trace(rho_osc @ n_op))
    mean_n2 = np.real(np.trace(rho_osc @ n_op @ n_op))

    var_n = mean_n2 - mean_n**2
    var_n = max(0.0, var_n)  # Numerical safety

    # F_Q = Var(n) + ⟨n⟩  (derived from 4·Var(n₁) after BS₁)
    return var_n + mean_n


# =============================================================================
# Oscillator Density Extraction
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
    return np.trace(rho_reshaped, axis1=1, axis2=3)


# =============================================================================
# MZI Embedding and Circuit
# =============================================================================


def embed_hybrid_in_mzi(
    hybrid_state: np.ndarray,
    N: int,
) -> np.ndarray:
    """Embed hybrid state into two-mode MZI space.

    Creates: ρ_2mode = |ψ⟩_hybrid ⊗ |0⟩_vacuum
    where |0⟩_vacuum is the vacuum state in mode 2.

    Accepts both:
    - Pure state vector (1D) — returns embedded vector of shape (dim_mzi,).
    - Density matrix (2D) — returns embedded matrix of shape (dim_mzi, dim_mzi).

    State ordering:
    - Mode 1: hybrid oscillator (N+1 Fock states)
    - Mode 2: vacuum mode (N+1 Fock states)
    - Spin: 2 states

    Total dimension: 2 × (N+1)²
    Index = (n1*(N+1) + n2) * 2 + s

    Args:
        hybrid_state: State vector of shape (2(N+1),) or density matrix of
            shape (2(N+1), 2(N+1)).
        N: Maximum photon number (truncation).

    Returns:
        Embedded state vector of shape (2(N+1)²,) if input is 1D, or
        embedded density matrix of shape (2(N+1)², 2(N+1)²) if input is 2D.

    """
    dim_osc = N + 1
    dim_hybrid = 2 * dim_osc
    dim_mzi = 2 * (dim_osc**2)  # hybrid ⊗ mode2

    # --- Pure state path ---
    if hybrid_state.ndim == 1:
        if hybrid_state.shape != (dim_hybrid,):
            raise ValueError(
                f"hybrid_state must have shape ({dim_hybrid},), "
                f"got {hybrid_state.shape}",
            )

        embedded = np.zeros(dim_mzi, dtype=complex)
        # Embed as: |n1⟩_mode1 ⊗ |0⟩_mode2 ⊗ |σ⟩_spin
        # Index in embedded space: (n1*(N+1) + 0) * 2 + s = n1*(N+1)*2 + s
        for n1 in range(dim_osc):
            for s in range(2):  # spin state
                hybrid_idx = n1 * 2 + s
                mzi_idx = n1 * dim_osc * 2 + s  # n2=0
                embedded[mzi_idx] = hybrid_state[hybrid_idx]
        return embedded

    # --- Density matrix path ---
    if hybrid_state.ndim == 2:
        if hybrid_state.shape != (dim_hybrid, dim_hybrid):
            raise ValueError(
                f"hybrid_state must have shape ({dim_hybrid}, {dim_hybrid}), "
                f"got {hybrid_state.shape}",
            )

        # Build embedding isometry E: maps hybrid index → two-mode index
        # E[n1*(N+1)*2 + s, n1*2 + s] = 1.0, all else 0.
        E = np.zeros((dim_mzi, dim_hybrid), dtype=complex)
        for n1 in range(dim_osc):
            for s in range(2):
                hybrid_idx = n1 * 2 + s
                mzi_idx = n1 * dim_osc * 2 + s
                E[mzi_idx, hybrid_idx] = 1.0

        # ρ_embedded = E @ ρ_hybrid @ E†
        return E @ hybrid_state @ E.conj().T

    raise ValueError(
        f"hybrid_state must be 1D (state vector) or 2D (density matrix), "
        f"got ndim={hybrid_state.ndim}",
    )


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
    dim_modes = dim_osc**2

    # Build annihilation operators for mode 1 and mode 2
    # Mode 1: a1 ⊗ I_2
    a1 = np.zeros((dim_modes, dim_modes), dtype=complex)
    for n1 in range(dim_osc):
        for n2 in range(dim_osc):
            idx = n1 * dim_osc + n2
            if n1 > 0:
                a1[idx - dim_osc, idx] = np.sqrt(n1)  # a1|n1,n2⟩ = √n1|n1-1,n2⟩

    # Mode 2: I_1 ⊗ a2
    a2 = np.zeros((dim_modes, dim_modes), dtype=complex)
    for n1 in range(dim_osc):
        for n2 in range(dim_osc):
            idx = n1 * dim_osc + n2
            if n2 > 0:
                a2[idx - 1, idx] = np.sqrt(n2)  # a2|n1,n2⟩ = √n2|n1,n2-1⟩

    # Beam splitter generator: G = i(a1†a2 - a1a2†)
    a1_dag = a1.conj().T
    a2_dag = a2.conj().T

    # Compute unitary: U = exp(-iθ G) = exp(θ * (a1†a2 - a1a2†))
    G = 1j * (a1_dag @ a2 - a1 @ a2_dag)
    bs_modes = scipy.linalg.expm(-1j * theta * G)

    # Embed with spin identity
    return np.kron(bs_modes, np.eye(2, dtype=complex))


def mzi_phase_shift(N: int, phi_phase: float) -> np.ndarray:
    """Construct phase shift unitary on mode 1.

    U_phase = exp(i φ n₁) ⊗ I_mode2 ⊗ I_spin

    Args:
        N: Maximum photon number.
        phi_phase: Phase shift in radians.

    Returns:
        Unitary of shape (2(N+1)², 2(N+1)²).

    """
    dim_osc = N + 1
    dim_modes = dim_osc**2

    phase_op = np.zeros((dim_modes, dim_modes), dtype=complex)

    for n1 in range(dim_osc):
        for n2 in range(dim_osc):
            idx = n1 * dim_osc + n2
            phase_op[idx, idx] = np.exp(1j * phi_phase * n1)

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
    dim_modes = dim_osc**2

    # n₁ in mode space: diagonal with value n1
    n1_op = np.zeros((dim_modes, dim_modes), dtype=complex)
    for n1 in range(dim_osc):
        for n2 in range(dim_osc):
            idx = n1 * dim_osc + n2
            n1_op[idx, idx] = n1

    # Embed with spin identity
    return np.kron(n1_op, np.eye(2, dtype=complex))


def evolve_hybrid_mzi(
    hybrid_state: np.ndarray,
    N: int,
    phi_phase: float,
    theta: float = np.pi / 4,
) -> np.ndarray:
    """Evolve hybrid state through MZI.

    Sequence: embed → BS1 → phase shift → BS2

    Args:
        hybrid_state: Input hybrid state of shape (2(N+1),).
        N: Maximum photon number.
        phi_phase: Phase shift in mode 1 (unknown parameter).
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
    ps = mzi_phase_shift(N, phi_phase)
    state = ps @ state

    # BS2
    return bs @ state


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
    return np.abs(final_state) ** 2


def mzi_marginal_photon_probs(
    final_state: np.ndarray,
    N: int,
) -> tuple[np.ndarray, np.ndarray]:
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
# Wigner Function Computation
# =============================================================================


def wigner_function_single(
    rho_osc: np.ndarray,
    x_range: np.ndarray,
    p_range: np.ndarray,
) -> np.ndarray:
    """Compute Wigner function for single-mode density matrix.

    Delegates to qutip.wigner with g=2 to match the α = x + ip convention
    and normalization ∫∫ W(x,p) dx dp = 1.

    Args:
        rho_osc: Density matrix of oscillator (dim N+1, N+1).
        x_range: Array of x quadrature values.
        p_range: Array of p quadrature values.

    Returns:
        2D array W[x_idx, p_idx] of Wigner function values.

    Raises:
        ValueError: If rho_osc is not square.

    """
    if rho_osc.ndim != 2 or rho_osc.shape[0] != rho_osc.shape[1]:
        raise ValueError(f"rho_osc must be square, got shape {rho_osc.shape}")

    rho_qobj = qutip.Qobj(rho_osc)
    # qutip.wigner returns (len(p), len(x)); transpose to (len(x), len(p))
    wigner_result = qutip.wigner(rho_qobj, x_range, p_range, g=2)
    assert wigner_result is not None
    return wigner_result.T


def wigner_from_hybrid_state(
    hybrid_state: np.ndarray,
    N: int,
    x_range: np.ndarray,
    p_range: np.ndarray,
    spin_component: str = "down",
) -> np.ndarray:
    """Extract oscillator density matrix from hybrid state and compute Wigner.

    Args:
        hybrid_state: State vector of shape (2(N+1),) - hybrid oscillator+spin.
        N: Maximum photon number.
        x_range: Array of x quadrature values.
        p_range: Array of p quadrature values.
        spin_component: Which spin to trace ("down" for |↓⟩, "up" for |↑⟩).

    Returns:
        2D array W[x_idx, p_idx].

    Raises:
        ValueError: If spin_component is invalid.

    """
    dim_osc = N + 1
    dim_hybrid = 2 * dim_osc

    if hybrid_state.shape != (dim_hybrid,):
        raise ValueError(
            f"hybrid_state must have shape ({dim_hybrid},), got {hybrid_state.shape}",
        )

    # Extract oscillator state for given spin component
    if spin_component == "down":
        osc_state = hybrid_state[::2]  # Even indices
    elif spin_component == "up":
        osc_state = hybrid_state[1::2]  # Odd indices
    else:
        raise ValueError(f"Unknown spin_component: {spin_component}")

    # Check if state is pure or mixed (for now assume pure from state vector)
    rho_osc = np.outer(osc_state, osc_state.conj())
    rho_qobj = qutip.Qobj(rho_osc)

    # qutip.wigner returns (len(p), len(x)); transpose to (len(x), len(p))
    wigner_result = qutip.wigner(rho_qobj, x_range, p_range, g=2)
    assert wigner_result is not None
    return wigner_result.T


def compute_wigner_for_state(
    hybrid_state: np.ndarray,
    N: int,
    x_max: float = 5.0,
    n_points: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def wigner_is_negative(W: np.ndarray, tol: float = 1e-10) -> bool:
    """Check if Wigner function has negative values.

    Args:
        W: Wigner function array.
        tol: Tolerance for considering negative.

    Returns:
        True if min(W) < -tol.

    """
    return float(np.min(W)) < -tol
