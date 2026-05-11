"""
Kerr-nonlinear Mach-Zehnder Interferometer.

Physical Model:
- Standard MZI architecture (BS1 -> phase + Kerr -> BS2)
- Kerr nonlinearity adds intensity-dependent phase shift in each arm:
    H_Kerr = chi (a1dag a1)^2 + chi (a2dag a2)^2 = chi (n1^2 + n2^2)
- Combined phase + nonlinear evolution:
    U = exp(i * [phi * n2 + chi * T * (n1^2 + n2^2)])
  where n2 = a2dag a2 is the number operator for mode 1 (second arm)

The Kerr term creates a photon-number-dependent phase that can
enhance phase sensitivity beyond the standard Heisenberg limit
through nonlinear phase magnification.

Hilbert Space:
- Two-mode Fock basis with max N photons (dimension: (N+1)^2)

Units:
- Dimensionless throughout. Phase phi in radians.
- chi has units of 1/time (dimensionless when multiplied by T).

Conventions:
- Beam splitter transformation: a -> cos(theta)a + i*e^{i*phi}*sin(theta)b
  (from src.physics.mzi_simulation)
- Phase shift on mode 1 (second arm): exp(i * phi * n2)
- State ordering: |n1, n2> with n1 as first mode, n2 as second mode
- Consistent with src.physics.mzi_simulation conventions
"""

from typing import Tuple

import numpy as np

from src.physics.mzi_simulation import beam_splitter_unitary
from src.utils.validators import validate_state_mzi


# =============================================================================
# Unitary Construction
# =============================================================================


def kerr_phase_shift_unitary(
    phi: float,
    chi: float,
    T: float,
    max_photons: int,
) -> np.ndarray:
    r"""Create combined linear phase shift and Kerr nonlinearity unitary.

    Constructs the diagonal unitary:
        U = exp(i * [phi * n2 + chi * T * (n1^2 + n2^2)])

    where n1 = a1dag a1 is the number operator for mode 0 (first arm),
    and n2 = a2dag a2 is the number operator for mode 1 (second arm).

    The linear phase phi * n2 is applied to the second mode, consistent
    with the standard MZI convention where the phase shift accumulates
    in one arm (see :func:`src.physics.mzi_simulation.phase_shift_unitary`).
    The Kerr terms add intensity-dependent phase shifts in each arm.

    The full unitary is diagonal in the two-mode Fock basis because all
    terms are functions of the number operators, which are themselves
    diagonal. This means the evolution is purely phase-space rotation
    with no population transfer between Fock states.

    Args:
        phi: Linear phase shift applied to mode 1 (radians). This is
            the parameter to be estimated. When chi=0, this reduces
            to the standard phase shift unitary.
        chi: Kerr nonlinearity strength. Dimensionless when T is
            also dimensionless. Controls the magnitude of the
            intensity-dependent phase per photon per unit time.
        T: Evolution time. The product chi * T controls the total
            nonlinear phase contribution.
        max_photons: Maximum photon number per mode for Hilbert
            space truncation. Determines the matrix dimension:
            (max_photons + 1)^2.

    Returns:
        Diagonal unitary matrix of dimension (max_photons + 1)^2 x
        (max_photons + 1)^2 representing the combined linear and
        nonlinear evolution.

    Raises:
        ValueError: If max_photons is negative, or if chi or T are
            negative (non-physical parameter values).

    Example:
        >>> U = kerr_phase_shift_unitary(1.0, 0.1, 1.0, max_photons=2)
        >>> U.shape
        (9, 9)
        >>> np.allclose(U @ U.conj().T, np.eye(9))
        True
        >>> # Verify diagonal structure
        >>> np.allclose(U, np.diag(np.diag(U)))
        True

    Notes:
        When chi = 0, this function reproduces the standard
        phase shift unitary from mzi_simulation:
        >>> U_lin = kerr_phase_shift_unitary(1.0, 0.0, 1.0, max_photons=2)
        >>> from src.physics.mzi_simulation import phase_shift_unitary
        >>> np.allclose(U_lin, phase_shift_unitary(1.0, 2))
        True
    """
    if max_photons < 0:
        raise ValueError(f"max_photons must be non-negative, got {max_photons}")
    if chi < 0:
        raise ValueError(f"Kerr nonlinearity chi must be non-negative, got {chi}")
    if T < 0:
        raise ValueError(f"Evolution time T must be non-negative, got {T}")

    dim = (max_photons + 1) ** 2
    U = np.zeros((dim, dim), dtype=complex)

    # Combined nonlinear coefficient: chi * T
    kerr_coeff = chi * T

    for n1 in range(max_photons + 1):
        for n2 in range(max_photons + 1):
            idx = n1 * (max_photons + 1) + n2

            # Linear phase: phi * n2 on mode 1 (second arm)
            # Kerr phase: chi * T * (n1^2 + n2^2)
            phase = phi * n2 + kerr_coeff * (n1**2 + n2**2)
            U[idx, idx] = np.exp(1j * phase)

    return U


# =============================================================================
# Full Kerr MZI Evolution
# =============================================================================


def kerr_mzi(
    initial_state: np.ndarray,
    phi: float,
    chi: float,
    T: float,
    max_photons: int,
    theta: float = np.pi / 4,
    phi_bs: float = 0.0,
) -> np.ndarray:
    r"""Run Kerr-nonlinear Mach-Zehnder interferometer.

    Implements the circuit:

        BS1(theta, phi_bs) -> U_kerr(phi, chi, T) -> BS2(theta, phi_bs)

    where U_kerr combines the linear phase shift and Kerr nonlinearity:

        U_kerr = exp(i * [phi * n2 + chi * T * (n1^2 + n2^2)])

    The beam splitters are implemented via the standard
    :func:`src.physics.mzi_simulation.beam_splitter_unitary`.

    Circuit sequence:
        1. BS1: First beam splitter creates path entanglement / mode mixing.
        2. U_kerr: Simultaneous linear phase and nonlinear Kerr evolution
           (diagonal in the Fock basis, no population transfer).
        3. BS2: Second beam splitter recombines paths for interference.

    Args:
        initial_state: Input state vector in the two-mode Fock basis
            of dimension (max_photons + 1)^2. Common choices include
            NOON states, Twin-Fock states, Fock states, or coherent
            states.
        phi: Linear phase shift (the parameter to estimate) applied
            to mode 1 (second arm). Measured in radians.
        chi: Kerr nonlinearity strength. When chi = 0, the evolution
            reduces to the standard MZI with a linear phase shift.
        T: Evolution time for the nonlinearity. The product chi * T
            controls the nonlinear phase contribution.
        max_photons: Maximum photon number per mode for Hilbert
            space truncation.
        theta: Beam splitter transmittance angle. theta = pi/4 gives
            a symmetric 50/50 beam splitter (default).
        phi_bs: Beam splitter phase parameter controlling the
            reflection phase (default: 0).

    Returns:
        Final state vector after the full Kerr-MZI circuit, of
        dimension (max_photons + 1)^2. The state is normalized
        (unitary evolution preserves norm).

    Raises:
        ValueError: If the input state is not normalized (invalid
            quantum state).

    Example:
        >>> from src.physics.mzi_simulation import noon_state
        >>> state = noon_state(3, max_photons=3)
        >>> final = kerr_mzi(state, phi=1.0, chi=0.1, T=1.0, max_photons=3)
        >>> np.isclose(np.sum(np.abs(final) ** 2), 1.0)
        True
        >>> # Without Kerr, final state norm is still preserved
        >>> final_lin = kerr_mzi(state, phi=1.0, chi=0.0, T=0.0, max_photons=3)
        >>> np.isclose(np.sum(np.abs(final_lin) ** 2), 1.0)
        True
    """
    if not validate_state_mzi(initial_state):
        raise ValueError(
            "Initial state is not normalized. "
            "Provide a valid quantum state with ||psi|| = 1."
        )

    expected_dim = (max_photons + 1) ** 2
    if len(initial_state) != expected_dim:
        raise ValueError(
            f"Initial state has dimension {len(initial_state)}, "
            f"expected {expected_dim} for max_photons={max_photons}"
        )

    # BS1: First beam splitter
    bs = beam_splitter_unitary(theta, phi_bs, max_photons)
    state = bs @ initial_state

    # Combined phase shift + Kerr evolution (diagonal unitary)
    U_kerr = kerr_phase_shift_unitary(phi, chi, T, max_photons)
    state = U_kerr @ state

    # BS2: Second beam splitter (identical to BS1)
    state = bs @ state

    return state


# =============================================================================
# Output Port Probabilities
# =============================================================================


def compute_kerr_output_probabilities(
    state: np.ndarray,
    max_photons: int,
) -> Tuple[float, float]:
    r"""Compute detection probabilities at the two output ports.

    Calculates the normalized probability of detecting photons at
    output mode 0 versus output mode 1 after the second beam
    splitter. For a pure state |psi>, the probability of detecting
    a photon in output mode k is:

        P_k = <psi|n_k|psi> / <psi|(n_0 + n_1)|psi>

    where n_k is the number operator for mode k. The probabilities
    are normalized by the total mean photon number.

    Args:
        state: Pure state vector after the full Kerr-MZI circuit.
            Must have dimension (max_photons + 1)^2.
        max_photons: Maximum photon number per mode used in the
            Hilbert space truncation.

    Returns:
        Tuple (P0, P1) where:
        - P0: Probability of detecting a photon in output mode 0.
        - P1: Probability of detecting a photon in output mode 1.
        Both are real, non-negative, and sum to 1 (when the total
        mean photon number is nonzero). For vacuum input, returns
        (0.5, 0.5).

    Example:
        >>> from src.physics.mzi_simulation import noon_state
        >>> state = kerr_mzi(noon_state(2, max_photons=2), 0.0, 0.0, 1.0, 2)
        >>> P0, P1 = compute_kerr_output_probabilities(state, 2)
        >>> np.isclose(P0 + P1, 1.0)
        True
    """
    P0 = 0.0
    P1 = 0.0

    for n1 in range(max_photons + 1):
        for n2 in range(max_photons + 1):
            idx = n1 * (max_photons + 1) + n2
            prob = np.real(state[idx] * np.conj(state[idx]))
            P0 += prob * n1
            P1 += prob * n2

    total = P0 + P1
    if total > 1e-15:
        return P0 / total, P1 / total
    else:
        return 0.5, 0.5


# =============================================================================
# Phase Sensitivity Analysis
# =============================================================================


def compute_kerr_phase_sensitivity(
    N: int,
    chi: float,
    T: float,
    max_photons: int | None = None,
) -> float:
    r"""Compute the quantum Fisher information for Kerr MZI with NOON input.

    For a NOON input state (|N, 0> + |0, N>)/sqrt(2) in the Kerr MZI,
    the QFI for estimating the phase parameter phi is computed as:

        F_Q = 4 * Var(G)

    where G = n2 is the generator of the phase shift (number operator
    for mode 1, the phase-shifted arm). The QFI is evaluated on the
    NOON state itself (the probe state entering the nonlinear phase
    evolution), which is the standard approach in quantum metrology.

    For a NOON state with N photons, the QFI is F_Q = N^2, achieving
    the Heisenberg limit. The Kerr nonlinearity preserves this result
    because the generator n2 commutes with the diagonal Kerr term
    [n2, n1^2 + n2^2] = 0, so the QFI is invariant under the
    nonlinear evolution.

    The phase sensitivity lower bound from the QFI is:

        Delta_phi >= 1 / sqrt(F_Q)

    which for a NOON state gives the Heisenberg scaling Delta_phi ~ 1/N.

    Args:
        N: Total photon number for the NOON input state. Must be
            non-negative.
        chi: Kerr nonlinearity strength. Included for API consistency
            with the full Kerr model. The QFI for the linear generator
            n2 is independent of chi since the generator commutes with
            the Kerr Hamiltonian (see notes above).
        T: Evolution time. Included for API consistency.
        max_photons: Maximum photon number per mode for Hilbert space
            truncation. If None, defaults to N (sufficient for NOON
            states with up to N photons per mode).

    Returns:
        Quantum Fisher information F_Q for estimating the linear phase
        shift phi. For a NOON state, this equals N^2 (Heisenberg limit)
        regardless of chi and T, reflecting the fact that the generator
        n2 commutes with the Kerr diagonal.

    Example:
        >>> # Standard NOON MZI (no Kerr): F_Q = N^2
        >>> F_noon = compute_kerr_phase_sensitivity(3, 0.0, 0.0)
        >>> np.isclose(F_noon, 9.0)
        True
        >>> # With Kerr: QFI unchanged since generator commutes
        >>> F_kerr = compute_kerr_phase_sensitivity(3, 0.5, 1.0)
        >>> np.isclose(F_kerr, 9.0)
        True
        >>> # Heisenberg scaling: F_Q = N^2
        >>> F_N2 = compute_kerr_phase_sensitivity(10, 0.0, 0.0)
        >>> np.isclose(F_N2, 100.0)
        True

    Notes:
        The QFI being independent of chi is a consequence of the
        generator n2 commuting with the Kerr Hamiltonian. In a more
        general nonlinear estimation scenario where the parameter
        couples through a nonlinear generator (e.g., n2^2), the QFI
        can show enhanced scaling. This function computes the QFI
        for the standard linear phase parameter phi coupled through
        n2.

        The QFI is computed on the NOON state directly (the probe
        state before entering the interferometer), not after BS1.
        This matches the standard quantum metrology convention
        where the NOON state is the path-entangled resource used
        for Heisenberg-limited phase estimation.
    """
    if max_photons is None:
        max_photons = N

    from src.physics.mzi_simulation import noon_state

    # Build NOON state: (|N, 0> + |0, N>)/sqrt(2)
    # The QFI is evaluated on the probe state directly (standard
    # quantum metrology convention for NOON-state interferometry).
    state = noon_state(N, max_photons)

    # Compute Var(n2) where n2 is the number operator for mode 1
    # n2 |n1, n2> = n2 |n1, n2> (diagonal in Fock basis)
    mean_n2 = 0.0
    mean_n2_sq = 0.0

    dim = max_photons + 1
    for n1 in range(dim):
        for n2 in range(dim):
            idx = n1 * dim + n2
            amp_sq = np.real(state[idx] * np.conj(state[idx]))
            mean_n2 += amp_sq * n2
            mean_n2_sq += amp_sq * n2**2

    var_n2 = mean_n2_sq - mean_n2**2

    # QFI = 4 * Var(G) where G = n2
    F_Q = 4.0 * var_n2

    return float(F_Q)


# =============================================================================
# Interference Fringe Computation
# =============================================================================


def compute_kerr_interference_fringe(
    phase_range: np.ndarray,
    chi: float,
    T: float,
    max_photons: int,
    initial_state: np.ndarray | None = None,
    N: int = 1,
    theta: float = np.pi / 4,
    phi_bs: float = 0.0,
) -> np.ndarray:
    r"""Compute the interference fringe for the Kerr MZI.

    Sweeps the linear phase shift phi and computes the output
    probability P0 at each value, producing the characteristic
    interference pattern used in phase estimation experiments.

    The resulting fringe encodes the phase information and shows
    how the Kerr nonlinearity modifies the interference pattern
    through intensity-dependent phase shifts.

    Args:
        phase_range: Array of phase values (radians) to evaluate.
            Typically np.linspace(0, 2*np.pi, num_points).
        chi: Kerr nonlinearity strength.
        T: Evolution time for the nonlinearity.
        max_photons: Maximum photon number per mode.
        initial_state: Input state vector. If None, a NOON state
            with N photons is used as the default.
        N: Photon number for the default NOON state. Ignored if
            initial_state is provided.
        theta: Beam splitter transmittance angle (default: pi/4).
        phi_bs: Beam splitter phase parameter (default: 0).

    Returns:
        Array of P0 probabilities (detection in output mode 0)
        corresponding to each phase value in phase_range.

    Example:
        >>> phases = np.linspace(0, 2*np.pi, 100)
        >>> from src.physics.mzi_simulation import noon_state
        >>> state = noon_state(3, max_photons=3)
        >>> fringe = compute_kerr_interference_fringe(
        ...     phases, chi=0.1, T=1.0, max_photons=3,
        ...     initial_state=state,
        ... )
        >>> fringe.shape
        (100,)
        >>> np.all(fringe >= 0) and np.all(fringe <= 1)
        True
    """
    if initial_state is None:
        from src.physics.mzi_simulation import noon_state

        initial_state = noon_state(N, max_photons)

    probs = []
    for phi in phase_range:
        state = kerr_mzi(
            initial_state,
            phi,
            chi,
            T,
            max_photons,
            theta,
            phi_bs,
        )
        p0, _ = compute_kerr_output_probabilities(state, max_photons)
        probs.append(p0)

    return np.array(probs)
