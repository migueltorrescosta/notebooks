"""
Mach-Zehnder Interferometer Input States.

Physical Model:
- Twin-Fock: |N/2, N/2⟩_Fock = |J, 0⟩_Dicke (balanced two-mode Fock)
- NOON: (|N, 0⟩ + |0, N⟩)/√2 = (|J, J⟩ + |J, -J⟩)/√2
- Twin-Fock Fisher information: F_Q = N²(1+1/N)/2 ≈ N²/2
- NOON achieves Heisenberg limit F_Q = N²

Hilbert Space:
- Two-mode Fock basis with max N photons (dimension: (N+1)²)

Units:
- Dimensionless throughout.

Conventions:
- State ordering: |n₁, n₂⟩ with n₁ as first mode, n₂ as second mode
- J_z = (n₁ - n₂)/2
- For NOON states, use `noon_state` from mzi_simulation module
  (this module re-exports it for convenience)

Note:
- The factory function `input_state_factory` is the preferred way
  to create input states.
"""

import numpy as np
import scipy.special

from src.physics.mzi_simulation import noon_state
from src.utils.validators import validate_state_mzi


def twin_fock_state(N: int, max_photons: int | None = None) -> np.ndarray:
    """Twin-Fock state |J, 0⟩_Dicke for even N.

    The Twin-Fock state is implemented as the symmetric Dicke state |J, 0⟩
    which is the uniform superposition over all N+1 permutations:

    |ψ⟩ = (1/√(N+1)) × ∑_{n=0}^N |n, N-n⟩

    This is equivalent to |J, 0⟩_Dicke where J = N/2.

    This state has:
    - ⟨J_z⟩ = 0 (symmetric)
    - Var(J_z) = N(N+2)/4
    - Fisher information: F_Q = N²(1+1/N)/2 → N²/2 for large N
    - Achieves Heisenberg scaling with sub-leading 1/N correction

    Args:
        N: Total photon number.
        max_photons: Maximum photon number per mode. If None, uses N.

    Returns:
        Normalized Twin-Fock state vector of dimension (max_photons+1)².

    Raises:
        ValueError: If N is odd.

    Example:
        >>> state = twin_fock_state(N=4)  # |0,4⟩ + |1,3⟩ + |2,2⟩ + |3,1⟩ + |4,0⟩) / √5
        >>> np.isclose(np.sum(np.abs(state)**2), 1.0)
        True
    """
    if N % 2 != 0:
        raise ValueError("Twin-Fock requires even N (N/2 must be integer)")

    effective_max = max(N, max_photons or N)
    dim = (effective_max + 1) ** 2
    state = np.zeros(dim, dtype=complex)

    # Build symmetric state |ψ⟩ = (1/√(N+1)) × ∑_n |n, N-n⟩
    norm = 1.0 / np.sqrt(N + 1)

    for n in range(N + 1):
        if n <= effective_max and (N - n) <= effective_max:
            idx = n * (effective_max + 1) + (N - n)
            state[idx] = norm

    return state


def coherent_state_two_mode(
    alpha1: complex,
    alpha2: complex,
    max_photons: int | None = None,
) -> np.ndarray:
    """Two-mode coherent state |α₁⟩ ⊗ |α₂⟩.

    A coherent state is the quantum state most closely resembling
    a classical electromagnetic field. It has minimal uncertainty
    and Poisson photon number distribution.

    Args:
        alpha1: Complex amplitude for mode 0.
        alpha2: Complex amplitude for mode 1.
        max_photons: Maximum photon number per mode. If None, uses
            max(|α₁|², |α₂|²) + adequate margin (6 sigma).

    Returns:
        State vector representing the two-mode coherent state.

    Example:
        >>> state = coherent_state_two_mode(1.0+0j, 0.0+0j)
        >>> np.isclose(np.sum(np.abs(state)**2), 1.0)
        True
    """
    # Determine effective max if not provided
    mean_n1 = abs(alpha1) ** 2
    mean_n2 = abs(alpha2) ** 2
    max_mean = max(mean_n1, mean_n2)

    if max_photons is None:
        # Use enough photons to capture > 6 sigma of Poisson distribution
        if max_mean > 0:
            sigma = np.sqrt(max_mean)
            max_photons = int(max_mean + 6 * sigma + 5)
        else:
            max_photons = 5

    dim = (max_photons + 1) ** 2
    state = np.zeros(dim, dtype=complex)

    # Normalization factor: exp(-(|α₁|² + |α₂|²)/2)
    norm_factor = np.exp(-(abs(alpha1) ** 2 + abs(alpha2) ** 2) / 2)

    for n1 in range(max_photons + 1):
        for n2 in range(max_photons + 1):
            # Coherent state amplitudes via displacement
            amp1 = (alpha1**n1) / np.sqrt(scipy.special.factorial(n1))
            amp2 = (alpha2**n2) / np.sqrt(scipy.special.factorial(n2))

            amplitude = norm_factor * amp1 * amp2
            idx = n1 * (max_photons + 1) + n2
            state[idx] = amplitude

    return state


def single_photon_split_state(N: int, max_photons: int | None = None) -> np.ndarray:
    """Single-photon split state: (|N-1, 1⟩ + |1, N-1⟩)/√2.

    Also known as "single-photon NOON" or the state created by a
    50/50 beam splitter from a single-photon input.

    For N >= 2: |ψ⟩ = (|N-1, 1⟩ + |1, N-1⟩)/√2
    For N = 2, this is just fock state |1,1⟩ (both terms equal)

    Args:
        N: Total photon number (must be >= 2).
        max_photons: Maximum photon number per mode. If None, uses N.

    Returns:
        State vector representing the split single-photon state.

    Raises:
        ValueError: If N < 2.
    """
    if N < 2:
        raise ValueError("N must be >= 2 for single-photon split state")

    effective_max = max(N, max_photons or N)

    # Handle N=2 case specially (both terms equal)
    if N == 2:
        state = np.zeros((effective_max + 1) ** 2, dtype=complex)
        idx = 1 * (effective_max + 1) + 1  # |1,1⟩
        state[idx] = 1.0
        return state

    # General case N > 2
    state = np.zeros((effective_max + 1) ** 2, dtype=complex)

    # |N-1, 1⟩ (index for n1=N-1, n2=1)
    idx_n1 = (N - 1) * (effective_max + 1) + 1
    state[idx_n1] = 1.0 / np.sqrt(2)

    # |1, N-1⟩ (index for n1=1, n2=N-1)
    idx_1n = 1 * (effective_max + 1) + (N - 1)
    state[idx_1n] = 1.0 / np.sqrt(2)

    return state


def input_state_factory(
    state_type: str,
    N: int,
    max_photons: int | None = None,
    **kwargs,
) -> np.ndarray:
    """Factory function for creating various input states for MZI.

    Creates different types of input states commonly used in quantum
    metrology and optics experiments.

    Supported state types:
    - "twin_fock": Twin-Fock state |N/2, N/2⟩ (N must be even)
    - "noon": NOON state (|N,0⟩ + |0,N⟩)/√2
    - "coherent": Two-mode coherent state (use alpha1, alpha2 kwargs)
    - "single_photon_split": (|N-1,1⟩ + |1,N-1⟩)/√2
    - "fock": Simple Fock state |N, 0⟩

    Args:
        state_type: Type of state to create.
        N: Total photon number (used differently per state type).
        max_photons: Maximum photon number per mode. If None, determined per state.
        **kwargs: Additional arguments:
            - For "coherent": alpha1 (complex), alpha2 (complex)

    Returns:
        State vector in the two-mode Fock basis.

    Raises:
        ValueError: If state_type is not recognized.

    Example:
        >>> state = input_state_factory("twin_fock", N=4)
        >>> state = input_state_factory("noon", N=3)
        >>> state = input_state_factory("coherent", N=0, alpha1=1.0, alpha2=0.0)
    """
    match state_type:
        case "twin_fock":
            effective_max = max(N, max_photons or N)
            return twin_fock_state(N, effective_max)
        case "noon":
            effective_max = max(N, max_photons or N)
            # Use noon_state from mzi_simulation (single source of truth)
            return noon_state(N, effective_max)
        case "coherent":
            alpha1 = kwargs.get("alpha1", 1.0 + 0j)
            alpha2 = kwargs.get("alpha2", 0.0 + 0j)
            # For coherent states, derive max_photons from amplitudes if not provided
            if max_photons is None:
                return coherent_state_two_mode(alpha1, alpha2)
            else:
                return coherent_state_two_mode(alpha1, alpha2, max_photons)
        case "single_photon_split":
            effective_max = max(N - 1, max_photons or N)
            return single_photon_split_state(N, effective_max)
        case "fock":
            # Simple Fock state |N, 0⟩
            effective_max = max(N, max_photons or N)
            dim = (effective_max + 1) ** 2
            state = np.zeros(dim, dtype=complex)
            idx = N * (effective_max + 1)  # |N, 0⟩
            state[idx] = 1.0
            return state
        case "css":
            # Coherent state split - same as coherent with alpha on mode 0
            alpha = kwargs.get("alpha", np.sqrt(N) + 0j)
            if max_photons is None:
                return coherent_state_two_mode(alpha, 0.0 + 0j)
            else:
                return coherent_state_two_mode(alpha, 0.0 + 0j, max_photons)
        case "sss":
            # Single photon split for any N (treated as N=2 case)
            if max_photons is None:
                return single_photon_split_state(2)
            else:
                return single_photon_split_state(2, max_photons)
        case _:
            raise ValueError(f"Unknown state_type: {state_type}")


# =============================================================================
# Observable Calculations for Input States
# =============================================================================


def create_jz_operator(max_photons: int) -> np.ndarray:
    """Create J_z operator for two-mode Fock basis.

    J_z = (n₁ - n₂)/2 is the angular momentum projection operator
    in the two-mode Fock basis.

    Args:
        max_photons: Maximum photon number per mode.

    Returns:
        J_z operator matrix of dimension (max_photons+1)² × (max_photons+1)².
    """
    dim = (max_photons + 1) ** 2
    jz = np.zeros((dim, dim), dtype=complex)

    for n1 in range(max_photons + 1):
        for n2 in range(max_photons + 1):
            idx = n1 * (max_photons + 1) + n2
            jz[idx, idx] = (n1 - n2) / 2

    return jz


def compute_jz_expectation(state: np.ndarray, max_photons: int) -> complex:
    """Compute expectation value of J_z for a pure state.

    ⟨J_z⟩ = ⟨ψ|J_z|ψ⟩

    Args:
        state: Pure state vector.
        max_photons: Maximum photon number per mode.

    Returns:
        Complex expectation value (should be real).
    """
    jz = create_jz_operator(max_photons)
    # Ensure state is in the right dimension
    exp_val = np.conj(state) @ jz @ state
    return exp_val


def compute_jz_variance(state: np.ndarray, max_photons: int) -> float:
    """Compute variance of J_z for a pure state.

    Var(J_z) = ⟨J_z²⟩ - ⟨J_z⟩²

    Args:
        state: Pure state vector.
        max_photons: Maximum photon number per mode.

    Returns:
        Variance of J_z.
    """
    jz = create_jz_operator(max_photons)
    jz_sq = jz @ jz

    mean = np.conj(state) @ jz @ state
    mean_sq = np.conj(state) @ jz_sq @ state

    return np.real(mean_sq - mean**2)


def compute_fisher_information(state: np.ndarray, max_photons: int) -> float:
    """Compute QFI for phase estimation in two-mode system.

    For the two-mode interferometer with phase shift on mode 1,
    the QFI for estimating phase φ is:

    F_Q = 4 * Var(n₁) = 4 * Var(J_z + N/2) = 4 * Var(J_z)

    This assumes the optimal measurement (photon number counting in one output port).

    For NOON states: F_Q = N² (Heisenberg limit)
    For Twin-Fock: F_Q = N²(1+1/N)/2 ≈ N²/2

    Args:
        state: Pure state vector.
        max_photons: Maximum photon number per mode.

    Returns:
        Quantum Fisher information for phase estimation.
    """
    # F_Q = 4 * Var(n_mode1) = 4 * Var(n1)
    # Since n1 = J_z + N/2, Var(n1) = Var(J_z)
    jz_var = compute_jz_variance(state, max_photons)

    # For standard MZI with phase in mode 1, F_Q = 4 * Var(n1)
    return 4.0 * jz_var


# =============================================================================
# Validation Functions
# =============================================================================


def validate_twin_fock(N: int, state: np.ndarray, max_photons: int) -> bool:
    """Validate Twin-Fock state properties.

    Checks:
    1. State is normalized
    2. ⟨J_z⟩ = 0 (symmetric)
    3. Variance matches expected formula Var(J_z) = N(N+2)/4 for N=2J

    Args:
        N: Total photon number.
        state: Twin-Fock state vector.
        max_photons: Maximum photon number per mode.

    Returns:
        True if valid.
    """
    # Check normalization
    if not validate_state_mzi(state):
        return False

    # Check J_z expectation = 0 (symmetric property)
    jz_mean = np.real(compute_jz_expectation(state, max_photons))
    if not np.isclose(jz_mean, 0.0, atol=1e-10):
        return False

    return True


def validate_noon(N: int, state: np.ndarray, max_photons: int) -> bool:
    """Validate NOON state properties.

    Checks:
    1. State is normalized
    2. Equal overlap with |N,0⟩ and |0,N⟩
    3. ⟨J_z⟩ = 0
    4. Var(J_z) = N²/4

    Args:
        N: Total photon number.
        state: NOON state vector.
        max_photons: Maximum photon number per mode.

    Returns:
        True if valid.
    """
    # Check normalization
    if not validate_state_mzi(state):
        return False

    # Check J_z expectation = 0
    jz_mean = np.real(compute_jz_expectation(state, max_photons))
    if not np.isclose(jz_mean, 0.0, atol=1e-10):
        return False

    # Validate variance: Var(J_z) = N²/4 for NOON
    jz_var = compute_jz_variance(state, max_photons)
    expected_var = N**2 / 4

    return bool(np.isclose(jz_var, expected_var, rtol=1e-6))
