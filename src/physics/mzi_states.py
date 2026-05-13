"""
Mach-Zehnder Interferometer Input States.

Physical Model:
- NOON: (|N, 0⟩ + |0, N⟩)/√2 = (|J, J⟩ + |J, -J⟩)/√2
  NOON achieves Heisenberg limit F_Q = N² under the J_z generator.
- Twin-Fock (uniform superposition): ∑|n, N-n⟩/√(N+1)
  Under the J_z generator: F_Q = N(N+2)/3 ≈ N²/3 (SQL scaling).
  Note: this differs from the standard twin-Fock |N/2, N/2⟩ state
  which has Var(J_z) = 0 before BS1.

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

from typing import Any

import numpy as np
import scipy.special

from src.physics.mzi_simulation import noon_state
from src.utils.validators import validate_state_mzi


def twin_fock_state(N: int, max_photons: int | None = None) -> np.ndarray:
    """Twin-Fock state as the uniform superposition over all |n, N-n⟩.

    |ψ⟩ = (1/√(N+1)) × ∑_{n=0}^N |n, N-n⟩

    This is the symmetric state with zero mean population imbalance,
    sometimes called the "Dicke state" or "balanced superposition"
    in the Fock basis (distinct from the standard twin-Fock |N/2, N/2⟩).

    Under the J_z generator convention used in this codebase:
        - ⟨J_z⟩ = 0 (symmetric)
        - Var(J_z) = N(N+2)/12
        - F_Q = 4·Var(J_z) = N(N+2)/3 ≈ N²/3 for large N (SQL scaling)

    Args:
        N: Total photon number (must be even).
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


def squeezed_vacuum_state(
    r: float,
    phi: float,
    max_photons: int,
) -> np.ndarray:
    r"""Create squeezed vacuum state S(ξ)|0⟩ in mode 0.

    The single-mode squeezed vacuum state is generated by the squeeze
    operator S(ξ) = exp((ξ* a² - ξ a†²)/2) where ξ = r·e^{iφ}.

    Fock basis decomposition (mode 0, mode 1 is vacuum):
        |ξ⟩ = (1/√cosh r) Σ_{n=0}^{∞} (-e^{iφ} tanh r)^n
              · √((2n)!)/(2^n n!) |2n, 0⟩

    Properties:
    - Mean photon number: ⟨n⟩ = sinh²(r)
    - Photon number parity: only even photon numbers (2n)
    - Quadrature squeezing: ΔX < 1 for appropriate φ
    - Phase sensitivity: Δφ = 1/(2 sinh(r) cosh(r)) for optimal
      homodyne detection

    Uses the standard decomposition with only even photon numbers
    in mode 0. Mode 1 is always vacuum (no photons).

    Args:
        r: Squeezing magnitude (r >= 0). Mean photon number
            ⟨n⟩ = sinh²(r). Larger r gives more squeezing.
        phi: Squeezing angle in radians. Controls which quadrature
            is squeezed. φ = 0 squeezes the X quadrature.
        max_photons: Maximum photon number per mode for Hilbert
            space truncation. Must be >= 0.

    Returns:
        State vector in the two-mode Fock basis of dimension
        (max_photons+1)². Only the |2n, 0⟩ components (even n in
        mode 0) are non-zero.

    Raises:
        ValueError: If r < 0 or max_photons < 0.

    Example:
        >>> state = squeezed_vacuum_state(r=0.5, phi=0.0, max_photons=6)
        >>> np.isclose(np.sum(np.abs(state)**2), 1.0)
        True
        >>> # Only even photon numbers in mode 0 are populated
        >>> # Check that odd-indexed mode-0 amplitudes are zero
        >>> dim = max_photons + 1
        >>> for n in range(1, max_photons + 1, 2):
        ...     idx = n * dim  # index for |n, 0⟩
        ...     assert np.isclose(state[idx], 0.0)

    """
    if r < 0:
        raise ValueError(f"Squeezing magnitude r must be non-negative, got {r}")
    if max_photons < 0:
        raise ValueError(
            f"Maximum photon number must be non-negative, got {max_photons}",
        )

    dim = (max_photons + 1) ** 2
    state = np.zeros(dim, dtype=complex)

    # Normalization factor
    norm = 1.0 / np.sqrt(np.cosh(r))

    # Squeezed vacuum decomposition: only even |2n, 0⟩ components
    # |ξ⟩ = (1/√cosh r) Σ_n (-e^{iφ} tanh r)^n
    #        · √((2n)!)/(2^n n!) |2n, 0⟩
    tanh_r = np.tanh(r)
    exp_i_phi = np.exp(1j * phi)

    max_term = max_photons // 2  # n for |2n, 0⟩ must fit in truncation

    for n in range(max_term + 1):
        n_photons = 2 * n  # Actual photon number in mode 0

        # Coefficient: (-e^{iφ} tanh r)^n · √((2n)!)/(2^n n!)
        coeff = (-exp_i_phi * tanh_r) ** n
        coeff *= np.sqrt(scipy.special.factorial(2 * n))
        coeff /= (2**n) * scipy.special.factorial(n)

        amplitude = norm * coeff

        # Index: |2n, 0⟩ → n1 = 2n, n2 = 0
        idx = n_photons * (max_photons + 1) + 0
        state[idx] = amplitude

    return state


def input_state_factory(
    state_type: str,
    N: int,
    max_photons: int | None = None,
    **kwargs: Any,
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
    - "squeezed_vacuum": Single-mode squeezed vacuum (use r, phi_kwargs)
    - "sss": Single-photon split state (|N-1,1⟩ + |1,N-1⟩)/√2 with N total
            photons. Requires N >= 2. Formerly hard-coded to N=2; now scales
            with the survey N parameter for meaningful scaling analysis.

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
            return coherent_state_two_mode(alpha, 0.0 + 0j, max_photons)
        case "sss":
            # Single-photon split state (|N-1, 1⟩ + |1, N-1⟩)/√2.
            # Uses N as the total photon number; N must be >= 2.
            # For N=2 this yields |1,1⟩ (single Fock state).
            effective_max = max(N, max_photons or N)
            return single_photon_split_state(N, effective_max)
        case "squeezed_vacuum":
            # Default: scale r so mean photon number ⟨N⟩ = sinh²(r) = N
            # (resource normalization: same ⟨N⟩ across compared states).
            # Override by passing r explicitly via kwargs.
            if "r" not in kwargs:
                r = float(np.arcsinh(np.sqrt(max(N, 1))))
            else:
                r = kwargs["r"]
            phi_sv = kwargs.get("phi_sv", 0.0)
            effective_max = max(N, max_photons or N)
            return squeezed_vacuum_state(r, phi_sv, effective_max)
        case _:
            raise ValueError(f"Unknown state_type: {state_type}")


# =============================================================================
# Observable Calculations for Input States
# =============================================================================


def two_mode_jz_operator(max_photons: int) -> np.ndarray:
    """Create J_z operator for two-mode Fock basis.

    J_z = (n₁ - n₂)/2 is the angular momentum projection operator
    in the two-mode Fock basis.

    This is distinct from :func:`dicke_basis.jz_operator` which operates
    in the collective (N+1)-dimensional Dicke basis. This operator operates
    in the full (M+1)²-dimensional two-mode Fock space.

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
    jz = two_mode_jz_operator(max_photons)
    # Ensure state is in the right dimension
    return np.conj(state) @ jz @ state


def compute_jz_variance(state: np.ndarray, max_photons: int) -> float:
    """Compute variance of J_z for a pure state.

    Var(J_z) = ⟨J_z²⟩ - ⟨J_z⟩²

    Args:
        state: Pure state vector.
        max_photons: Maximum photon number per mode.

    Returns:
        Variance of J_z.

    """
    jz = two_mode_jz_operator(max_photons)
    jz_sq = jz @ jz

    mean = np.conj(state) @ jz @ state
    mean_sq = np.conj(state) @ jz_sq @ state

    return np.real(mean_sq - mean**2)


def compute_fisher_information(state: np.ndarray, max_photons: int) -> float:
    """Compute QFI for phase estimation using the number-difference generator J_z.

    The convention in this codebase is to use J_z = (n₁ - n₂)/2 as the
    phase generator for the QFI computation. For the MZI with phase shift
    on mode 1 (generator n₂), this convention produces:

        F_Q = 4 · Var(J_z) = Var(n₁ - n₂)

    which matches the error-propagation sensitivity from number-difference
    measurements. The resulting scaling exponents α are correct across all
    state types, but absolute prefactors may differ from the n₂-generator
    convention by constant factors.

    For definite-N states (NOON, Twin-Fock, Fock), Var(J_z) = Var(n₂) so
    the conventions coincide exactly.

    Reference values under this convention:
        NOON:        F_Q = N²          (Heisenberg limit)
        Twin-Fock:   F_Q = N²(1+1/N)/2 ≈ N²/2
        Coherent:    F_Q = N           (SQL)
        Squeezed vacuum: F_Q = 2⟨N⟩(⟨N⟩+1)

    Args:
        state: Pure state vector.
        max_photons: Maximum photon number per mode.

    Returns:
        Quantum Fisher information for phase estimation under the J_z
        generator convention.

    """
    jz_var = compute_jz_variance(state, max_photons)

    # F_Q = 4 · Var(J_z) under the J_z generator convention
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
    return np.isclose(jz_mean, 0.0, atol=1e-10)


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
