"""
Dicke-basis input states for BEC spin-squeezing simulations.

Provides state vectors in the collective-spin (Dicke) basis |J, m⟩ with
J = N/2, dimension d = N+1, used for BEC interferometry and
one-axis twisting simulations.

Physical Model:
- Hilbert space: Dicke basis |J, m⟩ with J = N/2, d = N+1
- Twin-Fock state: |N/2, 0⟩ → index m=0 → state[N//2] = 1
- NOON state: (|N,0⟩ + |0,N⟩)/√2 → (|J,J⟩ + |J,-J⟩)/√2

Basis Conventions:
- m-ordered descending: m = N/2, N/2-1, ..., -N/2
- Index i corresponds to eigenvalue J - i
- |N,0⟩ ↔ |J,  J⟩ (index 0)
- |0,N⟩ ↔ |J, -J⟩ (index N)

Units:
- Dimensionless throughout.

See Also:
    src.physics.dicke_basis: Dicke basis operators and transformations
    src.algorithms.spin_squeezing: CSS and squeezed state generation
"""

from __future__ import annotations

import numpy as np


def generate_twin_fock_state(N: int) -> np.ndarray:
    """Generate Twin-Fock state |N/2, 0⟩ in the Dicke basis.

    The Twin-Fock state has equal population in both modes with minimal
    variance in the perpendicular direction, achieving near-Heisenberg
    scaling.

    Args:
        N: Total atom number (must be even).

    Returns:
        State vector in Dicke basis, dimension N+1.

    Raises:
        ValueError: If N is odd.

    Example:
        >>> state = generate_twin_fock_state(4)
        >>> state.shape
        (5,)
        >>> np.argmax(np.abs(state))
        2

    """
    if N % 2 != 0:
        raise ValueError(f"N must be even for Twin-Fock state, got N={N}")

    dim = N + 1
    state = np.zeros(dim, dtype=complex)
    state[N // 2] = 1.0
    return state


def generate_noon_state(N: int) -> np.ndarray:
    """Generate NOON state (|N,0⟩ + |0,N⟩)/√2 in the Dicke basis.

    Achieves Heisenberg limit: Δφ = 1/N.

    In the Dicke basis:
    - |N, 0⟩ = |J,  J⟩  (index 0)
    - |0, N⟩ = |J, -J⟩ (index N)

    Args:
        N: Total atom number.

    Returns:
        State vector in Dicke basis, dimension N+1.

    Example:
        >>> state = generate_noon_state(4)
        >>> state.shape
        (5,)
        >>> np.isclose(np.abs(state[0])**2, 0.5)
        True
        >>> np.isclose(np.abs(state[4])**2, 0.5)
        True

    """
    if N < 1:
        raise ValueError(f"N must be >= 1 for NOON state, got N={N}")

    dim = N + 1
    state = np.zeros(dim, dtype=complex)
    norm = 1.0 / np.sqrt(2)
    state[0] = norm  # |N, 0⟩ = |J, J⟩
    state[N] = norm  # |0, N⟩ = |J, -J⟩
    return state
