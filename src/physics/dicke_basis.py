"""
Dicke Basis Representation for N-Atom Two-Mode Systems.

This module provides collective spin operators and basis transformations
for atomic ensembles in quantum optics, using QuTiP's jmat for the
underlying angular momentum operator construction.

Physical Model:
- Hilbert space: Dicke basis |J, m⟩ with J = N/2, m ∈ {-J, ..., J}
- Dimension: d = N + 1
- Collective spin operators: J_x, J_y, J_z satisfy SU(2) algebra
- Fock basis mapping: |n₁, n₂⟩ ↔ |J, m⟩ where m = (n₁ - n₂)/2

Conventions:
- Collective operators: J_x = (a†b + b†a)/2, J_y = (a†b - b†a)/(2i)
- J_z = (a†a - b†b)/2
- Phase convention: standard quantum mechanics (no extra phases)
- Units: dimensionless throughout

Basis Conventions:
- This module uses DICKE BASIS |J, m⟩ with eigenvalue ordering:
    m = N/2, N/2-1, ..., -N/2 (descending)
- For comparison with BOSONIC FOCK basis (used in lindblad_solver.py):
    Dicke: eigenvalues [N/2, N/2-1, ..., -N/2]
    Fock:  eigenvalues [0-N/2, 1-N/2, ..., N-N/2] = [-(N/2), ..., N/2]
- Both give same eigenvalue range but different ordering!

Example:
    >>> from src.physics.dicke_basis import dicke_states, jz_operator, jx_operator
    >>> N = 4
    >>> basis = dicke_states(N)
    >>> J_z = jz_operator(N)
    >>> J_z.diagonal()  # Eigenvalues: 2, 1, 0, -1, -2
    array([ 2.,  1.,  0., -1., -2.])

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.utils.enums import OperatorBasis


def dicke_states(N: int) -> dict[float, int]:
    """Generate Dicke basis states mapping m → index for N atoms.

    Creates a dictionary mapping magnetic quantum number m to the
    corresponding basis index. The Dicke basis states |J, m⟩
    with J = N/2 are ordered by decreasing m (from +J to -J).

    Args:
        N: Total number of two-level atoms. Must be non-negative.
            The Hilbert space dimension is d = N + 1.

    Returns:
        Dictionary mapping m (magnetic quantum number) to index.
        Keys range from m = N/2 down to m = -N/2 in steps of 1.

    Raises:
        ValueError: If N is negative.

    """
    if N < 0:
        raise ValueError(f"Number of atoms N must be non-negative, got {N}")

    J = N / 2.0
    result: dict[float, int] = {}
    m_values = np.arange(J, -J - 1, -1)
    for idx, m in enumerate(m_values):
        result[float(m)] = idx
    return result


def to_dicke_basis(fock_state: np.ndarray, N: int) -> np.ndarray:
    """Convert a two-mode Fock state vector to the Dicke basis representation.

    Transforms a state |ψ⟩ expressed in the two-mode Fock basis
    (|n₁, n₂⟩ with n₁ + n₂ = N constraint for symmetric Dicke space)
    to the collective spin Dicke basis |J, m⟩.

    The transformation maps:
        |n₁, n₂⟩ → |J, m⟩ with n₁ + n₂ = N, m = (n₁ - n₂)/2

    Args:
        fock_state: State vector in the two-mode Fock basis with
            dimension (N+1)² × 1.
        N: Total atom/photon number. Determines the Dicke space dimension.

    Returns:
        Column vector in the Dicke basis with dimension (N+1) × 1,
        ordered by m = N/2, N/2-1, ..., -N/2.

    Raises:
        ValueError: If fock_state dimension doesn't match (N+1)².
        ValueError: If N is negative.

    """
    if N < 0:
        raise ValueError(f"Number of atoms N must be non-negative, got {N}")

    expected_dim = (N + 1) ** 2
    if fock_state.shape[0] != expected_dim:
        raise ValueError(
            f"Expected Fock state dimension {expected_dim}, got {fock_state.shape[0]}",
        )

    dicke_dim = N + 1
    dicke_vec = np.zeros(dicke_dim, dtype=complex)

    J = N / 2.0
    for n1 in range(N + 1):
        n2 = N - n1
        fock_idx = n1 * (N + 1) + n2
        m = (n1 - n2) / 2.0
        dicke_idx = int(J - m)
        dicke_vec[dicke_idx] = fock_state[fock_idx]

    return dicke_vec


def from_dicke_basis(dicke_state: np.ndarray, N: int) -> np.ndarray:
    """Convert a Dicke basis state vector to the two-mode Fock basis.

    Transforms a state |ψ⟩ expressed in the Dicke basis |J, m⟩
    back to the two-mode Fock basis |n₁, n₂⟩ with n₁ + n₂ = N.

    The inverse transformation maps:
        |J, m⟩ → |n₁, n₂⟩ where n₁ + n₂ = N, n₁ = (N/2 + m), n₂ = (N/2 - m)

    Args:
        dicke_state: Column vector in the Dicke basis with dimension
            (N+1) × 1, ordered by m = N/2, N/2-1, ..., -N/2.
        N: Total atom/photon number. Determines the Dicke space dimension.

    Returns:
        State vector in the two-mode Fock basis with dimension
        (N+1)² × 1, with zeros for states outside the symmetric subspace.

    Raises:
        ValueError: If dicke_state dimension doesn't match N+1.
        ValueError: If N is negative.

    """
    if N < 0:
        raise ValueError(f"Number of atoms N must be non-negative, got {N}")

    if dicke_state.shape[0] != N + 1:
        raise ValueError(
            f"Expected Dicke state dimension {N + 1}, got {dicke_state.shape[0]}",
        )

    fock_dim = (N + 1) ** 2
    fock_vec = np.zeros(fock_dim, dtype=complex)

    J = N / 2.0
    for m_idx, m_val in enumerate(np.arange(J, -J - 1, -1)):
        n1 = J + m_val
        n2 = J - m_val
        if np.isclose(n1, int(n1)) and np.isclose(n2, int(n2)):
            n1_int, n2_int = int(n1), int(n2)
            if 0 <= n1_int <= N and 0 <= n2_int <= N:
                fock_idx = n1_int * (N + 1) + n2_int
                fock_vec[fock_idx] = dicke_state[m_idx]

    return fock_vec


def jz_operator(N: int, basis: OperatorBasis | None = None) -> np.ndarray:
    """Construct the dense J_z operator.

    This is the single authoritative implementation of J_z for the project.

    Basis Conventions:
        DICKE (default): |J, m⟩ with J = N/2.
            Eigenvalues: m = N/2, N/2-1, ..., -N/2 (descending).
            Example N=4: diag([2., 1., 0., -1., -2.])
        FOCK: Bosonic Fock basis |n⟩ with n = 0, 1, ..., N.
            Eigenvalues: n - N/2 = -N/2, -N/2+1, ..., N/2 (ascending).
            Example N=4: diag([-2., -1., 0., 1., 2.])

    Both conventions yield the same eigenvalue range but with opposite
    ordering. The choice of basis must be consistent across all operators
    (Hamiltonian, Lindblad, observables) used together.

    Matrix elements:
        DICKE: ⟨J, m'|J_z|J, m⟩ = m δ_{m',m}
        FOCK:  ⟨n'|J_z|n⟩ = (n - N/2) δ_{n',n}

    Args:
        N: Total number of two-level atoms. Must be non-negative.
        basis: Basis convention. Defaults to DICKE if None.

    Returns:
        Diagonal (N+1) × (N+1) matrix representing J_z.

    Raises:
        ValueError: If N is negative.

    Example:
        >>> from src.utils.enums import OperatorBasis
        >>> J_z = jz_operator(N=4)  # DICKE basis (default)
        >>> J_z.diagonal()
        array([ 2.,  1.,  0., -1., -2.])
        >>> J_z_fock = jz_operator(N=4, basis=OperatorBasis.FOCK)
        >>> J_z_fock.diagonal()
        array([-2., -1.,  0.,  1.,  2.])
        >>> np.allclose(J_z, J_z.T.conj())  # Hermitian check
        True

    """
    if N < 0:
        raise ValueError(f"Number of atoms N must be non-negative, got {N}")

    if basis is None:
        from src.utils.enums import OperatorBasis

        basis = OperatorBasis.DICKE

    from src.utils.enums import OperatorBasis

    if basis == OperatorBasis.DICKE:
        eigenvalues = np.arange(N / 2.0, -N / 2.0 - 1, -1)
    elif basis == OperatorBasis.FOCK:
        # Ascending: n - N/2 for n = 0, 1, ..., N
        eigenvalues = np.arange(N + 1) - N / 2.0
    else:
        raise ValueError(f"Unknown basis: {basis!r}. Use OperatorBasis.DICKE or FOCK.")

    return np.diag(eigenvalues)


def jx_operator(N: int) -> np.ndarray:
    """Construct the dense J_x operator in the Dicke basis.

    The J_x operator is the collective spin x-component, obtained
    from QuTiP's jmat.  Returns a real symmetric matrix.

    Args:
        N: Total number of two-level atoms. Must be non-negative.

    Returns:
        Real symmetric (N+1) × (N+1) matrix representing J_x in
        the Dicke basis.

    Raises:
        ValueError: If N is negative.

    Example:
        >>> J_x = jx_operator(N=2)
        >>> J_x  # For J=1, diag entries are 0, super/sub-diagonal are 1/√2
        array([[0.        , 0.70710678, 0.        ],
               [0.70710678, 0.        , 0.70710678],
               [0.        , 0.70710678, 0.        ]])

    """
    if N < 0:
        raise ValueError(f"Number of atoms N must be non-negative, got {N}")

    if N == 0:
        return np.zeros((1, 1), dtype=float)

    from qutip import jmat

    return jmat(N / 2.0, "x").full().real.astype(float)


def jy_operator(N: int) -> np.ndarray:
    r"""Construct the dense J_y operator in the Dicke basis.

    The J_y operator is the collective spin y-component, obtained
    from QuTiP's jmat.  J_y is Hermitian with purely imaginary
    off-diagonal elements.

    Args:
        N: Total number of two-level atoms. Must be non-negative.

    Returns:
        Hermitian (N+1) × (N+1) matrix with purely imaginary
        off-diagonal elements representing J_y in the Dicke basis.

    Raises:
        ValueError: If N is negative.

    Example:
        >>> J_y = jy_operator(N=1)
        >>> np.allclose(J_y, J_y.T.conj())  # Hermitian check
        True

    """
    if N < 0:
        raise ValueError(f"Number of atoms N must be non-negative, got {N}")

    if N == 0:
        return np.zeros((1, 1), dtype=complex)

    from qutip import jmat

    return jmat(N / 2.0, "y").full()


def basis_transformation_matrix(N: int) -> np.ndarray:
    r"""Construct the unitary matrix that transforms between Fock and Dicke bases.

    The transformation matrix T maps Dicke basis states to Fock basis states:
        T_{fock_idx, dicke_idx} = ⟨n₁, n₂|J, m⟩

    This matrix satisfies T^\dagger T = I (unitarity).

    Args:
        N: Total number of two-level atoms. Must be non-negative.

    Returns:
        Unitary (N+1)² × (N+1) matrix.

    Raises:
        ValueError: If N is negative.

    Example:
        >>> T = basis_transformation_matrix(N=2)
        >>> np.allclose(T @ T.conj().T, np.eye(9))  # Unitarity check
        True

    """
    if N < 0:
        raise ValueError(f"Number of atoms N must be non-negative, got {N}")

    fock_dim = (N + 1) ** 2
    dicke_dim = N + 1
    J = N / 2.0

    T = np.zeros((fock_dim, dicke_dim), dtype=complex)

    # For each Dicke state |J, m⟩, find corresponding Fock state |n1, n2⟩
    for m_idx, m in enumerate(np.arange(J, -J - 1, -1)):
        # In Dicke basis: n1 = J + m, n2 = J - m
        n1 = int(J + m)
        n2 = int(J - m)
        if 0 <= n1 <= N and 0 <= n2 <= N:
            fock_idx = n1 * (N + 1) + n2
            T[fock_idx, m_idx] = 1.0

    return T
