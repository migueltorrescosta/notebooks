"""
Dicke Basis Representation for N-Atom Two-Mode Systems.

This module provides efficient computation of collective spin operators
and basis transformations for atomic ensembles in quantum optics.

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

import numpy as np


def dicke_states(N: int) -> dict[float, int]:
    """Generate Dicke basis states mapping m → fock_index for N atoms.

    Creates a dictionary mapping magnetic quantum number m to the
    corresponding Fock basis index. The Dicke basis states |J, m⟩
    with J = N/2 are ordered by decreasing m (from +J to -J).

    Args:
        N: Total number of two-level atoms. Must be non-negative.
            The Hilbert space dimension is d = N + 1.

    Returns:
        Dictionary mapping m (magnetic quantum number) to fock_index
        (1D index in the two-mode Fock basis). Keys are float values
        for half-integer m when N is odd, and integer values when N is even.
        Keys range from m = N/2 down to m = -N/2 in steps of 1.

    Raises:
        ValueError: If N is negative.

    Example:
        >>> basis = dicke_states(N=4)  # J = 2, m values: 2.0, 1.0, 0.0, -1.0, -2.0
        >>> len(basis)
        5
        >>> basis = dicke_states(N=5)  # J = 2.5, m values: 2.5, 1.5, 0.5, -0.5, -1.5, -2.5
        >>> len(basis)
        6
    """
    if N < 0:
        raise ValueError(f"Number of atoms N must be non-negative, got {N}")

    J = N / 2.0
    # Map m values from +J down to -J
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
            dimension (N+1)² × 1. Only states with total photon number
            = N are represented (symmetric Dicke subspace).
        N: Total atom/photon number. Determines the Dicke space dimension.

    Returns:
        Column vector in the Dicke basis with dimension (N+1) × 1,
        ordered by m = N/2, N/2-1, ..., -N/2.

    Raises:
        ValueError: If fock_state dimension doesn't match (N+1)².
        ValueError: If N is negative.

    Example:
        >>> N = 2
        >>> # Fock state |2,0⟩ (n1=2, n2=0) → m=1
        >>> fock_dim = (N+1)**2
        >>> fock = np.zeros(fock_dim)
        >>> fock[2*(N+1) + 0] = 1.0  # |2,0⟩ at index 2*3+0=6
        >>> dicke = to_dicke_basis(fock, N)
        >>> # m=1 at idx 0, m=0 at idx 1, m=-1 at idx 2
        >>> np.abs(dicke[0])  # amplitude for |J=1, m=1⟩
        1.0
    """
    if N < 0:
        raise ValueError(f"Number of atoms N must be non-negative, got {N}")

    expected_dim = (N + 1) ** 2
    if fock_state.shape[0] != expected_dim:
        raise ValueError(
            f"Expected Fock state dimension {expected_dim}, got {fock_state.shape[0]}"
        )

    dicke_dim = N + 1
    dicke_vec = np.zeros(dicke_dim, dtype=complex)

    J = N / 2.0
    # Map each Fock state |n1, n2⟩ with n1 + n2 = N to Dicke state |J, m⟩
    for n1 in range(N + 1):
        n2 = N - n1  # Constraint: total photon number = N
        fock_idx = n1 * (N + 1) + n2
        m = (n1 - n2) / 2.0
        # m ranges from N/2 down to -N/2, so idx = N//2 - m
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

    Example:
        >>> N = 2
        >>> # Dicke state |J=1, m=1⟩ (idx 0)
        >>> dicke = np.array([1, 0, 0], dtype=complex)
        >>> fock = from_dicke_basis(dicke, N)
        >>> # |2,0⟩ at index 2*3+0=6 has amplitude 1.0
        >>> np.abs(fock[6])
        1.0
    """
    if N < 0:
        raise ValueError(f"Number of atoms N must be non-negative, got {N}")

    if dicke_state.shape[0] != N + 1:
        raise ValueError(
            f"Expected Dicke state dimension {N + 1}, got {dicke_state.shape[0]}"
        )

    fock_dim = (N + 1) ** 2
    fock_vec = np.zeros(fock_dim, dtype=complex)

    J = N / 2.0
    for m_idx, m_val in enumerate(np.arange(J, -J - 1, -1)):
        # Use exact arithmetic: n1 = J + m, n2 = J - m
        n1 = J + m_val
        n2 = J - m_val
        if np.isclose(n1, int(n1)) and np.isclose(n2, int(n2)):
            n1_int, n2_int = int(n1), int(n2)
            if 0 <= n1_int <= N and 0 <= n2_int <= N:
                fock_idx = n1_int * (N + 1) + n2_int
                fock_vec[fock_idx] = dicke_state[m_idx]

    return fock_vec


def jz_eigenvalues(N: int) -> np.ndarray:
    """Compute the J_z eigenvalues for the Dicke basis.

    Returns the eigenvalues of the collective spin J_z operator,
    which are the magnetic quantum numbers m ∈ {-N/2, ..., N/2}.

    Args:
        N: Total number of two-level atoms. Must be non-negative.

    Returns:
        Array of eigenvalues of length N+1, ordered from
        m = N/2 down to m = -N/2.

    Raises:
        ValueError: If N is negative.

    Example:
        >>> jz_eigenvalues(4)  # J = 2
        array([ 2.,  1.,  0., -1., -2.])
        >>> jz_eigenvalues(3)  # J = 1.5
        array([ 1.5,  0.5, -0.5, -1.5])
    """
    if N < 0:
        raise ValueError(f"Number of atoms N must be non-negative, got {N}")

    J = N / 2.0
    return np.arange(J, -J - 1, -1)


def jz_operator(N: int) -> np.ndarray:
    """Construct the dense J_z operator in the Dicke basis.

    IMPORTANT: This implementation uses DICKE BASIS |J, m⟩ with J = N/2.
    Eigenvalues are m = N/2, N/2-1, ..., -N/2 (descending order).

    This differs from lindblad_solver.py which uses BOSONIC FOCK basis:
    - Dicke: [N/2, N/2-1, ..., -N/2]
    - Fock:  [0, 1, 2, ..., N] mapped to eigenvalues (n - N/2)

    Matrix elements:
        ⟨J, m'|J_z|J, m⟩ = m δ_{m',m}

    Args:
        N: Total number of two-level atoms. Must be non-negative.

    Returns:
        Diagonal (N+1) × (N+1) matrix representing J_z in the
        Dicke basis (m basis).

    Raises:
        ValueError: If N is negative.

    Example:
        >>> J_z = jz_operator(N=4)
        >>> J_z.diagonal()
        array([ 2.,  1.,  0., -1., -2.])
        >>> np.allclose(J_z, J_z.T.conj())  # Hermitian check
        True
    """
    if N < 0:
        raise ValueError(f"Number of atoms N must be non-negative, got {N}")

    eigenvalues = jz_eigenvalues(N)
    return np.diag(eigenvalues)


def jx_operator(N: int) -> np.ndarray:
    """Construct the dense J_x operator in the Dicke basis.

    The J_x operator is the collective spin x-component, obtained
    from the raising and lowering operators:
        J_x = (J_+ + J_-)/2

    Matrix elements (off-diagonal):
        ⟨J, m'|J_x|J, m⟩ = (1/2)√(J(J+1) - m(m∓1)) if m' = m∓1

    Args:
        N: Total number of two-level atoms. Must be non-negative.

    Returns:
        Real symmetric (N+1) × (N+1) matrix representing J_x in
        the Dicke basis, with non-zero elements only on the
        super/sub-diagonals.

    Raises:
        ValueError: If N is negative.

    Example:
        >>> J_x = jx_operator(N=2)
        >>> J_x  # For J=1, should be [[0, 1/√2, 0], [1/√2, 0, 1/√2], [0, 1/√2, 0]]
        array([[0.        , 0.70710678, 0.        ],
               [0.70710678, 0.        , 0.70710678],
               [0.        , 0.70710678, 0.        ]])
    """
    if N < 0:
        raise ValueError(f"Number of atoms N must be non-negative, got {N}")

    dim = N + 1
    J = N / 2.0
    J_x = np.zeros((dim, dim), dtype=float)

    # Compute off-diagonal elements using ladder operator matrix elements
    for i in range(dim - 1):
        # Magnetic quantum number for state |i⟩
        m = J - i
        # Matrix element: ⟨J, m-1|J_x|J, m⟩ = ⟨J, m|J_x|J, m-1⟩
        # = (1/2)√((J+m)(J-m+1))
        element = 0.5 * np.sqrt((J + m) * (J - m + 1))
        J_x[i + 1, i] = element
        J_x[i, i + 1] = element

    return J_x


def jy_operator(N: int) -> np.ndarray:
    r"""Construct the dense J_y operator in the Dicke basis.

    The J_y operator is the collective spin y-component:
        J_y = (J_+ - J_-)/(2i) = -i(J_+ - J_-)/2

    J_y is Hermitian (J_y = J_y^\dagger) with purely imaginary off-diagonal elements.

    Args:
        N: Total number of two-level atoms. Must be non-negative.

    Returns:
        Hermitian (N+1) × (N+1) matrix with purely imaginary
        off-diagonal elements representing J_y in the Dicke basis.

    Raises:
        ValueError: If N is negative.

    Example:
        >>> J_y = jy_operator(N=1)
        >>> # For spin-1/2 (N=1), J_y = [[0, -i/2], [i/2, 0]]
        >>> np.allclose(J_y, J_y.T.conj())  # Hermitian check
        True
    """
    if N < 0:
        raise ValueError(f"Number of atoms N must be non-negative, got {N}")

    dim = N + 1
    J = N / 2.0
    J_y = np.zeros((dim, dim), dtype=complex)

    # Hermitian: J_y[i,i+1] = J_y[i+1,i].conj()
    # J_y[i,i+1] = -i/2 * sqrt((J+m)(J-m+1)) where m = J - i
    # J_y[i+1,i] = +i/2 * sqrt((J+m)(J-m+1)) = -J_y[i,i+1].conj()
    for i in range(dim - 1):
        m = J - i
        element = 0.5 * np.sqrt((J + m) * (J - m + 1))
        J_y[i, i + 1] = -1j * element
        J_y[i + 1, i] = 1j * element

    return J_y


def j_squared_operator(N: int) -> np.ndarray:
    """Construct the J² operator in the Dicke basis.

    The total angular momentum squared operator J² is diagonal in
    the Dicke basis with eigenvalue J(J+1) for all states.

    Args:
        N: Total number of two-level atoms. Must be non-negative.

    Returns:
        Diagonal (N+1) × (N+1) matrix with all diagonal elements
        equal to J(J+1) = (N/2)(N/2 + 1).

    Raises:
        ValueError: If N is negative.

    Example:
        >>> J2 = j_squared_operator(N=4)
        >>> J = 4/2 = 2
        >>> expected = J * (J + 1) = 2 * 3 = 6
        >>> np.allclose(J2, expected * np.eye(5))
        True
    """
    if N < 0:
        raise ValueError(f"Number of atoms N must be non-negative, got {N}")

    J = N / 2.0
    eigenvalue = J * (J + 1)
    return eigenvalue * np.eye(N + 1)


def j_raising_operator(N: int) -> np.ndarray:
    """Construct the J_+ raising operator in the Dicke basis.

    J_+ = J_x + i J_y increases the magnetic quantum number m by 1.

    Matrix elements:
        ⟨J, m+1|J_+|J, m⟩ = √((J-m)(J+m+1))

    Args:
        N: Total number of two-level atoms. Must be non-negative.

    Returns:
        (N+1) × (N+1) matrix with non-zero elements only on the
        super-diagonal (above diagonal).

    Raises:
        ValueError: If N is negative.
    """
    if N < 0:
        raise ValueError(f"Number of atoms N must be non-negative, got {N}")

    dim = N + 1
    J = N / 2.0
    J_plus = np.zeros((dim, dim), dtype=complex)

    # For state i (m = J - i), connecting to state i-1 (m+1 = J - (i-1))
    # J_plus[i-1, i] connects |m⟩ to |m+1⟩
    for i in range(1, dim):
        m = J - i  # m for state i
        # ⟨J, m+1|J_+|J, m⟩ = √((J-m)(J+m+1))
        element = np.sqrt((J - m) * (J + m + 1))
        J_plus[i - 1, i] = element

    return J_plus


def j_lowering_operator(N: int) -> np.ndarray:
    r"""Construct the J_- lowering operator in the Dicke basis.

    J_- = J_x - i J_y decreases the magnetic quantum number m by 1.
    This is the Hermitian conjugate of J_+.

    Matrix elements:
        ⟨J, m-1|J_-|J, m⟩ = √((J+m)(J-m+1))

    Args:
        N: Total number of two-level atoms. Must be non-negative.

    Returns:
        (N+1) × (N+1) matrix with non-zero elements only on the
        sub-diagonal (below diagonal). J_- = (J_+)^\dagger.

    Raises:
        ValueError: If N is negative.
    """
    if N < 0:
        raise ValueError(f"Number of atoms N must be non-negative, got {N}")

    # J_- is the Hermitian conjugate of J_+
    return j_raising_operator(N).T.conj()


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
