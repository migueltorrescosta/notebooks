"""
Angular momentum operators for collective-spin (Dicke) systems.

Provides parameterised J_x, J_y, J_z operators that generalise the
Pauli matrices in :mod:`src.utils.constants` to arbitrary spin quantum
number *J* = N/2.  These are pure mathematical objects (generalised
Pauli matrices) and carry no physics-specific logic, which is why they
live in ``utils`` rather than ``physics``.

Conventions:
- Dimension: d = N + 1 = 2J + 1
- DICKE basis: |J, m⟩ with m = J, J-1, ..., -J (descending)
- FOCK basis:  |n⟩ with n = 0, 1, ..., N; eigenvalue n - N/2
- Units: dimensionless
"""

from __future__ import annotations

import numpy as np

from src.utils.enums import OperatorBasis


def jz_operator(N: int, *, basis: OperatorBasis) -> np.ndarray:
    """Construct the dense J_z operator.

    This is the single authoritative implementation of J_z for the project.

    Basis Conventions:
        DICKE: |J, m⟩ with J = N/2.
            Eigenvalues: m = N/2, N/2-1, ..., -N/2 (descending).
            Example N=4: diag([2., 1., 0., -1., -2.])
        FOCK: Bosonic Fock basis |n⟩ with n = 0, 1, ..., N.
            Eigenvalues: n - N/2 = -N/2, -N/2+1, ..., N/2 (ascending).
            Example N=4: diag([-2., -1.,  0.,  1.,  2.])

    Matrix elements:
        DICKE: :math:`\\langle J, m'|J_z|J, m\\rangle = m \\delta_{m',m}`
        FOCK:  :math:`\\langle n'|J_z|n\\rangle = (n - N/2) \\delta_{n',n}`

    Args:
        N: Total number of two-level atoms. Must be non-negative.
        basis: Basis convention (``OperatorBasis.DICKE`` or ``OperatorBasis.FOCK``).

    Returns:
        Diagonal (N+1) x (N+1) matrix representing J_z.

    Raises:
        ValueError: If N is negative.

    Example:
        >>> J_z = jz_operator(N=4, basis=OperatorBasis.DICKE)
        >>> J_z.diagonal()
        array([ 2.,  1.,  0., -1., -2.])

    """
    if N < 0:
        raise ValueError(f"Number of atoms N must be non-negative, got {N}")

    if basis == OperatorBasis.DICKE:
        eigenvalues = np.arange(N / 2.0, -N / 2.0 - 1, -1)
    elif basis == OperatorBasis.FOCK:
        eigenvalues = np.arange(N + 1) - N / 2.0
    else:
        raise ValueError(f"Unknown basis: {basis!r}. Use OperatorBasis.DICKE or FOCK.")

    return np.diag(eigenvalues)


def jx_operator(N: int, *, basis: OperatorBasis) -> np.ndarray:
    """Construct the dense J_x operator in the Dicke basis.

    The J_x operator is the collective spin x-component, obtained
    from QuTiP's jmat.  Returns a real symmetric matrix.
    Only the DICKE basis is supported; FOCK will raise ``ValueError``.

    Args:
        N: Total number of two-level atoms. Must be non-negative.
        basis: Basis convention. Only ``OperatorBasis.DICKE`` is supported.

    Returns:
        Real symmetric (N+1) x (N+1) matrix representing J_x in
        the Dicke basis.

    Raises:
        ValueError: If N is negative.
        ValueError: If ``basis`` is ``OperatorBasis.FOCK`` (not supported).

    """
    if N < 0:
        raise ValueError(f"Number of atoms N must be non-negative, got {N}")

    if basis == OperatorBasis.FOCK:
        raise ValueError(
            "J_x only supports the DICKE basis. Use basis=OperatorBasis.DICKE.",
        )

    if N == 0:
        return np.zeros((1, 1), dtype=float)

    from qutip import jmat

    return np.real(jmat(N / 2.0, "x").full()).astype(float)


def jy_operator(N: int, *, basis: OperatorBasis) -> np.ndarray:
    r"""Construct the dense J_y operator in the Dicke basis.

    The J_y operator is the collective spin y-component, obtained
    from QuTiP's jmat.  J_y is Hermitian with purely imaginary
    off-diagonal elements.
    Only the DICKE basis is supported; FOCK will raise ``ValueError``.

    Args:
        N: Total number of two-level atoms. Must be non-negative.
        basis: Basis convention. Only ``OperatorBasis.DICKE`` is supported.

    Returns:
        Hermitian (N+1) x (N+1) matrix with purely imaginary
        off-diagonal elements representing J_y in the Dicke basis.

    Raises:
        ValueError: If N is negative.
        ValueError: If ``basis`` is ``OperatorBasis.FOCK`` (not supported).

    """
    if N < 0:
        raise ValueError(f"Number of atoms N must be non-negative, got {N}")

    if basis == OperatorBasis.FOCK:
        raise ValueError(
            "J_y only supports the DICKE basis. Use basis=OperatorBasis.DICKE.",
        )

    if N == 0:
        return np.zeros((1, 1), dtype=complex)

    from qutip import jmat

    return jmat(N / 2.0, "y").full()
