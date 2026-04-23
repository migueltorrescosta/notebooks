"""
Angular momentum operator generation for quantum spin systems.

This module provides functions to construct angular momentum operators
(Jx, Jz) for a spin-J system in the standard basis. These operators are
essential for modeling nuclear spin systems, atomic ensembles, and other
few-level quantum systems with angular momentum symmetry.

The operators are returned in the basis where Jz is diagonal, with
magnetic quantum numbers m = -J, -J+1, ..., J.
"""

import numpy as np


def generate_spin_matrices(dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate Jx and Jz angular momentum operators for a spin-J system.

    Constructs the x and z components of the total angular momentum
    operator for a system with total angular momentum quantum number
    J = (dim - 1) / 2. The matrices are in the standard basis where
    Jz is diagonal (magnetic quantum number basis).

    Args:
        dim: Dimension of the spin space (must be odd for half-integer spins
            or even for integer spins). For spin-J, dim = 2J + 1.

    Returns:
        A tuple (Jx, Jz) where:
        - Jx: The x-component of angular momentum (Hermitian)
        - Jz: The z-component of angular momentum (diagonal, Hermitian)

    Raises:
        ValueError: If dim < 1.

    Example:
        >>> Jx, Jz = generate_spin_matrices(3)  # Spin-1 system (2*1+1 = 3)
        >>> Jz.diagonal()  # Eigenvalues are m = 1, 0, -1
        array([ 1.,  0., -1.])
        >>> np.allclose(Jx, Jx.conj().T)  # Jx is Hermitian
        True
    """
    if dim < 1:
        raise ValueError("Dimension must be at least 1")

    spin = (dim - 1) / 2
    # Vectorized construction of Jz (diagonal matrix)
    j = np.arange(dim)
    magnetic_numbers = spin - j
    jz = np.diag(magnetic_numbers)

    # Vectorized construction of Jx (off-diagonal matrix)
    off_diags = 0.5 * np.sqrt(
        (spin - magnetic_numbers[:-1] + 1) * (spin + magnetic_numbers[:-1])
    )
    jx = np.zeros((dim, dim))
    jx[np.arange(dim - 1), np.arange(1, dim)] = off_diags
    jx[np.arange(1, dim), np.arange(dim - 1)] = off_diags

    return np.array(jx, dtype=float), np.array(jz, dtype=float)
