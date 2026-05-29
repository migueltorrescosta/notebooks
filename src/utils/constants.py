"""
Shared physical constants for the MZI metrology project.

Provides canonical definitions of Pauli matrices, identity matrices,
and angular momentum operators. All modules should import from here
rather than redefining these constants locally.

Constants:
    SIGMA_X, SIGMA_Y, SIGMA_Z: Pauli matrices (2×2, complex).
    I_2: 2×2 identity matrix.
    I_4: 4×4 identity matrix.
    EYE: Alias for I_2 (for backward compatibility with heisenberg_model.py).
    J_X, J_Y, J_Z: Spin-1/2 angular momentum operators (J = σ/2).
"""

import numpy as np

# Pauli matrices (dimensionless, 2×2)
SIGMA_X: np.ndarray = np.array([[0, 1], [1, 0]], dtype=complex)
SIGMA_Y: np.ndarray = np.array([[0, -1j], [1j, 0]], dtype=complex)
SIGMA_Z: np.ndarray = np.array([[1, 0], [0, -1]], dtype=complex)

# Identity matrices
I_2: np.ndarray = np.eye(2, dtype=complex)
I_4: np.ndarray = np.eye(4, dtype=complex)

# Alias for backward compatibility with heisenberg_model.py
EYE: np.ndarray = I_2

# Angular-momentum operators for a single qubit (J = σ/2)
J_X: np.ndarray = SIGMA_X / 2.0
J_Y: np.ndarray = SIGMA_Y / 2.0
J_Z: np.ndarray = SIGMA_Z / 2.0
