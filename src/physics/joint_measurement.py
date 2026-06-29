"""
Shared weighted joint measurement operators.

Provides functions to build the measurement operator

    M(ψ) = cosψ · J_z^S + sinψ · J_z^A

for two common Hilbert-space configurations:

- **Qubit ancilla** (J_A = 1/2):  Total space dimension 2(N+1).
- **Multi-particle ancilla** (J_A = N/2):  Total space dimension (N+1)².

Originally duplicated across reports #20260613 and #20260628; promoted here
as reusable infrastructure.
"""

from __future__ import annotations

import numpy as np


def build_joint_measurement_operator(
    N: int,
    psi: float,
    ops: dict[str, np.ndarray],
) -> np.ndarray:
    """Build the weighted joint measurement operator M(ψ) for J_A = 1/2.

    M(ψ) = cosψ · J_z^S + sinψ · J_z^A

    Dimension: 2(N+1) × 2(N+1).  The coefficients satisfy
    m_s² + m_a² = 1 with m_s = cosψ, m_a = sinψ.

    Args:
        N: Number of system particles (for dimension check).
        psi: Measurement weight angle (radians).
        ops: Operators from ``build_n_particle_operators(N)``.

    Returns:
        2(N+1) × 2(N+1) Hermitian matrix.
    """
    d_tot = 2 * (N + 1)
    M = np.cos(psi) * ops["Jz_S"] + np.sin(psi) * ops["Jz_A"]
    M = 0.5 * (M + M.conj().T)
    assert M.shape == (d_tot, d_tot), (
        f"M has shape {M.shape}, expected ({d_tot}, {d_tot})"
    )
    assert np.allclose(M, M.conj().T, atol=1e-12), (
        "Joint measurement operator not Hermitian"
    )
    return M


def build_bipartite_joint_measurement_operator(
    N: int,
    psi: float,
    ops: dict[str, np.ndarray],
) -> np.ndarray:
    """Build the joint measurement operator for J_A = N/2.

    M(ψ) = cosψ · J_z^S + sinψ · J_z^A

    Dimension: (N+1)² × (N+1)².

    Args:
        N: Number of particles per subsystem (for dimension check).
        psi: Measurement weight angle (radians).
        ops: Operators from ``build_operators(N, N)``.

    Returns:
        (N+1)² × (N+1)² Hermitian matrix.
    """
    d_tot = (N + 1) ** 2
    M = np.cos(psi) * ops["Jz_S"] + np.sin(psi) * ops["Jz_A"]
    M = 0.5 * (M + M.conj().T)
    assert M.shape == (d_tot, d_tot), (
        f"M has shape {M.shape}, expected ({d_tot}, {d_tot})"
    )
    assert np.allclose(M, M.conj().T, atol=1e-12), (
        "Bipartite joint measurement operator not Hermitian"
    )
    return M
