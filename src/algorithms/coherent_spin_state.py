"""
General coherent spin state (CSS) construction for arbitrary spin J.

Provides ``coherent_spin_state(J, theta, phi)`` for constructing CSS
:math:`|\\theta, \\phi\\rangle` for any integer or half-integer spin J,
using the binomial expansion in the Dicke basis.
"""

from __future__ import annotations

import cmath
import math

import numpy as np


def coherent_spin_state(J: float, theta: float, phi: float) -> np.ndarray:
    r"""Construct a coherent spin state :math:`|\theta, \phi\rangle` for general J.

    .. math::

        |\psi(\theta, \phi)\rangle =
        \sum_{m=-J}^{J} \sqrt{\binom{2J}{J+m}}
        \cos(\theta/2)^{J+m} e^{-i\phi(J-m)}
        \sin(\theta/2)^{J-m} |J, m\rangle

    The Dicke basis ordering is descending :math:`m = +J, J-1, \dots, -J`.

    Args:
        J: Total spin :math:`J` (half-integer or integer). Must be non-negative.
        theta: Polar angle in :math:`[0, \pi]`.
        phi: Azimuthal angle in :math:`[0, 2\pi)`.

    Returns:
        Normalised :math:`(2J+1)`-dimensional complex vector in the Dicke basis.

    Raises:
        ValueError: If J is negative.
    """
    if J < 0:
        raise ValueError(f"J must be non-negative, got {J}")

    dim = int(2 * J + 1)
    if dim == 1:
        return np.array([1.0], dtype=complex)

    state = np.zeros(dim, dtype=complex)
    two_J = int(2 * J)
    cos_half = math.cos(theta / 2.0)
    sin_half = math.sin(theta / 2.0)

    # Dicke basis ordering: m = J, J-1, ..., -J
    for idx, m in enumerate(np.linspace(J, -J, dim)):
        j_plus_m = round(J + m)  # J + m in {0, ..., 2J}
        j_minus_m = round(J - m)  # J - m in {0, ..., 2J}
        coeff = math.sqrt(math.comb(two_J, j_plus_m))
        coeff *= cos_half**j_plus_m
        coeff *= cmath.exp(-1j * phi * j_minus_m)
        coeff *= sin_half**j_minus_m
        state[idx] = coeff

    norm = np.linalg.norm(state)
    assert abs(norm - 1.0) < 1e-12, (
        f"CSS norm={norm} deviates from 1 for J={J}, theta={theta}"
    )
    curr_norm = abs(norm - 1.0)
    if curr_norm > 1e-12:
        state = state / norm
    return state
