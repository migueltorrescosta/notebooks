r"""
Number-difference distribution for Mach-Zehnder interferometers.

Computes :math:`P(m|\omega)` where :math:`m = n_1 - n_2` from an output
state in the two-mode Fock basis.

Hilbert Space:
- Two-mode Fock basis with dimension ``(max_photons+1)²``.
- Basis ordering: |n₁, n₂⟩ with n₁ as first mode, n₂ as second mode.

Units:
- Dimensionless throughout.
"""

from __future__ import annotations

import numpy as np


def output_number_diff_distribution(
    state_out: np.ndarray,
    max_photons: int,
) -> np.ndarray:
    r"""Compute :math:`P(m|\omega)` where :math:`m = n_1 - n_2`.

    From the output state, collect the probability of each number-difference
    outcome :math:`m \in \{-M, \dots, M\}` with :math:`M = \text{max\_photons}`.

    Args:
        state_out: Output state vector in the two-mode Fock basis.
        max_photons: Maximum photon number per mode.

    Returns:
        Array of shape ``(2 * max_photons + 1,)`` indexed by ``m + max_photons``.
    """
    n_outcomes = 2 * max_photons + 1
    P = np.zeros(n_outcomes, dtype=float)
    offset = max_photons
    for n1 in range(max_photons + 1):
        for n2 in range(max_photons + 1):
            idx = n1 * (max_photons + 1) + n2
            prob = np.real(state_out[idx].conj() * state_out[idx])
            m = n1 - n2
            P[m + offset] += prob
    return P
