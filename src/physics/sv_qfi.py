r"""
Squeezed-vacuum analytical QFI and truncation-convergence utilities.

Shared across reports that use single-mode squeezed vacuum (SV) states
to avoid duplicating the analytical recurrence and QFI formula.

Physical Model:
    The single-mode squeezed vacuum state is :math:`S(r)\vert 0\rangle_1
    \otimes \vert 0\rangle_2 = \sum_{n} c_n \vert 2n, 0\rangle` with
    :math:`c_n = \sqrt{(2n)!}/(2^n n!) \cdot \tanh^n(r)/\sqrt{\cosh(r)}`.

    The mean photon number is :math:`\langle N \rangle = \sinh^2(r)`.
    The QFI for a :math:`J_z`-generated phase is
    :math:`F_Q = 2 t_{\text{hold}}^2 \langle N \rangle (\langle N \rangle + 1)`.
"""

from __future__ import annotations

import numpy as np


def compute_sv_captured_norm(mean_N: float, max_photons: int) -> float:
    r"""Analytical norm captured by SV truncated at ``max_photons`` per mode.

    The single-mode squeezed vacuum only populates states :math:`\vert 2n, 0\rangle`.
    The probability of the :math:`2n`-photon component satisfies the recurrence:

    .. math::

        P(0) = \frac{1}{\cosh(r)},\qquad
        P(2n) = \frac{2n-1}{2n}\,\tanh^{2}(r)\,P(2n-2)

    The captured norm is :math:`\sum_{n=0}^{\lfloor M/2\rfloor} P(2n)` where
    :math:`M =` ``max_photons`` and :math:`\langle N \rangle = \sinh^{2}(r)`.

    Args:
        mean_N: Mean photon number :math:`\langle N \rangle = \sinh^{2}(r)`.
        max_photons: Truncation per mode (maximum photon number).

    Returns:
        Fraction of total norm captured in :math:`[0, 1]`.
    """
    r = float(np.arcsinh(np.sqrt(mean_N)))
    tanh_sq = np.tanh(r) ** 2
    cosh_r = np.cosh(r)
    max_n = max_photons // 2  # only even n are populated

    prob = 1.0 / cosh_r  # P(0)
    captured = prob

    for n in range(1, max_n + 1):
        prob *= (2.0 * n - 1.0) / (2.0 * n) * tanh_sq
        captured += prob

    return float(captured)


def compute_sv_qfi(mean_N: float, t_hold: float = 10.0) -> float:
    r"""Analytical QFI for single-mode squeezed vacuum.

    :math:`F_Q = 2 \cdot t_{\text{hold}}^2 \cdot \langle N \rangle (\langle N \rangle + 1)`

    Args:
        mean_N: Mean photon number :math:`\langle N \rangle = \sinh^2(r)`.
        t_hold: Holding time.

    Returns:
        Quantum Fisher information.
    """
    return 2.0 * t_hold**2 * mean_N * (mean_N + 1.0)


def verify_sv_qfi(mean_N: float, var_probe: float) -> bool:
    r"""Verify that Var(J_z) satisfies the SV analytical formula.

    :math:`\text{Var}(J_z)_{\text{probe}} = \langle N \rangle (\langle N \rangle + 1) / 2`

    Args:
        mean_N: Mean photon number.
        var_probe: Computed variance of J_z from the probe state.

    Returns:
        True if the variance matches the analytical formula within tolerance.
    """
    expected_var = mean_N * (mean_N + 1.0) / 2.0
    return bool(np.isclose(var_probe, expected_var, rtol=1e-4))


def check_truncation_convergence(
    state: np.ndarray | None = None,
    threshold: float = 0.999,
    *,
    mean_n: float | None = None,
    mean_total: float | None = None,
    max_photons: int | None = None,
) -> bool:
    r"""Check that the truncated Hilbert space captures enough norm.

    For SV states, the analytical truncation error is computed using the
    photon-number recurrence of the squeezed vacuum *before* renormalisation
    (which would otherwise hide the truncation loss).

    For TMSV states, the analytical truncation error is:

    .. math::

        \sum_{n=0}^{M} \frac{\tanh^{2n}(r)}{\cosh^{2}(r)}
        = 1 - \tanh^{2(M+1)}(r)

    Args:
        state: State vector (only used as fallback when analytical parameters
            are not provided).
        threshold: Minimum captured fraction (default 0.999).
        mean_n: Mean photon number for SV (analytical check).
        mean_total: Total mean photon number for TMSV (analytical check).
        max_photons: Truncation per mode (required for analytical checks).

    Returns:
        True if the captured norm fraction >= threshold.

    Raises:
        ValueError: If neither analytical parameters nor state are provided.
    """
    if mean_n is not None and max_photons is not None:
        captured = compute_sv_captured_norm(mean_n, max_photons)
        return captured >= threshold
    if mean_total is not None and max_photons is not None:
        r = float(np.arcsinh(np.sqrt(mean_total / 2.0)))
        tanh_r = np.tanh(r)
        captured = 1.0 - tanh_r ** (2 * (max_photons + 1))
        return captured >= threshold
    if state is not None:
        return bool(np.linalg.norm(state) >= threshold)
    raise ValueError(
        "Must provide (mean_n, max_photons) for SV, "
        "(mean_total, max_photons) for TMSV, or state."
    )
