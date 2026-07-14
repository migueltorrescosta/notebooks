"""
Pure sensitivity-math utilities.

Provides the observable computation and analytical sensitivity derivative
for the reduced system-ancilla model.  These are pure mathematical
formulas that depend only on numpy and live in ``utils`` so that both
``utils.validators`` and ``analysis.sensitivity_analysis`` can use them
without creating upward dependency cycles.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_rabi_frequency(
    n: int,
    k: int,
    j_s: float,
    delta_s: float,
    alpha_x: float,
    alpha_z: float,
) -> float:
    """Compute Rabi frequency omega_k.

    omega_k = sqrt((alpha_z * (N-2k)/2 + delta_S)^2
                 + (alpha_x * (N-2k)/2 - J_S)^2)

    Args:
        n: Ancillary dimension N.
        k: Ancilla level k.
        j_s: System tunneling strength.
        delta_s: System energy shift.
        alpha_x: sigma_x coupling coefficient.
        alpha_z: sigma_z coupling coefficient.

    Returns:
        Rabi frequency omega_k.

    Raises:
        ValueError: If k > n or k < 0.

    """
    if k > n:
        raise ValueError(f"k={k} must be <= n={n}")
    if k < 0:
        raise ValueError(f"k={k} must be >= 0")

    x_coefficient = alpha_x * (n - 2 * k) / 2 - j_s
    z_coefficient = alpha_z * (n - 2 * k) / 2 + delta_s

    return float(np.sqrt(x_coefficient**2 + z_coefficient**2))


def sensitivity(
    n: int,
    k: int,
    j_s: float,
    delta_s: float,
    alpha_x: float,
    alpha_z: float,
    t: float,
) -> dict[str, Any]:
    """Compute sensitivity to J_S and delta_S.

    Calculates the sensitivity (derivative) of the observable
    <sigma_z> with respect to system parameters:
    - d<sigma_z>/dJ_S
    - d<sigma_z>/ddelta_S

    Args:
        n: Ancillary dimension N.
        k: Level k.
        j_s: System tunneling strength.
        delta_s: System energy shift.
        alpha_x: sigma_x coupling coefficient.
        alpha_z: sigma_z coupling coefficient.
        t: Evolution time.

    Returns:
        Dictionary with all parameters and sensitivities.

    Raises:
        ValueError: If k > n or k < 0.

    """
    if k > n:
        raise ValueError(f"k={k} must be <= n={n}")
    if k < 0:
        raise ValueError(f"k={k} must be >= 0")

    x_coefficient = alpha_x * (n - 2 * k) / 2 - j_s
    z_coefficient = alpha_z * (n - 2 * k) / 2 + delta_s
    omega_k = np.sqrt(x_coefficient**2 + z_coefficient**2)

    if omega_k < 1e-10:
        return {
            "n": n,
            "k": k,
            "j_s": j_s,
            "delta_s": delta_s,
            "alpha_x": alpha_x,
            "alpha_z": alpha_z,
            "t": t,
            "omega_k": 0.0,
            "sensitivity_to_j": 0.0,
            "sensitivity_to_delta": 0.0,
        }

    sin_sq = np.sin(omega_k * t) ** 2

    sensitivity_to_j = sin_sq * (alpha_x * x_coefficient) / (omega_k**2)
    sensitivity_to_delta = sin_sq * (alpha_z * z_coefficient) / (omega_k**2)

    return {
        "n": n,
        "k": k,
        "j_s": j_s,
        "delta_s": delta_s,
        "alpha_x": alpha_x,
        "alpha_z": alpha_z,
        "t": t,
        "omega_k": float(omega_k),
        "sensitivity_to_j": float(sensitivity_to_j),
        "sensitivity_to_delta": float(sensitivity_to_delta),
    }


def compute_observable(
    n: int,
    k: int,
    j_s: float,
    delta_s: float,
    alpha_x: float,
    alpha_z: float,
    t: float,
) -> float:
    """Compute the observable <sigma_z>(t).

    Args:
        n: Ancillary dimension.
        k: Initial ancilla state.
        j_s: System parameter.
        delta_s: System parameter.
        alpha_x: Coupling coefficient.
        alpha_z: Coupling coefficient.
        t: Time.

    Returns:
        Observable value <sigma_z>.

    """
    x_coefficient = alpha_x * (n - 2 * k) / 2 - j_s
    z_coefficient = alpha_z * (n - 2 * k) / 2 + delta_s
    omega_k = np.sqrt(x_coefficient**2 + z_coefficient**2)

    if omega_k < 1e-10:
        return 1.0  # cos^2(0) = 1

    cos_sq = np.cos(omega_k * t) ** 2
    sin_sq = np.sin(omega_k * t) ** 2

    return float(
        cos_sq
        + sin_sq * (z_coefficient / omega_k) ** 2
        - sin_sq * (x_coefficient / omega_k)
    )
