"""
Sensitivity Analysis Physics Module.

This module contains the core physics logic for sensitivity analysis:
- Rabi frequency computation
- Sensitivity calculations to parameters
- Heatmap generation

Physical Model:
- Reduced system-ancilla model with single Fock state
- H = (-J_S σ_x + δ_S σ_z) + α_x σ_x J_z + α_z σ_z J_z

Units:
- Dimensionless throughout.
"""

from typing import Any, Dict

import numpy as np


# =============================================================================
# Sensitivity Calculation
# =============================================================================


def compute_rabi_frequency(
    n: int,
    k: int,
    j_s: float,
    delta_s: float,
    alpha_x: float,
    alpha_z: float,
) -> float:
    """Compute Rabi frequency ω_k.

    ω_k = sqrt((α_z * (N-2k)/2 + δ_S)² + (α_x * (N-2k)/2 - J_S)²)

    Args:
        n: Ancillary dimension N.
        k: Ancilla level k.
        j_s: System tunneling strength.
        delta_s: System energy shift.
        alpha_x: σ_x coupling coefficient.
        alpha_z: σ_z coupling coefficient.

    Returns:
        Rabi frequency ω_k.

    Raises:
        ValueError: If k > n or k < 0.
    """
    if k > n:
        raise ValueError(f"k={k} must be <= n={n}")
    if k < 0:
        raise ValueError(f"k={k} must be >= 0")

    x_coefficient = alpha_x * (n - 2 * k) / 2 - j_s
    z_coefficient = alpha_z * (n - 2 * k) / 2 + delta_s

    return np.sqrt(x_coefficient**2 + z_coefficient**2)


def sensitivity(
    n: int,
    k: int,
    j_s: float,
    delta_s: float,
    alpha_x: float,
    alpha_z: float,
    t: float,
) -> Dict[str, Any]:
    """Compute sensitivity to J_S and δ_S.

    Calculates the sensitivity (derivative) of the observable
    ⟨σ_z⟩ with respect to system parameters:
    - ∂⟨σ_z⟩/∂J_S
    - ∂⟨σ_z⟩/∂δ_S

    The formulas are derived from:
    ⟨σ_z⟩ ≈ cos²(ω_k t) + sin²(ω_k t) * (z_coeff/ω_k)² - sin²(ω_k t) * (x_coeff/ω_k)

    Args:
        n: Ancillary dimension N.
        k: Level k.
        j_s: System tunneling strength.
        delta_s: System energy shift.
        alpha_x: σ_x coupling coefficient.
        alpha_z: σ_z coupling coefficient.
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

    # Coefficients
    x_coefficient = alpha_x * (n - 2 * k) / 2 - j_s
    z_coefficient = alpha_z * (n - 2 * k) / 2 + delta_s
    omega_k = np.sqrt(x_coefficient**2 + z_coefficient**2)

    # Handle omega_k ≈ 0 case
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

    # Sensitivities using chain rule
    sin_sq = np.sin(omega_k * t) ** 2

    # d⟨σ_z⟩/dJ_S = sin²(ω_k t) * α_x * x_coeff / ω_k²
    sensitivity_to_j = sin_sq * (alpha_x * x_coefficient) / (omega_k**2)

    # d⟨σ_z⟩/dδ_S = sin²(ω_k t) * α_z * z_coeff / ω_k²
    sensitivity_to_delta = sin_sq * (alpha_z * z_coefficient) / (omega_k**2)

    return {
        "n": n,
        "k": k,
        "j_s": j_s,
        "delta_s": delta_s,
        "alpha_x": alpha_x,
        "alpha_z": alpha_z,
        "t": t,
        "omega_k": omega_k,
        "sensitivity_to_j": sensitivity_to_j,
        "sensitivity_to_delta": sensitivity_to_delta,
    }


def compute_sensitivity_grid(
    n: int,
    k: int,
    j_s: float,
    delta_s: float,
    alpha_x_range: np.ndarray,
    alpha_z_range: np.ndarray,
    t: float,
) -> Dict[str, np.ndarray]:
    """Compute sensitivity over a grid of α_x, α_z values.

    Args:
        n: Ancillary dimension.
        k: Level.
        j_s: System parameter.
        delta_s: System parameter.
        alpha_x_range: Range of α_x values.
        alpha_z_range: Range of α_z values.
        t: Time.

    Returns:
        Dictionary with sensitivity grids.
    """
    omega_grid = np.zeros((len(alpha_x_range), len(alpha_z_range)))
    sens_j_grid = np.zeros_like(omega_grid)
    sens_delta_grid = np.zeros_like(omega_grid)

    for i, ax in enumerate(alpha_x_range):
        for j, az in enumerate(alpha_z_range):
            result = sensitivity(n, k, j_s, delta_s, ax, az, t)
            omega_grid[i, j] = result["omega_k"]
            sens_j_grid[i, j] = result["sensitivity_to_j"]
            sens_delta_grid[i, j] = result["sensitivity_to_delta"]

    return {
        "omega_k": omega_grid,
        "sensitivity_to_j": sens_j_grid,
        "sensitivity_to_delta": sens_delta_grid,
        "alpha_x": alpha_x_range,
        "alpha_z": alpha_z_range,
    }


# =============================================================================
# Observable Calculation
# =============================================================================


def compute_observable(
    n: int,
    k: int,
    j_s: float,
    delta_s: float,
    alpha_x: float,
    alpha_z: float,
    t: float,
) -> float:
    """Compute the observable ⟨σ_z⟩(t).

    Args:
        n: Ancillary dimension.
        k: Initial ancilla state.
        j_s: System parameter.
        delta_s: System parameter.
        alpha_x: Coupling coefficient.
        alpha_z: Coupling coefficient.
        t: Time.

    Returns:
        Observable value ⟨σ_z⟩.
    """
    x_coefficient = alpha_x * (n - 2 * k) / 2 - j_s
    z_coefficient = alpha_z * (n - 2 * k) / 2 + delta_s
    omega_k = np.sqrt(x_coefficient**2 + z_coefficient**2)

    if omega_k < 1e-10:
        return 1.0  # cos²(0) = 1

    cos_sq = np.cos(omega_k * t) ** 2
    sin_sq = np.sin(omega_k * t) ** 2

    return cos_sq + sin_sq * (z_coefficient / omega_k) ** 2 - sin_sq * (x_coefficient / omega_k)


# =============================================================================
# Validation
# =============================================================================


def validate_sensitivity(
    n: int,
    k: int,
    j_s: float,
    delta_s: float,
    alpha_x: float,
    alpha_z: float,
    t: float,
    tolerance: float = 1e-6,
) -> bool:
    """Validate sensitivity calculation via finite differences.

    Args:
        n: Ancillary dimension.
        k: Level.
        j_s: System parameter.
        delta_s: System parameter.
        alpha_x: Coupling.
        alpha_z: Coupling.
        t: Time.
        tolerance: Required accuracy.

    Returns:
        True if numerical derivative matches analytical.
    """
    eps = 1e-5

    # Get sensitivities
    result = sensitivity(n, k, j_s, delta_s, alpha_x, alpha_z, t)

    # Numerical derivative wrt j_s
    obs_plus = compute_observable(n, k, j_s + eps, delta_s, alpha_x, alpha_z, t)
    obs_minus = compute_observable(n, k, j_s - eps, delta_s, alpha_x, alpha_z, t)
    num_deriv_j = (obs_plus - obs_minus) / (2 * eps)

    # Numerical derivative wrt delta_s
    obs_plus = compute_observable(n, k, j_s, delta_s + eps, alpha_x, alpha_z, t)
    obs_minus = compute_observable(n, k, j_s, delta_s - eps, alpha_x, alpha_z, t)
    num_deriv_delta = (obs_plus - obs_minus) / (2 * eps)

    # Compare
    if abs(num_deriv_j - result["sensitivity_to_j"]) > tolerance:
        return False
    if abs(num_deriv_delta - result["sensitivity_to_delta"]) > tolerance:
        return False

    return True